"""Phase 2: throw the held ball at a cup. Two presets, both found by optimizing the swing on the
xArm850 URDF and checked live on sfhack11 (Oct 29):

  far    open-hand forward toss, cups ~80-100 cm. THE DEFAULT. The ball rides in the palm with the
         fingers pointing forward, and it flies off when the arm brakes, so finger timing matters little.
         Live: the 3 measured landings (19:33-19:37) were within ~8 cm of the model.
  power  full-force throw for distance (direct mode only), cups 1-2 m. Timed release: it scatters more.

    python throw.py --distance 900                                 # dry run: prints the plan
    python throw.py --distance 900 --go --i-checked-with-staff     # live (E-stop!)
    python throw.py --distance 1500 --power --go --i-checked-with-staff

Safety: the swing and the wind-up are DIRECT joint moves (no Viam motion planning), so every one is
checked first with throw_safety.path_problems (table / ceiling / walls, live obstacle values) and joint
limits. Raised speeds are restored in a `finally`.

Distance: the swing speed comes from the preset's model table, scaled by the correction learned from
logged throws in Atlas (throw_fit.py). --speed overrides it.
"""
import argparse
import asyncio
import math
import time

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.proto.component.arm import JointPositions
from viam.services.motion import MotionClient

from connection import connect
from throw_safety import path_problems

# Windows ticks its timers every ~15.6 ms by default; ask for 1 ms resolution (no-op elsewhere).
try:
    import sys as _sys
    if _sys.platform == "win32":
        import ctypes
        ctypes.WinDLL("winmm").timeBeginPeriod(1)
except Exception:  # noqa: BLE001
    pass

ARM, GRIPPER = "arm", "gripper"
NORMAL_SPEED = 60            # deg/s, the config value
MAX_SPEED, MAX_ACCEL = 180, 1145   # viam:ufactory module limits
# xArm850 joint limits, degrees (UFactory spec). A swing that leaves them is refused, not clipped.
JOINT_LIMITS_DEG = {0: (-360, 360), 1: (-132, 120), 2: (-242, 3.5), 3: (-360, 360),
                    4: (-124, 124), 5: (-360, 360)}
LIMIT_MARGIN_DEG = 5
HOME = dict(x=350, y=0, z=445, o_x=0, o_y=0, o_z=-1, theta=180)
GRIPPER_SPEED = 5000          # firmware max; faster fingers = earlier, steadier release
MAX_YAW_DEG = 45

# FAR: start ball (357, 0, 608) fingers forward -> end (622, 0, 799) fingers 22 deg up.
# Flange stays ~1.14 m/s at 180 deg/s, under the ServoJ linear-speed stop (~1.26-1.44 m/s).
FAR = dict(start_j=(55.9, -19.6, 11.1),            # J2, J3, J5
           swing={1: -49.9, 2: -40.0, 4: +35.4},
           table=[(105, 837), (120, 866), (135, 895), (150, 924), (165, 953), (180, 983)],  # deg/s -> mm
           lead=0.43,          # gripper starts moving ~0.3 s after open(); open ~0.13 s before braking
           min_speed=100.0, direct=False,
           min_delay=0.10)     # live 19:34: opening before the swing dropped the ball (fingers forward)
# POWER: shoulder, elbow, wrist all at 180 deg/s -> ball 3.8 m/s at 48 deg. Flange 3.26 m/s, so
# only in direct mode (no ServoJ stop). Start ball (286, 0, 89) fingers back/down; release ~(600, 0, 300).
POWER = dict(start_j=(-57.4, -65.5, -79.5),
             swing={1: +33.1, 2: -34.7, 4: +34.3},
             table=[(60, 900), (80, 1017), (100, 1187), (120, 1412), (140, 1679), (160, 1955), (180, 2179)],
             lead=0.17,        # ON THE ROBOT COMPUTER (Oct 30): partial open starts 0.10-0.18 s after the
                               # command; the laptop-era 0.40 s included 15-400 ms of network delay
             min_speed=50.0, direct=True,
             min_delay=0.0)    # fingers point back/down at the start: never open before the swing
# CRADLE (spoon): FAR's swing with the wrist rolled 90 deg so the fingers sit ABOVE and BELOW the ball
# (the lower finger is a spoon), and a PARTIAL open sent the moment the swing starts. If the fingers
# open early the ball just rests on the lower finger (instead of dropping out, as FAR did at 19:34),
# and it leaves the open end when the arm brakes - timing hardly matters.
# scripts/design_cradle.py (URDF + ball-exit simulation): with FAR's motion the ball clears the
# fingertips at 165-180 deg/s even from deep in the palm; below that only from the fingertips, so the
# preset is limited to 165-180 (cups ~0.9-1.0 m). A fingers-UP cradle (open before the swing) is NOT
# feasible on this arm: 1145 deg/s^2 brakes the ball at only ~1.2-1.8 g, so it never gets out.
# UNTESTED on the robot: run --cradle-check first (wind-up, open, hold 3 s, no swing).
CRADLE = dict(start_j=FAR["start_j"], swing=dict(FAR["swing"]),
              table=[(165, 953), (180, 983)],            # FAR's model (roll doesn't move the ball center)
              lead=0.0, min_speed=165.0, direct=False, min_delay=0.0, release="cradle", j6=90.0)
PRESETS = {"far": FAR, "power": POWER, "cradle": CRADLE}
ACCEL = 1145
HOLD_POS_TYPICAL = 370        # gripper position while holding a ball (status notes, Oct 29)
CRADLE_OPEN_DELTA = 90        # open this much past the holding position (~9 mm if 850 = full 85 mm stroke)
PARTIAL_OPEN_DELTA = 150      # --partial: timed release opens only this far instead of fully


def check_limits(start, end):
    """Raise if any joint of `end` is outside its limit (with a margin)."""
    for j, v in enumerate(end):
        lo, hi = JOINT_LIMITS_DEG[j]
        if not lo + LIMIT_MARGIN_DEG <= v <= hi - LIMIT_MARGIN_DEG:
            raise ValueError(f"swing leaves joint J{j + 1} limit: {start[j]:.1f} -> {v:.1f} "
                             f"(allowed {lo + LIMIT_MARGIN_DEG}..{hi - LIMIT_MARGIN_DEG})")


def check_path(q_from, q_to, what):
    """Raise if the direct joint move q_from -> q_to passes too close to the table/ceiling/walls."""
    probs = path_problems(q_from, q_to)
    if probs:
        raise RuntimeError(f"{what} path not clear: " + "; ".join(probs))


def yaw_for_cup(x, y):
    """Base rotation (deg) that points the throw plane at a cup at world (x, y) mm."""
    yaw = math.degrees(math.atan2(y, x))
    print(f"   aim: cup at ({x:.0f}, {y:.0f}) -> yaw {yaw:+.1f} deg, {math.hypot(x, y):.0f} mm from the base")
    return yaw


def reach_mm(preset, ratio=1.0):
    """(min, max) cup distance a preset can reach, mm."""
    p = PRESETS[preset]
    t = p["table"]
    # can't slow below the table's first speed -> only a small margin on the short side
    short = 20 if p["min_speed"] >= t[0][0] else 60
    return round(t[0][1] * ratio) - short, round(t[-1][1] * ratio) + 40


def speed_for(distance_mm, preset="far", ratio=1.0):
    """Swing speed (deg/s) whose landing is distance_mm from the base. `ratio` = observed / model
    landing distance from logged throws (throw_fit). Clamped to [min_speed, 180]."""
    p = PRESETS[preset]
    t = [(s, d * ratio) for s, d in p["table"]]
    if distance_mm <= t[0][1]:
        slope = (t[1][1] - t[0][1]) / (t[1][0] - t[0][0])
        return max(p["min_speed"], t[0][0] - (t[0][1] - distance_mm) / slope)
    for (s0, d0), (s1, d1) in zip(t, t[1:]):
        if distance_mm <= d1:
            return s0 + (s1 - s0) * (distance_mm - d0) / (d1 - d0)
    return float(MAX_SPEED)


def model_landing(speed_dps, preset="far"):
    """Model landing distance (mm) for a swing speed: the inverse of speed_for at ratio 1."""
    t = PRESETS[preset]["table"]
    if speed_dps <= t[0][0]:
        (s0, d0), (s1, d1) = t[0], t[1]
    else:
        (s0, d0), (s1, d1) = next(((a, b) for a, b in zip(t, t[1:]) if speed_dps <= b[0]), (t[-2], t[-1]))
    return d0 + (d1 - d0) * (speed_dps - s0) / (s1 - s0)


def release_delay(speed_dps, preset="far", accel=ACCEL):
    """Open-command time (s from swing start) so the fingers are open as the arm starts braking."""
    p = PRESETS[preset]
    dmax = max(abs(v) for v in p["swing"].values())
    t_brake = speed_dps / accel + (dmax - speed_dps ** 2 / accel) / speed_dps
    return max(t_brake - p["lead"], p["min_delay"])


def plan(distance_mm, preset="far", speed=None, release_delay_s=None, ratio=1.0, yaw_deg=0.0, scale=1.0,
         roll=None, tilt_up=0.0, cradle_open=None):
    """Everything for one throw, no robot needed. Raises on anything unsafe.
    roll: absolute wrist roll J6 (deg) for the swing (default: the preset's j6, or keep current).
    tilt_up: raise the fingers this many deg for the whole swing (J5 +; checked live: +1 deg J5 = +1 deg).
    cradle_open: cradle opening past the holding position (gripper units, default CRADLE_OPEN_DELTA)."""
    if not -15 <= tilt_up <= 25:
        raise ValueError("--tilt-up must be between -15 and 25 deg")
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r} (use {', '.join(PRESETS)})")
    if not 0.1 <= scale <= 1.0:
        raise ValueError("--scale must be between 0.1 and 1")
    if abs(yaw_deg) > MAX_YAW_DEG:
        raise ValueError(f"yaw must be within +-{MAX_YAW_DEG} deg (move the cup nearer the middle)")
    p = PRESETS[preset]
    spd = speed if speed is not None else speed_for(distance_mm, preset, ratio)
    spd = max(20.0, spd * scale)
    if spd > MAX_SPEED:
        raise ValueError(f"speed above the module limit ({MAX_SPEED} deg/s)")
    cradle = p.get("release") == "cradle"
    if release_delay_s is not None:
        delay = float(release_delay_s)
        if not -0.6 <= delay <= 0.8:
            raise ValueError("--release-delay must be between -0.6 and 0.8 s")
    else:   # computed: slow / scaled swings can brake late; never refuse a plan over the default
        delay = min(0.8, 0.0 if cradle else release_delay(spd, preset))
    swing = {j: d * scale for j, d in p["swing"].items()}
    if p["start_j"] is None:
        raise ValueError(f"preset {preset!r} has no design yet")
    j6 = roll if roll is not None else p.get("j6")
    start = [yaw_deg, p["start_j"][0], p["start_j"][1], 0.0, p["start_j"][2] + tilt_up, j6 or 0.0]
    end = list(start)
    for j, d in swing.items():
        end[j] += d
    check_limits(start, end)
    check_path(start, end, "swing")
    return dict(preset=preset, distance_mm=round(distance_mm), speed_dps=round(spd, 1), accel_dps2=ACCEL,
                release_delay_s=round(delay, 3), swing_deg=swing, start_j=start, end_j=end,
                yaw_deg=round(yaw_deg, 1), direct=p["direct"], ratio=round(ratio, 3), scale=scale,
                release_mode="cradle" if cradle else "timed", j6=j6, tilt_up=tilt_up,
                cradle_open=CRADLE_OPEN_DELTA if cradle_open is None else cradle_open,
                model_landing_mm=round(model_landing(spd, preset) * ratio))


async def _go_home(motion, tries=3):
    """The move home right after a fast swing was cancelled once by the session manager (Oct 29,
    "context canceled"). Moving home is safe to repeat."""
    for i in range(tries):
        try:
            ok = await motion.move(GRIPPER, PoseInFrame(reference_frame="world", pose=Pose(**HOME)))
            if ok is not False:
                return True
        except Exception as e:  # noqa: BLE001
            print(f"  home move failed ({type(e).__name__}: {str(e)[:80]}), try {i + 1}/{tries}")
        await asyncio.sleep(1.0)
    print("!! arm is NOT home: run `python home.py` or move it in the Viam app")
    return False


async def _gripper_pos(gripper):
    try:
        return float((await gripper.do_command({"get": True})).get("pos"))
    except Exception:  # noqa: BLE001
        return None


async def throw(robot, pl, go, release=True, direct=None, partial=False, check_only=False):
    """Run a plan from plan(). Starts from home; wind-up and swing are checked direct joint moves.
    partial: timed release opens only PARTIAL_OPEN_DELTA past the holding position (less finger travel).
    check_only: go to the wind-up pose (cradle: open the cradle), hold 3 s, close, go home. No swing."""
    direct = pl["direct"] if direct is None else direct
    cradle = pl.get("release_mode") == "cradle"
    print(f"1) wind up (joint move) to J2={pl['start_j'][1]} J3={pl['start_j'][2]} J5={pl['start_j'][4]}, "
          f"yaw {pl['yaw_deg']:+.1f} deg")
    print(f"2) {pl['preset'].upper()} swing at {pl['speed_dps']:.0f} deg/s"
          + (", spoon release (partial open at swing start)" if cradle and release else
             f", open at {pl['release_delay_s'] * 1000:+.0f} ms" + (" (partial)" if partial else "") if release
             else ", NO release")
          + (" [CHECK ONLY: no swing]" if check_only else "")
          + f" (cup {pl['distance_mm']} mm, model lands {pl['model_landing_mm']} mm, "
            f"{'direct' if direct else 'servo'} mode, scale {pl['scale']})")
    if not go:
        return {**pl, "dry_run": True}
    motion = MotionClient.from_robot(robot, "builtin")
    arm, gripper = Arm.from_robot(robot, ARM), Gripper.from_robot(robot, GRIPPER)
    pose = await arm.get_end_position()
    if abs(pose.x - HOME["x"]) > 90 or abs(pose.y - HOME["y"]) > 90 or pose.z < 350:
        raise RuntimeError(f"Start from home first (arm at {pose.x:.0f}, {pose.y:.0f}, {pose.z:.0f}). "
                           "Run `python home.py`.")
    cur = list((await arm.get_joint_positions()).values)
    j6 = pl["j6"] if pl.get("j6") is not None else cur[5]
    start = [cur[0] + pl["yaw_deg"], pl["start_j"][1], pl["start_j"][2], cur[3], pl["start_j"][4], j6]
    print(f"   wrist roll J6 {j6:+.0f} deg, fingers tilted {pl.get('tilt_up', 0):+.0f} deg")
    end = list(start)
    for j, d in pl["swing_deg"].items():
        end[j] += d
    check_limits(cur, start)
    check_limits(start, end)
    check_path(cur, start, "wind-up")
    check_path(start, end, "swing")
    extra = (lambda s, a: {"direct": True, "speed_d": float(s), "acceleration_d": float(a)}) if direct else (
        lambda s, a: None)
    prev_gs = None
    timing = {}
    try:
        await arm.do_command({"set_speed": NORMAL_SPEED})
        await arm.move_to_joint_positions(JointPositions(values=start), extra=extra(NORMAL_SPEED, 300.0))
        held = await _gripper_pos(gripper)
        cradle_pos = (held if held is not None else HOLD_POS_TYPICAL) + pl.get("cradle_open", CRADLE_OPEN_DELTA)
        if check_only:
            if cradle:
                print(f"   cradle: gripper {held} -> {cradle_pos:.0f}")
                await gripper.do_command({"set": float(cradle_pos)})
            print("   CHECK: fingers should be ABOVE and BELOW the ball and the ball should stay put. Holding 3 s...")
            await asyncio.sleep(3.0)
            await gripper.grab()
            await arm.move_to_joint_positions(JointPositions(values=cur), extra=extra(NORMAL_SPEED, 300.0))
            await _go_home(motion)
            return {**pl, "check_only": True}
        if release:
            try:
                prev_gs = (await gripper.do_command({"get_gripper_speed": True})).get("gripper_speed")
            except Exception:  # noqa: BLE001
                prev_gs = None
            await gripper.do_command({"set_gripper_speed": GRIPPER_SPEED})
        await arm.do_command({"set_speed": pl["speed_dps"], "set_acceleration": pl["accel_dps2"]})
        stamps = {}           # absolute monotonic times
        open_task = None

        async def open_gripper():
            stamps["open_called"] = time.monotonic()
            if cradle:
                await gripper.do_command({"set": float(cradle_pos)})
            elif partial:
                await gripper.do_command({"set": float((held or HOLD_POS_TYPICAL) + PARTIAL_OPEN_DELTA)})
            else:
                await gripper.open()
            stamps["open_returned"] = time.monotonic()

        def start_swing():
            stamps["swing_start"] = time.monotonic()
            return asyncio.create_task(arm.move_to_joint_positions(
                JointPositions(values=end), extra=extra(pl["speed_dps"], pl["accel_dps2"])))

        delay = pl["release_delay_s"]
        if release and delay < 0:   # override only: open command goes out BEFORE the swing
            open_task = asyncio.create_task(open_gripper())
            await asyncio.sleep(-delay)
            swing = start_swing()
        else:
            swing = start_swing()
            if release:
                await asyncio.sleep(delay)
                open_task = asyncio.create_task(open_gripper())
        await swing
        stamps["swing_done"] = time.monotonic()
        if open_task is not None:
            await open_task
        t0 = stamps["swing_start"]
        timing = {f"{k}_s": round(v - t0, 3) for k, v in stamps.items() if k != "swing_start"}
        print("   timing (s from swing start): " + ", ".join(
            f"{k.replace('_s', '')}={v:.2f}" for k, v in timing.items() if v is not None))
    except Exception as e:
        if "Linear speed exceeds limit" in str(e):
            print("!! the controller stopped the swing (ServoJ linear speed limit). Use --direct, "
                  "a lower --speed, or a smaller --scale.")
        raise
    finally:
        await arm.do_command({"set_speed": NORMAL_SPEED})
        if release and prev_gs:
            try:
                await gripper.do_command({"set_gripper_speed": prev_gs})
            except Exception:  # noqa: BLE001
                pass
        print("3) speed restored")
    await _go_home(motion)
    return {**pl, "direct": direct, "released": release, "partial_open": partial, "held_pos": held,
            "timing": timing}


def add_args(ap):
    """Throw options shared by throw.py and pick_and_throw.py."""
    ap.add_argument("--distance", type=float, default=None, help="cup distance from the base, mm")
    ap.add_argument("--power", action="store_true", help="full-force preset for 1-2 m (direct mode)")
    ap.add_argument("--cradle", action="store_true",
                    help="spoon throw: FAR swing, fingers above/below the ball, partial open at swing start")
    ap.add_argument("--cradle-check", action="store_true",
                    help="go to the cradle wind-up, open the cradle, hold 3 s, close, go home. NO swing")
    ap.add_argument("--partial", action="store_true", help="timed release opens partway, not fully")
    ap.add_argument("--roll", type=float, default=None,
                    help="wrist roll J6 (deg, absolute) during the throw; cradle default 90. Try 0 / 180 / -90")
    ap.add_argument("--tilt-up", type=float, default=0.0, help="raise the fingers this many deg (J5) for the throw")
    ap.add_argument("--cradle-open", type=float, default=None,
                    help=f"cradle opening past the holding position, gripper units (default {CRADLE_OPEN_DELTA})")
    ap.add_argument("--speed", type=float, default=None, help="override the swing speed, deg/s (max 180)")
    ap.add_argument("--release-delay", type=float, default=None, help="override the open time, s from swing start")
    ap.add_argument("--scale", type=float, default=1.0, help="0.1-1: shrink swing + speed for a first look")
    ap.add_argument("--yaw", type=float, default=0.0, help="turn the throw toward +y by this many deg")
    ap.add_argument("--no-release", action="store_true", help="swing but never open the gripper")
    ap.add_argument("--direct", action="store_true", help="point-to-point mode for FAR too (POWER always is)")
    ap.add_argument("--no-learn", action="store_true", help="ignore the Atlas correction; use the raw model")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--i-checked-with-staff", action="store_true")


def preset_from_args(a):
    if getattr(a, "cradle", False) or getattr(a, "cradle_check", False):
        return "cradle"
    return "power" if a.power else "far"


def learned_ratio(a):
    """Atlas correction for the chosen preset (1.0 with --no-learn or a fixed --speed)."""
    if a.no_learn or a.speed is not None:
        return 1.0
    import throw_fit
    return throw_fit.correction(preset_from_args(a))


def plan_from_args(a, distance_mm, yaw_deg=None, ratio=None):
    preset = preset_from_args(a)
    ratio = learned_ratio(a) if ratio is None else ratio
    return plan(distance_mm, preset, a.speed, a.release_delay, ratio,
                a.yaw if yaw_deg is None else yaw_deg, a.scale,
                getattr(a, "roll", None), getattr(a, "tilt_up", 0.0) or 0.0, getattr(a, "cradle_open", None))


async def _main():
    ap = argparse.ArgumentParser(description="Throw a ball that is already in the gripper.")
    add_args(ap)
    a = ap.parse_args()
    if a.distance is None:
        ap.error("--distance is required")
    go = a.go and a.i_checked_with_staff
    if a.go and not go:
        print("Refusing to throw live without --i-checked-with-staff (direct joint moves).")
    pl = plan_from_args(a, a.distance)
    robot = await connect()
    try:
        print(await throw(robot, pl, go, not a.no_release, True if a.direct else None,
                          partial=a.partial, check_only=a.cradle_check))
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(_main())
