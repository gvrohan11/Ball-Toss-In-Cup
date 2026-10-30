"""Phase 2 DRAFT: throw the held ball toward a far cup with a fast forward swing.

NOT TESTED ON THE ROBOT. Talk to Viam staff before the first live throw:
- the swing is a DIRECT joint move: it does NOT go through motion planning,
  so table / wall / ceiling obstacles are NOT checked;
- it raises the arm speed (restored afterwards, even on error).

    python throw.py --distance 1300                 # dry run: prints the plan
    python throw.py --distance 1300 --go --i-checked-with-staff   # live (E-stop!)

How it works:
 1. (planned move) gripper to WIND_UP: above the table, tilted back, ball held.
 2. read the joint angles there; the swing = rotate the elbow (J3) and wrist (J5)
    forward by SWING_DEG, so the gripper sweeps forward and up.
 3. start the joint move at THROW_SPEED; after RELEASE_DELAY_S, open the gripper.
 4. restore the normal speed, go home.

Tuning: distance is controlled by RELEASE_DELAY_S (earlier = higher arc, later =
flatter) and THROW_SPEED. Log every throw (main.py's logger) with the landing spot,
then fit delay vs distance from Atlas. Network latency (15-400 ms seen today) makes
release timing noisy; a module running on the robot computer would fix that.
"""
import argparse
import asyncio
import time

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.proto.component.arm import JointPositions
from viam.services.motion import MotionClient

from connection import connect

ARM, GRIPPER = "arm", "gripper"
NORMAL_SPEED = 60            # deg/s, the config value
THROW_SPEED = 150            # deg/s default; module max is 180 (use --speed 180)
THROW_ACCEL = 900            # deg/s^2 default; module max is 1145 (use --accel 1100)
MAX_SPEED, MAX_ACCEL = 180, 1145
# joint index -> forward swing (J3 elbow, J5 wrist), degrees. Directions MEASURED on the real arm
# at WIND_UP (Oct 29): J3 negative lifts the gripper up and forward (-15 deg: +94 mm x, +198 mm z);
# J5 NEGATIVE tilts the gripper BACKWARD (-15 deg: tilt -0.5 -> -0.71), so a forward flick is J5 positive.
SWING_DEG = {2: -35, 4: +45}
# xArm850 joint limits, degrees (UFactory spec). A swing that leaves them is refused, not clipped.
JOINT_LIMITS_DEG = {0: (-360, 360), 1: (-132, 120), 2: (-242, 3.5), 3: (-360, 360),
                    4: (-124, 124), 5: (-360, 360)}
LIMIT_MARGIN_DEG = 5
# release delay after the swing starts, per target distance (mm) - placeholders to tune
# First live throw (Oct 29, full speed): released at 200 ms and the ball just dropped when the arm
# stopped - the open command + finger travel add ~0.2 s, so 200 ms landed at the END of the ~0.4 s
# swing. Start much earlier and tune with --release-delay (try 0.0 / 0.05 / 0.1).
# Measured Oct 29 (gripper_timing_test): open() takes ~0.3 s to start and ~0.4 s until the ball is
# free, and the full swing lasts ~0.4 s. So the open command must go out BEFORE the swing:
# negative delay = seconds before the swing starts. Ball lets go ~ (0.4 + delay) s into the swing.
RELEASE_TABLE = [(900, -0.25), (1100, -0.20), (1300, -0.15)]
# theta=0 on purpose: with theta=180 the planner turned the wrist (J6) from ~0 to -180, a visible
# half turn before every throw. Same position/tilt/joints J1-J5 (checked on the real arm Oct 29);
# only the wrist angle differs, and the swing uses J3/J5 only.
WIND_UP = dict(x=300, y=0, z=450, o_x=-0.5, o_y=0, o_z=-0.866, theta=0)
HOME = dict(x=350, y=0, z=445, o_x=0, o_y=0, o_z=-1, theta=180)


def check_limits(start, end):
    """Raise if any joint of `end` is outside its limit (with a margin)."""
    for j, v in enumerate(end):
        lo, hi = JOINT_LIMITS_DEG[j]
        if not lo + LIMIT_MARGIN_DEG <= v <= hi - LIMIT_MARGIN_DEG:
            raise ValueError(f"swing leaves joint J{j + 1} limit: {start[j]:.1f} -> {v:.1f} "
                             f"(allowed {lo + LIMIT_MARGIN_DEG}..{hi - LIMIT_MARGIN_DEG})")


def release_delay(distance_mm):
    pts = sorted(RELEASE_TABLE)
    if distance_mm <= pts[0][0]:
        return pts[0][1]
    for (d0, t0), (d1, t1) in zip(pts, pts[1:]):
        if distance_mm <= d1:
            return t0 + (t1 - t0) * (distance_mm - d0) / (d1 - d0)
    return pts[-1][1]


async def _go_home(motion, tries=3):
    """The move home right after a fast swing was cancelled once by the robot's session manager
    (Oct 29, "context canceled"); the swing itself had finished. Moving home is safe to repeat."""
    for i in range(tries):
        try:
            ok = await motion.move(GRIPPER, PoseInFrame(reference_frame="world", pose=Pose(**HOME)))
            if ok is not False:
                return True
        except Exception as e:  # noqa: BLE001
            print(f"  home move failed ({type(e).__name__}: {str(e)[:80]}), try {i + 1}/{tries}")
        await asyncio.sleep(1.0)
    print("!! arm is NOT home: run `python main.py` (dry run does not move it) or move it by hand/Viam app")
    return False


async def throw(robot, distance_mm, go, scale=1.0, release=True, release_delay_s=None,
                swing_deg=None, speed_dps=None, accel_dps2=None, wind_z=None, start_j=None, yaw_deg=0.0,
                gripper_speed=None):
    """scale (0.1..1) shrinks the swing angles AND the speed: use 0.2-0.3 first, empty gripper,
    release=False, to watch the motion before any real throw."""
    if not 0.1 <= scale <= 1.0:
        raise ValueError("--scale must be between 0.1 and 1")
    motion = MotionClient.from_robot(robot, "builtin")
    arm, gripper = Arm.from_robot(robot, ARM), Gripper.from_robot(robot, GRIPPER)
    delay = release_delay(distance_mm) if release_delay_s is None else float(release_delay_s)
    if not -0.6 <= delay <= 0.6:
        raise ValueError("--release-delay must be between -0.6 and 0.6 s")
    if yaw_deg and start_j is None:
        raise ValueError("--yaw / --cup only work with --lob (direct joint swing)")
    if abs(yaw_deg) > 45:
        raise ValueError("yaw must be within +-45 deg (cup at the side: move the cup nearer the middle)")
    wind = dict(WIND_UP)
    if wind_z is not None:
        if not 300 <= wind_z <= 800:
            raise ValueError("--wind-z must be between 300 and 800 mm")
        wind["z"] = wind_z
    if start_j is not None:
        # Direct joint move to a known start configuration (J2, J3, J5; the other joints stay as
        # they are). Used by the lob, whose swing is designed around exact joint angles.
        print(f"1) wind up (joint move): J2={start_j[0]} J3={start_j[1]} J5={start_j[2]}, yaw {yaw_deg:+.1f} deg")
        if go:
            pose = await arm.get_end_position()
            if abs(pose.x - HOME["x"]) > 90 or abs(pose.y - HOME["y"]) > 90 or pose.z < 350:
                raise RuntimeError(f"Start from home first (arm at {pose.x:.0f}, {pose.y:.0f}, {pose.z:.0f}): "
                                   "this wind-up is a direct joint move. Run `python home.py`.")
            cur = list((await arm.get_joint_positions()).values)
            # J1 turns the whole swing plane about the base: +yaw = toward +y (checked live: J1 +4.3 deg -> y +25 at x 339)
            target = [cur[0] + yaw_deg, start_j[0], start_j[1], cur[3], start_j[2], cur[5]]
            check_limits(cur, target)
            await arm.do_command({"set_speed": NORMAL_SPEED})
            await arm.move_to_joint_positions(JointPositions(values=target))
    else:
        print(f"1) wind up (planned move) to {wind}")
        if go:
            ok = await motion.move(GRIPPER, PoseInFrame(reference_frame="world", pose=Pose(**wind)))
            if ok is False:
                raise RuntimeError("wind-up move failed")
    start = list((await arm.get_joint_positions()).values) if go else (
        [yaw_deg, start_j[0], start_j[1], 0.0, start_j[2], 0.0] if start_j else [0.0] * 6)
    end = list(start)
    swing_deg = dict(SWING_DEG if swing_deg is None else swing_deg)
    for j, d in swing_deg.items():
        end[j] += d * scale
    check_limits(start, end)
    speed = max(20, (speed_dps or (LOB_SPEED_DPS if start_j is not None else THROW_SPEED)) * scale)
    accel = max(100, (accel_dps2 or THROW_ACCEL) * scale)
    if speed > MAX_SPEED or accel > MAX_ACCEL:
        raise ValueError(f"speed/accel above the module limit ({MAX_SPEED} deg/s, {MAX_ACCEL} deg/s2)")
    print(f"2) swing joints {[round(v, 1) for v in start]} -> {[round(v, 1) for v in end]} "
          f"at {speed:.0f} deg/s, " + (f"release after {delay * 1000:.0f} ms" if release else "NO release")
          + f" (target {distance_mm} mm, scale {scale})")
    if not go:
        return {"distance_mm": distance_mm, "release_delay_s": delay, "dry_run": True}
    try:
        if gripper_speed and release:
            # firmware default is 1500 (G1) / 2000 (G2); max 5000. Faster fingers = earlier, steadier release.
            try:
                prev_gs = (await gripper.do_command({"get_gripper_speed": True})).get("gripper_speed")
            except Exception:  # noqa: BLE001
                prev_gs = None
            await gripper.do_command({"set_gripper_speed": gripper_speed})
        await arm.do_command({"set_speed": speed, "set_acceleration": accel})
        t0 = time.monotonic()
        t_open_call = t_open_done = None
        open_task = None

        async def open_gripper():
            nonlocal t_open_call, t_open_done
            t_open_call = time.monotonic() - t0
            await gripper.open()
            t_open_done = time.monotonic() - t0

        if release and delay < 0:
            # The gripper reacts ~0.3-0.4 s after the command (measured Oct 29), so to let go
            # mid-swing the open command goes out BEFORE the swing starts.
            open_task = asyncio.create_task(open_gripper())
            await asyncio.sleep(-delay)
            t0_swing = time.monotonic()
            swing = asyncio.create_task(arm.move_to_joint_positions(JointPositions(values=end)))
        else:
            t0_swing = time.monotonic()
            swing = asyncio.create_task(arm.move_to_joint_positions(JointPositions(values=end)))
            if release:
                await asyncio.sleep(delay)
                open_task = asyncio.create_task(open_gripper())
        await swing
        t_swing = time.monotonic() - t0_swing
        if open_task is not None:
            await open_task
        # times relative to the START OF THE SWING (negative = before it)
        shift = t0_swing - t0
        timing = {"open_called_s": None if t_open_call is None else t_open_call - shift,
                  "open_returned_s": None if t_open_done is None else t_open_done - shift,
                  "swing_done_s": t_swing}
        print("   timing (s from swing start): " + ", ".join(
            f"{k.replace('_s', '')}={v:.2f}" for k, v in timing.items() if v is not None))
    except Exception as e:
        if "Linear speed exceeds limit" in str(e):
            print("!! the arm's controller stopped the swing: the gripper was moving faster than its linear "
                  "speed limit. Re-run with a lower --speed (about 10 deg/s less) or a smaller --scale.")
        raise
    finally:
        await arm.do_command({"set_speed": NORMAL_SPEED})
        if gripper_speed and release and prev_gs:
            try:
                await gripper.do_command({"set_gripper_speed": prev_gs})
            except Exception:  # noqa: BLE001
                pass
        print("3) speed restored")
    await _go_home(motion)
    return {"distance_mm": distance_mm, "release_delay_s": delay,
            "throw_speed": speed, "throw_accel": accel, "swing_deg": swing_deg, "scale": scale, "released": release,
            "timing": timing}


LOB_RELEASE_RADIUS_MM = 480   # about where the lob lets go, measured from the base (landed on the cup's distance)


def yaw_for_cup(x, y, warn=True):
    """Base rotation (deg) that points the lob plane at a cup at world (x, y) mm; also warns on distance."""
    import math
    yaw = math.degrees(math.atan2(y, x))
    r = math.hypot(x, y)
    print(f"   aim: cup at ({x:.0f}, {y:.0f}) -> yaw {yaw:+.1f} deg, {r:.0f} mm from the base")
    if warn and abs(r - LOB_RELEASE_RADIUS_MM) > 80:
        print("!! cup is more than 8 cm from where the lob lands: move the cup to ~48 cm from the base")
    return yaw


# FAR: open-hand toss, found by optimizing the swing on the xArm850 URDF (Oct 29, 19:00).
# Key facts it is built on:
#  * The controller's "Linear speed exceeds limit in ServoJ mode" (error 0x3C) limits the FLANGE speed,
#    not the ball: lob OK at 1.26 m/s flange, tripped at 1.44; a wrist-heavy throw ran with the gripper
#    at 1.6 m/s and the flange at 1.2. This swing keeps the flange at 1.14 m/s at 180 deg/s.
#  * A ball can only leave through the OPEN END of the fingers. Earlier throws had the fingers pointing
#    down/back, so the ball just dropped. Here the fingers point forward (start) and 22 deg up (end);
#    once the fingers open, gravity + the swing press the ball into the palm, and it flies off when the
#    arm brakes - so the exact gripper timing matters much less.
#  * Start: ball at (357, 0, 608) mm, fingers forward. End: (622, 0, 799), fingers 22 deg up.
#    Release (start of braking) ~(540, 0, 772), 26 deg up. Model landing at cup-rim height (z 100):
FAR_START_J = (55.9, -19.6, 11.1)          # J2, J3, J5
FAR_SWING = {1: -49.9, 2: -40.0, 4: +35.4}
FAR_ACCEL = 1145
FAR_GRIPPER_SPEED = 5000
FAR_TABLE = [(105, 837), (120, 866), (135, 895), (150, 924), (165, 953), (180, 983)]  # deg/s -> mm from base
FAR_OPEN_LEAD_S = 0.43    # gripper starts moving ~0.3 s after open(); have it open ~0.13 s before braking


# WHIP: wrist-whip throw, ~2x the FAR ball speed (1.9 m/s at 180 deg/s), launched ~42-53 deg up.
# Same optimizer, but the ball is HELD until the fingers open (timed release), so the wrist can
# add speed the flange limit doesn't see. Start: ball (367, 0, 561), fingers down. End: (671, 0, 795).
# Release (start of braking) ~(577, 0, 683). Flange 1.12 m/s at 180 deg/s (limit ~1.26-1.44).
# Model landing x1.05: live 116 deg/s landed ~1040 mm vs 989 modeled (15 cm past a cup at 892).
WHIP_START_J = (3.6, -75.5, -91.1)
WHIP_SWING = {1: -16.2, 2: -38.1, 4: +37.9}
WHIP_TABLE = [(s, round(d * 1.05)) for s, d in
              [(90, 897), (100, 931), (110, 966), (120, 1004), (135, 1065), (150, 1129), (165, 1197), (180, 1266)]]
WHIP_OPEN_LEAD_S = 0.40   # ball should be free right as the arm starts braking (fastest point)

# PITCH: underhand pitch with a real backswing (optimized Oct 29 19:30): 109 deg of wrist at full speed.
# Start: ball (255, 0, 263), fingers pointing back/up. The wrist sweeps the ball down and under (lowest
# ~19 cm above the base plane), releases ~(598, 0, 367) going 1.54 m/s, 63 deg up at 180 deg/s; lands
# steep (75 deg). Flange only 0.94 m/s. Table = model x1.05 (the WHIP landed ~5% past its model).
PITCH_START_J = (-57.7, -59.9, -116.9)
PITCH_SWING = {1: +58.8, 2: +2.8, 4: +109.1}
PITCH_TABLE = [(s, round(d * 1.05)) for s, d in [(120, 750), (135, 778), (150, 810), (165, 845), (180, 883)]]

PRESETS = {
    "far": dict(start_j=FAR_START_J, swing=FAR_SWING, table=FAR_TABLE, lead=FAR_OPEN_LEAD_S, min_speed=100.0),
    "pitch": dict(start_j=PITCH_START_J, swing=PITCH_SWING, table=PITCH_TABLE, lead=WHIP_OPEN_LEAD_S,
                  min_speed=120.0),
    "whip": dict(start_j=WHIP_START_J, swing=WHIP_SWING, table=WHIP_TABLE, lead=WHIP_OPEN_LEAD_S, min_speed=90.0),
}


def far_speed_for(distance_mm, preset="far"):
    """Swing speed (deg/s) whose landing is distance_mm from the base (clamped to min-180)."""
    t = PRESETS[preset]["table"]
    lo = PRESETS[preset]["min_speed"]
    if distance_mm <= t[0][1]:
        return max(lo, t[0][0] - (t[0][1] - distance_mm) / ((t[1][1] - t[0][1]) / (t[1][0] - t[0][0])))
    for (s0, d0), (s1, d1) in zip(t, t[1:]):
        if distance_mm <= d1:
            return s0 + (s1 - s0) * (distance_mm - d0) / (d1 - d0)
    return 180.0


def far_release_delay(speed_dps, accel=FAR_ACCEL, preset="far"):
    """Open-command time (s from swing start, negative = before) for the arm's braking point."""
    dmax = max(abs(v) for v in PRESETS[preset]["swing"].values())
    t_brake = speed_dps / accel + (dmax - speed_dps ** 2 / accel) / speed_dps
    return t_brake - PRESETS[preset]["lead"]


def far_params(distance_mm, speed=None, accel=None, release_delay_s=None, preset="far"):
    p = PRESETS[preset]
    speed = speed or far_speed_for(distance_mm, preset)
    accel = accel or FAR_ACCEL
    delay = far_release_delay(speed, accel, preset) if release_delay_s is None else release_delay_s
    print(f"   {preset.upper()}: cup {distance_mm:.0f} mm -> {speed:.0f} deg/s, open at {delay:+.2f} s, "
          f"gripper speed {FAR_GRIPPER_SPEED}")
    return dict(speed_dps=speed, accel_dps2=accel, release_delay_s=delay, swing_deg=dict(p["swing"]),
                start_j=p["start_j"], gripper_speed=FAR_GRIPPER_SPEED)


def swing_from_args(a):
    """SWING_DEG with any --j2/--j3/--j5 overrides (joint index = J number - 1)."""
    sw = dict(LOB_SWING if getattr(a, "lob", False) else SWING_DEG)
    for j, v in ((1, a.j2), (2, a.j3), (4, a.j5)):
        if v is not None:
            sw[j] = v
    return sw


# xArm firmware limit seen live (Oct 29): "Linear speed exceeds limit in ServoJ mode" stopped the lob
# mid-swing at ~2.8 m/s of gripper speed (J2 180 deg/s). Earlier swings peaked ~1.65 m/s and ran
# fine. MEASURED with scripts/lob_speed_sweep.py (Oct 29, 18:36): lob swing OK at 50/60/70 deg/s,
# tripped at 80. So the controller caps the lob at roughly 1.1 m/s of gripper speed (my Jacobian
# estimate: 2.79 m/s * speed/180). 65 keeps a margin under the last good 70.
LOB_SPEED_DPS = 65

# LOB: ball leaves almost straight UP at full speed. Measured Oct 29 at the middle of the swing
# (gripper mm per degree, forward/up): J2+ (-2.2, +8.2), J3+ (-4.8, -7.3), J5+ (+3.4, 0).
# J2 +180, J3 -180, J5 -111 deg/s -> ~2.8 m/s up, ~0.1 m/s forward. The path is a vertical lift
# (gripper 20 -> 80 cm above the table at x~370-480), released mid-swing; lands ~0.5 m from the base.
LOB_START_J = (-25.0, -45.9, -48.3)            # J2, J3, J5 at the start of the swing
LOB_SWING = {1: +40.0, 2: -40.0, 4: -25.0}      # end: J2 +15, J3 -85.9, J5 -73.3 (all within limits)


async def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=float, required=True, help="cup distance from base, mm")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="0.1-1: shrink swing angles and speed (use 0.2-0.3 for the first live look)")
    ap.add_argument("--no-release", action="store_true", help="swing but never open the gripper")
    ap.add_argument("--release-delay", type=float, default=None,
                    help="seconds after the swing starts to send the open command (default: table). "
                         "NEGATIVE = open first: the gripper takes ~0.4 s to let go, so try -0.3 / -0.2 / -0.1")
    ap.add_argument("--speed", type=float, default=None, help="swing speed deg/s (default 150, max 180)")
    ap.add_argument("--accel", type=float, default=None, help="swing accel deg/s2 (default 900, max 1145)")
    ap.add_argument("--j2", type=float, default=None,
                    help="shoulder swing deg. MEASURED: negative = gripper forward+down, positive = back+up")
    ap.add_argument("--j3", type=float, default=None, help="elbow swing deg (default -35: up and forward)")
    ap.add_argument("--j5", type=float, default=None, help="wrist swing deg (default +45: tips forward)")
    ap.add_argument("--wind-z", type=float, default=None,
                    help="wind-up height mm (default 450). Higher start = longer swing path = a higher lob (try 650)")
    ap.add_argument("--far", action="store_true",
                    help="open-hand forward toss for a cup 84-98 cm from the base (speed picked from --distance)")
    ap.add_argument("--pitch", action="store_true",
                    help="underhand pitch: long 109-deg backswing at full speed, steep 63-deg launch (cup ~79-93 cm)")
    ap.add_argument("--whip", action="store_true",
                    help="wrist-whip throw, ~2x faster, 42-53 deg up (cup ~85-115 cm; speed from --distance)")
    ap.add_argument("--yaw", type=float, default=0.0, help="turn the lob toward +y by this many deg (--lob only)")
    ap.add_argument("--lob", action="store_true",
                    help="near-vertical full-speed toss (cup ~50 cm from the base); see LOB_* in this file")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--i-checked-with-staff", action="store_true")
    a = ap.parse_args()
    go = a.go and a.i_checked_with_staff
    if a.go and not go:
        print("Refusing to throw live without --i-checked-with-staff (direct joint move, no obstacle checks).")
    robot = await connect()
    try:
        if a.far or a.whip or a.pitch:
            fp = far_params(a.distance, a.speed, a.accel, a.release_delay,
                            "pitch" if a.pitch else ("whip" if a.whip else "far"))
            print(await throw(robot, a.distance, go, a.scale, not a.no_release, fp["release_delay_s"],
                              fp["swing_deg"], fp["speed_dps"], fp["accel_dps2"], None, fp["start_j"], a.yaw,
                              fp["gripper_speed"]))
        else:
            print(await throw(robot, a.distance, go, a.scale, not a.no_release, a.release_delay,
                              swing_from_args(a), a.speed, a.accel, a.wind_z,
                              LOB_START_J if a.lob else None, a.yaw))
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(_main())
