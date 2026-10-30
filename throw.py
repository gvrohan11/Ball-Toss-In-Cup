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
                swing_deg=None, speed_dps=None, accel_dps2=None, wind_z=None, start_j=None):
    """scale (0.1..1) shrinks the swing angles AND the speed: use 0.2-0.3 first, empty gripper,
    release=False, to watch the motion before any real throw."""
    if not 0.1 <= scale <= 1.0:
        raise ValueError("--scale must be between 0.1 and 1")
    motion = MotionClient.from_robot(robot, "builtin")
    arm, gripper = Arm.from_robot(robot, ARM), Gripper.from_robot(robot, GRIPPER)
    delay = release_delay(distance_mm) if release_delay_s is None else float(release_delay_s)
    if not -0.6 <= delay <= 0.6:
        raise ValueError("--release-delay must be between -0.6 and 0.6 s")
    wind = dict(WIND_UP)
    if wind_z is not None:
        if not 300 <= wind_z <= 800:
            raise ValueError("--wind-z must be between 300 and 800 mm")
        wind["z"] = wind_z
    if start_j is not None:
        # Direct joint move to a known start configuration (J2, J3, J5; the other joints stay as
        # they are). Used by the lob, whose swing is designed around exact joint angles.
        print(f"1) wind up (joint move): J2={start_j[0]} J3={start_j[1]} J5={start_j[2]}")
        if go:
            pose = await arm.get_end_position()
            if abs(pose.x - HOME["x"]) > 90 or abs(pose.y - HOME["y"]) > 90 or pose.z < 350:
                raise RuntimeError(f"Start from home first (arm at {pose.x:.0f}, {pose.y:.0f}, {pose.z:.0f}): "
                                   "this wind-up is a direct joint move. Run `python home.py`.")
            cur = list((await arm.get_joint_positions()).values)
            target = [cur[0], start_j[0], start_j[1], cur[3], start_j[2], cur[5]]
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
        [0.0, start_j[0], start_j[1], 0.0, start_j[2], 0.0] if start_j else [0.0] * 6)
    end = list(start)
    swing_deg = dict(SWING_DEG if swing_deg is None else swing_deg)
    for j, d in swing_deg.items():
        end[j] += d * scale
    check_limits(start, end)
    speed = max(20, (speed_dps or THROW_SPEED) * scale)
    accel = max(100, (accel_dps2 or THROW_ACCEL) * scale)
    if speed > MAX_SPEED or accel > MAX_ACCEL:
        raise ValueError(f"speed/accel above the module limit ({MAX_SPEED} deg/s, {MAX_ACCEL} deg/s2)")
    print(f"2) swing joints {[round(v, 1) for v in start]} -> {[round(v, 1) for v in end]} "
          f"at {speed:.0f} deg/s, " + (f"release after {delay * 1000:.0f} ms" if release else "NO release")
          + f" (target {distance_mm} mm, scale {scale})")
    if not go:
        return {"distance_mm": distance_mm, "release_delay_s": delay, "dry_run": True}
    try:
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
    finally:
        await arm.do_command({"set_speed": NORMAL_SPEED})
        print("3) speed restored")
    await _go_home(motion)
    return {"distance_mm": distance_mm, "release_delay_s": delay,
            "throw_speed": speed, "throw_accel": accel, "swing_deg": swing_deg, "scale": scale, "released": release,
            "timing": timing}


def swing_from_args(a):
    """SWING_DEG with any --j2/--j3/--j5 overrides (joint index = J number - 1)."""
    sw = dict(LOB_SWING if getattr(a, "lob", False) else SWING_DEG)
    for j, v in ((1, a.j2), (2, a.j3), (4, a.j5)):
        if v is not None:
            sw[j] = v
    return sw


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
        print(await throw(robot, a.distance, go, a.scale, not a.no_release, a.release_delay,
                          swing_from_args(a), a.speed, a.accel, a.wind_z,
                          LOB_START_J if a.lob else None))
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(_main())
