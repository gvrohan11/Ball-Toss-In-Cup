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
THROW_SPEED = 150            # deg/s, module max is 180
THROW_ACCEL = 900            # deg/s^2, module max is 1145
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
RELEASE_TABLE = [(900, 0.03), (1100, 0.05), (1300, 0.08)]
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


async def throw(robot, distance_mm, go, scale=1.0, release=True, release_delay_s=None):
    """scale (0.1..1) shrinks the swing angles AND the speed: use 0.2-0.3 first, empty gripper,
    release=False, to watch the motion before any real throw."""
    if not 0.1 <= scale <= 1.0:
        raise ValueError("--scale must be between 0.1 and 1")
    motion = MotionClient.from_robot(robot, "builtin")
    arm, gripper = Arm.from_robot(robot, ARM), Gripper.from_robot(robot, GRIPPER)
    delay = release_delay(distance_mm) if release_delay_s is None else float(release_delay_s)
    if not 0.0 <= delay <= 0.6:
        raise ValueError("--release-delay must be between 0 and 0.6 s")
    print(f"1) wind up (planned move) to {WIND_UP}")
    if go:
        ok = await motion.move(GRIPPER, PoseInFrame(reference_frame="world", pose=Pose(**WIND_UP)))
        if ok is False:
            raise RuntimeError("wind-up move failed")
    start = list((await arm.get_joint_positions()).values) if go else [0.0] * 6
    end = list(start)
    for j, d in SWING_DEG.items():
        end[j] += d * scale
    check_limits(start, end)
    speed, accel = max(20, THROW_SPEED * scale), max(100, THROW_ACCEL * scale)
    print(f"2) swing joints {[round(v, 1) for v in start]} -> {[round(v, 1) for v in end]} "
          f"at {speed:.0f} deg/s, " + (f"release after {delay * 1000:.0f} ms" if release else "NO release")
          + f" (target {distance_mm} mm, scale {scale})")
    if not go:
        return {"distance_mm": distance_mm, "release_delay_s": delay, "dry_run": True}
    try:
        await arm.do_command({"set_speed": speed, "set_acceleration": accel})
        t0 = time.monotonic()
        swing = asyncio.create_task(arm.move_to_joint_positions(JointPositions(values=end)))
        t_open_call = t_open_done = None
        if release:
            await asyncio.sleep(delay)
            t_open_call = time.monotonic() - t0
            await gripper.open()
            t_open_done = time.monotonic() - t0
        await swing
        t_swing = time.monotonic() - t0
        timing = {"open_called_s": t_open_call, "open_returned_s": t_open_done, "swing_done_s": t_swing}
        print("   timing (s from swing start): " + ", ".join(
            f"{k.replace('_s', '')}={v:.2f}" for k, v in timing.items() if v is not None))
    finally:
        await arm.do_command({"set_speed": NORMAL_SPEED})
        print("3) speed restored")
    await _go_home(motion)
    return {"distance_mm": distance_mm, "release_delay_s": delay,
            "throw_speed": speed, "swing_deg": SWING_DEG, "scale": scale, "released": release,
            "timing": timing}


async def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=float, required=True, help="cup distance from base, mm")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="0.1-1: shrink swing angles and speed (use 0.2-0.3 for the first live look)")
    ap.add_argument("--no-release", action="store_true", help="swing but never open the gripper")
    ap.add_argument("--release-delay", type=float, default=None,
                    help="seconds after the swing starts to open the gripper (default: table). "
                         "Dropping straight down = too late; try 0.0, 0.05, 0.1")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--i-checked-with-staff", action="store_true")
    a = ap.parse_args()
    go = a.go and a.i_checked_with_staff
    if a.go and not go:
        print("Refusing to throw live without --i-checked-with-staff (direct joint move, no obstacle checks).")
    robot = await connect()
    try:
        print(await throw(robot, a.distance, go, a.scale, not a.no_release, a.release_delay))
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(_main())
