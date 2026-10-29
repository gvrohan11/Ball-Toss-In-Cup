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
SWING_DEG = {2: -35, 4: -45}  # joint index -> forward swing (J3 elbow, J5 wrist), degrees
# release delay after the swing starts, per target distance (mm) - placeholders to tune
RELEASE_TABLE = [(900, 0.16), (1100, 0.20), (1300, 0.24)]
WIND_UP = dict(x=300, y=0, z=450, o_x=-0.5, o_y=0, o_z=-0.866, theta=180)
HOME = dict(x=350, y=0, z=445, o_x=0, o_y=0, o_z=-1, theta=180)


def release_delay(distance_mm):
    pts = sorted(RELEASE_TABLE)
    if distance_mm <= pts[0][0]:
        return pts[0][1]
    for (d0, t0), (d1, t1) in zip(pts, pts[1:]):
        if distance_mm <= d1:
            return t0 + (t1 - t0) * (distance_mm - d0) / (d1 - d0)
    return pts[-1][1]


async def throw(robot, distance_mm, go):
    motion = MotionClient.from_robot(robot, "builtin")
    arm, gripper = Arm.from_robot(robot, ARM), Gripper.from_robot(robot, GRIPPER)
    delay = release_delay(distance_mm)
    print(f"1) wind up (planned move) to {WIND_UP}")
    if go:
        ok = await motion.move(GRIPPER, PoseInFrame(reference_frame="world", pose=Pose(**WIND_UP)))
        if ok is False:
            raise RuntimeError("wind-up move failed")
    start = list((await arm.get_joint_positions()).values) if go else [0.0] * 6
    end = list(start)
    for j, d in SWING_DEG.items():
        end[j] += d
    print(f"2) swing joints {[round(v, 1) for v in start]} -> {[round(v, 1) for v in end]} "
          f"at {THROW_SPEED} deg/s, release after {delay * 1000:.0f} ms (target {distance_mm} mm)")
    if not go:
        return {"distance_mm": distance_mm, "release_delay_s": delay, "dry_run": True}
    try:
        await arm.do_command({"set_speed": THROW_SPEED, "set_acceleration": THROW_ACCEL})
        swing = asyncio.create_task(arm.move_to_joint_positions(JointPositions(values=end)))
        await asyncio.sleep(delay)
        await gripper.open()
        await swing
    finally:
        await arm.do_command({"set_speed": NORMAL_SPEED})
        print("3) speed restored")
    await motion.move(GRIPPER, PoseInFrame(reference_frame="world", pose=Pose(**HOME)))
    return {"distance_mm": distance_mm, "release_delay_s": delay,
            "throw_speed": THROW_SPEED, "swing_deg": SWING_DEG}


async def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=float, required=True, help="cup distance from base, mm")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--i-checked-with-staff", action="store_true")
    a = ap.parse_args()
    go = a.go and a.i_checked_with_staff
    if a.go and not go:
        print("Refusing to throw live without --i-checked-with-staff (direct joint move, no obstacle checks).")
    robot = await connect()
    try:
        print(await throw(robot, a.distance, go))
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(_main())
