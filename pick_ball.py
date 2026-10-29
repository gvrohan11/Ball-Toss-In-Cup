"""
pick_ball.py - find the orange ball with ball-3d and pick it up with the gripper.

Setup (once, on your laptop):
    see README (conda env)

Credentials come from .env via connection.py (see .env.example).

Run:
    python pick_ball.py          # DRY RUN: only prints where it would go
    python pick_ball.py --go     # actually moves the arm (E-stop person ready!)
"""
import asyncio
import sys

from viam.components.gripper import Gripper
from viam.services.vision import VisionClient
from viam.services.motion import MotionClient
from viam.proto.common import Pose, PoseInFrame

from connection import connect

CAMERA = "cam"
BALL_3D = "ball-3d"
GRIPPER = "gripper"

# "look" pose for the GRIPPER (world frame, mm). Gripper points straight down.
# (Arm flange at z=550 == gripper tip at about z=445.)
LOOK = dict(x=350, y=0, z=445)
HOVER_MM = 100          # how far above the ball to stop before descending
GRASP_OFFSET_MM = 5     # grab slightly above ball center
TABLE_SAFE_Z = 0        # never send the gripper below this (table top is ~ -23)

DOWN = dict(o_x=0, o_y=0, o_z=-1, theta=0)


def world_pose(x, y, z):
    return PoseInFrame(reference_frame="world", pose=Pose(x=x, y=y, z=z, **DOWN))


async def find_ball(robot):
    vision = VisionClient.from_robot(robot, BALL_3D)
    objs = await vision.get_object_point_clouds(CAMERA)
    if not objs:
        return None
    found = []
    for o in objs:
        frame = o.geometries.reference_frame
        for g in o.geometries.geometries:
            w = await robot.transform_pose(PoseInFrame(reference_frame=frame, pose=g.center), "world")
            p = w.pose
            found.append((p.x, p.y, p.z, g.label))
            print(f"  candidate {g.label!r}: world x={p.x:.0f} y={p.y:.0f} z={p.z:.0f}")
    # ball sits on the table, so prefer the lowest candidate near the table
    found.sort(key=lambda c: c[2])
    return found[0]


async def move_gripper(motion, x, y, z, go):
    z = max(z, TABLE_SAFE_Z)
    print(f"  -> gripper to x={x:.0f} y={y:.0f} z={z:.0f}")
    if go:
        await motion.move(component_name=GRIPPER, destination=world_pose(x, y, z))


async def main():
    go = "--go" in sys.argv
    print("MODE:", "LIVE (arm will move)" if go else "DRY RUN (no motion)")
    robot = await connect()
    try:
        motion = MotionClient.from_robot(robot, "builtin")
        gripper = Gripper.from_robot(robot, GRIPPER)

        print("1) go to look pose")
        await move_gripper(motion, LOOK["x"], LOOK["y"], LOOK["z"], go)

        print("2) find ball")
        ball = await find_ball(robot)
        if ball is None:
            print("  no ball found - check ball-detector in the CONTROL tab")
            return
        bx, by, bz, _ = ball

        print("3) open gripper")
        if go:
            await gripper.open()

        print("4) hover above ball")
        await move_gripper(motion, bx, by, bz + HOVER_MM, go)

        print("5) descend")
        await move_gripper(motion, bx, by, bz + GRASP_OFFSET_MM, go)

        print("6) grab")
        if go:
            holding = await gripper.grab()
            print("  holding something:", holding)

        print("7) lift")
        await move_gripper(motion, bx, by, bz + HOVER_MM, go)

        print("8) back to look pose")
        await move_gripper(motion, LOOK["x"], LOOK["y"], LOOK["z"], go)
        print("done")
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
