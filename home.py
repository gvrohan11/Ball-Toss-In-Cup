"""Send the arm to the saved safe home pose (350, 0, 445, gripper straight down, theta 180).

    python home.py            # move home (planned, obstacle-checked move)
    python home.py --open     # also open the gripper first (drops a held ball!)

Retries a few times because the robot sometimes cancels the first move after a fast swing.
"""
import asyncio
import sys

from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient

from connection import connect

HOME = dict(x=350, y=0, z=445, o_x=0, o_y=0, o_z=-1, theta=180)


async def go_home(robot, tries=4):
    motion = MotionClient.from_robot(robot, "builtin")
    for i in range(tries):
        try:
            if await motion.move("gripper", PoseInFrame(reference_frame="world", pose=Pose(**HOME))) is not False:
                return True
        except Exception as e:  # noqa: BLE001
            print(f"  try {i + 1}/{tries} failed: {type(e).__name__}: {str(e)[:80]}")
        await asyncio.sleep(1.0)
    return False


async def main():
    robot = await connect()
    try:
        if "--open" in sys.argv:
            await Gripper.from_robot(robot, "gripper").open()
        print("HOME" if await go_home(robot) else "!! could not reach home - use the Viam app")
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
