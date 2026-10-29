"""Manan: the full loop. find ball -> pick -> find cup -> drop -> home -> log.

python main.py          # dry run: prints the plan, moves nothing
python main.py --go     # runs on the arm (someone on the E-stop!)

Uses the sequences that worked on sfhack11 (Oct 29): pick_ball.py (2/2 picks),
cup_aim.py + drop.py (2/2 drops). Swap in balls.find_ball / cups.get_cups when
the ball and cup teammates' versions are reliable.
"""
import asyncio
import sys

from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient

from connection import connect
from cup_aim import CUP_LOOK, find_cup
from drop import drop_in_cup
from logger import log_attempt
from pick_ball import GRIPPER, LOOK, THETA, find_ball, pick


async def move_to(motion, p, go):
    print(f"  -> gripper to {p}")
    if go:
        await motion.move(component_name=GRIPPER,
                          destination=PoseInFrame(reference_frame="world", pose=Pose(**p)))


async def main():
    go = "--go" in sys.argv
    print("MODE:", "LIVE (arm will move)" if go else "DRY RUN (no motion)")
    home = dict(x=LOOK[0], y=LOOK[1], z=LOOK[2], o_x=0, o_y=0, o_z=-1, theta=THETA)
    robot = await connect()
    try:
        motion = MotionClient.from_robot(robot, "builtin")

        print("0) start from the saved home pose")
        await move_to(motion, home, go)

        # Find the cup FIRST, while the gripper is empty: a held orange ball in view
        # throws off cup-detector.
        print("1) find cup")
        await move_to(motion, CUP_LOOK, go)
        cup = await find_cup(robot)
        if cup is None:
            print("no cup found - check cup-detector at the cup-look pose")
            return
        print(f"  cup rim center: {cup[0]:.0f}, {cup[1]:.0f}, {cup[2]:.0f}")

        print("2) find ball")
        ball = await find_ball(robot, motion, go)
        if ball is None:
            log_attempt({"result": {"hit": False, "failure": "no_ball_found"}})
            print("no ball found")
            return

        print("3) pick")
        held = await pick(robot, ball, go)
        if go and not held:
            log_attempt({"ball": {"pos_mm": ball}, "result": {"hit": False, "failure": "grasp_failed"}})
            print("pick missed")
            await move_to(motion, home, go)
            return

        print("4) drop")
        result = await drop_in_cup(robot, cup, go)

        print("5) home")
        await move_to(motion, home, go)

        hit = None
        if go:
            hit = input("Did it go in? [y/n] ").strip().lower().startswith("y")
        log_attempt({"phase": "drop", "ball": {"pos_mm": ball}, "cup": {"pos_mm": cup},
                     "release": result, "result": {"hit": hit}})
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
