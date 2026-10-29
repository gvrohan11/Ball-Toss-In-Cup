"""Manan: the main loop. Glues everyone's modules together.

python main.py          # dry run, prints the plan, moves nothing
python main.py --go     # runs on the arm (someone on the E-stop!)
"""
import asyncio
import sys

from connection import connect
from cups import get_cups
from drop import drop_in_cup
from logger import log_attempt
from balls import find_ball      # ball teammate
# TODO (Manan): pick(robot, ball, go) from pick_ball.py


async def main():
    go = "--go" in sys.argv
    print("MODE:", "LIVE" if go else "DRY RUN")
    robot = await connect()
    try:
        ball = await find_ball(robot)
        if ball is None:
            print("no ball found")
            return
        print("ball:", ball[:3])
        # await pick(robot, ball, go)   # TODO (Manan)

        cups = await get_cups(robot)     # Rohan
        if not cups:
            print("no cups found")
            return
        cup = cups[0]
        print("cup:", cup)

        result = await drop_in_cup(robot, cup, go)  # Manan
        hit = None  # TODO: ask a human (input) or check the camera
        log_attempt({"ball": list(ball[:3]), "cup": list(cup), **result, "hit": hit})  # MongoDB teammate
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
