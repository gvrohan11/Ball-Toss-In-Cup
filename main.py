"""Manan: the full loop. find ball -> pick -> find cup -> drop -> home -> log.

python main.py          # dry run: prints the plan, moves nothing
python main.py --go     # runs on the arm (someone on the E-stop!)
python main.py --go --autotune   # also shift the aim by the average of recent misses in Atlas

Uses the sequences that worked on sfhack11 (Oct 29): pick_ball.py (2/2 picks),
cup_aim.py + drop.py (2/2 drops). Swap in balls.find_ball / cups.get_cups when
the ball and cup teammates' versions are reliable.
"""
import asyncio
import sys
import time

from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient

from connection import connect
from cup_aim import CUP_LOOKS, find_cup
from drop import drop_in_cup
from logger import log_attempt, suggest_offset
from pick_ball import GRIPPER, LOOK, THETA, find_ball, pick


async def move_to(motion, p, go):
    print(f"  -> gripper to {p}")
    if go:
        await motion.move(component_name=GRIPPER,
                          destination=PoseInFrame(reference_frame="world", pose=Pose(**p)))


def ask_result():
    """Ask the operator how it went. Returns (hit, miss_mm or None).
    Miss direction is from the robot's point of view: +x = past the cup (away from the
    robot), +y = to the robot's left. Used by logger.suggest_offset() to auto-tune."""
    hit = input("Did it go in? [y/n] ").strip().lower().startswith("y")
    if hit:
        return True, None
    raw = input("Where did it land vs the cup? cm past(+)/short(-), cm left(+)/right(-) "
                "as seen from the robot, e.g. '2,-1' (Enter to skip): ").strip()
    if not raw:
        return False, None
    try:
        dx, dy = (float(v) * 10 for v in raw.split(","))
        return False, {"x": dx, "y": dy}
    except ValueError:
        print("  couldn't read that, logging the miss without a direction")
        return False, None


async def main():
    go = "--go" in sys.argv
    print("MODE:", "LIVE (arm will move)" if go else "DRY RUN (no motion)")
    t0 = time.monotonic()
    home = dict(x=LOOK[0], y=LOOK[1], z=LOOK[2], o_x=0, o_y=0, o_z=-1, theta=THETA)
    robot = await connect()
    try:
        motion = MotionClient.from_robot(robot, "builtin")

        print("0) start from the saved home pose")
        await move_to(motion, home, go)

        print("1) find ball")
        ball = await find_ball(robot, motion, go)
        if ball is None:
            log_attempt({"result": {"hit": False, "failure": "no_ball_found"}})
            print("no ball found")
            return

        print("2) pick")
        held = await pick(robot, ball, go)
        if go and not held:
            log_attempt({"ball": {"pos_mm": ball}, "result": {"hit": False, "failure": "grasp_failed"}})
            print("pick missed")
            await move_to(motion, home, go)
            return

        # Zoom out and look across the table at an angle to find the cup:
        # 30 deg first (closer cups), then 45 deg (far end of the table).
        print("3) find cup")
        cup = None
        for look in CUP_LOOKS:
            await move_to(motion, look, go)
            cup = await find_cup(robot)
            if cup is not None:
                break
        if cup is None:
            print("no cup found - check cup-detector at the cup-look pose; going home holding the ball")
            await move_to(motion, home, go)
            return
        print(f"  cup rim center: {cup[0]:.0f}, {cup[1]:.0f}, {cup[2]:.0f}")

        extra = (0, 0)
        if "--autotune" in sys.argv:
            extra = suggest_offset()
            print(f"  autotune from Atlas misses: {extra[0]:+.0f}, {extra[1]:+.0f} mm")
        print("4) drop (centered on the cup)")
        result = await drop_in_cup(robot, cup, go, extra)

        print("5) home")
        await move_to(motion, home, go)

        hit, miss = None, None
        if go:
            hit, miss = ask_result()
        res = {"hit": hit}
        if miss is not None:
            res["miss_mm"] = miss
            res["failure"] = "missed"
        log_attempt({"phase": "drop", "ball": {"pos_mm": ball}, "cup": {"pos_mm": cup},
                     "release": result, "result": res,
                     "duration_s": round(time.monotonic() - t0, 1)})
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
