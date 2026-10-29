"""Remember cup in world frame -> pick -> planned transfer -> verify -> release.

python main.py          # dry run: prints the plan, moves nothing
python main.py --go     # runs on the arm (someone on the E-stop!)

Uses two-view rim triangulation before picking, world-frame cup memory,
continuous planned-motion feedback and visual release verification.
Default mode observes the current frame without moving hardware.
"""
import asyncio
import sys

from connection import connect
from cup_tracking import scan_cup, observe_rims
from drop import drop_at_remembered_cup
from motion_control import MotionController
from logger import log_attempt
from pick_ball import LOOK, THETA, find_ball, pick


def record_attempt(doc, go):
    if go:
        log_attempt(doc)
    else:
        print('DRY RUN: attempt not saved:', doc)


async def main():
    go = "--go" in sys.argv
    print("MODE:", "LIVE (arm will move)" if go else "DRY RUN (no motion)")
    home = dict(x=LOOK[0], y=LOOK[1], z=LOOK[2], o_x=0, o_y=0, o_z=-1, theta=THETA)
    robot = await connect()
    try:
        controller = MotionController(robot, go)
        if not go:
            rims = await observe_rims(robot)
            controller.event('dry_run', rim_candidates=len(rims),
                             sequence='remember cup in two views -> pick -> planned transfer -> verify -> release')
            print('Two-view cup acquisition needs camera motion; no saved cup is reused in dry run.')
            return
        await controller.arm.do_command({'set_speed': 10, 'set_acceleration': 20})
        print('1) locate and remember the nearest cup BEFORE picking')
        cup_memory = await scan_cup(robot, controller)
        controller.cup_memory = cup_memory
        cup = cup_memory.center
        print('2) find and pick orange ball')
        ball = await find_ball(robot, controller, go)
        if ball is None:
            record_attempt({'result': {'hit': False, 'failure': 'no_ball_found'}}, go)
            return
        held = await pick(robot, ball, go, controller=controller)
        if not held:
            record_attempt({'ball': {'pos_mm': ball}, 'result': {'hit': False, 'failure': 'grasp_failed'}}, go)
            await controller.move(home, 'home_after_missed_pick')
            return
        controller.event('holding_ball', cup_world_mm=cup)
        controller.monitor_payload = True
        print('3) transfer directly to remembered cup; verify before releasing')
        result = await drop_at_remembered_cup(robot, cup_memory, controller)

        print("5) home")
        await controller.move(home, "home")

        hit = None
        if go:
            hit = input("Did it go in? [y/n] ").strip().lower().startswith("y")
        record_attempt({"phase": "drop", "ball": {"pos_mm": ball}, "cup": {"pos_mm": cup},
                     "release": result, "result": {"hit": hit}}, go)
    except Exception as exc:
        print(f'RUN STOPPED: {exc}. No automatic release or recovery move.')
        raise
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
