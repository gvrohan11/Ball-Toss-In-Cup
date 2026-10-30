"""Phase 2: find the cup, pick the ball, THROW it, log the result to Atlas.

    python pick_and_throw.py --aim-cup                                  # dry run: no motion
    python pick_and_throw.py --aim-cup --go --i-checked-with-staff      # FAR throw at the cup it sees
    python pick_and_throw.py --distance 950 --go --i-checked-with-staff # no scan: cup straight ahead
    python pick_and_throw.py --aim-cup --power --go --i-checked-with-staff   # far cup, 1-2 m

Flow: home -> set down anything held -> (--aim-cup) scan for the cup, aim the base at it ->
plan + safety-check the throw BEFORE picking -> find + pick the ball -> throw -> home ->
ask what happened -> log to Atlas (phase "throw"). throw_fit learns from these logs.
If anything fails after the pick, the ball is put back on the table.
"""
import argparse
import asyncio
import math
import re
import time

import throw
from connection import connect
from main import move_to, put_ball_back, release_if_holding
from pick_ball import LOOK, THETA, find_ball, pick
from viam.services.motion import MotionClient

HOME = dict(x=LOOK[0], y=LOOK[1], z=LOOK[2], o_x=0, o_y=0, o_z=-1, theta=THETA)

PROMPT = ("Result? 'in' | 'rim' (reached the cup, bounced out) | cm long(+)/short(-), optionally "
          "',left(+)/right(-) cm' e.g. '12' or '-8,3' | 'drop' (fell out early) | blank = skip: ")


def parse_landing(s):
    """Typed answer -> result dict (None = skip). Every miss gets an explicit outcome, so a 0 is never
    ambiguous again (Oct 29: 34 of 69 misses were logged as long_cm 0)."""
    s = s.strip().lower()
    if not s:
        return None
    if s in ("in", "y", "yes", "hit"):
        return {"hit": True, "outcome": "in"}
    if s.startswith(("rim", "bounce", "b")):
        return {"hit": False, "outcome": "rim_out", "long_cm": 0.0}
    if s.startswith(("drop", "d", "fell")):
        return {"hit": False, "outcome": "dropped"}
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]
    if not nums:
        return None
    long_cm = nums[0]
    res = {"hit": False, "long_cm": long_cm,
           "outcome": "long" if long_cm > 0 else "short" if long_cm < 0 else "rim_out"}
    if len(nums) > 1:
        res["left_cm"] = nums[1]
    return res


def ask_landing():
    try:
        return parse_landing(input(PROMPT))
    except EOFError:
        return None


async def run(args):
    go = args.go and args.i_checked_with_staff
    if args.go and not go:
        print("Refusing to throw live without --i-checked-with-staff.")
    print("MODE:", "LIVE (arm will move and throw)" if go else "DRY RUN (no motion)")
    preset = throw.preset_from_args(args)
    t0 = time.monotonic()
    robot = await connect()
    try:
        motion = MotionClient.from_robot(robot, "builtin")
        await move_to(motion, HOME, go, "home")
        await release_if_holding(robot, motion, go)
        await move_to(motion, HOME, go, "home")

        cup_doc, distance, yaw = None, args.distance, args.yaw
        if args.aim_cup:
            from main import scan_for_cup
            import cup_aim
            cup_aim.MAX_CUP_REACH_MM = 2300   # throws reach past the drop reach
            print("0) find the cup")
            cup = await scan_for_cup(motion, robot, go)
            await move_to(motion, HOME, go, "home")
            if cup is None:
                print("no cup found - not throwing")
                return
            yaw = throw.yaw_for_cup(cup[0], cup[1])
            distance = math.hypot(cup[0], cup[1])
            cup_doc = {"pos_mm": list(cup), "dist_mm": round(distance)}
        if distance is None:
            print("give --distance (mm from the base) or --aim-cup")
            return

        pl = throw.plan_from_args(args, distance, yaw)   # raises before anything is picked if unsafe
        lo, hi = throw.reach_mm(preset, pl["ratio"])
        if not lo <= distance <= hi and args.speed is None and not args.force_aim:
            hint = "try --power" if preset == "far" and distance > hi else (
                "drop --power" if preset == "power" else "move the cup")
            print(f"cup is {distance:.0f} mm from the base; {preset.upper()} reaches {lo}-{hi} mm: "
                  f"{hint}, or --force-aim.")
            await move_to(motion, HOME, go, "home")
            return

        print("1) find ball")
        ball = await find_ball(robot, motion, go)
        if ball is None:
            print("no ball found")
            return
        print("2) pick")
        held = await pick(robot, ball, go)
        if go and not held:
            print("pick missed; nothing to throw")
            await move_to(motion, HOME, go, "home")
            return

        try:
            print("3) throw")
            await move_to(motion, HOME, go, "home")
            info = await throw.throw(robot, pl, go, not args.no_release, True if args.direct else None,
                                     partial=args.partial, check_only=args.cradle_check)
        except BaseException as e:  # error or Ctrl+C: don't leave the ball in the gripper
            print(f"!! stopped ({type(e).__name__}: {e}); putting the ball back")
            await asyncio.sleep(2.0)  # after a controller stop the arm refuses commands for a moment
            await put_ball_back(robot, motion, ball, go)
            raise
        await move_to(motion, HOME, go, "home")
        if go and args.cradle_check:
            print("cradle check done: did the ball stay in? were the fingers above/below it? (not logged)")
            await put_ball_back(robot, motion, ball, go)
            return
        if go:
            landing = ask_landing()
            doc = {"phase": "throw", "ball": {"pos_mm": ball}, "cup": cup_doc, "throw": info,
                   "duration_s": round(time.monotonic() - t0, 1),
                   "result": landing or {"hit": None, "outcome": "not_recorded"}}
            try:
                import logger
                logger.log_attempt(doc)
            except Exception as e:  # noqa: BLE001
                print(f"  [log failed: {e}]")
    finally:
        await robot.close()


def main():
    ap = argparse.ArgumentParser(description="Pick the ball, then throw it at the cup.")
    throw.add_args(ap)
    ap.add_argument("--aim-cup", action="store_true", help="scan for the cup and aim + range the throw at it")
    ap.add_argument("--force-aim", action="store_true", help="throw even if the cup is outside the preset's range")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
