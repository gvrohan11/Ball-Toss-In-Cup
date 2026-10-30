"""Phase 2: pick the ball up, THEN throw it. (throw.py alone only swings: it never picks.)

    python pick_and_throw.py --distance 1100                       # dry run: no motion
    python pick_and_throw.py --distance 1100 --scale 0.3 --go --i-checked-with-staff
    python pick_and_throw.py --distance 1100 --go --i-checked-with-staff   # full throw

Flow: home -> set down anything held -> find + pick the ball (same code as main.py) -> check the
gripper really holds it -> throw.py's swing -> back home -> ask where it landed -> log to Atlas
(phase "throw", never counted in the drop hit rate).
If anything fails after the pick, the ball is put back on the table.
"""
import argparse
import asyncio
import time

import throw
from connection import connect
from main import move_to, put_ball_back, release_if_holding
from pick_ball import LOOK, THETA, find_ball, pick
from viam.services.motion import MotionClient

HOME = dict(x=LOOK[0], y=LOOK[1], z=LOOK[2], o_x=0, o_y=0, o_z=-1, theta=THETA)


def ask_landing():
    """+cm = long (past the cup), -cm = short, blank = skip."""
    try:
        s = input("Where did it land? cm past the cup (+) or short of it (-), 'in' if it went in, "
                  "blank to skip: ").strip().lower()
    except EOFError:
        return None
    if not s:
        return None
    if s in ("in", "y", "yes"):
        return {"hit": True}
    import re
    m = re.search(r"-?\d+(?:\.\d+)?", s)   # accepts "10", "-10", "10cm", "10cms"
    return {"hit": False, "long_cm": float(m.group())} if m else None


async def run(args):
    go = args.go and args.i_checked_with_staff
    if args.go and not go:
        print("Refusing to throw live without --i-checked-with-staff.")
    print("MODE:", "LIVE (arm will move and throw)" if go else "DRY RUN (no motion)")
    t0 = time.monotonic()
    robot = await connect()
    try:
        motion = MotionClient.from_robot(robot, "builtin")
        await move_to(motion, HOME, go, "home")
        await release_if_holding(robot, motion, go)
        await move_to(motion, HOME, go, "home")

        yaw = args.yaw
        if args.aim_cup:
            from main import scan_for_cup
            print("0) find cup (to aim the lob)")
            cup = await scan_for_cup(motion, robot, go)
            if cup is None:
                print("no cup found - not throwing")
                return
            yaw = throw.yaw_for_cup(cup[0], cup[1])
            import math
            if abs(math.hypot(cup[0], cup[1]) - throw.LOB_RELEASE_RADIUS_MM) > 80 and not getattr(args, "force_aim", False):
                print(f"cup is {math.hypot(cup[0], cup[1]):.0f} mm from the base but the lob only lands "
                      f"~{throw.LOB_RELEASE_RADIUS_MM} mm out: not picking or throwing. Move the cup to ~48 cm "
                      "from the base (measure with a tape), or use --force-aim to throw anyway.")
                await move_to(motion, HOME, go, "home")
                return
            await move_to(motion, HOME, go, "home")

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
            info = await throw.throw(robot, args.distance, go, args.scale, not args.no_release,
                                     args.release_delay, throw.swing_from_args(args), args.speed, args.accel, args.wind_z,
                                     throw.LOB_START_J if args.lob else None, yaw)
        except BaseException as e:  # error or Ctrl+C: don't leave the ball in the gripper
            print(f"!! stopped ({type(e).__name__}: {e}); putting the ball back")
            await asyncio.sleep(2.0)  # after a controller stop the arm refuses commands for a moment
            await put_ball_back(robot, motion, ball, go)
            raise
        await move_to(motion, HOME, go, "home")
        if go:
            landing = ask_landing()
            doc = {"phase": "throw", "ball": {"pos_mm": ball}, "throw": info,
                   "duration_s": round(time.monotonic() - t0, 1),
                   "result": landing or {"hit": None}}
            try:
                import logger
                logger.log_attempt(doc)
            except Exception as e:  # noqa: BLE001
                print(f"  [log failed: {e}]")
    finally:
        await robot.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=float, required=True, help="cup distance from base, mm")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--no-release", action="store_true")
    ap.add_argument("--speed", type=float, default=None, help="swing speed deg/s (default 150, max 180)")
    ap.add_argument("--accel", type=float, default=None, help="swing accel deg/s2 (default 900, max 1145)")
    ap.add_argument("--wind-z", type=float, default=None, help="wind-up height mm (default 450; try 650 for a lob)")
    ap.add_argument("--lob", action="store_true", help="near-vertical full-speed toss (cup ~50 cm from the base)")
    ap.add_argument("--yaw", type=float, default=0.0, help="turn the lob toward +y by this many deg (--lob only)")
    ap.add_argument("--force-aim", action="store_true", help="with --aim-cup: throw even if the cup is out of range")
    ap.add_argument("--aim-cup", action="store_true",
                    help="scan for the cup first and turn the lob toward it (--lob only)")
    ap.add_argument("--j2", type=float, default=None, help="shoulder swing deg (negative = forward+down)")
    ap.add_argument("--j3", type=float, default=None, help="elbow swing deg (default -35)")
    ap.add_argument("--j5", type=float, default=None, help="wrist swing deg (default +45)")
    ap.add_argument("--release-delay", type=float, default=None,
                    help="seconds after the swing starts to open the gripper; negative = open before the swing; try -0.3 / -0.2 / -0.1")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--i-checked-with-staff", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
