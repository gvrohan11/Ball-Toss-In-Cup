"""Phase 2: survey the cups, pick the ball, THROW it, log the result to Atlas.

    python pick_and_throw.py --aim-cup                                   # dry run: survey + plan, no motion
    python pick_and_throw.py --aim-cup --cradle --go --i-checked-with-staff     # one throw, nearest cup in range
    python pick_and_throw.py --aim-cup --cup 3 --go --i-checked-with-staff      # one throw at survey cup 3
    python pick_and_throw.py --fill-cups --cradle --go --i-checked-with-staff   # one ball per cup until all filled
    python pick_and_throw.py --survey-only --cradle --go --i-checked-with-staff # look at the cups only, list them
    python pick_and_throw.py --distance 950 --go --i-checked-with-staff         # no survey: cup straight ahead

Cups (--aim-cup / --fill-cups): the table is surveyed ONCE with the gripper empty, from every scan
view (main.survey_cups: sightings merged into distinct cups, numbered nearest first, remembered in
the world frame). Only cups inside the preset's distance band are targets; the rest are listed.
Layout tip: the throw's DISTANCE band is narrow (cradle ~93-102 cm) but aiming left/right is just a
base turn, so put the cups on an ARC at one distance, not in a triangle.
After each throw you say what happened; "in 3" = it went into cup 3 (any cup, not only the target).
Filled cups are never targeted again, and balls near a cup are never picked (pick_ball.AVOID_XY).

Flow per ball: plan + safety-check the throw BEFORE picking -> find + pick -> throw -> home ->
ask -> log to Atlas (phase "throw"). If anything fails after the pick, the ball is put back.
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
MAX_TRIES_PER_CUP = 3     # misses at one cup before --fill-cups moves on to the next
MAX_PICK_FAILS = 3        # failed picks in a row before stopping

PROMPT = ("Result? 'in' (target) or 'in 3' (went into cup 3) | 'rim' (reached the cup, bounced out) | "
          "cm long(+)/short(-), optionally ',left(+)/right(-) cm' e.g. '12' or '-8,3' | "
          "'drop' (fell out early) | blank = skip: ")


def parse_landing(s):
    """Typed answer -> result dict (None = skip). Every miss gets an explicit outcome, so a 0 is never
    ambiguous again (Oct 29: 34 of 69 misses were logged as long_cm 0)."""
    s = s.strip().lower()
    if not s:
        return None
    m = re.fullmatch(r"(?:in|y|yes|hit)(?:\s*(?:cup)?\s*#?(\d+))?", s)
    if m:
        res = {"hit": True, "outcome": "in"}
        if m.group(1):
            res["cup_number"] = int(m.group(1))
        return res
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


def _dist(c):
    return math.hypot(c["pos_mm"][0], c["pos_mm"][1])


def _cup_str(c):
    x, y, _ = c["pos_mm"]
    return f"cup {c['number']} ({x:.0f}, {y:+.0f}) {_dist(c) / 10:.0f} cm"


async def survey_targets(motion, robot, go, preset, ratio):
    """Survey once; returns every cup (numbered, nearest first) with c['reachable'] = inside the
    preset's distance band."""
    import cup_aim
    import main
    cup_aim.MAX_CUP_REACH_MM = 2300   # the detector marks cups past the DROP reach; throws go farther
    lo, hi = throw.reach_mm(preset, ratio)
    print(f"0) survey the cups ({preset.upper()} throws land {lo / 10:.0f}-{hi / 10:.0f} cm from the base)")
    cups = await main.survey_cups(motion, robot, go, in_range=lambda p: lo <= math.hypot(p[0], p[1]) <= hi,
                                  range_label=f"the {preset} band {lo / 10:.0f}-{hi / 10:.0f} cm")
    for n, c in enumerate(cups, 1):
        c["number"] = n
    return cups


async def throw_at(robot, motion, args, go, pl, cup, t0):
    """Find + pick a ball, throw with plan `pl`, ask, log. Returns the result dict, or
    {"failure": ...} if no ball / missed pick."""
    print("1) find ball")
    ball = await find_ball(robot, motion, go)
    if ball is None:
        print("no ball found")
        return {"failure": "no_ball_found"}
    print("2) pick")
    held = await pick(robot, ball, go)
    if go and not held:
        print("pick missed; nothing to throw")
        await move_to(motion, HOME, go, "home")
        return {"failure": "grasp_failed"}
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
        return {"check_only": True}
    if not go:
        return {"dry_run": True}
    if args.no_release:
        print("no-release run: the ball is still in the gripper (the next run sets it down first); not logged")
        return {"no_release": True}
    landing = ask_landing() or {"hit": None, "outcome": "not_recorded"}
    cup_doc = None
    if cup is not None:
        cup_doc = {"number": cup.get("number"), "pos_mm": list(cup["pos_mm"]), "dist_mm": round(_dist(cup)),
                   "of": cup.get("of")}
    doc = {"phase": "throw", "ball": {"pos_mm": ball}, "cup": cup_doc, "throw": info,
           "duration_s": round(time.monotonic() - t0, 1), "result": landing}
    try:
        import logger
        logger.log_attempt(doc)
    except Exception as e:  # noqa: BLE001
        print(f"  [log failed: {e}]")
    return landing


def plan_for(args, cup, ratio):
    yaw = throw.yaw_for_cup(cup["pos_mm"][0], cup["pos_mm"][1])
    return throw.plan_from_args(args, _dist(cup), yaw, ratio)


async def fill_cups(robot, motion, args, go, cups, ratio):
    """One ball per cup, nearest first, until every in-band cup is filled, balls run out, or picks
    keep failing. A hit fills whichever cup the ball went into ("in 3")."""
    targets = [c for c in cups if c["reachable"]]
    tries, pick_fails, throws, filled = {}, 0, 0, []
    while targets:
        cup = targets[0]
        print(f"\n=== ball {throws + 1} -> {_cup_str(cup)} ({len(targets)} cup(s) left) ===")
        pl = plan_for(args, cup, ratio)                      # safety-checked BEFORE picking
        res = await throw_at(robot, motion, args, go, pl, cup, time.monotonic())
        if res.get("failure") == "no_ball_found":
            print("  no ball left near the robot; stopping")
            break
        if res.get("failure") == "grasp_failed":
            pick_fails += 1
            if pick_fails >= MAX_PICK_FAILS:
                print(f"  {pick_fails} picks failed in a row; stopping")
                break
            continue
        pick_fails, throws = 0, throws + 1
        if not go or res.get("check_only") or res.get("no_release"):
            break                                            # dry run / check: one pass is enough
        if res.get("hit"):
            n = res.get("cup_number", cup["number"])
            hit = next((c for c in targets if c["number"] == n), None)
            if hit is not None:
                targets.remove(hit)
                filled.append(hit)
                print(f"  FILLED {_cup_str(hit)}; {len(targets)} cup(s) left")
            continue
        tries[cup["number"]] = tries.get(cup["number"], 0) + 1
        if tries[cup["number"]] >= MAX_TRIES_PER_CUP:
            targets.remove(cup)
            print(f"  missed {_cup_str(cup)} {MAX_TRIES_PER_CUP} times: moving on")
    print(f"\nFILL SUMMARY: {len(filled)} cup(s) filled in {throws} throw(s)"
          + (": " + ", ".join(_cup_str(c) for c in filled) if filled else ""))
    return filled


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
        ratio = throw.learned_ratio(args)

        if not (args.aim_cup or args.fill_cups or args.survey_only):
            if args.distance is None:
                print("give --distance (mm from the base), --aim-cup or --fill-cups")
                return
            pl = throw.plan_from_args(args, args.distance, None, ratio)
            await throw_at(robot, motion, args, go, pl, None, t0)
            return

        cups = await survey_targets(motion, robot, go, preset, ratio)
        await move_to(motion, HOME, go, "home")
        for c in cups:
            c["of"] = len(cups)
        targets = [c for c in cups if c["reachable"]]
        if args.survey_only:
            print(f"SURVEY ONLY: {len(cups)} cup(s), {len(targets)} in the {preset} band: "
                  + (", ".join(_cup_str(c) for c in targets) or "none")
                  + "  (python detection_log.py shows what the camera saw)")
            return
        if args.cup is not None:
            cup = next((c for c in cups if c["number"] == args.cup), None)
            if cup is None:
                print(f"no cup {args.cup} in the survey ({len(cups)} found)")
                return
            if not cup["reachable"] and not args.force_aim:
                print(f"{_cup_str(cup)} is outside the {preset} band: move it, or --force-aim")
                return
            targets = [cup]
        if not targets:
            lo, hi = throw.reach_mm(preset, ratio)
            print(f"no cup {lo / 10:.0f}-{hi / 10:.0f} cm from the base - move the cups"
                  + (" or try --power" if preset != "power" else "") + "; not picking or throwing")
            return
        if args.fill_cups:
            await fill_cups(robot, motion, args, go, targets, ratio)
            return
        cup = targets[0]
        print(f"target: {_cup_str(cup)}")
        pl = plan_for(args, cup, ratio)                      # raises before anything is picked if unsafe
        await throw_at(robot, motion, args, go, pl, cup, t0)
    finally:
        await robot.close()


def main():
    ap = argparse.ArgumentParser(description="Pick the ball, then throw it at a cup.")
    throw.add_args(ap)
    ap.add_argument("--aim-cup", action="store_true", help="survey the cups and throw at the nearest one in range")
    ap.add_argument("--fill-cups", action="store_true", help="survey once, then one ball per cup until all are filled")
    ap.add_argument("--survey-only", action="store_true",
                    help="run the cup survey (planned moves only), list the cups, no pick/throw")
    ap.add_argument("--cup", type=int, default=None, help="with --aim-cup: throw at survey cup N (1 = nearest)")
    ap.add_argument("--force-aim", action="store_true", help="with --cup: throw even if it's outside the band")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
