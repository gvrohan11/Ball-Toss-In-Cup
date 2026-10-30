"""Manan: the full loop. survey cups -> remember them -> find ball -> pick -> drop -> home -> log.

python main.py          # dry run: prints the plan, moves nothing
python main.py --go     # runs on the arm (someone on the E-stop!)
python main.py --go --autotune   # also shift the aim by the average of recent misses in Atlas
python main.py --go --runs 10    # 10 attempts in a row; put the ball back between them
python main.py --go --cup 2      # drop into cup 2 of the survey (1 = nearest, default)
python main.py --go --no-yolo    # cup survey with the rim method only (YOLO off)
python main.py --go --fill-cups  # several balls near the robot, cups stay put: survey once,
                                 # one ball per cup (nearest first); filled cups are discarded
python detection_log.py          # what the camera saw + detected in the last attempt

The table is surveyed FIRST, with the gripper empty (the held ball used to hide the cup's
red side): every scan view is visited, sightings are merged into distinct cups, and all of
them are listed and remembered in the world frame. After the pick the arm goes straight to
the chosen cup's remembered rim center: it does not need to be in view any more.

Uses the sequences that worked on sfhack11 (Oct 29): pick_ball.py (2/2 picks),
cup_aim.py + drop.py (2/2 drops). Swap in balls.find_ball / cups.get_cups when
the ball and cup teammates' versions are reliable.
"""
import asyncio
import math
import sys
import time
from collections import Counter

from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient

import detection_log
from connection import connect
from cup_aim import CUP_SCAN, find_cups   # rim method + YOLO (cup_detection.py) per frame
from drop import REACH_MAX_MM, drop_in_cup
from logger import log_attempt, suggest_offset
from pick_ball import GRIPPER, LOOK, THETA, find_ball, pick
from telemetry import Telemetry

TEL = None  # Telemetry for the current run (set in main)


def fmt_pose(p):
    """x=550 y=+220 z=650, tilted 30 deg  (instead of the raw pose dict)."""
    tilt = math.degrees(math.atan2(math.hypot(p.get("o_x", 0), p.get("o_y", 0)), -p.get("o_z", -1)))
    return (f"x={p['x']:.0f} y={p['y']:+.0f} z={p['z']:.0f}, "
            + ("pointing down" if tilt < 1 else f"tilted {tilt:.0f} deg"))


async def move_to(motion, p, go, label="move", say=True):
    if say:
        print(f"  -> gripper to {fmt_pose(p)}")
    if go:
        t = time.monotonic()
        ok = None
        try:
            ok = await motion.move(component_name=GRIPPER,
                                   destination=PoseInFrame(reference_frame="world", pose=Pose(**p)))
        finally:
            if TEL:
                TEL.move(label, p, ok, time.monotonic() - t)
        if ok is False:
            raise RuntimeError("Motion did not complete")


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


async def put_ball_back(robot, motion, ball, go):
    """Safe fail: carry the held ball back to where it was picked, set it down, open, go up."""
    from viam.components.gripper import Gripper
    from pick_ball import GRAB_Z, HOVER_Z
    print("  putting the ball back")
    gripper = Gripper.from_robot(robot, GRIPPER)
    try:
        if ball is not None:
            await move_to(motion, dict(x=ball[0], y=ball[1], z=HOVER_Z, o_x=0, o_y=0, o_z=-1, theta=THETA), go)
            await move_to(motion, dict(x=ball[0], y=ball[1], z=GRAB_Z + 10, o_x=0, o_y=0, o_z=-1, theta=THETA), go)
    except Exception as e:
        print(f"  couldn't reach the pick spot ({e}); releasing where I am")
    if go:
        await gripper.open()
    if ball is not None:
        await move_to(motion, dict(x=ball[0], y=ball[1], z=HOVER_Z, o_x=0, o_y=0, o_z=-1, theta=THETA), go)


CUP_READINGS = 3      # frames per view
CUP_MERGE_MM = 50     # sightings this close are the same cup (solo cup openings are ~95 mm wide)
CUP_MIN_SIGHTINGS = 3  # fewer than this across the whole survey = glare/noise, ignored
                       # (real cups: 9-20 sightings per survey on Oct 29; strays: 2)
VIEW_SETTLE_S = 0.5    # after arriving at a view: the first frames were motion-blurred


def merge_sightings(sightings, min_sightings=CUP_MIN_SIGHTINGS):
    """Group cup sightings from all views and frames into distinct cups.
    Returns [{"pos_mm": mean (x, y, z), "sightings": n, "reachable": bool}] nearest first."""
    groups = []
    for c in sightings:
        near = [g for g in groups if math.dist(c[:2], g["mean"][:2]) <= CUP_MERGE_MM]
        g = min(near, key=lambda g: math.dist(c[:2], g["mean"][:2]), default=None)
        if g is None:
            g = {"points": []}
            groups.append(g)
        g["points"].append(c)
        g["mean"] = tuple(sum(v) / len(g["points"]) for v in zip(*g["points"]))
    cups = [{"pos_mm": g["mean"], "sightings": len(g["points"]), "reachable": reachable(g["mean"])}
            for g in groups if len(g["points"]) >= min_sightings]
    return sorted(cups, key=lambda c: math.hypot(c["pos_mm"][0], c["pos_mm"][1]))


def reachable(cup):
    """The drop can only release over cups up to REACH_MAX_MM from the arm base."""
    return math.hypot(cup[0], cup[1]) <= REACH_MAX_MM


def _xy(cups):
    return " ".join(f"({c['pos_mm'][0]:.0f},{c['pos_mm'][1]:+.0f})" for c in cups) or "-"


async def scan_for_cup(motion, robot, go):
    """Kept for pick_and_throw.py --aim-cup: try the CUP_SCAN views in order and return the first
    cup found (x, y, z), or None. Uses cup_aim.find_cup (rim method + YOLO)."""
    import cup_aim
    for i, look in enumerate(CUP_SCAN, 1):
        print(f"  view {i}/{len(CUP_SCAN)}: {fmt_pose(look)}")
        try:
            await move_to(motion, look, go, label=f"cup_view_{i}")
        except Exception as e:  # noqa: BLE001
            print(f"    can't reach this view ({str(e)[:80]}); skipping")
            continue
        if go:
            await asyncio.sleep(VIEW_SETTLE_S)
        cup = await cup_aim.find_cup(robot, topdown=(look["o_z"] == -1))
        if cup is not None:
            return cup
    return None


async def survey_cups(motion, robot, go, in_range=None, range_label=None):
    """Look at the table from EVERY view in CUP_SCAN (skipping views the planner can't
    reach) and return all cups found, in the world frame, nearest first.
    in_range(pos_mm) -> bool replaces the drop's reach test (pick_and_throw passes the throw's
    distance band); range_label says what the range is in the printout."""
    import cup_aim
    cup_aim.USE_YOLO = "--no-yolo" not in sys.argv
    cup_aim.YOLO_OFFSETS.clear()
    sightings = []
    for i, look in enumerate(CUP_SCAN, 1):
        name = f"  view {i}/{len(CUP_SCAN)} ({fmt_pose(look)})"
        try:
            await move_to(motion, look, go, label=f"cup_view_{i}", say=False)
        except Exception as e:
            print(f"{name}: can't reach, skipped ({str(e)[:60]})")
            continue
        if go:
            await asyncio.sleep(VIEW_SETTLE_S)
        detection_log.context(f"cup_view_{i}", max_frames=1, view=look)   # images for 1 frame per view
        seen = []
        for _ in range(CUP_READINGS):
            seen += await find_cups(robot, topdown=(look["o_z"] == -1))
        here = merge_sightings(seen, min_sightings=1)
        print(f"{name}: {len(here)} cup(s) {_xy(here)}")
        if TEL:
            await TEL.snap(f"cup_view_{i}", found=len(seen), view=look)
            if seen and not sightings:
                await TEL.thumbnail("cup_view")
        sightings += seen
    cups = merge_sightings(sightings)
    if in_range is not None:
        for c in cups:
            c["reachable"] = bool(in_range(c["pos_mm"]))
    label = range_label or f"drop reach {REACH_MAX_MM / 10:.0f} cm"
    import pick_ball
    pick_ball.AVOID_XY = [c["pos_mm"][:2] for c in cups]   # never pick a ball out of a cup
    detection_log.context("survey")
    detection_log.event("survey", cups=cups, sightings=sightings)
    yolo_vs_rim(cup_aim.YOLO_OFFSETS)
    print(f"  TABLE SURVEY: {len(cups)} cup(s)")
    for n, c in enumerate(cups, 1):
        x, y, z = c["pos_mm"]
        print(f"    cup {n}: x={x:.0f} y={y:+.0f}  {math.hypot(x, y) / 10:.0f} cm out, "
              f"seen {c['sightings']}x" + ("" if c["reachable"] else
                                           f"  <- outside {label}" if range_label else "  <- out of reach"))
    for n, c in enumerate(cups, 1):              # out-of-reach cups are discarded right away
        if not c["reachable"]:
            x, y, z = c["pos_mm"]
            why = (f"{math.hypot(x, y) / 10:.0f} cm is outside {label}" if range_label else
                   f"out of reach: {math.hypot(x, y) / 10:.0f} cm > {REACH_MAX_MM / 10:.0f} cm")
            print(f"  DISCARDED cup {n} at x={x:.0f} y={y:+.0f} z={z:.0f} ({why})")
            detection_log.event("discarded", number=n, cup_mm=c["pos_mm"], why=why,
                                left=sum(1 for k in cups if k["reachable"]))
    return cups


def yolo_vs_rim(offsets):
    """Median (rim - YOLO) offset over cups both methods saw: what cup_detection.
    CUP_CORRECTION_MM should be. Printed + logged every survey so it can be set from data."""
    if not offsets:
        return None
    import cup_detection
    dx = sorted(o[0] for o in offsets)[len(offsets) // 2]
    dy = sorted(o[1] for o in offsets)[len(offsets) // 2]
    print(f"  YOLO vs rim on {len(offsets)} shared sightings: rim - YOLO = ({dx:+.0f}, {dy:+.0f}) mm "
          f"(cup_detection.CUP_CORRECTION_MM is {cup_detection.CUP_CORRECTION_MM})")
    detection_log.event("yolo_vs_rim", median_offset_mm=(dx, dy), n=len(offsets))
    return dx, dy


def chosen_cup_number(cups):
    """--cup N picks cup N from the survey; default = the nearest cup the drop can reach."""
    if "--cup" in sys.argv:
        return int(sys.argv[sys.argv.index("--cup") + 1])
    return next((n for n, c in enumerate(cups, 1) if c["reachable"]), 1)


# Where to set down a ball found in the gripper at startup: clear table near the robot.
SET_DOWN_XY = (420, 0)


async def release_if_holding(robot, motion, go):
    """Start-of-run check: if the gripper is holding something (left over from an
    interrupted run), set it down on the table, open, and lift. Always ends open."""
    from viam.components.gripper import Gripper
    from pick_ball import GRAB_Z, HOLDING_MIN_POS, HOVER_Z
    gripper = Gripper.from_robot(robot, GRIPPER)
    try:
        pos = (await gripper.do_command({"get": True})).get("pos")
    except Exception:
        pos = None
    holding = pos is not None and HOLDING_MIN_POS < pos < 800
    print(f"  gripper position {pos}: {'HOLDING something - setting it down' if holding else 'empty'}")
    if holding:
        x, y = SET_DOWN_XY
        await move_to(motion, dict(x=x, y=y, z=HOVER_Z, o_x=0, o_y=0, o_z=-1, theta=THETA), go)
        await move_to(motion, dict(x=x, y=y, z=GRAB_Z + 10, o_x=0, o_y=0, o_z=-1, theta=THETA), go)
    if go:
        await gripper.open()
    if holding:
        await move_to(motion, dict(x=x, y=y, z=HOVER_Z, o_x=0, o_y=0, o_z=-1, theta=THETA), go)


async def _snap(stage, **kw):
    if TEL:
        await TEL.snap(stage, **kw)


def _phase(name):
    if TEL:
        TEL.phase(name)


async def run_once(robot, motion, go, home, t0):
    print("0) start from the saved home pose")
    _phase("start")
    await move_to(motion, home, go, label="home")
    await release_if_holding(robot, motion, go)
    await move_to(motion, home, go, label="home")
    await _snap("home")

    # Cups first, gripper empty: nothing hides them, and nothing is held if none is found.
    print("1) survey the table for cups and remember them")
    _phase("cup_scan")
    cups = await survey_cups(motion, robot, go)
    n = chosen_cup_number(cups)
    if not cups or not 1 <= n <= len(cups):
        print(f"no cup {n} on the table ({len(cups)} found); not picking")
        return {"cups": cups, "result": {"hit": False, "failure": "no_cup_found"}}, None
    cup = cups[n - 1]["pos_mm"]
    if not cups[n - 1]["reachable"]:
        print(f"cup {n} was discarded (out of reach, {math.hypot(cup[0], cup[1]) / 10:.0f} cm); "
              f"no cup within {REACH_MAX_MM / 10:.0f} cm to use; not picking")
        detection_log.event("target", cup_mm=cup, number=n, of=len(cups), refused="too far")
        return {"cups": cups, "result": {"hit": False, "failure": "cup_out_of_reach"}}, None
    print(f"  target: cup {n} of {len(cups)} at x={cup[0]:.0f} y={cup[1]:+.0f} (rim z={cup[2]:.0f})")
    await _snap("cup_remembered", cup_mm=list(cup), cups=len(cups))
    detection_log.event("target", cup_mm=cup, number=n, of=len(cups))
    cup_doc = {"pos_mm": cup, "frame": "world", "seen_before_pick": True,
               "number": n, "of": len(cups)}
    return await pick_and_drop(robot, motion, go, cup_doc, cups)


async def pick_and_drop(robot, motion, go, cup_doc, cups):
    """Find a ball near the robot, pick it slowly, drop it at the remembered cup_doc."""
    cup = cup_doc["pos_mm"]
    print("2) find ball")
    _phase("find_ball")
    detection_log.context("find_ball")
    ball = await find_ball(robot, motion, go)
    await _snap("ball_measured", ball_mm=list(ball) if ball else None)
    if TEL and ball:
        await TEL.thumbnail("ball_view")
    if ball is None:
        print("no ball found")
        return {"cup": cup_doc, "cups": cups, "result": {"hit": False, "failure": "no_ball_found"}}, None

    print("3) pick (slowly)")
    _phase("pick")
    held = await pick(robot, ball, go)
    await _snap("after_pick", held=held)
    if go and not held:
        print("pick missed")
        return {"ball": {"pos_mm": ball}, "cup": cup_doc, "cups": cups,
                "result": {"hit": False, "failure": "grasp_failed"}}, None

    # Ball is in the gripper from here on: any failure puts it back first.
    extra = (0, 0)
    if "--autotune" in sys.argv:
        extra = suggest_offset()
        print(f"  autotune from Atlas misses: {extra[0]:+.0f}, {extra[1]:+.0f} mm")
    print("4) drop at the remembered cup (no new look needed)")
    _phase("drop")
    try:
        result = await drop_in_cup(robot, cup, go, extra)
        await _snap("released", cup_mm=list(cup))
    except Exception as e:
        print(f"  can't reach the cup ({str(e)[:80]})")
        await put_ball_back(robot, motion, ball, go)
        return {"ball": {"pos_mm": ball}, "cup": cup_doc, "cups": cups,
                "result": {"hit": False, "failure": "cup_out_of_reach"}}, None
    return {"phase": "drop", "ball": {"pos_mm": ball}, "cup": cup_doc, "cups": cups,
            "release": result}, "dropped"


def _run_config():
    import cup_aim
    import cup_detection
    import drop
    import pick_ball
    return {"theta": THETA, "home": list(LOOK),
            "ball_cam_correction_mm": list(pick_ball.CAM_CORRECTION_MM),
            "grab_z": pick_ball.GRAB_Z, "hover_z": pick_ball.HOVER_Z,
            "rim_correction_mm": list(cup_aim.RIM_CORRECTION_MM),
            "topdown_correction_mm": list(cup_aim.TOPDOWN_CORRECTION_MM),
            "cup_detection": "rim" if "--no-yolo" in sys.argv else "rim+yolo",
            "yolo_cup_correction_mm": list(cup_detection.CUP_CORRECTION_MM),
            "release_above_rim_mm": drop.RELEASE_ABOVE_RIM_MM,
            "pick_speed_deg_s": pick_ball.PICK_SPEED_DEG_S,
            "ball_color": pick_ball.BALL_COLOR,
            "autotune": "--autotune" in sys.argv}


async def attempt(robot, motion, go, home, body=None):
    """One full attempt; always ends at home. Asks hit/miss, logs, returns the document.
    body(t0) -> (doc, state) does the work; default run_once (start check, survey, pick, drop)."""
    global TEL
    t0 = time.monotonic()
    TEL = Telemetry(robot, go)
    TEL.config.update(_run_config())
    log = detection_log.start()
    try:
        doc, state = await (body(t0) if body else run_once(robot, motion, go, home, t0))
    except BaseException as e:  # error or Ctrl+C: never leave the ball hanging
        print(f"!! stopped ({type(e).__name__}: {e}) - opening gripper and going home")
        if go:
            from viam.components.gripper import Gripper
            await Gripper.from_robot(robot, GRIPPER).open()
        raise
    finally:
        print("5) home")
        await move_to(motion, home, go)

    if state == "dropped" and go:
        hit, miss = ask_result()
        res = {"hit": hit}
        if miss is not None:
            res["miss_mm"] = miss
            res["failure"] = "missed"
        doc["result"] = res
    if log:
        doc["detection_log"] = str(log.dir)
        detection_log.context("result")
        detection_log.event("result", **doc.get("result", {}))
        detection_log.stop()
    if go:
        doc["duration_s"] = round(time.monotonic() - t0, 1)
        doc["telemetry"] = TEL.as_doc()
        log_attempt(doc)
    return doc


def summary(results):
    hits = sum(1 for r in results if r.get("hit"))
    fails = Counter(r["failure"] for r in results if r.get("failure"))
    return (f"SERIES: {hits}/{len(results)} in the cup"
            + (f"; failures: {dict(fails)}" if fails else ""))


async def run_series(robot, motion, go, home, runs):
    """`runs` attempts in a row. Stops on an error (after going home) or 'q' between runs."""
    results = []
    try:
        for i in range(runs):
            if runs > 1:
                print(f"\n=== attempt {i + 1}/{runs} ===")
                if go and i and input("Take the ball out of the cup, put it on the table, "
                                      "Enter to go (q = stop): ").strip().lower() == "q":
                    break
            results.append((await attempt(robot, motion, go, home)).get("result", {}))
    finally:
        if runs > 1:
            print(summary(results))
    return results


MAX_TRIES_PER_CUP = 2   # misses before a cup is given up on
MAX_PICK_FAILS = 3      # failed picks in a row before stopping


def _cup_at(c):
    x, y, z = c["pos_mm"]
    return f"x={x:.0f} y={y:+.0f} z={z:.0f}"


async def fill_all_cups(robot, motion, go, home):
    """Stationary cups, several balls near the robot. Survey the cups ONCE, then one ball per
    cup, nearest cup first. A cup that got its ball is discarded (its location printed) and
    the next ball goes to the next cup. A miss leaves the cup for the next ball, up to
    MAX_TRIES_PER_CUP. Stops when every cup is done, no ball is left, or picks keep failing.
    Returns the list of discarded cups."""
    print("0) start from the saved home pose")
    await move_to(motion, home, go, label="home")
    await release_if_holding(robot, motion, go)
    await move_to(motion, home, go, label="home")

    session = detection_log.start()          # this folder holds the survey + the cup bookkeeping
    try:
        print("1) survey the table ONCE (the cups stay put)")
        cups = await survey_cups(motion, robot, go)
    finally:
        detection_log.stop()
    targets = [dict(c, number=n) for n, c in enumerate(cups, 1) if c["reachable"]]
    print(f"  PLAN: {len(targets)} cup(s) to fill, nearest first: "
          + (", ".join(f"cup {c['number']} ({_cup_at(c)})" for c in targets) or "none"))

    def note(kind, **data):
        if session:
            session.event(kind, **data)

    discarded = [{"number": n, "pos_mm": c["pos_mm"], "why": "out of reach"}     # (survey printed these)
                 for n, c in enumerate(cups, 1) if not c["reachable"]]
    results, tries, pick_fails = [], Counter(), 0

    def discard(c, why):
        targets.remove(c)
        discarded.append({"number": c["number"], "pos_mm": c["pos_mm"], "why": why})
        print(f"  DISCARDED cup {c['number']} at {_cup_at(c)} ({why}); {len(targets)} cup(s) left"
              + (": " + ", ".join(f"cup {t['number']} ({_cup_at(t)})" for t in targets) if targets else ""))
        note("discarded", number=c["number"], cup_mm=c["pos_mm"], why=why, left=len(targets))

    try:
        while targets:
            c = targets[0]
            print(f"\n=== ball {len(results) + 1} -> cup {c['number']} at {_cup_at(c)} "
                  f"({len(targets)} cup(s) left) ===")
            cup_doc = {"pos_mm": c["pos_mm"], "frame": "world", "seen_before_pick": True,
                       "number": c["number"], "of": len(cups)}

            async def body(t0, cup_doc=cup_doc):     # find_ball starts at the look pose itself
                return await pick_and_drop(robot, motion, go, cup_doc, cups)

            doc = await attempt(robot, motion, go, home, body=body)
            res = doc.get("result", {})
            results.append(res)
            failure = res.get("failure")
            if failure == "no_ball_found":
                print("  no ball left near the robot; stopping")
                break
            if failure == "grasp_failed":
                pick_fails += 1
                if pick_fails >= MAX_PICK_FAILS:
                    print(f"  {pick_fails} picks failed in a row; stopping")
                    break
                continue
            pick_fails = 0
            if failure == "cup_out_of_reach":
                discard(c, "arm can't reach it")
                continue
            tries[c["number"]] += 1
            if res.get("hit") or not go:
                discard(c, "ball in" if go else "dry run")
            elif tries[c["number"]] >= MAX_TRIES_PER_CUP:
                discard(c, f"missed {tries[c['number']]} times, giving up")
            else:
                print(f"  missed: cup {c['number']} stays, the next ball tries it again")
    finally:
        filled = sum(1 for d in discarded if d["why"] == "ball in")
        print(f"\nFILL SUMMARY: {filled} cup(s) filled with {len(results)} ball attempt(s); "
              f"{len(targets)} cup(s) left")
        for d in discarded:
            x, y, z = d["pos_mm"]
            print(f"  cup {d['number']}: x={x:.0f} y={y:+.0f} z={z:.0f}  {d['why']}")
        note("session", filled=filled, attempts=len(results), left=[t["number"] for t in targets],
             discarded=discarded)
    return discarded


def _runs():
    if "--runs" not in sys.argv:
        return 1
    n = int(sys.argv[sys.argv.index("--runs") + 1])
    if n < 1:
        raise SystemExit("--runs must be at least 1")
    return n


async def main():
    go = "--go" in sys.argv
    runs = _runs()
    import pick_ball
    print("BALL:", pick_ball.set_ball_color_from_argv())
    print("MODE:", "LIVE (arm will move)" if go else "DRY RUN (no motion, nothing sent to Atlas)")
    home = dict(x=LOOK[0], y=LOOK[1], z=LOOK[2], o_x=0, o_y=0, o_z=-1, theta=THETA)
    robot = await connect()
    try:
        motion = MotionClient.from_robot(robot, "builtin")
        if "--fill-cups" in sys.argv:
            await fill_all_cups(robot, motion, go, home)
        else:
            await run_series(robot, motion, go, home, runs)
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
