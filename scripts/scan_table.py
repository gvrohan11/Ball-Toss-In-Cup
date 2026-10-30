"""Scan the table for balls and list them. Moves the arm to look; never picks or throws.

    python scripts/scan_table.py                    # dry run: current camera view only, no motion
    python scripts/scan_table.py --go               # look pose, top view, side views: every ball
    python scripts/scan_table.py --go --survey      # survey the cups first, so balls in cups and
                                                    # past the search line are marked as such

Visits the same views the picker uses (pick_ball.LOOK, then pick_ball.SEARCH_VIEWS: the top
view first), finds EVERY orange ball in each, merges balls seen from several views, and prints:

    ball 1: (452, +51)  45 cm out, seen in 3 view(s)  -> would be picked
    ball 2: (880, +205) 90 cm out, seen in 1 view(s)  -> in or next to a cup
    TABLE: 1 ball(s) the robot would pick

or "TABLE CLEAR: no ball the robot would pick". Every frame is saved in the detection log
(python detection_log.py; python scripts/debug_ball.py <folder> for how each ball was found).
"""
import argparse
import asyncio
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SAME_BALL_MM = 30     # sightings from different views this close are one ball


def merge(sightings):
    """[(view, {"xy", "why"})] -> [{"xy", "why", "views"}], nearest the base first."""
    balls = []
    for view, s in sightings:
        b = next((b for b in balls if math.dist(b["xy"], s["xy"]) <= SAME_BALL_MM), None)
        if b is None:
            balls.append({"xy": s["xy"], "why": s["why"], "views": [view], "_pts": [s["xy"]]})
            continue
        b["_pts"].append(s["xy"])
        b["xy"] = tuple(sum(v) / len(b["_pts"]) for v in zip(*b["_pts"]))
        b["views"].append(view)
        b["why"] = b["why"] or s["why"]           # any view that rejects it (e.g. touching) wins
    for b in balls:
        del b["_pts"]
    return sorted(balls, key=lambda b: math.hypot(*b["xy"]))


def report(balls, limit):
    print(f"\nBALLS ON THE TABLE (search line: {limit / 10:.0f} cm from the base)")
    for n, b in enumerate(balls, 1):
        x, y = b["xy"]
        print(f"  ball {n}: ({x:.0f}, {y:+.0f})  {math.hypot(x, y) / 10:.0f} cm out, "
              f"seen in {len(set(b['views']))} view(s)  -> {b['why'] or 'would be picked'}")
    pickable = [b for b in balls if not b["why"]]
    if pickable:
        print(f"TABLE: {len(pickable)} ball(s) the robot would pick")
    else:
        print("TABLE CLEAR: no ball the robot would pick")
    return pickable


async def scan(robot, motion, go):
    import detection_log
    import pick_ball as pb
    views = [pb.LOOK] + pb.search_views()
    sightings = []
    for i, (gx, gy, gz) in enumerate(views, 1):
        name = "look pose" if (gx, gy, gz) == pb.LOOK else ("top" if (gx, gy, gz) == pb.TOP_VIEW else "side")
        detection_log.context(f"ball_scan_{i}", view=(gx, gy, gz))
        try:
            await pb.move(motion, gx, gy, gz, go)
        except Exception as e:
            print(f"  view {i}/{len(views)} {name} ({gx}, {gy}, z {gz}): can't reach, skipped ({str(e)[:50]})")
            continue
        if go:
            await asyncio.sleep(pb.LOOK_SETTLE_S)
        seen = await pb.balls_in_view(robot)
        print(f"  view {i}/{len(views)} {name} ({gx}, {gy}, z {gz}): {len(seen)} ball(s) "
              + " ".join(f"({s['xy'][0]:.0f},{s['xy'][1]:+.0f})" for s in seen))
        sightings += [(i, s) for s in seen]
        if not go:
            print("  dry run: the arm didn't move, so only the current view was checked")
            break
    return merge(sightings)


async def _main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="move the arm through the views (E-stop!)")
    ap.add_argument("--survey", action="store_true", help="survey the cups first (with --go)")
    ap.add_argument("--no-yolo", action="store_true", help="color detection only")
    a = ap.parse_args()
    import ball_finder
    import detection_log
    import pick_ball as pb
    from connection import connect
    from viam.services.motion import MotionClient

    ball_finder.USE_YOLO = not a.no_yolo
    print("MODE:", "LIVE (arm moves to look; nothing is picked)" if a.go else "DRY RUN (no motion)")
    robot = await connect()
    log = detection_log.start()
    try:
        motion = MotionClient.from_robot(robot, "builtin")
        if a.survey:
            import main
            print("survey the cups first (marks balls in cups and past the search line)")
            await main.survey_cups(motion, robot, a.go)
        balls = await scan(robot, motion, a.go)
        report(balls, pb.search_max_r())
    finally:
        detection_log.stop()
        if a.go:
            await pb.move(MotionClient.from_robot(robot, "builtin"), *pb.LOOK, True)
        await robot.close()
    if log:
        print(f"frames: {log.dir}")


if __name__ == "__main__":
    asyncio.run(_main())
