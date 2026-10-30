"""Pick calibration test: measure where the camera THINKS the ball is vs where it really is.

Why: the grab goes to the camera's close-up position + pick_ball.CAM_CORRECTION_MM, which was
measured on a single pick (-15, +20). If it's wrong, the gripper lands a bit "back" or to the
side and the fingers close on nothing.

How (the robot provides the ground truth itself):
  1. find + pick one ball (the normal pick, with retries)
  2. for each spot P in SPOTS: set the ball down with the gripper at P (the ball is centered
     between the fingers, so it sits at P), lift, center the camera over it, measure it
     (median of 3 close-up frames, no correction) -> offset = P - measured
  3. pick it up again (using the offset learned so far) and go to the next spot
  4. print per-spot offsets, the median, the spread, and the CAM_CORRECTION_MM to use

    python scripts/calibrate_pick.py              # dry run: prints the plan, moves nothing
    python scripts/calibrate_pick.py --go         # live (E-stop!): one orange ball near the robot
    python scripts/calibrate_pick.py --go --spots 3

Results: observations/calibration/<time>.json, frames in observations/detections/<time>/.
Nothing is changed in the code: copy the printed CAM_CORRECTION_MM into pick_ball.py.
"""
import argparse
import asyncio
import json
import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Ball set-down spots (world mm), all inside the home camera view and the pick area.
SPOTS = [(400, 0), (370, -60), (430, -60), (370, 60), (430, 60), (400, 0)]
SPREAD_OK_MM = 6     # offsets agreeing within this: the correction is trustworthy


def recommend(offsets):
    """Median (dx, dy) and spread of the measured offsets (truth - camera), mm."""
    if not offsets:
        return None
    med = tuple(sorted(v)[len(v) // 2] for v in zip(*offsets))
    spread = max(math.hypot(o[0] - med[0], o[1] - med[1]) for o in offsets)
    return med, spread


async def set_down(robot, at, go):
    """Put the held ball on the table with the gripper at `at`, open, lift."""
    import pick_ball as pb
    from viam.components.gripper import Gripper
    motion = _motion(robot)
    await pb.move(motion, at[0], at[1], pb.HOVER_Z, go)
    async with pb.slow(robot, go):
        await pb.move(motion, at[0], at[1], pb.GRAB_Z + 5, go)
        if go:
            await asyncio.sleep(pb.SETTLE_S)
            await Gripper.from_robot(robot, pb.GRIPPER).open()
            await asyncio.sleep(0.5)                   # let the ball settle on the table
        await pb.move(motion, at[0], at[1], pb.HOVER_Z, go)


async def measure_at(robot, at, correction, go):
    """Camera centered over the ball expected at `at`; raw (uncorrected) close-up position."""
    import pick_ball as pb
    motion = _motion(robot)
    await pb.move(motion, at[0] - correction[0] + pb.CAM_OFFSET_XY[0],
                  at[1] - correction[1] + pb.CAM_OFFSET_XY[1], pb.MEASURE_HEIGHT_Z, go)
    return await pb.closeup(robot, go) if go else (at[0] - correction[0], at[1] - correction[1], 0)


async def calibrate(robot, spots, go):
    import detection_log
    import pick_ball as pb
    pb.AVOID_XY = []
    correction = pb.CAM_CORRECTION_MM          # improves as offsets come in
    rows = []
    log = detection_log.start()
    try:
        detection_log.context("find_ball")
        ball = await pb.find_ball(robot, _motion(robot), go)
        if ball is None:
            print("no ball found near the robot; put one orange ball alone on the table")
            return rows, correction
        if not await pb.pick(robot, ball, go) and go:
            print("couldn't pick the ball even with retries: the correction is far off or the ball "
                  "isn't graspable here. Check scripts/debug_ball.py on the frames above.")
            return rows, correction
        for i, spot in enumerate(spots, 1):
            print(f"\n--- spot {i}/{len(spots)}: set the ball down at ({spot[0]}, {spot[1]})")
            detection_log.context(f"calib_spot_{i}")
            await set_down(robot, spot, go)
            seen = await measure_at(robot, spot, correction, go)
            if seen is None:
                print("  ball not seen after setting it down (rolled away?); stopping")
                break
            off = (spot[0] - seen[0], spot[1] - seen[1])
            rows.append({"spot": spot, "camera_mm": seen[:2], "offset_mm": off})
            print(f"  camera says ({seen[0]:.0f}, {seen[1]:.0f}) -> truth - camera = ({off[0]:+.0f}, {off[1]:+.0f}) mm")
            detection_log.event("calibration", spot=spot, camera_mm=seen[:2], offset_mm=off)
            correction = recommend([r["offset_mm"] for r in rows])[0]
            if i == len(spots):
                break
            target = (seen[0] + correction[0], seen[1] + correction[1], seen[2])
            if not await pb.pick(robot, target, go) and go:
                print("  re-pick failed; stopping with the spots measured so far")
                break
    finally:
        detection_log.stop()
        if log:
            print(f"\nframes: {log.dir}")
    return rows, correction


def _motion(robot):
    from viam.services.motion import MotionClient
    return MotionClient.from_robot(robot, "builtin")


def report(rows, current):
    print("\nCALIBRATION")
    for r in rows:
        print(f"  spot ({r['spot'][0]}, {r['spot'][1]}): offset ({r['offset_mm'][0]:+.0f}, {r['offset_mm'][1]:+.0f}) mm")
    rec = recommend([r["offset_mm"] for r in rows])
    if rec is None:
        print("  no measurements")
        return None
    (dx, dy), spread = rec
    print(f"  median offset ({dx:+.0f}, {dy:+.0f}) mm, spread {spread:.0f} mm over {len(rows)} spot(s)")
    print(f"  current CAM_CORRECTION_MM = {current}; "
          f"grabs land ({current[0] - dx:+.0f}, {current[1] - dy:+.0f}) mm off the ball with it")
    if spread > SPREAD_OK_MM:
        print(f"  spread > {SPREAD_OK_MM} mm: not consistent; re-run (ball rolling? blurred frames?)")
    elif len(rows) >= 3:
        print(f"  -> set in pick_ball.py:  CAM_CORRECTION_MM = ... (-15, 20) -> ({dx:.0f}, {dy:.0f})")
    return {"median_offset_mm": (dx, dy), "spread_mm": spread}


async def _main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--go", action="store_true", help="move the arm (someone on the E-stop)")
    ap.add_argument("--spots", type=int, default=len(SPOTS), help="how many set-down spots to use")
    a = ap.parse_args()
    import pick_ball as pb
    from connection import connect
    spots = SPOTS[:max(1, a.spots)]
    print("MODE:", "LIVE (arm will move)" if a.go else "DRY RUN (no motion)")
    print(f"plan: pick 1 ball, then set it down / measure / re-pick at {len(spots)} spots: {spots}")
    robot = await connect()
    try:
        rows, _ = await calibrate(robot, spots, a.go)
        summary = report(rows, pb.CAM_CORRECTION_MM)
        if a.go and rows:
            out = ROOT / "observations" / "calibration"
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"{datetime.now():%Y%m%d-%H%M%S}.json"
            path.write_text(json.dumps({"rows": rows, "summary": summary,
                                        "current": pb.CAM_CORRECTION_MM}, indent=2) + "\n")
            print(f"saved {path}")
    finally:
        if a.go:
            from viam.components.gripper import Gripper
            await Gripper.from_robot(robot, pb.GRIPPER).open()
            await pb.move(_motion(robot), *pb.LOOK, True)
        await robot.close()


if __name__ == "__main__":
    asyncio.run(_main())
