"""Why isn't the ball found? Wide look + close-up look with EVERY circle candidate printed,
including the ones the filters reject and why. Planned (obstacle-checked) moves only; no pick.

    python scripts/ball_check.py --go        # someone near the E-stop

Paste the whole output. Things it tells you:
  r_ratio  = measured radius / expected for a 40 mm ball (1.0 = same size as the Oct 29 balls)
  fill     = share of the circle that is orange (needs >= 0.7)
  ring     = share of the ring just outside that is orange (needs <= 0.5; high = a neighbour is
             very close, or the orange region is bigger than a ball)
  orange_px = how many pixels in the whole frame pass the orange color test
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from viam.components.camera import Camera  # noqa: E402
from viam.proto.common import Pose, PoseInFrame  # noqa: E402
from viam.services.motion import MotionClient  # noqa: E402

import ball_finder  # noqa: E402
import pick_ball  # noqa: E402
from connection import connect  # noqa: E402


async def report(robot, label):
    cam = Camera.from_robot(robot, pick_ball.CAMERA)
    k = (await cam.get_properties()).intrinsic_parameters
    frame = await pick_ball._frame(cam)
    campos = (await robot.transform_pose(PoseInFrame(reference_frame=pick_ball.CAMERA, pose=Pose()), "world")).pose
    dist = campos.z - pick_ball.BALL_CENTER_Z
    exp_r = k.focal_x_px * (pick_ball.BALL_DIAMETER_MM / 2) / dist
    m = ball_finder.orange_mask(frame)
    print(f"\n[{label}] camera z={campos.z:.0f} mm, expected ball radius {exp_r:.0f}px, "
          f"orange_px={int(np.count_nonzero(m))}")
    # every orange blob, with its equivalent radius
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
    blobs = sorted(((stats[i, cv2.CC_STAT_AREA], cent[i]) for i in range(1, n)), reverse=True)[:8]
    for area, (u, v) in blobs:
        if area < 200:
            continue
        r_eq = (area / np.pi) ** 0.5
        print(f"  orange blob at ({u:.0f},{v:.0f}) area {area}px  equiv r {r_eq:.0f}px  r_ratio {r_eq / exp_r:.2f}")
    # Hough with a WIDE radius window so rejected sizes show up too
    blur = cv2.GaussianBlur(m, (9, 9), 2)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=exp_r * 1.2, param1=100,
                               param2=max(10, int(exp_r * 0.5)), minRadius=int(exp_r * 0.6),
                               maxRadius=int(exp_r * 1.8))
    if circles is None:
        print("  no circles at all (even with a 0.6-1.8x size window)")
    else:
        for u, v, r in circles[0][:10]:
            disk = np.zeros_like(m); cv2.circle(disk, (int(u), int(v)), int(r), 255, -1)
            fill = np.count_nonzero(m & disk) / max(1, np.count_nonzero(disk))
            ring = np.zeros_like(m); cv2.circle(ring, (int(u), int(v)), int(r * 1.45), 255, -1)
            cv2.circle(ring, (int(u), int(v)), int(r * 1.15), 0, -1)
            ro = np.count_nonzero(m & ring) / max(1, np.count_nonzero(ring))
            why = []
            if not 0.8 <= r / exp_r <= 1.25:
                why.append("SIZE out of 0.8-1.25x")
            if fill < 0.7:
                why.append("FILL < 0.7")
            if ro > 0.5:
                why.append("RING > 0.5 (neighbour too close / big orange area)")
            print(f"  circle ({u:.0f},{v:.0f}) r={r:.0f}px r_ratio {r / exp_r:.2f} fill {fill:.2f} ring {ro:.2f}"
                  f"  -> {'OK' if not why else ', '.join(why)}")
    kept = ball_finder.find_balls(frame, exp_r)
    print(f"  find_balls keeps {len(kept)}: " + ", ".join(
        f"({b['u']:.0f},{b['v']:.0f}) r={b['r']:.0f}{' TOUCHING' if b['touching'] else ''}" for b in kept))
    name = f"ball_check_{label}.jpg"
    cv2.imwrite(name, frame)
    print(f"  saved {name}")


async def main():
    go = "--go" in sys.argv
    robot = await connect()
    try:
        motion = MotionClient.from_robot(robot, "builtin")
        await pick_ball.move(motion, *pick_ball.LOOK, go)
        await report(robot, "wide")
        coarse = await pick_ball.locate_ball(robot)
        if coarse is None:
            print("\nwide look picked no ball; stopping here")
            return
        x, y = coarse[0] + pick_ball.CAM_OFFSET_XY[0], coarse[1] + pick_ball.CAM_OFFSET_XY[1]
        print(f"\nmoving the camera over the chosen ball ({coarse[0]:.0f}, {coarse[1]:.0f})")
        await pick_ball.move(motion, x, y, pick_ball.MEASURE_HEIGHT_Z, go)
        if go:
            await asyncio.sleep(0.5)
        await report(robot, "closeup")
        await pick_ball.move(motion, *pick_ball.LOOK, go)
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
