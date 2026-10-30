"""Ball detection debugger: shows HOW each ball was found and why a pick could miss.

Read-only: never moves the arm.

    python scripts/debug_ball.py                          # replay the newest attempts' ball frames
    python scripts/debug_ball.py observations/detections/20261029-191013 [more folders...]
    python scripts/debug_ball.py frame.jpg --r 76         # any image; --r = expected ball radius, px
    python scripts/debug_ball.py live                     # camera at the arm's current pose

For every frame it writes a panel to observations/ball_debug/<time>/:
  left : the frame. Orange mask outline (cyan), each blob's centroid (white x), the fitted
         circle (green = edge fit, yellow = Hough split, magenta = YOLO outline), the
         expected ball size at that center (thin grey), YOLO boxes (magenta).
  right: the orange mask, with any rejected blob's reason.
and prints one line per ball:
  method  center px  r vs expected  bulge  fill  yolo  sharpness
  bulge     = px between the blob's centroid and the fitted circle: a shadow or motion blur
              pulling the blob to one side (the edge fit ignores it, a centroid would not).
  sharpness = Laplacian variance around the ball; well below the frame's others = blurred,
              i.e. the arm was still moving when the frame was taken.
live also prints the ball's world position and where the gripper would grab
(+ pick_ball.CAM_CORRECTION_MM), the part to check when the gripper lands "slightly back".
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ball_finder  # noqa: E402

OUT = ROOT / "observations" / "ball_debug"
COLORS = {"edge_fit": (0, 220, 0), "split": (0, 220, 255), "hough": (0, 220, 255), "yolo_mask": (255, 0, 255)}


def panel(bgr, expected_r, balls, dbg, title=""):
    """Side-by-side debug image: annotated frame | orange mask."""
    img = bgr.copy()
    mask = dbg["mask"]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (255, 255, 0), 1)
    for (x0, y0, x1, y1), conf, label in dbg.get("yolo", []):
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 255), 1)
        cv2.putText(img, f"{label} {conf:.2f}", (int(x0), max(12, int(y0) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
    for blob in dbg["blobs"]:
        cu, cv_ = (int(round(c)) for c in blob["centroid"])
        cv2.drawMarker(img, (cu, cv_), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 2)
    for b in balls:
        u, v, r = (int(round(b[k])) for k in ("u", "v", "r"))
        color = COLORS.get(b["method"], (255, 255, 255))
        cv2.circle(img, (u, v), int(expected_r), (160, 160, 160), 1)
        cv2.circle(img, (u, v), r, color, 2)
        cv2.drawMarker(img, (u, v), color, cv2.MARKER_CROSS, 14, 2)
        cv2.putText(img, f"{b['method']} r{r}/{expected_r:.0f} bulge {b.get('bulge_px')}",
                    (u - r, min(img.shape[0] - 6, v + r + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                    cv2.LINE_AA)
    cv2.putText(img, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    right = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    y = 24
    for blob in dbg["blobs"]:
        if blob.get("why"):
            cv2.putText(right, f"{blob['method']} at {blob['centroid']}: {blob['why']}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            y += 20
    return np.hstack([img, right])


def describe(bgr, balls):
    if not balls:
        return "    no ball"
    lines = []
    for b in balls:
        sharp = ball_finder.sharpness(bgr, b["u"], b["v"], b["r"])
        lines.append(f"    {b['method']:<9} center ({b['u']:.0f},{b['v']:.0f}) r {b['r']:.0f}  "
                     f"bulge {b.get('bulge_px')}  fill {b['fill']}  yolo {b.get('yolo')}  "
                     f"sharpness {sharp:.0f}" + ("  TOUCHING" if b.get("touching") else ""))
    return "\n".join(lines)


def run_frame(bgr, expected_r, name, out):
    dbg = {}
    balls = ball_finder.find_balls(bgr, expected_r, debug=dbg)
    cv2.imwrite(str(out / f"{name}.jpg"), panel(bgr, expected_r, balls, dbg, name),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{name}: expected r {expected_r:.0f} px, {len(dbg['blobs'])} orange blob(s)")
    print(describe(bgr, balls))
    return balls


def offline(paths, r):
    out = OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    if not paths:                                    # newest 5 attempt folders that have ball frames
        folders = sorted((ROOT / "observations" / "detections").glob("*"), reverse=True)
        paths = [f for f in folders if (f / "log.jsonl").exists()
                 and '"ball_' in (f / "log.jsonl").read_text()][:5]
    for p in map(Path, paths):
        if p.is_dir():
            for line in map(json.loads, (p / "log.jsonl").read_text().splitlines()):
                if str(line.get("detector", "")).startswith("ball") and line.get("image"):
                    er = line.get("expected_r_px") or r or 35
                    run_frame(cv2.imread(str(p / line["image"])), er,
                              f"{p.name}_{line['n']:02d}_{line['stage']}", out)
        else:
            run_frame(cv2.imread(str(p)), r or 35, p.stem, out)
    print(f"\npanels: {out}")


async def live():
    from viam.components.camera import Camera
    from viam.proto.common import Pose, PoseInFrame

    import pick_ball
    from connection import connect
    from cup_aim import _pixel_to_table

    out = OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    robot = await connect()
    try:
        cam = Camera.from_robot(robot, pick_ball.CAMERA)
        k = (await cam.get_properties()).intrinsic_parameters
        bgr = await pick_ball._frame(cam)
        cam_pos = (await robot.transform_pose(
            PoseInFrame(reference_frame=pick_ball.CAMERA, pose=Pose(x=0, y=0, z=0)), "world")).pose
        dist = cam_pos.z - pick_ball.BALL_CENTER_Z
        expected_r = k.focal_x_px * (pick_ball.BALL_DIAMETER_MM / 2) / dist
        print(f"camera at ({cam_pos.x:.0f},{cam_pos.y:.0f},{cam_pos.z:.0f}) mm, {dist:.0f} mm above ball centers")
        balls = run_frame(bgr, expected_r, "live", out)
        cx, cy = pick_ball.CAM_CORRECTION_MM
        for b in balls:
            hit = await _pixel_to_table(robot, k, b["u"], b["v"], pick_ball.BALL_CENTER_Z)
            if hit:
                (x, y, _), _ = hit
                print(f"    -> ball at world ({x:.0f},{y:.0f}); gripper would grab at ({x + cx:.0f},{y + cy:.0f})"
                      f"  [CAM_CORRECTION_MM {pick_ball.CAM_CORRECTION_MM}]"
                      + ("  (in/next to a cup: skipped)" if pick_ball.near_cup((x, y)) else ""))
        print(f"\npanel: {out / 'live.jpg'}")
    finally:
        await robot.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="detection-log folders or images (default: newest attempts); "
                                             "or the word 'live'")
    ap.add_argument("--r", type=float, help="expected ball radius in px, for plain images")
    ap.add_argument("--no-yolo", action="store_true", help="color detection only")
    ap.add_argument("--panels", action="store_true", help="accepted for symmetry with "
                    "compare_ball_detection.py; panels are always saved")
    a = ap.parse_args()
    ball_finder.USE_YOLO = not a.no_yolo
    view = "camera view at the look pose: about x 230-640 mm, y -350..+380 mm (world)"
    print(view)
    if a.paths == ["live"]:
        asyncio.run(live())
    else:
        offline(a.paths, a.r)


if __name__ == "__main__":
    main()
