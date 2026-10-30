"""Manan: find the cup with the angled camera view and aim the drop at its center.

This is the integration-side cup locator that produced 2/2 drops on Oct 29.
Rohan's cups.py (get_cups) can replace it once it's reliable; keep the same output:
(x, y, z) of the cup RIM CENTER in mm, world frame.

    python cup_aim.py        # read-only: uses the current camera view, moves nothing

How it works:
 1. Arm at CUP_LOOK (gripper tilted 45 deg forward, wrist 180) so the camera sees the red cup sides.
 2. cup-detector gives a box on the red body. The bottom-center of that box is the point of the
    cup base nearest the camera; project that pixel onto the table plane.
 3. Push it back by the base radius (along the viewing direction) to get the cup center.
 4. Apply CUP_CORRECTION_MM, measured from the two successful drops.
"""
import asyncio
import math

from viam.components.camera import Camera
from viam.proto.common import Pose, PoseInFrame
from viam.services.vision import VisionClient

import detection_log

CAMERA = "cam"
CUP_DETECTOR = "cup-detector"

# Gripper pose for looking at the far end of the table.
# Scan these tilts in order. 30 deg sees ~50-90 cm out, 45 deg sees ~87-130 cm out.
# (A fixed 45 deg view missed a cup placed at ~78 cm.)
CUP_LOOK_30 = dict(x=550, y=0, z=650, o_x=0.5, o_y=0, o_z=-0.866, theta=180)
CUP_LOOK_45 = dict(x=550, y=0, z=650, o_x=0.7, o_y=0, o_z=-0.7, theta=180)
CUP_LOOK = CUP_LOOK_45       # kept for older imports
CUP_LOOK_FAR = CUP_LOOK_30

# Wide scan. The arm slides SIDEWAYS (same orientation) rather than rotating, so the
# camera calibration measured at y=0 still holds in every view.
SCAN_SIDE_MM = 220
# Straight-down high view: sees cups ~30-90 cm out, wide; the pink inside is visible from above.
CUP_LOOK_TOP = dict(x=480, y=0, z=800, o_x=0, o_y=0, o_z=-1, theta=180)


def _shift(p, dy):
    return {**p, "y": p["y"] + dy}


CUP_SCAN = (
    CUP_LOOK_30,
    CUP_LOOK_45,
    _shift(CUP_LOOK_30, SCAN_SIDE_MM), _shift(CUP_LOOK_30, -SCAN_SIDE_MM),
    _shift(CUP_LOOK_45, SCAN_SIDE_MM), _shift(CUP_LOOK_45, -SCAN_SIDE_MM),
    CUP_LOOK_TOP,
)
CUP_LOOKS = CUP_SCAN

TABLE_Z = -10              # real tabletop height (fingers hit the table lower than the config's -23)
CUP_HEIGHT_MM = 120        # rim height above the table
CUP_BASE_RADIUS_MM = 30

# Measured Oct 29: this method said the cup center was at (926, -5); the drop that scored
# twice was released at (870, 20). Correction = truth - vision. Assumes the cup had not
# moved between the drops and the measurement - re-check with a moved cup.
CUP_CORRECTION_MM = (-56, 17)  # 3rd drop at y=+25 went in but slightly off-center in y; trimmed
# Same correction used from the 30 deg view (4th drop, cup at ~78 cm): it went in.
# With the ball held, the ball can hide the cup's red side and cup-detector returns nothing;
# find_cup() now tries the rim method (cup_rim.py) first for that reason.


def best_cup(dets):
    """Biggest red box that is taller than it is thin (ignores flat red lids and slivers)."""
    best = None
    for d in dets:
        w, h = d.x_max - d.x_min, d.y_max - d.y_min
        if w < 25 or h < 25:
            continue
        if best is None or w * h > (best.x_max - best.x_min) * (best.y_max - best.y_min):
            best = d
    return best


async def _pixel_to_table(robot, k, u, v, z_world):
    """Intersect the camera ray through pixel (u, v) with the horizontal plane z = z_world."""
    near = await robot.transform_pose(
        PoseInFrame(reference_frame=CAMERA, pose=Pose(x=0, y=0, z=0)), "world")
    far = await robot.transform_pose(
        PoseInFrame(reference_frame=CAMERA, pose=Pose(
            x=(u - k.center_x_px) / k.focal_x_px * 1000,
            y=(v - k.center_y_px) / k.focal_y_px * 1000,
            z=1000)), "world")
    a, b = near.pose, far.pose
    dz = b.z - a.z
    if abs(dz) < 1e-6:
        return None
    s = (z_world - a.z) / dz
    if not math.isfinite(s) or s <= 0:  # plane is behind the camera
        return None
    return (a.x + s * (b.x - a.x), a.y + s * (b.y - a.y), z_world), (a.x, a.y)


# Rim method (cup_rim.py): the pink-white opening, projected at rim height.
# Measured Oct 29: 45 deg view said (925, 6) for the cup that scored at (870, 20) -> (-55, +14);
# the 30 deg drops that went in used (-50, +17). Average:
RIM_CORRECTION_MM = (-52, 16)
# Straight-down views: the camera error measured on the ball picks at theta=180.
TOPDOWN_CORRECTION_MM = (-15, 20)
MAX_CUP_REACH_MM = 950        # farthest cup that scored was 870 from the base (main.py won't target beyond)
RIM_WIDTH_OK_MM = (70, 130)   # a solo cup opening is ~95 mm; reject blobs far off that


async def _color_frame(robot):
    import cv2
    import numpy as np
    images, _ = await Camera.from_robot(robot, CAMERA).get_images(filter_source_names=["color"])
    if not images:
        return None
    return cv2.imdecode(np.frombuffer(images[0].data, np.uint8), cv2.IMREAD_COLOR)


async def find_cups_by_rim(robot, topdown=False, seen=None):
    """Rim centers of EVERY cup opening in the current view (pink-white opening).
    Works when the held ball hides the red side.
    topdown=True when the camera points straight down (different calibration).
    seen: optional dict; gets the frame and every candidate (kept or not) for the log."""
    from cup_rim import find_cup_rims
    seen = {} if seen is None else seen
    seen.setdefault("items", [])
    frame = await _color_frame(robot)
    seen["frame"] = frame
    if frame is None:
        return []
    k = (await Camera.from_robot(robot, CAMERA).get_properties()).intrinsic_parameters
    rim_z = TABLE_Z + CUP_HEIGHT_MM
    cups = []
    for r in find_cup_rims(frame):
        item = {"kind": "rim", "box": r["box"], "px": (r["u"], r["v"]),
                "red_fraction": r.get("red_fraction"), "score": r.get("score")}
        seen["items"].append(item)
        c = await _pixel_to_table(robot, k, r["u"], r["v"], rim_z)
        left = await _pixel_to_table(robot, k, r["box"][0], r["v"], rim_z)
        right = await _pixel_to_table(robot, k, r["box"][2], r["v"], rim_z)
        if c is None or left is None or right is None:
            item.update(kept=False, why="ray misses table")
            continue
        width = math.dist(left[0][:2], right[0][:2])
        item["width_mm"] = round(width, 1)
        if not (RIM_WIDTH_OK_MM[0] <= width <= RIM_WIDTH_OK_MM[1]):
            item.update(kept=False, why=f"{width:.0f} mm wide")
            continue
        (x, y, z), _ = c
        if math.hypot(x, y) > MAX_CUP_REACH_MM:
            item["beyond_reach"] = True     # still a cup: listed in the survey, never targeted
        corr = TOPDOWN_CORRECTION_MM if topdown else RIM_CORRECTION_MM
        cups.append((x + corr[0], y + corr[1], z))
        item.update(kept=True, world_mm=cups[-1])
    return cups


def _first_in_reach(cups):
    return next((c for c in cups if math.hypot(c[0], c[1]) <= MAX_CUP_REACH_MM), None)


async def find_cup_by_rim(robot, topdown=False):
    """The first cup opening within reach in the current view, or None."""
    return _first_in_reach(await find_cups_by_rim(robot, topdown))


def cup_boxes(dets):
    """Every red box big enough to be a cup (best_cup keeps only the biggest)."""
    return [d for d in dets if d.x_max - d.x_min >= 25 and d.y_max - d.y_min >= 25]


# YOLO (cup_detection.py) runs on every frame next to the rim method. Rim positions are
# calibrated by real drops, so when both see the same cup the rim position is used and the
# YOLO-vs-rim offset is recorded (YOLO_OFFSETS) to calibrate cup_detection.CUP_CORRECTION_MM.
# Cups only YOLO sees (rim rejected or missed them) are added with YOLO's depth position.
USE_YOLO = True           # main.py --no-yolo: rim method only
YOLO_OFFSETS = []         # (dx, dy) rim minus raw YOLO, mm, one per matched sighting
_yolo_off = None          # reason YOLO was switched off for this run (missing model/library)
_yolo_errors = set()


async def _yolo_cups(robot, seen):
    """Run YOLO on the current view; fills seen (frame, depth, items). Never raises."""
    global _yolo_off
    if not USE_YOLO or _yolo_off:
        return
    try:
        import cup_detection
        await cup_detection.find_cups(robot, seen=seen)
    except (ImportError, FileNotFoundError) as e:     # can't work this run: say so once
        _yolo_off = f"{type(e).__name__}: {e}"
        print(f"  [cup] YOLO unavailable ({str(e)[:80]}); rim method only")
    except Exception as e:                            # bad frame / arm moved: skip this frame
        msg = f"{type(e).__name__}: {str(e)[:80]}"
        if msg not in _yolo_errors:                   # each distinct problem printed once
            _yolo_errors.add(msg)
            print(f"  [cup] YOLO skipped a frame ({msg})")


def combine(calibrated_items, yolo_items):
    """Cups for one frame: every calibrated (rim / red-body) cup, plus YOLO cups they don't
    already cover. Same cup = the calibrated box's center lies inside the YOLO box.
    Marks the YOLO items in place; returns the extra cups (world mm, corrected)."""
    import cup_detection
    extra = []
    for it in yolo_items:
        if not it.get("kept") or it.get("world_mm") is None:
            continue
        x0, y0, x1, y1 = it["box"]
        raw = tuple(it["world_mm"])
        same = next((c for c in calibrated_items
                     if x0 <= (c["box"][0] + c["box"][2]) / 2 <= x1
                     and y0 <= (c["box"][1] + c["box"][3]) / 2 <= y1), None)
        if same is not None:
            off = (same["world_mm"][0] - raw[0], same["world_mm"][1] - raw[1])
            it.update(duplicate_of=same["kind"], offset_vs_rim_mm=off)
            YOLO_OFFSETS.append(off)
            continue
        cx, cy = cup_detection.CUP_CORRECTION_MM
        cup = (raw[0] + cx, raw[1] + cy, raw[2])
        it.update(world_mm=cup, raw_world_mm=raw)
        extra.append(cup)
    return extra


async def find_cups(robot, topdown=False):
    """Rim centers (x, y, z) in mm, world frame, of every cup in the CURRENT view ([] if none).
    Rim method + YOLO on every frame (see combine); the red-body detector only when
    neither found anything in an angled view."""
    seen = {"items": []}
    try:
        cups = await find_cups_by_rim(robot, topdown, seen)
    except Exception as e:  # never let one method break the loop
        print(f"  [cup] rim method failed ({e})")
        cups = []
    ys = {}
    await _yolo_cups(robot, ys)
    yolo_items = ys.get("items", [])
    detector = "cup_rim" + ("+yolo" if ys else "")
    try:
        if not cups and not topdown and not any(it.get("kept") for it in yolo_items):
            detector += "+detector"
            cups = await _cups_from_red_bodies(robot, seen["items"])
        calibrated = [it for it in seen["items"] if it.get("kept") and it.get("world_mm") is not None]
        cups = cups + combine(calibrated, yolo_items)
    finally:
        detection_log.record(detector, seen.get("frame") if seen.get("frame") is not None else ys.get("frame"),
                             seen["items"] + yolo_items, depth=ys.get("depth"), topdown=topdown)
    return cups


async def _cups_from_red_bodies(robot, items):
    cups = []
    all_dets = await VisionClient.from_robot(robot, CUP_DETECTOR).get_detections_from_camera(CAMERA)
    dets = cup_boxes(all_dets)
    for d in all_dets:
        if d not in dets:
            items.append({"kind": "red_body", "box": (d.x_min, d.y_min, d.x_max, d.y_max),
                          "kept": False, "why": "smaller than 25 px"})
    if not dets:
        return []
    k = (await Camera.from_robot(robot, CAMERA).get_properties()).intrinsic_parameters
    for d in dets:
        item = {"kind": "red_body", "box": (d.x_min, d.y_min, d.x_max, d.y_max),
                "confidence": round(d.confidence, 3)}
        items.append(item)
        hit = await _pixel_to_table(robot, k, (d.x_min + d.x_max) / 2, d.y_max, TABLE_Z)
        if hit is None:
            item.update(kept=False, why="ray misses table")
            continue
        (bx, by, _), (camx, camy) = hit
        # push from the nearest base point to the base center, away from the camera
        dx, dy = bx - camx, by - camy
        n = math.hypot(dx, dy) or 1.0
        cx = bx + CUP_BASE_RADIUS_MM * dx / n + CUP_CORRECTION_MM[0]
        cy = by + CUP_BASE_RADIUS_MM * dy / n + CUP_CORRECTION_MM[1]
        cups.append((cx, cy, TABLE_Z + CUP_HEIGHT_MM))
        item.update(kept=True, world_mm=cups[-1])
    return cups


async def find_cup(robot, topdown=False):
    """The first cup within reach in the CURRENT view, or None (see find_cups)."""
    return _first_in_reach(await find_cups(robot, topdown))


if __name__ == "__main__":
    from connection import connect

    async def _main():
        robot = await connect()
        try:
            print("cup rim centers:", await find_cups(robot))
        finally:
            await robot.close()

    asyncio.run(_main())
