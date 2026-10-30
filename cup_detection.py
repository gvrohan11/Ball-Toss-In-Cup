"""YOLO cup detection: find cup rims in the world frame from the arm's RGB-D camera.

This is the cup detection main.py uses. A YOLO11m-seg instance-segmentation model
(COCO "cup" class, cup_vision/core.py) finds cups in the color image, then the
aligned depth on the visible rim is fit with a plane + circle to get the 3D rim
center. The camera point is transformed to the world frame with Viam's frame system.
No red pixels are required, so it works when the held ball hides the cup's red side.

    from cup_detection import find_cup, find_cups
    cup = await find_cup(robot)          # (x, y, z) rim center, mm, world frame, or None
    cups = await find_cups(robot)        # every cup with a validated rim, best first

Command line (read-only, never moves the arm):

    python cup_detection.py                      # locate cups from the current view
    python cup_detection.py live                 # OpenCV preview with boxes and rims
    python cup_detection.py capture --frames 30 --output /tmp/cup-data   # save RGB-D pairs
    python cup_detection.py offline --input /tmp/cup-data --output /tmp/cup-results

Detector settings: config/cup-detector.thresholds.json (DetectorConfig fields).
rotation_search is off: the wrist camera is always upright (theta=180), and on the Oct 29
survey frames the 4 quarter-turn passes only added duplicate boxes (cup + neighbor) while
taking ~1.7 s per frame instead of ~0.5 s. Every red cup was found without them.
The same detector runs as a Viam vision service from cup_vision/ (see docs/cup-detection.md).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "cup-detector.thresholds.json"

# Accept a cup only if the model is this sure and the fitted rim looks like a cup opening.
MIN_CONFIDENCE = 0.30
RIM_DIAMETER_OK_MM = (60, 130)   # a red solo cup opening is ~95 mm
# Camera calibration correction (truth - measured), world mm. Measure on a real cup and
# update; the ball picks at theta=180 needed (-15, +20) for the same camera.
CUP_CORRECTION_MM = (0.0, 0.0)
TABLE_Z_MM = -10       # real tabletop height (cup_aim.TABLE_Z)
CUP_RADIUS_MM = 47     # solo cup rim radius

# COCO "cup" is not "red solo cup": on the Oct 29 survey frames YOLO also boxed a white paper
# cup (0.92), a Coke can (0.51) and a pink soda can. What only the solo cups have is the
# pink-white opening (cup_rim's pink mask): 0.28-0.68 of the YOLO mask on every red cup,
# <= 0.09 on the paper cup and both cans. Every survey view looks down into the opening.
MIN_PINK_FRACTION = 0.18
EDGE_MARGIN_PX = 3          # an OPENING touching the frame edge is cut off: center unreliable
                            # (a body cut off at the bottom is fine: the rim is what's located)
CONTAINED_FRACTION = 0.5    # a box mostly inside a stronger box is the same cup (or cup + neighbor)
RIM_BAND_MM = 15            # depth points this close to the top of the cup are the rim


def _box_fraction_inside(box, other):
    """Share of `box`'s area that lies inside `other`."""
    w = max(0, min(box[2], other[2]) - max(box[0], other[0]))
    h = max(0, min(box[3], other[3]) - max(box[1], other[1]))
    return w * h / max(1, (box[2] - box[0]) * (box[3] - box[1]))


def _touches_edge(mask, margin):
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return xs.min() <= margin or ys.min() <= margin or xs.max() >= w - 1 - margin or ys.max() >= h - 1 - margin


def screen(detections: Sequence[Any], bgr: np.ndarray) -> list[tuple[Any, dict[str, Any]]]:
    """Decide which YOLO "cup" boxes are red solo cups worth locating.
    Returns (detection, log item) pairs, strongest first; item["kept"] says which, and
    item["why"] says why not. Items use the detection-log format (kind/box/kept/why)."""
    from cup_rim import _masks
    pink, red = _masks(bgr)
    h, w = bgr.shape[:2]
    out, kept_boxes = [], []
    for d in sorted(detections, key=lambda d: d.confidence, reverse=True):
        x0, y0, x1, y1 = (int(v) for v in d.bbox)
        mask = getattr(d, "object_mask", None)
        if mask is None or not mask.any():
            mask = np.zeros((h, w), bool)
            mask[y0:y1, x0:x1] = True
        opening = (pink > 0) & mask
        pink_frac = float(np.count_nonzero(opening)) / max(1, int(mask.sum()))
        red_frac = float(np.count_nonzero(red[mask])) / max(1, int(mask.sum()))
        item = {"kind": "yolo", "box": (x0, y0, x1, y1), "confidence": round(float(d.confidence), 3),
                "pink_fraction": round(pink_frac, 3), "red_fraction": round(red_frac, 3), "kept": False}
        if d.confidence < MIN_CONFIDENCE:
            item["why"] = f"confidence {d.confidence:.2f}"
        elif pink_frac < MIN_PINK_FRACTION:
            item["why"] = f"not a red cup (opening {pink_frac:.0%})"
        elif _touches_edge(opening, EDGE_MARGIN_PX):
            item["why"] = "opening cut off at frame edge"
        elif any(_box_fraction_inside((x0, y0, x1, y1), k) >= CONTAINED_FRACTION for k in kept_boxes):
            item["why"] = "overlaps a stronger box"
        else:
            item["kept"] = True
            kept_boxes.append((x0, y0, x1, y1))
        out.append((d, item))
    return out

_DETECTOR = None  # loading YOLO takes seconds; reuse it across calls


def load_detector(config: Path | None = None, model: str | None = None) -> Any:
    """CupDetector built from the thresholds JSON (relative model paths resolve from the repo)."""
    from cup_vision.core import CupDetector, DetectorConfig

    path = Path(config) if config else DEFAULT_CONFIG
    attributes: dict[str, Any] = {}
    if path.exists():
        attributes = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(attributes, dict):
            raise ValueError(f"{path} must contain a JSON object of DetectorConfig attributes")
    if model:
        attributes["model_path"] = model
    model_path = attributes.get("model_path", DetectorConfig().model_path)
    if not Path(model_path).is_absolute() and (ROOT / model_path).exists():
        attributes["model_path"] = str(ROOT / model_path)
    detector_config = DetectorConfig(**attributes)
    detector_config.validate()
    return CupDetector(detector_config)


def detector() -> Any:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = load_detector()
    return _DETECTOR


@dataclass
class FoundCup:
    center_mm: tuple[float, float, float]   # rim center, world frame
    confidence: float
    method: str                              # "rim_fit" or "body_depth"
    diameter_mm: float
    residual_mm: float
    bbox: tuple[int, int, int, int]


def _camera_to_world(robot: Any, camera: str):
    """4x4 camera->world transform from three transform_pose calls (frame system, arm still)."""
    from viam.proto.common import Pose, PoseInFrame

    async def build() -> np.ndarray:
        pts = []
        for x, y, z in ((0, 0, 0), (100, 0, 0), (0, 100, 0), (0, 0, 100)):
            r = await robot.transform_pose(PoseInFrame(reference_frame=camera, pose=Pose(x=x, y=y, z=z, o_z=1)), "world")
            pts.append(np.array([r.pose.x, r.pose.y, r.pose.z]))
        T = np.eye(4)
        T[:3, :3] = np.column_stack([(pts[i] - pts[0]) / 100 for i in (1, 2, 3)])
        T[:3, 3] = pts[0]
        return T
    return build()


def body_estimate(det: Any, depth: np.ndarray, k: dict[str, float], T: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Rim center from the YOLO mask + aligned depth when no rim ellipse was fit.
    The top RIM_BAND_MM of the cup's depth points is the rim; a circle fit to them gives
    the center. Every survey view looks down into the opening, so the rim ring is visible
    all round; a partial arc still fits. (Falls back to the band's centroid.)"""
    x0, y0, x1, y1 = det.bbox
    mask = np.zeros(depth.shape, bool)
    if getattr(det, "object_mask", None) is not None:
        mask = det.object_mask.copy()
    else:
        mask[y0:y1, x0:x1] = True
    z = depth.astype(float)
    mask &= (z > 100) & (z < 3000)
    if mask.sum() < 50:
        return None
    zs = z[mask]
    mask &= np.abs(z - np.median(zs)) < 150          # drop background leaking through the mask
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    zc = z[ys, xs]
    cam = np.column_stack(((xs - k["cx"]) * zc / k["fx"], (ys - k["cy"]) * zc / k["fy"], zc, np.ones_like(zc)))
    world = (T @ cam.T).T[:, :3]
    above = world[world[:, 2] > TABLE_Z_MM + 15]      # cup body, not the table
    if len(above) < 30:
        return None
    rim_z = float(np.percentile(above[:, 2], 95))      # 95th: ignore flying depth pixels
    band = above[above[:, 2] >= rim_z - RIM_BAND_MM][:, :2]
    if len(band) < 20:
        return None
    fit = fit_circle(band)
    if fit is not None and 0.6 * CUP_RADIUS_MM <= fit[2] <= 1.4 * CUP_RADIUS_MM:
        cx, cy, _, residual = fit
    else:
        (cx, cy), residual = np.median(band, axis=0), float("nan")
    return np.array([cx, cy, rim_z]), residual


def fit_circle(xy: np.ndarray) -> tuple[float, float, float, float] | None:
    """Least-squares circle through 2D points: (cx, cy, radius, median residual) or None."""
    a = np.column_stack((2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))))
    b = (xy ** 2).sum(axis=1)
    if np.linalg.cond(a) > 1e8:
        return None
    (cx, cy, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    r2 = c + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    residual = float(np.median(np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r)))
    return float(cx), float(cy), r, residual


def locate_cups(det: Any, bgr: np.ndarray, depth: np.ndarray, k: dict[str, float],
           T: np.ndarray) -> tuple[list[FoundCup], list[dict[str, Any]]]:
    """The robot-free part of find_cups: detect, screen (red solo cups only), locate in 3D.
    Returns (cups best first, detection-log items for every YOLO box)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cups, items = [], []
    for d, item in screen(det.detect(rgb), bgr):
        items.append(item)
        if not item["kept"]:
            continue
        rim = det.localize_rim(d, rgb, depth, k)
        if rim is not None and RIM_DIAMETER_OK_MM[0] <= rim.diameter_mm <= RIM_DIAMETER_OK_MM[1]:
            c = T @ np.r_[np.asarray(rim.center_mm), 1.0]
            cup = FoundCup(tuple(map(float, c[:3])), float(d.confidence), "rim_fit",
                           float(rim.diameter_mm), float(rim.residual_mm), tuple(d.bbox))
        else:
            body = body_estimate(d, depth, k, T)
            if body is None:
                item.update(kept=False, why="no usable depth on the cup")
                continue
            c, residual = body
            cup = FoundCup(tuple(map(float, c)), float(d.confidence), "body_depth",
                           float(2 * CUP_RADIUS_MM), residual, tuple(d.bbox))
        item.update(world_mm=cup.center_mm, method=cup.method)
        cups.append(cup)
    cups.sort(key=lambda c: (c.method == "rim_fit", c.confidence), reverse=True)
    return cups, items


async def find_cups(robot: Any, *, camera: str = "cam", arm: str = "arm", det: Any = None,
                    seen: dict | None = None) -> list[FoundCup]:
    """Every red solo cup YOLO finds in the current view, located in 3D, best first.
    Uses the precise rim fit when the rim ellipse is visible, otherwise the rim-band
    circle fit on depth. Keep the arm still while this runs.
    seen: optional dict; gets the frame, depth and every YOLO box (kept or not) for the log."""
    from viam.components.arm import Arm
    from viam.components.camera import Camera
    from viam.media.video import CameraMimeType

    det = det or detector()
    cam, arm_c = Camera.from_robot(robot, camera), Arm.from_robot(robot, arm)
    before = np.asarray((await arm_c.get_joint_positions()).values)
    props = await cam.get_properties()
    kp = props.intrinsic_parameters
    k = {"fx": kp.focal_x_px, "fy": kp.focal_y_px, "cx": kp.center_x_px, "cy": kp.center_y_px}
    images, _ = await cam.get_images()
    src = {im.name: im for im in images}
    if not {"color", "depth"} <= set(src) or src["depth"].mime_type != CameraMimeType.VIAM_RAW_DEPTH:
        raise ValueError(f"camera must return color + raw depth, got {list(src)}")
    bgr = cv2.imdecode(np.frombuffer(src["color"].data, np.uint8), cv2.IMREAD_COLOR)
    depth = np.asarray(src["depth"].bytes_to_depth_array(), dtype=np.uint16)
    T = await _camera_to_world(robot, camera)
    cups, items = await asyncio.to_thread(locate_cups, det, bgr, depth, k, T)
    if seen is not None:
        seen.update(frame=bgr, depth=depth, items=items)
    after = np.asarray((await arm_c.get_joint_positions()).values)
    if np.max(np.abs(after - before)) > 0.2:
        raise RuntimeError("arm moved during capture; hold still")
    return cups


# Close-up look: put the CAMERA (via the frame system, so the camera-gripper offset is
# handled by Viam) this far from the coarse cup estimate, looking at it, and re-measure.
REFINE_DISTANCE_MM = 380
REFINE_MATCH_MM = 150          # the close-up cup must be within this of the coarse estimate
CAM_THETA_FOR_GRIPPER_180 = 82.3   # camera frame theta when the gripper is at theta=180


def refine_view(cup: Sequence[float]) -> dict[str, float]:
    """Camera pose (world) looking at the cup: straight down for near cups, 45 deg for far ones."""
    cx, cy, cz = cup
    o = (0.0, 0.0, -1.0) if cx <= 650 else (0.7071, 0.0, -0.7071)
    return dict(x=cx - REFINE_DISTANCE_MM * o[0], y=cy, z=cz - REFINE_DISTANCE_MM * o[2],
                o_x=o[0], o_y=o[1], o_z=o[2], theta=CAM_THETA_FOR_GRIPPER_180)


async def refine_cup(robot: Any, cup: Sequence[float], go: bool, camera: str = "cam") -> tuple[float, float, float]:
    """Move the camera close to the coarse cup, re-detect with YOLO (the rim circle is fit
    when visible) and return the refined rim center. Falls back to the coarse estimate."""
    from viam.proto.common import Pose, PoseInFrame
    from viam.services.motion import MotionClient

    view = refine_view(cup)
    print(f"  refine: camera to ({view['x']:.0f}, {view['y']:.0f}, {view['z']:.0f}) looking at the cup")
    if not go:
        return tuple(cup)
    try:
        motion = MotionClient.from_robot(robot, "builtin")
        await motion.move(component_name=camera, destination=PoseInFrame(reference_frame="world", pose=Pose(**view)))
        await asyncio.sleep(0.3)  # settle
        cups = await find_cups(robot, camera=camera)
    except Exception as e:
        print(f"  refine failed ({str(e)[:80]}); using the coarse estimate")
        return tuple(cup)
    near = [c for c in cups if np.hypot(c.center_mm[0] - cup[0], c.center_mm[1] - cup[1]) <= REFINE_MATCH_MM]
    if not near:
        print("  refine: target cup not seen up close; using the coarse estimate")
        return tuple(cup)
    best = min(near, key=lambda c: (c.method != "rim_fit", np.hypot(c.center_mm[0] - cup[0], c.center_mm[1] - cup[1])))
    x, y, z = best.center_mm
    x, y = x + CUP_CORRECTION_MM[0], y + CUP_CORRECTION_MM[1]
    print(f"  refine: rim center ({x:.0f}, {y:.0f}, {z:.0f}) via {best.method} "
          f"(moved {np.hypot(x - cup[0], y - cup[1]):.0f} mm from coarse)")
    return (x, y, z)


async def find_cup(robot: Any, topdown: bool = False, **kwargs: Any) -> tuple[float, float, float] | None:
    """Rim center (x, y, z) in mm, world frame, of the best cup in the current view, or None.
    Same output as the old cup_aim.find_cup; `topdown` is accepted for compatibility (the
    3D rim fit does not depend on the viewing angle)."""
    try:
        cups = await find_cups(robot, **kwargs)
    except Exception as e:  # a bad frame must not crash the loop: treat it as "no cup here"
        print(f"  [cup] YOLO cup detection failed: {e}")
        return None
    if not cups:
        print("  [cup] YOLO: no cup in this view")
        return None
    c = cups[0]
    x, y, z = c.center_mm
    print(f"  [cup] YOLO cup {c.confidence:.2f} via {c.method}: ({x:.0f}, {y:.0f}, {z:.0f}) "
          f"({len(cups)} cup(s) in view)")
    return (x + CUP_CORRECTION_MM[0], y + CUP_CORRECTION_MM[1], z)



def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_image(path: Path, image_bgr: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"Cannot write {path}")


def make_detector(args: argparse.Namespace) -> Any:
    return load_detector(args.config, args.model)


def decode_pair(images: Sequence[Any], expected_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Decode by source name; do not assume that RGB arrives before depth."""
    from viam.media.video import CameraMimeType

    sources = {image.name: image for image in images}
    if not {"color", "depth"}.issubset(sources):
        raise ValueError(f"Camera must return color and depth, received {list(sources)}")
    color_bgr = cv2.imdecode(np.frombuffer(sources["color"].data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise ValueError("Color stream is not a decodable JPEG or PNG")
    if sources["depth"].mime_type != CameraMimeType.VIAM_RAW_DEPTH:
        raise ValueError("Expected image/vnd.viam.dep. Set little_endian_depth=false on the camera.")
    depth_mm = np.asarray(sources["depth"].bytes_to_depth_array(), dtype=np.uint16)
    if color_bgr.shape[:2] != expected_size or depth_mm.shape != expected_size:
        raise ValueError(f"RGB/depth/intrinsics size mismatch: {color_bgr.shape[:2]}, "
                         f"{depth_mm.shape}, expected {expected_size}")
    return color_bgr, depth_mm


def annotate(
    color_bgr: np.ndarray,
    detections: Sequence[Any],
    depth_mm: np.ndarray | None,
    intrinsics: dict[str, float] | None,
    detector: Any,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Draw full-frame detections and expose failed rim localization explicitly."""
    from cup_vision.core import localize_rim

    rendered = color_bgr.copy()
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    results: list[dict[str, Any]] = []
    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = (int(value) for value in detection.bbox)
        mask = getattr(detection, "object_mask", None)
        if mask is not None:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rendered, contours, -1, (255, 220, 0), 1)
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{index}: {detection.label} {detection.confidence:.2f}"
        cv2.putText(rendered, label, (max(0, x1), max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        ellipse = getattr(detection, "rim_ellipse", None)
        estimate = None
        if depth_mm is not None and intrinsics is not None:
            estimate = localize_rim(detection, rgb, depth_mm, intrinsics, config=detector.config)
        entry: dict[str, Any] = {
            "index": index,
            "bbox": [x1, y1, x2, y2],
            "confidence": float(detection.confidence),
            "label": detection.label,
            "rim_ellipse": list(ellipse) if ellipse is not None else None,
            "rim_center_camera_mm": None,
        }
        if estimate is not None:
            if ellipse is not None:
                cx, cy, axis_a, axis_b, angle = ellipse
                cv2.ellipse(rendered, ((float(cx), float(cy)), (float(axis_a), float(axis_b)), float(angle)),
                            (0, 255, 255), 2)
            entry.update(rim_center_camera_mm=[float(v) for v in estimate.center_mm],
                         rim_normal_camera=[float(v) for v in estimate.normal],
                         rim_support=int(estimate.support),
                         rim_residual_mm=float(estimate.residual_mm))
            position = ", ".join(f"{v:.0f}" for v in estimate.center_mm)
            cv2.putText(rendered, f"rim camera mm: {position}", (max(0, x1), min(rendered.shape[0] - 5, y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        results.append(entry)
    return rendered, results


async def camera_loop(args: argparse.Namespace, *, preview: bool) -> None:
    from google.protobuf.json_format import MessageToDict
    from viam.components.camera import Camera

    from connection import connect

    detector = make_detector(args) if preview or args.detect else None
    output = args.output.resolve() if args.output else None
    if output:
        output.mkdir(parents=True, exist_ok=True)
    robot = await connect()
    frames: list[str] = []
    try:
        camera = Camera.from_robot(robot, args.camera)
        properties = await camera.get_properties(timeout=20)
        k = properties.intrinsic_parameters
        intrinsics = {"fx": k.focal_x_px, "fy": k.focal_y_px,
                      "cx": k.center_x_px, "cy": k.center_y_px}
        if not np.isfinite(list(intrinsics.values())).all() or min(intrinsics["fx"], intrinsics["fy"]) <= 0:
            raise ValueError("Missing or invalid camera intrinsics")
        index = 0
        previous_timestamp: str | None = None
        while preview or index < args.frames:
            images, metadata = await camera.get_images(timeout=30)
            timestamp = metadata.captured_at.ToJsonString()
            if timestamp == previous_timestamp:
                await asyncio.sleep(args.interval)
                continue
            previous_timestamp = timestamp
            color, depth = decode_pair(images, (k.height_px, k.width_px))
            rendered, results = color, None
            if detector is not None:
                detections = await asyncio.to_thread(detector.detect, cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
                rendered, results = await asyncio.to_thread(annotate, color, detections, depth, intrinsics, detector)
            if output:
                stem = f"frame_{index:06d}"
                write_image(output / f"{stem}_color.png", color)
                np.save(output / f"{stem}_depth_mm.npy", depth)
                record: dict[str, Any] = {
                    "schema_version": 1,
                    "camera_name": args.camera,
                    "captured_at": timestamp,
                    "color_file": f"{stem}_color.png",
                    "depth_file": f"{stem}_depth_mm.npy",
                    "intrinsics": intrinsics,
                    "properties": MessageToDict(properties, preserving_proto_field_name=True),
                    "valid_depth_fraction": float(np.mean(depth > 0)),
                }
                if results is not None:
                    record["detections"] = results
                    write_image(output / f"{stem}_annotated.png", rendered)
                write_json(output / f"{stem}.json", record)
                frames.append(f"{stem}.json")
            LOGGER.info("Frame %d at %s: depth %.1f%% valid%s", index, timestamp,
                        100 * np.mean(depth > 0), f", {len(results)} cups" if results is not None else "")
            index += 1
            if preview:
                cv2.imshow("YOLO cup detection - q to quit", rendered)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            await asyncio.sleep(args.interval)
    finally:
        if output:
            write_json(output / "capture.json", {"schema_version": 1, "frames": frames})
        if preview:
            cv2.destroyAllWindows()
        await robot.close()


def offline_sources(paths: Sequence[Path]) -> Iterable[tuple[Path, Path | None]]:
    """Prefer paired records over image globs, avoiding duplicate annotated images."""
    for path in paths:
        if path.is_dir():
            records = sorted(path.glob("frame_[0-9]*.json"))
            if records:
                for record in records:
                    data = json.loads(record.read_text(encoding="utf-8"))
                    yield record.parent / data["color_file"], record
            else:
                for image in sorted(path.iterdir()):
                    if image.suffix.lower() in IMAGE_SUFFIXES and "annotated" not in image.stem:
                        yield image, None
        elif path.suffix.lower() in IMAGE_SUFFIXES:
            yield path, None
        else:
            raise ValueError(f"Expected an image or a capture directory, got {path}")


def offline(args: argparse.Namespace) -> None:
    detector = make_detector(args)
    args.output.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    for index, (path, record_path) in enumerate(offline_sources(args.input)):
        color = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if color is None:
            raise ValueError(f"Cannot read {path}")
        depth, intrinsics = None, None
        if record_path:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            depth = np.load(record_path.parent / record["depth_file"], allow_pickle=False)
            intrinsics = record["intrinsics"]
            if depth.shape != color.shape[:2]:
                raise ValueError(f"Saved RGB/depth dimensions differ for {path}")
        detections = detector.detect(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
        rendered, results = annotate(color, detections, depth, intrinsics, detector)
        output_name = f"{index:06d}_{path.stem}_annotated.png"
        write_image(args.output / output_name, rendered)
        summary.append({"input": str(path.resolve()), "annotation": output_name, "detections": results})
        LOGGER.info("%s: %d cups", path.name, len(results))
    if not summary:
        raise ValueError("No source images found")
    write_json(args.output / "detections.json", summary)


async def locate(args: argparse.Namespace) -> None:
    from connection import connect

    det = load_detector(getattr(args, "config", None), getattr(args, "model", None))
    robot = await connect()
    try:
        cups = await find_cups(robot, camera=getattr(args, "camera", "cam"), det=det)
        if not cups:
            print("no cup in the current view")
        for i, c in enumerate(cups):
            x, y, z = c.center_mm
            print(f"cup {i}: rim center ({x:.0f}, {y:.0f}, {z:.0f}) mm world, confidence {c.confidence:.2f}, "
                  f"method {c.method}, box {c.bbox}")
    finally:
        await robot.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode")
    locate = modes.add_parser("locate", help="Print cup rim centers (world mm) from the current view (default)")
    locate.add_argument("--config", type=Path, help="JSON object of detector attributes")
    locate.add_argument("--model", help="YOLO weights path; overrides model_path in --config")
    locate.add_argument("--camera", default="cam")
    for name, help_text in (("capture", "Save N synchronized RGB-D pairs; optionally annotate"),
                            ("offline", "Replay saved pairs or photographs"),
                            ("live", "Show detection boxes in a local OpenCV window")):
        mode = modes.add_parser(name, help=help_text)
        mode.add_argument("--config", type=Path, help="JSON object of detector attributes")
        mode.add_argument("--model", help="YOLO weights path; overrides model_path in --config")
        mode.add_argument("--output", type=Path, required=name != "live")
        if name == "offline":
            mode.add_argument("--input", type=Path, nargs="+", required=True)
        else:
            mode.add_argument("--camera", default="cam")
            mode.add_argument("--interval", type=float, default=0.1, help="Delay between captures in seconds")
            mode.add_argument("--frames", type=int, default=30)
            mode.add_argument("--detect", action="store_true", help="Also run detection during capture")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "frames") and args.frames < 1:
        parser.error("--frames must be positive")
    if hasattr(args, "interval") and args.interval < 0:
        parser.error("--interval cannot be negative")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args.mode = args.mode or "locate"
    try:
        if args.mode == "locate":
            asyncio.run(locate(args))
        elif args.mode == "offline":
            offline(args)
        else:
            asyncio.run(camera_loop(args, preview=args.mode == "live"))
    except KeyboardInterrupt:
        LOGGER.info("Stopped")
    except Exception:
        LOGGER.exception("Cup detection failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
