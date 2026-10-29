"""Capture paired RGB-D, replay detections offline, or preview cups. Never moves hardware.

Run from the project root with ``python -m scripts.cup_test --help``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_image(path: Path, image_bgr: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image_bgr):
        raise OSError(f"Cannot write {path}")


def make_detector(args: argparse.Namespace) -> Any:
    from cup_detection.core import CupDetector, DetectorConfig

    attributes: dict[str, Any] = {}
    if args.config:
        attributes = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(attributes, dict):
            raise ValueError("--config must contain a JSON object of DetectorConfig attributes")
    if args.model:
        attributes["model_path"] = args.model
    return CupDetector(DetectorConfig(**attributes))


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
    from cup_detection.core import localize_rim

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
                cv2.imshow("Red Solo cup detector — q to quit", rendered)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
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
    try:
        if args.mode == "offline":
            offline(args)
        else:
            asyncio.run(camera_loop(args, preview=args.mode == "live"))
    except KeyboardInterrupt:
        LOGGER.info("Stopped")
    except Exception:
        LOGGER.exception("Cup test failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
