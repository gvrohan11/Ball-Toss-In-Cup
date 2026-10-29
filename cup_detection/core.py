"""Learned cup instance segmentation and conservative optional rim geometry.

Inputs are RGB uint8 images and aligned depth in millimeters. The pretrained
COCO ``cup`` class supplies object identity without requiring visible red pixels.
Classical rim fitting estimates geometry only. A hidden red exterior cannot
establish Solo identity; this model recognizes general cup candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import logging
from typing import Any, Mapping

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)
Ellipse = tuple[float, float, float, float, float]


@dataclass(frozen=True)
class DetectorConfig:
    model_path: str = ".cup-models/yolo11m-seg.pt"
    semantic_enabled: bool = True
    semantic_confidence: float = 0.12
    image_size: int = 1280
    device: str = "cpu"
    rotation_search: bool = True
    red_hue_low_max: int = 12
    red_hue_high_min: int = 168
    red_saturation_min: float = 0.40
    red_value_min: float = 0.15
    red_fraction_min: float = 0.06
    white_saturation_max: float = 0.28
    white_value_min: float = 0.48
    pink_saturation_min: float = 0.08
    pink_hue_min: int = 150
    pink_hue_max: int = 179
    pink_saturation_max: float = 0.55
    pink_value_min: float = 0.40
    pink_fraction_min: float = 0.15
    rim_white_fraction_min: float = 0.55
    rim_interior_pale_min: float = 0.50
    rim_interior_red_max: float = 0.20
    ring_min_area_px: int = 200
    ring_image_size: int = 1024
    ring_axis_ratio_min: float = 0.08
    top_axis_ratio_min: float = 0.45
    ellipse_error_max: float = 0.12
    rim_band_fraction: float = 0.08
    morphology_size: int = 3
    nms_iou: float = 0.45
    top_view_enabled: bool = False
    top_view_confidence: float = 0.60
    cup_height_mm: float | None = None
    rim_diameter_mm: float | None = None
    size_tolerance_fraction: float = 0.25
    depth_min_mm: float = 100
    depth_max_mm: float = 3000
    rim_min_points: int = 40
    rim_plane_threshold_mm: float = 4
    rim_plane_inlier_fraction_min: float = 0.65
    rim_circle_residual_max_mm: float = 4
    rim_coverage_degrees_min: float = 240
    rim_coverage_with_size_degrees_min: float = 180
    rim_ransac_iterations: int = 100
    support_region_expansion: float = 0.35
    support_plane_threshold_mm: float = 5
    support_plane_min_points: int = 100
    support_plane_normal_tolerance_degrees: float = 15
    inferred_height_min_mm: float = 10
    inferred_height_max_mm: float = 400

    def validate(self) -> None:
        """Reject malformed thresholds rather than silently clamping config."""
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not np.isfinite(value):
                raise ValueError(f"{field.name} must be finite")
        fractions = ("semantic_confidence", "red_saturation_min", "red_value_min",
                     "red_fraction_min", "white_saturation_max", "white_value_min",
                     "pink_saturation_min", "pink_saturation_max", "pink_value_min",
                     "pink_fraction_min", "rim_white_fraction_min", "ring_axis_ratio_min",
                     "top_axis_ratio_min", "ellipse_error_max", "rim_band_fraction",
                     "nms_iou", "top_view_confidence", "size_tolerance_fraction",
                     "rim_plane_inlier_fraction_min", "rim_interior_pale_min", "rim_interior_red_max")
        for name in fractions:
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and between 0 and 1")
        for name in ("image_size", "ring_image_size", "ring_min_area_px", "morphology_size", "rim_min_points", "rim_ransac_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.morphology_size % 2 != 1:
            raise ValueError("morphology_size must be odd")
        if self.rim_min_points < 6:
            raise ValueError("rim_min_points must be at least six")
        for name in ("semantic_enabled", "rotation_search", "top_view_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if not isinstance(self.model_path, str) or not self.model_path:
            raise ValueError("model_path must be a nonempty path or Ultralytics checkpoint name")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a nonempty string")
        if not 0 <= self.red_hue_low_max < self.red_hue_high_min <= 179:
            raise ValueError("red hue ranges must wrap around OpenCV's 0..179 hue axis")
        if not 0 <= self.pink_hue_min < self.pink_hue_max <= 179:
            raise ValueError("pink hue bounds must be ordered within 0..179")
        if self.pink_saturation_min >= self.pink_saturation_max:
            raise ValueError("pink_saturation_min must be below pink_saturation_max")
        for name in ("cup_height_mm", "rim_diameter_mm"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a positive measured value or null")
        for name in ("depth_min_mm", "depth_max_mm", "rim_plane_threshold_mm", "rim_circle_residual_max_mm"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.depth_min_mm >= self.depth_max_mm:
            raise ValueError("depth_min_mm must be below depth_max_mm")
        for name in ("rim_coverage_degrees_min", "rim_coverage_with_size_degrees_min"):
            if not 0 < getattr(self, name) <= 360:
                raise ValueError(f"{name} must be in (0,360]")
        if not 0 < self.support_region_expansion <= 2:
            raise ValueError("support_region_expansion must be in (0,2]")
        if self.support_plane_threshold_mm <= 0 or self.support_plane_min_points < 3:
            raise ValueError("support plane needs a positive threshold and at least three points")
        if not 0 < self.support_plane_normal_tolerance_degrees < 90:
            raise ValueError("support_plane_normal_tolerance_degrees must be in (0,90)")
        if not 0 < self.inferred_height_min_mm < self.inferred_height_max_mm:
            raise ValueError("inferred height bounds must be positive and ordered")


@dataclass
class CupDetection:
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str = "red_solo_cup"
    rim_ellipse: Ellipse | None = None
    source: str = "semantic"
    object_mask: np.ndarray | None = None
    rim_failure_reason: str | None = None


@dataclass(frozen=True)
class RimEstimate:
    center_mm: tuple[float, float, float]
    normal: tuple[float, float, float]
    support: int
    residual_mm: float
    diameter_mm: float
    height_mm: float | None = None


def _masks(rgb: np.ndarray, config: DetectorConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    red_hue = (hue <= config.red_hue_low_max) | (hue >= config.red_hue_high_min)
    red = red_hue & (saturation >= 255 * config.red_saturation_min) & (value >= 255 * config.red_value_min)
    white = (saturation <= 255 * config.white_saturation_max) & (value >= 255 * config.white_value_min)
    pink_hue = (hue >= config.pink_hue_min) & (hue <= config.pink_hue_max)
    pink = pink_hue & (saturation >= 255 * config.pink_saturation_min) & (saturation <= 255 * config.pink_saturation_max) & (value >= 255 * config.pink_value_min)
    return red, white, pink


def _ellipse_coordinates(ellipse: Ellipse, shape: tuple[int, int]) -> np.ndarray:
    cx, cy, a, b, angle = ellipse
    ys, xs = np.indices(shape)
    theta = np.deg2rad(angle)
    xx, yy = xs - cx, ys - cy
    u = xx * np.cos(theta) + yy * np.sin(theta)
    v = -xx * np.sin(theta) + yy * np.cos(theta)
    return np.sqrt((u / (a / 2)) ** 2 + (v / (b / 2)) ** 2)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    w, h = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w * h
    return intersection / max(1, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection)


class CupDetector:
    """Pretrained semantic proposals with transparent appearance specialization."""

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self.config.validate()
        self._model: Any = None
        self._proposal_masks: dict[tuple[int, int, int, int], np.ndarray] = {}

    def _semantic_boxes(self, rgb: np.ndarray) -> list[tuple[tuple[int, int, int, int], float]]:
        if not self.config.semantic_enabled:
            return []
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Install cup_detection/requirements.txt before semantic cup detection") from exc
            LOGGER.info("Loading semantic cup detector %s", self.config.model_path)
            self._model = YOLO(self.config.model_path)
        names = self._model.names
        names_items = names.items() if isinstance(names, dict) else enumerate(names)
        class_ids = [int(i) for i, name in names_items if name in ("cup", "red_solo_cup")]
        if not class_ids:
            raise ValueError("Model has neither a cup nor red_solo_cup class")
        proposals: list[tuple[tuple[int, int, int, int], float]] = []
        self._proposal_masks = {}
        h, w = rgb.shape[:2]
        # Optional quarter-turn search makes rotated side views explicit, at 4x inference cost.
        for turns in (range(4) if self.config.rotation_search else (0,)):
            rotated = np.ascontiguousarray(np.rot90(rgb, turns))
            results = self._model.predict(source=cv2.cvtColor(rotated, cv2.COLOR_RGB2BGR),
                                          conf=self.config.semantic_confidence, classes=class_ids,
                                          imgsz=self.config.image_size, device=self.config.device, verbose=False)
            for result in results:
                if result.boxes is None:
                    continue
                for index, (box, confidence) in enumerate(zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())):
                    x1, y1, x2, y2 = map(float, box)
                    if turns == 1:
                        x1, y1, x2, y2 = w - y2, x1, w - y1, x2
                    elif turns == 2:
                        x1, y1, x2, y2 = w - x2, h - y2, w - x1, h - y1
                    elif turns == 3:
                        x1, y1, x2, y2 = y1, h - x2, y2, h - x1
                    bbox = (max(0, int(x1)), max(0, int(y1)), min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2))))
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        proposals.append((bbox, float(confidence)))
                        if result.masks is not None and index < len(result.masks.xy):
                            mask = np.zeros(rotated.shape[:2], np.uint8)
                            polygon = result.masks.xy[index].astype(np.int32)
                            if len(polygon) >= 3:
                                cv2.fillPoly(mask, [polygon], 1)
                                self._proposal_masks[bbox] = np.ascontiguousarray(np.rot90(mask, -turns)).astype(bool)
        return proposals

    def _rings(self, rgb: np.ndarray) -> list[tuple[Ellipse, float, float]]:
        scale = min(1.0, self.config.ring_image_size / max(rgb.shape[:2]))
        if scale < 1:
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        red, white, pink = _masks(rgb, self.config)
        kernel = np.ones((self.config.morphology_size,) * 2, np.uint8)
        mask = cv2.morphologyEx(white.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        rings = []
        for contour in contours:
            if len(contour) < 20 or cv2.contourArea(contour) < self.config.ring_min_area_px:
                continue
            (cx, cy), (a, b), angle = cv2.fitEllipse(contour)
            if min(a, b) / max(a, b) < self.config.ring_axis_ratio_min:
                continue
            ellipse = (float(cx), float(cy), float(a), float(b), float(angle))
            points = contour[:, 0, :].astype(float) - (cx, cy)
            theta = np.deg2rad(angle)
            u = points[:, 0] * np.cos(theta) + points[:, 1] * np.sin(theta)
            v = -points[:, 0] * np.sin(theta) + points[:, 1] * np.cos(theta)
            error = np.median(np.abs(np.sqrt((u / (a / 2)) ** 2 + (v / (b / 2)) ** 2) - 1))
            if error > self.config.ellipse_error_max:
                continue
            radial = _ellipse_coordinates(ellipse, white.shape)
            band = (radial > 1 - self.config.rim_band_fraction) & (radial < 1 + self.config.rim_band_fraction)
            interior = radial < 0.78
            if not band.any() or not interior.any():
                continue
            inside_band = band & (radial <= 1)
            outside_band = band & (radial >= 1)
            white_fraction = max(float(white[side].mean()) for side in (inside_band, outside_band) if side.any())
            pink_fraction = float(pink[interior].mean())
            # Ring contours have their white material predominantly to one side.
            # A rim surrounds the pale opening, not the glossy red sidewall.
            # This specifically rejects ellipses fitted around white reflections.
            pale_fraction = float(white[interior].mean())
            red_fraction = float(red[interior].mean())
            if (white_fraction >= self.config.rim_white_fraction_min
                    and pale_fraction >= self.config.rim_interior_pale_min
                    and red_fraction <= self.config.rim_interior_red_max):
                rings.append(((ellipse[0] / scale, ellipse[1] / scale, ellipse[2] / scale,
                               ellipse[3] / scale, ellipse[4]), white_fraction, pink_fraction))
        return rings

    def detect(self, rgb: np.ndarray) -> list[CupDetection]:
        """Return all matching cups anywhere in the frame, with half-open boxes."""
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must be an HxWx3 RGB uint8 image")
        rings = self._rings(rgb) if self.config.top_view_enabled else []
        detections: list[CupDetection] = []
        for bbox, confidence in self._semantic_boxes(rgb):
            x1, y1, x2, y2 = bbox
            local_rings = self._rings(rgb[y1:y2, x1:x2])
            local_rings = [((e[0]+x1,e[1]+y1,e[2],e[3],e[4]), wf, pf) for e,wf,pf in local_rings]
            matching = [(e, wf, pf) for e, wf, pf in local_rings if x1 <= e[0] <= x2 and y1 <= e[1] <= y2
                        and 0.35 * min(x2 - x1, y2 - y1) <= max(e[2:4]) < 1.25 * max(x2 - x1, y2 - y1)]
            # Trust the learned cup class even when the red exterior is hidden.
            # Reject fits extrapolating outside the actual object box. A partial
            # arc can otherwise yield a convincing but oversized false circle.
            contained = []
            for candidate in matching:
                e = candidate[0]
                theta = np.deg2rad(e[4])
                rx = np.hypot(e[2]*np.cos(theta)/2, e[3]*np.sin(theta)/2)
                ry = np.hypot(e[2]*np.sin(theta)/2, e[3]*np.cos(theta)/2)
                margin = .05 * min(x2-x1, y2-y1)
                if e[0]-rx >= x1-margin and e[0]+rx <= x2+margin and e[1]-ry >= y1-margin and e[1]+ry <= y2+margin:
                    contained.append(candidate)
            rim = max(contained, key=lambda item: item[0][2] * item[0][3])[0] if contained else None
            detections.append(CupDetection(bbox, confidence, rim_ellipse=rim,
                                           object_mask=self._proposal_masks.get(bbox)))
        if self.config.top_view_enabled:
            h, w = rgb.shape[:2]
            for ellipse, white_fraction, pink_fraction in rings:
                cx, cy, a, b, _ = ellipse
                if min(a, b) / max(a, b) < self.config.top_axis_ratio_min or pink_fraction < self.config.pink_fraction_min:
                    continue
                theta = np.deg2rad(ellipse[4])
                rx = np.sqrt((a * np.cos(theta) / 2) ** 2 + (b * np.sin(theta) / 2) ** 2)
                ry = np.sqrt((a * np.sin(theta) / 2) ** 2 + (b * np.cos(theta) / 2) ** 2)
                # A clipped opening has an ambiguous center: leave it to semantic proposals.
                if cx - rx < 0 or cy - ry < 0 or cx + rx >= w or cy + ry >= h:
                    continue
                bbox = (int(cx - rx), int(cy - ry), int(np.ceil(cx + rx)), int(np.ceil(cy + ry)))
                confidence = min(self.config.top_view_confidence, 0.4 + 0.2 * min(white_fraction, pink_fraction))
                detections.append(CupDetection(bbox, confidence, rim_ellipse=ellipse, source="rim_geometry"))
        kept: list[CupDetection] = []
        # An embossed inner bottom circle is not another cup inside the same rim.
        detections = [d for d in detections if d.source != "rim_geometry" or not any(
            other.source == "rim_geometry" and other is not d
            and other.bbox[0] <= d.bbox[0] and other.bbox[1] <= d.bbox[1]
            and other.bbox[2] >= d.bbox[2] and other.bbox[3] >= d.bbox[3]
            and (other.bbox[2] - other.bbox[0]) * (other.bbox[3] - other.bbox[1]) >
            1.5 * (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]) for other in detections)]
        for detection in sorted(detections, key=lambda d: d.confidence, reverse=True):
            if all(_iou(detection.bbox, other.bbox) < self.config.nms_iou for other in kept):
                kept.append(detection)
        return kept

    def localize_rim(self, detection: CupDetection, rgb: np.ndarray, depth_mm: np.ndarray,
                     intrinsics: Mapping[str, float]) -> RimEstimate | None:
        return localize_rim(detection, rgb, depth_mm, intrinsics, self.config)


def localize_rim(detection: CupDetection, rgb: np.ndarray, depth_mm: np.ndarray,
                 intrinsics: Mapping[str, float], config: DetectorConfig | None = None) -> RimEstimate | None:
    """Fit a 3D circle to white rim samples, never to pixels inside the opening.

    Returns None for missing rim, inadequate depth/arc coverage, nonplanar samples,
    or implausible measured diameter. A semantic 2D box can remain valid then.
    """
    cfg = config or DetectorConfig()
    cfg.validate()
    def reject(reason: str) -> None:
        detection.rim_failure_reason = reason
        return None
    if detection.rim_ellipse is None:
        return reject("no validated 2D rim ellipse")
    if depth_mm.shape != rgb.shape[:2]:
        raise ValueError("Depth must be aligned to RGB with the same shape")
    fx, fy, cx, cy = (float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy"))
    if not np.isfinite((fx, fy, cx, cy)).all() or min(fx, fy) <= 0:
        raise ValueError("Intrinsics must be finite with positive focal lengths")
    _, white, _ = _masks(rgb, cfg)
    radial = _ellipse_coordinates(detection.rim_ellipse, depth_mm.shape)
    ring = (np.abs(radial - 1) <= cfg.rim_band_fraction) & white
    valid = ring & np.isfinite(depth_mm) & (depth_mm >= cfg.depth_min_mm) & (depth_mm <= cfg.depth_max_mm)
    ys, xs = np.nonzero(valid)
    if len(xs) < cfg.rim_min_points:
        return reject(f"insufficient rim depth: {len(xs)} points")
    # Bound fitting cost while retaining samples over the complete ring.
    if len(xs) > 3000:
        select = np.linspace(0, len(xs) - 1, 3000).astype(int)
        ys, xs = ys[select], xs[select]
    z = depth_mm[ys, xs].astype(float)
    points = np.column_stack(((xs - cx) * z / fx, (ys - cy) * z / fy, z))
    rng = np.random.default_rng(0)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(cfg.rim_ransac_iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        inliers = np.abs((points - sample[0]) @ normal) <= cfg.rim_plane_threshold_mm
        if inliers.sum() > best.sum():
            best = inliers
    if best.sum() < cfg.rim_min_points or best.mean() < cfg.rim_plane_inlier_fraction_min:
        return reject(f"rim depth not planar: {best.mean():.0%} inliers")
    points = points[best]
    mean = points.mean(axis=0)
    _, _, basis = np.linalg.svd(points - mean, full_matrices=False)
    normal = basis[2]
    plane_residual = float(np.sqrt(np.mean(((points - mean) @ normal) ** 2)))
    if plane_residual > cfg.rim_plane_threshold_mm:
        return reject(f"rim plane residual {plane_residual:.1f}mm")
    uv = (points - mean) @ basis[:2].T
    design = np.column_stack((2 * uv[:, 0], 2 * uv[:, 1], np.ones(len(uv))))
    if np.linalg.cond(design) > 1e5:
        return None
    solution, *_ = np.linalg.lstsq(design, np.sum(uv * uv, axis=1), rcond=None)
    center_uv = solution[:2]
    radius_sq = float(solution[2] + center_uv @ center_uv)
    if radius_sq <= 0:
        return None
    radius = np.sqrt(radius_sq)
    radial_residual = np.abs(np.linalg.norm(uv - center_uv, axis=1) - radius)
    residual = float(np.sqrt(np.mean(radial_residual ** 2)))
    if residual > cfg.rim_circle_residual_max_mm:
        return reject(f"rim circle residual {residual:.1f}mm")
    angles = np.sort(np.mod(np.arctan2(uv[:, 1] - center_uv[1], uv[:, 0] - center_uv[0]), 2 * np.pi))
    coverage = np.rad2deg(2 * np.pi - np.diff(np.r_[angles, angles[0] + 2 * np.pi]).max())
    required_coverage = cfg.rim_coverage_with_size_degrees_min if cfg.rim_diameter_mm else cfg.rim_coverage_degrees_min
    if coverage < required_coverage:
        return reject(f"rim coverage {coverage:.0f} degrees, need {required_coverage:.0f}")
    diameter = 2 * float(radius)
    if cfg.rim_diameter_mm and abs(diameter / cfg.rim_diameter_mm - 1) > cfg.size_tolerance_fraction:
        return None
    center = mean + center_uv @ basis[:2]
    if np.dot(normal, center) > 0:
        normal = -normal
    height = _support_height(detection, depth_mm, intrinsics, center, normal, cfg)
    return RimEstimate(tuple(map(float, center)), tuple(map(float, normal)), len(points), max(plane_residual, residual), diameter, height)


def _support_height(detection: CupDetection, depth_mm: np.ndarray, intrinsics: Mapping[str, float],
                    center: np.ndarray, normal: np.ndarray, cfg: DetectorConfig) -> float | None:
    """Infer upright cup height only when a nearby parallel support plane is visible."""
    x1, y1, x2, y2 = detection.bbox
    h, w = depth_mm.shape
    dx = int((x2 - x1) * cfg.support_region_expansion)
    dy = int((y2 - y1) * cfg.support_region_expansion)
    mask = np.zeros(depth_mm.shape, bool)
    mask[max(0, y1 - dy):min(h, y2 + dy), max(0, x1 - dx):min(w, x2 + dx)] = True
    mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = False
    mask &= np.isfinite(depth_mm) & (depth_mm >= cfg.depth_min_mm) & (depth_mm <= cfg.depth_max_mm)
    ys, xs = np.nonzero(mask)
    if len(xs) < cfg.support_plane_min_points:
        return None
    if len(xs) > 3000:
        select = np.linspace(0, len(xs) - 1, 3000).astype(int)
        ys, xs = ys[select], xs[select]
    z = depth_mm[ys, xs].astype(float)
    points = np.column_stack(((xs - intrinsics["cx"]) * z / intrinsics["fx"],
                              (ys - intrinsics["cy"]) * z / intrinsics["fy"], z))
    heights = -(points - center) @ normal
    eligible = (heights >= cfg.inferred_height_min_mm) & (heights <= cfg.inferred_height_max_mm)
    if eligible.sum() < cfg.support_plane_min_points:
        return None
    # Find the dominant parallel plane by its signed offset from the rim plane.
    bins = np.floor(heights[eligible] / cfg.support_plane_threshold_mm).astype(int)
    mode = np.bincount(bins).argmax()
    offset = (mode + .5) * cfg.support_plane_threshold_mm
    inliers = np.abs(heights - offset) <= cfg.support_plane_threshold_mm
    if inliers.sum() < cfg.support_plane_min_points:
        return None
    support = points[inliers]
    _, _, basis = np.linalg.svd(support - support.mean(axis=0), full_matrices=False)
    agreement = abs(float(basis[2] @ normal))
    if agreement < np.cos(np.deg2rad(cfg.support_plane_normal_tolerance_degrees)):
        return None
    height = float(np.median(heights[inliers]))
    if cfg.cup_height_mm and abs(height / cfg.cup_height_mm - 1) > cfg.size_tolerance_fraction:
        return None
    return height
