"""Viam adapter for the cup detector; no arm or motion dependencies."""
from __future__ import annotations

import asyncio
import dataclasses
from typing import Any, Mapping, Optional, Sequence, cast

import cv2
import numpy as np
from viam.components.camera import Camera
from viam.errors import NotSupportedError
from viam.media.video import CameraMimeType, ViamImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import GeometriesInFrame, Geometry, PointCloudObject, Pose, RectangularPrism, ResourceName, Vector3
from viam.proto.service.vision import Classification, Detection, Detection3D
from viam.resource.base import ResourceBase
from viam.resource.easy_resource import EasyResource
from viam.services.vision import Vision
from viam.services.vision.vision import CaptureAllResult
from viam.utils import ValueTypes, struct_to_dict

from .core import CupDetection, CupDetector, DetectorConfig


def decode_rgb(image: ViamImage) -> np.ndarray:
    """Decode a standard camera JPEG/PNG to RGB for the model."""
    if image.mime_type not in (CameraMimeType.JPEG, CameraMimeType.PNG):
        raise ValueError(f"Cup detector expects JPEG/PNG color, got {image.mime_type}")
    bgr = cv2.imdecode(np.frombuffer(image.data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode camera image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def as_detection(item: CupDetection) -> Detection:
    x0, y0, x1, y1 = item.bbox
    return Detection(x_min=x0, y_min=y0, x_max=x1, y_max=y1,
                     confidence=float(item.confidence), class_name=item.label)


def parse_config(config: ComponentConfig) -> tuple[str, DetectorConfig, int, float]:
    attrs = dict(struct_to_dict(config.attributes))
    camera = attrs.pop("camera_name", "cam")
    if not isinstance(camera, str) or not camera.strip():
        raise ValueError("camera_name must be a nonempty string")
    limit = attrs.pop("max_pointcloud_points", 12000)
    band = attrs.pop("pointcloud_depth_band_mm", 150.0)
    if isinstance(limit, bool) or not isinstance(limit, (int, float)) or int(limit) != limit or not 20 <= limit <= 100000:
        raise ValueError("max_pointcloud_points must be an integer from 20 through 100000")
    if isinstance(band, bool) or not isinstance(band, (int, float)) or not np.isfinite(band) or band <= 0:
        raise ValueError("pointcloud_depth_band_mm must be finite and positive")
    fields = {field.name: field for field in dataclasses.fields(DetectorConfig)}
    unknown = set(attrs) - fields.keys()
    if unknown:
        raise ValueError(f"Unknown detector attributes: {sorted(unknown)}")
    # Protobuf Struct decodes all JSON numbers as float; preserve integer thresholds.
    defaults = DetectorConfig()
    for name, value in attrs.items():
        default = getattr(defaults, name)
        if isinstance(default, bool) and not isinstance(value, bool):
            raise ValueError(f"{name} must be boolean")
        if isinstance(default, int) and not isinstance(default, bool):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or int(value) != value:
                raise ValueError(f"{name} must be an integer")
            attrs[name] = int(value)
    detector_config = DetectorConfig(**attrs)
    detector_config.validate()
    return camera, detector_config, int(limit), float(band)


class RedSoloCupVision(Vision, EasyResource):
    MODEL = "rohan:cup-detection:red-solo-cup"

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._lock = asyncio.Lock()

    @classmethod
    def validate_config(cls, config: ComponentConfig) -> tuple[Sequence[str], Sequence[str]]:
        camera, _, _, _ = parse_config(config)
        return [camera], []

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> RedSoloCupVision:
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> None:
        camera_name, detector_config, limit, band = parse_config(config)
        camera_key = Camera.get_resource_name(camera_name)
        if camera_key not in dependencies:
            raise ValueError(f"Missing configured camera dependency {camera_name!r}")
        self.camera_name = camera_name
        self.camera = cast(Camera, dependencies[camera_key])
        self.detector = CupDetector(detector_config)
        self.max_pointcloud_points = limit
        self.pointcloud_depth_band_mm = band
        self.logger.info("Configured red-solo-cup using camera %s", camera_name)

    def _check_camera(self, name: str) -> None:
        if name and name != self.camera_name:
            raise ValueError(f"This service is calibrated for {self.camera_name!r}, not {name!r}")

    async def _capture(self, camera_name: str, timeout: Optional[float]) -> tuple[ViamImage, np.ndarray, np.ndarray, dict[str, float], str, str]:
        self._check_camera(camera_name)
        images, metadata = await self.camera.get_images(timeout=timeout)
        sources = {image.name: image for image in images}
        if not {"color", "depth"}.issubset(sources):
            raise ValueError("RGB-D cup detection requires color and depth sources from GetImages")
        rgb = decode_rgb(sources["color"])
        if sources["depth"].mime_type != CameraMimeType.VIAM_RAW_DEPTH:
            raise ValueError("Expected image/vnd.viam.dep depth; configure little_endian_depth=false")
        depth = np.asarray(sources["depth"].bytes_to_depth_array(), dtype=np.uint16)
        props = await self.camera.get_properties(timeout=timeout)
        intrinsics = props.intrinsic_parameters
        if depth.shape != rgb.shape[:2] or depth.shape != (intrinsics.height_px, intrinsics.width_px):
            raise ValueError("RGB, aligned depth and intrinsics dimensions must match")
        if min(intrinsics.focal_x_px, intrinsics.focal_y_px) <= 0:
            raise ValueError("Camera does not report valid focal lengths")
        if any(abs(value) > 1e-9 for value in props.distortion_parameters.parameters):
            raise ValueError("Nonzero distortion requires rectification before pinhole rim localization")
        k = {"fx": intrinsics.focal_x_px, "fy": intrinsics.focal_y_px,
             "cx": intrinsics.center_x_px, "cy": intrinsics.center_y_px}
        return sources["color"], rgb, depth, k, metadata.captured_at.ToJsonString(), props.default_reference_frame or self.camera_name

    def _clouds(self, detections: list[CupDetection], depth: np.ndarray, k: dict[str, float], frame: str) -> list[PointCloudObject]:
        clouds: list[PointCloudObject] = []
        for detection in detections:
            x0, y0, x1, y1 = detection.bbox
            yy, xx = np.mgrid[max(0, y0):min(depth.shape[0], y1), max(0, x0):min(depth.shape[1], x1)]
            z = depth[yy, xx].astype(float)
            valid = z > 0
            if detection.object_mask is not None:
                valid &= detection.object_mask[yy, xx]
            if not valid.any():
                continue
            # The learned instance mask excludes pixels belonging to neighbors.
            median = float(np.median(z[valid]))
            valid &= abs(z - median) <= self.pointcloud_depth_band_mm
            if int(valid.sum()) < 20:
                continue
            points = np.column_stack(((xx[valid] - k["cx"]) * z[valid] / k["fx"],
                                      (yy[valid] - k["cy"]) * z[valid] / k["fy"], z[valid]))
            stride = max(1, int(np.ceil(len(points) / self.max_pointcloud_points)))
            points = points[::stride]
            header = ("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                      f"WIDTH {len(points)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA binary\n")
            # Viam's PCD wire format is meters; geometry/Pose protobufs stay mm.
            pcd = header.encode("ascii") + (points / 1000.0).astype("<f4").tobytes()
            lower, upper = points.min(axis=0), points.max(axis=0)
            center, size = (lower + upper) / 2, np.maximum(upper - lower, 1.0)
            geometry = Geometry(center=Pose(x=center[0], y=center[1], z=center[2], o_z=1),
                                box=RectangularPrism(dims_mm=Vector3(x=size[0], y=size[1], z=size[2])),
                                label=detection.label)
            clouds.append(PointCloudObject(point_cloud=pcd, geometries=GeometriesInFrame(reference_frame=frame, geometries=[geometry])))
        return clouds

    async def get_detections(self, image: ViamImage, *, extra: Optional[Mapping[str, ValueTypes]] = None,
                             timeout: Optional[float] = None) -> list[Detection]:
        async with self._lock:
            items = await asyncio.to_thread(self.detector.detect, decode_rgb(image))
            return [as_detection(item) for item in items]

    async def get_detections_from_camera(self, camera_name: str, *, extra: Optional[Mapping[str, ValueTypes]] = None,
                                         timeout: Optional[float] = None) -> list[Detection]:
        result = await self.capture_all_from_camera(camera_name, return_detections=True, timeout=timeout)
        return result.detections or []

    async def capture_all_from_camera(self, camera_name: str, return_image: bool = False,
                                      return_classifications: bool = False, return_detections: bool = False,
                                      return_object_point_clouds: bool = False, return_detections_3d: bool = False,
                                      *, extra: Optional[Mapping[str, ValueTypes]] = None,
                                      timeout: Optional[float] = None) -> CaptureAllResult:
        if return_classifications or return_detections_3d:
            raise NotSupportedError("Cup service supports 2D detections and object point clouds")
        async with self._lock:
            color, rgb, depth, k, timestamp, frame = await self._capture(camera_name, timeout)
            items = await asyncio.to_thread(self.detector.detect, rgb)
            clouds = await asyncio.to_thread(self._clouds, items, depth, k, frame) if return_object_point_clouds else None
            return CaptureAllResult(image=color if return_image else None,
                                    detections=[as_detection(item) for item in items] if return_detections else None,
                                    objects=clouds, extra={"captured_at": timestamp, "reference_frame": frame})

    async def get_object_point_clouds(self, camera_name: str, *, extra: Optional[Mapping[str, ValueTypes]] = None,
                                      timeout: Optional[float] = None) -> list[PointCloudObject]:
        result = await self.capture_all_from_camera(camera_name, return_object_point_clouds=True, timeout=timeout)
        return result.objects or []

    async def get_properties(self, *, extra: Optional[Mapping[str, ValueTypes]] = None,
                             timeout: Optional[float] = None) -> Vision.Properties:
        return Vision.Properties(detections_supported=True, object_point_clouds_supported=True,
                                 classifications_supported=False, detections_3d_supported=False,
                                 default_camera=self.camera_name)

    async def get_classifications(self, image: ViamImage, count: int, **kwargs: Any) -> list[Classification]:
        raise NotSupportedError("This service does not classify images")

    async def get_classifications_from_camera(self, camera_name: str, count: int, **kwargs: Any) -> list[Classification]:
        raise NotSupportedError("This service does not classify images")

    async def get_detections_3d(self, camera_name: str, **kwargs: Any) -> list[Detection3D]:
        raise NotSupportedError("Use rim_centers DoCommand for validated rim geometry")

    async def do_command(self, command: Mapping[str, ValueTypes], *, timeout: Optional[float] = None,
                         **kwargs: Any) -> Mapping[str, ValueTypes]:
        if command.get("command") != "rim_centers":
            raise ValueError("Supported command: {\"command\": \"rim_centers\"}")
        camera_name = command.get("camera_name", self.camera_name)
        if not isinstance(camera_name, str):
            raise ValueError("camera_name must be a string")
        async with self._lock:
            _, rgb, depth, k, timestamp, frame = await self._capture(camera_name, timeout)
            items = await asyncio.to_thread(self.detector.detect, rgb)
            results: list[dict[str, Any]] = []
            for item in items:
                estimate = await asyncio.to_thread(self.detector.localize_rim, item, rgb, depth, k)
                results.append({"bbox": list(item.bbox), "confidence": float(item.confidence), "label": item.label,
                                "center_mm": list(estimate.center_mm) if estimate is not None else None,
                                "normal": list(estimate.normal) if estimate is not None else None,
                                "diameter_mm": float(estimate.diameter_mm) if estimate is not None else None,
                                "height_mm": getattr(estimate, "height_mm", None),
                                "support": int(estimate.support) if estimate is not None else 0,
                                "residual_mm": float(estimate.residual_mm) if estimate is not None else None})
            return {"reference_frame": frame, "captured_at": timestamp, "cups": results}
