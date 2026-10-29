"""Reusable, read-only local RGB-D cup localization through a Viam camera."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import json
from pathlib import Path

import cv2
import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.media.video import CameraMimeType
from viam.proto.common import Pose, PoseInFrame
from viam.robot.client import RobotClient

from cup_detection.core import CupDetector, CupDetection, RimEstimate

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocatedCup:
    bbox: tuple[int, int, int, int]
    confidence: float
    center_mm: tuple[float, float, float]
    normal: tuple[float, float, float]
    diameter_mm: float
    height_mm: float | None
    support: int
    residual_mm: float


class CupLocator:
    """Find reliably localized rim centers in a fixed frame, never moving anything.

    Inference runs locally. The same detector is also available as a Viam module.
    Keep the arm still: frame transforms use current joint positions, not historical
    capture-time transforms. Joint checks reject motion during a capture/inference.
    """

    def __init__(self, robot: RobotClient, detector: CupDetector, *, camera: str = "cam",
                 arm: str = "arm", base_frame: str = "world") -> None:
        self.robot = robot
        self.detector = detector
        self.camera_name = camera
        self.camera = Camera.from_robot(robot, camera)
        self.arm = Arm.from_robot(robot, arm)
        self.base_frame = base_frame

    async def sample(self) -> list[LocatedCup]:
        before = np.asarray((await self.arm.get_joint_positions(timeout=10)).values)
        properties = await self.camera.get_properties(timeout=10)
        k = properties.intrinsic_parameters
        intrinsics = dict(fx=k.focal_x_px, fy=k.focal_y_px, cx=k.center_x_px, cy=k.center_y_px)
        if (not np.isfinite(list(intrinsics.values())).all()
                or min(intrinsics["fx"], intrinsics["fy"]) <= 0):
            raise ValueError("Camera intrinsics are unavailable")
        if any(abs(value) > 1e-9 for value in properties.distortion_parameters.parameters):
            raise ValueError("This localizer requires rectified/aligned images; nonzero distortion reported")
        images, metadata = await self.camera.get_images(timeout=20)
        sources = {image.name: image for image in images}
        if not {"color", "depth"}.issubset(sources):
            raise ValueError("Camera must supply paired color and depth images")
        if sources["depth"].mime_type != CameraMimeType.VIAM_RAW_DEPTH:
            raise ValueError("Expected standard big-endian Viam depth; use little_endian_depth=false")
        bgr = cv2.imdecode(np.frombuffer(sources["color"].data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Cannot decode camera RGB frame")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = np.asarray(sources["depth"].bytes_to_depth_array(), dtype=np.uint16)
        if rgb.shape[:2] != depth.shape or depth.shape != (k.height_px, k.width_px):
            raise ValueError("Aligned RGB, depth and intrinsics have different dimensions")

        def infer() -> list[tuple[CupDetection, RimEstimate]]:
            result = []
            detections = self.detector.detect(rgb)
            details = []
            for detection in detections:
                rim = self.detector.localize_rim(detection, rgb, depth, intrinsics)
                details.append({"bbox": detection.bbox, "confidence": detection.confidence,
                                "rim_ellipse": detection.rim_ellipse,
                                "status": "ok" if rim else getattr(detection, "rim_failure_reason", "rim_fit_rejected")})
                if rim is not None:
                    result.append((detection, rim))
            if not result:
                directory = Path('/tmp/cup-hover-debug')
                directory.mkdir(exist_ok=True)
                cv2.imwrite(str(directory / 'color.png'), bgr)
                np.save(directory / 'depth_mm.npy', depth)
                (directory / 'diagnostics.json').write_text(json.dumps({"cups": details, "intrinsics": intrinsics}, indent=2))
                LOGGER.warning("2D cups=%d, but no usable 3D rim: %s; diagnostic frame saved in %s",
                               len(detections), [item['status'] for item in details], directory)
            return result

        inferred = await asyncio.to_thread(infer)
        result: list[LocatedCup] = []
        for detection, rim in inferred:
            c = np.asarray(rim.center_mm)
            endpoint = c + np.asarray(rim.normal) * 100
            transformed = []
            for point in (c, endpoint):
                pose = await self.robot.transform_pose(PoseInFrame(
                    reference_frame=self.camera_name,
                    pose=Pose(x=float(point[0]), y=float(point[1]), z=float(point[2]), o_z=1)),
                    self.base_frame)
                transformed.append(np.array([pose.pose.x, pose.pose.y, pose.pose.z]))
            normal = transformed[1] - transformed[0]
            normal /= np.linalg.norm(normal)
            if normal[2] < 0:
                normal = -normal
            result.append(LocatedCup(
                detection.bbox, detection.confidence, tuple(map(float, transformed[0])),
                tuple(map(float, normal)), float(rim.diameter_mm), getattr(rim, "height_mm", None),
                int(rim.support), float(rim.residual_mm)))
        after = np.asarray((await self.arm.get_joint_positions(timeout=10)).values)
        if before.size == 0 or before.shape != after.shape or np.max(np.abs(after - before)) > 0.2:
            raise RuntimeError("Arm moved during capture/inference; discard these coordinates and hold still")
        LOGGER.info("Capture %s: %d cups with usable 3D rims", metadata.captured_at.ToJsonString(), len(result))
        return result
