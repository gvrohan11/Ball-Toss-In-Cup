"""SDK adapter tests with a paired synthetic RGB-D camera; no hardware access."""
import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from google.protobuf.timestamp_pb2 import Timestamp
from viam.components.camera import Camera
from viam.media.video import CameraMimeType, NamedImage
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResponseMetadata
from viam.utils import dict_to_struct

from cup_vision.core import CupDetection
from cup_vision.service import RedSoloCupVision


class PairedCamera:
    def __init__(self):
        self.calls = 0

    async def get_images(self, **kwargs):
        self.calls += 1
        encoded = cv2.imencode(".png", np.zeros((20, 30, 3), np.uint8))[1].tobytes()
        raw_depth = b"DEPTHMAP" + (30).to_bytes(8, "big") + (20).to_bytes(8, "big")
        raw_depth += np.full((20, 30), 500, dtype=">u2").tobytes()
        timestamp = Timestamp(seconds=1700000000)
        # Intentionally reversed order: image names, not array order, identify streams.
        return [NamedImage("depth", raw_depth, CameraMimeType.VIAM_RAW_DEPTH),
                NamedImage("color", encoded, CameraMimeType.PNG)], ResponseMetadata(captured_at=timestamp)

    async def get_properties(self, **kwargs):
        return SimpleNamespace(
            intrinsic_parameters=SimpleNamespace(width_px=30, height_px=20, focal_x_px=100., focal_y_px=100., center_x_px=15., center_y_px=10.),
            distortion_parameters=SimpleNamespace(parameters=[]), default_reference_frame="cam")


def make_service():
    camera = PairedCamera()
    config = ComponentConfig(name="cups", attributes=dict_to_struct({"camera_name": "cam", "semantic_enabled": False}))
    service = RedSoloCupVision.new(config, {Camera.get_resource_name("cam"): camera})
    service.detector = Mock()
    service.detector.detect.return_value = [CupDetection((2, 2, 28, 18), .8)]
    service.detector.localize_rim.return_value = None
    return service, camera


def test_capture_image_and_detections_share_one_exposure():
    service, camera = make_service()
    result = asyncio.run(service.capture_all_from_camera("cam", return_image=True, return_detections=True))
    assert camera.calls == 1
    assert result.image.name == "color"
    assert result.detections[0].class_name == "red_solo_cup"
    assert result.detections[0].x_max == 28


def test_pcd_units_are_meters_while_geometry_is_millimeters():
    service, _ = make_service()
    objects = asyncio.run(service.get_object_point_clouds("cam"))
    raw = objects[0].point_cloud.split(b"DATA binary\n", 1)[1]
    points = np.frombuffer(raw, "<f4").reshape(-1, 3)
    assert np.allclose(points[:, 2], .5)
    assert objects[0].geometries.reference_frame == "cam"
    assert objects[0].geometries.geometries[0].center.z == 500


def test_missing_rim_is_explicit_not_fabricated_from_blob_center():
    service, _ = make_service()
    result = asyncio.run(service.do_command({"command": "rim_centers"}))
    assert result["cups"][0]["center_mm"] is None
    assert result["cups"][0]["support"] == 0
    assert result["captured_at"].endswith("Z")


def test_reject_different_camera_to_prevent_wrong_calibration():
    service, _ = make_service()
    with pytest.raises(ValueError, match="calibrated"):
        asyncio.run(service.get_detections_from_camera("other"))


def test_config_rejects_unknown_and_fractional_integer_attributes():
    with pytest.raises(ValueError, match="Unknown"):
        RedSoloCupVision.validate_config(ComponentConfig(attributes=dict_to_struct({"threshold_typo": .5})))
    with pytest.raises(ValueError, match="integer"):
        RedSoloCupVision.validate_config(ComponentConfig(attributes=dict_to_struct({"image_size": 640.5})))


def test_config_returns_camera_dependency():
    deps = RedSoloCupVision.validate_config(ComponentConfig(attributes=dict_to_struct({"camera_name": "wrist"})))
    assert deps == (["wrist"], [])
