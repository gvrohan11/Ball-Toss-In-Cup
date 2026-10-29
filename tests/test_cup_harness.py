"""Tests for paired capture decoding and offline dataset replay."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from viam.media.video import CameraMimeType, NamedImage

from scripts.cup_test import annotate, decode_pair, offline_sources


def pair() -> tuple[list[NamedImage], np.ndarray]:
    color = np.full((20, 30, 3), (50, 100, 180), np.uint8)
    encoded, jpeg = cv2.imencode(".jpg", color)
    assert encoded
    depth = np.full((20, 30), 500, dtype=np.uint16)
    data = b"DEPTHMAP" + (30).to_bytes(8, "big") + (20).to_bytes(8, "big") + depth.astype(">u2").tobytes()
    return [NamedImage("depth", data, CameraMimeType.VIAM_RAW_DEPTH),
            NamedImage("color", jpeg.tobytes(), CameraMimeType.JPEG)], depth


class CaptureTests(unittest.TestCase):
    def test_streams_are_decoded_by_name_instead_of_order(self) -> None:
        images, expected = pair()
        color, depth = decode_pair(images, (20, 30))
        self.assertEqual(color.shape, (20, 30, 3))
        np.testing.assert_array_equal(depth, expected)

    def test_calibration_shape_mismatch_is_rejected(self) -> None:
        images, _ = pair()
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            decode_pair(images, (10, 15))

    def test_missing_depth_is_rejected(self) -> None:
        images, _ = pair()
        with self.assertRaisesRegex(ValueError, "color and depth"):
            decode_pair(images[1:], (20, 30))

    def test_saved_frame_records_prevent_duplicate_annotated_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frame_000000.json").write_text(json.dumps({"color_file": "frame_000000_color.png"}))
            (root / "frame_000000_color.png").touch()
            (root / "frame_000000_annotated.png").touch()
            self.assertEqual(list(offline_sources([root])),
                             [(root / "frame_000000_color.png", root / "frame_000000.json")])

    def test_annotation_preserves_source_and_reports_missing_rim(self) -> None:
        original = np.zeros((80, 100, 3), dtype=np.uint8)
        detection = types.SimpleNamespace(bbox=(10, 10, 40, 50), confidence=0.9,
                                          label="red_solo_cup", rim_ellipse=None)
        fake_core = types.SimpleNamespace(localize_rim=lambda *args, **kwargs: None)
        with patch.dict("sys.modules", {"cup_detection.core": fake_core}):
            image, records = annotate(original, [detection], np.ones((80, 100)),
                                      {"fx": 100, "fy": 100, "cx": 50, "cy": 40},
                                      types.SimpleNamespace(config=None))
        self.assertFalse(original.any())
        self.assertTrue(image.any())
        self.assertIsNone(records[0]["rim_center_camera_mm"])
        self.assertEqual(records[0]["bbox"], [10, 10, 40, 50])


if __name__ == "__main__":
    unittest.main()
