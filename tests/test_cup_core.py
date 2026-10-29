"""Geometry and appearance regression tests without downloading a neural model."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from cup_detection.core import CupDetection, CupDetector, DetectorConfig, _masks


def overhead_cup(center=(160, 120), pink=True):
    rgb = np.full((240, 320, 3), (60, 80, 60), np.uint8)
    cv2.circle(rgb, center, 65, (245, 245, 245), -1)
    cv2.circle(rgb, center, 57, (220, 170, 185) if pink else (230, 230, 230), -1)
    return rgb


class CupCoreTests(unittest.TestCase):
    def setUp(self):
        self.detector = CupDetector(DetectorConfig(semantic_enabled=False, top_view_enabled=True))

    def test_red_hue_wrap_excludes_orange(self):
        hsv = np.array([[[0, 255, 200], [179, 255, 200], [20, 255, 200]]], np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        red, _, _ = _masks(rgb, self.detector.config)
        self.assertEqual(red.tolist(), [[True, True, False]])

    def test_top_down_cup_anywhere_in_frame(self):
        for center in ((80, 80), (230, 160)):
            detections = self.detector.detect(overhead_cup(center))
            self.assertEqual(len(detections), 1)
            self.assertAlmostEqual(detections[0].rim_ellipse[0], center[0], delta=2)

    def test_white_plate_and_solid_red_patch_rejected(self):
        self.assertEqual(self.detector.detect(overhead_cup(pink=False)), [])
        red = np.full((240, 320, 3), (80, 80, 80), np.uint8)
        cv2.rectangle(red, (40, 40), (160, 200), (180, 5, 10), -1)
        self.assertEqual(self.detector.detect(red), [])

    def test_semantic_red_cup_keeps_full_object_box(self):
        image = np.full((240, 320, 3), (80, 80, 80), np.uint8)
        cv2.rectangle(image, (50, 50), (160, 200), (180, 5, 10), -1)
        with patch.object(self.detector, "_semantic_boxes", return_value=[((40, 30, 180, 215), 0.8)]):
            detections = self.detector.detect(image)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox, (40, 30, 180, 215))

    def test_model_cup_not_rejected_when_no_red_exterior_visible(self):
        detector = CupDetector(DetectorConfig(top_view_enabled=False))
        image = np.full((240, 320, 3), (190, 195, 225), np.uint8)
        with patch.object(detector, "_semantic_boxes", return_value=[((40,30,180,215), .8)]):
            detections = detector.detect(image)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].confidence, .8)

    def test_reflective_red_body_is_not_a_white_rim(self):
        image = np.full((240,320,3), (180,5,10), np.uint8)
        cv2.ellipse(image, (160,120), (80,65), 0, 0, 360, (255,255,255), 10)
        self.assertEqual(self.detector._rings(image), [])

    def test_oversized_rim_fit_is_rejected(self):
        detector = CupDetector(DetectorConfig(top_view_enabled=False))
        image = np.full((240,320,3), (200,180,190), np.uint8)
        with patch.object(detector, "_semantic_boxes", return_value=[((80,60,200,190),.8)]), patch.object(
            detector, "_rings", return_value=[((60,60,150,150,0),.9,.5)]):
            detections = detector.detect(image)
        self.assertIsNone(detections[0].rim_ellipse)

    def test_rim_uses_ring_depth_not_interior_bottom(self):
        rgb = overhead_cup()
        depth = np.full((240, 320), 750.0)
        ys, xs = np.indices(depth.shape)
        radius = np.hypot(xs - 160, ys - 120)
        depth[(radius >= 57) & (radius <= 65)] = 500
        detection = CupDetection((95, 55, 226, 186), .8, rim_ellipse=(160, 120, 130, 130, 0))
        estimate = self.detector.localize_rim(detection, rgb, depth, {"fx": 650, "fy": 650, "cx": 160, "cy": 120})
        self.assertIsNotNone(estimate)
        self.assertAlmostEqual(estimate.center_mm[2], 500, delta=1)
        self.assertAlmostEqual(estimate.center_mm[0], 0, delta=1)
        self.assertAlmostEqual(estimate.diameter_mm, 100, delta=6)
        self.assertAlmostEqual(estimate.height_mm, 250, delta=1)

    def test_height_missing_support_plane_is_unknown(self):
        rgb = overhead_cup()
        depth = np.zeros((240, 320), float)
        ys, xs = np.indices(depth.shape)
        r = np.hypot(xs - 160, ys - 120)
        depth[(r >= 57) & (r <= 65)] = 500
        detection = CupDetection((95, 55, 226, 186), .8, rim_ellipse=(160, 120, 130, 130, 0))
        estimate = self.detector.localize_rim(detection, rgb, depth, {"fx": 650, "fy": 650, "cx": 160, "cy": 120})
        self.assertIsNotNone(estimate)
        self.assertIsNone(estimate.height_mm)

    def test_missing_rim_depth_refuses_3d(self):
        rgb = overhead_cup()
        depth = np.full((240, 320), 750.0)
        ys, xs = np.indices(depth.shape)
        depth[np.hypot(xs - 160, ys - 120) > 55] = 0
        detection = CupDetection((95, 55, 226, 186), .8, rim_ellipse=(160, 120, 130, 130, 0))
        self.assertIsNone(self.detector.localize_rim(detection, rgb, depth, {"fx": 650, "fy": 650, "cx": 160, "cy": 120}))

    def test_wrong_measured_diameter_refuses_3d(self):
        detector = CupDetector(DetectorConfig(semantic_enabled=False, rim_diameter_mm=40))
        rgb = overhead_cup()
        detection = CupDetection((95, 55, 226, 186), .8, rim_ellipse=(160, 120, 130, 130, 0))
        self.assertIsNone(detector.localize_rim(detection, rgb, np.full((240, 320), 500),
                                              {"fx": 650, "fy": 650, "cx": 160, "cy": 120}))

    def test_invalid_thresholds_fail_config(self):
        for kwargs in ({"red_fraction_min": -1}, {"morphology_size": 2}, {"rim_diameter_mm": 0}, {"semantic_enabled": "false"}):
            with self.assertRaises(ValueError):
                DetectorConfig(**kwargs).validate()


if __name__ == "__main__":
    unittest.main()
