import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from scene_detection import Instance, SceneDetector, ball_candidates


def sphere_scene(centers):
    height, width = 200, 300
    v, u = np.indices((height, width))
    rays = np.stack([(u-150)/500, (v-100)/500, np.ones_like(u)], axis=-1)
    a = (rays*rays).sum(axis=-1)
    depth = np.zeros((height, width), float)
    instances = []
    for center in centers:
        center = np.array(center)
        b = (rays*center).sum(axis=-1)
        disc = b*b-a*((center*center).sum()-400)
        mask = disc > 0
        z = (b-np.sqrt(np.maximum(disc, 0)))/a
        depth[mask] = np.where((depth[mask] == 0) | (z[mask] < depth[mask]), z[mask], depth[mask])
        yy, xx = np.nonzero(mask)
        instances.append(Instance('sports ball', .9,
                         (xx.min(), yy.min(), xx.max()+1, yy.max()+1), mask))
    return instances, depth


class PlateGeometry(unittest.TestCase):
    def test_multiple_ball_spheres_without_color_filter(self):
        instances, depth = sphere_scene([(-35, 0, 500), (35, 0, 500)])
        reports = ball_candidates(instances, depth, (500, 500, 150, 100), (0, 0, 300, 200))
        self.assertEqual(len(reports), 2)
        self.assertTrue(all(r['accepted'] for r in reports))
        np.testing.assert_allclose(reports[0]['center_camera_mm'], [-35, 0, 500], atol=.5)
        self.assertTrue(all(r['grasp_ready'] is False for r in reports))

    def test_touching_balls_are_not_isolated(self):
        instances, depth = sphere_scene([(-20, 0, 500), (20, 0, 500)])
        reports = ball_candidates(instances, depth, (500, 500, 150, 100), (0, 0, 300, 200))
        self.assertTrue(all(r['accepted'] for r in reports))
        self.assertTrue(all(not r['isolated_from_detected_neighbors'] for r in reports))

    def test_flat_circular_print_is_rejected(self):
        instances, depth = sphere_scene([(0, 0, 500)])
        depth[depth > 0] = 500
        reports = ball_candidates(instances, depth, (500,500,150,100), (0,0,300,200))
        self.assertFalse(reports[0]['accepted'])

    def test_missing_depth_rejected(self):
        instances, depth = sphere_scene([(0, 0, 500)])
        reports = ball_candidates(instances, np.zeros_like(depth), (500,500,150,100), (0,0,300,200))
        self.assertFalse(reports[0]['accepted'])

    def test_plate_boundary_rejected(self):
        instances, depth = sphere_scene([(0, 0, 500)])
        reports = ball_candidates(instances, depth, (500,500,150,100), (145,0,300,200))
        self.assertFalse(reports[0]['accepted'])

    def test_cup_not_treated_as_ball(self):
        instances, depth = sphere_scene([(0, 0, 500)])
        instances[0].label = 'cup'
        self.assertEqual(ball_candidates(instances, depth, (500,500,150,100), (0,0,300,200)), [])

    def test_one_shared_model_requests_both_classes(self):
        model = SimpleNamespace(names={32:'sports ball', 41:'cup', 0:'person'}, predict=Mock(return_value=[]))
        detector = SceneDetector(model=model)
        for color in ((255,255,255), (0,140,255)):
            detector.detect(np.full((20,20,3), color, np.uint8))
        self.assertEqual(model.predict.call_args.kwargs['classes'], [32,41])


if __name__ == '__main__':
    unittest.main()
