import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np

from cup_tracking import CupMemory, match_memory, triangulate, rim_circles, held_ball_circle
from drop import drop_at_remembered_cup
from motion_control import MotionController


def observation(center=(400, 150, 110), origin=(0, 0, 500), radius=50):
    origin = np.array(origin, dtype=float)
    scale = (origin[2]-center[2])/1000
    return dict(origin=origin, direction=(np.array(center)-origin)/scale,
                circle=(600, 400, radius*900/(scale*1000)), focal=900)


class CupGeometry(unittest.TestCase):
    def test_two_view_world_coordinates(self):
        cup = triangulate(observation(), observation(origin=(0, 60, 500)))
        np.testing.assert_allclose(cup.center, [400, 150, 110], atol=1e-6)
        self.assertAlmostEqual(cup.radius_mm, 50)

    def test_same_view_is_not_stereo(self):
        self.assertIsNone(triangulate(observation(), observation()))

    def test_memory_independent_of_camera_origin(self):
        memory = CupMemory([400, 150, 110], 50, time.time(), 0)
        np.testing.assert_allclose(match_memory([observation(origin=(50, -80, 600))], memory), memory.center)

    def test_moved_cup_blocks_release(self):
        memory = CupMemory([400, 150, 110], 50, time.time(), 0)
        with self.assertRaisesRegex(RuntimeError, 'moved'):
            match_memory([observation(center=(420, 150, 110))], memory)

    def test_missing_cup_blocks_release(self):
        with self.assertRaisesRegex(RuntimeError, 'not visible'):
            match_memory([], CupMemory([400, 150, 110], 50, time.time(), 0))

    def test_expired_memory_blocks_release(self):
        with self.assertRaisesRegex(RuntimeError, 'expired'):
            CupMemory([400, 150, 110], 50, time.time()-200, 0).require_fresh()

    def test_ambiguous_rims_block_release(self):
        with self.assertRaisesRegex(RuntimeError, 'Ambiguous'):
            match_memory([observation(), observation(center=(405, 150, 110))],
                         CupMemory([400, 150, 110], 50, time.time(), 0))

    def test_blank_image_has_no_rim(self):
        self.assertEqual(rim_circles(np.zeros((720, 1280, 3), dtype=np.uint8)), [])

    def test_partial_held_sphere_can_be_measured(self):
        import cv2
        image = np.zeros((720, 1280, 3), np.uint8)
        cv2.circle(image, (855, 760), 130, (20, 130, 240), -1)
        x, y, radius = held_ball_circle(image)
        np.testing.assert_allclose([x, y, radius], [855, 760, 130], atol=2)

    def test_missing_payload_measurement_blocks(self):
        with self.assertRaises(RuntimeError):
            held_ball_circle(np.zeros((720, 1280, 3), np.uint8))


class ReleaseChecks(unittest.IsolatedAsyncioTestCase):
    async def test_loop_remembers_cup_before_pick(self):
        import main
        order = []
        robot = SimpleNamespace(close=AsyncMock())
        controller = SimpleNamespace(go=True, arm=SimpleNamespace(do_command=AsyncMock()),
                                     move=AsyncMock(), event=Mock())
        async def scan(*args):
            order.append('remember_cup')
            return CupMemory([400, 150, 110], 50, time.time(), 0)
        async def find(*args):
            order.append('find_ball')
            return (350, 0, 20)
        async def pick(*args, **kwargs):
            order.append('pick')
            return True
        with patch.object(main.sys, 'argv', ['main.py', '--go']), \
             patch.object(main, 'connect', new=AsyncMock(return_value=robot)), \
             patch.object(main, 'MotionController', return_value=controller), \
             patch.object(main, 'scan_cup', side_effect=scan), \
             patch.object(main, 'find_ball', side_effect=find), \
             patch.object(main, 'pick', side_effect=pick), \
             patch.object(main, 'drop_at_remembered_cup', new=AsyncMock(return_value={'released': True})), \
             patch.object(main, 'record_attempt'), patch('builtins.input', return_value='y'):
            await main.main()
        self.assertEqual(order, ['remember_cup', 'find_ball', 'pick'])
        robot.close.assert_awaited_once()

    async def run_drop(self, payload, visible=None):
        gripper = SimpleNamespace(is_holding_something=AsyncMock(return_value=
            SimpleNamespace(is_holding_something=True)), open=AsyncMock())
        controller = SimpleNamespace(go=True, move=AsyncMock(), event=Mock(), monitor_payload=True)
        memory = CupMemory([400, 150, 110], 50, time.time(), 0)
        with patch('drop.Gripper.from_robot', return_value=gripper), \
             patch('drop.measure_held_ball', new=AsyncMock(return_value=np.array([20., 20., 50.]))), \
             patch('drop.verify_cup', new=AsyncMock(return_value=np.array(visible or memory.center))), \
             patch('drop.world_point', new=AsyncMock(return_value=np.array(payload))):
            try:
                result = await drop_at_remembered_cup(object(), memory, controller)
            except RuntimeError:
                gripper.open.assert_not_awaited()
                raise
        return result, controller, gripper

    async def test_aligned_payload_releases_after_single_transfer(self):
        result, controller, gripper = await self.run_drop([400, 150, 250])
        self.assertTrue(result['released'])
        controller.move.assert_awaited_once()
        gripper.open.assert_awaited_once()
        self.assertFalse(controller.monitor_payload)

    async def test_unaligned_payload_keeps_gripper_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'not aligned'):
            await self.run_drop([415, 150, 250])

    async def test_bad_height_keeps_gripper_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'not aligned'):
            await self.run_drop([400, 150, 120])

    async def test_telemetry_failure_stops_arm_and_cancels_move(self):
        motion = SimpleNamespace(move=AsyncMock(side_effect=lambda *a, **k: None))
        async def slow_move(*args, **kwargs):
            import asyncio
            await asyncio.sleep(60)
        motion.move = AsyncMock(side_effect=slow_move)
        arm = SimpleNamespace(stop=AsyncMock())
        with patch('motion_control.MotionClient.from_robot', return_value=motion), \
             patch('motion_control.Arm.from_robot', return_value=arm):
            controller = MotionController(object(), True)
        controller.event = Mock()
        controller._feedback = AsyncMock(side_effect=RuntimeError('telemetry failed'))
        with self.assertRaisesRegex(RuntimeError, 'telemetry failed'):
            await controller.move(dict(x=400, y=150, z=300, o_z=-1), 'transfer')
        arm.stop.assert_awaited_once()


if __name__ == '__main__':
    unittest.main()
