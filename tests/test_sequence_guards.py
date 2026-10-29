"""Offline regression checks: no robot connection or motion."""
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import cup_aim
import drop
import pick_ball
import main


class SequenceGuards(unittest.IsolatedAsyncioTestCase):
    def test_partial_ball_box_is_rejected(self):
        box = SimpleNamespace(x_min=746, x_max=779, y_min=556, y_max=613)
        self.assertIsNone(pick_ball.best_ball([box]))

    async def test_localization_jump_blocks_descent(self):
        with patch.object(pick_ball, 'move', new_callable=AsyncMock) as move, \
             patch.object(pick_ball, 'locate_ball', new=AsyncMock(side_effect=[
                 (308, -44, 63), (351, 45, 41)])):
            with self.assertRaisesRegex(RuntimeError, 'descent blocked'):
                await pick_ball.find_ball(object(), object(), True)
        self.assertEqual(move.await_count, 2)  # look and measurement only

    def test_dry_run_does_not_write_attempts(self):
        with patch.object(main, 'log_attempt') as log:
            main.record_attempt({'phase': 'drop'}, False)
        log.assert_not_called()

    async def test_failed_release_motion_keeps_gripper_closed(self):
        motion = SimpleNamespace(move=AsyncMock(return_value=False))
        gripper = SimpleNamespace(open=AsyncMock())
        with patch.object(drop.MotionClient, 'from_robot', return_value=motion), \
             patch.object(drop.Gripper, 'from_robot', return_value=gripper):
            with self.assertRaises(RuntimeError):
                await drop.drop_in_cup(object(), (500, 0, 110), go=True)
        gripper.open.assert_not_awaited()

    async def test_failed_pick_motion_stops(self):
        motion = SimpleNamespace(move=AsyncMock(return_value=False))
        with self.assertRaises(RuntimeError):
            await pick_ball.move(motion, 350, 0, 150, True)

    async def test_missing_gripper_position_is_an_error(self):
        gripper = SimpleNamespace(do_command=AsyncMock(return_value={}))
        with patch.object(pick_ball.asyncio, 'sleep', new_callable=AsyncMock):
            with self.assertRaises(RuntimeError):
                await pick_ball.wait_gripper(gripper)

    async def test_gripper_must_settle_before_timeout(self):
        gripper = SimpleNamespace(do_command=AsyncMock(side_effect=[{'pos': 1}, {'pos': 2}]))
        with patch.object(pick_ball.asyncio, 'sleep', new_callable=AsyncMock):
            with self.assertRaises(TimeoutError):
                await pick_ball.wait_gripper(gripper, timeout_s=0.4)

    async def test_gripper_settles(self):
        gripper = SimpleNamespace(do_command=AsyncMock(side_effect=[
            {'pos': 300}, {'pos': 200}, {'pos': 200}, {'pos': 200}]))
        with patch.object(pick_ball.asyncio, 'sleep', new_callable=AsyncMock):
            self.assertEqual(await pick_ball.wait_gripper(gripper), 200)

    async def test_empty_grasp_does_not_lift(self):
        motion = SimpleNamespace(move=AsyncMock(return_value=True))
        gripper = SimpleNamespace(open=AsyncMock(), grab=AsyncMock())
        with patch.object(pick_ball.MotionClient, 'from_robot', return_value=motion), \
             patch.object(pick_ball.Gripper, 'from_robot', return_value=gripper), \
             patch.object(pick_ball, 'wait_gripper', new=AsyncMock(return_value=0)):
            self.assertFalse(await pick_ball.pick(object(), (350, 0, 20), True))
        self.assertEqual(motion.move.await_count, 2)  # hover and descend; no lift

    async def test_table_intersection_behind_camera_is_rejected(self):
        robot = SimpleNamespace(transform_pose=AsyncMock(side_effect=[
            SimpleNamespace(pose=SimpleNamespace(x=0, y=0, z=100)),
            SimpleNamespace(pose=SimpleNamespace(x=0, y=0, z=1100))]))
        intrinsics = SimpleNamespace(center_x_px=0, center_y_px=0, focal_x_px=500, focal_y_px=500)
        self.assertIsNone(await cup_aim._pixel_to_table(robot, intrinsics, 0, 0, 0))


if __name__ == '__main__':
    unittest.main()
