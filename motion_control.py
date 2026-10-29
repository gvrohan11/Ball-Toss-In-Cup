"""Single-owner Viam planned motion with continuous read-only pose feedback."""
import asyncio
import contextlib
import json
import time
from pathlib import Path

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient


class MotionController:
    def __init__(self, robot, go=False):
        self.robot, self.go = robot, go
        self.motion = MotionClient.from_robot(robot, 'builtin')
        self.arm = Arm.from_robot(robot, 'arm')
        self.stage = 'idle'
        self.cup_memory = None
        self.monitor_payload = False
        self.held_ball_local = None
        self.lock = asyncio.Lock()
        self.path = Path(__file__).resolve().parent / 'observations/run-feedback.jsonl'

    def event(self, state, **values):
        record = dict(time=time.time(), stage=self.stage, state=state, **values)
        print(json.dumps(record), flush=True)
        self.path.parent.mkdir(exist_ok=True)
        with self.path.open('a') as stream:
            stream.write(json.dumps(record)+'\n')

    async def _feedback(self):
        while True:
            pose = await self.motion.get_pose('gripper', 'world', timeout=2)
            p = pose.pose
            values = dict(gripper_world_mm=[p.x, p.y, p.z])
            if self.cup_memory:
                values['cup_world_mm'] = self.cup_memory.center
            if self.monitor_payload:
                from cup_tracking import world_point
                import math
                payload = await asyncio.wait_for(
                    world_point(self.robot, self.held_ball_local, source='gripper'), 2)
                center = self.cup_memory.center
                distance = math.hypot(payload[0]-center[0], payload[1]-center[1])
                values.update(ball_world_mm=payload.tolist(), cup_xy_error_mm=distance,
                              over_remembered_cup=distance <= 8,
                              vision_status='awaiting stationary recheck')
                if not (await Gripper.from_robot(self.robot, 'gripper').is_holding_something(timeout=2)).is_holding_something:
                    raise RuntimeError('Payload lost during transfer')
            self.event('moving', **values)
            await asyncio.sleep(0.5)

    async def move(self, pose, stage):
        async with self.lock:
            self.stage = stage
            self.event('planning', target_world=pose)
            if not self.go:
                self.event('dry_run')
                return
            feedback = asyncio.create_task(self._feedback())
            motion = asyncio.create_task(self.motion.move('gripper',
                PoseInFrame(reference_frame='world', pose=Pose(**pose)), timeout=60))
            try:
                done, _ = await asyncio.wait([motion, feedback], return_when=asyncio.FIRST_COMPLETED)
                if feedback in done:
                    await feedback  # propagate telemetry failure
                    raise RuntimeError('Feedback stopped during motion')
                if not await motion:
                    raise RuntimeError('Viam planner did not complete motion')
                self.event('arrived')
            except BaseException:
                with contextlib.suppress(Exception):
                    await self.arm.stop(timeout=3)
                raise
            finally:
                motion.cancel()
                feedback.cancel()
                await asyncio.gather(motion, feedback, return_exceptions=True)
