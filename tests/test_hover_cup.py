import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from viam.proto.common import Pose, PoseInFrame

from cup_vision.client import LocatedCup
from hover_cup import CupHoverController, HoverSettings, HoverPlan


def controller(cups):
    obj = object.__new__(CupHoverController)
    obj.settings = HoverSettings(samples=3)
    obj.locator = SimpleNamespace(sample=AsyncMock(return_value=cups))
    obj.arm = SimpleNamespace(get_joint_positions=AsyncMock(return_value=SimpleNamespace(values=[0]*6)), stop=AsyncMock(), do_command=AsyncMock(return_value={}))
    obj.motion = SimpleNamespace(get_pose=AsyncMock(return_value=PoseInFrame(
        reference_frame='world', pose=Pose(x=300,y=0,z=500,o_z=-1))), move=AsyncMock(return_value=True))
    return obj


def cup(x=400, normal=(0,0,1)):
    return LocatedCup((10,10,100,100), .9, (x,0,100), normal, 95, 120, 100, 1)


def test_preview_targets_gripper_400mm_above_rim_without_motion():
    obj = controller([cup()])
    plan = asyncio.run(obj.plan())
    assert plan.gripper_target_mm == (400,0,500)
    obj.motion.move.assert_not_called()
    obj.arm.do_command.assert_not_called()
    obj.motion.get_pose.assert_awaited_once_with('gripper', 'world', timeout=10)
    assert plan.joint_speed_deg_per_sec == 180


def test_multiple_cups_require_selection():
    with pytest.raises(ValueError, match='Multiple cups'):
        asyncio.run(controller([cup(),cup(500)]).plan())


def test_transient_missing_rim_is_retried_without_motion():
    obj = controller([])
    obj.locator.sample.side_effect = [[], [cup()], [cup()], [cup()]]
    plan = asyncio.run(obj.plan())
    assert plan.gripper_target_mm == (400,0,500)
    assert obj.locator.sample.await_count == 4
    obj.motion.move.assert_not_called()


def test_persistent_missing_rim_stops_after_bounded_retries():
    obj = controller([])
    with pytest.raises(RuntimeError, match='Only 0/3 usable rim observations'):
        asyncio.run(obj.plan())
    assert obj.locator.sample.await_count == 9
    obj.motion.move.assert_not_called()


def test_cli_expected_localization_error_has_no_traceback(monkeypatch, caplog):
    import hover_cup
    async def fail(_args):
        raise RuntimeError('No usable rim')
    monkeypatch.setattr(hover_cup, '_run', fail)
    monkeypatch.setattr('sys.argv', ['hover_cup.py'])
    with pytest.raises(SystemExit) as error:
        hover_cup.main()
    assert error.value.code == 2
    assert 'Hover stopped: No usable rim' in caplog.text


def test_tilted_cup_refuses_hover():
    with pytest.raises(RuntimeError, match='tilted'):
        asyncio.run(controller([cup(normal=(1,0,0))]).plan())


def test_stale_plan_refuses_motion():
    obj = controller([])
    plan = HoverPlan((400,0,100),(400,0,400),(300,0,500),95,120,1,time.monotonic()-20)
    with pytest.raises(RuntimeError, match='stale'):
        asyncio.run(obj.execute(plan))
    obj.motion.move.assert_not_called()


def test_failed_motion_stops_arm():
    obj = controller([])
    obj.motion.move.return_value = False
    plan = HoverPlan((400,0,100),(400,0,400),(300,0,500),95,120,1,time.monotonic())
    with pytest.raises(RuntimeError, match='returned false'):
        asyncio.run(obj.execute(plan))
    obj.arm.stop.assert_awaited_once()
    assert obj.motion.move.call_args.kwargs['component_name'] == 'gripper'
    assert [call.args[0] for call in obj.arm.do_command.await_args_list] == [
        {'set_speed': 180.0}, {'set_speed': 60.0}]


def test_success_moves_gripper_and_restores_speed():
    obj = controller([])
    obj.motion.get_pose.side_effect = [
        PoseInFrame(reference_frame='world', pose=Pose(x=300,y=0,z=500,o_z=-1)),
        PoseInFrame(reference_frame='world', pose=Pose(x=400,y=0,z=500,o_z=-1))]
    plan = HoverPlan((400,0,100),(400,0,500),(300,0,500),95,120,1,time.monotonic())
    asyncio.run(obj.execute(plan))
    assert obj.motion.move.await_count == 3
    assert all(call.kwargs['component_name'] == 'gripper' for call in obj.motion.move.await_args_list)
    final = obj.motion.move.call_args.kwargs['destination']
    assert (final.pose.x,final.pose.y,final.pose.z,final.pose.o_z) == (400,0,500,-1)
    assert [call.args[0] for call in obj.arm.do_command.await_args_list] == [
        {'set_speed': 180.0}, {'set_speed': 60.0}]


def test_speed_above_driver_limit_rejected():
    with pytest.raises(ValueError, match='joint speed'):
        HoverSettings(speed_multiplier=4).validate()


def test_invalid_workspace_rejected():
    with pytest.raises(ValueError, match='Workspace'):
        HoverSettings(minimum_mm=(0,0,0), maximum_mm=(0,0,0)).validate()
