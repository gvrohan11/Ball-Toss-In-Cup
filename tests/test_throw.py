import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import throw  # noqa: E402

WIND_UP_JOINTS = [0.0, 5.0, -48.4, 0.0, -83.4, 180.0]   # measured at WIND_UP on the real arm


def swung(scale=1.0):
    end = list(WIND_UP_JOINTS)
    for j, d in throw.SWING_DEG.items():
        end[j] += d * scale
    return end


def test_default_swing_is_inside_joint_limits():
    throw.check_limits(WIND_UP_JOINTS, swung())


def test_old_draft_swing_was_out_of_limits():
    end = list(WIND_UP_JOINTS)
    end[2] += -35
    end[4] += -45                  # the draft's wrist direction: backward AND past -124 deg
    with pytest.raises(ValueError):
        throw.check_limits(WIND_UP_JOINTS, end)


def test_directions_are_the_measured_ones():
    assert throw.SWING_DEG[2] < 0 and throw.SWING_DEG[4] > 0


def test_scale_validated_and_dry_run_moves_nothing(monkeypatch):
    for cls in (throw.MotionClient, throw.Arm, throw.Gripper):
        monkeypatch.setattr(cls, "from_robot", staticmethod(lambda r, n: object()))
    with pytest.raises(ValueError):
        asyncio.run(throw.throw(object(), 1000, False, scale=2.0))
    out = asyncio.run(throw.throw(object(), 1000, False, scale=0.3))
    assert out["dry_run"] is True


def test_release_delay_interpolates():
    assert throw.release_delay(1000) == pytest.approx(-0.225)
    assert throw.release_delay(500) == -0.25 and throw.release_delay(2000) == -0.15


def test_wind_up_does_not_turn_the_wrist():
    assert throw.WIND_UP["theta"] == 0      # theta=180 spun J6 by 180 deg before every throw


def test_release_delay_override_validated(monkeypatch):
    for cls in (throw.MotionClient, throw.Arm, throw.Gripper):
        monkeypatch.setattr(cls, "from_robot", staticmethod(lambda r, n: object()))
    with pytest.raises(ValueError):
        asyncio.run(throw.throw(object(), 1100, False, release_delay_s=1.5))
    out = asyncio.run(throw.throw(object(), 1100, False, release_delay_s=0.0))
    assert out["release_delay_s"] == 0.0


def test_negative_delay_opens_before_swing(monkeypatch):
    """Open command first (gripper lags ~0.4 s), swing starts |delay| later."""
    order = []

    class G:
        async def open(self):
            order.append("open")

    class A:
        async def get_joint_positions(self):
            return type("J", (), {"values": [0, 5, -48.4, 0, -83.4, 0]})()

        async def do_command(self, c):
            return {}

        async def move_to_joint_positions(self, j):
            order.append("swing")

    class M:
        async def move(self, *a, **k):
            return True

    monkeypatch.setattr(throw.MotionClient, "from_robot", staticmethod(lambda r, n: M()))
    monkeypatch.setattr(throw.Arm, "from_robot", staticmethod(lambda r, n: A()))
    monkeypatch.setattr(throw.Gripper, "from_robot", staticmethod(lambda r, n: G()))
    out = asyncio.run(throw.throw(object(), 1100, True, scale=1.0, release=True, release_delay_s=-0.05))
    assert order == ["open", "swing"] and out["released"] is True


def test_swing_overrides_and_limits(monkeypatch):
    a = SimpleNamespace(j2=-20.0, j3=None, j5=50.0)
    assert throw.swing_from_args(a) == {1: -20.0, 2: -35, 4: 50.0}
    for cls in (throw.MotionClient, throw.Arm, throw.Gripper):
        monkeypatch.setattr(cls, "from_robot", staticmethod(lambda r, n: object()))
    out = asyncio.run(throw.throw(object(), 1100, False, swing_deg={1: -20, 2: -35, 4: 45}, speed_dps=180, accel_dps2=1100))
    assert out["dry_run"] is True
    with pytest.raises(ValueError):
        asyncio.run(throw.throw(object(), 1100, False, speed_dps=400))


def test_wind_z_validated(monkeypatch):
    for cls in (throw.MotionClient, throw.Arm, throw.Gripper):
        monkeypatch.setattr(cls, "from_robot", staticmethod(lambda r, n: object()))
    with pytest.raises(ValueError):
        asyncio.run(throw.throw(object(), 1100, False, wind_z=1500))
    assert asyncio.run(throw.throw(object(), 1100, False, wind_z=650))["dry_run"] is True


def test_lob_swing_is_inside_limits_and_goes_up():
    start = [0.0, throw.LOB_START_J[0], throw.LOB_START_J[1], 0.0, throw.LOB_START_J[2], 0.0]
    end = list(start)
    for j, d in throw.LOB_SWING.items():
        end[j] += d
    throw.check_limits(start, end)
    assert end[1] > start[1] and end[2] < start[2]      # shoulder back/up, elbow up


def test_lob_dry_run(monkeypatch):
    for cls in (throw.MotionClient, throw.Arm, throw.Gripper):
        monkeypatch.setattr(cls, "from_robot", staticmethod(lambda r, n: object()))
    out = asyncio.run(throw.throw(object(), 500, False, swing_deg=throw.LOB_SWING, start_j=throw.LOB_START_J))
    assert out["dry_run"] is True
    a = SimpleNamespace(j2=None, j3=None, j5=None, lob=True)
    assert throw.swing_from_args(a) == throw.LOB_SWING


def test_lob_default_speed_is_below_the_controller_limit(monkeypatch):
    for cls in (throw.MotionClient, throw.Arm, throw.Gripper):
        monkeypatch.setattr(cls, "from_robot", staticmethod(lambda r, n: object()))
    # peak gripper speed ~ 2.79 m/s * speed/180; 180 deg/s tripped the xArm limit, so stay well under it
    assert throw.LOB_SPEED_DPS <= 130
    out = asyncio.run(throw.throw(object(), 500, False, swing_deg=throw.LOB_SWING, start_j=throw.LOB_START_J))
    assert out["dry_run"] is True


def test_yaw_for_cup_points_at_cup(capsys):
    assert abs(throw.yaw_for_cup(480, 0)) < 1e-9
    assert abs(throw.yaw_for_cup(480, 480) - 45) < 1e-6
    assert throw.yaw_for_cup(480, -85) < 0


def test_yaw_needs_lob_and_is_limited(monkeypatch):
    for cls in (throw.MotionClient, throw.Arm, throw.Gripper):
        monkeypatch.setattr(cls, "from_robot", staticmethod(lambda r, n: object()))
    with pytest.raises(ValueError):
        asyncio.run(throw.throw(object(), 500, False, yaw_deg=10))
    with pytest.raises(ValueError):
        asyncio.run(throw.throw(object(), 500, False, swing_deg=throw.LOB_SWING, start_j=throw.LOB_START_J, yaw_deg=60))
    out = asyncio.run(throw.throw(object(), 500, False, swing_deg=throw.LOB_SWING, start_j=throw.LOB_START_J, yaw_deg=12))
    assert out["dry_run"] is True


def test_far_swing_inside_limits_and_speed_table():
    start = [0.0, throw.FAR_START_J[0], throw.FAR_START_J[1], 0.0, throw.FAR_START_J[2], 0.0]
    end = list(start)
    for j, d in throw.FAR_SWING.items():
        end[j] += d
    throw.check_limits(start, end)
    assert abs(throw.far_speed_for(983) - 180) < 1e-6
    assert 105 < throw.far_speed_for(850) < 120
    assert throw.far_speed_for(5000) == 180.0
    assert throw.far_speed_for(100) >= 100.0
    # open command goes out before the arm starts braking (fingers lag ~0.3 s)
    assert -0.3 < throw.far_release_delay(180) < 0.0


def test_far_dry_run(monkeypatch):
    for cls in (throw.MotionClient, throw.Arm, throw.Gripper):
        monkeypatch.setattr(cls, "from_robot", staticmethod(lambda r, n: object()))
    fp = throw.far_params(850)
    out = asyncio.run(throw.throw(object(), 850, False, 1.0, True, fp["release_delay_s"], fp["swing_deg"],
                                  fp["speed_dps"], fp["accel_dps2"], None, fp["start_j"], -9.1, fp["gripper_speed"]))
    assert out["dry_run"] is True
