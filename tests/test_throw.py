import asyncio
import os
import sys

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
