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
    assert throw.release_delay(1000) == pytest.approx(0.18)
    assert throw.release_delay(500) == 0.16 and throw.release_delay(2000) == 0.24


def test_wind_up_does_not_turn_the_wrist():
    assert throw.WIND_UP["theta"] == 0      # theta=180 spun J6 by 180 deg before every throw
