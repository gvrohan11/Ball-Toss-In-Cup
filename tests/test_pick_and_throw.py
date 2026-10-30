import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pick_and_throw as pt  # noqa: E402


class Motion:
    def __init__(self):
        self.moves = []

    async def move(self, component_name, destination):
        self.moves.append(destination.pose)
        return True


class Robot:
    async def close(self):
        pass


@pytest.fixture
def rig(monkeypatch):
    m, events = Motion(), []

    async def connect():
        return Robot()

    async def release(robot, motion, go):
        events.append("release_check")

    async def find(robot, motion, go):
        return (400.0, 0.0, 10.0)

    async def back(robot, motion, ball, go):
        events.append("put_back")

    monkeypatch.setattr(pt, "connect", connect)
    monkeypatch.setattr(pt.MotionClient, "from_robot", staticmethod(lambda r, n: m))
    monkeypatch.setattr(pt, "release_if_holding", release)
    monkeypatch.setattr(pt, "find_ball", find)
    monkeypatch.setattr(pt, "put_ball_back", back)
    monkeypatch.setattr(pt, "ask_landing", lambda: {"hit": False, "long_cm": -30})
    return m, events


def args(**kw):
    d = dict(distance=1100.0, scale=1.0, no_release=False, release_delay=None, speed=None, accel=None, j2=None, j3=None, j5=None, wind_z=None, lob=False, yaw=0.0, aim_cup=False, go=True, i_checked_with_staff=True)
    d.update(kw)
    return SimpleNamespace(**d)


def test_throws_only_after_a_good_pick(rig, monkeypatch):
    m, events = rig
    thrown = []

    async def pick(robot, ball, go):
        return True

    async def throw(robot, d, go, scale, release, delay=None, swing=None, speed=None, accel=None, wind_z=None, start_j=None, yaw=0.0, gripper_speed=None, direct=False):
        thrown.append((d, go))
        return {"distance_mm": d}

    logged = []
    import logger
    monkeypatch.setattr(logger, "log_attempt", lambda doc: logged.append(doc))
    monkeypatch.setattr(pt, "pick", pick)
    monkeypatch.setattr(pt.throw, "throw", throw)
    asyncio.run(pt.run(args()))
    assert thrown == [(1100.0, True)]
    assert logged and logged[0]["phase"] == "throw" and logged[0]["result"]["long_cm"] == -30


def test_missed_pick_never_throws(rig, monkeypatch):
    m, events = rig

    async def pick(robot, ball, go):
        return False

    async def throw(*a, **k):
        raise AssertionError("must not throw without the ball")

    monkeypatch.setattr(pt, "pick", pick)
    monkeypatch.setattr(pt.throw, "throw", throw)
    asyncio.run(pt.run(args()))


def test_failure_after_pick_puts_ball_back(rig, monkeypatch):
    m, events = rig

    async def pick(robot, ball, go):
        return True

    async def throw(*a, **k):
        raise RuntimeError("swing failed")

    monkeypatch.setattr(pt, "pick", pick)
    monkeypatch.setattr(pt.throw, "throw", throw)
    with pytest.raises(RuntimeError):
        asyncio.run(pt.run(args()))
    assert "put_back" in events


def test_refuses_live_without_staff_flag(rig, monkeypatch):
    m, events = rig
    seen = []

    async def pick(robot, ball, go):
        seen.append(go)
        return None

    async def throw(robot, d, go, scale, release, delay=None, swing=None, speed=None, accel=None, wind_z=None, start_j=None, yaw=0.0, gripper_speed=None, direct=False):
        seen.append(("throw", go))
        return {}

    monkeypatch.setattr(pt, "pick", pick)
    monkeypatch.setattr(pt.throw, "throw", throw)
    asyncio.run(pt.run(args(i_checked_with_staff=False)))
    assert seen == [None, ("throw", False)] or all(x in (None, False, ("throw", False)) for x in seen)
    assert m.moves == []                      # dry run: nothing moved


def test_throw_doc_with_int_keys_is_stored(monkeypatch):
    """Live 17:54: swing_deg {2: -35, 4: 45} made Atlas reject the throw doc ("only string keys")."""
    import logger
    import mongomock
    monkeypatch.setattr(logger, "_collection", mongomock.MongoClient()["balltoss"]["attempts"])
    doc = logger.log_attempt({"phase": "throw", "throw": {"swing_deg": {2: -35, 4: 45}}})
    assert doc["throw"]["swing_deg"] == {"2": -35, "4": 45}
    assert logger._collection.count_documents({"phase": "throw"}) == 1


def test_landing_answer_accepts_units(monkeypatch):
    for typed, want in (("10", 10.0), ("-10", -10.0), ("10cms", 10.0), ("short -25 cm", -25.0)):
        monkeypatch.setattr("builtins.input", lambda prompt="", typed=typed: typed)
        assert pt.ask_landing()["long_cm"] == want
    monkeypatch.setattr("builtins.input", lambda prompt="": "in")
    assert pt.ask_landing() == {"hit": True}


def test_aim_cup_refuses_when_cup_is_out_of_range(rig, monkeypatch):
    m, events = rig
    import main as main_mod
    async def scan(motion, robot, go):
        return (836.0, -135.0, 100.0)
    monkeypatch.setattr(main_mod, "scan_for_cup", scan)
    thrown = []
    async def throw(*a, **k):
        thrown.append(1)
        return {}
    monkeypatch.setattr(pt.throw, "throw", throw)
    asyncio.run(pt.run(args(lob=True, aim_cup=True)))
    assert thrown == []
