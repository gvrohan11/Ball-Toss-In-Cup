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
    monkeypatch.setattr(pt, "ask_landing", lambda: {"hit": False, "long_cm": -30, "outcome": "short"})
    return m, events


def args(**kw):
    d = dict(distance=900.0, scale=1.0, no_release=False, release_delay=None, speed=None, yaw=0.0, power=False,
             direct=False, no_learn=True, aim_cup=False, force_aim=False, go=True, i_checked_with_staff=True)
    d.update(kw)
    return SimpleNamespace(**d)


def test_throws_only_after_a_good_pick(rig, monkeypatch):
    m, events = rig
    thrown = []

    async def pick(robot, ball, go):
        return True

    async def throw(robot, pl, go, release=True, direct=None):
        thrown.append((pl["distance_mm"], go))
        return dict(pl)

    logged = []
    import logger
    monkeypatch.setattr(logger, "log_attempt", lambda doc: logged.append(doc))
    monkeypatch.setattr(pt, "pick", pick)
    monkeypatch.setattr(pt.throw, "throw", throw)
    asyncio.run(pt.run(args()))
    assert thrown == [(900, True)]
    assert logged and logged[0]["phase"] == "throw" and logged[0]["result"]["long_cm"] == -30
    assert logged[0]["throw"]["preset"] == "far"


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

    async def throw(robot, pl, go, release=True, direct=None):
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


def test_landing_answer_accepts_units_and_outcomes():
    for typed, want in (("10", 10.0), ("-10", -10.0), ("10cms", 10.0), ("short -25 cm", -25.0)):
        assert pt.parse_landing(typed)["long_cm"] == want
    assert pt.parse_landing("in") == {"hit": True, "outcome": "in"}
    assert pt.parse_landing("rim")["outcome"] == "rim_out"
    assert pt.parse_landing("0")["outcome"] == "rim_out"
    assert pt.parse_landing("drop")["outcome"] == "dropped"
    r = pt.parse_landing("-8,3")
    assert r["long_cm"] == -8 and r["left_cm"] == 3 and r["outcome"] == "short"
    assert pt.parse_landing("") is None


def test_aim_cup_refuses_when_cup_is_out_of_range(rig, monkeypatch):
    m, events = rig
    import main as main_mod
    async def scan(motion, robot, go):
        return (2600.0, -135.0, 100.0)          # beyond FAR and POWER
    monkeypatch.setattr(main_mod, "scan_for_cup", scan)
    thrown = []
    async def throw(*a, **k):
        thrown.append(1)
        return {}
    monkeypatch.setattr(pt.throw, "throw", throw)
    asyncio.run(pt.run(args(aim_cup=True)))
    assert thrown == []


def test_aim_cup_aims_and_logs_the_cup(rig, monkeypatch):
    m, events = rig
    import main as main_mod
    async def scan(motion, robot, go):
        return (880.0, 90.0, 100.0)
    monkeypatch.setattr(main_mod, "scan_for_cup", scan)
    async def pick(robot, ball, go):
        return True
    plans, logged = [], []
    async def throw(robot, pl, go, release=True, direct=None):
        plans.append(pl)
        return dict(pl)
    import logger
    monkeypatch.setattr(logger, "log_attempt", lambda doc: logged.append(doc))
    monkeypatch.setattr(pt, "pick", pick)
    monkeypatch.setattr(pt.throw, "throw", throw)
    asyncio.run(pt.run(args(aim_cup=True, distance=None)))
    assert plans and plans[0]["yaw_deg"] > 0 and abs(plans[0]["distance_mm"] - 885) < 2
    assert logged[0]["cup"]["dist_mm"] == 885
