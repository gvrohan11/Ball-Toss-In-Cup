"""Offline tests for the full loop in main.py with a fake arm/gripper (no robot needed).

    python -m pytest -q tests
"""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cup_aim  # noqa: E402
import drop  # noqa: E402
import main  # noqa: E402
import pick_ball  # noqa: E402
from viam.components import gripper as gripper_mod  # noqa: E402

HOME = dict(x=350, y=0, z=445, o_x=0, o_y=0, o_z=-1, theta=180)


class FakeMotion:
    """Records moves; refuses poses listed in `unreachable` (by (x, y, z))."""

    def __init__(self, unreachable=()):
        self.moves, self.unreachable = [], set(unreachable)

    async def move(self, component_name, destination):
        p = destination.pose
        key = (round(p.x), round(p.y), round(p.z))
        if key in self.unreachable:
            raise RuntimeError("all IK solutions failed")
        self.moves.append(key)
        return True


class FakeGripper:
    def __init__(self, pos=850):
        self.pos, self.events = pos, []

    async def do_command(self, cmd):
        return {"pos": self.pos}

    async def open(self):
        self.pos = 850
        self.events.append("open")

    async def grab(self):
        self.pos = 370
        self.events.append("grab")
        return True


@pytest.fixture
def rig(monkeypatch):
    motion, grip = FakeMotion(), FakeGripper()
    monkeypatch.setattr(gripper_mod.Gripper, "from_robot", staticmethod(lambda r, n: grip))
    monkeypatch.setattr(drop.Gripper, "from_robot", staticmethod(lambda r, n: grip))
    monkeypatch.setattr(drop.MotionClient, "from_robot", staticmethod(lambda r, n: motion))

    async def find_ball(robot, m, go):
        return (420.0, 10.0, 15.0)

    async def pick(robot, ball, go):
        await grip.grab()
        return True

    monkeypatch.setattr(main, "find_ball", find_ball)
    monkeypatch.setattr(main, "pick", pick)
    return motion, grip


def run(motion, cup_fn, monkeypatch):
    monkeypatch.setattr(main, "find_cup", cup_fn)
    return asyncio.run(main.run_once(object(), motion, True, HOME, 0))


def test_happy_path_drops_over_cup(rig, monkeypatch):
    motion, grip = rig

    async def cup(robot, topdown=False):
        return (730.0, -66.0, 110.0)

    doc, state = run(motion, cup, monkeypatch)
    assert state == "dropped"
    assert (730, -66, 300) in motion.moves          # release pose 19 cm above the rim
    assert grip.events[-1] == "open"                # released
    assert doc["release"]["release_pose"]["o_x"] == 0.7   # far cup -> tilted reach


def test_no_cup_puts_ball_back(rig, monkeypatch):
    motion, grip = rig

    async def none(robot, topdown=False):
        return None

    doc, state = run(motion, none, monkeypatch)
    assert state is None and doc["result"]["failure"] == "no_cup_found"
    assert len([m for m in motion.moves if m[2] in (650, 800)]) == len(cup_aim.CUP_SCAN)
    assert (420, 10, 80) in motion.moves            # set back down at the pick spot
    assert grip.events[-1] == "open"


def test_cup_out_of_reach_puts_ball_back(rig, monkeypatch):
    motion, grip = rig
    motion.unreachable.add((1300, 0, 300))

    async def far(robot, topdown=False):
        return (1300.0, 0.0, 110.0)

    doc, state = run(motion, far, monkeypatch)
    assert doc["result"]["failure"] == "cup_out_of_reach"
    assert (420, 10, 80) in motion.moves and grip.events[-1] == "open"


def test_unreachable_scan_views_are_skipped(rig, monkeypatch):
    motion, grip = rig
    for v in cup_aim.CUP_SCAN[2:6]:
        motion.unreachable.add((v["x"], v["y"], v["z"]))
    seen = []

    async def only_top(robot, topdown=False):
        seen.append(topdown)
        return (600.0, 0.0, 110.0) if topdown else None

    doc, state = run(motion, only_top, monkeypatch)
    assert state == "dropped" and seen[-1] is True


def test_start_check_sets_down_held_ball(rig, monkeypatch):
    motion, grip = rig
    grip.pos = 370                                   # left holding from an earlier run
    asyncio.run(main.release_if_holding(object(), motion, True))
    assert (420, 0, 80) in motion.moves and grip.events == ["open"]


def test_empty_gripper_start_check_just_opens(rig):
    motion, grip = rig
    grip.pos = -2                                     # real closed-empty reading
    asyncio.run(main.release_if_holding(object(), motion, True))
    assert motion.moves == [] and grip.events == ["open"]


def test_failed_release_move_keeps_gripper_closed(rig, monkeypatch):
    motion, grip = rig

    class NoGo(FakeMotion):
        async def move(self, component_name, destination):
            return False

    monkeypatch.setattr(drop.MotionClient, "from_robot", staticmethod(lambda r, n: NoGo()))
    with pytest.raises(RuntimeError):
        asyncio.run(drop.drop_in_cup(object(), (730, -66, 110), True))
    assert "open" not in grip.events


def test_autotune_is_capped():
    p = drop.release_pose((730, -66, 110), (-80, 5))
    assert p["x"] == 700 and p["y"] == -61


def test_wait_gripper_rejects_bad_reading():
    class Bad:
        async def do_command(self, c):
            return {"pos": None}

    with pytest.raises(RuntimeError):
        asyncio.run(pick_ball.wait_gripper(Bad()))


def test_telemetry_records_timeline(rig, monkeypatch):
    import mongomock
    import logger
    import telemetry

    motion, grip = rig

    async def cup(robot, topdown=False):
        return (730.0, -66.0, 110.0)

    tel = telemetry.Telemetry(None, True)    # robot=None: timeline only, no hardware reads
    monkeypatch.setattr(main, "TEL", tel)
    tel.images.append({"t": 0.1, "stage": "cup_view", "jpeg": b"\xff\xd8fake"})
    doc, state = run(motion, cup, monkeypatch)
    doc["telemetry"] = tel.as_doc()
    t = doc["telemetry"]
    assert [e["stage"] for e in t["events"]][:2] == ["home", "ball_measured"]
    assert "released" in [e["stage"] for e in t["events"]]
    assert set(t["phases"]) >= {"start", "find_ball", "pick", "cup_scan", "drop"}
    assert any(m["stage"].startswith("cup_view") for m in t["moves"])
    monkeypatch.setattr(logger, "_collection", mongomock.MongoClient()["balltoss"]["attempts"])
    logger.log_attempt(doc)                     # stores cleanly, incl. binary thumbnail
    assert logger._collection.count_documents({"telemetry.events.stage": "released"}) == 1
