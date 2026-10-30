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

    async def scan(*a):
        events.append("scan")
        return (900.0, 0.0, 110.0)

    async def align(robot, bearing, go):
        events.append("align")
        return bearing

    import main
    monkeypatch.setattr(main, "scan_for_cup", scan)
    monkeypatch.setattr(pt, "align_with_cup", align)
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

    async def throw(robot, d, go, scale, release, delay=None, swing=None, speed=None, accel=None, wind_z=None, start_j=None, yaw=0.0, gripper_speed=None, direct=False, aligned_base_deg=None):
        thrown.append((d, go))
        return {"distance_mm": d}

    logged = []
    import logger
    monkeypatch.setattr(logger, "log_attempt", lambda doc: logged.append(doc))
    monkeypatch.setattr(pt, "pick", pick)
    monkeypatch.setattr(pt.throw, "throw", throw)
    asyncio.run(pt.run(args()))
    assert thrown == [(900.0, True)]
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

    async def throw(robot, d, go, scale, release, delay=None, swing=None, speed=None, accel=None, wind_z=None, start_j=None, yaw=0.0, gripper_speed=None, direct=False, aligned_base_deg=None):
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


# ---- required cup scan + joint-0 alignment -------------------------------------------------
class FakeArm:
    def __init__(self, joints):
        self.j = list(joints)
        self.cmds = []

    async def get_joint_positions(self):
        return SimpleNamespace(values=list(self.j))

    async def do_command(self, c):
        self.cmds.append(c)

    async def move_to_joint_positions(self, jp, extra=None):
        self.j = list(jp.values)

    async def get_end_position(self):
        import math
        r = 350.0   # home: gripper 350 mm out along the arm's heading
        a = math.radians(self.j[0])
        return SimpleNamespace(x=r * math.cos(a), y=r * math.sin(a), z=445.0)


def test_cup_bearing_is_world_heading():
    yaw, d = pt.cup_bearing((800.0, 800.0, 110.0))
    assert abs(yaw - 45) < 1e-9 and abs(d - 800 * 2 ** 0.5) < 1e-9
    assert pt.cup_bearing((900.0, -100.0, 110.0))[0] < 0       # cup toward -y -> negative heading


def test_align_turns_only_joint0_to_absolute_bearing(monkeypatch):
    arm = FakeArm([2.0, 13.2, -44.1, 0.0, -57.3, 0.1])        # joint 0 slightly off zero at home
    monkeypatch.setattr(pt.Arm, "from_robot", staticmethod(lambda r, n: arm))
    j0 = asyncio.run(pt.align_with_cup(object(), -6.3, True))
    assert abs(j0 - (-6.3)) < 1e-9                            # absolute heading, not 2.0 + (-6.3)
    assert arm.j[1:] == [13.2, -44.1, 0.0, -57.3, 0.1]        # other joints untouched


def test_align_refuses_outside_sector():
    with pytest.raises(ValueError):
        asyncio.run(pt.align_with_cup(object(), 60.0, True))


def test_scan_uses_throw_reach_and_restores_it(monkeypatch):
    import cup_aim
    import main
    seen = []

    async def scan(motion, robot, go):
        seen.append(cup_aim.MAX_CUP_REACH_MM)
        return (1000.0, 0.0, 110.0)

    monkeypatch.setattr(main, "scan_for_cup", scan)
    before = cup_aim.MAX_CUP_REACH_MM
    assert asyncio.run(pt.scan_for_throw_cup(None, None, False)) == (1000.0, 0.0, 110.0)
    assert seen == [pt.THROW_SCAN_REACH_MM] and cup_aim.MAX_CUP_REACH_MM == before


def test_rotate_about_base_keeps_radius_height_and_tilt():
    import math
    import throw
    p = throw.rotate_about_base(throw.WIND_UP, 30)
    assert abs(math.hypot(p["x"], p["y"]) - math.hypot(throw.WIND_UP["x"], throw.WIND_UP["y"])) < 1e-9
    assert abs(math.degrees(math.atan2(p["y"], p["x"])) - 30) < 1e-9
    assert p["z"] == throw.WIND_UP["z"] and p["o_z"] == throw.WIND_UP["o_z"] and p["theta"] == throw.WIND_UP["theta"]
    assert abs(math.degrees(math.atan2(p["o_y"], p["o_x"])) - (180 + 30 - 360)) < 1e-9   # tilt turned too


def test_pipeline_order(rig, monkeypatch):
    m, events = rig

    async def pick(robot, ball, go):
        events.append("pick")
        return True

    async def throw(robot, d, go, *a, aligned_base_deg=None, **k):
        events.append(("throw", aligned_base_deg))
        return {}

    async def find(robot, motion, go):
        events.append("find_ball")
        return (400.0, 0.0, 10.0)

    monkeypatch.setattr(pt, "find_ball", find)
    monkeypatch.setattr(pt, "pick", pick)
    monkeypatch.setattr(pt.throw, "throw", throw)
    monkeypatch.setattr(pt, "ask_landing", lambda: None)
    import logger
    monkeypatch.setattr(logger, "log_attempt", lambda d: None)
    asyncio.run(pt.run(args()))
    order = [e for e in events if e != "release_check"]
    assert order == ["scan", "find_ball", "pick", "align", ("throw", 0.0)]
