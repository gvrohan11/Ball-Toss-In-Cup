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
             direct=False, no_learn=True, cradle=False, cradle_check=False, partial=False, aim_cup=False, fill_cups=False, survey_only=False, cup=None, force_aim=False, go=True, i_checked_with_staff=True)
    d.update(kw)
    return SimpleNamespace(**d)


def test_throws_only_after_a_good_pick(rig, monkeypatch):
    m, events = rig
    thrown = []

    async def pick(robot, ball, go):
        return True

    async def throw(robot, pl, go, release=True, direct=None, partial=False, check_only=False):
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
    assert logged[0]["throw"]["preset"] == "power"          # the default since Oct 30


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

    async def throw(robot, pl, go, release=True, direct=None, partial=False, check_only=False):
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


def test_in_with_cup_number():
    assert pt.parse_landing("in 3") == {"hit": True, "outcome": "in", "cup_number": 3}
    assert pt.parse_landing("in cup 2")["cup_number"] == 2
    assert pt.parse_landing("in #4")["cup_number"] == 4


def fake_survey(monkeypatch, positions):
    """main.survey_cups stand-in: cups at `positions`, nearest first, band applied like the real one."""
    import main as main_mod
    calls = []

    async def survey(motion, robot, go, in_range=None, range_label=None):
        calls.append(range_label)
        cups = [{"pos_mm": p, "sightings": 9, "reachable": in_range(p) if in_range else True} for p in positions]
        return sorted(cups, key=lambda c: (c["pos_mm"][0] ** 2 + c["pos_mm"][1] ** 2))
    monkeypatch.setattr(main_mod, "survey_cups", survey)
    return calls


def recorder(monkeypatch, results):
    """Fake pick (always holds) + throw (records plans) + answers from `results`, in order."""
    plans, logged = [], []

    async def pick(robot, ball, go):
        return True

    async def throw(robot, pl, go, release=True, direct=None, partial=False, check_only=False):
        plans.append(pl)
        return dict(pl)
    answers = iter(results)
    import logger
    monkeypatch.setattr(logger, "log_attempt", lambda doc: logged.append(doc))
    monkeypatch.setattr(pt, "pick", pick)
    monkeypatch.setattr(pt.throw, "throw", throw)
    monkeypatch.setattr(pt, "ask_landing", lambda: next(answers))
    return plans, logged


def test_aim_cup_refuses_when_no_cup_is_in_the_band(rig, monkeypatch):
    fake_survey(monkeypatch, [(2600.0, -135.0, 100.0), (500.0, 0.0, 100.0)])
    plans, _ = recorder(monkeypatch, [])
    asyncio.run(pt.run(args(aim_cup=True, distance=None)))
    assert plans == []


def test_aim_cup_targets_most_straight_ahead_cup_in_the_band_and_logs_it(rig, monkeypatch):
    # 600 mm: too close for POWER's band (from ~76 cm); 880/90 (6 deg left) and 950/-200 (12 deg right) are in it
    # -> the one closest to straight ahead = 880/90; numbered right->left: (950,-200)=1, (600,0)=2, (880,90)=3
    calls = fake_survey(monkeypatch, [(600.0, 0.0, 100.0), (950.0, -200.0, 100.0), (880.0, 90.0, 100.0)])
    plans, logged = recorder(monkeypatch, [{"hit": False, "long_cm": 5, "outcome": "long"}])
    asyncio.run(pt.run(args(aim_cup=True, distance=None)))
    assert "power band" in calls[0]
    assert len(plans) == 1 and plans[0]["yaw_deg"] > 0 and abs(plans[0]["distance_mm"] - 885) < 2
    assert logged[0]["cup"]["dist_mm"] == 885 and logged[0]["cup"]["number"] == 3 and logged[0]["cup"]["of"] == 3


def test_cup_n_picks_that_cup(rig, monkeypatch):
    fake_survey(monkeypatch, [(880.0, 90.0, 100.0), (950.0, -200.0, 100.0)])
    plans, logged = recorder(monkeypatch, [{"hit": True, "outcome": "in"}])
    asyncio.run(pt.run(args(aim_cup=True, distance=None, cup=1)))       # cup 1 = rightmost = (950,-200)
    assert plans[0]["yaw_deg"] < 0 and logged[0]["cup"]["number"] == 1


def test_fill_cups_never_reuses_a_filled_cup_and_counts_lucky_cups(rig, monkeypatch):
    arc = [(950.0, -250.0, 100.0), (980.0, 0.0, 100.0), (950.0, 250.0, 100.0)]   # cradle band
    fake_survey(monkeypatch, arc)
    answers = [{"hit": True, "outcome": "in"},                       # cup 1 (rightmost) filled
               {"hit": True, "outcome": "in", "cup_number": 3},      # aimed at cup 2 (middle), went into cup 3
               {"hit": False, "long_cm": 4, "outcome": "long"},      # cup 2 miss
               {"hit": True, "outcome": "in"}]                       # cup 2 filled
    plans, logged = recorder(monkeypatch, answers)
    asyncio.run(pt.run(args(fill_cups=True, cradle=True, distance=None)))
    aimed = [round(p["yaw_deg"]) for p in plans]
    assert len(plans) == 4
    assert aimed == [-15, 0, 0, 0]         # right to left: cup 1, then cup 2 only (1 and 3 are filled)
    assert {d["cup"]["number"] for d in logged} == {1, 2}
    assert all(p["preset"] == "cradle" for p in plans)


def test_fill_cups_stops_when_balls_run_out(rig, monkeypatch):
    fake_survey(monkeypatch, [(950.0, -250.0, 100.0), (980.0, 0.0, 100.0)])
    plans, _ = recorder(monkeypatch, [{"hit": True, "outcome": "in"}])
    found = iter([(400.0, 0.0, 10.0), None])

    async def find(robot, motion, go):
        return next(found)
    monkeypatch.setattr(pt, "find_ball", find)
    asyncio.run(pt.run(args(fill_cups=True, cradle=True, distance=None)))
    assert len(plans) == 1       # one throw; ball 2 never found, so no second throw


def test_survey_only_never_picks_or_throws(rig, monkeypatch):
    fake_survey(monkeypatch, [(980.0, 0.0, 100.0)])
    plans, _ = recorder(monkeypatch, [])
    async def find(*a):
        raise AssertionError("must not look for a ball")
    monkeypatch.setattr(pt, "find_ball", find)
    asyncio.run(pt.run(args(survey_only=True, cradle=True, distance=None)))
    assert plans == []


def test_no_release_run_never_asks_or_logs(rig, monkeypatch):
    fake_survey(monkeypatch, [(980.0, 0.0, 100.0)])
    plans, logged = recorder(monkeypatch, [])
    monkeypatch.setattr(pt, "ask_landing", lambda: (_ for _ in ()).throw(AssertionError("asked")))
    asyncio.run(pt.run(args(aim_cup=True, cradle=True, no_release=True, distance=None)))
    assert len(plans) == 1 and logged == []


def test_plain_phrases():
    assert pt.parse_landing("hit number 1") == {"hit": True, "outcome": "in", "cup_number": 1}
    assert pt.parse_landing("went in cup 2")["cup_number"] == 2
    assert pt.parse_landing("hit") == {"hit": True, "outcome": "in"}
    r = pt.parse_landing("rim of 4")
    assert r["outcome"] == "rim_out" and r["rim_cup"] == 4 and r["hit"] is False
    assert pt.parse_landing("it dropped")["outcome"] == "dropped"
    assert pt.parse_landing("-12")["outcome"] == "short"
    assert pt.parse_landing("12, -3") == {"hit": False, "long_cm": 12.0, "outcome": "long", "left_cm": -3.0}


def test_default_is_power_and_far_still_available():
    import throw
    assert throw.preset_from_args(args()) == "power"
    assert throw.preset_from_args(args(far=True)) == "far"
    assert throw.preset_from_args(args(cradle=True)) == "cradle"
