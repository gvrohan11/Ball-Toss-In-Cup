"""Offline tests for the full loop in main.py with a fake arm/gripper (no robot needed).

    python -m pytest -q tests
"""
import asyncio
import json
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


@pytest.fixture(autouse=True)
def det_log_dir(tmp_path, monkeypatch):
    """Detection logs from these tests go to a temp folder, not observations/."""
    import detection_log
    monkeypatch.setattr(detection_log, "ROOT", tmp_path / "detections")
    yield tmp_path / "detections"
    detection_log.stop()


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


def one_cup(cup_fn):
    """Adapt a fake 'one cup or None' view to main.find_cups (all cups in the view)."""
    async def find_cups(robot, topdown=False):
        c = await cup_fn(robot, topdown)
        return [] if c is None else [c]
    return find_cups


def run(motion, cup_fn, monkeypatch, cups_fn=None):
    monkeypatch.setattr(main, "find_cups", cups_fn or one_cup(cup_fn))
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


def test_no_cup_never_picks(rig, monkeypatch):
    motion, grip = rig

    async def none(robot, topdown=False):
        return None

    doc, state = run(motion, none, monkeypatch)
    assert state is None and doc["result"]["failure"] == "no_cup_found"
    assert len([m for m in motion.moves if m[2] in (650, 800)]) == len(cup_aim.CUP_SCAN)
    assert "grab" not in grip.events                # cup is found first: no ball picked


def test_cup_remembered_before_pick_and_used_when_out_of_view(rig, monkeypatch):
    motion, grip = rig
    looks = []

    async def visible_until_picked(robot, topdown=False):
        looks.append("grab" in grip.events)
        return None if "grab" in grip.events else (730.0, -66.0, 110.0)

    doc, state = run(motion, visible_until_picked, monkeypatch)
    assert state == "dropped"
    assert not any(looks)                           # never looked for the cup after the pick
    assert (730, -66, 300) in motion.moves and grip.events[-1] == "open"
    assert doc["cup"] == {"pos_mm": (730.0, -66.0, 110.0), "frame": "world",
                          "seen_before_pick": True, "number": 1, "of": 1}


def test_stray_single_sightings_are_ignored(rig, monkeypatch):
    motion, grip = rig
    jumpy = iter([(730.0, -66.0, 110.0), (800.0, 40.0, 110.0), (650.0, -150.0, 110.0)])

    async def cup(robot, topdown=False):
        return next(jumpy, (600.0, 0.0, 110.0))     # 3 one-off glints, then a steady cup

    doc, state = run(motion, cup, monkeypatch)
    assert state == "dropped" and doc["cup"]["pos_mm"] == (600.0, 0.0, 110.0)
    assert len(doc["cups"]) == 1


THREE_CUPS = [(820.0, 150.0, 110.0), (600.0, 0.0, 110.0), (700.0, -200.0, 110.0)]


async def three_cups(robot, topdown=False):
    """Each view sees all three cups, a few mm off each time (different views/frames)."""
    three_cups.n = getattr(three_cups, "n", 0) + 1
    j = (three_cups.n % 3 - 1) * 8
    return [(x + j, y - j, z) for x, y, z in THREE_CUPS]


def test_survey_lists_every_cup_nearest_first(rig, monkeypatch):
    motion, grip = rig
    doc, state = run(motion, None, monkeypatch, cups_fn=three_cups)
    assert state == "dropped"
    found = [c["pos_mm"][:2] for c in doc["cups"]]
    assert len(found) == 3                           # repeat sightings merged, not 21 cups
    for got, want in zip(found, [(600, 0), (700, -200), (820, 150)]):   # nearest first
        assert abs(got[0] - want[0]) < 10 and abs(got[1] - want[1]) < 10
    assert all(c["sightings"] == 3 * len(cup_aim.CUP_SCAN) for c in doc["cups"])
    assert doc["cup"]["number"] == 1 and doc["cup"]["of"] == 3
    assert len([m for m in motion.moves if m[2] in (650, 800)]) == len(cup_aim.CUP_SCAN)  # every view


def test_cup_flag_picks_which_cup(rig, monkeypatch):
    motion, grip = rig
    monkeypatch.setattr(sys, "argv", ["main.py", "--go", "--cup", "3"])
    doc, state = run(motion, None, monkeypatch, cups_fn=three_cups)
    assert state == "dropped" and doc["cup"]["number"] == 3
    x, y, _ = doc["cup"]["pos_mm"]
    assert abs(x - 820) < 10 and abs(y - 150) < 10


def test_cup_number_beyond_survey_never_picks(rig, monkeypatch):
    motion, grip = rig
    monkeypatch.setattr(sys, "argv", ["main.py", "--go", "--cup", "4"])
    doc, state = run(motion, None, monkeypatch, cups_fn=three_cups)
    assert doc["result"]["failure"] == "no_cup_found" and len(doc["cups"]) == 3
    assert "grab" not in grip.events


def test_merge_sightings():
    near = [(600, 0, 110), (610, 5, 110), (590, -5, 110)]      # one cup seen 3x
    other = [(600, 120, 110), (604, 118, 110)]                  # a neighbor ~12 cm away
    cups = main.merge_sightings(near + other + [(900, 300, 110)])   # + a one-off glint
    assert [c["sightings"] for c in cups] == [3, 2]
    assert cups[0]["pos_mm"] == (600, 0, 110)


def test_ten_attempts_in_a_row(rig, monkeypatch):
    import telemetry
    motion, grip = rig
    cups = [(600.0 + 20 * i, -100.0 + 20 * i, 110.0) for i in range(10)]   # cup moved each time
    current = {}

    async def cup(robot, topdown=False):
        return None if "grab" in grip.events[-1:] else current["cup"]

    async def survey(m, robot, go):
        current["cup"] = cups[len(logged)]
        return await real_survey(m, robot, go)

    logged = []
    real_survey = main.survey_cups
    monkeypatch.setattr(main, "find_cups", one_cup(cup))
    monkeypatch.setattr(main, "survey_cups", survey)
    monkeypatch.setattr(main, "TEL", None)        # attempt() sets the global; undo afterwards
    monkeypatch.setattr(main, "Telemetry", lambda r, go: telemetry.Telemetry(None, go))
    monkeypatch.setattr(main, "log_attempt", logged.append)
    monkeypatch.setattr(main, "ask_result", lambda: (True, None))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    results = asyncio.run(main.run_series(object(), motion, True, HOME, 10))
    assert len(results) == len(logged) == 10 and all(r["hit"] for r in results)
    assert main.summary(results) == "SERIES: 10/10 in the cup"
    assert grip.events.count("grab") == 10
    for doc, c in zip(logged, cups):                # each drop went to that run's remembered cup
        assert doc["cup"]["pos_mm"] == c
        p = doc["release"]["release_pose"]
        assert (p["x"], p["y"], p["z"]) == (c[0], c[1], c[2] + drop.RELEASE_ABOVE_RIM_MM)


def test_each_attempt_gets_a_detection_log(rig, monkeypatch, det_log_dir):
    import telemetry
    motion, grip = rig

    async def cup(robot, topdown=False):
        return (730.0, -66.0, 110.0)

    logged = []
    monkeypatch.setattr(main, "find_cups", one_cup(cup))
    monkeypatch.setattr(main, "TEL", None)
    monkeypatch.setattr(main, "Telemetry", lambda r, go: telemetry.Telemetry(None, go))
    monkeypatch.setattr(main, "log_attempt", logged.append)
    monkeypatch.setattr(main, "ask_result", lambda: (True, None))
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    asyncio.run(main.run_series(object(), motion, True, HOME, 2))
    folders = sorted(det_log_dir.iterdir())
    assert len(folders) == 2 and [d["detection_log"] for d in logged] == [str(f) for f in folders]
    events = [json.loads(s)["event"] for s in (folders[0] / "log.jsonl").read_text().splitlines()]
    assert events == ["survey", "target", "result"]


def test_series_stops_on_q(rig, monkeypatch):
    import telemetry
    motion, grip = rig

    async def cup(robot, topdown=False):
        return (730.0, -66.0, 110.0)

    monkeypatch.setattr(main, "find_cups", one_cup(cup))
    monkeypatch.setattr(main, "TEL", None)        # attempt() sets the global; undo afterwards
    monkeypatch.setattr(main, "Telemetry", lambda r, go: telemetry.Telemetry(None, go))
    monkeypatch.setattr(main, "log_attempt", lambda d: None)
    monkeypatch.setattr(main, "ask_result", lambda: (False, {"x": 10, "y": 0}))
    monkeypatch.setattr("builtins.input", lambda prompt="": "q")
    results = asyncio.run(main.run_series(object(), motion, True, HOME, 10))
    assert len(results) == 1 and main.summary(results) == "SERIES: 0/1 in the cup; failures: {'missed': 1}"


class FakeArm:
    def __init__(self):
        self.speed, self.commands = pick_ball.NORMAL_SPEED_DEG_S, []

    async def do_command(self, cmd):
        self.commands.append(cmd)
        self.speed = cmd.get("set_speed", self.speed)
        return {}


@pytest.fixture
def pick_rig(monkeypatch):
    arm, grip = FakeArm(), FakeGripper()

    class SpeedMotion(FakeMotion):
        async def move(self, component_name, destination):
            await super().move(component_name, destination)
            self.moves[-1] += (arm.speed,)
            return True

    motion = SpeedMotion()
    monkeypatch.setattr(pick_ball.Arm, "from_robot", staticmethod(lambda r, n: arm))
    monkeypatch.setattr(pick_ball.Gripper, "from_robot", staticmethod(lambda r, n: grip))
    monkeypatch.setattr(pick_ball.MotionClient, "from_robot", staticmethod(lambda r, n: motion))
    monkeypatch.setattr(pick_ball, "SETTLE_S", 0)
    return arm, grip, motion


def test_pick_goes_down_and_lifts_slowly(pick_rig):
    arm, grip, motion = pick_rig
    assert asyncio.run(pick_ball.pick(object(), (420, 10, 15), True)) is True
    slow, normal = pick_ball.PICK_SPEED_DEG_S, pick_ball.NORMAL_SPEED_DEG_S
    assert motion.moves == [(420, 10, pick_ball.HOVER_Z, normal),     # transit at normal speed
                            (420, 10, pick_ball.GRAB_Z, slow),        # descend slowly
                            (420, 10, pick_ball.HOVER_Z, slow)]       # lift slowly
    assert arm.speed == normal                                        # restored


def test_pick_restores_speed_on_error(pick_rig):
    arm, grip, motion = pick_rig
    motion.unreachable.add((420, 10, pick_ball.GRAB_Z))
    with pytest.raises(RuntimeError):
        asyncio.run(pick_ball.pick(object(), (420, 10, 15), True))
    assert arm.speed == pick_ball.NORMAL_SPEED_DEG_S and "grab" not in grip.events


def test_cup_beyond_reach_is_refused_before_picking(rig, monkeypatch):
    motion, grip = rig

    async def far(robot, topdown=False):
        return (994.0, -133.0, 110.0)               # the 100 cm cup from the Oct 29 run

    doc, state = run(motion, far, monkeypatch)
    assert doc["result"]["failure"] == "cup_out_of_reach"
    assert doc["cups"][0]["reachable"] is False
    assert "grab" not in grip.events                # ball never picked up


def test_default_target_skips_too_far_cups(rig, monkeypatch):
    motion, grip = rig
    monkeypatch.setattr(main, "REACH_MAX_MM", 500)   # every cup (60-83 cm) is now too far
    doc, state = run(motion, None, monkeypatch, cups_fn=three_cups)
    assert doc["result"]["failure"] == "cup_out_of_reach" and "grab" not in grip.events
    monkeypatch.setattr(main, "REACH_MAX_MM", 950)
    assert main.chosen_cup_number([{"reachable": False}, {"reachable": True}]) == 2


def test_planner_failure_at_cup_puts_ball_back(rig, monkeypatch):
    motion, grip = rig
    motion.unreachable.add((900, 0, 300))

    async def cup(robot, topdown=False):
        return (900.0, 0.0, 110.0)                  # within reach on paper, planner says no

    doc, state = run(motion, cup, monkeypatch)
    assert doc["result"]["failure"] == "cup_out_of_reach"
    assert (420, 10, 80) in motion.moves and grip.events[-1] == "open"


def test_terminal_output_is_short_and_readable(rig, monkeypatch, capsys):
    motion, grip = rig
    doc, state = run(motion, None, monkeypatch, cups_fn=three_cups)
    out = capsys.readouterr().out
    assert "{'x'" not in out                         # no raw pose dicts
    assert "view 1/7 (x=550 y=+0 z=650, tilted 30 deg): 3 cup(s)" in out
    assert "TABLE SURVEY: 3 cup(s)" in out
    assert len(out.splitlines()) < 40


def test_without_atlas_attempt_is_saved_as_readable_json(tmp_path, monkeypatch, capsys):
    import logger
    monkeypatch.setattr(logger, "_collection", None)
    monkeypatch.delenv("MONGODB_URI", raising=False)
    monkeypatch.setattr(logger, "LOCAL_DIR", tmp_path / "attempts")
    logger.log_attempt({"result": {"hit": True}, "cup": {"pos_mm": (819.312073773267, -70.9, 110.0)},
                        "telemetry": {"images": [{"jpeg": b"\xff\xd8" * 5000}]}})
    out = capsys.readouterr().out
    assert len(out) < 300 and "xff" not in out and "saved attempt locally, hit=True" in out
    (saved,) = (tmp_path / "attempts").iterdir()
    doc = json.loads(saved.read_text())
    assert doc["cup"]["pos_mm"] == [819.3, -70.9, 110.0]
    assert doc["telemetry"]["images"][0]["jpeg"] == "<10000 bytes jpeg>"


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
    stages = [e["stage"] for e in t["events"]]
    assert stages[0] == "home" and stages[1] == "cup_view_1"
    assert stages.index("cup_remembered") < stages.index("ball_measured") < stages.index("after_pick")
    assert "released" in [e["stage"] for e in t["events"]]
    assert set(t["phases"]) >= {"start", "find_ball", "pick", "cup_scan", "drop"}
    assert any(m["stage"].startswith("cup_view") for m in t["moves"])
    monkeypatch.setattr(logger, "_collection", mongomock.MongoClient()["balltoss"]["attempts"])
    logger.log_attempt(doc)                     # stores cleanly, incl. binary thumbnail
    assert logger._collection.count_documents({"telemetry.events.stage": "released"}) == 1


def test_rim_beyond_reach_is_ignored(monkeypatch):
    """A false rim far past the arm's reach (seen live at x=1241) must not end the scan."""
    import cup_rim

    async def frame(robot):
        return object()

    async def props(self):
        class P:
            class intrinsic_parameters:
                focal_x_px = center_x_px = focal_y_px = center_y_px = 1.0
        return P

    spots = iter([(1241.0, 168.0), (729.0, -86.0)])

    async def table(robot, k, u, v, z):
        return (next(spots)[0:2] + (z,)), (0, 0)

    class Cam:
        async def get_properties(self):
            return await props(self)

    monkeypatch.setattr(cup_aim, "_color_frame", frame)
    monkeypatch.setattr(cup_aim.Camera, "from_robot", staticmethod(lambda r, n: Cam()))
    monkeypatch.setattr(cup_rim, "find_cup_rims",
                        lambda f: [dict(u=1, v=1, box=(0, 0, 1, 1), red_fraction=0.3)] * 2)
    calls = []

    async def pix(robot, k, u, v, z):
        calls.append(1)
        n = len(calls)
        # candidate 1: (1241,168) for centre/left/right; candidate 2: (729,-86), widths ~95 mm
        base = (1241.0, 168.0) if n <= 3 else (729.0, -86.0)
        dx = {1: 0, 2: -47, 3: 47, 4: 0, 5: -47, 6: 47}[n]
        return (base[0] + dx, base[1], z), (0, 0)

    monkeypatch.setattr(cup_aim, "_pixel_to_table", pix)
    cup = asyncio.run(cup_aim.find_cup_by_rim(object()))
    assert cup is not None and round(cup[0]) == 729 - 52
