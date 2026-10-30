"""Detection log: frames + candidates saved per attempt (no robot needed).

    python -m pytest -q tests
"""
import asyncio
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cup_aim  # noqa: E402
import detection_log  # noqa: E402
import pick_ball  # noqa: E402

K = SimpleNamespace(focal_x_px=600.0, focal_y_px=600.0, center_x_px=640.0, center_y_px=360.0)


class FakeCam:
    async def get_properties(self):
        return SimpleNamespace(intrinsic_parameters=K)


@pytest.fixture(autouse=True)
def rim_only(monkeypatch):
    """These tests drive the rim / ball detectors; YOLO has its own tests."""
    monkeypatch.setattr(cup_aim, "USE_YOLO", False)


@pytest.fixture
def log(tmp_path):
    yield detection_log.start(tmp_path)
    detection_log.stop()


def lines(log):
    return [json.loads(s) for s in (log.dir / "log.jsonl").read_text().splitlines()]


@pytest.fixture
def fake_view(monkeypatch):
    """A blank frame; pixel (u, v) maps to world (u, v, z) so widths are box widths."""
    frame = np.zeros((720, 1280, 3), np.uint8)

    async def color_frame(robot):
        return frame

    async def to_table(robot, k, u, v, z):
        return (float(u), float(v), z), (0.0, 0.0)

    monkeypatch.setattr(cup_aim, "_color_frame", color_frame)
    monkeypatch.setattr(cup_aim, "_pixel_to_table", to_table)
    monkeypatch.setattr(cup_aim.Camera, "from_robot", staticmethod(lambda r, n: FakeCam()))
    return frame


def test_cup_frames_and_candidates_are_saved(log, fake_view, monkeypatch):
    import cup_rim
    rims = [dict(u=500, v=300, red_fraction=0.2, score=9.0, box=(450, 280, 550, 320)),   # 100 mm
            dict(u=900, v=300, red_fraction=0.1, score=5.0, box=(750, 280, 1050, 320))]  # 300 mm
    monkeypatch.setattr(cup_rim, "find_cup_rims", lambda bgr: rims)
    detection_log.context("cup_view_2", view={"x": 550})

    cups = asyncio.run(cup_aim.find_cups(object(), topdown=False))
    assert len(cups) == 1
    (rec,) = lines(log)
    assert rec["stage"] == "cup_view_2" and rec["view"] == {"x": 550}
    assert rec["detector"] == "cup_rim" and rec["found"] == 1
    kept, rejected = rec["items"]
    assert kept["kept"] and kept["world_mm"] == list(cups[0])
    assert not rejected["kept"] and rejected["why"] == "300 mm wide"
    assert (log.dir / rec["image"]).stat().st_size > 0
    assert (log.dir / rec["annotated"]).stat().st_size > 0


def test_red_body_fallback_is_logged(log, fake_view, monkeypatch):
    import cup_rim
    monkeypatch.setattr(cup_rim, "find_cup_rims", lambda bgr: [])
    big = SimpleNamespace(x_min=600, y_min=200, x_max=700, y_max=400, confidence=0.9)
    tiny = SimpleNamespace(x_min=10, y_min=10, x_max=20, y_max=20, confidence=0.5)

    class Det:
        async def get_detections_from_camera(self, cam):
            return [big, tiny]

    monkeypatch.setattr(cup_aim.VisionClient, "from_robot", staticmethod(lambda r, n: Det()))
    cups = asyncio.run(cup_aim.find_cups(object(), topdown=False))
    (rec,) = lines(log)
    assert rec["detector"] == "cup_rim+detector" and rec["found"] == len(cups) == 1
    assert {(i["kind"], i["kept"]) for i in rec["items"]} == {("red_body", True), ("red_body", False)}


def test_ball_candidates_and_choice_are_logged(log, fake_view, monkeypatch):
    import ball_finder
    balls = [dict(u=600, v=300, r=30, touching=False), dict(u=900, v=300, r=30, touching=True)]
    monkeypatch.setattr(ball_finder, "find_balls", lambda frame, r: balls)
    monkeypatch.setattr(pick_ball, "BALL_COLOR", "orange")
    monkeypatch.setattr(pick_ball.Camera, "from_robot", staticmethod(lambda r, n: FakeCam()))

    async def frame(cam):
        return fake_view

    class Robot:
        async def transform_pose(self, p, dest):
            return SimpleNamespace(pose=SimpleNamespace(x=0, y=0, z=400))

    monkeypatch.setattr(pick_ball, "_frame", frame)
    detection_log.context("find_ball")
    pos = asyncio.run(pick_ball.locate_ball(Robot()))
    (rec,) = lines(log)
    assert rec["detector"] == "ball_orange" and rec["closeup"] is False
    chosen, touching = rec["items"]
    assert chosen["chosen"] and chosen["world_mm"] == list(pos)
    assert not touching["kept"] and touching["why"] == "touching another ball"


def test_ball_in_a_cup_is_never_picked(log, fake_view, monkeypatch):
    """--fill-cups: a ball already dropped in a cup can be in view; pick the free one."""
    import ball_finder
    balls = [dict(u=600, v=300, r=30, touching=False), dict(u=300, v=300, r=30, touching=False)]
    monkeypatch.setattr(ball_finder, "find_balls", lambda frame, r: balls)
    monkeypatch.setattr(pick_ball, "BALL_COLOR", "orange")
    monkeypatch.setattr(pick_ball, "AVOID_XY", [(610.0, 290.0)])     # a surveyed cup at the 1st ball
    monkeypatch.setattr(pick_ball.Camera, "from_robot", staticmethod(lambda r, n: FakeCam()))

    async def frame(cam):
        return fake_view

    class Robot:
        async def transform_pose(self, p, dest):
            return SimpleNamespace(pose=SimpleNamespace(x=0, y=0, z=400))

    monkeypatch.setattr(pick_ball, "_frame", frame)
    pos = asyncio.run(pick_ball.locate_ball(Robot()))
    assert pos[:2] == (300.0, 300.0)
    in_cup, free = lines(log)[0]["items"]
    assert in_cup["why"] == "in or next to a cup" and free["chosen"]


def test_ball_past_the_search_line_is_not_picked(log, fake_view, monkeypatch):
    """Cups ~92 cm out: balls farther than 77 cm (15 cm short of the nearest cup) are left."""
    import ball_finder
    balls = [dict(u=800, v=250, r=30, touching=False), dict(u=420, v=20, r=30, touching=False)]   # 84 / 42 cm
    monkeypatch.setattr(ball_finder, "find_balls", lambda frame, r: balls)
    monkeypatch.setattr(pick_ball, "BALL_COLOR", "orange")
    monkeypatch.setattr(pick_ball, "AVOID_XY", [(915.0, 70.0), (914.0, -85.0)])
    monkeypatch.setattr(pick_ball.Camera, "from_robot", staticmethod(lambda r, n: FakeCam()))

    async def frame(cam):
        return fake_view

    class Robot:
        async def transform_pose(self, p, dest):
            return SimpleNamespace(pose=SimpleNamespace(x=0, y=0, z=400))

    monkeypatch.setattr(pick_ball, "_frame", frame)
    assert asyncio.run(pick_ball.locate_ball(Robot()))[:2] == (420.0, 20.0)
    far, near = lines(log)[0]["items"]
    assert far["why"] == "past the search line (too near the cups)" and near["chosen"]


def test_events_and_summary(log):
    detection_log.context("survey")
    detection_log.event("survey", cups=[{"pos_mm": (600, 0, 110), "sightings": 4}])
    text = detection_log.summarize(log.dir)
    assert "SURVEY: 1 cup(s)" in text and "(600,+0)" in text


def test_log_numbers_are_rounded(log):
    detection_log.event("target", cup_mm=(993.5606224470915, -133.37903472222737, 110.0))
    assert lines(log)[0]["cup_mm"] == [993.6, -133.4, 110.0]


def test_summary_is_readable(log):
    detection_log.context("cup_view_3")
    detection_log.record("cup_rim", None, [
        {"kind": "rim", "kept": True, "world_mm": (1000.4, -121.2, 110)},
        {"kind": "rim", "kept": False, "why": "66 mm wide"}])
    detection_log.context("survey")
    detection_log.event("survey", cups=[{"pos_mm": (994, -133, 110), "sightings": 18,
                                         "reachable": False}], sightings=[])
    detection_log.event("target", cup_mm=(994, -133, 110), number=1, of=1, refused="too far")
    detection_log.event("result", hit=False, failure="cup_out_of_reach")
    text = detection_log.summarize(log.dir)
    assert "(1000,-121)  rejected: 66 mm wide" in text
    assert "100 cm out, seen 18x  <- too far to drop" in text
    assert "REFUSED: too far" in text and "RESULT: MISS cup_out_of_reach" in text


def test_survey_view_saves_one_frame(log):
    frame, depth = np.zeros((40, 60, 3), np.uint8), np.ones((40, 60), np.uint16)
    detection_log.context("cup_view_1", max_frames=1)
    for _ in range(3):                                        # 3 readings per view
        detection_log.record("cup_rim+yolo", frame, [], depth=depth)
    detection_log.context("find_ball")                        # ball stages: every frame
    for _ in range(2):
        detection_log.record("ball_orange", frame, [])
    recs = lines(log)
    assert len(recs) == 5                                     # every reading is still logged
    assert [bool(r.get("annotated")) for r in recs] == [True, False, False, True, True]
    assert recs[0]["depth"] == "0001_cup_view_1_depth.png"
    saved = sorted(p.name for p in log.dir.glob("*_cup_view_1*"))
    assert saved == ["0001_cup_view_1.jpg", "0001_cup_view_1_depth.png", "0001_cup_view_1_det.jpg"]
    assert "[img]" in detection_log.summarize(log.dir)


def test_annotate_draws_on_a_copy():
    frame = np.zeros((100, 200, 3), np.uint8)
    out = detection_log.annotate(frame, [{"kind": "rim", "box": (10, 20, 60, 50), "kept": True,
                                          "world_mm": (1, 2, 3)},
                                         {"kind": "ball", "circle": (150, 50, 20), "kept": False,
                                          "why": "touching"}], "t")
    assert out.shape == frame.shape and out.any() and not frame.any()


def test_logging_problems_never_stop_a_run(log, monkeypatch):
    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(log, "record", broken)
    detection_log.record("cup_rim", None, [])            # no exception


def test_no_log_started_is_a_no_op(tmp_path):
    detection_log.stop()
    detection_log.record("cup_rim", np.zeros((4, 4, 3), np.uint8), [])
    detection_log.event("survey")
    assert not any(tmp_path.iterdir())
