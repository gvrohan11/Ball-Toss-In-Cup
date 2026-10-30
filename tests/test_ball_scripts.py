"""scripts/calibrate_pick.py and scripts/debug_ball.py, offline (no robot)."""
import asyncio
import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ball_finder  # noqa: E402
import detection_log  # noqa: E402
import pick_ball  # noqa: E402


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


calib = _load("calibrate_pick")
debug_ball = _load("debug_ball")


@pytest.fixture
def fake_arm(monkeypatch, tmp_path):
    """Camera that reports every ball (TRUE_ERR) off; a gripper that always holds."""
    true_err = (-20.0, 25.0)                   # truth - camera, what calibration must find
    placed = {}
    monkeypatch.setattr(detection_log, "ROOT", tmp_path)
    monkeypatch.setattr(calib, "_motion", lambda robot: object())

    async def move(motion, x, y, z, go):
        placed.setdefault("moves", []).append((x, y, z))

    async def set_down(robot, at, go):
        placed["ball"] = at                    # the ball now really sits at `at`

    async def closeup(robot, go):
        x, y = placed["ball"]
        return (x - true_err[0], y - true_err[1], 10.0)

    async def find_ball(robot, motion, go):
        placed["ball"] = (390.0, 10.0)
        return (390.0, 10.0, 10.0)

    picks = []

    async def pick(robot, ball, go):
        picks.append(ball)
        return True

    monkeypatch.setattr(pick_ball, "move", move)
    monkeypatch.setattr(pick_ball, "closeup", closeup)
    monkeypatch.setattr(pick_ball, "find_ball", find_ball)
    monkeypatch.setattr(pick_ball, "pick", pick)
    monkeypatch.setattr(calib, "set_down", set_down)
    return true_err, picks


def test_calibration_recovers_the_camera_error(fake_arm, capsys):
    true_err, picks = fake_arm
    rows, correction = asyncio.run(calib.calibrate(object(), calib.SPOTS[:4], True))
    assert len(rows) == 4 and correction == true_err
    assert all(r["offset_mm"] == true_err for r in rows)
    # re-picks use the learned correction: they target exactly where the ball was set down
    for (x, y, _), r in zip(picks[1:], rows):
        assert (round(x), round(y)) == r["spot"]
    summary = calib.report(rows, pick_ball.CAM_CORRECTION_MM)
    out = capsys.readouterr().out
    assert summary["median_offset_mm"] == true_err and summary["spread_mm"] == 0
    assert "-> set in pick_ball.py:  CAM_CORRECTION_MM = ... (-15, 20) -> (-20, 25)" in out


def test_recommend_flags_inconsistent_offsets(capsys):
    rows = [{"spot": (400, 0), "offset_mm": o} for o in [(-15, 20), (-5, 32), (-28, 8)]]
    calib.report(rows, (-15, 20))
    assert "not consistent" in capsys.readouterr().out


def test_debug_panel_and_offline_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(ball_finder, "USE_YOLO", False)
    monkeypatch.setattr(debug_ball, "OUT", tmp_path)
    img = np.full((720, 1280, 3), (150, 180, 215), np.uint8)
    cv2.circle(img, (640, 360), 75, (60, 140, 245), -1)
    path = tmp_path / "frame.jpg"
    cv2.imwrite(str(path), img)
    debug_ball.offline([str(path)], 76.6)
    (out,) = [p for p in tmp_path.rglob("frame.jpg") if p.parent != tmp_path]
    panel = cv2.imread(str(out))
    assert panel.shape == (720, 2560, 3)                    # frame | mask side by side


compare = _load("compare_ball_detection")


def test_compare_script_reports_old_vs_new(tmp_path, monkeypatch, capsys):
    """A logged close-up with two touching balls: old finds one, new finds both."""
    import json
    folder = tmp_path / "20261030-100000"
    folder.mkdir()
    img = np.full((720, 1280, 3), (150, 180, 215), np.uint8)
    for c in [(600, 360), (748, 372)]:
        cv2.circle(img, c, 75, (60, 140, 245), -1)
    cv2.imwrite(str(folder / "0002_find_ball.jpg"), img)
    (folder / "log.jsonl").write_text(json.dumps({"n": 2, "stage": "find_ball", "detector": "ball_orange",
                                                  "image": "0002_find_ball.jpg", "expected_r_px": 76.5,
                                                  "closeup": True, "items": []}) + "\n")
    monkeypatch.setattr(sys, "argv", ["compare", str(folder), "--no-yolo"])
    compare.main()
    out = capsys.readouterr().out
    assert "SUMMARY over 1 frames" in out and "new 1/1" in out
    line = next(l for l in out.splitlines() if l.startswith("20261030-100000#2"))
    assert line.split()[3] == "2"                                   # new column: both balls


scan_table = _load("scan_table")


def test_scan_visits_every_view_lists_every_ball_and_never_picks(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(detection_log, "ROOT", tmp_path)
    monkeypatch.setattr(pick_ball, "AVOID_XY", [(880.0, 205.0)])       # one cup ~90 cm out
    monkeypatch.setattr(pick_ball, "LOOK_SETTLE_S", 0)
    moves = []

    async def move(motion, x, y, z, go):
        moves.append((x, y, z))

    seen_from = {pick_ball.LOOK: [{"xy": (452.0, 51.0), "why": None}],
                 pick_ball.TOP_VIEW: [{"xy": (455.0, 49.0), "why": None},          # same ball, from above
                                      {"xy": (880.0, 205.0), "why": "in or next to a cup"},
                                      {"xy": (300.0, -420.0), "why": None}]}       # only the top sees it

    async def balls_in_view(robot):
        return seen_from.get(moves[-1], [])

    async def no_pick(*a, **k):
        raise AssertionError("scan_table must never pick")

    monkeypatch.setattr(pick_ball, "move", move)
    monkeypatch.setattr(pick_ball, "balls_in_view", balls_in_view)
    monkeypatch.setattr(pick_ball, "pick", no_pick)
    balls = asyncio.run(scan_table.scan(object(), object(), True))
    assert moves[0] == pick_ball.LOOK and moves[1] == pick_ball.TOP_VIEW
    assert len(moves) == 1 + len(pick_ball.search_views())                # every view visited
    assert len(balls) == 3                                                 # the double sighting merged
    pickable = scan_table.report(balls, pick_ball.search_max_r())
    assert len(pickable) == 2
    out = capsys.readouterr().out
    assert "-> in or next to a cup" in out and "TABLE: 2 ball(s) the robot would pick" in out


def test_scan_report_table_clear(capsys):
    scan_table.report([{"xy": (880.0, 205.0), "why": "in or next to a cup", "views": [2]}], 740)
    assert "TABLE CLEAR" in capsys.readouterr().out


def test_merge_keeps_touching_pair_apart_and_joins_views(monkeypatch):
    """Two touching balls 46 mm apart (same view) stay two balls; one ball seen from 3 views
    up to 26 mm apart is one ball; status follows the merged position / majority."""
    monkeypatch.setattr(pick_ball, "AVOID_XY", [])
    s = [(1, {"xy": (457.0, 2.0), "why": "touching another ball", "touching": True}),
         (1, {"xy": (500.0, -14.0), "why": "touching another ball", "touching": True}),
         (2, {"xy": (448.0, -3.0), "why": "touching another ball", "touching": True}),
         (2, {"xy": (489.0, -21.0), "why": "touching another ball", "touching": True}),
         (1, {"xy": (633.0, 81.0), "why": None, "touching": False}),
         (2, {"xy": (624.0, 75.0), "why": None, "touching": False}),
         (3, {"xy": (646.0, 85.0), "why": None, "touching": False})]
    balls = scan_table.merge(s)
    assert len(balls) == 3
    pair = [b for b in balls if b["touching"]]
    assert len(pair) == 2 and all(b["why"] == "touching another ball" for b in pair)
    single = next(b for b in balls if not b["touching"])
    assert sorted(single["views"]) == [1, 2, 3] and single["why"] is None and single["spread_mm"] < 15


def test_one_view_on_the_search_line_does_not_discard_a_ball(monkeypatch):
    monkeypatch.setattr(pick_ball, "AVOID_XY", [(905.0, 0.0)])          # line at 75.5 cm
    s = [(v, {"xy": xy, "why": None, "touching": False})
         for v, xy in [(1, (745.0, -42.0)), (2, (762.0, -41.0)), (3, (752.0, -39.0))]]
    (b,) = scan_table.merge(s)
    assert b["why"] is None                                            # merged 75.3 cm: pickable


def test_cut_off_ball_is_left_out():
    assert pick_ball.cut_off({"u": 30, "v": 300, "r": 36}, 1280, 720)
    assert not pick_ball.cut_off({"u": 600, "v": 300, "r": 36}, 1280, 720)


def test_ball_map_is_saved_in_world_frame(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(scan_table, "MAP_DIR", tmp_path / "ball_map")
    balls = [{"xy": (474.0, 81.0), "why": None, "views": [1, 2], "spread_mm": 12.0},
             {"xy": (456.0, 1.0), "why": "touching another ball", "views": [1], "spread_mm": 0.0}]
    path = scan_table.save_map(balls, 760, tmp_path)
    doc = json.loads((tmp_path / "ball_map" / "latest.json").read_text())
    assert path.exists() and (tmp_path / "ball_map.json").exists()
    assert doc["frame"] == "world" and doc["count"] == 2 and doc["pickable"] == 1
    assert doc["balls"][0] == {"n": 1, "x": 474.0, "y": 81.0, "dist_mm": 481, "status": "pickable",
                               "views": [1, 2], "spread_mm": 12.0}
