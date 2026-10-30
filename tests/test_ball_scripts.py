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
