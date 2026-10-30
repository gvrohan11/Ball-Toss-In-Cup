"""Offline tests for the white-ball finder and the --ball switch (no robot)."""
import asyncio
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pick_ball  # noqa: E402
import white_ball  # noqa: E402


class K:
    focal_x_px = 910.0
    focal_y_px = 910.0
    center_x_px = 640.0
    center_y_px = 378.0


def scene(color_bgr, u=640, v=378, r=36):
    img = np.full((720, 1280, 3), (120, 160, 190), np.uint8)       # wood-ish table
    cv2.circle(img, (u, v), r, color_bgr, -1)
    cv2.circle(img, (u, v), r, (90, 100, 110), 2)                    # soft edge
    return cv2.GaussianBlur(img, (5, 5), 1.5)


WHITE = (250, 247, 240)   # bright, slightly bluish like the real ball
ORANGE = (40, 140, 250)


def test_classify_colors():
    assert white_ball.classify_color(scene(WHITE), 640, 378, 36) == "white"
    assert white_ball.classify_color(scene(ORANGE), 640, 378, 36) == "orange"
    assert white_ball.classify_color(scene((120, 160, 190)), 640, 378, 36) == "other"


def test_matches():
    assert white_ball.matches("white", "white") and not white_ball.matches("white", "orange")
    assert white_ball.matches("any", "orange") and not white_ball.matches("any", "other")


def test_closeup_finds_white_not_orange():
    found = white_ball.closeup_ball(scene(WHITE), K, 36, "white")
    assert len(found) == 1 and abs(found[0]["u"] - 640) < 4 and found[0]["color"] == "white"
    assert white_ball.closeup_ball(scene(ORANGE), K, 36, "white") == []


def test_ball_switch_dispatches_to_shape_finder(monkeypatch):
    calls = []

    async def shape(robot, prefer_center=False):
        calls.append(prefer_center)
        return (400.0, 10.0, 10.0)

    monkeypatch.setattr(pick_ball, "locate_ball_shape", shape)
    monkeypatch.setattr(pick_ball, "BALL_COLOR", "white")
    assert asyncio.run(pick_ball.locate_ball(object(), True)) == (400.0, 10.0, 10.0)
    assert calls == [True]


def test_ball_flag_validates(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--ball", "purple"])
    with pytest.raises(SystemExit):
        pick_ball.set_ball_color_from_argv()
    monkeypatch.setattr(sys, "argv", ["x", "--ball", "white"])
    monkeypatch.setattr(pick_ball, "BALL_COLOR", "orange")
    assert pick_ball.set_ball_color_from_argv() == "white"
