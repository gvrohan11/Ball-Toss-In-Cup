"""Ball finder: split touching balls, prefer an isolated one, reject non-ball blobs."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ball_finder as bf  # noqa: E402

WOOD, ORANGE = (150, 180, 215), (60, 140, 245)


@pytest.fixture(autouse=True)
def color_only(monkeypatch):
    """Synthetic scenes test the color path; YOLO gets its own (faked) tests below."""
    monkeypatch.setattr(bf, "USE_YOLO", False)


def scene(centers, r=35, extra=None):
    img = np.full((720, 1280, 3), WOOD, np.uint8)
    for c in centers:
        cv2.circle(img, c, r, ORANGE, -1)
    if extra:
        extra(img)
    return img


def test_cluster_is_split_and_isolated_ball_first():
    balls = bf.find_balls(scene([(400, 300), (470, 300), (435, 360), (900, 500)]), 35)
    assert len(balls) == 4
    assert abs(balls[0]["u"] - 900) < 5 and not balls[0]["touching"]
    assert all(b["touching"] for b in balls[1:])


def test_wrong_size_and_big_blobs_are_rejected():
    def big(img):
        cv2.rectangle(img, (100, 100), (400, 300), ORANGE, -1)   # e.g. an orange box
    balls = bf.find_balls(scene([(900, 500)], extra=big), 35)
    assert len(balls) == 1 and abs(balls[0]["u"] - 900) < 5
    assert bf.find_balls(scene([(600, 400)], r=18), 35) == []   # half-size blob


def test_empty_table():
    assert bf.find_balls(scene([]), 35) == []


# ---------- shadows / blur: the edge fit ignores a bulge on one side ----------
def test_shadow_bulge_does_not_move_the_center():
    """A dark-orange shadow merged into the mask on one side: the RANSAC edge fit keeps the
    ball's true center (a centroid would shift ~10 px toward the shadow)."""
    def shadow(img):
        cv2.ellipse(img, (628, 412), (30, 18), 35, 0, 360, (40, 110, 200), -1)   # orange-ish shadow
    img = scene([(600, 400)], r=35, extra=shadow)
    (b,) = bf.find_balls(img, 35)
    assert abs(b["u"] - 600) < 2.5 and abs(b["v"] - 400) < 2.5
    assert b["method"] == "edge_fit" and b["bulge_px"] > 3


def test_same_frame_same_answer():
    """The old Hough-only search flipped between found / not found on re-encoded frames."""
    img = scene([(600, 400)], r=75)
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    again = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    for frame in (img, again):
        (b,) = bf.find_balls(frame, 76.6)
        assert abs(b["u"] - 600) < 2 and abs(b["v"] - 400) < 2


def test_sharpness_drops_with_motion_blur():
    img = scene([(600, 400)], r=75)
    blurred = cv2.blur(img, (25, 1))
    assert bf.sharpness(blurred, 600, 400, 75) < 0.5 * bf.sharpness(img, 600, 400, 75)


# ---------- YOLO (faked): confirms color balls, adds balls color missed ----------
def _yolo_mask(c, r, shape=(720, 1280)):
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, c, r, 255, -1)
    return m


def test_yolo_confirms_and_adds_a_ball_color_missed(monkeypatch):
    monkeypatch.setattr(bf, "USE_YOLO", True)
    # color sees the ball at (300,300); a ball with marker writing at (900,400) is too dark
    img = scene([(300, 300)], r=75)
    cv2.circle(img, (900, 400), 75, (40, 40, 60), -1)
    monkeypatch.setattr(bf, "yolo_balls", lambda bgr: [
        ((225, 225, 375, 375), 0.8, "orange", _yolo_mask((300, 300), 75)),
        ((825, 325, 975, 475), 0.7, "orange", _yolo_mask((900, 400), 75)),
        ((1200, 650, 1280, 720), 0.6, "orange", _yolo_mask((1270, 760), 75))])   # mostly off-frame
    balls = {b["method"]: b for b in bf.find_balls(img, 76.6)}
    assert balls["edge_fit"]["yolo"] == 0.8
    assert abs(balls["yolo_mask"]["u"] - 900) < 3 and balls["yolo_mask"]["yolo"] == 0.7
    assert len(balls) == 2                                          # the off-frame one is rejected


def test_without_yolo_library_it_is_color_only(monkeypatch, capsys):
    monkeypatch.setattr(bf, "USE_YOLO", True)
    monkeypatch.setattr(bf, "_yolo", None)
    monkeypatch.setattr(bf, "YOLO_MODEL", "/nonexistent/yolo.pt")
    for _ in range(2):
        (b,) = bf.find_balls(scene([(300, 300)], r=75), 76.6)
        assert b["yolo"] is None
    assert capsys.readouterr().out.count("YOLO unavailable") == 1


def test_orange_paper_is_not_a_ball_cluster():
    """Seen live: a ball-sized circle inside a big orange sheet by the plate."""
    def paper(img):
        pts = np.array([[1100, 40], [1180, 20], [1230, 160], [1150, 230], [1120, 150]], np.int32)
        cv2.fillPoly(img, [pts], ORANGE)
    balls = bf.find_balls(scene([(500, 400)], r=22, extra=paper), 21)
    assert len(balls) == 1 and abs(balls[0]["u"] - 500) < 3


def test_two_touching_balls_are_both_found():
    """Seen live (19:16:52 close-up): Hough found only one of two touching balls."""
    balls = bf.find_balls(scene([(600, 360), (748, 372)], r=75), 76.5)
    assert len(balls) == 2 and all(b["touching"] for b in balls)
    xs = sorted(round(b["u"]) for b in balls)
    assert abs(xs[0] - 600) < 4 and abs(xs[1] - 748) < 4
