"""Ball finder: split touching balls, prefer an isolated one, reject non-ball blobs."""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ball_finder as bf  # noqa: E402

WOOD, ORANGE = (150, 180, 215), (60, 140, 245)


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
