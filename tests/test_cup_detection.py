"""main.py must use the YOLO cup detection in cup_detection.py (offline, no robot)."""
import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cup_detection  # noqa: E402
import main  # noqa: E402


def test_main_uses_yolo_cup_detection():
    assert main.find_cup is cup_detection.find_cup


import numpy as np  # noqa: E402


def test_find_cup_uses_best_and_applies_correction(monkeypatch):
    best = cup_detection.FoundCup((700.0, 0.0, 105.0), 0.8, "rim_fit", 95, 1, (0, 0, 1, 1))

    async def fake(robot, **kw):
        return [best]

    monkeypatch.setattr(cup_detection, "find_cups", fake)
    monkeypatch.setattr(cup_detection, "CUP_CORRECTION_MM", (10.0, -5.0))
    assert asyncio.run(cup_detection.find_cup(None)) == (710.0, -5.0, 105.0)


def test_find_cup_returns_none_on_error(monkeypatch):
    async def broken(robot, **kw):
        raise RuntimeError("arm moved during capture")

    monkeypatch.setattr(cup_detection, "find_cups", broken)
    assert asyncio.run(cup_detection.find_cup(None)) is None


def test_body_estimate_on_synthetic_cylinder():
    """Camera at world (0,0,600) looking straight down; cup axis at (150, 0), rim z=110."""
    k = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])  # cam x->world x, cam y->-world y, cam z->-world z
    T[:3, 3] = (0, 0, 600)
    depth = np.zeros((480, 640), np.uint16)
    mask = np.zeros_like(depth, bool)
    for zw in np.linspace(0, 110, 60):          # near half of the cup wall
        for a in np.linspace(np.pi / 2, 3 * np.pi / 2, 90):
            xw, yw = 150 + 47 * np.cos(a), 47 * np.sin(a)
            zc = 600 - zw
            u, v = int(round(xw * 600 / zc + 320)), int(round(-yw * 600 / zc + 240))
            if depth[v, u] == 0 or depth[v, u] > zc:
                depth[v, u] = zc
            mask[v, u] = True
    det = types.SimpleNamespace(bbox=(0, 0, 640, 480), object_mask=mask)
    c, _ = cup_detection.body_estimate(det, depth, k, T)
    assert abs(c[2] - 110) < 8 and abs(c[1]) < 10 and abs(c[0] - 150) < 30
