"""main.py must use the YOLO cup detection in cup_detection.py (offline, no robot)."""
import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cup_aim  # noqa: E402
import cup_detection  # noqa: E402
import main  # noqa: E402


def test_main_survey_uses_yolo_through_cup_aim():
    """The survey calls cup_aim.find_cups, which runs the rim method AND YOLO per frame."""
    import inspect
    import cup_aim
    assert main.find_cups is cup_aim.find_cups
    assert "cup_detection.find_cups" in inspect.getsource(cup_aim._yolo_cups)


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


# ---------- screening: COCO "cup" -> red solo cup only (Oct 29 survey frames) ----------
import cv2  # noqa: E402
import pytest  # noqa: E402
from pathlib import Path  # noqa: E402


def _det(box, conf, mask=None):
    return types.SimpleNamespace(bbox=box, confidence=conf, object_mask=mask)


def _scene():
    """Wood table, a pink-white cup opening at (100..200, 100..200), a white paper cup
    at (300..400, 100..200), a red can at (500..560, 100..220)."""
    img = np.full((480, 640, 3), (90, 140, 190), np.uint8)               # wood (BGR)
    pink = cv2.cvtColor(np.uint8([[[140, 50, 210]]]), cv2.COLOR_HSV2BGR)[0, 0]
    img[100:200, 100:200] = pink
    img[100:200, 300:400] = (245, 245, 245)
    img[100:220, 500:560] = (30, 30, 200)
    img[420:480, 100:200] = pink                                           # opening cut by the bottom
    img[300:380, 300:400] = pink                                           # whole opening, body below
    return img


def test_screen_keeps_only_red_solo_cups():
    img = _scene()
    dets = [_det((95, 95, 205, 205), 0.94), _det((295, 95, 405, 205), 0.92),
            _det((495, 95, 565, 225), 0.51), _det((95, 410, 205, 480), 0.77),
            _det((295, 295, 405, 480), 0.85),
            _det((95, 95, 260, 205), 0.40), _det((95, 95, 205, 205), 0.13)]
    why = {d.bbox + (d.confidence,): (it["kept"], it.get("why", "")) for d, it in cup_detection.screen(dets, img)}
    assert why[(95, 95, 205, 205, 0.94)] == (True, "")
    assert why[(295, 95, 405, 205, 0.92)][1].startswith("not a red cup")      # paper cup
    assert why[(495, 95, 565, 225, 0.51)][1].startswith("not a red cup")      # can
    assert why[(95, 410, 205, 480, 0.77)][1] == "opening cut off at frame edge"
    assert why[(295, 295, 405, 480, 0.85)] == (True, "")                    # body cut off: still fine
    assert why[(95, 95, 260, 205, 0.40)][1] == "overlaps a stronger box"      # cup + neighbor
    assert why[(95, 95, 205, 205, 0.13)][1] == "confidence 0.13"


def test_screen_suppresses_box_around_cup_and_neighbor():
    img = _scene()
    img[100:200, 200:260] = img[150, 150]                                      # more opening
    _, item = cup_detection.screen([_det((95, 95, 205, 205), 0.94), _det((95, 95, 262, 205), 0.40)], img)[1]
    assert item["why"] == "overlaps a stronger box"


def test_body_estimate_top_down_full_rim_has_no_bias():
    """Camera straight above at (0,0,600); cup axis at (150, 0), rim z=110, whole rim ring
    visible (as in every survey view). The old fixed 0.6r push put this ~28 mm off."""
    k = {"fx": 600.0, "fy": 600.0, "cx": 320.0, "cy": 240.0}
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = (0, 0, 600)
    depth = np.zeros((480, 640), np.uint16)
    mask = np.zeros_like(depth, bool)
    for zw in np.linspace(60, 110, 30):                     # upper part of the wall, all round
        for a in np.linspace(0, 2 * np.pi, 180, endpoint=False):
            xw, yw, zc = 150 + 47 * np.cos(a), 47 * np.sin(a), 600 - zw
            u, v = int(round(xw * 600 / zc + 320)), int(round(-yw * 600 / zc + 240))
            if depth[v, u] == 0 or depth[v, u] > zc:
                depth[v, u] = zc
            mask[v, u] = True
    c, _ = cup_detection.body_estimate(types.SimpleNamespace(bbox=(0, 0, 640, 480), object_mask=mask), depth, k, T)
    assert abs(c[0] - 150) < 5 and abs(c[1]) < 5 and abs(c[2] - 110) < 8


def test_fit_circle():
    a = np.linspace(0.3, 2.5, 50)                           # a partial arc is enough
    cx, cy, r, res = cup_detection.fit_circle(np.column_stack((10 + 47 * np.cos(a), -5 + 47 * np.sin(a))))
    assert abs(cx - 10) < 1e-6 and abs(cy + 5) < 1e-6 and abs(r - 47) < 1e-6 and res < 1e-6


# ---------- cup_aim: rim method + YOLO per frame ----------
@pytest.fixture
def frame_with(monkeypatch):
    import detection_log
    detection_log.stop()
    monkeypatch.setattr(cup_aim, "YOLO_OFFSETS", [])
    monkeypatch.setattr(cup_aim, "USE_YOLO", True)
    monkeypatch.setattr(cup_aim, "_yolo_off", None)
    monkeypatch.setattr(cup_detection, "CUP_CORRECTION_MM", (5.0, -5.0))   # test value, not the real one

    def setup(rim_items, yolo_items):
        async def rim(robot, topdown, seen, frame=None):
            seen["items"] += rim_items
            return [tuple(i["world_mm"]) for i in rim_items if i.get("kept")]

        async def yolo(robot, *, seen=None, **kw):
            seen.update(items=yolo_items, frame=None, depth=None)
            return []

        monkeypatch.setattr(cup_aim, "find_cups_by_rim", rim)
        monkeypatch.setattr(cup_detection, "find_cups", yolo)
    return setup


def test_yolo_adds_cups_the_rim_method_missed(frame_with):
    rim = [{"kind": "rim", "box": (100, 100, 200, 160), "kept": True, "world_mm": (800.0, -70.0, 110.0)},
           {"kind": "rim", "box": (500, 100, 560, 140), "kept": False, "why": "66 mm wide"}]
    yolo = [{"kind": "yolo", "box": (90, 90, 210, 230), "kept": True, "world_mm": (790.0, -60.0, 105.0)},
            {"kind": "yolo", "box": (490, 90, 570, 220), "kept": True, "world_mm": (1000.0, 120.0, 108.0)},
            {"kind": "yolo", "box": (300, 50, 380, 150), "kept": False, "why": "not a red cup (opening 0%)"}]
    frame_with(rim, yolo)
    cups = asyncio.run(cup_aim.find_cups(object(), topdown=False))
    assert cups == [(800.0, -70.0, 110.0), (1005.0, 115.0, 108.0)]   # rim kept as is; YOLO-only corrected
    assert yolo[0]["duplicate_of"] == "rim" and cup_aim.YOLO_OFFSETS == [(10.0, -10.0)]
    assert yolo[1]["raw_world_mm"] == (1000.0, 120.0, 108.0)


def test_without_yolo_it_is_rim_only_and_says_so_once(monkeypatch, capsys):
    import detection_log
    detection_log.stop()
    monkeypatch.setattr(cup_aim, "USE_YOLO", True)
    monkeypatch.setattr(cup_aim, "_yolo_off", None)

    async def rim(robot, topdown, seen, frame=None):
        seen["items"].append({"kind": "rim", "box": (0, 0, 10, 10), "kept": True, "world_mm": (700, 0, 110)})
        return [(700, 0, 110)]

    async def missing(robot, **kw):
        raise ImportError("No module named 'ultralytics'")

    monkeypatch.setattr(cup_aim, "find_cups_by_rim", rim)
    monkeypatch.setattr(cup_detection, "find_cups", missing)
    for _ in range(3):
        assert asyncio.run(cup_aim.find_cups(object())) == [(700, 0, 110)]
    assert capsys.readouterr().out.count("YOLO unavailable") == 1


# ---------- regression on a real survey frame (needs the local model + observations) ----------
FRAME = Path(__file__).resolve().parent.parent / "observations/detections/20261029-175207/0004_cup_view_2.jpg"
MODEL = Path(__file__).resolve().parent.parent / ".cup-models/yolo11m-seg.pt"


@pytest.mark.skipif(not (FRAME.exists() and MODEL.exists()), reason="needs the Oct 29 frame and YOLO weights")
def test_real_frame_paper_cup_and_cans_rejected():
    """View 2 of the 3-cup run: 2 whole red cups, a red cup cut off at the bottom edge,
    a white paper cup, a Coke can, a pink can. Only the 2 whole red cups survive."""
    bgr = cv2.imread(str(FRAME))
    det = cup_detection.load_detector()
    items = [it for _, it in cup_detection.screen(det.detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), bgr)]
    kept = sorted(it["box"][0] for it in items if it["kept"])
    assert len(kept) == 2 and abs(kept[0] - 510) < 15 and abs(kept[1] - 769) < 15


def test_yolo_correction_matches_measured_offset():
    """Set from 163 rim/YOLO shared sightings on sfhack11; (0, 0) made duplicate cups."""
    assert cup_detection.CUP_CORRECTION_MM == (-54.0, 22.0)


def test_rim_and_yolo_judge_the_same_frame(monkeypatch):
    """The rim method must use YOLO's captured frame, not grab a second one."""
    import detection_log
    detection_log.stop()
    monkeypatch.setattr(cup_aim, "USE_YOLO", True)
    monkeypatch.setattr(cup_aim, "_yolo_off", None)
    yolo_frame = np.zeros((4, 4, 3), np.uint8)
    got = {}

    async def yolo(robot, *, seen=None, **kw):
        seen.update(items=[], frame=yolo_frame, depth=None)

    async def rim(robot, topdown, seen, frame=None):
        got["frame"] = frame
        return []

    async def no_red(robot, items):
        return []

    monkeypatch.setattr(cup_detection, "find_cups", yolo)
    monkeypatch.setattr(cup_aim, "find_cups_by_rim", rim)
    monkeypatch.setattr(cup_aim, "_cups_from_red_bodies", no_red)
    asyncio.run(cup_aim.find_cups(object(), topdown=True))
    assert got["frame"] is yolo_frame
