"""Old vs new ball detection on the frames the robot actually saw. Read-only.

    python scripts/compare_ball_detection.py                 # every logged ball frame
    python scripts/compare_ball_detection.py --panels        # + side-by-side images per frame
    python scripts/compare_ball_detection.py observations/detections/20261029-191013 ...

OLD = the original detector (Hough circle search on the orange mask), frozen below so the
      comparison still works after ball_finder.py changes.
NEW = ball_finder.find_balls: RANSAC edge fit per blob, repeated edge fits to split
      touching balls (circles must cover the blob), color only;  NEW+YOLO = plus YOLO "orange".

Per frame and in total:
  found     ball found at all (the frame was logged because the robot looked for a ball)
  stable    same answer on the frame and on a re-encoded copy (JPEG q85): the old search
            flipped between found / not found on identical-looking frames
  shift     old chosen center vs the nearest new center (same ball), px and mm (radius = 20 mm)
  bulge     how far the blob's centroid is from the new circle: shadow / blur pull
  ms        time per frame
Panels (--panels): observations/ball_compare/<time>/: old (red) vs new (green) circles.
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ball_finder  # noqa: E402

OUT = ROOT / "observations" / "ball_compare"


def old_find_balls(bgr, expected_r):
    """The original ball_finder.find_balls (Hough only), kept verbatim for comparison."""
    m = ball_finder.orange_mask(bgr)
    h, w = m.shape
    r_lo, r_hi = int(expected_r * 0.8), int(expected_r * 1.25) + 1
    blur = cv2.GaussianBlur(m, (9, 9), 2)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=expected_r * 1.5,
                               param1=100, param2=max(10, int(expected_r * 0.5)),
                               minRadius=r_lo, maxRadius=r_hi)
    found = []
    if circles is not None:
        for u, v, r in circles[0]:
            disk = np.zeros_like(m)
            cv2.circle(disk, (int(u), int(v)), int(r), 255, -1)
            fill = np.count_nonzero(m & disk) / max(1, np.count_nonzero(disk))
            if fill < 0.7:
                continue
            ring = np.zeros_like(m)
            cv2.circle(ring, (int(u), int(v)), int(r * 1.45), 255, -1)
            cv2.circle(ring, (int(u), int(v)), int(r * 1.15), 0, -1)
            if np.count_nonzero(m & ring) / max(1, np.count_nonzero(ring)) > 0.5:
                continue
            found.append(dict(u=float(u), v=float(v), r=float(r)))
    for b in found:
        others = [np.hypot(b["u"] - o["u"], b["v"] - o["v"]) - o["r"] - b["r"] for o in found if o is not b]
        b["isolation_px"] = min(others) if others else float(max(h, w))
    found.sort(key=lambda b: -min(b["isolation_px"], expected_r * 3))
    return found


def new_find_balls(bgr, expected_r, yolo):
    ball_finder.USE_YOLO = yolo
    return ball_finder.find_balls(bgr, expected_r)


def reencode(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def chosen(balls, closeup, shape):
    """The ball the picker would take: nearest the image center in a close-up, else first."""
    if not balls:
        return None
    if closeup:
        cy, cx = shape[0] / 2, shape[1] / 2
        return min(balls, key=lambda b: (b["u"] - cx) ** 2 + (b["v"] - cy) ** 2)
    return balls[0]


def frames(paths):
    if not paths:
        paths = sorted((ROOT / "observations" / "detections").glob("*"))
    for p in map(Path, paths):
        log = p / "log.jsonl"
        if not log.exists():
            continue
        for line in map(json.loads, log.read_text().splitlines()):
            if str(line.get("detector", "")).startswith("ball_orange") and line.get("image"):
                img = cv2.imread(str(p / line["image"]))
                if img is not None:
                    yield f"{p.name}#{line['n']}", img, line["expected_r_px"], bool(line.get("closeup"))


def timed(fn, *a):
    t = time.perf_counter()
    out = fn(*a)
    return out, (time.perf_counter() - t) * 1000


def panel(img, old, new, name):
    out = img.copy()
    for b in old:
        cv2.circle(out, (int(b["u"]), int(b["v"])), int(b["r"]), (40, 40, 230), 2)
    for b in new:
        cv2.circle(out, (int(round(b["u"])), int(round(b["v"]))), int(b["r"]), (40, 200, 40), 2)
        cv2.drawMarker(out, (int(round(b["u"])), int(round(b["v"]))), (40, 200, 40), cv2.MARKER_CROSS, 12, 2)
    cv2.putText(out, f"{name}  old (red): {len(old)}  new (green): {len(new)}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="detection-log folders (default: all)")
    ap.add_argument("--panels", action="store_true", help="save old-vs-new images")
    ap.add_argument("--no-yolo", action="store_true", help="skip the NEW+YOLO column (faster)")
    a = ap.parse_args()
    out = None
    if a.panels:
        out = OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True, exist_ok=True)

    tot = {k: 0 for k in ("n", "old", "new", "yolo", "old_flip", "new_flip", "yolo_flip",
                          "old_ms", "new_ms", "yolo_ms")}
    shifts, bulges, gained, lost = [], [], [], []
    print(f"{'frame':<20}{'view':<7}{'old':>4}{'new':>5}{'+yolo':>7}  {'shift':>14}  {'bulge':>6}  stable(old/new)")
    for name, img, er, closeup in frames(a.paths):
        again = reencode(img)
        old, t_old = timed(old_find_balls, img, er)
        new, t_new = timed(new_find_balls, img, er, False)
        yl, t_y = timed(new_find_balls, img, er, True) if not a.no_yolo else ([], 0)
        old_flip = bool(old) != bool(old_find_balls(again, er))
        new_flip = bool(new) != bool(new_find_balls(again, er, False))
        yolo_flip = (bool(yl) != bool(new_find_balls(again, er, True))) if not a.no_yolo else False
        tot["n"] += 1
        tot["old"] += bool(old)
        tot["new"] += bool(new)
        tot["yolo"] += bool(yl)
        tot["old_flip"] += old_flip
        tot["new_flip"] += new_flip
        tot["yolo_flip"] += yolo_flip
        tot["old_ms"] += t_old
        tot["new_ms"] += t_new
        tot["yolo_ms"] += t_y
        best = yl if (yl and not a.no_yolo) else new
        co, cn = chosen(old, closeup, img.shape), chosen(best, closeup, img.shape)
        shift = ""
        if co and cn:          # same ball: the new circle nearest the old chosen one
            d = min(float(np.hypot(co["u"] - b["u"], co["v"] - b["v"])) for b in best)
            mm = d * 20 / er                       # expected radius er px = 20 mm
            shifts.append(mm)
            shift = f"{d:5.1f}px {mm:4.1f}mm"
        elif cn and not co:
            gained.append(name)
            shift = "new only"
        elif co and not cn:
            lost.append(name)
            shift = "old only"
        bulge = ""
        if cn and cn.get("bulge_px") is not None:
            bulges.append(cn["bulge_px"] * 20 / er)
            bulge = f"{cn['bulge_px']:.1f}"
        print(f"{name:<20}{'close' if closeup else 'look':<7}{len(old):>4}{len(new):>5}"
              f"{len(yl) if not a.no_yolo else '-':>7}  {shift:>14}  {bulge:>6}  "
              f"{'FLIPS' if old_flip else 'ok'}/{'FLIPS' if new_flip else 'ok'}")
        if out is not None:
            cv2.imwrite(str(out / f"{name.replace('#', '_')}.jpg"), panel(img, old, best, name),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])

    n = tot["n"]
    if not n:
        print("no logged ball frames found under observations/detections/")
        return
    print("\nSUMMARY over", n, "frames")
    print(f"  ball found        old {tot['old']}/{n}   new {tot['new']}/{n}"
          + ("" if a.no_yolo else f"   new+YOLO {tot['yolo']}/{n}"))
    print(f"  flips on re-save  old {tot['old_flip']}   new {tot['new_flip']}"
          + ("" if a.no_yolo else f"   new+YOLO {tot['yolo_flip']}"))
    if shifts:
        print(f"  center old vs new median {np.median(shifts):.1f} mm, max {max(shifts):.1f} mm")
    if bulges:
        print(f"  shadow/blur pull a centroid would have had: median {np.median(bulges):.1f} mm, "
              f"max {max(bulges):.1f} mm")
    if gained:
        print(f"  found only by new: {', '.join(gained)}")
    if lost:
        print(f"  found only by old: {', '.join(lost)}  (check these panels)")
    print(f"  time per frame    old {tot['old_ms'] / n:.0f} ms   new {tot['new_ms'] / n:.0f} ms"
          + ("" if a.no_yolo else f"   new+YOLO {tot['yolo_ms'] / n:.0f} ms"))
    if out is not None:
        print(f"\npanels: {out}")


if __name__ == "__main__":
    main()
