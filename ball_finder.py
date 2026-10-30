"""Find individual ping-pong balls in a frame, even when several touch (e.g. in a plate).

The color detector draws one box around touching balls, which then looks like one huge,
close ball. Here we know how big a ball must look from the camera's height, so we look
for circles of exactly that size in the orange mask and split clusters into single balls.

    find_balls(bgr, expected_radius_px) -> list of dict(u, v, r, isolation_px, fill), best first
    "best" = most isolated from other orange blobs (room for the fingers), then roundest.

    python ball_finder.py frame.jpg 35      # writes frame_balls.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

# YOLO (the COCO model already used for cups) as a second, color-independent detector.
# On the 39 logged sfhack11 ball frames it found the orange ping-pong ball in 37, always as
# COCO "orange" (the fruit), never "sports ball". Color + edge fit stays the position source;
# YOLO confirms each candidate and, when the color mask finds nothing (shadow, lighting),
# its segmentation outline is circle-fit instead.
CLUSTER_COVER_MIN = 0.7  # a blob of touching balls: the split circles must cover this much of it
USE_YOLO = True
YOLO_MODEL = Path(__file__).resolve().parent / ".cup-models" / "yolo11m-seg.pt"
YOLO_CLASSES = ("orange", "sports ball")
YOLO_MIN_CONF = 0.15
_yolo = None            # loaded model, or the reason it can't be used

# Orange ping-pong ball (OpenCV hue 0-180). Measured on sfhack11 frames.
ORANGE_LO = np.array([3, 110, 110])
ORANGE_HI = np.array([25, 255, 255])


def orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, ORANGE_LO, ORANGE_HI)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def fit_circle_ransac(pts, r_lo, r_hi, iters=200, tol=3.0, seed=0):
    """Circle through most of the edge points (u, v, r, inlier share), or None.
    Points on a shadow / blur bulge or a cut-off edge are outliers, so they can't drag
    the center the way a centroid or a least-squares fit would."""
    if len(pts) < 10:
        return None
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < 1e-6:
            continue
        sa, sb, sc = (a ** 2).sum(), (b ** 2).sum(), (c ** 2).sum()
        u = (sa * (b[1] - c[1]) + sb * (c[1] - a[1]) + sc * (a[1] - b[1])) / d
        v = (sa * (c[0] - b[0]) + sb * (a[0] - c[0]) + sc * (b[0] - a[0])) / d
        r = np.hypot(a[0] - u, a[1] - v)
        if not r_lo <= r <= r_hi:
            continue
        n = np.count_nonzero(np.abs(np.hypot(pts[:, 0] - u, pts[:, 1] - v) - r) < tol)
        if n > best_n:
            best, best_n = (u, v, r), n
    if best is None:
        return None
    u, v, r = best                                     # refine on the inliers (least squares)
    inl = pts[np.abs(np.hypot(pts[:, 0] - u, pts[:, 1] - v) - r) < tol]
    a = np.column_stack((2 * inl[:, 0], 2 * inl[:, 1], np.ones(len(inl))))
    (u, v, c), *_ = np.linalg.lstsq(a, (inl ** 2).sum(axis=1), rcond=None)
    r = float(np.sqrt(max(c + u * u + v * v, 0)))
    return float(u), float(v), r, best_n / len(pts)


def _checks(m, u, v, r):
    """(fill, ring_orange): share of the disk that is orange, share of the ring just outside."""
    disk = np.zeros_like(m)
    cv2.circle(disk, (int(round(u)), int(round(v))), int(r), 255, -1)
    fill = np.count_nonzero(m & disk) / max(1, np.count_nonzero(disk))
    ring = np.zeros_like(m)
    cv2.circle(ring, (int(round(u)), int(round(v))), int(r * 1.45), 255, -1)
    cv2.circle(ring, (int(round(u)), int(round(v))), int(r * 1.15), 0, -1)
    return fill, np.count_nonzero(m & ring) / max(1, np.count_nonzero(ring))


def _hough(m, expected_r, r_lo, r_hi):
    blur = cv2.GaussianBlur(m, (9, 9), 2)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=expected_r * 1.5,
                               param1=100, param2=max(10, int(expected_r * 0.5)),
                               minRadius=r_lo, maxRadius=r_hi)
    return [] if circles is None else [tuple(map(float, c)) for c in circles[0]]


def yolo_balls(bgr):
    """COCO "orange" / "sports ball" detections: [(box, conf, label, mask)]; [] without YOLO."""
    global _yolo
    if not USE_YOLO or isinstance(_yolo, str):
        return []
    try:
        if _yolo is None:
            from ultralytics import YOLO
            _yolo = YOLO(str(YOLO_MODEL))
        ids = [k for k, v in _yolo.names.items() if v in YOLO_CLASSES]
        r = _yolo.predict(bgr, classes=ids, conf=YOLO_MIN_CONF, imgsz=1280, device="cpu", verbose=False)[0]
    except Exception as e:                      # missing library / weights: color only, say once
        _yolo = f"{type(e).__name__}: {e}"
        print(f"  [ball] YOLO unavailable ({str(e)[:80]}); color detection only")
        return []
    out = []
    for i, (box, conf, c) in enumerate(zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(),
                                           r.boxes.cls.cpu().numpy())):
        mask = None
        if r.masks is not None and i < len(r.masks.xy) and len(r.masks.xy[i]) >= 3:
            mask = np.zeros(bgr.shape[:2], np.uint8)
            cv2.fillPoly(mask, [r.masks.xy[i].astype(np.int32)], 255)
        out.append((tuple(float(v) for v in box), float(conf), _yolo.names[int(c)], mask))
    return out


def _edge_points(blob):
    h, w = blob.shape
    contour = max(cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
    pts = contour[:, 0, :].astype(float)
    return pts[(pts[:, 0] > 1) & (pts[:, 1] > 1) & (pts[:, 0] < w - 2) & (pts[:, 1] < h - 2)]


def _split_cluster(blob, expected_r, r_lo, r_hi):
    """Touching balls: fit a circle to the blob's edge, remove that circle's edge points,
    fit the next one. Each ball keeps most of its own outer edge, so each is found; the
    Hough search alone often found just one of two touching balls. Hough fills in if the
    edge fits find nothing."""
    pts = _edge_points(blob)
    n_max = int(np.ceil(np.count_nonzero(blob) / (np.pi * expected_r ** 2))) + 1
    min_pts = 0.3 * 2 * np.pi * expected_r          # a third of a ball's rim must be visible
    circles = []
    for k in range(n_max):
        fit = fit_circle_ransac(pts, r_lo, r_hi, iters=400, seed=k)
        if fit is None or fit[3] * len(pts) < min_pts:
            break
        u, v, r, _ = fit
        circles.append((u, v, r))
        pts = pts[np.abs(np.hypot(pts[:, 0] - u, pts[:, 1] - v) - r) > 6]
        if len(pts) < min_pts:
            break
    return circles or _hough(blob, expected_r, r_lo, r_hi)


def find_balls(bgr, expected_r, debug=None):
    """Balls of radius ~expected_r (pixels) in the orange mask.

    A blob the size of one ball (most frames) is located by a RANSAC circle fit to its
    edge: stable frame to frame, and a shadow or blur bulge on one side doesn't move it.
    A blob bigger than one ball (touching balls) is split by repeated edge fits
    (_split_cluster); the circles must cover most of it, or it isn't balls.
    (Hough alone was the old method: its vote threshold sat on the edge for a single
    ball, so the same frame flipped between found / not found.)
    debug: optional dict, gets the mask and every blob with its numbers."""
    m = orange_mask(bgr)
    h, w = m.shape
    # 1.4: one of the Oct 30 balls measured 1.19x in the wide view and tipped over 1.25 up close
    r_lo, r_hi = int(expected_r * 0.8), int(expected_r * 1.4) + 1
    one = np.pi * expected_r ** 2
    blobs = []
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m)
    found = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.3 * one:
            continue
        blob = (labels == i).astype(np.uint8) * 255
        info = {"area_r": round(float(np.sqrt(area / np.pi)), 1),
                "centroid": (round(float(cents[i][0]), 1), round(float(cents[i][1]), 1))}
        blobs.append(info)
        if area > 1.6 * one:                         # several balls touching: split them
            info["method"] = "split"
            cands = _split_cluster(blob, expected_r, r_lo, r_hi)
            # real touching balls are mostly circles; a ball-sized circle inside a big orange
            # thing (paper, box, a hand) is not a ball (seen live: orange paper by a plate)
            cover = np.zeros_like(blob)
            for u, v, r in cands:
                cv2.circle(cover, (int(round(u)), int(round(v))), int(r), 255, -1)
            covered = np.count_nonzero(cover & blob) / area
            info["covered"] = round(covered, 2)
            if covered < CLUSTER_COVER_MIN:
                info["why"] = f"not a ball cluster (circles cover {covered:.0%})"
                continue
        else:
            fit = fit_circle_ransac(_edge_points(blob), r_lo, r_hi)
            info["method"] = "edge_fit"
            if fit is None or fit[3] < 0.5:
                info["why"] = "edge not a ball-sized circle"
                continue
            info["edge_inliers"] = round(fit[3], 2)
            cands = [fit[:3]]
        for u, v, r in cands:
            fill, ring_orange = _checks(m, u, v, r)
            if fill < 0.7:                          # most of the circle must actually be orange
                info["why"] = f"fill {fill:.2f}"
                continue
            # the ring just outside must be mostly NOT orange: rejects circles that are
            # part of a bigger orange/skin-colored region (hands, cups, boxes)
            if ring_orange > 0.5:
                info["why"] = f"ring orange {ring_orange:.2f}"
                continue
            # how far the blob's centroid sits from the circle: a shadow / blur bulge
            shift = float(np.hypot(cents[i][0] - u, cents[i][1] - v))
            found.append(dict(u=float(u), v=float(v), r=float(r), fill=round(fill, 2),
                              ring_orange=round(ring_orange, 2), method=info["method"],
                              bulge_px=round(shift, 1) if info["method"] == "edge_fit" else None))
    # YOLO: confirm color balls; add balls only YOLO sees (circle fit to its outline)
    ys = yolo_balls(bgr)
    for b in found:
        b["yolo"] = next((round(c, 2) for (x0, y0, x1, y1), c, _, _ in ys
                          if x0 <= b["u"] <= x1 and y0 <= b["v"] <= y1), None)
    for (x0, y0, x1, y1), conf, label, ymask in ys:
        if ymask is None or any(x0 <= b["u"] <= x1 and y0 <= b["v"] <= y1 for b in found):
            continue
        fit = fit_circle_ransac(_edge_points(ymask), r_lo, r_hi)
        info = {"method": "yolo_mask", "yolo": round(conf, 2), "label": label,
                "centroid": (round((x0 + x1) / 2, 1), round((y0 + y1) / 2, 1))}
        blobs.append(info)
        if fit is None or fit[3] < 0.5:
            info["why"] = "YOLO outline not a ball-sized circle"
            continue
        u, v, r, _ = fit
        h, w = ymask.shape
        disk = np.zeros_like(ymask)
        cv2.circle(disk, (int(round(u)), int(round(v))), int(r), 255, -1)
        if np.count_nonzero(disk) < 0.75 * np.pi * r * r:   # mostly off-frame: center is a guess
            info["why"] = "cut off at frame edge"
            continue
        fill, _ = _checks(ymask, u, v, r)
        if fill < 0.7:
            info["why"] = f"fill {fill:.2f}"
            continue
        found.append(dict(u=u, v=v, r=r, fill=round(fill, 2), ring_orange=None, method="yolo_mask",
                          bulge_px=None, yolo=round(conf, 2)))
    # isolation: distance to the nearest OTHER ball / orange pixel outside this ball
    for b in found:
        others = [np.hypot(b["u"] - o["u"], b["v"] - o["v"]) - o["r"] - b["r"]
                  for o in found if o is not b]
        b["isolation_px"] = round(min(others), 1) if others else float(max(h, w))
        b["touching"] = bool(others) and min(others) < expected_r * 0.6
    found.sort(key=lambda b: (-min(b["isolation_px"], expected_r * 3), -b["fill"]))
    if debug is not None:
        debug.update(mask=m, blobs=blobs, expected_r=expected_r,
                     yolo=[(box, round(c, 2), label) for box, c, label, _ in ys])
    return found


def sharpness(bgr, u, v, r):
    """Variance of the Laplacian around the ball: low = motion blur (arm still moving)."""
    x0, y0 = max(0, int(u - 1.5 * r)), max(0, int(v - 1.5 * r))
    roi = cv2.cvtColor(bgr[y0:int(v + 1.5 * r), x0:int(u + 1.5 * r)], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    r = float(sys.argv[2]) if len(sys.argv) > 2 else 35
    balls = find_balls(img, r)
    print(balls)
    for b in balls:
        color = (0, 0, 255) if b["touching"] else (0, 255, 0)
        cv2.circle(img, (int(b["u"]), int(b["v"])), int(b["r"]), color, 3)
    cv2.imwrite(sys.argv[1].rsplit(".", 1)[0] + "_balls.jpg", img)
