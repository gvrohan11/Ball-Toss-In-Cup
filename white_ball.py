"""Find a WHITE (or any-colour) ping-pong ball: depth sphere fit for the first look,
plain circle search for the close-up look.

Why two methods: the colour finder (ball_finder.py) only knows orange. A white ball on a
light table has no colour to threshold, so we use shape instead:
  * coarse look (camera ~50 cm up, depth is reliable there): circle proposals + RGB-D sphere
    fit (scene_detection.py, Saurabh) -> only things that are really 40 mm spheres survive;
  * close-up look (camera ~25 cm up, depth too close to trust): the ball is near the image
    centre by construction, so take the circle nearest the centre that has the right colour,
    then use the same calibrated ray-plane position as the orange ball.

Never moves the arm.
"""
import math

import cv2
import numpy as np

from scene_detection import ball_candidates, circle_proposals


def classify_color(bgr, u, v, r):
    """'orange', 'white' or 'other' from the middle of a circle (ignores the logo/edges)."""
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(u), int(v)), max(3, int(r * 0.6)), 255, -1)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[mask > 0]
    if len(hsv) == 0:
        return "other"
    hue, sat, val = (float(np.median(hsv[:, i])) for i in range(3))
    if 3 <= hue <= 25 and sat >= 110 and val >= 110:
        return "orange"
    # measured Oct 29: white ball H~100 S~80 V~247 (bluish, very bright); wood table V~155
    if sat <= 110 and val >= 200:
        return "white"
    return "other"


def matches(want, color):
    return want == "any" and color in ("orange", "white") or want == color


def coarse_balls(bgr, depth_mm, k, want, roi=None):
    """Individually fitted, isolated balls of colour `want` as dicts with camera-frame centre
    (mm), pixel centre and radius. Needs aligned depth at a camera height of ~40+ cm."""
    h, w = depth_mm.shape
    roi = roi or (0, 0, w, h)
    x0, y0, x1, y1 = roi
    d = depth_mm[y0:y1, x0:x1]
    d = d[np.isfinite(d) & (d > 0)]
    if not len(d):
        return []
    radius_px = k.focal_x_px * 20 / float(np.median(d))
    props = circle_proposals(bgr, roi, radius_px)
    reports = ball_candidates(props, depth_mm, (k.focal_x_px, k.focal_y_px, k.center_x_px,
                                                k.center_y_px), roi, 40)
    out = []
    for rep in reports:
        if not rep.get("accepted"):
            continue
        bx0, by0, bx1, by1 = rep["bbox"]
        u, v, r = (bx0 + bx1) / 2, (by0 + by1) / 2, (bx1 - bx0) / 2
        color = classify_color(bgr, u, v, r)
        out.append(dict(u=u, v=v, r=r, color=color, center_camera_mm=rep["center_camera_mm"],
                        sphere_error_mm=rep["sphere_error_mm"]))
    # keep balls with free space around them (fingers are ~ one ball wide on each side)
    for b in out:
        gaps = [math.hypot(b["u"] - o["u"], b["v"] - o["v"]) - b["r"] - o["r"]
                for o in out if o is not b]
        b["isolation_px"] = min(gaps) if gaps else 9999.0
        b["touching"] = b["isolation_px"] < b["r"] * 0.6
    chosen = [b for b in out if matches(want, b["color"])]
    chosen.sort(key=lambda b: (b["touching"], -min(b["isolation_px"], b["r"] * 3)))
    return chosen


def closeup_ball(bgr, k, expected_r, want, search_px=260):
    """Circle nearest the image centre with the wanted colour (ball is centred by the move)."""
    h, w = bgr.shape[:2]
    cx, cy = int(k.center_x_px), int(k.center_y_px)
    roi = (max(0, cx - search_px), max(0, cy - search_px), min(w, cx + search_px),
           min(h, cy + search_px))
    found = []
    for p in circle_proposals(bgr, roi, expected_r):
        x0, y0, x1, y1 = p.bbox
        u, v, r = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2
        color = classify_color(bgr, u, v, r)
        if matches(want, color):
            found.append(dict(u=u, v=v, r=r, color=color))
    found.sort(key=lambda b: (b["u"] - cx) ** 2 + (b["v"] - cy) ** 2)
    return found
