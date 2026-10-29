"""Find individual ping-pong balls in a frame, even when several touch (e.g. in a plate).

The color detector draws one box around touching balls, which then looks like one huge,
close ball. Here we know how big a ball must look from the camera's height, so we look
for circles of exactly that size in the orange mask and split clusters into single balls.

    find_balls(bgr, expected_radius_px) -> list of dict(u, v, r, isolation_px, fill), best first
    "best" = most isolated from other orange blobs (room for the fingers), then roundest.

    python ball_finder.py frame.jpg 35      # writes frame_balls.jpg
"""
import sys

import cv2
import numpy as np

# Orange ping-pong ball (OpenCV hue 0-180). Measured on sfhack11 frames.
ORANGE_LO = np.array([3, 110, 110])
ORANGE_HI = np.array([25, 255, 255])


def orange_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, ORANGE_LO, ORANGE_HI)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def find_balls(bgr, expected_r):
    """Circles of radius ~expected_r (pixels) in the orange mask."""
    m = orange_mask(bgr)
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
            inside = np.count_nonzero(disk)
            fill = np.count_nonzero(m & disk) / max(1, inside)
            if fill < 0.7:          # most of the circle must actually be orange
                continue
            # the ring just outside must be mostly NOT orange: rejects circles that are
            # part of a bigger orange/skin-colored region (hands, cups, boxes)
            ring = np.zeros_like(m)
            cv2.circle(ring, (int(u), int(v)), int(r * 1.45), 255, -1)
            cv2.circle(ring, (int(u), int(v)), int(r * 1.15), 0, -1)
            ring_orange = np.count_nonzero(m & ring) / max(1, np.count_nonzero(ring))
            if ring_orange > 0.5:
                continue
            found.append(dict(u=float(u), v=float(v), r=float(r), fill=round(fill, 2),
                              ring_orange=round(ring_orange, 2)))
    # isolation: distance to the nearest OTHER ball / orange pixel outside this ball
    for b in found:
        others = [np.hypot(b["u"] - o["u"], b["v"] - o["v"]) - o["r"] - b["r"]
                  for o in found if o is not b]
        b["isolation_px"] = round(min(others), 1) if others else float(max(h, w))
        b["touching"] = bool(others) and min(others) < expected_r * 0.6
    found.sort(key=lambda b: (-min(b["isolation_px"], expected_r * 3), -b["fill"]))
    return found


if __name__ == "__main__":
    img = cv2.imread(sys.argv[1])
    r = float(sys.argv[2]) if len(sys.argv) > 2 else 35
    balls = find_balls(img, r)
    print(balls)
    for b in balls:
        color = (0, 0, 255) if b["touching"] else (0, 255, 0)
        cv2.circle(img, (int(b["u"]), int(b["v"])), int(b["r"]), color, 3)
    cv2.imwrite(sys.argv[1].rsplit(".", 1)[0] + "_balls.jpg", img)
