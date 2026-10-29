"""Find red solo cups in a camera frame from their pinkish-white opening + red body.

Why: when the gripper holds the orange ball at the cup-look pose, the ball covers most of
the cup's red side, so the plain red color detector finds nothing. The inside of a red solo
cup is white with a pink/magenta tint (red light bouncing inside), which nothing else on the
table has: wood is orange-ish, paper and plates are blue-ish white.

    find_cup_rims(bgr_image) -> list of dict(u, v, width_px, height_px, score), best first
    (u, v) = pixel center of the cup opening.

Pure OpenCV, no robot needed:
    python cup_rim.py image.jpg [more.jpg ...]   # prints detections, writes *_rim.jpg overlays
"""
import sys

import cv2
import numpy as np

# Inside of the cup: hue ~120-150 on OpenCV's 0-180 scale (pink/magenta tint),
# low-medium saturation, fairly bright. Measured on sfhack11 frames, Oct 29.
PINK_LO = np.array([115, 22, 110])
PINK_HI = np.array([165, 90, 245])
# Cup body: saturated red.
RED1_LO, RED1_HI = np.array([0, 110, 50]), np.array([10, 255, 255])
RED2_LO, RED2_HI = np.array([170, 110, 50]), np.array([180, 255, 255])

MIN_AREA_PX = 1500
MIN_RED_FRACTION = 0.03     # share of red pixels in the band around/under the opening


def _masks(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    pink = cv2.inRange(hsv, PINK_LO, PINK_HI)
    red = cv2.inRange(hsv, RED1_LO, RED1_HI) | cv2.inRange(hsv, RED2_LO, RED2_HI)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    pink = cv2.morphologyEx(pink, cv2.MORPH_OPEN, k)
    pink = cv2.morphologyEx(pink, cv2.MORPH_CLOSE, k)
    return pink, red


def find_cup_rims(bgr):
    pink, red = _masks(bgr)
    h_img, w_img = pink.shape
    n, labels, stats, cents = cv2.connectedComponentsWithStats(pink)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA_PX:
            continue
        aspect = min(w, h) / max(w, h)
        if aspect < 0.35:                       # openings are round-ish ellipses
            continue
        fill = area / float(w * h)
        if fill < 0.45:                         # an ellipse fills ~0.79 of its box
            continue
        # red body: band from the middle of the opening down to 1.2x its height below
        x0, x1 = max(0, x - w // 4), min(w_img, x + w + w // 4)
        y0, y1 = max(0, y + h // 2), min(h_img, y + h + int(1.2 * h))
        band = red[y0:y1, x0:x1]
        red_frac = float(np.count_nonzero(band)) / max(1, band.size)
        if red_frac < MIN_RED_FRACTION:
            continue
        score = area * (0.5 + red_frac) * fill
        u, v = cents[i]
        out.append(dict(u=float(u), v=float(v), width_px=int(w), height_px=int(h),
                        red_fraction=round(red_frac, 3), score=float(score),
                        box=(int(x), int(y), int(x + w), int(y + h))))
    out.sort(key=lambda d: -d["score"])
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        dets = find_cup_rims(img)
        print(path.split("/")[-1], dets[:3])
        for d in dets[:3]:
            cv2.rectangle(img, d["box"][:2], d["box"][2:], (0, 255, 0), 3)
            cv2.circle(img, (int(d["u"]), int(d["v"])), 6, (0, 255, 0), -1)
        cv2.imwrite(path.rsplit(".", 1)[0] + "_rim.jpg", img)
