"""Design study for a no-timing CRADLE throw on the xArm850 URDF (no robot).

RESULT (Oct 29, 23:00): a fingers-UP cradle opened before the swing is NOT feasible: the module's
max 1145 deg/s^2 brakes the ball at only ~1.2-1.8 g, so it never clears the fingertips (best
candidate: 30-75 mm short). exit_margin() is still the useful tool: FAR's motion clears the tips at
165-180 deg/s, which is what throw.CRADLE (spoon release) is built on.

Idea: the wrist is rolled 90 deg so the fingers sit ABOVE and BELOW the ball (the lower finger is a
spoon). At the wind-up pose the fingers point upward and open slightly, so the ball rests loosely
in the hand before anything moves. The swing (one direct-mode joint move, all joints synchronized,
lead joint at full speed) keeps the ball pressed into the palm; when the arm brakes the ball slides
out of the open end of the fingers. No release timing at all.

Constraints checked for each candidate, at several swing speeds and ball offsets:
  * start pose: the tool axis points >= ~15 deg up (the ball sits in the palm by gravity)
  * accelerate + cruise: the felt acceleration (a + g) points into the palm (ball can't slide out)
  * release (start of braking): launch 30-50 deg up, fingers within 20 deg of the ball's velocity
  * swept path clear of table / ceiling / walls (throw_safety), joint limits, home -> start clear
Objective: longest landing at 180 deg/s.

    python scripts/design_cradle.py            # ~2-4 min
"""
import math
import os
import sys

import numpy as np
from scipy.optimize import differential_evolution

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import throw_safety as ts  # noqa: E402

G = 9810.0                  # mm/s^2
ACCEL = 1145.0              # deg/s^2 (module max)
RIM_Z = 110.0               # cup rim height, mm (world)
BALL_OFFSETS = (130.0, 175.0)  # ball center from the flange along the tool axis, mm (unknown exactly
                                # when cradled: check both ends of the plausible range)
EXIT_TIP_MM = 205.0         # ball center must pass flange + this (fingertips ~195 + margin) to be free
TOP_GAP_MM = 8.0            # ball may drift this far toward the upper finger before it's free
CRADLE_J6 = 90.0            # wrist roll: fingers above/below the ball (jaws assumed along flange y)
HOME_J = [0.0, 13.26, -44.18, 0.0, -57.29, 0.0]   # measured at home (Oct 29, control tab)
LIMITS = [(-360, 360), (-132, 120), (-242, 3.5), (-360, 360), (-124, 124), (-360, 360)]
MARGIN = 6.0
SPEEDS = (120.0, 180.0)


def frame(q):
    M = np.eye(4)
    for (rpy, xyz), qq in zip(ts.URDF_JOINTS, np.radians(q)):
        M = M @ ts._tf(ts._rpy(*rpy), xyz)
        c, s = math.cos(qq), math.sin(qq)
        M = M @ ts._tf(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), (0, 0, 0))
    return M[:3, 3], M[:3, 2]


def ball(q, off):
    f, t = frame(q)
    return f + off * t, t


def unit(v):
    return v / np.linalg.norm(v)


def exit_margin(start, u, D, w, off):
    """Simulate the ball (free flight from the start of braking) against the braking hand.
    Returns (mm past EXIT_TIP_MM it reaches, max drift toward the upper finger before exiting, exit time s)."""
    ta = w * w / (2 * ACCEL)
    th_rel = D - ta
    q_rel = start + u * th_rel
    p0, t0 = ball(q_rel, off)
    pp, _ = ball(q_rel + u * 0.5, off)
    pm, _ = ball(q_rel - u * 0.5, off)
    v = (pp - pm) / 1.0 * w                       # mm/s
    t_brake = w / ACCEL
    best, drift = -1e9, -1e9
    for t in np.arange(0.0, t_brake + 0.35, 0.002):
        th = th_rel + (w * t - 0.5 * ACCEL * t * t if t < t_brake else w * t_brake - 0.5 * ACCEL * t_brake ** 2)
        fl, tool = frame(start + u * th)
        pb = p0 + v * t - np.array([0, 0, 0.5 * G * t * t])
        rel = pb - fl
        along = rel @ tool
        up = np.array([0, 0, 1.0]) - tool[2] * tool
        n = np.linalg.norm(up)
        side = 0.0 if n < 1e-6 else rel @ (up / n)
        drift = max(drift, side)
        best = max(best, along - EXIT_TIP_MM)
        if along >= EXIT_TIP_MM:
            return best, drift, t
    return best, drift, None


def joints(x):
    s2, s3, s5, d2, d3, d5 = x
    start = np.array([0.0, s2, s3, 0.0, s5, CRADLE_J6])
    delta = np.array([0.0, d2, d3, 0.0, d5, 0.0])
    return start, delta


def evaluate(x, detail=False):
    """(score, info). score < 0 is better (negative landing distance) when feasible."""
    start, delta = joints(x)
    D = np.max(np.abs(delta))
    pen = 0.0
    if D < 1e-6:
        return 1e6, None
    u = delta / D                              # deg of each joint per deg of the lead joint
    end = start + delta
    for q in (start, end):
        for j, (lo, hi) in enumerate(LIMITS):
            pen += max(0, lo + MARGIN - q[j]) + max(0, q[j] - hi + MARGIN)
    h = 0.5
    info = {}
    for off in BALL_OFFSETS:
        _, t0 = ball(start, off)
        pen += max(0, 0.26 - t0[2]) * 400           # tool >= 15 deg above horizontal at rest
        for w in SPEEDS:
            ta = w * w / (2 * ACCEL)                # deg of lead joint to reach speed
            if D < 2 * ta + 6:
                pen += (2 * ta + 6 - D) * 20
                continue
            th_rel = D - ta
            # felt acceleration into the palm: accel phase (alpha) + cruise (centripetal)
            for th in np.linspace(0, th_rel, 9):
                q = start + u * th
                p0, t = ball(q, off)
                pp, _ = ball(q + u * h, off)
                pm, _ = ball(q - u * h, off)
                dp = (pp - pm) / (2 * h)                       # mm per deg
                d2p = (pp - 2 * p0 + pm) / (h * h)             # mm per deg^2
                wcur = math.sqrt(min(w * w, 2 * ACCEL * max(th, 1e-9)))
                alpha = ACCEL if th < ta else 0.0
                a = d2p * wcur ** 2 + dp * alpha               # mm/s^2
                f = a + np.array([0, 0, G])
                c = f @ t / np.linalg.norm(f)
                pen += max(0, 0.30 - c) * 200
            q = start + u * th_rel
            p0, t = ball(q, off)
            pp, _ = ball(q + u * h, off)
            pm, _ = ball(q - u * h, off)
            v = (pp - pm) / (2 * h) * w                        # mm/s
            elev = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
            ang = math.degrees(math.acos(np.clip(unit(v) @ t, -1, 1)))
            pen += max(0, 30 - elev) * 5 + max(0, elev - 50) * 5 + max(0, ang - 20) * 5
            pen += max(0, v[0] * -1) * 0.01                    # must throw forward
            if pen == 0:
                past, drift, t_exit = exit_margin(start, u, D, w, off)
                pen += max(0, -past) * 2 + max(0, drift - TOP_GAP_MM) * 2
                if t_exit is None:
                    pen += 20
            # ballistic landing at rim height
            a_, b_, c_ = -0.5 * G, v[2], p0[2] - RIM_Z
            disc = b_ * b_ - 4 * a_ * c_
            if disc < 0:
                pen += 100
                continue
            tf = (-b_ - math.sqrt(disc)) / (2 * a_)
            land = math.hypot(p0[0] + v[0] * tf, p0[1] + v[1] * tf)
            info[(off, w)] = dict(land=land, elev=elev, ang=ang, speed=np.linalg.norm(v) / 1000,
                                  release=p0.round(0), ta=ta)
    if pen > 0:
        return 1e3 + pen, info
    for a_, b_ in ((HOME_J[:5] + [CRADLE_J6], start), (start, end)):
        probs = ts.path_problems(list(a_), list(b_))
        if probs:
            return 1e3 + 50 * len(probs), info
    return -min(info[(o, 180.0)]["land"] for o in BALL_OFFSETS), info


def main():
    bounds = [(-120, 110), (-230, -5), (-115, 115), (-110, 110), (-110, 110), (-110, 110)]
    best = differential_evolution(lambda x: evaluate(x)[0], bounds, seed=3, popsize=15, maxiter=160,
                                  tol=1e-7, polish=False, workers=1, updating="immediate")
    x = best.x
    score, info = evaluate(x, detail=True)
    start, delta = joints(x)
    print("feasible:", score < 0, "score", round(score, 1))
    print("START_J (J2, J3, J5) =", tuple(np.round(start[[1, 2, 4]], 1)))
    print("SWING {1: d2, 2: d3, 4: d5} =", {1: round(delta[1], 1), 2: round(delta[2], 1), 4: round(delta[4], 1)})
    u = delta / np.max(np.abs(delta))
    for k in sorted(info):
        i = info[k]
        past, drift, t_exit = exit_margin(start, u, np.max(np.abs(delta)), k[1], k[0])
        print(f"    exit: {past:+.0f} mm past the tips, drift {drift:.1f} mm, at {t_exit}")
        print(f"  ball +{k[0]:.0f} mm, {k[1]:.0f} deg/s: lands {i['land']:.0f} mm, {i['speed']:.2f} m/s, "
              f"launch {i['elev']:.0f} deg, fingers {i['ang']:.0f} deg off, release {i['release']}")
    return x


if __name__ == "__main__":
    main()
