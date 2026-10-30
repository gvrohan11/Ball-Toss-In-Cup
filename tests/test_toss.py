"""Offline checks for toss_ball.py physics and safety (no robot)."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import toss_ball as t  # noqa: E402


def test_release_speed_lands_at_distance():
    d = t.design(400)
    th, v, h = math.radians(d["theta_deg"]), d["v"], d["h_mm"] / 1000
    vx, vz = v * math.cos(th), v * math.sin(th)
    tf = (vz + math.sqrt(vz * vz + 2 * t.G * h)) / t.G
    assert abs(vx * tf - 0.4) < 1e-6


def test_jacobian_matches_finite_difference():
    d = t.design(400)
    q, dq, eps = np.array(d["release_q"]), np.zeros(6), 1e-4
    dq[2], dq[4] = t.FORWARD[2], t.FORWARD[4] * t.K
    v = (t.fk(q + dq * math.degrees(eps))["ball"] - t.fk(q - dq * math.degrees(eps))["ball"]) / (2 * eps)
    v = v * math.radians(d["omega_deg"]) / 1000
    assert abs(np.linalg.norm(v) - d["v"]) < 1e-3


def test_short_toss_is_synchronized_and_safe():
    d = t.design(400)
    assert not d["long_toss"]
    assert all(p["start_delay"] < 1e-3 for p in d["plan"].values())
    assert t.check(d) == []


def test_low_accel_triggers_backswing():
    d = t.design(700, alpha_max_deg=300)
    assert d["long_toss"] and d["plan"][2]["backswing"] > 0


def test_out_of_range_is_refused():
    assert any("deg/s" in p for p in t.check(t.design(1600)))
