import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import throw  # noqa: E402
import throw_fit  # noqa: E402
import throw_safety  # noqa: E402


@pytest.mark.parametrize("preset", list(throw.PRESETS))
@pytest.mark.parametrize("yaw", [-40, 0, 40])
def test_presets_clear_limits_and_obstacles(preset, yaw):
    pl = throw.plan(1000, preset, yaw_deg=yaw)          # raises if unsafe
    assert pl["preset"] == preset and pl["speed_dps"] <= throw.MAX_SPEED


def test_far_is_servo_power_is_direct():
    assert throw.plan(900)["direct"] is False
    assert throw.plan(1500, "power")["direct"] is True


def test_speed_follows_table_and_learned_ratio():
    assert abs(throw.speed_for(953, "far") - 165) < 0.5
    assert throw.speed_for(5000, "far") == 180.0
    assert throw.speed_for(100, "far") == throw.FAR["min_speed"]
    # ball flies 10 % farther than the model -> same cup needs less speed
    assert throw.speed_for(953, "far", ratio=1.1) < 150
    for s in (105, 140, 180):
        assert abs(throw.speed_for(throw.model_landing(s, "far"), "far") - s) < 1e-6


def test_far_never_opens_before_the_swing_power_never_before_start():
    for s in (100, 140, 180):
        assert throw.release_delay(s, "far") >= 0.10
        assert throw.release_delay(s, "power") >= 0.0


def test_unsafe_requests_are_refused():
    with pytest.raises(ValueError):
        throw.plan(900, yaw_deg=60)
    with pytest.raises(ValueError):
        throw.plan(900, speed=250)
    with pytest.raises(ValueError):
        throw.plan(900, scale=2)
    with pytest.raises(ValueError):
        throw.plan(900, "whip")


def test_path_check_catches_a_swing_into_the_table():
    start = [0, -57.4, -65.5, 0, -79.5, 0]              # POWER start: ball 9 cm above the table plane
    down = [0, -77.4, -65.5, 0, -79.5, 0]               # shoulder 20 deg further back: ball reaches z~8
    probs = throw_safety.path_problems(start, down)
    assert probs and "table" in probs[0]
    with pytest.raises(RuntimeError):
        throw.check_path(start, down, "swing")


def test_path_check_uses_the_live_ceiling():
    assert throw_safety.CEILING_Z == 1200


def test_yaw_for_cup_points_at_cup():
    assert abs(throw.yaw_for_cup(480, 0)) < 1e-9
    assert abs(throw.yaw_for_cup(480, 480) - 45) < 1e-6
    assert throw.yaw_for_cup(480, -85) < 0


def doc(preset, speed, cup, **result):
    return {"phase": "throw", "throw": {"preset": preset, "throw_speed": speed, "distance_mm": cup, "scale": 1},
            "result": {"hit": result.pop("hit", False), **result}}


def test_landing_rules():
    assert throw_fit.landing_mm(doc("far", 180, 786, long_cm=20)) == 986
    assert throw_fit.landing_mm(doc("far", 165, 955, long_cm=0)) is None         # legacy 0 = not measured
    assert throw_fit.landing_mm(doc("far", 165, 955, long_cm=0, outcome="rim_out")) == 955
    assert throw_fit.landing_mm(doc("far", 165, 955, hit=True)) == 955
    assert throw_fit.landing_mm(doc("far", 165, 955, outcome="dropped")) is None
    assert throw_fit.landing_mm(doc("power", 180, 10, long_cm=50)) is None       # hand-typed bogus distance
    d = doc("far", 165, 955, long_cm=12)
    d["exclude_from_fit"] = True
    assert throw_fit.landing_mm(d) is None


def test_fit_uses_median_and_needs_data():
    m = throw.model_landing(180, "far")
    docs = [doc("far", 180, round(m * 1.1), hit=True) for _ in range(4)] + [doc("far", 180, 400, long_cm=200)]
    assert abs(throw_fit.fit(docs, "far") - 1.1) < 0.01         # one wild throw doesn't move the median
    assert throw_fit.fit(docs[:2], "far") == 1.0                 # too few throws
    assert throw_fit.fit(docs, "power") == 1.0                   # other preset's data ignored
    wild = [doc("far", 180, round(m * 2), hit=True) for _ in range(5)]
    assert throw_fit.fit(wild, "far") == throw_fit.CLAMP[1]
