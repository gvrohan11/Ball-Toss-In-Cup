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


# ---- cradle flow on a fake robot: open BEFORE the swing, nothing during it ----
import asyncio  # noqa: E402
from types import SimpleNamespace  # noqa: E402

HOME_J = [0.0, 13.26, -44.18, 0.0, -57.29, 0.0]


class FakeArm:
    def __init__(self, log):
        self.log, self.q = log, list(HOME_J)

    async def get_end_position(self):
        return SimpleNamespace(x=350.5, y=0.0, z=550.7)

    async def get_joint_positions(self):
        return SimpleNamespace(values=list(self.q))

    async def move_to_joint_positions(self, jp, extra=None):
        self.log.append(("move", [round(v, 1) for v in jp.values], extra))
        self.q = list(jp.values)

    async def do_command(self, cmd):
        self.log.append(("arm_cmd", cmd))
        return {}


class FakeGripper:
    def __init__(self, log):
        self.log = log

    async def do_command(self, cmd):
        self.log.append(("grip_cmd", cmd))
        return {"pos": 372} if "get" in cmd else {}

    async def open(self):
        self.log.append(("open",))

    async def grab(self):
        self.log.append(("grab",))


@pytest.fixture
def fake(monkeypatch):
    log = []
    arm, grip = FakeArm(log), FakeGripper(log)
    monkeypatch.setattr(throw.Arm, "from_robot", staticmethod(lambda r, n: arm))
    monkeypatch.setattr(throw.Gripper, "from_robot", staticmethod(lambda r, n: grip))

    class Motion:
        async def move(self, *a, **k):
            log.append(("home",))
            return True
    monkeypatch.setattr(throw.MotionClient, "from_robot", staticmethod(lambda r, n: Motion()))
    real_sleep = asyncio.sleep
    monkeypatch.setattr(throw.asyncio, "sleep", lambda s: real_sleep(0))
    return log


def test_cradle_opens_partially_at_swing_start(fake):
    pl = throw.plan(960, "cradle")
    assert pl["release_delay_s"] == 0.0 and 165 <= pl["speed_dps"] <= 180
    out = asyncio.run(throw.throw(object(), pl, True))
    moves = [i for i, e in enumerate(fake) if e[0] == "move"]
    sets = [i for i, e in enumerate(fake) if e[0] == "grip_cmd" and "set" in e[1]]
    assert len(moves) == 2 and sets and sets[0] > moves[1]           # open goes out right after the swing starts
    assert ("open",) not in fake                                       # never a full open
    assert fake[sets[0]][1]["set"] == 372 + throw.CRADLE_OPEN_DELTA
    assert fake[moves[0]][1][5] == throw.CRADLE["j6"]                  # wrist rolled: fingers above/below
    assert ("grip_cmd", {"set_gripper_speed": throw.GRIPPER_SPEED}) in fake
    last_arm = [e for e in fake if e[0] == "arm_cmd"][-1]
    assert last_arm == ("arm_cmd", {"set_speed": throw.NORMAL_SPEED})  # speed restored
    assert out["release_mode"] == "cradle" and out["held_pos"] == 372


def test_cradle_only_near_full_speed():
    assert throw.speed_for(800, "cradle") == 165.0
    lo, hi = throw.reach_mm("cradle")
    assert lo < 953 and hi > 983 and hi < 1100


def test_cradle_check_never_swings(fake):
    pl = throw.plan(1100, "cradle")
    out = asyncio.run(throw.throw(object(), pl, True, check_only=True))
    moves = [e for e in fake if e[0] == "move"]
    assert len(moves) == 2 and moves[1][1] == [round(v, 1) for v in HOME_J]   # wind-up, then straight back
    assert ("grab",) in fake and out["check_only"] is True
    assert not any(e[0] == "arm_cmd" and "set_acceleration" in e[1] for e in fake)


def test_partial_open_uses_set_not_open(fake):
    pl = throw.plan(900, "far")
    asyncio.run(throw.throw(object(), pl, True, partial=True))
    assert ("open",) not in fake
    assert ("grip_cmd", {"set": float(372 + throw.PARTIAL_OPEN_DELTA)}) in fake


def test_roll_tilt_and_open_overrides(fake):
    pl = throw.plan(960, "cradle", roll=0.0, tilt_up=8.0, cradle_open=40)
    assert pl["j6"] == 0.0 and pl["start_j"][4] == throw.CRADLE["start_j"][2] + 8.0 and pl["cradle_open"] == 40
    asyncio.run(throw.throw(object(), pl, True, check_only=True))
    wind = [e for e in fake if e[0] == "move"][0][1]
    assert wind[5] == 0.0 and abs(wind[4] - (throw.CRADLE["start_j"][2] + 8.0)) < 0.05
    assert ("grip_cmd", {"set": 412.0}) in fake
    with pytest.raises(ValueError):
        throw.plan(960, "cradle", tilt_up=40)
