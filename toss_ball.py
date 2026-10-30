"""toss_ball.py - toss the held ball forward with a two-joint (elbow + wrist) swing.

No cup aiming: the arm faces +x (joint 0 = 0) and throws the ball to land a chosen
horizontal distance past the release point.

    python toss_ball.py                       # DRY RUN: plan + safety checks, moves nothing
    python toss_ball.py --distance 400        # dry run for a 40 cm toss
    python toss_ball.py --go                  # LIVE toss of a ball already in the gripper
    python toss_ball.py --pick --go           # LIVE: pick_ball (find + pick), then toss

Someone on the E-stop. The swing is a DIRECT joint move (no motion planning), so this
script checks the whole swept path itself against the table, ceiling and back wall.

Joints use Viam's 0-5 numbering: 0 base yaw, 1 shoulder, 2 elbow, 3 forearm roll,
4 wrist pitch, 5 wrist roll. Joint 3 stays at 0 so joints 2 and 4 swing in the same
vertical plane; joints 0, 1, 3, 5 are locked during the throw.

Physics (see plan_throw):
 1. Ballistics: release point p (from forward kinematics), launch angle theta (the
    direction of the TCP velocity the swing produces), landing height = table + ball
    radius.  v^2 = g D^2 / (2 cos^2(theta) (D tan(theta) + h)).
 2. Split v over the joints with a fixed ratio k = w4/w2:  w2 = v / (r_e + k r_w),
    where r_e, r_w are the elbow / wrist lever arms projected on the launch direction
    (|dp/dq| from the URDF Jacobian, not hand-measured lengths).
 3. The arm waits in a HOLD pose a runway Delta behind the release pose. Each joint
    accelerates from rest at alpha_i = w_i^2 / (2 Delta_i) so it hits full speed exactly
    at the release pose. If alpha_i > alpha_max: long toss, back up by
    w_i^2/(2 alpha_max) - Delta_i slowly, pause, then throw at alpha_max.
 4. Open the gripper tau early (its latency). After release the joints coast at full
    speed for a short follow-through and then brake over w^2/(2 alpha).

Execution detail: the viam:ufactory module runs one joint move as a single trapezoid
in joint space, with every joint scaled to the joint that moves furthest. Choosing
Delta_4 = k * Delta_2 makes that exact: both joints get ratio k in speed AND
acceleration, both reach the release pose at the same instant, and the stagger from
plan_throw is zero (asserted).
"""
import argparse
import asyncio
import math
import sys
import time

import numpy as np

G = 9.81  # m/s^2

# ---------------- arm geometry (viam-ufactory-xarm arm/uf850.urdf, use_urdfs: true) ----
# (rpy of the joint origin, xyz of the joint origin in mm); every joint rotates about its local z.
URDF_JOINTS = [
    ((0, 0, 0), (0, 0, 364)),
    ((1.5708, -1.5708, 0), (0, 0, 0)),
    ((-3.1416, 0, -1.5708), (390, 0, 0)),
    ((-1.5708, 0, 0), (150, 426, 0)),
    ((-1.5708, 0, 0), (0, 0, 0)),
    ((1.5708, 0, 0), (0, -90, 0)),
]
JOINT_LIMITS_DEG = [(-359, 359), (-132, 132), (-242, 3.5), (-359, 359), (-124, 124), (-359, 359)]
LIMIT_MARGIN_DEG = 3
GRIPPER_FRAME_MM = 105   # Viam "gripper" frame: flange + 105 mm (config/sfhack11.json)
BALL_CENTER_MM = 175     # ball center along the tool axis from the flange (grab z=70, ball center ~-3)
BALL_RADIUS_MM = 20

# ---------------- workcell (world frame, mm; from the Viam obstacles) -------------------
TABLE_TOP_Z = -23
CEILING_Z = 1000          # bottom of the ceiling obstacle box
WALL_BACK_X = -525        # front face of wall-back
WALL_LEFT_Y = 625         # face of wall-left
CLEARANCE_MM = 60         # keep ball / flange / wrist this far from every surface

# ---------------- throw design ---------------------------------------------------------
# Release pose: forearm reaching forward-down, gripper tilted 35 deg below horizontal;
# the elbow + wrist swing then launches the ball forward-up at ~45 deg, ball released
# ~33 cm above the table, ~54 cm in front of the base.
RELEASE_Q = [0.0, 0.0, -40.0, 0.0, -5.0, 0.0]
WRIST_ROLL_DEG = 0.0      # joint 5. 0 = fingers close along world y (assumed jaw axis = flange y),
                          # so the ball leaves through the open side of the jaws, not into a finger.
FORWARD = {2: -1, 4: +1}  # joint direction that swings the ball forward/up (from the Jacobian)
K = 1.0                   # w4 / w2
DELTA_DEG = {2: 20.0, 4: 20.0 * K}  # holding-pose runway; Delta_4 = k * Delta_2 (see docstring)
ALPHA_MAX_DEG = 1145.0    # module maximum = 20 rad/s^2. Lower it here to make long tosses back up.
OMEGA_MAX_DEG = 180.0     # module maximum
FOLLOW_CRUISE_DEG = 10.0  # constant-speed coast after release: a late release still gets full speed
GRIPPER_LATENCY_S = 0.10  # tau: open command -> jaws free the ball (measure and update!)
MODULE_START_S = 0.02     # request -> first joint step (module ready-state check)
SLOW_SPEED_DEG = 30.0     # positioning / backswing moves
SLOW_ACCEL_DEG = 100.0
NORMAL_SPEED_DEG = 60.0
DEFAULT_DISTANCE_MM = 400  # landing distance past the release point (horizontal)

HOME = dict(x=350, y=0, z=445, o_x=0, o_y=0, o_z=-1, theta=180)
ARM, GRIPPER = "arm", "gripper"
HOLDING_MIN_POS = 50


# ============================== kinematics ==============================================
def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = map(lambda f: f, (math.cos(r), math.sin(r), math.cos(p),
                                               math.sin(p), math.cos(y), math.sin(y)))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _tf(R, t):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def fk(q_deg):
    """Forward kinematics. Returns dict: flange 4x4, joint (origin, axis) list, points."""
    M, axes = np.eye(4), []
    for (rpy, xyz), q in zip(URDF_JOINTS, np.radians(q_deg)):
        M = M @ _tf(_rpy(*rpy), xyz)
        axes.append((M[:3, 3].copy(), M[:3, 2].copy()))
        c, s = math.cos(q), math.sin(q)
        M = M @ _tf(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), (0, 0, 0))
    tool = M[:3, 2]
    return {"flange": M, "axes": axes, "tool": tool,
            "ball": M[:3, 3] + BALL_CENTER_MM * tool,
            "points": {"elbow": axes[2][0], "wrist": axes[4][0], "flange": M[:3, 3],
                       "gripper": M[:3, 3] + GRIPPER_FRAME_MM * tool,
                       "ball": M[:3, 3] + BALL_CENTER_MM * tool}}


def lever(q_deg, j):
    """dp_ball/dq_j in mm/rad (column j of the position Jacobian)."""
    f = fk(q_deg)
    o, a = f["axes"][j]
    return np.cross(a, f["ball"] - o)


# ============================== physics =================================================
def release_speed(distance_mm, theta, h_mm):
    """Speed (m/s) to land distance_mm horizontally away, launching at theta (rad)
    from h_mm above the landing height."""
    D, h = distance_mm / 1000, h_mm / 1000
    den = 2 * math.cos(theta) ** 2 * (D * math.tan(theta) + h)
    if den <= 0:
        raise ValueError("target unreachable at this launch angle")
    return math.sqrt(G * D * D / den)


def plan_throw(v, delta, r_e, r_w, k, alpha_max):
    """The plan from the design note: speeds, accelerations, backswings, start delays.
    Units: v in m/s, radii in m, angles in rad, alpha in rad/s^2."""
    w2 = v / (r_e + k * r_w)
    w = {2: w2, 4: k * w2}
    plan = {}
    for j in (2, 4):
        alpha_needed = w[j] ** 2 / (2 * delta[j])
        if alpha_needed <= alpha_max[j]:      # short toss: forward only
            plan[j] = dict(omega=w[j], alpha=alpha_needed, backswing=0.0)
        else:                                  # long toss: back up first
            runup = w[j] ** 2 / (2 * alpha_max[j])
            plan[j] = dict(omega=w[j], alpha=alpha_max[j], backswing=runup - delta[j])
    # stagger start times so both joints reach the release pose together
    t = {j: plan[j]["omega"] / plan[j]["alpha"] for j in plan}
    t_max = max(t.values())
    for j in plan:
        plan[j]["start_delay"] = t_max - t[j]
    return plan


def design(distance_mm, alpha_max_deg=ALPHA_MAX_DEG, k=K):
    """Everything for one toss: ballistics -> plan_throw -> one synchronized joint move."""
    rel = list(RELEASE_Q)
    rel[5] = WRIST_ROLL_DEG
    f = fk(rel)
    c2, c4 = FORWARD[2] * lever(rel, 2), FORWARD[4] * lever(rel, 4)
    vel_dir = c2 + k * c4                      # TCP velocity per rad/s of joint 2
    u = vel_dir / np.linalg.norm(vel_dir)
    theta = math.atan2(u[2], math.hypot(u[0], u[1]))
    r_e, r_w = (c2 @ u) / 1000, (c4 @ u) / 1000  # m, projected on the launch direction
    h = f["ball"][2] - (TABLE_TOP_Z + BALL_RADIUS_MM)
    v = release_speed(distance_mm, theta, h)

    delta = {j: math.radians(DELTA_DEG[j]) for j in (2, 4)}
    amax = {2: math.radians(alpha_max_deg), 4: math.radians(alpha_max_deg)}
    plan = plan_throw(v, delta, r_e, r_w, k, amax)

    # One synchronized move: the joint with the larger speed leads; the other joint's
    # runway, accel and speed are the lead's times k (so it never exceeds alpha_max).
    lead = max(plan, key=lambda j: plan[j]["omega"])
    ratio = {j: plan[j]["omega"] / plan[lead]["omega"] for j in plan}
    w = plan[lead]["omega"]
    runup = delta[lead] + plan[lead]["backswing"]          # rad, from rest to release
    alpha = w * w / (2 * runup)
    cruise = max(math.radians(FOLLOW_CRUISE_DEG), 0.15 * 2 * runup)  # module needs a plateau
    brake = w * w / (2 * alpha)
    t_release = 2 * runup / w                                # rest -> release, constant alpha

    if plan[lead]["backswing"] == 0:
        for j in plan:
            assert plan[j]["start_delay"] < 1e-3, "Delta_4 must equal k * Delta_2 (one synchronized move)"

    def pose(offset_lead_rad):
        q = list(rel)
        for j in plan:
            q[j] += FORWARD[j] * math.degrees(offset_lead_rad * ratio[j])
        return q

    return dict(
        distance_mm=distance_mm, v=v, theta_deg=math.degrees(theta), h_mm=h,
        release_ball=f["ball"], r_e=r_e, r_w=r_w, k=k, plan=plan, lead=lead,
        omega_deg=math.degrees(w), alpha_deg=math.degrees(alpha),
        long_toss=plan[lead]["backswing"] > 0, t_release=t_release,
        release_q=rel, hold_q=pose(-delta[lead]), start_q=pose(-runup), end_q=pose(cruise + brake),
        cruise_deg=math.degrees(cruise), brake_deg=math.degrees(brake),
    )


# ============================== safety checks ===========================================
def check(d):
    """Return a list of problems; empty means OK to throw."""
    bad = []
    for j in d["plan"]:
        if abs(math.degrees(d["plan"][j]["omega"])) > OMEGA_MAX_DEG:
            bad.append(f"joint {j} needs {math.degrees(d['plan'][j]['omega']):.0f} deg/s > {OMEGA_MAX_DEG} (out of range)")
    for name in ("hold_q", "start_q", "end_q"):
        for j, (lo, hi) in enumerate(JOINT_LIMITS_DEG):
            if not lo + LIMIT_MARGIN_DEG <= d[name][j] <= hi - LIMIT_MARGIN_DEG:
                bad.append(f"{name} joint {j} = {d[name][j]:.1f} deg outside limits [{lo}, {hi}]")
    bad += path_problems(d["start_q"], d["end_q"])
    return bad


def path_problems(q_from, q_to, n=60):
    """Sample the straight joint-space path and check elbow/wrist/flange/ball clearance."""
    bad = []
    for s in np.linspace(0, 1, n):
        q = [a + s * (b - a) for a, b in zip(q_from, q_to)]
        for name, p in fk(q)["points"].items():
            if name == "elbow":
                continue  # the elbow is on the upper arm, far from everything
            if p[2] < TABLE_TOP_Z + CLEARANCE_MM + (BALL_RADIUS_MM if name == "ball" else 0):
                bad.append(f"{name} at z={p[2]:.0f} hits the table ({s:.0%} along the path)")
            if p[2] > CEILING_Z - CLEARANCE_MM:
                bad.append(f"{name} at z={p[2]:.0f} hits the ceiling ({s:.0%})")
            if p[0] < WALL_BACK_X + CLEARANCE_MM or abs(p[1]) > WALL_LEFT_Y - CLEARANCE_MM:
                bad.append(f"{name} at ({p[0]:.0f},{p[1]:.0f}) hits a wall ({s:.0%})")
        if bad:
            return bad[:3]
    return bad


def describe(d):
    p = d["plan"]
    lines = [
        f"target: land {d['distance_mm']:.0f} mm past the release point",
        f"release: ball at ({d['release_ball'][0]:.0f}, {d['release_ball'][1]:.0f}, {d['release_ball'][2]:.0f}) mm, "
        f"{d['h_mm']:.0f} mm above landing height, launch {d['theta_deg']:.1f} deg",
        f"ballistics: v = {d['v']:.2f} m/s;  r_e = {d['r_e'] * 1000:.0f} mm, r_w = {d['r_w'] * 1000:.0f} mm, k = {d['k']}",
    ]
    for j in sorted(p):
        lines.append(f"  joint {j}: w = {math.degrees(p[j]['omega']):6.1f} deg/s, alpha = {math.degrees(p[j]['alpha']):6.1f} deg/s^2, "
                     f"backswing = {math.degrees(p[j]['backswing']):5.1f} deg, start delay = {p[j]['start_delay'] * 1000:.0f} ms")
    lines += [
        f"{'LONG toss (backswing first)' if d['long_toss'] else 'SHORT toss (forward only)'}: lead joint {d['lead']} "
        f"at {d['omega_deg']:.1f} deg/s, {d['alpha_deg']:.0f} deg/s^2",
        f"release {d['t_release'] * 1000:.0f} ms after the swing starts; gripper open sent at "
        f"{(d['t_release'] + MODULE_START_S - GRIPPER_LATENCY_S) * 1000:.0f} ms (tau = {GRIPPER_LATENCY_S * 1000:.0f} ms)",
        f"follow-through: coast {d['cruise_deg']:.1f} deg, brake {d['brake_deg']:.1f} deg",
        "joints: hold  " + _fmt(d["hold_q"]),
        "        start " + _fmt(d["start_q"]),
        "        rel   " + _fmt(d["release_q"]),
        "        end   " + _fmt(d["end_q"]),
    ]
    return "\n".join(lines)


def _fmt(q):
    return "[" + ", ".join(f"{v:6.1f}" for v in q) + "]"


# ============================== robot =========================================================
async def _joints(arm):
    return list((await arm.get_joint_positions()).values)


async def _move_joints(arm, q, speed, accel):
    from viam.proto.component.arm import JointPositions
    await arm.move_to_joint_positions(JointPositions(values=q),
                                      extra={"speed_d": speed, "acceleration_d": accel})


async def verify_kinematics(arm):
    """Our FK must match the arm's own flange pose before we trust any path check."""
    q = await _joints(arm)
    p = await arm.get_end_position()
    mine = fk(q)["flange"][:3, 3]
    err = math.dist(mine, (p.x, p.y, p.z))
    print(f"  FK check: joints {_fmt(q)} -> ours {np.round(mine)} vs arm ({p.x:.0f}, {p.y:.0f}, {p.z:.0f}), err {err:.1f} mm")
    if err > 10:
        raise RuntimeError("forward kinematics disagree with the arm by >10 mm; not throwing")
    return q


async def toss(robot, distance_mm, go):
    from viam.components.arm import Arm
    from viam.components.gripper import Gripper
    from viam.proto.common import Pose, PoseInFrame
    from viam.services.motion import MotionClient

    d = design(distance_mm)
    print(describe(d))
    problems = check(d)
    if problems:
        print("REFUSING to throw:\n  " + "\n  ".join(problems))
        return {"thrown": False, "problems": problems}
    print("safety checks: OK (limits, speeds, swept path vs table/ceiling/walls)")

    arm, gripper = Arm.from_robot(robot, ARM), Gripper.from_robot(robot, GRIPPER)
    motion = MotionClient.from_robot(robot, "builtin")
    q_now = await verify_kinematics(arm)
    pos = (await gripper.do_command({"get": True})).get("pos")
    print(f"  gripper position {pos} ({'holding' if pos is not None and pos > HOLDING_MIN_POS else 'EMPTY?'})")

    async def home():
        print("  -> home (planned)")
        if go:
            await motion.move(component_name=GRIPPER,
                              destination=PoseInFrame(reference_frame="world", pose=Pose(**HOME)))

    if not go:
        probs = path_problems(q_now, d["hold_q"])
        print(f"  (dry run) current -> hold path: {'OK' if not probs else probs}")
        return {"thrown": False, "dry_run": True}

    released = False
    try:
        # 1) get to a known-safe pose with the planner, then a slow checked joint move to HOLD
        await home()
        q_now = await _joints(arm)
        probs = path_problems(q_now, d["hold_q"])
        if probs:
            raise RuntimeError(f"home -> hold path not clear: {probs}")
        print("  -> hold pose (slow joint move)")
        await _move_joints(arm, d["hold_q"], SLOW_SPEED_DEG, SLOW_ACCEL_DEG)
        # 2) long toss: back up slowly, settle
        if d["long_toss"]:
            print("  -> backswing (slow)")
            await _move_joints(arm, d["start_q"], SLOW_SPEED_DEG, SLOW_ACCEL_DEG)
        await asyncio.sleep(0.5)  # settle: start from rest
        # 3) throw: one synchronized move; open the gripper tau before the release instant
        t_open = d["t_release"] + MODULE_START_S - GRIPPER_LATENCY_S
        print(f"  THROW: {d['omega_deg']:.0f} deg/s, {d['alpha_deg']:.0f} deg/s^2, open at {t_open * 1000:.0f} ms")
        t0 = time.perf_counter()
        swing = asyncio.create_task(_move_joints(arm, d["end_q"], d["omega_deg"], d["alpha_deg"]))
        await asyncio.sleep(max(0.0, t_open - (time.perf_counter() - t0)))
        t_sent = time.perf_counter() - t0
        await gripper.open()
        released = True
        await swing
        print(f"  open sent at {t_sent * 1000:.0f} ms; swing done at {(time.perf_counter() - t0) * 1000:.0f} ms")
    except Exception:
        try:
            await arm.stop()
        except Exception:
            pass
        raise
    finally:
        if released:
            await home()
        else:
            print("  ball NOT released; leaving it in the gripper")
    return {"thrown": released, "distance_mm": distance_mm, "v": d["v"],
            "omega_deg": d["omega_deg"], "alpha_deg": d["alpha_deg"], "t_open_s": t_sent if released else None}


async def _main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--distance", type=float, default=DEFAULT_DISTANCE_MM,
                    help="landing distance past the release point, mm (default %(default)s)")
    ap.add_argument("--pick", action="store_true", help="run pick_ball first (find + pick)")
    ap.add_argument("--go", action="store_true", help="actually move the arm")
    ap.add_argument("--offline", action="store_true", help="print the plan only, no robot connection")
    a = ap.parse_args()
    print("MODE:", "LIVE (arm will move)" if a.go else "DRY RUN (no motion)")
    if a.offline:
        d = design(a.distance)
        print(describe(d))
        print("checks:", check(d) or "OK")
        return
    from connection import connect
    robot = await connect()
    try:
        if a.pick:
            from viam.services.motion import MotionClient
            from pick_ball import find_ball, pick
            motion = MotionClient.from_robot(robot, "builtin")
            ball = await find_ball(robot, motion, a.go)
            if ball is None:
                print("no ball found; not tossing")
                return
            held = await pick(robot, ball, a.go)
            print("pick result:", {None: "dry run", True: "PICKED", False: "MISSED"}[held])
            if held is False:
                return
        print(await toss(robot, a.distance, a.go))
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(_main())
