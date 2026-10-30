"""Swept-path check for throws (direct joint moves skip Viam's obstacle checks).

Forward kinematics from the xArm850 URDF (from Rohan's toss_ball.py, origin/rohan/tossball).
`path_problems(q_from, q_to)` samples the straight joint-space path and reports any point of
the wrist / flange / gripper / ball that comes within CLEARANCE_MM of the table, ceiling or walls.
Joint angles are Viam 0-5, degrees.

    python throw_safety.py      # checks every preset in throw.py, no robot
"""
import math

import numpy as np

# (rpy of the joint origin, xyz of the joint origin in mm); every joint rotates about its local z.
URDF_JOINTS = [
    ((0, 0, 0), (0, 0, 364)),
    ((1.5708, -1.5708, 0), (0, 0, 0)),
    ((-3.1416, 0, -1.5708), (390, 0, 0)),
    ((-1.5708, 0, 0), (150, 426, 0)),
    ((-1.5708, 0, 0), (0, 0, 0)),
    ((1.5708, 0, 0), (0, -90, 0)),
]
GRIPPER_FRAME_MM = 105   # Viam "gripper" frame = flange + 105 mm
BALL_CENTER_MM = 175     # held ball center along the tool axis from the flange
BALL_RADIUS_MM = 20

# Workcell, world frame mm. Matches the LIVE sfhack11 config read Oct 29 21:50:
# table box center z=-123, 200 tall -> top -23; ceiling center z=1250, 100 tall -> bottom 1200;
# wall-back center x=-550, 50 thick -> face -525; wall-left center y=650 -> face 625.
TABLE_TOP_Z = -23
CEILING_Z = 1200
WALL_BACK_X = -525
WALL_LEFT_Y = 625
CLEARANCE_MM = 50


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _tf(R, t):
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, t
    return M


def fk(q_deg):
    """Points (world mm) of wrist, flange, gripper and held ball for joint angles q_deg."""
    M, wrist = np.eye(4), None
    for i, ((rpy, xyz), q) in enumerate(zip(URDF_JOINTS, np.radians(q_deg))):
        M = M @ _tf(_rpy(*rpy), xyz)
        if i == 4:
            wrist = M[:3, 3].copy()
        c, s = math.cos(q), math.sin(q)
        M = M @ _tf(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]), (0, 0, 0))
    tool, flange = M[:3, 2], M[:3, 3]
    return {"wrist": wrist, "flange": flange.copy(), "gripper": flange + GRIPPER_FRAME_MM * tool,
            "ball": flange + BALL_CENTER_MM * tool}


def path_problems(q_from, q_to, n=60):
    """Problems (strings) along the straight joint path q_from -> q_to; [] = clear."""
    bad = []
    for s in np.linspace(0, 1, n):
        q = [a + s * (b - a) for a, b in zip(q_from, q_to)]
        for name, p in fk(q).items():
            floor = TABLE_TOP_Z + CLEARANCE_MM + (BALL_RADIUS_MM if name == "ball" else 0)
            if p[2] < floor:
                bad.append(f"{name} at z={p[2]:.0f} too close to the table ({s:.0%} along the path)")
            if p[2] > CEILING_Z - CLEARANCE_MM:
                bad.append(f"{name} at z={p[2]:.0f} too close to the ceiling ({s:.0%})")
            if p[0] < WALL_BACK_X + CLEARANCE_MM or abs(p[1]) > WALL_LEFT_Y - CLEARANCE_MM:
                bad.append(f"{name} at ({p[0]:.0f}, {p[1]:.0f}) too close to a wall ({s:.0%})")
        if bad:
            return bad[:3]
    return bad


if __name__ == "__main__":
    import throw
    for name, p in throw.PRESETS.items():
        for yaw in (0, -30, 30):
            start = [yaw, p["start_j"][0], p["start_j"][1], 0.0, p["start_j"][2], 0.0]
            end = list(start)
            for j, d in p["swing"].items():
                end[j] += d
            probs = path_problems(start, end)
            print(f"{name:6s} yaw {yaw:+3d}: {'OK' if not probs else probs}  "
                  f"ball start {np.round(fk(start)['ball'])} end {np.round(fk(end)['ball'])}")
