"""
pick_ball.py - find the orange ball and pick it up.

Tested by hand on sfhack11 (Oct 29): this sequence picked the ball successfully.

    python pick_ball.py          # DRY RUN: finds the ball from the current view, prints targets, moves nothing
    python pick_ball.py --go     # actually moves the arm (someone on the E-stop!)

Credentials come from .env via connection.py (see .env.example).

How it works (what worked on the real arm):
 1. Gripper to the LOOK pose, detect the ball (2D box from ball-detector).
 2. Move so the CAMERA is directly over the ball and detect again
    (the estimate is much more accurate when the ball is near the image center).
 3. Pixel + ball size -> 3D point in the camera frame -> world frame.
 4. Apply the camera calibration correction (CAM_CORRECTION_MM).
 5. Hover above the ball, then SLOWLY (PICK_SPEED_DEG_S): go down to GRAB_Z, settle, close,
    WAIT for the fingers to stop, lift.
"""
import asyncio
import contextlib
import math
import sys

from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

import detection_log
from connection import connect

CAMERA = "cam"
BALL_DETECTOR = "ball-detector"
ARM = "arm"
GRIPPER = "gripper"

BALL_DIAMETER_MM = 40
# Gripper poses, world frame, mm. Gripper points straight down.
LOOK = (350, 0, 445)
# Wrist rotation for every move. 180 = camera image upright (far end of table at the top).
# The calibration below was MEASURED at theta=0; at theta=180 the camera sits on the
# other side of the gripper, so both offsets flip sign.
THETA = 180
MEASURE_HEIGHT_Z = 160      # gripper z when the camera is centered over the ball (camera ~250 mm above table)
_CAM_OFFSET_THETA0 = (83, 14)     # gripper xy minus camera xy, measured at theta=0
_CAM_CORRECTION_THETA0 = (0, -20)  # measured at theta=0: camera estimate was 2 cm off in +y
_SIGN = 1 if THETA == 0 else -1
CAM_OFFSET_XY = (_SIGN * _CAM_OFFSET_THETA0[0], _SIGN * _CAM_OFFSET_THETA0[1])
# Measured at theta=180 on the 2nd pick: flipped y (+20) was right, plus ~1.5 cm back toward the base in x.
CAM_CORRECTION_MM = _CAM_CORRECTION_THETA0 if THETA == 0 else (-15, 20)
HOVER_Z = 150
GRAB_Z = 70                 # fingers swing DOWN while closing; lower than ~60 drags them on the table
GRIP_BLOCKED_POS = 800      # fingers barely closed: they hit something (a plate, other balls)
HOLDING_MIN_POS = 50        # gripper position below this after closing = closed on nothing
# Pick slowly: from the hover point down, the close and the lift run at PICK_SPEED so the
# fingers don't knock the ball away or yank it. Transit moves keep the normal speed.
NORMAL_SPEED_DEG_S = 60     # speed_degs_per_sec in the sfhack11 arm config
PICK_SPEED_DEG_S = 15
SETTLE_S = 0.5              # let the arm stop swaying at GRAB_Z before closing


def pose(x, y, z):
    return PoseInFrame(reference_frame="world",
                       pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=THETA))


async def move(motion, x, y, z, go):
    print(f"  -> gripper to x={x:.0f} y={y:.0f} z={z:.0f}")
    if go:
        ok = await motion.move(component_name=GRIPPER, destination=pose(x, y, z))
        if ok is False:
            raise RuntimeError("Gripper motion did not complete; stopping")


def best_ball(dets):
    """Pick the detection that looks most like a ball: roughly square, biggest."""
    best = None
    for d in dets:
        w, h = d.x_max - d.x_min, d.y_max - d.y_min
        if w < 20 or h < 20:
            continue
        squareness = min(w, h) / max(w, h)
        if squareness < 0.6:
            continue
        if best is None or w * h > (best.x_max - best.x_min) * (best.y_max - best.y_min):
            best = d
    return best


BALL_CENTER_Z = -10 + BALL_DIAMETER_MM / 2   # real tabletop (-10) + radius
# Which ball to pick: "orange" (default, colour finder, the validated path), "white" or "any"
# (shape finder: depth sphere fit for the first look, circle search for the close-up).
BALL_COLOR = "orange"


async def _frame(cam):
    import cv2
    import numpy as np
    images, _ = await cam.get_images(filter_source_names=["color"])
    if not images:
        return None
    return cv2.imdecode(np.frombuffer(images[0].data, np.uint8), cv2.IMREAD_COLOR)


async def _color_and_depth(cam):
    import cv2
    import numpy as np
    images, _ = await cam.get_images(timeout=20)
    src = {im.name: im for im in images}
    if "color" not in src or "depth" not in src:
        raise RuntimeError("Need aligned color + depth from the camera")
    bgr = cv2.imdecode(np.frombuffer(src["color"].data, np.uint8), cv2.IMREAD_COLOR)
    depth = np.asarray(src["depth"].bytes_to_depth_array())
    if bgr is None or bgr.shape[:2] != depth.shape:
        raise RuntimeError("Color/depth images are not aligned")
    return bgr, depth


def _ball_items(balls):
    """Detection-log entries for ball candidates (touching ones are skipped by the picker)."""
    return [{"kind": "ball", "circle": (b["u"], b["v"], b["r"]), "color": b.get("color"),
             "kept": not b.get("touching", False),
             **({"why": "touching another ball"} if b.get("touching") else {})} for b in balls]


def _chose(items, i, hit):
    """Mark the candidate the picker went for, with its world position."""
    items[i]["chosen"] = True
    if hit is not None:
        items[i]["world_mm"] = hit[0]
    else:
        items[i].update(kept=False, why="ray misses table")


async def locate_ball_shape(robot, prefer_center=False):
    """Non-orange (white / any) ball, same return as locate_ball. Coarse look uses depth
    (camera must be ~40+ cm up); the close-up look (prefer_center) uses the circle search
    and the same calibrated ray-plane position as the orange ball."""
    import white_ball
    from cup_aim import _pixel_to_table

    cam = Camera.from_robot(robot, CAMERA)
    k = (await cam.get_properties()).intrinsic_parameters
    if k.focal_x_px <= 0 or k.focal_y_px <= 0:
        raise ValueError("Camera intrinsics must have positive focal lengths")
    if not prefer_center:
        bgr, depth = await _color_and_depth(cam)
        balls = white_ball.coarse_balls(bgr, depth, k, BALL_COLOR)
        items = _ball_items(balls)
        meta = dict(closeup=False, want=BALL_COLOR)
        alone = [b for b in balls if not b["touching"]]
        if balls and not alone:
            print(f"  {len(balls)} {BALL_COLOR} ball(s) found but all touching others; spread them out")
            detection_log.record("ball_depth", bgr, items, **meta)
            return None
        if not alone:
            detection_log.record("ball_depth", bgr, items, **meta)
            return None
        b = alone[0]
        cx, cy, cz = b["center_camera_mm"]
        w = (await robot.transform_pose(PoseInFrame(reference_frame=CAMERA,
                                                    pose=Pose(x=cx, y=cy, z=cz)), "world")).pose
        pos = (w.x, w.y, BALL_CENTER_Z)
        _chose(items, balls.index(b), ((pos, None)))
        detection_log.record("ball_depth", bgr, items, sphere_error_mm=b.get("sphere_error_mm"), **meta)
        print(f"  {b['color']} ball (depth fit, err {b['sphere_error_mm']:.1f} mm), "
              f"{len(balls)} {BALL_COLOR} ball(s) in view")
        if not all(math.isfinite(v) for v in pos):
            raise ValueError("Ball localization returned invalid coordinates")
        return pos
    frame = await _frame(cam)
    if frame is None:
        return None
    cam_pos = (await robot.transform_pose(
        PoseInFrame(reference_frame=CAMERA, pose=Pose(x=0, y=0, z=0)), "world")).pose
    dist = cam_pos.z - BALL_CENTER_Z
    if dist <= 50:
        raise ValueError("Camera is too low to look for balls")
    expected_r = k.focal_x_px * (BALL_DIAMETER_MM / 2) / dist
    found = white_ball.closeup_ball(frame, k, expected_r, BALL_COLOR)
    items = _ball_items(found)
    meta = dict(expected_r_px=round(expected_r, 1), closeup=True, want=BALL_COLOR)
    if not found:
        detection_log.record("ball_shape", frame, items, **meta)
        return None
    b = found[0]
    hit = await _pixel_to_table(robot, k, b["u"], b["v"], BALL_CENTER_Z)
    _chose(items, 0, hit)
    detection_log.record("ball_shape", frame, items, **meta)
    if hit is None:
        return None
    (x, y, z), _ = hit
    if not all(math.isfinite(v) for v in (x, y, z)):
        raise ValueError("Ball localization returned invalid coordinates")
    print(f"  {b['color']} ball at pixel ({b['u']:.0f},{b['v']:.0f}) r={b['r']:.0f}px "
          f"(expected {expected_r:.0f})")
    return (x, y, z)


async def locate_ball(robot, prefer_center=False):
    """Ball center (x, y, z) in mm, world frame, from the current camera view, or None.

    Finds INDIVIDUAL balls as circles of the size a ball must have at this camera height
    (ball_finder.py), so touching balls in a cluster are split instead of merged into one
    "huge, close" ball. Position = camera ray through the circle center intersected with
    the plane at ball-center height (not the blob size, which clusters break).
    Balls touching other balls are skipped: the fingers would hit the neighbours."""
    if BALL_COLOR != "orange":
        return await locate_ball_shape(robot, prefer_center)
    from ball_finder import find_balls
    from cup_aim import _pixel_to_table

    cam = Camera.from_robot(robot, CAMERA)
    k = (await cam.get_properties()).intrinsic_parameters
    if k.focal_x_px <= 0 or k.focal_y_px <= 0:
        raise ValueError("Camera intrinsics must have positive focal lengths")
    frame = await _frame(cam)
    if frame is None:
        return None
    cam_pos = (await robot.transform_pose(
        PoseInFrame(reference_frame=CAMERA, pose=Pose(x=0, y=0, z=0)), "world")).pose
    dist = cam_pos.z - BALL_CENTER_Z
    if dist <= 50:
        raise ValueError("Camera is too low to look for balls")
    expected_r = k.focal_x_px * (BALL_DIAMETER_MM / 2) / dist
    balls = find_balls(frame, expected_r)
    items = _ball_items(balls)
    meta = dict(expected_r_px=round(expected_r, 1), closeup=prefer_center)
    alone = [b for b in balls if not b["touching"]]
    if balls and not alone:
        print(f"  {len(balls)} ball(s) found but all touching others; spread them out")
        detection_log.record("ball_orange", frame, items, **meta)
        return None
    if not alone:
        detection_log.record("ball_orange", frame, items, **meta)
        return None
    if prefer_center:
        alone.sort(key=lambda b: (b["u"] - k.center_x_px) ** 2 + (b["v"] - k.center_y_px) ** 2)
    b = alone[0]
    hit = await _pixel_to_table(robot, k, b["u"], b["v"], BALL_CENTER_Z)
    _chose(items, balls.index(b), hit)
    detection_log.record("ball_orange", frame, items, **meta)
    if hit is None:
        return None
    (x, y, z), _ = hit
    if not all(math.isfinite(v) for v in (x, y, z)):
        raise ValueError("Ball localization returned invalid coordinates")
    print(f"  ball at pixel ({b['u']:.0f},{b['v']:.0f}) r={b['r']:.0f}px "
          f"(expected {expected_r:.0f}), {len(balls)} ball(s) in view")
    return (x, y, z)


async def find_ball(robot, motion, go):
    """Coarse look, then center the camera over the ball for an accurate fix."""
    await move(motion, *LOOK, go)
    coarse = await locate_ball(robot)
    if coarse is None:
        return None
    print(f"  coarse ball: {coarse[0]:.0f}, {coarse[1]:.0f}, {coarse[2]:.0f}")
    await move(motion, coarse[0] + CAM_OFFSET_XY[0], coarse[1] + CAM_OFFSET_XY[1], MEASURE_HEIGHT_Z, go)
    fine = await locate_ball(robot, prefer_center=True) if go else coarse
    if fine is None:
        return None
    jump = math.hypot(fine[0] - coarse[0], fine[1] - coarse[1])
    if jump > 40:  # coarse/fine normally agree within ~15 mm
        print(f"  ball moved {jump:.0f} mm between views (bumped or wrong object); not descending")
        return None
    x, y, z = fine[0] + CAM_CORRECTION_MM[0], fine[1] + CAM_CORRECTION_MM[1], fine[2]
    print(f"  fine ball (corrected): {x:.0f}, {y:.0f}, {z:.0f}")
    return (x, y, z)


async def wait_gripper(gripper, timeout_s=3.0):
    """Grab() returns before the fingers finish; wait until the position stops changing."""
    last, still = None, 0
    for _ in range(int(timeout_s / 0.2)):
        await asyncio.sleep(0.2)
        pos = (await gripper.do_command({"get": True})).get("pos")
        if not isinstance(pos, (int, float)) or not math.isfinite(pos) or not -20 <= pos <= 900:
            raise RuntimeError(f"Invalid gripper position {pos!r}; stopping")
        still = still + 1 if pos == last else 0
        last = pos
        if still >= 2:
            break
    return last


@contextlib.asynccontextmanager
async def slow(robot, go):
    """Run the block at PICK_SPEED_DEG_S; the normal speed is restored even on error."""
    print(f"  (slow: {PICK_SPEED_DEG_S} deg/s)")
    if not go:
        yield
        return
    arm = Arm.from_robot(robot, ARM)
    await arm.do_command({"set_speed": PICK_SPEED_DEG_S})
    try:
        yield
    finally:
        await asyncio.shield(arm.do_command({"set_speed": NORMAL_SPEED_DEG_S}))


async def pick(robot, ball, go):
    """Pick the ball at (x, y, _). Returns True if it's in the gripper after lifting."""
    motion = MotionClient.from_robot(robot, "builtin")
    gripper = Gripper.from_robot(robot, GRIPPER)
    x, y, _ = ball
    if go:
        await gripper.open()
    await move(motion, x, y, HOVER_Z, go)
    async with slow(robot, go):
        await move(motion, x, y, GRAB_Z, go)
        if not go:
            return None
        await asyncio.sleep(SETTLE_S)
        await gripper.grab()
        pos = await wait_gripper(gripper)
        print(f"  gripper closed at position {pos}")
        if pos is not None and pos <= HOLDING_MIN_POS:
            print("  gripper closed on nothing; not lifting")
            return False
        if pos is not None and pos >= GRIP_BLOCKED_POS:
            print("  fingers blocked (landed on something, not around the ball); opening and backing off")
            await gripper.open()
            await move(motion, x, y, HOVER_Z, go)
            return False
        await move(motion, x, y, HOVER_Z, go)
    pos = (await gripper.do_command({"get": True})).get("pos")
    held = pos is not None and pos > HOLDING_MIN_POS
    print("  holding ball:", held, f"(position {pos})")
    return held


def set_ball_color_from_argv():
    global BALL_COLOR
    if "--ball" in sys.argv:
        c = sys.argv[sys.argv.index("--ball") + 1]
        if c not in ("orange", "white", "any"):
            raise SystemExit("--ball must be orange, white or any")
        BALL_COLOR = c
    return BALL_COLOR


async def main():
    go = "--go" in sys.argv
    set_ball_color_from_argv()
    print("MODE:", "LIVE (arm will move)" if go else "DRY RUN (no motion)")
    robot = await connect()
    try:
        motion = MotionClient.from_robot(robot, "builtin")
        ball = await find_ball(robot, motion, go)
        if ball is None:
            print("no ball found - check ball-detector in the CONTROL tab")
            return
        held = await pick(robot, ball, go)
        print("result:", {None: "dry run", True: "PICKED", False: "MISSED"}[held])
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
