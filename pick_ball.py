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
LOOK_SETTLE_S = 0.5         # before a camera look: close-ups taken on arrival were motion-blurred
CLOSEUP_READINGS = 3        # close-up frames per measurement; the median position is used
CLOSEUP_SPREAD_MM = 6       # readings farther apart than this: the arm/ball was still moving


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
# Cups found by the survey (main.survey_cups sets this). A ball within AVOID_RADIUS_MM of a
# cup is in it or right next to it: never picked (the fingers would hit or lift the cup).
AVOID_XY = []
AVOID_RADIUS_MM = 90        # cup rim radius 47 + ball radius 20 + margin


def near_cup(xy):
    return any(math.hypot(xy[0] - cx, xy[1] - cy) < AVOID_RADIUS_MM for cx, cy in AVOID_XY)


# Ball search: when the look pose shows no ball, more of the table is searched, but only the
# part between the robot and the cups: up to CUP_MARGIN_MM short of the nearest cup.
CUP_MARGIN_MM = 150
DEFAULT_SEARCH_MAX_R = 700      # no cups surveyed yet: search out to 70 cm from the base
# Extra look poses (gripper x, y at the look height, camera straight down). The camera sits
# (83, 14) mm ahead of the gripper and sees ~41 cm (x) by ~73 cm (y) from there, so these
# cover the sides, the far strip and the strip near the base around the normal LOOK view.
# First the TOP view: gripper at z 800 straight down (the cup survey's view 7) sees ~1.2 x 0.7 m
# of table in one frame (a ball is ~21 px across there). Then the closer side views.
TOP_VIEW = (480, 0, 800)
SEARCH_VIEWS = [TOP_VIEW] + [(x, y, LOOK[2]) for x, y in
                             [(350, 300), (350, -300), (520, 0), (520, 300), (520, -300), (250, 0)]]


def search_max_r():
    """Farthest a ball may be from the base (mm): CUP_MARGIN_MM short of the nearest cup."""
    if AVOID_XY:
        return min(math.hypot(x, y) for x, y in AVOID_XY) - CUP_MARGIN_MM
    return DEFAULT_SEARCH_MAX_R


def skip_reason(xy):
    """Why a ball at world xy must not be picked, or None."""
    if near_cup(xy):
        return "in or next to a cup"
    if math.hypot(xy[0], xy[1]) > search_max_r():
        return "past the search line (too near the cups)"
    return None


def _skip(items, i, why):
    items[i].update(kept=False, why=why)


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


def _ball_items(balls, frame=None):
    """Detection-log entries for ball candidates (touching ones are skipped by the picker):
    how each was found (edge_fit / hough / yolo_mask), YOLO's confidence, the shadow/blur
    bulge and a sharpness score, so a bad pick can be traced to the frame."""
    from ball_finder import sharpness
    items = []
    for b in balls:
        it = {"kind": "ball", "circle": (b["u"], b["v"], b["r"]), "color": b.get("color"),
              "kept": not b.get("touching", False)}
        if b.get("touching"):
            it["why"] = "touching another ball"
        for k in ("method", "yolo", "bulge_px", "fill"):
            if b.get(k) is not None:
                it[k] = b[k]
        if frame is not None:
            it["sharpness"] = round(sharpness(frame, b["u"], b["v"], b["r"]), 1)
        items.append(it)
    return items


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
        b = pos = None
        for cand in alone:
            cx, cy, cz = cand["center_camera_mm"]
            w = (await robot.transform_pose(PoseInFrame(reference_frame=CAMERA,
                                                        pose=Pose(x=cx, y=cy, z=cz)), "world")).pose
            why = skip_reason((w.x, w.y))
            if why:
                _skip(items, balls.index(cand), why)
                continue
            b, pos = cand, (w.x, w.y, BALL_CENTER_Z)
            break
        if b is None:
            print("  the only balls in view are in/next to cups or past the search line")
            detection_log.record("ball_depth", bgr, items, **meta)
            return None
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
    b = hit = None
    for i, cand in enumerate(found):
        h = await _pixel_to_table(robot, k, cand["u"], cand["v"], BALL_CENTER_Z)
        if h is not None and skip_reason(h[0]):
            _skip(items, i, skip_reason(h[0]))
            continue
        b, hit = cand, h
        break
    if b is None:
        detection_log.record("ball_shape", frame, items, **meta)
        return None
    _chose(items, found.index(b), hit)
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
    items = _ball_items(balls, frame)
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
    b = hit = None
    for cand in alone:
        h = await _pixel_to_table(robot, k, cand["u"], cand["v"], BALL_CENTER_Z)
        if h is not None and skip_reason(h[0]):
            _skip(items, balls.index(cand), skip_reason(h[0]))
            continue
        b, hit = cand, h
        break
    if b is None:
        print("  the only balls in view are in/next to cups or past the search line")
        detection_log.record("ball_orange", frame, items, **meta)
        return None
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


async def closeup(robot, go):
    """Ball position from CLOSEUP_READINGS close-up frames (camera centered over it): the
    median, after LOOK_SETTLE_S. None if the ball is seen in fewer than 2 frames."""
    if go:
        await asyncio.sleep(LOOK_SETTLE_S)
    seen = []
    for _ in range(CLOSEUP_READINGS if go else 1):
        p = await locate_ball(robot, prefer_center=True)
        if p is not None:
            seen.append(p)
    if len(seen) < min(2, CLOSEUP_READINGS if go else 1):
        print(f"  close-up: ball seen in {len(seen)}/{CLOSEUP_READINGS} frames; not trusting it")
        return None
    med = tuple(sorted(v)[len(v) // 2] for v in zip(*seen))
    spread = max(math.hypot(p[0] - med[0], p[1] - med[1]) for p in seen)
    print(f"  close-up: {len(seen)} readings, spread {spread:.1f} mm")
    if spread > CLOSEUP_SPREAD_MM:
        print(f"  close-up readings disagree by {spread:.0f} mm (still moving?); using the median")
    detection_log.event("closeup", readings=seen, median_mm=med, spread_mm=round(spread, 1))
    return med


def search_views():
    """SEARCH_VIEWS whose camera is not past the search line (the top view always: it sees
    the whole table; balls past the line are filtered out anyway)."""
    limit = search_max_r()
    return [v for v in SEARCH_VIEWS if v == TOP_VIEW or
            math.hypot(v[0] - CAM_OFFSET_XY[0], v[1] - CAM_OFFSET_XY[1]) <= limit]


def cut_off(b, w, h, margin=0.9):
    """The ball's circle runs past the frame edge (its fitted center is unreliable)."""
    r = margin * b["r"]
    return b["u"] - r < 0 or b["v"] - r < 0 or b["u"] + r > w or b["v"] + r > h


async def balls_in_view(robot):
    """EVERY orange ball in the current view (not just the one the picker would take):
    [{"xy": (x, y) world mm, "why": reason it would not be picked or None, "touching": bool}].
    Balls cut off at the frame edge are left out (logged as rejected). Read-only."""
    from ball_finder import find_balls
    from cup_aim import _pixel_to_table

    cam = Camera.from_robot(robot, CAMERA)
    k = (await cam.get_properties()).intrinsic_parameters
    frame = await _frame(cam)
    if frame is None:
        return []
    cam_pos = (await robot.transform_pose(
        PoseInFrame(reference_frame=CAMERA, pose=Pose(x=0, y=0, z=0)), "world")).pose
    expected_r = k.focal_x_px * (BALL_DIAMETER_MM / 2) / (cam_pos.z - BALL_CENTER_Z)
    balls = find_balls(frame, expected_r)
    items = _ball_items(balls, frame)
    out = []
    h, w = frame.shape[:2]
    for i, b in enumerate(balls):
        hit = await _pixel_to_table(robot, k, b["u"], b["v"], BALL_CENTER_Z)
        if hit is None:
            continue
        xy = hit[0][:2]
        if cut_off(b, w, h):          # its center is a guess; another view (the top) sees it whole
            items[i].update(world_mm=hit[0], kept=False, why="cut off at the frame edge")
            continue
        why = "touching another ball" if b["touching"] else skip_reason(xy)
        items[i].update(world_mm=hit[0], kept=why is None, **({"why": why} if why else {}))
        out.append({"xy": xy, "why": why, "touching": bool(b["touching"])})
    detection_log.record("ball_scan", frame, items, expected_r_px=round(expected_r, 1), closeup=False)
    return out


async def search_table(robot, motion, go):
    """No ball at the look pose: look from SEARCH_VIEWS over the table between the robot and
    the cups (views whose camera is past the search line are skipped, unreachable ones too).
    Returns the first ball's coarse position, or None when that whole area is empty."""
    limit = search_max_r()
    views = search_views()
    print(f"  no ball at the look pose: searching the table up to {limit / 10:.0f} cm from the base "
          f"(before the cups), {len(views)} more view(s), from the top first")
    for i, (gx, gy, gz) in enumerate(views, 1):
        detection_log.context(f"ball_search_{i}")
        try:
            await move(motion, gx, gy, gz, go)
        except Exception as e:
            print(f"    view {i}/{len(views)} ({gx}, {gy}, z {gz}): can't reach, skipped ({str(e)[:50]})")
            continue
        if go:
            await asyncio.sleep(LOOK_SETTLE_S)
        ball = await locate_ball(robot)
        print(f"    view {i}/{len(views)} ({gx}, {gy}, z {gz}): "
              + (f"ball at ({ball[0]:.0f}, {ball[1]:.0f})" if ball else "no ball"))
        if ball is not None:
            return ball
    print("  table searched: no ball left before the cups")
    return None


FIND_TRIES = 3              # a ball seen but lost in the close-up: look again this many times


async def find_ball(robot, motion, go):
    """Coarse look, then center the camera over the ball for an accurate fix. If the look
    pose shows no ball, the rest of the table before the cups is searched (search_table).
    Returns None ("no ball found") only when the look pose AND the search find nothing, or
    when a ball that was seen can't be measured in FIND_TRIES close-ups."""
    for attempt in range(1, FIND_TRIES + 1):
        await move(motion, *LOOK, go)
        if go:
            await asyncio.sleep(LOOK_SETTLE_S)
        coarse = await locate_ball(robot)
        if coarse is None:
            coarse = await search_table(robot, motion, go)
        if coarse is None:
            return None                               # look pose + whole search: no ball
        print(f"  coarse ball: {coarse[0]:.0f}, {coarse[1]:.0f}, {coarse[2]:.0f}")
        await move(motion, coarse[0] + CAM_OFFSET_XY[0], coarse[1] + CAM_OFFSET_XY[1], MEASURE_HEIGHT_Z, go)
        fine = await closeup(robot, go) if go else coarse
        if fine is None:
            print(f"  lost the ball in the close-up; looking again ({attempt}/{FIND_TRIES})")
            continue
        jump = math.hypot(fine[0] - coarse[0], fine[1] - coarse[1])
        if jump > 40:  # coarse/fine normally agree within ~15 mm
            print(f"  ball moved {jump:.0f} mm between views (bumped or wrong object); "
                  f"looking again ({attempt}/{FIND_TRIES})")
            continue
        x, y, z = fine[0] + CAM_CORRECTION_MM[0], fine[1] + CAM_CORRECTION_MM[1], fine[2]
        print(f"  fine ball (corrected): {x:.0f}, {y:.0f}, {z:.0f}")
        return (x, y, z)
    print(f"  a ball was seen but couldn't be measured in {FIND_TRIES} close-ups; giving up on it")
    return None


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


PICK_RETRIES = 3            # after a missed grab: go up, look again, retry this many times
LOOK_AGAIN_MAX_MM = 100     # a ball farther than this from the missed spot is not "the" ball


async def pick(robot, ball, go):
    """Pick the ball at (x, y, _). Returns True if it's in the gripper after lifting.
    A missed grab (closed on nothing, fingers landed on something, or the ball slipped out
    on the lift): open, go up, center the camera over the spot, and if the ball is still
    there grab again, up to PICK_RETRIES times. Ends open and up if it gives up."""
    held = await grab_at(robot, ball, go)
    for retry in range(1, PICK_RETRIES + 1):
        if not go or held:
            return held
        print(f"  missed the ball; going up to look again (retry {retry}/{PICK_RETRIES})")
        detection_log.context(f"pick_retry_{retry}")
        ball = await look_again(robot, ball, go)
        if ball is None:
            return False
        held = await grab_at(robot, ball, go)
    if go and not held:
        print(f"  still no ball after {PICK_RETRIES} retries; giving up on it")
        await _open_and_lift(robot, ball, go)
    return held


async def _open_and_lift(robot, ball, go):
    await Gripper.from_robot(robot, GRIPPER).open()
    await move(MotionClient.from_robot(robot, "builtin"), ball[0], ball[1], HOVER_Z, go)


async def look_again(robot, ball, go):
    """After a miss: open, lift straight up (fingers clear of the ball), put the camera
    over the spot and re-measure. Returns the corrected ball position, or None if no ball
    is there any more (rolled away, or it was never a ball)."""
    motion = MotionClient.from_robot(robot, "builtin")
    await _open_and_lift(robot, ball, go)
    await move(motion, ball[0] - CAM_CORRECTION_MM[0] + CAM_OFFSET_XY[0],
               ball[1] - CAM_CORRECTION_MM[1] + CAM_OFFSET_XY[1], MEASURE_HEIGHT_Z, go)
    seen = await closeup(robot, go)
    if seen is None:
        print("  no ball under the camera any more")
        return None
    x, y, z = seen[0] + CAM_CORRECTION_MM[0], seen[1] + CAM_CORRECTION_MM[1], seen[2]
    moved = math.hypot(x - ball[0], y - ball[1])
    if moved > LOOK_AGAIN_MAX_MM:
        print(f"  nearest ball is {moved:.0f} mm from the missed spot; not the same ball")
        return None
    print(f"  ball still there, {moved:.0f} mm from the last try: ({x:.0f}, {y:.0f})")
    return (x, y, z)


async def grab_at(robot, ball, go):
    """One grab at (x, y, _): hover, slowly down, close, slowly up. True if holding.
    On a miss the gripper may be left closed and low; pick() / look_again() recover."""
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
        outcome = ("closed on nothing" if pos is not None and pos <= HOLDING_MIN_POS else
                   "blocked" if pos is not None and pos >= GRIP_BLOCKED_POS else "closed on something")
        detection_log.event("grab", target_mm=(x, y, GRAB_Z), finger_pos=pos, outcome=outcome,
                            cam_correction_mm=CAM_CORRECTION_MM)
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
