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
 5. Hover above the ball, go down to GRAB_Z, close, WAIT for the fingers to stop, lift.
"""
import asyncio
import sys

from viam.components.camera import Camera
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient
from viam.services.vision import VisionClient

from connection import connect

CAMERA = "cam"
BALL_DETECTOR = "ball-detector"
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
CAM_CORRECTION_MM = (_SIGN * _CAM_CORRECTION_THETA0[0], _SIGN * _CAM_CORRECTION_THETA0[1])  # TODO: re-verify at theta=180
HOVER_Z = 150
GRAB_Z = 70                 # fingers swing DOWN while closing; lower than ~60 drags them on the table
HOLDING_MIN_POS = 50        # gripper position below this after closing = closed on nothing


def pose(x, y, z):
    return PoseInFrame(reference_frame="world",
                       pose=Pose(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=THETA))


async def move(motion, x, y, z, go):
    print(f"  -> gripper to x={x:.0f} y={y:.0f} z={z:.0f}")
    if go:
        await motion.move(component_name=GRIPPER, destination=pose(x, y, z))


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


async def locate_ball(robot):
    """Return the ball center (x, y, z) in mm, world frame, from the current camera view."""
    vision = VisionClient.from_robot(robot, BALL_DETECTOR)
    cam = Camera.from_robot(robot, CAMERA)
    d = best_ball(await vision.get_detections_from_camera(CAMERA))
    if d is None:
        return None
    k = (await cam.get_properties()).intrinsic_parameters
    u, v = (d.x_min + d.x_max) / 2, (d.y_min + d.y_max) / 2
    size_px = max(d.x_max - d.x_min, d.y_max - d.y_min)
    depth = k.focal_x_px * BALL_DIAMETER_MM / size_px
    p_cam = Pose(x=(u - k.center_x_px) / k.focal_x_px * depth,
                 y=(v - k.center_y_px) / k.focal_y_px * depth,
                 z=depth)
    w = await robot.transform_pose(PoseInFrame(reference_frame=CAMERA, pose=p_cam), "world")
    return (w.pose.x, w.pose.y, w.pose.z)


async def find_ball(robot, motion, go):
    """Coarse look, then center the camera over the ball for an accurate fix."""
    await move(motion, *LOOK, go)
    coarse = await locate_ball(robot)
    if coarse is None:
        return None
    print(f"  coarse ball: {coarse[0]:.0f}, {coarse[1]:.0f}, {coarse[2]:.0f}")
    await move(motion, coarse[0] + CAM_OFFSET_XY[0], coarse[1] + CAM_OFFSET_XY[1], MEASURE_HEIGHT_Z, go)
    fine = await locate_ball(robot) if go else coarse
    if fine is None:
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
        still = still + 1 if pos == last else 0
        last = pos
        if still >= 2:
            break
    return last


async def pick(robot, ball, go):
    """Pick the ball at (x, y, _). Returns True if it's in the gripper after lifting."""
    motion = MotionClient.from_robot(robot, "builtin")
    gripper = Gripper.from_robot(robot, GRIPPER)
    x, y, _ = ball
    if go:
        await gripper.open()
    await move(motion, x, y, HOVER_Z, go)
    await move(motion, x, y, GRAB_Z, go)
    if not go:
        return None
    await gripper.grab()
    pos = await wait_gripper(gripper)
    print(f"  gripper closed at position {pos}")
    await move(motion, x, y, HOVER_Z, go)
    pos = (await gripper.do_command({"get": True})).get("pos")
    held = pos is not None and pos > HOLDING_MIN_POS
    print("  holding ball:", held, f"(position {pos})")
    return held


async def main():
    go = "--go" in sys.argv
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
