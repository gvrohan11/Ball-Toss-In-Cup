"""Manan: find the cup with the angled camera view and aim the drop at its center.

This is the integration-side cup locator that produced 2/2 drops on Oct 29.
Rohan's cups.py (get_cups) can replace it once it's reliable; keep the same output:
(x, y, z) of the cup RIM CENTER in mm, world frame.

    python cup_aim.py        # read-only: uses the current camera view, moves nothing

How it works:
 1. Arm at CUP_LOOK (gripper tilted 45 deg forward, wrist 180) so the camera sees the red cup sides.
 2. cup-detector gives a box on the red body. The bottom-center of that box is the point of the
    cup base nearest the camera; project that pixel onto the table plane.
 3. Push it back by the base radius (along the viewing direction) to get the cup center.
 4. Apply CUP_CORRECTION_MM, measured from the two successful drops.
"""
import asyncio
import math

from viam.components.camera import Camera
from viam.proto.common import Pose, PoseInFrame
from viam.services.vision import VisionClient

CAMERA = "cam"
CUP_DETECTOR = "cup-detector"

# Gripper pose for looking at the far end of the table.
# Scan these tilts in order. 30 deg sees ~50-90 cm out, 45 deg sees ~87-130 cm out.
# (A fixed 45 deg view missed a cup placed at ~78 cm.)
CUP_LOOK_30 = dict(x=550, y=0, z=650, o_x=0.5, o_y=0, o_z=-0.866, theta=180)
CUP_LOOK_45 = dict(x=550, y=0, z=650, o_x=0.7, o_y=0, o_z=-0.7, theta=180)
CUP_LOOKS = (CUP_LOOK_30, CUP_LOOK_45)
CUP_LOOK = CUP_LOOK_45       # kept for older imports
CUP_LOOK_FAR = CUP_LOOK_30

TABLE_Z = -10              # real tabletop height (fingers hit the table lower than the config's -23)
CUP_HEIGHT_MM = 120        # rim height above the table
CUP_BASE_RADIUS_MM = 30

# Measured Oct 29: this method said the cup center was at (926, -5); the drop that scored
# twice was released at (870, 20). Correction = truth - vision. Assumes the cup had not
# moved between the drops and the measurement - re-check with a moved cup.
CUP_CORRECTION_MM = (-56, 17)  # 3rd drop at y=+25 went in but slightly off-center in y; trimmed
# Same correction used from the 30 deg view (4th drop, cup at ~78 cm): it went in.
# Known gap: with the ball held, the ball can hide the cup's red side and cup-detector
# returns nothing. Next: measure the white rim (OpenCV) or use YOLO "cup" (Rohan).


def best_cup(dets):
    """Biggest red box that is taller than it is thin (ignores flat red lids and slivers)."""
    best = None
    for d in dets:
        w, h = d.x_max - d.x_min, d.y_max - d.y_min
        if w < 25 or h < 25:
            continue
        if best is None or w * h > (best.x_max - best.x_min) * (best.y_max - best.y_min):
            best = d
    return best


async def _pixel_to_table(robot, k, u, v, z_world):
    """Intersect the camera ray through pixel (u, v) with the horizontal plane z = z_world."""
    near = await robot.transform_pose(
        PoseInFrame(reference_frame=CAMERA, pose=Pose(x=0, y=0, z=0)), "world")
    far = await robot.transform_pose(
        PoseInFrame(reference_frame=CAMERA, pose=Pose(
            x=(u - k.center_x_px) / k.focal_x_px * 1000,
            y=(v - k.center_y_px) / k.focal_y_px * 1000,
            z=1000)), "world")
    a, b = near.pose, far.pose
    dz = b.z - a.z
    if abs(dz) < 1e-6:
        return None
    s = (z_world - a.z) / dz
    if not math.isfinite(s) or s <= 0:
        return None
    return (a.x + s * (b.x - a.x), a.y + s * (b.y - a.y), z_world), (a.x, a.y)


async def find_cup(robot):
    """Return the cup rim center (x, y, z) in mm, world frame, from the CURRENT view, or None."""
    d = best_cup(await VisionClient.from_robot(robot, CUP_DETECTOR).get_detections_from_camera(CAMERA))
    if d is None:
        return None
    k = (await Camera.from_robot(robot, CAMERA).get_properties()).intrinsic_parameters
    hit = await _pixel_to_table(robot, k, (d.x_min + d.x_max) / 2, d.y_max, TABLE_Z)
    if hit is None:
        return None
    (bx, by, _), (camx, camy) = hit
    # push from the nearest base point to the base center, away from the camera
    dx, dy = bx - camx, by - camy
    n = math.hypot(dx, dy) or 1.0
    cx = bx + CUP_BASE_RADIUS_MM * dx / n + CUP_CORRECTION_MM[0]
    cy = by + CUP_BASE_RADIUS_MM * dy / n + CUP_CORRECTION_MM[1]
    return (cx, cy, TABLE_Z + CUP_HEIGHT_MM)


if __name__ == "__main__":
    from connection import connect

    async def _main():
        robot = await connect()
        try:
            print("cup rim center:", await find_cup(robot))
        finally:
            await robot.close()

    asyncio.run(_main())
