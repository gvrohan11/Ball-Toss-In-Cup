"""Ball teammate: ball detection.

find_ball(robot) -> (x, y, z) in mm, world frame, or None.
Test: python balls.py   (read-only, moves nothing)
"""
import asyncio
import os

from viam.proto.common import PoseInFrame
from viam.services.vision import VisionClient

CAMERA = "cam"
BALL_3D = os.getenv("BALL_VISION_SERVICE", "ball-3d")


async def find_ball(robot):
    vision = VisionClient.from_robot(robot, BALL_3D)
    objs = await vision.get_object_point_clouds(CAMERA)
    found = []
    for o in objs:
        frame = o.geometries.reference_frame
        for g in o.geometries.geometries:
            w = await robot.transform_pose(PoseInFrame(reference_frame=frame, pose=g.center), "world")
            found.append((w.pose.x, w.pose.y, w.pose.z))
    if not found:
        return None
    # TODO: filter to the reachable table area, handle several balls, white balls.
    found.sort(key=lambda p: p[2])  # ball sits on the table -> lowest candidate
    return found[0]


if __name__ == "__main__":
    from connection import connect

    async def _main():
        robot = await connect()
        try:
            print(await find_ball(robot))
        finally:
            await robot.close()

    asyncio.run(_main())
