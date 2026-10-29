"""Teammate 1: cup detection.

get_cups(robot) -> list of (x, y, z) rim centers in mm, world frame.
Test: python cups.py   (read-only, moves nothing)
"""
import asyncio

from viam.proto.common import PoseInFrame
from viam.services.vision import VisionClient

CAMERA = "cam"
CUP_3D = "cup-3d"


async def get_cups(robot):
    vision = VisionClient.from_robot(robot, CUP_3D)
    objs = await vision.get_object_point_clouds(CAMERA)
    cups = []
    for o in objs:
        frame = o.geometries.reference_frame
        for g in o.geometries.geometries:
            w = await robot.transform_pose(PoseInFrame(reference_frame=frame, pose=g.center), "world")
            cups.append((w.pose.x, w.pose.y, w.pose.z))
    # TODO: center is the middle of the detected blob; adjust to the rim / opening.
    return cups


if __name__ == "__main__":
    from connection import connect

    async def _main():
        robot = await connect()
        try:
            print(await get_cups(robot))
        finally:
            await robot.close()

    asyncio.run(_main())
