"""Manan: drop the held ball into a cup (Phase 1). Throw is Phase 2.

drop_in_cup(robot, cup, go=False) -> dict describing what was done.
Dry run by default; pass go=True to move.
"""
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient

GRIPPER = "gripper"
RELEASE_HEIGHT_MM = 150     # TODO: tune - how far above the cup rim to let go
OFFSET_X_MM = 0             # TODO: tune from hit/miss data
OFFSET_Y_MM = 0


async def drop_in_cup(robot, cup, go=False):
    cx, cy, cz = cup
    x, y, z = cx + OFFSET_X_MM, cy + OFFSET_Y_MM, cz + RELEASE_HEIGHT_MM
    release = dict(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=0)
    print(f"  drop: gripper to x={x:.0f} y={y:.0f} z={z:.0f}, then open")
    if go:
        motion = MotionClient.from_robot(robot, "builtin")
        gripper = Gripper.from_robot(robot, GRIPPER)
        await motion.move(component_name=GRIPPER,
                          destination=PoseInFrame(reference_frame="world", pose=Pose(**release)))
        await gripper.open()
    return {"release_pose": release, "height_mm": RELEASE_HEIGHT_MM,
            "offset_mm": [OFFSET_X_MM, OFFSET_Y_MM]}
