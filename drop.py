"""Manan: drop the held ball into a cup (Phase 1). Throw is Phase 2.

drop_in_cup(robot, cup, go=False) -> dict describing what was done.
Dry run by default; pass go=True to move.

First successful drop (sfhack11, Oct 29): cup upright at the far end, ~86-95 cm
from the base - beyond straight-down reach. The gripper was TILTED forward
(o = 0.7, 0, -0.7) to reach over it at x=870, y=20, z=300, then opened.
The ball fell ~19 cm and landed in the cup.

Reach notes:
- Straight down (o_z = -1) works out to about 65-70 cm.
- Tilting the gripper forward reaches further; the ball still falls straight down.
"""
from viam.components.gripper import Gripper
from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient

GRIPPER = "gripper"
RELEASE_ABOVE_RIM_MM = 190   # worked: z=300 with the rim at ~110
OFFSET_X_MM = 0              # tune from hit/miss data
OFFSET_Y_MM = 0
STRAIGHT_DOWN_MAX_X_MM = 650  # beyond this, tilt forward to reach

# The exact pose that put the ball in the cup on the first try.
FIRST_HIT_POSE = dict(x=870, y=20, z=300, o_x=0.7, o_y=0, o_z=-0.7, theta=180)


def release_pose(cup):
    cx, cy, cz = cup
    x, y, z = cx + OFFSET_X_MM, cy + OFFSET_Y_MM, cz + RELEASE_ABOVE_RIM_MM
    if x <= STRAIGHT_DOWN_MAX_X_MM:
        return dict(x=x, y=y, z=z, o_x=0, o_y=0, o_z=-1, theta=180)
    return dict(x=x, y=y, z=z, o_x=0.7, o_y=0, o_z=-0.7, theta=180)


async def drop_in_cup(robot, cup, go=False):
    """cup = (x, y, z) of the rim center, mm, world frame."""
    p = release_pose(cup)
    print(f"  drop: gripper to x={p['x']:.0f} y={p['y']:.0f} z={p['z']:.0f} "
          f"o=({p['o_x']},{p['o_y']},{p['o_z']}), then open")
    if go:
        motion = MotionClient.from_robot(robot, "builtin")
        gripper = Gripper.from_robot(robot, GRIPPER)
        await motion.move(component_name=GRIPPER,
                          destination=PoseInFrame(reference_frame="world", pose=Pose(**p)))
        await gripper.open()
    return {"release_pose": p, "above_rim_mm": RELEASE_ABOVE_RIM_MM,
            "offset_mm": [OFFSET_X_MM, OFFSET_Y_MM]}
