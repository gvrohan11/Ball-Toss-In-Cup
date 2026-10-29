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

from cup_tracking import verify_cup, world_point, measure_held_ball

GRIPPER = "gripper"
RELEASE_ABOVE_RIM_MM = 190   # worked: z=300 with the rim at ~110
OFFSET_X_MM = 0              # tune from hit/miss data
OFFSET_Y_MM = 0
STRAIGHT_DOWN_MAX_X_MM = 650  # beyond this, tilt forward to reach

# The exact pose that put the ball in the cup on the first try.
FIRST_HIT_POSE = dict(x=870, y=20, z=300, o_x=0.7, o_y=0, o_z=-0.7, theta=180)

# Measured from the held ball during the successful rim-guided drop on sfhack11.
# Local gripper coordinates; valid for this grasp/mount, revalidate after changes.
HELD_BALL_LOCAL_MM = (20.8454, 19.7199, 50.8800)


async def drop_at_remembered_cup(robot, memory, controller):
    """One planned transfer, then fresh visual and actual payload alignment gates."""
    memory.require_fresh()
    cx, cy, cz = memory.center
    if cx > 650:
        raise RuntimeError('Rim-guided transfer currently supports nearby cups with a vertical gripper')
    local_ball = await measure_held_ball(robot) if controller.go else HELD_BALL_LOCAL_MM
    controller.held_ball_local = local_ball
    bx, by, bz = local_ball
    if controller.go:
        controller.event('payload_measured', gripper_local_mm=local_ball.tolist())
    # At down/theta=180, gripper axes map to world (+X, -Y, -Z).
    # Extra camera stand-off keeps the full rim in view despite the held ball.
    target = dict(x=cx-bx, y=cy+by, z=cz+165+bz, o_z=-1, theta=180)
    await controller.move(target, 'transfer_to_remembered_cup')
    if not controller.go:
        return {'release_pose': target, 'released': False, 'dry_run': True}
    gripper = Gripper.from_robot(robot, GRIPPER)
    if not (await gripper.is_holding_something()).is_holding_something:
        raise RuntimeError('Ball no longer held; release blocked')
    visible = await verify_cup(robot, memory, controller)
    release_ball = await measure_held_ball(robot)
    payload = await world_point(robot, release_ball, source='gripper')
    import math
    error = math.hypot(payload[0]-visible[0], payload[1]-visible[1])
    height = payload[2]-visible[2]
    controller.event('release_alignment', ball_world_mm=payload.tolist(),
                     cup_world_mm=visible.tolist(), xy_error_mm=error, height_mm=float(height))
    if error > min(8, memory.radius_mm-25) or not 100 <= height <= 180:
        raise RuntimeError('Payload is not aligned above cup; release blocked')
    if not (await gripper.is_holding_something()).is_holding_something:
        raise RuntimeError('Ball lost during verification; release blocked')
    await gripper.open()
    controller.monitor_payload = False
    controller.event('released')
    return {'release_pose': target, 'released': True, 'xy_error_mm': error,
            'cup_world_mm': visible.tolist(), 'ball_world_mm': payload.tolist()}


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
        succeeded = await motion.move(component_name=GRIPPER,
                          destination=PoseInFrame(reference_frame="world", pose=Pose(**p)))
        if not succeeded:
            raise RuntimeError('Release motion did not complete; keeping gripper closed')
        await gripper.open()
    return {"release_pose": p, "above_rim_mm": RELEASE_ABOVE_RIM_MM,
            "offset_mm": [OFFSET_X_MM, OFFSET_Y_MM]}
