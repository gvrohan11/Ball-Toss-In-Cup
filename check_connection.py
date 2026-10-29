"""Read-only connection test. Moves nothing.

    python check_connection.py
"""
import asyncio

from viam.components.arm import Arm
from viam.services.vision import VisionClient

from connection import connect


async def main():
    robot = await connect()
    try:
        print("Connected. Resources:")
        for r in robot.resource_names:
            print(f"  {r.name:15s} {r.namespace}:{r.type}:{r.subtype}")

        arm = Arm.from_robot(robot, "arm")
        p = await arm.get_end_position()
        print(f"\nArm flange pose: x={p.x:.0f} y={p.y:.0f} z={p.z:.0f} "
              f"o=({p.o_x:.2f},{p.o_y:.2f},{p.o_z:.2f}) th={p.theta:.0f}")

        for name in ("ball-detector", "cup-detector"):
            dets = await VisionClient.from_robot(robot, name).get_detections_from_camera("cam")
            print(f"{name}: {len(dets)} detection(s)",
                  [(d.class_name, d.x_min, d.y_min, d.x_max, d.y_max) for d in dets])
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
