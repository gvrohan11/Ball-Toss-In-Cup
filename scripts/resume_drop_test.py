"""Explicit live recovery: verify cached cup and held ball, finish drop, go home."""
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from connection import connect
from cup_tracking import CupMemory, MEMORY_PATH
from drop import drop_at_remembered_cup
from motion_control import MotionController


async def main():
    memory = CupMemory(**json.loads(MEMORY_PATH.read_text()))
    memory.require_fresh()
    robot = await connect()
    try:
        controller = MotionController(robot, True)
        controller.cup_memory = memory
        controller.monitor_payload = True
        result = await drop_at_remembered_cup(robot, memory, controller)
        (ROOT / 'observations/automated-release-result.json').write_text(json.dumps(result, indent=2)+'\n')
        await controller.move(dict(x=350, y=0, z=445, o_z=-1, theta=180), 'home')
    finally:
        await robot.close()


if __name__ == '__main__':
    asyncio.run(main())
