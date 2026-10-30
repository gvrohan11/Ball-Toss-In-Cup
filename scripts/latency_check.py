"""Read-only: how long does one command take to reach the arm and come back? Moves nothing.

    python scripts/latency_check.py          # 60 joint reads + 20 gripper reads

Run it on your laptop and then on the robot computer. Release timing jitter is roughly the
spread here (p95 - median). Oct 29 from a laptop on venue Wi-Fi: 15-400 ms.
"""
import asyncio
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from viam.components.arm import Arm  # noqa: E402
from viam.components.gripper import Gripper  # noqa: E402

from connection import connect  # noqa: E402


async def timed(fn, n):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        await fn()
        out.append((time.perf_counter() - t) * 1000)
    return out


def show(name, ms):
    ms = sorted(ms)
    p95 = ms[int(0.95 * (len(ms) - 1))]
    print(f"{name:22s} median {statistics.median(ms):6.1f} ms   p95 {p95:6.1f} ms   max {ms[-1]:6.1f} ms   "
          f"jitter (p95 - median) {p95 - statistics.median(ms):6.1f} ms")


async def main():
    robot = await connect()
    try:
        arm, gripper = Arm.from_robot(robot, "arm"), Gripper.from_robot(robot, "gripper")
        await arm.get_joint_positions()            # warm up
        show("arm.get_joint_positions", await timed(arm.get_joint_positions, 60))
        show("gripper get", await timed(lambda: gripper.do_command({"get": True}), 20))
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
