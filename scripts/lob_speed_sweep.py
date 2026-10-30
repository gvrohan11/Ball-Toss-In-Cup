"""Find the fastest lob swing the xArm controller accepts ("Linear speed exceeds limit in ServoJ mode").

Empty gripper, no release. Runs the lob swing at rising speeds (50, 60, 70 ... deg/s), goes home
after each, and stops at the first speed the controller rejects. Prints the last good speed.

    python scripts/lob_speed_sweep.py             # dry run: prints the plan
    python scripts/lob_speed_sweep.py --go        # runs it (someone near the E-stop, hand clear)
    python scripts/lob_speed_sweep.py --go --start 80 --step 5 --stop 140   # finer, if you know the range

Then use  --speed <last good>  (or 5-10 below it) with throw.py / pick_and_throw.py --lob.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import throw
from connection import connect
from home import go_home


def speeds(start, step, stop):
    out, s = [], start
    while s <= stop:
        out.append(s)
        s += step
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=float, default=50)
    ap.add_argument("--step", type=float, default=10)
    ap.add_argument("--stop", type=float, default=120)
    ap.add_argument("--accel", type=float, default=1100)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    plan = speeds(a.start, a.step, a.stop)
    print(f"lob swing, no release, speeds {plan} deg/s, accel {a.accel}")
    if not a.go:
        print("(dry run - add --go)")
        return
    robot = await connect()
    last_ok = None
    try:
        for s in plan:
            print(f"\n--- {s:.0f} deg/s ---")
            try:
                await throw.throw(robot, 500, True, 1.0, False, None, throw.LOB_SWING, s, a.accel,
                                  None, throw.LOB_START_J)
                last_ok = s
                print(f"OK at {s:.0f}")
            except Exception as e:  # noqa: BLE001
                if "Linear speed exceeds limit" not in str(e):
                    raise
                print(f"TRIPPED at {s:.0f} deg/s")
                await asyncio.sleep(1.5)
                print("HOME" if await go_home(robot) else "!! not home - use home.py")
                break
        print(f"\nRESULT: last good speed = {last_ok} deg/s"
              + ("" if last_ok is None else f"  -> use --speed {max(20, last_ok - 5):.0f}"))
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
