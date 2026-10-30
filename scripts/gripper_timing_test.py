"""Does the gripper open WHILE the arm is moving, or does it wait until the arm stops?

Needed for the throw: the ball only flies if the fingers open mid-swing. Empty gripper, arm at
home, a slow small joint move (elbow -20 deg at 25 deg/s, ~1 s) - nothing like the throw speed.

    python scripts/gripper_timing_test.py          # dry run: prints the plan
    python scripts/gripper_timing_test.py --go     # runs it (someone near the E-stop)
    python scripts/gripper_timing_test.py --go --partial   # the throw's open: +90 units at gripper speed 5000

It prints, in seconds from the start of the move:
  - when the gripper position first changed (the fingers started opening)
  - when the open() call returned
  - when the arm move finished
If the fingers start opening BEFORE the move finishes, mid-swing release is possible and the
throw only needs better timing. If they only start AFTER the move ends, the controller queues the
gripper behind the arm and we need another way (ask Viam staff).
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from viam.components.arm import Arm
from viam.components.gripper import Gripper
from viam.proto.component.arm import JointPositions

from connection import connect

JOINT, DELTA_DEG, SPEED = 2, -20.0, 25.0   # elbow, gentle
OPEN_AFTER_S = 0.3


async def pos(gripper):
    return (await gripper.do_command({"get": True})).get("pos")


async def main():
    go = "--go" in sys.argv
    print(f"Plan: close the empty gripper, then move the elbow {DELTA_DEG} deg at {SPEED} deg/s and "
          f"call open() {OPEN_AFTER_S} s into the move, polling the finger position the whole time.")
    if not go:
        print("DRY RUN. Add --go to run.")
        return
    robot = await connect()
    arm, gripper = Arm.from_robot(robot, "arm"), Gripper.from_robot(robot, "gripper")
    start = list((await arm.get_joint_positions()).values)
    end_pose = await arm.get_end_position()
    if end_pose.z < 350:
        raise SystemExit(f"Start from home first (arm z={end_pose.z:.0f}): run `python home.py`")
    try:
        await gripper.grab()
        await asyncio.sleep(2.0)
        closed = await pos(gripper)
        print(f"gripper closed at {closed}")

        target = list(start)
        target[JOINT] += DELTA_DEG
        await arm.do_command({"set_speed": SPEED, "set_acceleration": 200})
        t0 = time.monotonic()
        log = {"open_called": None, "open_returned": None, "move_done": None, "first_change": None}
        samples = []
        stop = False

        async def poll():
            last = closed
            while not stop:
                p = await pos(gripper)
                t = time.monotonic() - t0
                samples.append((t, p))
                if log["first_change"] is None and p is not None and closed is not None and abs(p - closed) > 8:
                    log["first_change"] = t
                last = p
                await asyncio.sleep(0.03)

        partial = "--partial" in sys.argv
        if partial:
            await gripper.do_command({"set_gripper_speed": 5000})

        async def opener():
            await asyncio.sleep(OPEN_AFTER_S)
            log["open_called"] = time.monotonic() - t0
            if partial:
                await gripper.do_command({"set": float(closed + 90)})
            else:
                await gripper.open()
            log["open_returned"] = time.monotonic() - t0

        async def mover():
            await arm.move_to_joint_positions(JointPositions(values=target))
            log["move_done"] = time.monotonic() - t0

        poller = asyncio.create_task(poll())
        await asyncio.gather(mover(), opener())
        await asyncio.sleep(1.0)
        stop = True
        await poller
        print("\n--- results (seconds from the start of the move) ---")
        for k, v in log.items():
            print(f"{k:14s} {v if v is None else round(v, 2)}")
        print(f"samples: {len(samples)}; position trace (every 5th): "
              + ", ".join(f"{t:.2f}s:{p}" for t, p in samples[::5]))
        fc, md = log["first_change"], log["move_done"]
        if fc is not None and log["open_called"] is not None:
            print(f"\nFINGER REACTION: started moving {fc - log['open_called']:.2f} s after the open command "
                  f"({'partial +90 @5000' if partial else 'full open'})")
        if fc is not None and md is not None:
            print("\nVERDICT:", "fingers started opening DURING the move -> mid-swing release is possible"
                  if fc < md - 0.1 else "fingers only moved AFTER the move ended -> the gripper is queued behind the arm")
        else:
            print("\nVERDICT: unclear; paste all of this output")
    finally:
        await arm.do_command({"set_speed": 60, "set_acceleration": 200})
        try:
            await arm.move_to_joint_positions(JointPositions(values=start))
        except Exception as e:  # noqa: BLE001
            print(f"could not return by joint move ({e}); run python home.py")
        await gripper.open()
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
