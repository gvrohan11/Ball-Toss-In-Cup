"""Manan: the full loop. find ball -> pick -> find cup -> drop -> home -> log.

python main.py          # dry run: prints the plan, moves nothing
python main.py --go     # runs on the arm (someone on the E-stop!)
python main.py --go --autotune   # also shift the aim by the average of recent misses in Atlas

Uses the sequences that worked on sfhack11 (Oct 29): pick_ball.py (2/2 picks),
cup_aim.py + drop.py (2/2 drops). Swap in balls.find_ball / cups.get_cups when
the ball and cup teammates' versions are reliable.
"""
import asyncio
import sys
import time

from viam.proto.common import Pose, PoseInFrame
from viam.services.motion import MotionClient

from connection import connect
from cup_aim import CUP_SCAN, find_cup
from drop import drop_in_cup
from logger import log_attempt, suggest_offset
from pick_ball import GRIPPER, LOOK, THETA, find_ball, pick
from telemetry import Telemetry

TEL = None  # Telemetry for the current run (set in main)


async def move_to(motion, p, go, label="move"):
    print(f"  -> gripper to {p}")
    if go:
        t = time.monotonic()
        ok = None
        try:
            ok = await motion.move(component_name=GRIPPER,
                                   destination=PoseInFrame(reference_frame="world", pose=Pose(**p)))
        finally:
            if TEL:
                TEL.move(label, p, ok, time.monotonic() - t)
        if ok is False:
            raise RuntimeError("Motion did not complete")


def ask_result():
    """Ask the operator how it went. Returns (hit, miss_mm or None).
    Miss direction is from the robot's point of view: +x = past the cup (away from the
    robot), +y = to the robot's left. Used by logger.suggest_offset() to auto-tune."""
    hit = input("Did it go in? [y/n] ").strip().lower().startswith("y")
    if hit:
        return True, None
    raw = input("Where did it land vs the cup? cm past(+)/short(-), cm left(+)/right(-) "
                "as seen from the robot, e.g. '2,-1' (Enter to skip): ").strip()
    if not raw:
        return False, None
    try:
        dx, dy = (float(v) * 10 for v in raw.split(","))
        return False, {"x": dx, "y": dy}
    except ValueError:
        print("  couldn't read that, logging the miss without a direction")
        return False, None


async def put_ball_back(robot, motion, ball, go):
    """Safe fail: carry the held ball back to where it was picked, set it down, open, go up."""
    from viam.components.gripper import Gripper
    from pick_ball import GRAB_Z, HOVER_Z
    print("  putting the ball back")
    gripper = Gripper.from_robot(robot, GRIPPER)
    try:
        if ball is not None:
            await move_to(motion, dict(x=ball[0], y=ball[1], z=HOVER_Z, o_x=0, o_y=0, o_z=-1, theta=THETA), go)
            await move_to(motion, dict(x=ball[0], y=ball[1], z=GRAB_Z + 10, o_x=0, o_y=0, o_z=-1, theta=THETA), go)
    except Exception as e:
        print(f"  couldn't reach the pick spot ({e}); releasing where I am")
    if go:
        await gripper.open()
    if ball is not None:
        await move_to(motion, dict(x=ball[0], y=ball[1], z=HOVER_Z, o_x=0, o_y=0, o_z=-1, theta=THETA), go)


async def scan_for_cup(motion, robot, go):
    """Try every view in CUP_SCAN; skip views the planner can't reach."""
    for i, look in enumerate(CUP_SCAN, 1):
        print(f"  view {i}/{len(CUP_SCAN)}: y={look['y']:+.0f}, "
              f"{'top-down' if look['o_z'] == -1 else 'tilted'}")
        try:
            await move_to(motion, look, go, label=f"cup_view_{i}")
        except Exception as e:
            print(f"    can't reach this view ({str(e)[:80]}); skipping")
            continue
        cup = await find_cup(robot, topdown=(look["o_z"] == -1))
        if TEL:
            await TEL.snap(f"cup_view_{i}", found=cup is not None, view=look)
        if cup is not None:
            if TEL:
                await TEL.thumbnail("cup_view")
            return cup
    return None


# Where to set down a ball found in the gripper at startup: clear table near the robot.
SET_DOWN_XY = (420, 0)


async def release_if_holding(robot, motion, go):
    """Start-of-run check: if the gripper is holding something (left over from an
    interrupted run), set it down on the table, open, and lift. Always ends open."""
    from viam.components.gripper import Gripper
    from pick_ball import GRAB_Z, HOLDING_MIN_POS, HOVER_Z
    gripper = Gripper.from_robot(robot, GRIPPER)
    try:
        pos = (await gripper.do_command({"get": True})).get("pos")
    except Exception:
        pos = None
    holding = pos is not None and HOLDING_MIN_POS < pos < 800
    print(f"  gripper position {pos}: {'HOLDING something - setting it down' if holding else 'empty'}")
    if holding:
        x, y = SET_DOWN_XY
        await move_to(motion, dict(x=x, y=y, z=HOVER_Z, o_x=0, o_y=0, o_z=-1, theta=THETA), go)
        await move_to(motion, dict(x=x, y=y, z=GRAB_Z + 10, o_x=0, o_y=0, o_z=-1, theta=THETA), go)
    if go:
        await gripper.open()
    if holding:
        await move_to(motion, dict(x=x, y=y, z=HOVER_Z, o_x=0, o_y=0, o_z=-1, theta=THETA), go)


async def _snap(stage, **kw):
    if TEL:
        await TEL.snap(stage, **kw)


def _phase(name):
    if TEL:
        TEL.phase(name)


async def run_once(robot, motion, go, home, t0):
    print("0) start from the saved home pose")
    _phase("start")
    await move_to(motion, home, go, label="home")
    await release_if_holding(robot, motion, go)
    await move_to(motion, home, go, label="home")
    await _snap("home")

    print("1) find ball")
    _phase("find_ball")
    ball = await find_ball(robot, motion, go)
    await _snap("ball_measured", ball_mm=list(ball) if ball else None)
    if TEL and ball:
        await TEL.thumbnail("ball_view")
    if ball is None:
        print("no ball found")
        return {"result": {"hit": False, "failure": "no_ball_found"}}, None

    print("2) pick")
    _phase("pick")
    held = await pick(robot, ball, go)
    await _snap("after_pick", held=held)
    if go and not held:
        print("pick missed")
        return {"ball": {"pos_mm": ball}, "result": {"hit": False, "failure": "grasp_failed"}}, None

    # Ball is in the gripper from here on: any failure puts it back first.
    print("3) find cup (wide scan)")
    _phase("cup_scan")
    cup = await scan_for_cup(motion, robot, go)
    if cup is None:
        print("no cup found in any view")
        await put_ball_back(robot, motion, ball, go)
        return {"ball": {"pos_mm": ball}, "result": {"hit": False, "failure": "no_cup_found"}}, None
    print(f"  cup rim center: {cup[0]:.0f}, {cup[1]:.0f}, {cup[2]:.0f}")

    extra = (0, 0)
    if "--autotune" in sys.argv:
        extra = suggest_offset()
        print(f"  autotune from Atlas misses: {extra[0]:+.0f}, {extra[1]:+.0f} mm")
    print("4) drop (centered on the cup)")
    _phase("drop")
    try:
        result = await drop_in_cup(robot, cup, go, extra)
        await _snap("released", cup_mm=list(cup))
    except Exception as e:
        print(f"  can't reach the cup ({str(e)[:80]})")
        await put_ball_back(robot, motion, ball, go)
        return {"ball": {"pos_mm": ball}, "cup": {"pos_mm": cup},
                "result": {"hit": False, "failure": "cup_out_of_reach"}}, None
    return {"phase": "drop", "ball": {"pos_mm": ball}, "cup": {"pos_mm": cup},
            "release": result}, "dropped"


def _run_config():
    import cup_aim
    import drop
    import pick_ball
    return {"theta": THETA, "home": list(LOOK),
            "ball_cam_correction_mm": list(pick_ball.CAM_CORRECTION_MM),
            "grab_z": pick_ball.GRAB_Z, "hover_z": pick_ball.HOVER_Z,
            "rim_correction_mm": list(cup_aim.RIM_CORRECTION_MM),
            "topdown_correction_mm": list(cup_aim.TOPDOWN_CORRECTION_MM),
            "release_above_rim_mm": drop.RELEASE_ABOVE_RIM_MM,
            "ball_color": pick_ball.BALL_COLOR,
            "autotune": "--autotune" in sys.argv}


async def main():
    global TEL
    go = "--go" in sys.argv
    import pick_ball
    print("BALL:", pick_ball.set_ball_color_from_argv())
    print("MODE:", "LIVE (arm will move)" if go else "DRY RUN (no motion, nothing logged)")
    t0 = time.monotonic()
    home = dict(x=LOOK[0], y=LOOK[1], z=LOOK[2], o_x=0, o_y=0, o_z=-1, theta=THETA)
    robot = await connect()
    TEL = Telemetry(robot, go)
    TEL.config.update(_run_config())
    try:
        motion = MotionClient.from_robot(robot, "builtin")
        try:
            doc, state = await run_once(robot, motion, go, home, t0)
        except BaseException as e:  # error or Ctrl+C: never leave the ball hanging
            print(f"!! stopped ({type(e).__name__}: {e}) - opening gripper and going home")
            if go:
                from viam.components.gripper import Gripper
                await Gripper.from_robot(robot, GRIPPER).open()
            raise
        finally:
            print("5) home")
            await move_to(motion, home, go)

        if state == "dropped" and go:
            hit, miss = ask_result()
            res = {"hit": hit}
            if miss is not None:
                res["miss_mm"] = miss
                res["failure"] = "missed"
            doc["result"] = res
        if go:
            doc["duration_s"] = round(time.monotonic() - t0, 1)
            doc["telemetry"] = TEL.as_doc()
            log_attempt(doc)
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
