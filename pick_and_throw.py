"""Phase 2: find the cup, pick the ball up, turn toward the cup, THEN throw it.

    python pick_and_throw.py                                   # dry run: no motion
    python pick_and_throw.py --scale 0.3 --go --i-checked-with-staff
    python pick_and_throw.py --go --i-checked-with-staff       # full throw (add --far/--lob/... for a preset)

Flow: home -> set down anything held -> REQUIRED cup scan (main.scan_for_cup: the CUP_SCAN views in
order, unreachable views skipped, first cup found wins; returns the rim center in world mm) ->
find + pick the ball (same code as main.py) -> check the gripper really holds it -> home ->
turn ONLY joint 0 so the gripper/camera plane points at the cup -> throw.py's swing (keeps that
heading) -> back home -> ask where it landed -> log to Atlas (phase "throw", never counted in
the drop hit rate). No cup = no pick, no throw. If anything fails after the pick, the ball is
put back on the table.
"""
import argparse
import asyncio
import math
import time

import throw
from connection import connect
from main import move_to, put_ball_back, release_if_holding
from pick_ball import LOOK, THETA, find_ball, pick
from viam.services.motion import MotionClient
from viam.components.arm import Arm
from viam.proto.component.arm import JointPositions

HOME = dict(x=LOOK[0], y=LOOK[1], z=LOOK[2], o_x=0, o_y=0, o_z=-1, theta=THETA)


def ask_landing():
    """+cm = long (past the cup), -cm = short, blank = skip."""
    try:
        s = input("Where did it land? cm past the cup (+) or short of it (-), 'in' if it went in, "
                  "blank to skip: ").strip().lower()
    except EOFError:
        return None
    if not s:
        return None
    if s in ("in", "y", "yes"):
        return {"hit": True}
    import re
    m = re.search(r"-?\d+(?:\.\d+)?", s)   # accepts "10", "-10", "10cm", "10cms"
    return {"hit": False, "long_cm": float(m.group())} if m else None


# Throws reach farther than the drop: cup_aim.find_cup ignores cups beyond MAX_CUP_REACH_MM
# (950, the drop's reach), so the scan uses this limit instead (a cup was seen at 955 mm).
THROW_SCAN_REACH_MM = 1150
ALIGN_TOL_DEG = 1.0   # gripper bearing must match the cup bearing this closely after the turn


def cup_bearing(cup):
    """Base heading (deg, + = toward +y) and horizontal distance (mm) of a world-frame cup.
    The arm base frame is the world frame (zero translation/rotation in the Viam config), and
    joint 0 = 0 points the arm along +X, so the heading is also the absolute joint-0 angle."""
    return math.degrees(math.atan2(cup[1], cup[0])), math.hypot(cup[0], cup[1])


async def scan_for_throw_cup(motion, robot, go):
    """main.scan_for_cup (the CUP_SCAN views in order, unreachable views skipped, first cup
    found wins), with the cup-distance limit raised to the throw's range for this scan only."""
    import cup_aim
    from main import scan_for_cup
    saved = cup_aim.MAX_CUP_REACH_MM
    cup_aim.MAX_CUP_REACH_MM = THROW_SCAN_REACH_MM
    try:
        return await scan_for_cup(motion, robot, go)
    finally:
        cup_aim.MAX_CUP_REACH_MM = saved


async def align_with_cup(robot, bearing_deg: float, go: bool) -> float:
    """Turn ONLY joint 0 so the arm's vertical plane (gripper, camera, and the throw's swing
    plane) passes through the cup. Joints 1-5 keep their angles. Run at home.
    Sets joint 0 to the cup's absolute bearing (not current + bearing), then checks the
    gripper's measured world position actually points at that bearing.
    Returns the joint-0 angle so the throw keeps exactly this heading."""
    if not math.isfinite(bearing_deg) or abs(bearing_deg) > 45:
        raise ValueError("cup bearing must be finite and within +/-45 degrees")
    if not go:
        print(f"   preview: joint 0 -> {bearing_deg:+.1f} deg (gripper and camera face the cup)")
        return bearing_deg
    arm = Arm.from_robot(robot, throw.ARM)
    current = list((await arm.get_joint_positions()).values)
    target = list(current)
    target[0] = bearing_deg + 360.0 * round((current[0] - bearing_deg) / 360.0)  # same turn, no wrap
    throw.check_limits(current, target)
    print(f"   joint 0: {current[0]:+.1f} -> {target[0]:+.1f} deg (other joints unchanged)")
    await arm.do_command({"set_speed": throw.NORMAL_SPEED})
    await arm.move_to_joint_positions(JointPositions(values=target))
    actual = list((await arm.get_joint_positions()).values)
    if abs(actual[0] - target[0]) > 0.5 or max(abs(a - c) for a, c in zip(actual[1:], current[1:])) > 0.5:
        raise RuntimeError(f"joint move did not land on the cup heading ({actual}); refusing to throw")
    pose = await arm.get_end_position()
    seen = math.degrees(math.atan2(pose.y, pose.x))
    if abs(seen - bearing_deg) > ALIGN_TOL_DEG:
        raise RuntimeError(f"gripper points at {seen:+.1f} deg, cup is at {bearing_deg:+.1f} deg; refusing to throw")
    print(f"   aligned: gripper at ({pose.x:.0f}, {pose.y:.0f}) -> bearing {seen:+.1f} deg")
    return target[0]


async def run(args):
    go = args.go and args.i_checked_with_staff
    if args.go and not go:
        print("Refusing to throw live without --i-checked-with-staff.")
    print("MODE:", "LIVE (arm will move and throw)" if go else "DRY RUN (no motion)")
    t0 = time.monotonic()
    robot = await connect()
    try:
        motion = MotionClient.from_robot(robot, "builtin")
        await move_to(motion, HOME, go, "home")
        await release_if_holding(robot, motion, go)
        await move_to(motion, HOME, go, "home")

        print("0) required cup scan")
        cup = await scan_for_throw_cup(motion, robot, go)
        if cup is None:
            print("no suitable cup found; not picking or throwing")
            await move_to(motion, HOME, go, "home")
            return
        if len(cup) != 3 or not all(math.isfinite(v) for v in cup):
            raise ValueError("cup scanner returned an invalid world-frame position")
        yaw, distance = cup_bearing(cup)
        print(f"   cup rim at ({cup[0]:.0f}, {cup[1]:.0f}, {cup[2]:.0f}) mm world: "
              f"bearing {yaw:+.1f} deg, {distance:.0f} mm from the base")
        if abs(yaw) > 45:
            print("cup is outside the supported +/-45 degree throw sector; not picking")
            await move_to(motion, HOME, go, "home")
            return
        preset = next((n for n in ("power", "pitch", "whip", "far") if getattr(args, n, False)), None)
        # Range check (same rules as the old --aim-cup): presets have a calibrated distance table,
        # the lob lands ~LOB_RELEASE_RADIUS_MM out; the default swing has no range table.
        rng = None
        if preset:
            table = throw.PRESETS[preset]["table"]
            rng = (table[0][1] - 60, table[-1][1] + 40)
        elif args.lob:
            rng = (throw.LOB_RELEASE_RADIUS_MM - 80, throw.LOB_RELEASE_RADIUS_MM + 80)
        if rng and not rng[0] <= distance <= rng[1] and not getattr(args, "force_aim", False) and args.speed is None:
            print(f"cup is {distance:.0f} mm from the base; this throw reaches {rng[0]:.0f}-{rng[1]:.0f} mm. "
                  "Move the cup, or use --force-aim.")
            await move_to(motion, HOME, go, "home")
            return
        await move_to(motion, HOME, go, "home")

        print("1) find ball")
        ball = await find_ball(robot, motion, go)
        if ball is None:
            print("no ball found")
            return
        print("2) pick")
        held = await pick(robot, ball, go)
        if go and not held:
            print("pick missed; nothing to throw")
            await move_to(motion, HOME, go, "home")
            return

        try:
            await move_to(motion, HOME, go, "home")
            print("3) align joint 0 with cup")
            base_heading = await align_with_cup(robot, yaw, go)
            print("4) throw")
            if preset:
                fp = throw.far_params(distance, args.speed, args.accel, args.release_delay, preset)
                info = await throw.throw(robot, distance, go, args.scale, not args.no_release,
                                         fp["release_delay_s"], fp["swing_deg"], fp["speed_dps"], fp["accel_dps2"],
                                         None, fp["start_j"], yaw, fp["gripper_speed"],
                                         getattr(args, "direct", False) or fp["direct"],
                                         aligned_base_deg=base_heading)
            else:
                info = await throw.throw(robot, distance, go, args.scale, not args.no_release,
                                         args.release_delay, throw.swing_from_args(args), args.speed, args.accel,
                                         args.wind_z, throw.LOB_START_J if args.lob else None, yaw, None,
                                         getattr(args, "direct", False), aligned_base_deg=base_heading)
        except BaseException as e:  # error or Ctrl+C: don't leave the ball in the gripper
            print(f"!! stopped ({type(e).__name__}: {e}); putting the ball back")
            await asyncio.sleep(2.0)  # after a controller stop the arm refuses commands for a moment
            await put_ball_back(robot, motion, ball, go)
            raise
        await move_to(motion, HOME, go, "home")
        if go:
            landing = ask_landing()
            doc = {"phase": "throw", "ball": {"pos_mm": ball}, "throw": info,
                   "cup": {"rim_world_mm": list(cup), "bearing_deg": yaw},
                   "duration_s": round(time.monotonic() - t0, 1),
                   "result": landing or {"hit": None}}
            try:
                import logger
                logger.log_attempt(doc)
            except Exception as e:  # noqa: BLE001
                print(f"  [log failed: {e}]")
    finally:
        await robot.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance", type=float, default=None, help="legacy argument; distance is measured by the required cup scan")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--no-release", action="store_true")
    ap.add_argument("--speed", type=float, default=None, help="swing speed deg/s (default 150, max 180)")
    ap.add_argument("--accel", type=float, default=None, help="swing accel deg/s2 (default 900, max 1145)")
    ap.add_argument("--wind-z", type=float, default=None, help="wind-up height mm (default 450; try 650 for a lob)")
    ap.add_argument("--lob", action="store_true", help="near-vertical full-speed toss (cup ~50 cm from the base)")
    ap.add_argument("--yaw", type=float, default=0.0, help="legacy argument; heading is measured by the required cup scan")
    ap.add_argument("--far", action="store_true",
                    help="open-hand forward toss for a cup 84-98 cm out (with --aim-cup the speed comes from the cup distance)")
    ap.add_argument("--power", action="store_true", help="full-force throw (always direct mode)")
    ap.add_argument("--direct", action="store_true", help="point-to-point mode: no ServoJ speed stop, steadier release")
    ap.add_argument("--pitch", action="store_true", help="underhand pitch: long fast backswing, steep launch")
    ap.add_argument("--whip", action="store_true", help="wrist-whip throw, ~2x faster, 42-53 deg up")
    ap.add_argument("--force-aim", action="store_true", help="with --aim-cup: throw even if the cup is out of range")
    ap.add_argument("--aim-cup", action="store_true",
                    help="legacy flag; scanning is always required")
    ap.add_argument("--j2", type=float, default=None, help="shoulder swing deg (negative = forward+down)")
    ap.add_argument("--j3", type=float, default=None, help="elbow swing deg (default -35)")
    ap.add_argument("--j5", type=float, default=None, help="wrist swing deg (default +45)")
    ap.add_argument("--release-delay", type=float, default=None,
                    help="seconds after the swing starts to open the gripper; negative = open before the swing; try -0.3 / -0.2 / -0.1")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--i-checked-with-staff", action="store_true")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
