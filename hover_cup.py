"""Temporary gripper-hover tester. Preview by default; --execute enables motion.

python hover_cup.py --detector-config config/cup-detector.thresholds.json
python hover_cup.py --detector-config config/cup-detector.thresholds.json --execute
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time

import numpy as np
from viam.components.arm import Arm
from viam.proto.common import (GeometriesInFrame, Geometry, Pose, PoseInFrame,
                               RectangularPrism, Vector3, WorldState)
from viam.robot.client import RobotClient
from viam.services.motion import MotionClient

from connection import connect
from cup_vision.client import CupLocator, LocatedCup
from cup_vision.core import CupDetector, DetectorConfig

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class HoverSettings:
    clearance_mm: float = 400
    speed_multiplier: float = 3
    baseline_joint_speed_deg_per_sec: float = 60
    gripper: str = "gripper"
    samples: int = 5
    stability_mm: float = 12
    target_match_mm: float = 35
    max_tilt_degrees: float = 25
    max_translation_mm: float = 600
    min_detection_confidence: float = 0.5
    # Bounds for this workcell, in the current world == arm-base coordinates.
    minimum_mm: tuple[float, float, float] = (0, -500, -23)
    maximum_mm: tuple[float, float, float] = (750, 500, 800)
    camera: str = "cam"
    arm: str = "arm"
    motion_service: str = "builtin"
    base_frame: str = "world"

    def validate(self) -> None:
        speed = self.baseline_joint_speed_deg_per_sec * self.speed_multiplier
        if (not np.isfinite([speed, self.baseline_joint_speed_deg_per_sec, self.speed_multiplier]).all()
                or self.speed_multiplier <= 0 or not 3 <= self.baseline_joint_speed_deg_per_sec <= 180
                or not 3 <= speed <= 180):
            raise ValueError("Requested xArm joint speed must be between 3 and 180 degrees/s")
        if not 0 < self.min_detection_confidence <= 1:
            raise ValueError("min_detection_confidence must be in (0,1]")
        for name in ("clearance_mm", "stability_mm", "target_match_mm", "max_translation_mm", "max_tilt_degrees"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_tilt_degrees >= 90:
            raise ValueError("max_tilt_degrees must be below 90")
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 3:
            raise ValueError("samples must be an integer >= 3")
        low, high = np.asarray(self.minimum_mm), np.asarray(self.maximum_mm)
        if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(low >= high):
            raise ValueError("Workspace bounds must be finite XYZ triples with minimum < maximum")


@dataclass(frozen=True)
class HoverPlan:
    rim_center_mm: tuple[float, float, float]
    gripper_target_mm: tuple[float, float, float]
    current_gripper_mm: tuple[float, float, float]
    diameter_mm: float
    height_mm: float | None
    observed_spread_mm: float
    created_monotonic: float
    joint_speed_deg_per_sec: float = 180


class CupHoverController:
    """Estimate an upright cup's rim, then hover the gripper tip above it.

    This class is independent of the vision module. Motion targets `gripper`, so
    Viam's frame system accounts for camera/arm/gripper offsets and orientation.
    No guessed flange offsets, direct joint moves or gripper operations.
    """

    def __init__(self, robot: RobotClient, detector: CupDetector,
                 settings: HoverSettings | None = None) -> None:
        self.robot = robot
        self.settings = settings or HoverSettings()
        self.settings.validate()
        self.locator = CupLocator(robot, detector, camera=self.settings.camera,
                                  arm=self.settings.arm, base_frame=self.settings.base_frame)
        self.motion = MotionClient.from_robot(robot, self.settings.motion_service)
        self.arm = Arm.from_robot(robot, self.settings.arm)

    def _check_bounds(self, point: tuple[float, float, float] | np.ndarray) -> None:
        value = np.asarray(point)
        if not np.isfinite(value).all() or np.any(value < self.settings.minimum_mm) or np.any(value > self.settings.maximum_mm):
            raise ValueError(f"Point {tuple(value)} falls outside configured workcell bounds")

    async def plan(self, cup_index: int | None = None) -> HoverPlan:
        """Compute targets only. This does not invoke the motion planner or move."""
        selected: list[LocatedCup] = []
        joint_start = np.asarray((await self.arm.get_joint_positions(timeout=10)).values)
        for attempt in range(self.settings.samples * 3):
            cups = sorted((cup for cup in await self.locator.sample()
                           if cup.confidence >= self.settings.min_detection_confidence),
                          key=lambda cup: cup.bbox[0])
            if not cups:
                LOGGER.warning("No usable rim in observation %d; retrying while arm stays still", attempt + 1)
                await asyncio.sleep(0.1)
                continue
            if not selected:
                if cup_index is None and len(cups) != 1:
                    raise ValueError("Multiple cups visible; pass --cup-index (left-to-right, zero-based)")
                index = 0 if cup_index is None else cup_index
                if index < 0 or index >= len(cups):
                    raise ValueError("cup-index is outside the detected cup list")
                cup = cups[index]
            else:
                reference = np.asarray(selected[0].center_mm)
                cup = min(cups, key=lambda candidate: np.linalg.norm(np.asarray(candidate.center_mm) - reference))
                if np.linalg.norm(np.asarray(cup.center_mm) - reference) > self.settings.target_match_mm:
                    raise RuntimeError("Selected cup was lost or moved; no motion planned")
            if abs(cup.normal[2]) < np.cos(np.deg2rad(self.settings.max_tilt_degrees)):
                raise RuntimeError("Cup rim is too tilted for this upright-cup hover tester")
            selected.append(cup)
            if len(selected) == self.settings.samples:
                break
            await asyncio.sleep(0.1)
        if len(selected) < self.settings.samples:
            raise RuntimeError(f"Only {len(selected)}/{self.settings.samples} usable rim observations. "
                               "No motion planned. See /tmp/cup-hover-debug/diagnostics.json")
        joint_end = np.asarray((await self.arm.get_joint_positions(timeout=10)).values)
        if joint_start.size == 0 or joint_start.shape != joint_end.shape or np.max(np.abs(joint_end - joint_start)) > 0.2:
            raise RuntimeError("Arm moved between samples; discard plan")
        centers = np.asarray([cup.center_mm for cup in selected])
        center = np.median(centers, axis=0)
        spread = float(np.linalg.norm(centers - center, axis=1).max())
        if spread > self.settings.stability_mm:
            raise RuntimeError(f"Rim estimate is unstable ({spread:.1f} mm); no motion planned")
        self._check_bounds(center)
        target = center + (0, 0, self.settings.clearance_mm)
        self._check_bounds(target)
        current_pose = await self.motion.get_pose(self.settings.gripper, self.settings.base_frame, timeout=10)
        current = np.array([current_pose.pose.x, current_pose.pose.y, current_pose.pose.z])
        self._check_bounds(current)
        if np.linalg.norm(target - current) > self.settings.max_translation_mm:
            raise ValueError("Hover target exceeds max_translation_mm; reposition manually and retest")
        heights = [cup.height_mm for cup in selected if cup.height_mm is not None]
        return HoverPlan(tuple(map(float, center)), tuple(map(float, target)), tuple(map(float, current)),
                         float(np.median([cup.diameter_mm for cup in selected])),
                         float(np.median(heights)) if heights else None, spread, time.monotonic(),
                         self.settings.baseline_joint_speed_deg_per_sec * self.settings.speed_multiplier)

    async def execute(self, plan: HoverPlan) -> None:
        """Use collision-aware Viam motion; stop the arm on failed/cancelled moves."""
        if time.monotonic() - plan.created_monotonic > 5:
            raise RuntimeError("Plan is stale; obtain a fresh plan before executing")
        current = await self.motion.get_pose(self.settings.gripper, self.settings.base_frame, timeout=10)
        xyz = np.array([current.pose.x, current.pose.y, current.pose.z])
        if np.linalg.norm(xyz - plan.current_gripper_mm) > 2:
            raise RuntimeError("Gripper moved after planning; obtain a fresh plan")
        target = np.asarray(plan.gripper_target_mm)
        transit_z = max(xyz[2], target[2])
        # Conservatively include the whole cup column from the workspace floor
        # when a surrounding support plane was not visible to estimate height.
        rim = np.asarray(plan.rim_center_mm)
        bottom = max(self.settings.minimum_mm[2], rim[2] - plan.height_mm) if plan.height_mm else self.settings.minimum_mm[2]
        height = max(1, rim[2] - bottom)
        cup_obstacle = GeometriesInFrame(reference_frame=self.settings.base_frame, geometries=[Geometry(
            label="detected-cup", center=Pose(x=rim[0], y=rim[1], z=bottom + height / 2, o_z=1),
            box=RectangularPrism(dims_mm=Vector3(x=plan.diameter_mm + 20,
                                                y=plan.diameter_mm + 20, z=height + 10)))])
        world_state = WorldState(obstacles=[cup_obstacle])
        lift = Pose()
        lift.CopyFrom(current.pose)
        lift.z = transit_z
        # Point the gripper's +Z axis down; Viam resolves the tool mount.
        over = Pose(x=target[0], y=target[1], z=transit_z, o_z=-1)
        final = Pose(x=target[0], y=target[1], z=target[2], o_z=-1)
        speed_set = False
        try:
            speed = self.settings.baseline_joint_speed_deg_per_sec * self.settings.speed_multiplier
            LOGGER.warning("Setting joint speed to %.1f degrees/s (%.1fx baseline)", speed, self.settings.speed_multiplier)
            await self.arm.do_command({"set_speed": float(speed)}, timeout=10)
            speed_set = True
            for pose in (lift, over, final):
                self._check_bounds((pose.x, pose.y, pose.z))
                LOGGER.warning("Moving GRIPPER to (%.1f, %.1f, %.1f) mm in %s",
                               pose.x, pose.y, pose.z, self.settings.base_frame)
                moved = await self.motion.move(
                    component_name=self.settings.gripper,
                    destination=PoseInFrame(reference_frame=self.settings.base_frame, pose=pose),
                    world_state=world_state, timeout=45)
                if not moved:
                    raise RuntimeError("Motion service returned false; stop and inspect the plan")
        except BaseException:
            await asyncio.shield(self.arm.stop(timeout=5))
            raise
        finally:
            if speed_set:
                await asyncio.shield(self.arm.do_command(
                    {"set_speed": float(self.settings.baseline_joint_speed_deg_per_sec)}, timeout=10))
        actual = await self.motion.get_pose(self.settings.gripper, self.settings.base_frame, timeout=10)
        error = np.linalg.norm(np.array([actual.pose.x, actual.pose.y, actual.pose.z]) - target)
        LOGGER.info("Hover finished; reported gripper target error %.1f mm", error)
        if error > 10:
            raise RuntimeError("Gripper did not reach the hover target within 10 mm")


async def _run(args: argparse.Namespace) -> None:
    config_data = json.loads(args.detector_config.read_text()) if args.detector_config else {}
    settings_data = json.loads(args.hover_config.read_text()) if args.hover_config else {}
    settings = HoverSettings(**settings_data)
    detector = CupDetector(DetectorConfig(**config_data))
    robot = await connect()
    try:
        controller = CupHoverController(robot, detector, settings)
        plan = await controller.plan(args.cup_index)
        print(json.dumps(asdict(plan), indent=2))
        if args.execute:
            await controller.execute(plan)
        else:
            LOGGER.info("PREVIEW ONLY: no motion requested. Path feasibility is checked only by Move during execution.")
    finally:
        await robot.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector-config", type=Path)
    parser.add_argument("--hover-config", type=Path)
    parser.add_argument("--cup-index", type=int)
    parser.add_argument("--execute", action="store_true", help="Enable real arm motion; have an operator at the E-stop")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        asyncio.run(_run(args))
    except (RuntimeError, ValueError) as exc:
        LOGGER.error("Hover stopped: %s", exc)
        raise SystemExit(2) from None
    except KeyboardInterrupt:
        LOGGER.info("Stopped by user")


if __name__ == "__main__":
    main()
