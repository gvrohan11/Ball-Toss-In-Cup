"""Stationary rim observations, stereo cup memory, and release verification."""
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.proto.common import Pose, PoseInFrame

MEMORY_PATH = Path(__file__).resolve().parent / 'observations/cup-memory.json'
SCAN_POSES = [dict(x=350, y=y, z=445, o_z=-1, theta=180) for y in (0, 60)]


@dataclass
class CupMemory:
    center: list
    radius_mm: float
    observed_at: float
    ray_error_mm: float
    frame: str = 'world'

    def require_fresh(self, max_age=180):
        age = time.time() - self.observed_at
        if self.frame != 'world' or not 0 <= age <= max_age:
            raise RuntimeError('Cup memory expired; rescan before picking')

    def save(self):
        MEMORY_PATH.parent.mkdir(exist_ok=True)
        MEMORY_PATH.write_text(json.dumps(asdict(self), indent=2) + '\n')


def rim_circles(image):
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (7, 7), 1.5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 120,
                              param1=100, param2=45, minRadius=50, maxRadius=200)
    if circles is None:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    results = []
    angles = np.linspace(0, 2*np.pi, 180, endpoint=False)
    for x, y, radius in circles[0]:
        if x-radius < 5 or y-radius < 5 or x+radius >= image.shape[1]-5 or y+radius >= image.shape[0]-5:
            continue  # clipped rim cannot establish its center reliably
        support = np.zeros(len(angles), bool)
        for delta in (-4, -2, 0, 2, 4):
            u = np.rint(x+(radius+delta)*np.cos(angles)).astype(int)
            v = np.rint(y+(radius+delta)*np.sin(angles)).astype(int)
            samples = hsv[v, u]
            support |= (samples[:, 1] < 110) & (samples[:, 2] > 145)
        if support.mean() >= 0.65:
            results.append((float(x), float(y), float(radius)))
    return results


async def world_point(robot, point, source='cam', destination='world'):
    result = await robot.transform_pose(PoseInFrame(reference_frame=source,
        pose=Pose(x=float(point[0]), y=float(point[1]), z=float(point[2]))), destination)
    return np.array([result.pose.x, result.pose.y, result.pose.z])


def held_ball_circle(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (3, 100, 80), (28, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 100 or points[:, 1].max() < image.shape[0]-2:
            continue  # held ball is near the bottom edge on this wrist mount
        points = points[(points[:, 1] < image.shape[0]-3) &
                        (points[:, 0] > 2) & (points[:, 0] < image.shape[1]-3)]
        if len(points) < 100:
            continue
        design = np.column_stack([2*points[:, 0], 2*points[:, 1], np.ones(len(points))])
        x, y, d = np.linalg.lstsq(design, (points.astype(float)**2).sum(axis=1), rcond=None)[0]
        r2 = d+x*x+y*y
        if r2 <= 0:
            continue
        radius = math.sqrt(r2)
        residual = np.median(np.abs(np.linalg.norm(points-[x, y], axis=1)-radius))
        if 80 <= radius <= 200 and residual < 3:
            candidates.append((x, y, radius))
    if len(candidates) != 1:
        raise RuntimeError('Held ball cannot be measured unambiguously; release blocked')
    return candidates[0]


async def measure_held_ball(robot):
    arm, camera = Arm.from_robot(robot, 'arm'), Camera.from_robot(robot, 'cam')
    before = np.array((await arm.get_joint_positions()).values)
    images, metadata = await camera.get_images(timeout=10)
    age = time.time()-(metadata.captured_at.seconds+metadata.captured_at.nanos/1e9)
    if not -2 <= age <= 5:
        raise RuntimeError('Held-ball image is stale')
    color = next(i for i in images if i.name == 'color')
    image = cv2.imdecode(np.frombuffer(color.data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('Held-ball color image is unavailable')
    x, y, radius = held_ball_circle(image)
    k = (await camera.get_properties()).intrinsic_parameters
    if k.focal_x_px <= 0 or k.focal_y_px <= 0:
        raise RuntimeError('Invalid camera intrinsics')
    depth = k.focal_x_px*20/radius
    point = ((x-k.center_x_px)/k.focal_x_px*depth,
             (y-k.center_y_px)/k.focal_y_px*depth, depth)
    local = await world_point(robot, point, destination='gripper')
    after = np.array((await arm.get_joint_positions()).values)
    if np.max(np.abs(after-before)) > 0.05:
        raise RuntimeError('Arm moved during held-ball measurement')
    if not (abs(local[0]) < 60 and abs(local[1]) < 60 and 10 < local[2] < 100):
        raise RuntimeError('Orange object is outside expected gripper volume')
    return local


async def observe_rims(robot):
    arm = Arm.from_robot(robot, 'arm')
    camera = Camera.from_robot(robot, 'cam')
    before = np.array((await arm.get_joint_positions()).values)
    images, metadata = await camera.get_images(timeout=10)
    color = next(i for i in images if i.name == 'color')
    image = cv2.imdecode(np.frombuffer(color.data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError('No encoded color image')
    age = time.time() - (metadata.captured_at.seconds + metadata.captured_at.nanos / 1e9)
    if age > 5 or age < -2:
        raise RuntimeError('Camera image is stale')
    k = (await camera.get_properties()).intrinsic_parameters
    if k.focal_x_px <= 0 or k.focal_y_px <= 0:
        raise RuntimeError('Invalid camera intrinsics')
    origin = await world_point(robot, (0, 0, 0))
    observations = []
    for x, y, radius in rim_circles(image):
        ray = ((x-k.center_x_px)/k.focal_x_px*1000,
               (y-k.center_y_px)/k.focal_y_px*1000, 1000)
        far = await world_point(robot, ray)
        observations.append(dict(origin=origin, direction=far-origin,
                                 circle=(x, y, radius), focal=k.focal_x_px))
    after = np.array((await arm.get_joint_positions()).values)
    if np.max(np.abs(after-before)) > 0.05:
        raise RuntimeError('Arm moved during rim capture; wait and retry')
    return observations


def triangulate(first, second):
    a, b = first['origin'], second['origin']
    u, v = first['direction'], second['direction']
    if np.linalg.norm(b-a) < 40:
        return None
    design = np.column_stack([u, -v])
    if np.linalg.cond(design) > 50:
        return None
    s, t = np.linalg.lstsq(design, b-a, rcond=None)[0]
    if s <= 0 or t <= 0:
        return None
    p, q = a+s*u, b+t*v
    error = float(np.linalg.norm(p-q))
    center = (p+q)/2
    radii = [first['circle'][2]*s*1000/first['focal'],
             second['circle'][2]*t*1000/second['focal']]
    radius = sum(radii)/2
    if error > 3 or not 30 <= radius <= 65 or abs(radii[0]-radii[1]) > 5:
        return None
    if not (200 <= center[0] <= 1000 and abs(center[1]) <= 450 and 60 <= center[2] <= 180):
        return None
    return CupMemory(center.tolist(), radius, time.time(), error)


async def scan_cup(robot, controller):
    views = []
    for index, pose in enumerate(SCAN_POSES):
        await controller.move(pose, f'cup_scan_{index+1}')
        views.append(await observe_rims(robot))
    candidates = [cup for a in views[0] for b in views[1]
                  if (cup := triangulate(a, b)) is not None]
    if not candidates:
        raise RuntimeError('No cup verified in two views; ball will not be picked')
    cup = min(candidates, key=lambda c: math.hypot(c.center[0], c.center[1]))
    cup.save()
    controller.event('cup_remembered', center_world_mm=cup.center, radius_mm=cup.radius_mm)
    return cup


def match_memory(observations, memory):
    memory.require_fresh()
    matches = []
    for obs in observations:
        o, d = obs['origin'], obs['direction']
        if abs(d[2]) < 1e-6:
            continue
        s = (memory.center[2]-o[2])/d[2]
        if s <= 0:
            continue
        point = o+s*d
        radius = obs['circle'][2]*s*1000/obs['focal']
        distance = float(np.linalg.norm(point[:2]-np.array(memory.center[:2])))
        if distance <= 40 and abs(radius-memory.radius_mm) <= 8:
            matches.append((distance, point))
    if not matches:
        raise RuntimeError('Remembered cup is not visible; release blocked')
    matches.sort(key=lambda x: x[0])
    if len(matches) > 1 and matches[1][0]-matches[0][0] < 10:
        raise RuntimeError('Ambiguous cup association; release blocked')
    if matches[0][0] > 10:
        raise RuntimeError('Cup moved more than 10 mm; release blocked, rescan required')
    return matches[0][1]


async def verify_cup(robot, memory, controller=None):
    checks = []
    for _ in range(2):
        checks.append(match_memory(await observe_rims(robot), memory))
        if controller:
            controller.event('cup_verified', center_world_mm=checks[-1].tolist())
    if np.linalg.norm(checks[0]-checks[1]) > 5:
        raise RuntimeError('Cup observation is unstable; release blocked')
    return np.mean(checks, axis=0)
