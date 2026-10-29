"""Read-only orange sphere localization and candidate contacts. Never moves hardware."""
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from google.protobuf.json_format import MessageToDict
from viam.components.arm import Arm
from viam.components.camera import Camera
from viam.proto.common import Pose, PoseInFrame

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from connection import connect


def read_pcd(data):
    """Parse standard little-endian binary PCD, in meters, into millimeters."""
    end = data.index(b'DATA binary\n') + len(b'DATA binary\n')
    header = dict(line.split(' ', 1) for line in data[:end].decode().splitlines()
                  if ' ' in line and not line.startswith('#'))
    types = {('F', '4'): '<f4', ('F', '8'): '<f8', ('I', '4'): '<i4', ('U', '4'): '<u4'}
    if any(c != '1' for c in header['COUNT'].split()):
        raise ValueError('Unsupported PCD vector field')
    dtype = [(name, types[(kind, size)]) for name, kind, size in
             zip(header['FIELDS'].split(), header['TYPE'].split(), header['SIZE'].split())]
    values = np.frombuffer(data[end:], dtype=dtype)
    points = np.column_stack([values[k] for k in ('x', 'y', 'z')]).astype(float) * 1000
    return points[np.isfinite(points).all(axis=1) & (points[:, 2] > 0)]


def fit_sphere(points, radius=20):
    if len(points) < 100:
        raise ValueError('Too few depth points on the orange region')
    guess = np.median(points, axis=0) + [0, 0, radius / 2]
    fit = least_squares(lambda c: np.linalg.norm(points - c, axis=1) - radius,
                        guess, loss='soft_l1', f_scale=2)
    errors = np.abs(np.linalg.norm(points - fit.x, axis=1) - radius)
    fraction = float(np.mean(errors < 3))
    if np.median(errors) > 2.5 or fraction < 0.75:
        raise ValueError('Orange region does not fit a 40 mm sphere reliably')
    return fit.x, float(np.median(errors)), fraction


def fit_plane(points):
    if len(points) < 200:
        raise ValueError('Too few nearby table points')
    rng = np.random.default_rng(0)
    sample = points[rng.choice(len(points), min(len(points), 10000), replace=False)]
    best = None
    for _ in range(150):
        p = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        length = np.linalg.norm(n)
        if length < 1e-6:
            continue
        n /= length
        if n[2] < 0:
            n = -n
        if n[2] < 0.85:
            continue
        mask = np.abs((sample - p[0]) @ n) < 3
        if best is None or mask.sum() > best.sum():
            best = mask
    if best is None or best.mean() < 0.6:
        raise ValueError('No reliable horizontal support plane')
    q = sample[best]
    center = q.mean(axis=0)
    normal = np.linalg.svd(q - center, full_matrices=False)[2][-1]
    if normal[2] < 0:
        normal = -normal
    return center, normal, float(best.mean())


async def basis(robot, source, destination):
    poses = []
    for xyz in ([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]):
        p = await robot.transform_pose(PoseInFrame(reference_frame=source,
                    pose=Pose(x=xyz[0], y=xyz[1], z=xyz[2], o_z=1)), destination)
        poses.append(np.array([p.pose.x, p.pose.y, p.pose.z]))
    return np.column_stack([p - poses[0] for p in poses[1:]]), poses[0]


async def main():
    robot = await asyncio.wait_for(connect(), 30)
    output = ROOT / 'observations'
    output.mkdir(exist_ok=True)
    try:
        camera, arm = Camera.from_robot(robot, 'cam'), Arm.from_robot(robot, 'arm')
        before = np.array((await arm.get_joint_positions()).values)
        images, metadata = await camera.get_images(timeout=20)
        color = next(i for i in images if i.name == 'color')
        rgb = cv2.imdecode(np.frombuffer(color.data, np.uint8), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError('Camera did not return an encoded color image')
        cloud, _ = await camera.get_point_cloud(timeout=25)
        props = await camera.get_properties()
        rotation, origin = await basis(robot, 'cam', 'world')
        grip_rotation, grip_origin = await basis(robot, 'gripper', 'world')
        _, grip_in_cam = await basis(robot, 'gripper', 'cam')
        after = np.array((await arm.get_joint_positions()).values)
        drift = float(np.max(np.abs(after - before)))
        if drift > 0.05:
            raise ValueError(f'Arm moved {drift:.3f} degrees during capture; retry while stationary')

        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (3, 100, 60), (28, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        points = read_pcd(cloud)
        intr = props.intrinsic_parameters
        u = np.rint(points[:, 0] / points[:, 2] * intr.focal_x_px + intr.center_x_px).astype(int)
        v = np.rint(points[:, 1] / points[:, 2] * intr.focal_y_px + intr.center_y_px).astype(int)
        valid = (u >= 0) & (u < rgb.shape[1]) & (v >= 0) & (v < rgb.shape[0])
        points, u, v = points[valid], u[valid], v[valid]
        world_points = points @ rotation.T + origin
        candidates, rejected = [], []
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if area < 250 or perimeter == 0 or 4 * np.pi * area / perimeter**2 < 0.65:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if not 0.7 < w / h < 1.4:
                continue
            region = np.zeros_like(mask)
            cv2.drawContours(region, [contour], -1, 255, -1)
            region = cv2.erode(region, np.ones((3, 3), np.uint8))
            selected = points[region[v, u] > 0]
            try:
                center_cam, residual, fraction = fit_sphere(selected)
                center = rotation @ center_cam + origin
                radial = ((u - (x + w / 2)) / w)**2 + ((v - (y + h / 2)) / h)**2
                surrounding = world_points[(radial > 0.8**2) & (radial < 1.7**2)]
                plane, normal, plane_fraction = fit_plane(surrounding)
                height = float((center - plane) @ normal)
                axis = grip_rotation[:, 1]
                axis = axis - normal * (axis @ normal)
                axis /= np.linalg.norm(axis)
                candidates.append({
                    'center_camera_mm': center_cam.tolist(), 'center_world_mm': center.tolist(),
                    'contact_points_world_mm': [(center - 20 * axis).tolist(), (center + 20 * axis).tolist()],
                    'pregrasp_center_world_mm': (center + 100 * normal).tolist(),
                    'radius_assumed_mm': 20, 'sphere_residual_median_mm': residual,
                    'sphere_inlier_fraction': fraction, 'support_plane_point_world_mm': plane.tolist(),
                    'support_plane_normal_world': normal.tolist(), 'support_plane_inlier_fraction': plane_fraction,
                    'center_height_above_support_mm': height,
                    'support_consistent_with_40mm_ball': abs(height - 20) < 5,
                    'bbox_px': [x, y, x + w, y + h],
                    'validated_for_motion': False,
                    'contact_axis_note': 'Assumes gripper local Y is the finger closing axis; physically verify.',
                    'tcp_note': 'Contacts and pregrasp center are object geometry, not a commanded gripper TCP pose.',
                })
                cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            except ValueError as exc:
                rejected.append({'bbox_px': [x, y, x + w, y + h], 'reason': str(exc)})
        result = {'captured_at': metadata.captured_at.ToJsonString(),
                  'processed_at': datetime.now(timezone.utc).isoformat(),
                  'arm_drift_degrees': drift, 'camera_origin_world_mm': origin.tolist(),
                  'camera_rotation_to_world': rotation.tolist(), 'gripper_origin_world_mm': grip_origin.tolist(),
                  'gripper_origin_camera_mm': grip_in_cam.tolist(),
                  'candidates': candidates, 'rejected': rejected, 'hardware_moved_by_script': False}
        (output / 'orange-grasp.latest.json').write_text(json.dumps(result, indent=2) + '\n')
        cv2.imwrite(str(output / 'orange-grasp.latest.jpg'), rgb)
        print(json.dumps(result, indent=2))
    finally:
        await robot.close()


if __name__ == '__main__':
    asyncio.run(main())
