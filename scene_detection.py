"""CPU ball geometry and optional YOLO11 inspection; never commands motion.

This does not replace the validated single-orange-ball motion loop.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares


@dataclass
class Instance:
    label: str
    confidence: float
    bbox: tuple
    mask: np.ndarray
    source: str = 'yolo11'


def circle_proposals(bgr, roi, radius_px=None):
    """Color-independent circular proposals, not semantic object recognition.

    All proposals must subsequently pass 3D sphere fitting. This is useful when
    COCO labels orange ping-pong balls as fruit and misses white ping-pong balls.
    """
    x0,y0,x1,y1=roi
    gray=cv2.cvtColor(bgr[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),1.0)
    minimum,maximum=(max(5,int(radius_px*.85)),int(radius_px*1.12)) if radius_px else (15,60)
    circles=cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1.2,max(25,minimum*1.5),
                            param1=60,param2=25,minRadius=minimum,maxRadius=maximum)
    if circles is None:
        return []
    proposals=[]
    for x,y,radius in circles[0]:
        x,y=float(x+x0),float(y+y0)
        mask=np.zeros(bgr.shape[:2],np.uint8)
        cv2.circle(mask,(round(x),round(y)),round(radius),1,-1)
        proposals.append(Instance('sports ball',0.0,
                         (round(x-radius),round(y-radius),round(x+radius),round(y+radius)),
                         mask.astype(bool),'circle_proposal_not_semantically_classified'))
    return proposals


class SceneDetector:
    def __init__(self, checkpoint='.cup-models/yolo11m-seg.pt', model=None):
        self.checkpoint, self.model = checkpoint, model

    def detect(self, bgr):
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.checkpoint)
        names = self.model.names
        items = names.items() if isinstance(names, dict) else enumerate(names)
        classes = [int(i) for i, name in items if name in ('sports ball', 'cup')]
        if not classes:
            raise ValueError('Checkpoint has no sports ball or cup classes')
        results = self.model.predict(bgr, classes=classes, conf=0.25, imgsz=1280,
                                     device='cpu', verbose=False, max_det=200, retina_masks=True)
        instances = []
        for result in results:
            if result.boxes is None or result.masks is None:
                continue  # boxes alone cannot separate touching balls for grasp geometry
            for i, (box, confidence, cls) in enumerate(zip(result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy())):
                if i >= len(result.masks.xy):
                    continue
                polygon = np.asarray(result.masks.xy[i], dtype=np.int32)
                if len(polygon) < 3:
                    continue
                mask = np.zeros(bgr.shape[:2], np.uint8)
                cv2.fillPoly(mask, [polygon], 1)
                instances.append(Instance(names[int(cls)], float(confidence),
                                           tuple(map(int, box)), mask.astype(bool)))
        return instances


def ball_candidates(instances, depth_mm, intrinsics, roi, diameter_mm=40):
    """Fit separate known-size spheres using each mask; no color thresholds.

    3D output is in camera coordinates, not gripper commands. Clearance tests
    account only for detected neighboring balls, not hidden balls or plate walls.
    """
    if not 10 <= diameter_mm <= 100:
        raise ValueError('Provide a measured ball diameter from 10 to 100 mm')
    x0, y0, x1, y1 = roi
    height, width = depth_mm.shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('Plate ROI must be inside the depth image')
    fx, fy, cx, cy = intrinsics
    if not np.isfinite(intrinsics).all() or min(fx, fy) <= 0:
        raise ValueError('Invalid intrinsics')
    radius = diameter_mm/2
    reports = []
    for instance in instances:
        if instance.label != 'sports ball':
            continue
        report = dict(bbox=list(instance.bbox), confidence=instance.confidence, source=instance.source,
                      accepted=False, grasp_ready=False)
        reports.append(report)
        mask = instance.mask
        if mask.shape != depth_mm.shape:
            report['reason'] = 'mask/depth shape mismatch'
            continue
        # Entire visible instance must be in the plate ROI, never a neighboring cup.
        ys, xs = np.nonzero(mask)
        if len(xs) < 80 or xs.min() <= x0+2 or xs.max() >= x1-2 or ys.min() <= y0+2 or ys.max() >= y1-2:
            report['reason'] = 'insufficient mask or touches plate ROI boundary'
            continue
        ratio = (xs.max()-xs.min()+1)/(ys.max()-ys.min()+1)
        if not 0.65 <= ratio <= 1.55:
            report['reason'] = 'visible instance is not sufficiently round'
            continue
        interior = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        v, u = np.nonzero(interior & np.isfinite(depth_mm) & (depth_mm > 0))
        if len(u) < 80:
            report['reason'] = 'insufficient aligned depth'
            continue
        z = depth_mm[v, u].astype(float)
        keep = np.abs(z-np.median(z)) < diameter_mm
        z, u, v = z[keep], u[keep], v[keep]
        if len(z) < 80:
            report['reason'] = 'depth inconsistent with ball size'
            continue
        points = np.column_stack([(u-cx)/fx*z, (v-cy)/fy*z, z])[::max(1, len(z)//5000)]
        initial = np.median(points, axis=0)+[0, 0, radius/2]
        fit = least_squares(lambda c: np.linalg.norm(points-c, axis=1)-radius,
                            initial, loss='soft_l1', f_scale=2)
        error = np.abs(np.linalg.norm(points-fit.x, axis=1)-radius)
        inliers = float(np.mean(error < 3))
        report.update(sphere_error_mm=float(np.median(error)), inlier_fraction=inliers)
        extent = np.ptp(points, axis=0)
        expected_px=fx*diameter_mm/fit.x[2]
        observed_px=max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)
        if np.median(error) > 2.5 or inliers < 0.75 or min(extent[:2]) < 0.7*diameter_mm or max(extent[:2]) > 1.3*diameter_mm or extent[2] < 0.12*diameter_mm or not .9 <= observed_px/expected_px <= 1.2:
            report['reason'] = 'sphere geometry not validated'
            continue
        report.update(accepted=True, center_camera_mm=fit.x.tolist(),
                      radius_assumed_mm=radius, reason='sphere fitted; motion/plate clearance not verified')
    accepted = [r for r in reports if r['accepted']]
    for item in accepted:
        center = np.array(item['center_camera_mm'])
        neighbors = [float(np.linalg.norm(center-np.array(r['center_camera_mm'])))-diameter_mm
                     for r in accepted if r is not item]
        item['nearest_detected_ball_gap_mm'] = min(neighbors) if neighbors else None
        item['isolated_from_detected_neighbors'] = not neighbors or min(neighbors) >= 20
    return reports
