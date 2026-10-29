"""CPU multicolor ball inspection. Never commands hardware motion."""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from viam.components.arm import Arm
from viam.components.camera import Camera

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from connection import connect
from scene_detection import SceneDetector, ball_candidates, circle_proposals
from viam.proto.common import Pose, PoseInFrame


async def world_point(robot, point, source='cam', destination='world'):
    result = await robot.transform_pose(PoseInFrame(reference_frame=source,
        pose=Pose(x=float(point[0]), y=float(point[1]), z=float(point[2]))), destination)
    return np.array([result.pose.x, result.pose.y, result.pose.z])


async def main(args):
    detector = SceneDetector(args.model)
    # Load/download weights before capturing a frame; initial startup can be slow.
    if args.backend == 'yolo':
        from ultralytics import YOLO
        detector.model = YOLO(args.model)
    robot = await asyncio.wait_for(connect(), 30)
    try:
        arm, camera = Arm.from_robot(robot, 'arm'), Camera.from_robot(robot, 'cam')
        before = np.array((await arm.get_joint_positions()).values)
        images, metadata = await camera.get_images(timeout=20)
        sources = {image.name: image for image in images}
        if not {'color', 'depth'}.issubset(sources):
            raise RuntimeError('Paired aligned color and depth are required')
        image = cv2.imdecode(np.frombuffer(sources['color'].data, np.uint8), cv2.IMREAD_COLOR)
        depth = np.asarray(sources['depth'].bytes_to_depth_array())
        if image is None or image.shape[:2] != depth.shape:
            raise RuntimeError('Color/depth images are not aligned')
        props = await camera.get_properties()
        k = props.intrinsic_parameters
        if (k.height_px, k.width_px) != depth.shape or any(abs(v)>1e-9 for v in props.distortion_parameters.parameters):
            raise RuntimeError('Rectified camera intrinsics must match aligned images')
        x0,y0,x1,y1=args.plate_roi
        roi_depth=depth[y0:y1,x0:x1]
        valid_depth=roi_depth[np.isfinite(roi_depth)&(roi_depth>0)]
        if not len(valid_depth):
            raise RuntimeError('No depth available in plate ROI')
        radius_px=k.focal_x_px*args.diameter_mm/2/np.median(valid_depth)
        instances = (await asyncio.to_thread(detector.detect, image) if args.backend=='yolo'
                     else await asyncio.to_thread(circle_proposals, image, args.plate_roi,radius_px))
        reports = ball_candidates(instances, depth, (k.focal_x_px, k.focal_y_px,
                                  k.center_x_px, k.center_y_px), args.plate_roi, args.diameter_mm)
        for report in reports:
            if report['accepted']:
                report['center_world_mm'] = (await world_point(robot, report['center_camera_mm'])).tolist()
        after = np.array((await arm.get_joint_positions()).values)
        if before.size == 0 or before.shape != after.shape or np.max(np.abs(before-after)) > 0.05:
            raise RuntimeError('Arm moved during capture/inference; discard coordinates')
        age = time.time()-(metadata.captured_at.seconds+metadata.captured_at.nanos/1e9)
        result = dict(captured_at=metadata.captured_at.ToJsonString(), age_seconds=age,
                      model=args.model if args.backend=='yolo' else 'color-independent circles + RGB-D sphere fit', plate_roi=args.plate_roi, balls=reports,
                      cups=[dict(bbox=list(i.bbox), confidence=i.confidence) for i in instances if i.label=='cup'],
                      motion_performed=False, motion_ready=False,
                      limitations='Pretrained model may miss crowded ping-pong balls. Plate geometry, occlusion, '
                                  'individual target tracking, and finger clearance need validation before picking.')
        output = ROOT/'observations'
        output.mkdir(exist_ok=True)
        cv2.imwrite(str(output/'plate.raw.jpg'),image)
        np.save(output/'plate.depth.npy',depth)
        (output/'plate.intrinsics.json').write_text(json.dumps([k.focal_x_px,k.focal_y_px,k.center_x_px,k.center_y_px]))
        for index, report in enumerate(reports):
            x0,y0,x1,y1=report['bbox']
            color=(0,255,0) if report['accepted'] else (0,0,255)
            cv2.rectangle(image,(x0,y0),(x1,y1),color,2)
            label=f"ball {index+1}" if report['accepted'] else 'rejected'
            cv2.putText(image,label,(x0,max(15,y0-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
        cv2.imwrite(str(output/'plate.latest.jpg'),image)
        (output/'plate.latest.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:
        await robot.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='.cup-models/yolo11m-seg.pt')
    parser.add_argument('--backend',choices=['yolo','geometry'],default='geometry')
    parser.add_argument('--plate-roi',nargs=4,type=int,required=True,metavar=('X0','Y0','X1','Y1'))
    parser.add_argument('--diameter-mm',type=float,default=40)
    asyncio.run(main(parser.parse_args()))
