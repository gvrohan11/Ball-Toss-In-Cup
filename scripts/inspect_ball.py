"""Read ball detections and 3D geometry through Viam, without moving hardware."""
import asyncio
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from google.protobuf.json_format import MessageToDict
from viam.robot.client import RobotClient
from viam.services.vision import VisionClient
from viam.proto.common import PoseInFrame


async def main():
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    # our .env uses VIAM_ADDRESS; accept either name
    if os.getenv('VIAM_ADDRESS') and not os.getenv('VIAM_MACHINE_ADDRESS'):
        os.environ['VIAM_MACHINE_ADDRESS'] = os.environ['VIAM_ADDRESS']
    names = ('VIAM_MACHINE_ADDRESS', 'VIAM_API_KEY_ID', 'VIAM_API_KEY')
    missing = [name for name in names if not os.getenv(name)]
    if missing:
        raise SystemExit('Set these values in .env: ' + ', '.join(missing))
    address = os.environ['VIAM_MACHINE_ADDRESS'].strip()
    if re.fullmatch(r'[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', address):
        raise SystemExit('VIAM_MACHINE_ADDRESS must be the CONNECT-tab network '
                         'hostname, not the machine UUID.')
    options = RobotClient.Options.with_api_key(
        api_key=os.environ['VIAM_API_KEY'],
        api_key_id=os.environ['VIAM_API_KEY_ID'],
    )
    robot = await asyncio.wait_for(RobotClient.at_address(address, options), timeout=30)
    try:
        detector = VisionClient.from_robot(robot, 'ball-detector')
        segmenter = VisionClient.from_robot(robot, 'ball-3d')
        detections = await detector.get_detections_from_camera('cam', timeout=20)
        objects = await segmenter.get_object_point_clouds('cam', timeout=30)
        output_dir = Path(__file__).resolve().parents[1] / 'observations'
        output_dir.mkdir(exist_ok=True)
        for index, obj in enumerate(objects):
            (output_dir / f'ball.{index}.pcd').write_bytes(obj.point_cloud)
        result = {
            'detections': [MessageToDict(d) for d in detections],
            'objects': [
                {'geometries': MessageToDict(obj.geometries),
                 'point_cloud_bytes': len(obj.point_cloud)}
                for obj in objects
            ],
            'note': 'Raw segmenter geometry; verify reference frame, ball size, '
                    'camera calibration and gripper TCP before planning a grasp.',
        }
        candidates = []
        for obj in objects:
            for geometry in obj.geometries.geometries:
                if not geometry.HasField('box'):
                    continue
                dims = geometry.box.dims_mm
                if not (30 <= dims.x <= 50 and 30 <= dims.y <= 50 and 10 <= dims.z <= 55):
                    continue
                world = await robot.transform_pose(
                    PoseInFrame(reference_frame=obj.geometries.reference_frame,
                                pose=geometry.center), 'world')
                center = world.pose
                candidates.append({
                    'reference_frame': 'world',
                    'estimated_center_mm': {'x': center.x, 'y': center.y, 'z': center.z},
                    'assumed_ball_diameter_mm': 40,
                    'example_contact_pair_mm': [
                        {'x': center.x, 'y': center.y - 20, 'z': center.z},
                        {'x': center.x, 'y': center.y + 20, 'z': center.z},
                    ],
                    'validated_for_motion': False,
                    'limitations': 'Bounding-box center is not a fitted sphere center. '
                                   'Contact axis is an example; verify finger closing axis, '
                                   'TCP, table height and camera calibration.',
                })
        result['grasp_candidates'] = candidates
        print(json.dumps(result, indent=2))
        (output_dir / 'ball.latest.json').write_text(json.dumps(result, indent=2) + '\n')
    finally:
        await robot.close()


if __name__ == '__main__':
    asyncio.run(main())
