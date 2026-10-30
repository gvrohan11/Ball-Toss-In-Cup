"""Shared connection to the sfhack11 machine. Everyone imports this.

    from connection import connect
    robot = await connect()

Network hiccups: the venue Wi-Fi drops briefly and Viam then rejects the next call with
SESSION_EXPIRED (the SDK starts a fresh session right after). connect() installs a small
retry on the arm / gripper / camera / vision / motion client calls: transient errors
(session expired, unavailable, deadline exceeded, lost connection while the SDK redials)
are retried up to RETRIES times with backoff (~15 s). All of
these calls are safe to repeat (move to the same pose, open, grab, read).
"""
import asyncio
import functools
import os

from dotenv import load_dotenv
from viam.robot.client import RobotClient

load_dotenv()

RETRIES = 6
RETRY_WAIT_S = (0.5, 1, 2, 3, 4, 5)   # ~15 s total: long enough for the SDK to redial after a drop


def _is_transient(exc):
    try:
        from grpclib.const import Status
        from grpclib.exceptions import GRPCError, StreamTerminatedError
    except ImportError:  # pragma: no cover
        return False
    if isinstance(exc, StreamTerminatedError):
        return True
    if isinstance(exc, GRPCError):
        if exc.status == Status.INVALID_ARGUMENT and exc.message == "SESSION_EXPIRED":
            return True
        return exc.status in (Status.UNAVAILABLE, Status.DEADLINE_EXCEEDED, Status.CANCELLED)
    # After a full drop the SDK's local WebRTC proxy socket disappears until it redials:
    # connecting to it raises FileNotFoundError / ConnectionRefusedError. Wait it out.
    return isinstance(exc, (ConnectionError, FileNotFoundError, asyncio.TimeoutError, TimeoutError))


def _with_retry(fn):
    if getattr(fn, "_retrying", False):
        return fn

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):
        for attempt in range(RETRIES + 1):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                if attempt == RETRIES or not _is_transient(e):
                    raise
                print(f"  [net] {type(e).__name__}: {getattr(e, 'message', e)} - retrying "
                      f"{fn.__qualname__} ({attempt + 1}/{RETRIES})")
                await asyncio.sleep(RETRY_WAIT_S[min(attempt, len(RETRY_WAIT_S) - 1)])

    wrapper._retrying = True
    return wrapper


def install_retries():
    """Wrap the client methods we use. Idempotent."""
    from viam.components.arm.client import ArmClient
    from viam.components.camera.client import CameraClient
    from viam.components.gripper.client import GripperClient
    from viam.services.motion.client import MotionClient
    from viam.services.vision.client import VisionClient

    targets = {
        ArmClient: ["get_joint_positions", "get_end_position", "move_to_joint_positions", "stop", "do_command"],
        GripperClient: ["open", "grab", "stop", "is_holding_something", "is_moving", "do_command"],
        CameraClient: ["get_images", "get_properties", "get_point_cloud"],
        MotionClient: ["move", "get_pose"],
        VisionClient: ["get_detections_from_camera", "get_object_point_clouds"],
    }
    for cls, names in targets.items():
        for name in names:
            if hasattr(cls, name):
                setattr(cls, name, _with_retry(getattr(cls, name)))
    RobotClient.transform_pose = _with_retry(RobotClient.transform_pose)


async def connect() -> RobotClient:
    address = os.getenv("VIAM_ADDRESS")
    api_key = os.getenv("VIAM_API_KEY")
    api_key_id = os.getenv("VIAM_API_KEY_ID")
    missing = [n for n, v in [("VIAM_ADDRESS", address), ("VIAM_API_KEY", api_key),
                              ("VIAM_API_KEY_ID", api_key_id)] if not v]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)} in .env (copy .env.example to .env and fill it in)")
    install_retries()
    opts = RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
    return await RobotClient.at_address(address, opts)
