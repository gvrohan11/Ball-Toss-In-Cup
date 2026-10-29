"""Shared connection to the sfhack11 machine. Everyone imports this.

    from connection import connect
    robot = await connect()
"""
import os

from dotenv import load_dotenv
from viam.robot.client import RobotClient

load_dotenv()


async def connect() -> RobotClient:
    address = os.getenv("VIAM_ADDRESS") or os.getenv("VIAM_MACHINE_ADDRESS")
    api_key = os.getenv("VIAM_API_KEY")
    api_key_id = os.getenv("VIAM_API_KEY_ID")
    missing = [n for n, v in [("VIAM_ADDRESS", address), ("VIAM_API_KEY", api_key),
                              ("VIAM_API_KEY_ID", api_key_id)] if not v]
    if missing:
        raise SystemExit(f"Missing {', '.join(missing)} in .env (copy .env.example to .env and fill it in)")
    opts = RobotClient.Options.with_api_key(api_key=api_key, api_key_id=api_key_id)
    return await RobotClient.at_address(address, opts)
