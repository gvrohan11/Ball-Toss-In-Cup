"""Viam local-module entry point: python -m cup_detection SOCKET_PATH."""
import asyncio

from viam.module.module import Module

from .service import RedSoloCupVision


if __name__ == "__main__":
    asyncio.run(Module.run_with_models(RedSoloCupVision))
