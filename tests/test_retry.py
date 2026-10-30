"""Transient network errors are retried; real errors are not."""
import asyncio
import os
import sys

import pytest
from grpclib.const import Status
from grpclib.exceptions import GRPCError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import connection  # noqa: E402

connection.RETRY_WAIT_S = (0,)


def flaky(errors, result="ok"):
    calls = []

    async def fn(*a, **k):
        calls.append(1)
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        return result
    return connection._with_retry(fn), calls


def test_session_expired_is_retried():
    fn, calls = flaky([GRPCError(Status.INVALID_ARGUMENT, "SESSION_EXPIRED")])
    assert asyncio.run(fn()) == "ok" and len(calls) == 2


def test_gives_up_after_retries():
    err = GRPCError(Status.UNAVAILABLE, "down")
    fn, calls = flaky([err] * (connection.RETRIES + 1))
    with pytest.raises(GRPCError):
        asyncio.run(fn())
    assert len(calls) == connection.RETRIES + 1


def test_real_errors_are_not_retried():
    fn, calls = flaky([GRPCError(Status.INVALID_ARGUMENT, "bad pose")])
    with pytest.raises(GRPCError):
        asyncio.run(fn())
    assert len(calls) == 1


def test_install_wraps_client_methods():
    connection.install_retries()
    connection.install_retries()   # idempotent
    from viam.components.gripper.client import GripperClient
    from viam.services.motion.client import MotionClient
    assert getattr(GripperClient.grab, "_retrying", False)
    assert getattr(MotionClient.move, "_retrying", False)


def test_lost_proxy_socket_is_retried():
    fn, calls = flaky([FileNotFoundError(2, "No such file or directory")] * 2)
    assert asyncio.run(fn()) == "ok" and len(calls) == 3
