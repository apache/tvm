# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

""" Hexagon testing fixtures used to deduce testing argument
    values from testing parameters """

import os
import random
import socket
from typing import Optional

import pytest

import tvm
import tvm.rpc.tracker
from tvm.contrib.hexagon.build import HexagonLauncher

HEXAGON_TOOLCHAIN = "HEXAGON_TOOLCHAIN"
TVM_TRACKER_HOST = "TVM_TRACKER_HOST"
TVM_TRACKER_PORT = "TVM_TRACKER_PORT"
ANDROID_REMOTE_DIR = "ANDROID_REMOTE_DIR"
ANDROID_SERIAL_NUMBER = "ANDROID_SERIAL_NUMBER"
ADB_SERVER_SOCKET = "ADB_SERVER_SOCKET"


@tvm.testing.fixture
def shape_nhwc(batch, in_channel, in_size):
    return (batch, in_size, in_size, in_channel)


def _compose(args, decs):
    """Helper to apply multiple markers"""
    if len(args) > 0:
        f = args[0]
        for d in reversed(decs):
            f = d(f)
        return f
    return decs


def requires_hexagon_toolchain(*args):
    _requires_hexagon_toolchain = [
        pytest.mark.skipif(
            os.environ.get(HEXAGON_TOOLCHAIN) == None,
            reason=f"Missing environment variable {HEXAGON_TOOLCHAIN}.",
        ),
    ]

    return _compose(args, _requires_hexagon_toolchain)


@tvm.testing.fixture
def android_serial_number() -> Optional[str]:
    serial = os.getenv(ANDROID_SERIAL_NUMBER, default="")
    # Setting ANDROID_SERIAL_NUMBER to an empty string should be
    # equivalent to having it unset.
    if not serial.strip():
        serial = None
    return serial


# NOTE on server ports:
# These tests use different port numbers for the RPC server (7070 + ...).
# The reason is that an RPC session cannot be gracefully closed without
# triggering TIME_WAIT state on the server socket. This prevents another
# server to bind to the same port until the wait time elapses.

listen_port_min = 2000  # Well above the privileged ports (1024 or lower)
listen_port_max = 9000  # Below the search range end (port_end=9199) of RPC server
previous_port = None


def get_free_port():

    global previous_port
    if previous_port is None:
        port = random.randint(listen_port_min, listen_port_max)
    else:
        port = previous_port + 1

    while tvm.contrib.hexagon.build._is_port_in_use(port):
        port = port + 1 if port < listen_port_max else listen_port_min

    previous_port = port
    return port


@pytest.fixture(scope="session")
def _tracker_info() -> (str, int):
    env_tracker_host = os.getenv(TVM_TRACKER_HOST, default="")
    env_tracker_port = os.getenv(TVM_TRACKER_PORT, default="")

    if env_tracker_host or env_tracker_port:
        # A tracker is already running, and we should connect to it
        # when running tests.
        assert env_tracker_host, "TVM_TRACKER_PORT is defined, but TVM_TRACKER_HOST is not"
        assert env_tracker_port, "TVM_TRACKER_HOST is defined, but TVM_TRACKER_PORT is not"
        env_tracker_port = int(env_tracker_port)

        try:
            tvm.rpc.connect_tracker(env_tracker_host, env_tracker_port)
        except RuntimeError as exc:
            message = (
                "Could not connect to external tracker "
                "specified by $TVM_TRACKER_HOST and $TVM_TRACKER_PORT "
                f"({env_tracker_host}:{env_tracker_port})"
            )
            raise RuntimeError(message) from exc

        yield (env_tracker_host, env_tracker_port)

    else:
        # No tracker is provided to the tests, so we should start one
        # for the tests to use.
        tracker = tvm.rpc.tracker.Tracker("127.0.0.1", get_free_port())
        try:
            yield (tracker.host, tracker.port)
        finally:
            tracker.terminate()


@pytest.fixture(scope="session")
def tvm_tracker_host(_tracker_info) -> str:
    host, port = _tracker_info
    return host


@pytest.fixture(scope="session")
def tvm_tracker_port(_tracker_info) -> int:
    host, port = _tracker_info
    return port


@tvm.testing.fixture
def rpc_server_port() -> int:
    return get_free_port()


@tvm.testing.fixture
def adb_server_socket() -> str:
    return os.getenv(ADB_SERVER_SOCKET, default="tcp:5037")


@tvm.testing.fixture
def hexagon_launcher(request, android_serial_number, rpc_server_port, adb_server_socket):
    if android_serial_number is None:
        yield None
    else:
        # Requesting these fixtures sets up a local tracker, if one
        # hasn't been provided to us.  Delaying the evaluation of
        # these fixtures avoids starting a tracker unless necessary.
        tvm_tracker_host = request.getfixturevalue("tvm_tracker_host")
        tvm_tracker_port = request.getfixturevalue("tvm_tracker_port")

        rpc_info = {
            "rpc_tracker_host": tvm_tracker_host,
            "rpc_tracker_port": tvm_tracker_port,
            "rpc_server_port": rpc_server_port,
            "adb_server_socket": adb_server_socket,
        }
        launcher = HexagonLauncher(serial_number=android_serial_number, rpc_info=rpc_info)
        launcher.start_server()
        try:
            yield launcher
        finally:
            launcher.stop_server()


@tvm.testing.fixture
def hexagon_session(hexagon_launcher):
    if hexagon_launcher is None:
        yield None
    else:
        with hexagon_launcher.start_session() as session:
            yield session


# If the execution aborts while an RPC server is running, the python
# code that is supposed to shut it dowm will never execute. This will
# keep pytest from terminating (indefinitely), so add a cleanup
# fixture to terminate any still-running servers.
@pytest.fixture(scope="session", autouse=True)
def terminate_rpc_servers():
    # Since this is a fixture that runs regardless of whether the
    # execution happens on simulator or on target, make sure the
    # yield happens every time.
    serial = os.environ.get(ANDROID_SERIAL_NUMBER)
    yield []
    if serial == "simulator":
        os.system("ps ax | grep tvm_rpc_x86 | awk '{print $1}' | xargs kill")


aot_host_target = tvm.testing.parameter(
    "c",
    "llvm -keys=hexagon -link-params=0 -mattr=+hvxv68,+hvx-length128b,+hvx-qfloat,-hvx-ieee-fp -mcpu=hexagonv68 -mtriple=hexagon",
)


@tvm.testing.fixture
def aot_target(aot_host_target):
    if aot_host_target == "c":
        yield tvm.target.hexagon("v68")
    elif aot_host_target.startswith("llvm"):
        yield aot_host_target
    else:
        assert False, "Incorrect AoT host target: {aot_host_target}. Options are [c, llvm]."
