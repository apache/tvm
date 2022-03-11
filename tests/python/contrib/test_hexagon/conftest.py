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


@tvm.testing.fixture
def tvm_tracker_host() -> str:
    return os.getenv(TVM_TRACKER_HOST, default=None)


@tvm.testing.fixture
def tvm_tracker_port() -> int:
    port = os.getenv(TVM_TRACKER_PORT, default=None)
    port = int(port) if port else None
    return port


# NOTE on server ports:
# These tests use different port numbers for the RPC server (7070 + ...).
# The reason is that an RPC session cannot be gracefully closed without
# triggering TIME_WAIT state on the server socket. This prevents another
# server to bind to the same port until the wait time elapses.

# rpc_port_min = 1024  # Lowest unprivileged port
rpc_port_min = 2000  # Well above the privileged ports (1024 or lower)
rpc_port_max = 9000  # Below the search range end (port_end=9199) of RPC server
previous_port = [None]


@tvm.testing.fixture
def rpc_server_port() -> int:
    print(rpc_port_min)

    # https://stackoverflow.com/a/52872579/2689797
    def is_port_in_use(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    if previous_port[0] is None:
        port = random.randint(rpc_port_min, rpc_port_max)
    else:
        port = previous_port[0] + 1

    while is_port_in_use(port):
        port = port + 1 if port < rpc_port_max else rpc_port_min

    previous_port[0] = port
    return port


@tvm.testing.fixture
def adb_server_socket() -> str:
    return os.getenv(ADB_SERVER_SOCKET, default="tcp:5037")


@tvm.testing.fixture
def tvm_tracker(tvm_tracker_port):
    tracker = tvm.rpc.tracker.Tracker("127.0.0.1", tvm_tracker_port)
    try:
        yield tracker
    finally:
        tracker.terminate()


@tvm.testing.fixture
def rpc_info(tvm_tracker, rpc_server_port, adb_server_socket):
    return {
        "rpc_tracker_host": tvm_tracker.host,
        "rpc_tracker_port": tvm_tracker.port,
        "rpc_server_port": rpc_server_port,
        "adb_server_socket": adb_server_socket,
    }


@tvm.testing.fixture
def hexagon_launcher(android_serial_number, tvm_tracker, rpc_server_port, adb_server_socket):
    if android_serial_number is None:
        yield None
    else:
        rpc_info = {
            "rpc_tracker_host": tvm_tracker.host,
            "rpc_tracker_port": tvm_tracker.port,
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
