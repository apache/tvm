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

# pylint: disable=invalid-name,redefined-outer-name
""" Hexagon testing fixtures used to deduce testing argument
    values from testing parameters """

import os
import random
from typing import Optional, Union

import pytest

import tvm
import tvm.rpc.tracker
from tvm.contrib.hexagon.build import HexagonLauncher, HexagonLauncherRPC
from tvm.contrib.hexagon.session import Session

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
        func = args[0]
        for dec in reversed(decs):
            func = dec(func)
        return func
    return decs


requires_hexagon_toolchain = tvm.testing.requires_hexagon(support_required="compile-only")


def android_serial_number() -> Optional[str]:
    """Return the android serial number"""
    serial = os.getenv(ANDROID_SERIAL_NUMBER, default="")
    # Setting ANDROID_SERIAL_NUMBER to an empty string should be
    # equivalent to having it unset.
    if not serial.strip():
        return None

    # Split android serial numbers into a list
    serial = serial.split(",")
    return serial


# NOTE on server ports:
# These tests use different port numbers for the RPC server (7070 + ...).
# The reason is that an RPC session cannot be gracefully closed without
# triggering TIME_WAIT state on the server socket. This prevents another
# server to bind to the same port until the wait time elapses.

LISTEN_PORT_MIN = 6000  # Avoid hitting well-known Android debug ports
LISTEN_PORT_MAX = 9000  # Below the search range end (port_end=9199) of RPC server
PREVIOUS_PORT = None


def get_free_port() -> int:
    """Return the next port that is available to listen on"""
    global PREVIOUS_PORT
    if PREVIOUS_PORT is None:
        port = random.randint(LISTEN_PORT_MIN, LISTEN_PORT_MAX)
    else:
        port = PREVIOUS_PORT + 1

    while tvm.contrib.hexagon.build._is_port_in_use(port):
        port = port + 1 if port < LISTEN_PORT_MAX else LISTEN_PORT_MIN

    PREVIOUS_PORT = port
    return port


@pytest.fixture(scope="session")
def _tracker_info() -> Union[str, int]:
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
    host, _ = _tracker_info
    return host


@pytest.fixture(scope="session")
def tvm_tracker_port(_tracker_info) -> int:
    _, port = _tracker_info
    return port


@pytest.fixture(scope="session")
def rpc_server_port_for_session() -> int:
    return get_free_port()


@pytest.fixture()
def rpc_server_port() -> int:
    return get_free_port()


@pytest.fixture(scope="session")
def adb_server_socket() -> str:
    return os.getenv(ADB_SERVER_SOCKET, default="tcp:5037")


@pytest.fixture(scope="session")
def hexagon_server_process(
    request,
    rpc_server_port_for_session,
    adb_server_socket,
    skip_rpc,
    hexagon_debug,
    sysmon_profile,
    clear_logcat,
) -> HexagonLauncherRPC:
    """Initials and returns hexagon launcher if ANDROID_SERIAL_NUMBER is defined.
    This launcher is started only once per test session.
    """
    android_serial_num = android_serial_number()

    if android_serial_num is None:
        pytest.skip("ANDROID_SERIAL_NUMBER is not set.")
    if android_serial_num == ["simulator"]:
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
            "rpc_server_port": rpc_server_port_for_session,
            "adb_server_socket": adb_server_socket,
        }
        workerinput = getattr(request.config, "workerinput", None)
        if workerinput is None:  # single-process execution
            device_adr = read_device_list()[0]
        else:  # running in a subprocess here
            device_adr = workerinput["device_adr"]
        launcher = HexagonLauncher(
            serial_number=device_adr,
            rpc_info=rpc_info,
            hexagon_debug=hexagon_debug,
            sysmon_profile=sysmon_profile,
            clear_logcat=clear_logcat,
        )
        try:
            if not skip_rpc:
                launcher.start_server()
            yield {"launcher": launcher, "device_adr": device_adr}
        finally:
            if not skip_rpc:
                launcher.stop_server()


def read_device_list():
    return android_serial_number()


def pytest_configure(config):
    # read device list if we are on the master
    if not hasattr(config, "workerinput"):
        config.iplist = read_device_list()


def pytest_configure_node(node):
    # the master for each node fills slaveinput dictionary
    # which pytest-xdist will transfer to the subprocess
    if node.config.iplist is not None:
        node.workerinput["device_adr"] = node.config.iplist.pop()


@pytest.fixture
def hexagon_launcher(
    hexagon_server_process,
    rpc_server_port,
    tvm_tracker_host,
    tvm_tracker_port,
    adb_server_socket,
    hexagon_debug,
    sysmon_profile,
    clear_logcat,
) -> HexagonLauncherRPC:
    """Initials and returns hexagon launcher which reuses RPC info and Android serial number."""
    android_serial_num = android_serial_number()

    if android_serial_num != ["simulator"]:
        rpc_info = hexagon_server_process["launcher"]._rpc_info
    else:
        rpc_info = {
            "rpc_tracker_host": tvm_tracker_host,
            "rpc_tracker_port": tvm_tracker_port,
            "rpc_server_port": rpc_server_port,
            "adb_server_socket": adb_server_socket,
        }
    try:
        if android_serial_num == ["simulator"]:
            launcher = HexagonLauncher(serial_number=android_serial_num[0], rpc_info=rpc_info)
            launcher.start_server()
        else:
            launcher = HexagonLauncher(
                serial_number=hexagon_server_process["device_adr"],
                rpc_info=rpc_info,
                hexagon_debug=hexagon_debug,
                sysmon_profile=sysmon_profile,
                clear_logcat=clear_logcat,
            )
        yield launcher
    finally:
        if android_serial_num == ["simulator"]:
            launcher.stop_server()
        elif not hexagon_debug:
            launcher.cleanup_directory()


@pytest.fixture
def hexagon_session(hexagon_launcher: HexagonLauncherRPC) -> Session:
    if hexagon_launcher is None:
        yield None
    else:
        with hexagon_launcher.start_session() as session:
            yield session


# If the execution aborts while an RPC server is running, the python
# code that is supposed to shut it down will never execute. This will
# keep pytest from terminating (indefinitely), so add a cleanup
# fixture to terminate any still-running servers.
@pytest.fixture(scope="session", autouse=True)
def terminate_rpc_servers():
    # Since this is a fixture that runs regardless of whether the
    # execution happens on simulator or on target, make sure the
    # yield happens every time.
    serial = os.environ.get(ANDROID_SERIAL_NUMBER)
    yield []
    if serial == ["simulator"]:
        os.system("ps ax | grep tvm_rpc_x86 | awk '{print $1}' | xargs kill")


aot_host_target = tvm.testing.parameter(
    "c",
    "llvm -keys=hexagon "
    "-mattr=+hvxv68,+hvx-length128b,+hvx-qfloat,-hvx-ieee-fp "
    "-mcpu=hexagonv68 -mtriple=hexagon",
)


@tvm.testing.fixture
def aot_target(aot_host_target):
    if aot_host_target == "c":
        yield tvm.target.hexagon("v68")
    elif aot_host_target.startswith("llvm"):
        yield aot_host_target
    else:
        assert False, "Incorrect AoT host target: {aot_host_target}. Options are [c, llvm]."


@pytest.fixture(scope="session")
def skip_rpc(request) -> bool:
    return request.config.getoption("--skip-rpc")


@pytest.fixture(scope="session")
def hexagon_debug(request) -> bool:
    return request.config.getoption("--hexagon-debug")


@pytest.fixture(scope="session")
def sysmon_profile(request) -> bool:
    return request.config.getoption("--sysmon-profile")


@pytest.fixture(scope="session")
def clear_logcat(request) -> bool:
    return request.config.getoption("--clear-logcat")


def pytest_addoption(parser):
    """Add pytest options."""

    parser.addoption("--gtest_args", action="store", default="")

    parser.addoption(
        "--skip-rpc",
        action="store_true",
        default=False,
        help="If set true, the RPC server initialization on Android would be skipped",
    )
    parser.addoption(
        "--hexagon-debug",
        action="store_true",
        default=False,
        help="If set true, it will keep the hexagon test directories on the target. "
        + "Additionally logcat logs will be copied from device and cdsp errors printed out.",
    )
    parser.addoption(
        "--sysmon-profile",
        action="store_true",
        default=False,
        help="If set true, it will run sysmon profiler during the tests.",
    )
    parser.addoption(
        "--clear-logcat",
        action="store_true",
        default=False,
        help="If set true, it will clear logcat before execution.",
    )


def pytest_generate_tests(metafunc):
    option_value = metafunc.config.option.gtest_args
    if "gtest_args" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("gtest_args", [option_value])
