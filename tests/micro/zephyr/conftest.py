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
import datetime
import os
import pathlib
import sys

import pytest

import tvm.contrib.utils
import tvm.target.target

TEMPLATE_PROJECT_DIR = (
    pathlib.Path(__file__).parent
    / ".."
    / ".."
    / ".."
    / "apps"
    / "microtvm"
    / "zephyr"
    / "template_project"
).resolve()

def zephyr_micro_devices() -> dict:
    sys.path.insert(0, str(TEMPLATE_PROJECT_DIR))
    try:
        import microtvm_api_server
    finally:
        sys.path.pop(0)

    return microtvm_api_server.MICRO_DEVICES

def pytest_addoption(parser):
    parser.addoption(
        "--microtvm-device",
        default="qemu_x86",
        choices=zephyr_micro_devices().keys(),
        help=(
            "Specify a microtvm device (i.e. as passed to tvm.target.micro()) for microTVM tests."
        ),
    )
    parser.addoption(
        "--west-cmd", default="west", help="Path to `west` command for flashing device."
    )
    parser.addoption(
        "--tvm-debug",
        action="store_true",
        default=False,
        help="If set true, enable a debug session while the test is running. Before running the test, in a separate shell, you should run: <python -m tvm.exec.microtvm_debug_shell>",
    )


def pytest_generate_tests(metafunc):
    if "device" in metafunc.fixturenames:
        metafunc.parametrize("device", [metafunc.config.getoption("microtvm_device")])


@pytest.fixture
def west_cmd(request):
    return request.config.getoption("--west-cmd")


@pytest.fixture
def tvm_debug(request):
    return request.config.getoption("--tvm-debug")


@pytest.fixture
def temp_dir(device):
    _, zephyr_board = zephyr_micro_devices()[device]
    parent_dir = pathlib.Path(os.path.dirname(__file__))
    filename = os.path.splitext(os.path.basename(__file__))[0]
    board_workspace = (
        parent_dir
        / f"workspace_{filename}_{zephyr_board}"
        / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    )
    board_workspace_base = str(board_workspace)
    number = 1
    while board_workspace.exists():
        board_workspace = pathlib.Path(board_workspace_base + f"-{number}")
        number += 1

    if not os.path.exists(board_workspace.parent):
        os.makedirs(board_workspace.parent)

    return tvm.contrib.utils.tempdir(board_workspace)
