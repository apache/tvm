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
""" microTVM testing fixtures used to deduce testing argument
    values from testing parameters """

import pathlib
import json
import os
import datetime
import pytest

import tvm
from tvm.contrib.utils import tempdir


def zephyr_boards() -> dict:
    """Returns a dict mapping board to target model"""
    with open(
        pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr")) / "boards.json"
    ) as f:
        board_properties = json.load(f)

    boards_model = {board: info["model"] for board, info in board_properties.items()}
    return boards_model


def get_boards(platform: str) -> dict:
    """Returns a dict mapping board to target model"""
    with open(
        pathlib.Path(tvm.micro.get_microtvm_template_projects(platform)) / "boards.json"
    ) as f:
        board_properties = json.load(f)

    boards_model = {board: info["model"] for board, info in board_properties.items()}
    return boards_model


def pytest_addoption(parser):
    """Adds more pytest arguments"""
    parser.addoption(
        "--board",
        required=True,
        choices=list(get_boards("zephyr").keys()) + list(get_boards("arduino").keys()),
        help="microTVM boards for tests",
    )
    parser.addoption(
        "--test-build-only",
        action="store_true",
        help="Only run tests that don't require physical hardware.",
    )
    parser.addoption(
        "--tvm-debug",
        action="store_true",
        default=False,
        help="If set true, it will keep the project directory for debugging.",
    )


@pytest.fixture
def board(request):
    return request.config.getoption("--board")


@pytest.fixture
def tvm_debug(request):
    return request.config.getoption("--tvm-debug")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--test-build-only"):
        skip_hardware_tests = pytest.mark.skip(reason="--test-build-only was passed")
        for item in items:
            if "requires_hardware" in item.keywords:
                item.add_marker(skip_hardware_tests)


@pytest.fixture
def workspace_dir(board, tvm_debug):
    """Creates workspace directory for each test."""
    parent_dir = pathlib.Path(os.path.dirname(__file__))
    filename = os.path.splitext(os.path.basename(__file__))[0]
    board_workspace = (
        parent_dir
        / f"workspace_{filename}_{board}"
        / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    )
    board_workspace_base = str(board_workspace)
    number = 1
    while board_workspace.exists():
        board_workspace = pathlib.Path(board_workspace_base + f"-{number}")
        number += 1

    if not os.path.exists(board_workspace.parent):
        os.makedirs(board_workspace.parent)

    keep_for_debug = tvm_debug if tvm_debug else None
    test_temp_dir = tempdir(custom_path=board_workspace, keep_for_debug=keep_for_debug)
    return test_temp_dir


@pytest.fixture(autouse=True)
def skip_by_board(request, board):
    """Skip test if board is in the list."""
    if request.node.get_closest_marker("skip_boards"):
        if board in request.node.get_closest_marker("skip_boards").args[0]:
            pytest.skip("skipped on this board: {}".format(board))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_boards(board): skip test for the given board",
    )
