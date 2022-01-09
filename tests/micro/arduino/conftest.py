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

import pytest

from test_utils import ARDUINO_BOARDS


def pytest_addoption(parser):
    parser.addoption(
        "--arduino-board",
        nargs="+",
        required=True,
        choices=ARDUINO_BOARDS.keys(),
        help="Arduino board for tests.",
    )
    parser.addoption(
        "--arduino-cli-cmd",
        default="arduino-cli",
        help="Path to `arduino-cli` command for flashing device.",
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
        help="If given, enable a debug session while the test is running. Before running the test, in a separate shell, you should run: <python -m tvm.exec.microtvm_debug_shell>",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_hardware: mark test to run only when an Arduino board is connected"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--test-build-only"):
        skip_hardware_tests = pytest.mark.skip(reason="--test-build-only was passed")
        for item in items:
            if "requires_hardware" in item.keywords:
                item.add_marker(skip_hardware_tests)


# We might do project generation differently for different boards in the future
# (to take advantage of multiple cores / external memory / etc.), so all tests
# are parameterized by board
def pytest_generate_tests(metafunc):
    board = metafunc.config.getoption("arduino_board")
    metafunc.parametrize("board", board, scope="session")


@pytest.fixture(scope="session")
def arduino_cli_cmd(request):
    return request.config.getoption("--arduino-cli-cmd")


@pytest.fixture(scope="session")
def tvm_debug(request):
    return request.config.getoption("--tvm-debug")
