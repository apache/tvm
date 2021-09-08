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
import pathlib

import pytest
import tvm.target.target
from tvm.micro import project
from tvm import micro, relay

TEMPLATE_PROJECT_DIR = (
    pathlib.Path(__file__).parent
    / ".."
    / ".."
    / ".."
    / "apps"
    / "microtvm"
    / "arduino"
    / "template_project"
).resolve()


def arduino_boards() -> dict:
    """Returns a dict mapping board to target model"""
    template = project.TemplateProject.from_directory(TEMPLATE_PROJECT_DIR)
    project_options = template.info()["project_options"]
    for option in project_options:
        if option["name"] == "arduino_board":
            boards = option["choices"]
        if option["name"] == "arduino_model":
            models = option["choices"]

    arduino_boards = {boards[i]: models[i] for i in range(len(boards))}
    return arduino_boards


ARDUINO_BOARDS = arduino_boards()


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


def make_workspace_dir(test_name, board):
    filepath = pathlib.Path(__file__)
    board_workspace = (
        filepath.parent
        / f"workspace_{test_name}_{board}"
        / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    )

    number = 0
    while board_workspace.exists():
        number += 1
        board_workspace = pathlib.Path(str(board_workspace) + f"-{number}")
    board_workspace.parent.mkdir(exist_ok=True, parents=True)
    t = tvm.contrib.utils.tempdir(board_workspace)
    # time.sleep(200)
    return t


def make_kws_project(board, arduino_cli_cmd, tvm_debug, workspace_dir):
    this_dir = pathlib.Path(__file__).parent
    model = ARDUINO_BOARDS[board]
    build_config = {"debug": tvm_debug}

    with open(this_dir.parent / "testdata" / "kws" / "yes_no.tflite", "rb") as f:
        tflite_model_buf = f.read()

    # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
    try:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)

    mod, params = relay.frontend.from_tflite(tflite_model)
    target = tvm.target.target.micro(
        model, options=["--link-params=1", "--unpacked-api=1", "--executor=aot"]
    )

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = relay.build(mod, target, params=params)

    return tvm.micro.generate_project(
        str(TEMPLATE_PROJECT_DIR),
        mod,
        workspace_dir / "project",
        {
            "arduino_board": board,
            "arduino_cli_cmd": arduino_cli_cmd,
            "project_type": "example_project",
            "verbose": bool(build_config.get("debug")),
        },
    )
