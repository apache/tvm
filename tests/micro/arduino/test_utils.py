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

import json
import pathlib
import requests
import datetime

import tvm.micro
import tvm.target.target
from tvm.micro import project
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from tvm.testing.utils import fetch_model_from_url

TEMPLATE_PROJECT_DIR = pathlib.Path(tvm.micro.get_microtvm_template_projects("arduino"))

BOARDS = TEMPLATE_PROJECT_DIR / "boards.json"


def arduino_boards() -> dict:
    """Returns a dict mapping board to target model"""
    with open(BOARDS) as f:
        board_properties = json.load(f)

    boards_model = {board: info["model"] for board, info in board_properties.items()}
    return boards_model


ARDUINO_BOARDS = arduino_boards()


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
    return t


def make_kws_project(board, microtvm_debug, workspace_dir, serial_number: str):
    this_dir = pathlib.Path(__file__).parent
    model = ARDUINO_BOARDS[board]
    build_config = {"debug": microtvm_debug}

    mod, params = fetch_model_from_url(
        url="https://github.com/tensorflow/tflite-micro/raw/a56087ffa2703b4d5632f024a8a4c899815c31bb/tensorflow/lite/micro/examples/micro_speech/micro_speech.tflite",
        model_format="tflite",
        sha256="09e5e2a9dfb2d8ed78802bf18ce297bff54281a66ca18e0c23d69ca14f822a83",
    )

    target = tvm.target.target.micro(model)
    runtime = Runtime("crt")
    executor = Executor("aot", {"unpacked-api": True})

    with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
        mod = relay.build(mod, target, runtime=runtime, executor=executor, params=params)

    return tvm.micro.generate_project(
        str(TEMPLATE_PROJECT_DIR),
        mod,
        workspace_dir / "project",
        {
            "board": board,
            "project_type": "example_project",
            "verbose": bool(build_config.get("debug")),
            "serial_number": serial_number,
        },
    )
