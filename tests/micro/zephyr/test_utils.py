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

BOARDS = TEMPLATE_PROJECT_DIR / "boards.json"


def zephyr_boards() -> dict:
    """Returns a dict mapping board to target model"""
    with open(BOARDS) as f:
        board_properties = json.load(f)

    boards_model = {board: info["model"] for board, info in board_properties.items()}
    return boards_model


ZEPHYR_BOARDS = zephyr_boards()


def qemu_boards(board: str):
    """Returns True if board is QEMU."""
    with open(BOARDS) as f:
        board_properties = json.load(f)

    qemu_boards = [name for name, board in board_properties.items() if board["is_qemu"]]
    return board in qemu_boards


def has_fpu(board: str):
    """Returns True if board has FPU."""
    with open(BOARDS) as f:
        board_properties = json.load(f)

    fpu_boards = [name for name, board in board_properties.items() if board["fpu"]]
    return board in fpu_boards
