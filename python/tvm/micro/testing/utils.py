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

"""Defines the test methods used with microTVM."""

from functools import lru_cache
import json
import logging
from pathlib import Path
import tarfile
import time
from typing import Union

import tvm
from tvm import relay
from tvm.micro.project_api.server import IoTimeoutError

# Timeout in seconds for AOT transport.
TIMEOUT_SEC = 10


@lru_cache(maxsize=None)
def get_supported_platforms():
    return ["arduino", "zephyr"]


@lru_cache(maxsize=None)
def get_supported_boards(platform: str):
    template = Path(tvm.micro.get_microtvm_template_projects(platform))
    with open(template / "boards.json") as f:
        return json.load(f)


def get_target(platform: str, board: str) -> tvm.target.Target:
    """Intentionally simple function for making Targets for microcontrollers.
    If you need more complex arguments, one should call target.micro directly. Note
    that almost all, but not all, supported microcontrollers are Arm-based."""
    model = get_supported_boards(platform)[board]["model"]
    return tvm.target.target.micro(model, options=["-device=arm_cpu"])


def check_tune_log(log_path: Union[Path, str]):
    """Read the tuning log and check each result."""
    with open(log_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if len(line) > 0:
            tune_result = json.loads(line)
            assert tune_result["result"][0][0] < 1000000000.0


def aot_transport_init_wait(transport):
    """Send init message to microTVM device until it receives wakeup sequence."""
    while True:
        try:
            aot_transport_find_message(transport, "wakeup", timeout_sec=TIMEOUT_SEC)
            break
        except IoTimeoutError:
            transport.write(b"init%", timeout_sec=TIMEOUT_SEC)


def aot_transport_find_message(transport, expression: str, timeout_sec: int) -> str:
    """Read transport message until it finds the expression."""
    timeout = timeout_sec
    start_time = time.monotonic()
    while True:
        data = _read_line(transport, timeout)
        logging.debug("new line: %s", data)
        if expression in data:
            return data
        timeout = max(0, timeout_sec - (time.monotonic() - start_time))


def _read_line(transport, timeout_sec: int) -> str:
    data = bytearray()
    while True:
        new_data = transport.read(1, timeout_sec=timeout_sec)
        logging.debug("read data: %s", new_data)
        for item in new_data:
            data.append(item)
            if str(chr(item)) == "\n":
                return data.decode(encoding="utf-8")


def mlf_extract_workspace_size_bytes(mlf_tar_path: Union[Path, str]) -> int:
    """Extract an MLF archive file and read workspace size from metadata file."""

    workspace_size = 0
    with tarfile.open(mlf_tar_path, "r:*") as tar_file:
        tar_members = [ti.name for ti in tar_file.getmembers()]
        assert "./metadata.json" in tar_members
        with tar_file.extractfile("./metadata.json") as f:
            metadata = json.load(f)
            for mod_name in metadata["modules"].keys():
                workspace_size += metadata["modules"][mod_name]["memory"]["functions"]["main"][0][
                    "workspace_size_bytes"
                ]
            return workspace_size


def get_conv2d_relay_module():
    """Generate a conv2d Relay module for testing."""
    data_shape = (1, 3, 64, 64)
    weight_shape = (8, 3, 5, 5)
    data = relay.var("data", relay.TensorType(data_shape, "int8"))
    weight = relay.var("weight", relay.TensorType(weight_shape, "int8"))
    y = relay.nn.conv2d(
        data,
        weight,
        padding=(2, 2),
        channels=8,
        kernel_size=(5, 5),
        data_layout="NCHW",
        kernel_layout="OIHW",
        out_dtype="int32",
    )
    f = relay.Function([data, weight], y)
    mod = tvm.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)
    return mod
