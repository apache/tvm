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

import pathlib
import json
import logging
import tarfile
import time
from typing import Union

from tvm.micro.project_api.server import IoTimeoutError

# Timeout in seconds for AOT transport.
TIMEOUT_SEC = 10


def check_tune_log(log_path: Union[pathlib.Path, str]):
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


def mlf_extract_workspace_size_bytes(mlf_tar_path: Union[pathlib.Path, str]) -> int:
    """Extract an MLF archive file and read workspace size from metadata file."""

    with tarfile.open(mlf_tar_path, "r:*") as tar_file:
        tar_members = [ti.name for ti in tar_file.getmembers()]
        assert "./metadata.json" in tar_members
        with tar_file.extractfile("./metadata.json") as f:
            metadata = json.load(f)
            return metadata["memory"]["functions"]["main"][0]["workspace_size_bytes"]
