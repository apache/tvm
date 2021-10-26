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
import os
import tarfile
from typing import Union
from tvm.micro.project_api.server import IoTimeoutError


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
    timeout = 5
    while True:
        try:
            aot_transport_find_message(transport, "wakeup", timeout_sec=timeout)
            break
        except IoTimeoutError:
            transport.write(b"init%", timeout_sec=timeout)


def aot_transport_find_message(transport, expression: str, timeout_sec: int):
    """Read transport message until it finds the expression."""
    while True:
        data = _read_line(transport, timeout_sec)
        logging.debug("new line: %s", data)
        if expression in data:
            return data


def _read_line(transport, timeout_sec: int):
    data = ""
    new_line = False
    while True:
        if new_line:
            break
        new_data = transport.read(1, timeout_sec=timeout_sec)
        logging.debug("read data: %s", new_data)
        for item in new_data:
            new_c = chr(item)
            data = data + new_c
            if new_c == "\n":
                new_line = True
                break
    return data


def mlf_extract_workspace_size_bytes(
    mlf_tar_path: Union[pathlib.Path, str], extract_path: Union[pathlib.Path, str]
):
    """Extract an MLF archive file and read workspace size from metadata file."""

    tar_file = str(mlf_tar_path)
    base_path = str(extract_path)
    t = tarfile.open(tar_file)
    t.extractall(base_path)

    with open(os.path.join(base_path, "metadata.json")) as json_f:
        metadata = json.load(json_f)
        return metadata["memory"]["functions"]["main"][0]["workspace_size_bytes"]
