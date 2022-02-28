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

""" Hexagon testing fixtures used to deduce testing argument
    values from testing parameters """

import os
import pytest

import tvm
from tvm import rpc

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
        f = args[0]
        for d in reversed(decs):
            f = d(f)
        return f
    return decs


def requires_hexagon_toolchain(*args):
    _requires_hexagon_toolchain = [
        pytest.mark.skipif(
            os.environ.get(HEXAGON_TOOLCHAIN) == None,
            reason=f"Missing environment variable {HEXAGON_TOOLCHAIN}.",
        ),
    ]

    return _compose(args, _requires_hexagon_toolchain)


@tvm.testing.fixture
def android_serial_number() -> str:
    return os.getenv(ANDROID_SERIAL_NUMBER, default=None)


@tvm.testing.fixture
def tvm_tracker_host() -> str:
    return os.getenv(TVM_TRACKER_HOST, default=None)


@tvm.testing.fixture
def tvm_tracker_port() -> int:
    port = os.getenv(TVM_TRACKER_PORT, default=None)
    port = int(port) if port else None
    return port


@tvm.testing.fixture
def adb_server_socket() -> str:
    return os.getenv(ADB_SERVER_SOCKET, default="tcp:5037")
