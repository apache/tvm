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
ANDROID_TRACKER_KEY = "ANDROID_TRACKER_KEY"
ANDROID_REMOTE_DIR = "ANDROID_REMOTE_DIR"


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
            os.environ.get("HEXAGON_TOOLCHAIN") == None,
            reason="HEXAGON_TOOLCHAIN environment variable is required to run this test.",
        ),
    ]

    return _compose(args, _requires_hexagon_toolchain)


@tvm.testing.fixture
def android_tracker_key():
    return os.environ["ANDROID_TRACKER_KEY"]


@tvm.testing.fixture
def tvm_tracker_host():
    return os.environ["TVM_TRACKER_HOST"]


@tvm.testing.fixture
def tvm_tracker_port():
    return int(os.environ["TVM_TRACKER_PORT"])


@tvm.testing.fixture
def remote_path():
    dso_binary = "test_binary.so"
    return os.path.join(os.environ["ANDROID_REMOTE_DIR"], dso_binary)


@tvm.testing.fixture
def rpc_sess(android_tracker_key, tvm_tracker_host, tvm_tracker_port):
    from tvm import rpc

    tracker = rpc.connect_tracker(tvm_tracker_host, tvm_tracker_port)
    remote = tracker.request(android_tracker_key, priority=0, session_timeout=600)
    return remote


def requires_rpc_tracker_and_android_key(*args):
    """Mark a test as requiring an RPC tracker to exist in
    the host environment to run."""
    _requires_rpc_tracker = [
        *tvm.testing.requires_rpc(),
        pytest.mark.skipif(
            os.environ.get(TVM_TRACKER_HOST) == None,
            reason="Missing environment variable, TVM_TRACKER_HOST",
        ),
        pytest.mark.skipif(
            os.environ.get(TVM_TRACKER_PORT) == None,
            reason="Missing environment variable, TVM_TRACKER_PORT",
        ),
        pytest.mark.skipif(
            os.environ.get(ANDROID_TRACKER_KEY) == None,
            reason="Missing environment variable, ANDROID_TRACKER_KEY",
        ),
        pytest.mark.skipif(
            os.environ.get(ANDROID_REMOTE_DIR) == None,
            reason="Missing environment variable, ANDROID_REMOTE_DIR",
        ),
    ]

    return _compose(args, _requires_rpc_tracker)


def requires_rpc_tracker(*args):
    """Mark a test as requiring an RPC tracker to exist in
    the host environment to run."""
    _requires_rpc_tracker = [
        *tvm.testing.requires_rpc(),
        pytest.mark.skipif(
            os.environ.get("TVM_TRACKER_HOST") == None,
            reason="Missing environment variable, TVM_TRACKER_HOST",
        ),
        pytest.mark.skipif(
            os.environ.get("TVM_TRACKER_PORT") == None,
            reason="Missing environment variable, TVM_TRACKER_PORT",
        ),
    ]

    return _compose(args, _requires_rpc_tracker)
