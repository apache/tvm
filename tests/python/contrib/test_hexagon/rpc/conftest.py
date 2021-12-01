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
import os

import tvm.testing


def pytest_addoption(parser):
    parser.addoption(
        "--serial-number",
        required=True,
        help=("Android device serial number list from 'adb' command."),
    )


@pytest.fixture
def android_serial_number(request):
    return request.config.getoption("--serial-number")


@tvm.testing.fixture
def tvm_tracker_host():
    return os.environ["TVM_TRACKER_HOST"]


@tvm.testing.fixture
def tvm_tracker_port():
    return int(os.environ["TVM_TRACKER_PORT"])


def _compose(args, decs):
    """Helper to apply multiple markers"""
    if len(args) > 0:
        f = args[0]
        for d in reversed(decs):
            f = d(f)
        return f
    return decs


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


def requires_ndk_cc(*args):
    _requires_ndk_cc = [
        pytest.mark.skipif(
            os.environ.get("TVM_NDK_CC") == None,
            reason="TVM_NDK_CC environment variable is required to run this test.",
        ),
    ]

    return _compose(args, _requires_ndk_cc)
