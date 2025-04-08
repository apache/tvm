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

import os
import sys
import tvm
import pytest
from tvm import rpc as _rpc


@pytest.fixture(scope="session")
def rpc():
    rpc_target = os.getenv("RPC_TARGET", None)
    if rpc_target:
        connection_type = "tracker"
        host = os.getenv("TVM_TRACKER_HOST", "localhost")
        port = int(os.getenv("TVM_TRACKER_PORT", 9090))
        target = "opencl"
        target_host = "llvm -mtriple=aarch64-linux-gnu"
        device_key = os.getenv("RPC_DEVICE_KEY", "android")
        cross_compile = os.getenv("TVM_NDK_CC", "aarch64-linux-android-g++")
        tracker = _rpc.connect_tracker(host, port)
        return tracker.request(device_key, priority=1, session_timeout=1000)
    else:
        return None
