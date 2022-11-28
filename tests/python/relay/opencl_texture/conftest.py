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
import tvm
from tvm import rpc
import pytest


@pytest.fixture(scope="session")
def remote():
    if (
        "TVM_TRACKER_HOST" in os.environ
        and "TVM_TRACKER_PORT" in os.environ
        and "RPC_DEVICE_KEY" in os.environ
    ):

        rpc_tracker_host = os.environ["TVM_TRACKER_HOST"]
        rpc_tracker_port = int(os.environ["TVM_TRACKER_PORT"])
        rpc_device_key = os.environ["RPC_DEVICE_KEY"]
        tracker = rpc.connect_tracker(rpc_tracker_host, rpc_tracker_port)
        remote = tracker.request(rpc_device_key, priority=0, session_timeout=600)
        return remote
    else:
        return None
