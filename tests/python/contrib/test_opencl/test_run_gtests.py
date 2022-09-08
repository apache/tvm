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
import pytest
import numpy as np

import tvm
from tvm import rpc


# use pytest -sv to observe gtest output
# use --gtest_args to pass arguments to gtest
# for example to run all "foo" tests twice and observe gtest output run
# pytest -sv <this file> --gtests_args="--gtest_filter=*foo* --gtest_repeat=2"
@tvm.testing.requires_opencl
@pytest.mark.skipif(tvm.testing.utils.IS_IN_CI, reason="failed due to nvidia libOpencl in the CI")
def test_run_gtests(gtest_args):
    if (
        "TVM_TRACKER_HOST" in os.environ
        and "TVM_TRACKER_PORT" in os.environ
        and "TVM_TRACKER_KEY" in os.environ
    ):
        rpc_tracker_host = os.environ["TVM_TRACKER_HOST"]
        rpc_tracker_port = os.environ["TVM_TRACKER_PORT"]
        rpc_tracker_port = int(rpc_tracker_port)
        rpc_key = os.environ["TVM_TRACKER_KEY"]
        tracker = rpc.connect_tracker(rpc_tracker_host, rpc_tracker_port)
        rpc_connection = tracker.request(rpc_key, priority=0, session_timeout=600)
    else:
        rpc_connection = rpc.LocalSession()

    try:
        func = rpc_connection.get_function("opencl.run_gtests")
    except:
        print(
            "This test requires TVM Runtime to be built with a OpenCL gtest version using OpenCL API cmake flag -DUSE_OPENCL_GTEST=/path/to/opencl/googletest/gtest"
        )
        raise

    gtest_error_code = func(gtest_args)
    np.testing.assert_equal(gtest_error_code, 0)
