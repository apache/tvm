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
import hashlib
import pytest
import os
from collections import OrderedDict

pytest_plugins = ["tvm.testing.plugin"]


# These are long running tests (manually curated and extracted from CI logs)
# that should be allocated to test shards in a round-robin fashion. These are
# taken from the 20 (arbitrary number) of tests as from
# https://ci.tlcpack.ai/job/tvm/job/main/2907/testReport
_slowest_tests = [
    "tests/python/frontend/tensorflow/test_forward.py::test_forward_broadcast_args",
    "tests/python/frontend/tensorflow/test_forward.py::test_forward_broadcast_to",
    "tests/python/topi/python/test_topi_conv2d_int8.py::test_conv2d_nchw[int8]",
    "tests/python/topi/python/test_topi_conv2d_int8.py::test_conv2d_nchw[uint8]",
    "tests/python/topi/python/test_topi_upsampling.py::test_upsampling3d",
    "tests/python/topi/python/test_topi_upsampling.py::test_upsampling3d",
    "tests/python/topi/python/test_topi_conv2d_int8.py::test_conv2d_nchw[int8]",
    "tests/python/frontend/tflite/test_forward.py::test_all_elemwise",
    "tests/python/frontend/pytorch/test_object_detection.py::test_detection_models",
    "tests/python/topi/python/test_topi_conv2d_int8.py::test_conv2d_nchw[uint8]",
    "tests/python/topi/python/test_topi_conv2d_NCHWc.py::test_conv2d_NCHWc",
    "tests/python/topi/python/test_topi_conv2d_hwnc_tensorcore.py::test_conv2d_hwnc_tensorcore",
    "tests/python/contrib/test_tensorrt.py::test_binary[compile]",
    "tests/python/frontend/pytorch/test_forward.py::test_segmentation_models",
    "tests/python/topi/python/test_topi_conv2d_NCHWc.py::test_conv2d_NCHWc",
    "tests/python/relay/test_py_converter.py::test_global_recursion",
    "tests/python/frontend/tensorflow/test_forward.py::test_forward_ptb",
    "tests/python/relay/test_op_level6.py::test_topk",
    "tests/python/topi/python/test_topi_conv2d_winograd.py::test_conv2d_nchw",
    "tests/python/relay/test_py_converter.py::test_global_recursion",
]
HARDCODED_ALLOCATIONS = {}
for idx, test in enumerate(_slowest_tests):
    HARDCODED_ALLOCATIONS[test] = idx


def should_run(nodeid: str, num_shards: int, shard_index: int) -> bool:
    """
    Return true if this test should run on this shard
    """
    if nodeid in HARDCODED_ALLOCATIONS:
        hash = HARDCODED_ALLOCATIONS[nodeid]
    else:
        hash = hashlib.md5(nodeid.encode())
        hash = int(hash.hexdigest(), 16)

    return hash % num_shards == shard_index


def pytest_collection_modifyitems(config, items):
    if not all(k in os.environ for k in ["CI", "TVM_NUM_SHARDS", "TVM_SHARD_INDEX"]):
        # Only apportion tests if in CI and in a job that is set up for it
        return

    num_shards = int(os.environ["TVM_NUM_SHARDS"])
    shard_index = int(os.environ["TVM_SHARD_INDEX"])

    print(f"Marking tests for shard {shard_index} of {num_shards}")
    for item in items:
        if not should_run(item.nodeid, num_shards=num_shards, shard_index=shard_index):
            item.add_marker(pytest.mark.skip())
