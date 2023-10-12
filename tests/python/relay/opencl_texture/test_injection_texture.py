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

import re
import pytest
import tvm
import numpy as np
from tvm import relay
from utils.adreno_utils import build_run_compare, build_run_compare_vm


executor_type = tvm.testing.parameter("ge", "vm")
dtype = tvm.testing.parameter("float32")


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_layout_transform_to_block_nchw4c(remote, target, executor_type, dtype):
    """Verification of the case NCHW->NCHW4c"""
    input_shape = (1, 32, 720, 1280)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    lt = relay.layout_transform(A, "NCHW", "NCHW4c")
    mod = relay.Function([A], lt)

    if executor_type == "ge":
        build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_layout_transform_to_block_nchw(remote, target, executor_type, dtype):
    """Verification of the case NCHW4c->NCHW"""
    input_shape = (1, 36, 1, 1, 4)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    lt = relay.layout_transform(A, "NCHW4c", "NCHW")
    mod = relay.Function([A], lt)

    if executor_type == "ge":
        build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_layout_transform_to_block_nhwc4c(remote, target, executor_type, dtype):
    """Verification of the case NHWC->NHWC4c"""
    input_shape = (1, 1, 1, 144)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    lt = relay.layout_transform(A, "NHWC", "NHWC4c")
    mod = relay.Function([A], lt)

    if executor_type == "ge":
        build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@pytest.mark.skipif(
    tvm.testing.utils.IS_IN_CI, reason="Skip because GPU in CI doesn't support FP16"
)
@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_layout_transform_to_block_nhwc(remote, target, executor_type, dtype):
    """Verification of the case NHWC4c->NHWC"""
    input_shape = (1, 80, 80, 36, 4)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    mean = relay.mean(A, axis=[1, 2], keepdims=True)
    cast = relay.cast(mean, "float16")
    lt = relay.layout_transform(cast, "NHWC4c", "NHWC")
    mod = relay.Function([A], lt)

    if executor_type == "ge":
        build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


if __name__ == "__main__":
    test_layout_transform_to_block_nhwc(None, "opencl -device=adreno", "float16")
