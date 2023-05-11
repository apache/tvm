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
import tvm
import numpy as np
from tvm import relay
from tvm.relay import testing
from tvm.contrib import utils
from utils.adreno_utils import gpu_preprocess, build_run_compare


dtype = tvm.testing.parameter("float32")


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_mean(remote, target, dtype):
    # NCHW
    input_shape = (1, 3, 720, 1280)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    mean = relay.mean(A, axis=1, keepdims=True)
    mod = relay.Function([A], mean)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_argmax(remote, target, dtype):
    # NCHW
    input_shape = (1, 3, 720, 1280)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    argmax = relay.op.argmax(A, axis=[1])
    mod = relay.Function([A], argmax)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_reduction_max(remote, target, dtype):
    # NCHW
    input_shape = (1, 3, 720, 1280)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    argmax = relay.op.max(A, axis=[1])
    mod = relay.Function([A], argmax)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_mean_nd4(remote, target, dtype):
    # NCHW
    input_shape = (1, 3, 729, 729)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    mean = relay.mean(A, axis=1, keepdims=True)
    mod = relay.Function([A], mean)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_argmax_nd4(remote, target, dtype):
    # NCHW
    input_shape = (1, 3, 729, 729)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    argmax = relay.op.argmax(A, axis=[1])
    mod = relay.Function([A], argmax)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_reduction_max_nd4(remote, target, dtype):
    # NCHW
    input_shape = (1, 3, 729, 729)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    argmax = relay.op.max(A, axis=[1])
    mod = relay.Function([A], argmax)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_mean_b4(remote, target, dtype):
    # NCHW
    input_shape = (1, 3, 720, 320, 4)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    mean = relay.mean(A, axis=1, keepdims=True)
    mod = relay.Function([A], mean)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_argmax_b4(remote, target, dtype):
    # NCHW
    input_shape = (1, 3, 720, 320, 4)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    argmax = relay.op.argmax(A, axis=[1])
    mod = relay.Function([A], argmax)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_reduction_max_b4(remote, target, dtype):
    # NCHW
    input_shape = (1, 3, 720, 320, 4)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    argmax = relay.op.max(A, axis=[1])
    mod = relay.Function([A], argmax)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_mean_global_pooling(remote, target, dtype):
    """
    Use case of blocked NCHW4c global pooling with big spatial valies
    """
    input_shape = (1, 160, 160, 32)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    mean = relay.mean(A, axis=[1, 2], keepdims=True)
    mod = relay.Function([A], mean)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_mean_global_pooling_block4(remote, target, dtype):
    """
    Use case of blocked NCHW4c global pooling with big spatial valies
    """
    input_shape = (1, 160, 160, 8, 4)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    mean = relay.mean(A, axis=[1, 2], keepdims=True)
    mod = relay.Function([A], mean)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_max_global_pooling_block4(remote, target, dtype):
    """
    Use case of blocked NCHW4c global pooling with big spatial valies
    """
    input_shape = (1, 160, 160, 8, 4)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    mean = relay.max(A, axis=[1, 2], keepdims=True)
    mod = relay.Function([A], mean)

    build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


if __name__ == "__main__":
    tvm.testing.main()
