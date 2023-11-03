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

import tvm
from tvm import relay
from utils.adreno_utils import build_run_compare, build_run_compare_vm


executor_type = tvm.testing.parameter("ge", "vm")
dtype = tvm.testing.parameter("float32")


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_global_pool2d_nchw_wide(remote, target, executor_type, dtype):
    """
    Use case of NCHW global pooling with big spatial valies
    """
    input_shape = (1, 32, 160, 160)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    C = relay.nn.global_avg_pool2d(A)
    mod = relay.Function([A], C)

    if executor_type == "ge":
        build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_global_pool2d_nchw4c_wide(remote, target, executor_type, dtype):
    """
    Use case of blocked NCHW4c global pooling with big spatial valies
    """
    input_shape = (1, 8, 160, 160, 4)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    C = relay.nn.global_avg_pool2d(A, layout="NCHW4c")
    mod = relay.Function([A], C)

    if executor_type == "ge":
        build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_global_pool2d_nchw_deep(remote, target, executor_type, dtype):
    """
    Use case of NCHW deep global pooling
    """
    input_shape = (1, 2048, 20, 20)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    C = relay.nn.global_avg_pool2d(A)
    mod = relay.Function([A], C)

    if executor_type == "ge":
        build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_global_pool2d_nchw4c_deep(remote, target, executor_type, dtype):
    """
    Use case of blocked NCHW4c deep global pooling
    """
    input_shape = (1, 512, 20, 20, 4)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    C = relay.nn.global_avg_pool2d(A, layout="NCHW4c")
    mod = relay.Function([A], C)

    if executor_type == "ge":
        build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_global_pool2d_nhwc(remote, target, executor_type, dtype):
    """
    Use case of NHWC global pooling with big spatial valies
    """
    input_shape = (1, 160, 160, 32)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    C = relay.nn.global_avg_pool2d(A, layout="NHWC")
    mod = relay.Function([A], C)

    if executor_type == "ge":
        build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_global_pool2d_nhwc4c(remote, target, executor_type, dtype):
    """
    Use case of NHWC deep global pooling
    """
    input_shape = (1, 160, 160, 8, 4)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    C = relay.nn.global_avg_pool2d(A, layout="NHWC4c")
    mod = relay.Function([A], C)

    if executor_type == "ge":
        build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_global_max_pool2d_nchw_wide(remote, target, executor_type, dtype):
    """
    Use case of NCHW global pooling with big spatial valies
    """
    input_shape = (1, 32, 160, 160)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    C = relay.nn.global_max_pool2d(A)
    mod = relay.Function([A], C)

    if executor_type == "ge":
        build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_global_max_pool2d_nchw4c_wide(remote, target, executor_type, dtype):
    """
    Use case of blocked NCHW4c global pooling with big spatial valies
    """
    input_shape = (1, 8, 160, 160, 4)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    C = relay.nn.global_max_pool2d(A, layout="NCHW4c")
    mod = relay.Function([A], C)

    if executor_type == "ge":
        build_run_compare(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, {}, {"data": input_shape}, {"data": dtype}, target)


if __name__ == "__main__":
    tvm.testing.main()
