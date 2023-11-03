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
import numpy as np
from tvm import relay
from tvm.relay import testing
from utils.adreno_utils import gpu_preprocess, build_run_compare, build_run_compare_vm

executor_type = tvm.testing.parameter("ge", "vm")
dtype = tvm.testing.parameter("float32")


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_depthwise_conv2d_bias_nchwc(remote, target, executor_type, dtype):
    input_shape = (1, 64, 112, 112)
    filter_shape = (64, 1, 3, 3)
    bias_shape = (1, 64, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[1, 1, 1, 1],
        strides=[2, 2],
        out_dtype=dtype,
        channels=64,
        groups=64,
        kernel_size=(3, 3),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    # mod, params = relay.testing.init.create_workload(func)
    np.random.seed(1)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    if executor_type == "ge":
        build_run_compare(
            remote, mod, params1, {"data": input_shape}, {"data": dtype}, target, [], gpu_preprocess
        )
    else:
        build_run_compare_vm(
            remote, mod, params1, {"data": input_shape}, {"data": dtype}, target, [], gpu_preprocess
        )


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_depthwise_conv2d_nchwc(remote, target, executor_type, dtype):
    input_shape = (1, 64, 112, 112)
    filter_shape = (64, 1, 3, 3)
    bias_shape = (1, 64, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[1, 1, 1, 1],
        strides=[2, 2],
        out_dtype=dtype,
        channels=64,
        groups=64,
        kernel_size=(3, 3),
    )

    mod = relay.Function([A, B], conv)
    # mod, params = relay.testing.init.create_workload(func)
    np.random.seed(1)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
    }

    if executor_type == "ge":
        build_run_compare(
            remote, mod, params1, {"data": input_shape}, {"data": dtype}, target, [], gpu_preprocess
        )
    else:
        build_run_compare_vm(
            remote, mod, params1, {"data": input_shape}, {"data": dtype}, target, [], gpu_preprocess
        )


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_depthwise_conv2d_bias_nchw(remote, target, executor_type, dtype):
    input_shape = (1, 64, 112, 112)
    filter_shape = (64, 1, 3, 3)
    bias_shape = (1, 64, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[1, 1, 1, 1],
        strides=[2, 2],
        out_dtype=dtype,
        channels=64,
        groups=64,
        kernel_size=(3, 3),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    # mod, params = relay.testing.init.create_workload(func)
    np.random.seed(1)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    if executor_type == "ge":
        build_run_compare(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_depthwise_conv2d_repack_bias_nchw(remote, target, executor_type, dtype):
    input_shape = (1, 63, 112, 112)
    filter_shape = (63, 1, 3, 3)
    bias_shape = (1, 63, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[1, 1, 1, 1],
        strides=[2, 2],
        out_dtype=dtype,
        channels=63,
        groups=63,
        kernel_size=(3, 3),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    # mod, params = relay.testing.init.create_workload(func)
    np.random.seed(1)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    if executor_type == "ge":
        build_run_compare(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_to_3_channels(remote, target, executor_type, dtype):
    input_shape = (1, 3, 200, 200)
    filter_shape = (3, 1, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)

    D = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        out_dtype=dtype,
        channels=3,
        groups=3,
        kernel_size=(1, 1),
    )
    mod = relay.Function([A, B], D)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    initializer("weight", filter_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
    }

    if executor_type == "ge":
        build_run_compare(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target, [])
    else:
        build_run_compare_vm(
            remote, mod, params1, {"data": input_shape}, {"data": dtype}, target, []
        )


if __name__ == "__main__":
    tvm.testing.main()
