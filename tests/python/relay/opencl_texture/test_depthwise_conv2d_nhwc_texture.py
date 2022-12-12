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
from utils.adreno_utils import build_run_compare

dtype = tvm.testing.parameter("float32")


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_depthwise_conv2d_deeplabv3_1_129_129_144x3_3_144_1(remote, target, dtype):
    input_shape = (1, 129, 129, 144)
    filter_shape = (3, 3, 144, 1)
    kernel_size = (filter_shape[0], filter_shape[1])
    bias_shape = (filter_shape[2],)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWOI",
        out_dtype=dtype,
        groups=filter_shape[2],
        channels=filter_shape[2],
        kernel_size=kernel_size,
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    mod = relay.Function([A, B, bias], conv)
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

    build_run_compare(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_depthwise_conv2d_deeplabv3_4_35_35_576x3_3_576_1(remote, target, dtype):
    input_shape = (4, 35, 35, 576)
    filter_shape = (3, 3, 576, 1)
    kernel_size = (filter_shape[0], filter_shape[1])
    bias_shape = (filter_shape[2],)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWOI",
        out_dtype=dtype,
        groups=filter_shape[2],
        channels=filter_shape[2],
        kernel_size=kernel_size,
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    mod = relay.Function([A, B, bias], conv)
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

    build_run_compare(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_depthwise_conv2d_deeplabv3_1_129_129_144x3_3_144_1_with_padding(remote, target, dtype):
    input_shape = (1, 129, 129, 144)
    filter_shape = (3, 3, 144, 1)
    kernel_size = (filter_shape[0], filter_shape[1])
    bias_shape = (filter_shape[2],)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWOI",
        padding=[3, 3, 3, 3],
        strides=[2, 2],
        out_dtype=dtype,
        groups=filter_shape[2],
        channels=filter_shape[2],
        kernel_size=kernel_size,
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

    build_run_compare(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_depthwise_conv2d_1_513_513_7x3_3_7_1(remote, target, dtype):
    input_shape = (1, 513, 513, 7)
    filter_shape = (3, 3, 7, 1)
    bias_shape = (filter_shape[2],)
    kernel_size = (filter_shape[0], filter_shape[1])
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWOI",
        out_dtype=dtype,
        channels=filter_shape[2],
        groups=filter_shape[2],
        kernel_size=kernel_size,
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(1)
    initializer = relay.testing.init.Xavier()
    filter_data = np.ones(filter_shape).astype(dtype)
    bias_data = np.ones(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_depthwise_conv2d_1_513_513_3x3_3_3_1(remote, target, dtype):
    input_shape = (1, 513, 513, 3)
    filter_shape = (3, 3, 3, 1)
    bias_shape = (filter_shape[2],)
    kernel_size = (filter_shape[0], filter_shape[1])
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWOI",
        out_dtype=dtype,
        channels=filter_shape[2],
        groups=filter_shape[2],
        kernel_size=kernel_size,
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(1)
    initializer = relay.testing.init.Xavier()
    filter_data = np.ones(filter_shape).astype(dtype)
    bias_data = np.ones(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)


if __name__ == "__main__":
    tvm.testing.main()
