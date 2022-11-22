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
import re
import tvm
import numpy as np
from tvm import relay
from tvm.relay import testing
from tvm.contrib import utils
from utils.adreno_utils import gpu_preprocess, build_run_compare
import pytest


dtype = tvm.testing.parameter("float32")


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_deeplabv3_1_257_257_32x1_1_32_16(remote, target, dtype):
    input_shape = (1, 257, 257, 32)
    filter_shape = (1, 1, 32, 16)
    bias_shape = (filter_shape[-1],)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype=dtype,
        channels=filter_shape[-1],
        kernel_size=(1, 1),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
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
def test_conv2d_deeplabv3_1_257_257_32x1_1_32_16_with_padding(remote, target, dtype):
    input_shape = (1, 257, 257, 32)
    filter_shape = (1, 1, 32, 16)
    bias_shape = (filter_shape[-1],)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWIO",
        padding=[3, 3, 3, 3],
        strides=[2, 2],
        out_dtype=dtype,
        channels=filter_shape[-1],
        kernel_size=(1, 1),
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
def test_conv2d_4_35_35_32x3_3_144_16(remote, target, dtype):
    input_shape = (4, 35, 35, 32)
    filter_shape = (3, 3, 32, 16)
    bias_shape = (filter_shape[-1],)
    kernel_size = (filter_shape[0], filter_shape[1])
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype=dtype,
        channels=filter_shape[-1],
        kernel_size=kernel_size,
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
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
def test_conv2d_deeplabv3_1_513_513_3x3_3_3_32(remote, target, dtype):
    input_shape = (1, 513, 513, 3)
    filter_shape = (3, 3, 3, 32)
    bias_shape = (filter_shape[-1],)
    kernel_size = (filter_shape[0], filter_shape[1])
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWIO",
        out_dtype=dtype,
        channels=filter_shape[-1],
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
def test_conv2d_inceptionv3_64x35x35_96x64x3x3_nopad(remote, target, dtype):
    input_shape = (1, 42, 42, 32)
    filter_shape = (3, 3, 32, 96)
    bias_shape = (1, 1, 1, 96)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWIO",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=96,
        kernel_size=(3, 3),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(
        remote, mod, params1, {"data": input_shape}, {"data": dtype}, target, [], gpu_preprocess
    )


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_inceptionv3_64x35x35_96x64x3x3_nopad_pass(remote, target, dtype):
    input_shape = (1, 40, 40, 32)
    filter_shape = (2, 2, 32, 96)
    bias_shape = (1, 1, 1, 96)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWIO",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=96,
        kernel_size=(2, 2),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(
        remote, mod, params1, {"data": input_shape}, {"data": dtype}, target, [], gpu_preprocess
    )


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_inceptionv3_35_35_strides(remote, target, dtype):
    input_shape = (1, 35, 35, 48)
    filter_shape = (5, 5, 48, 64)
    bias_shape = (1, 1, 1, 64)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWIO",
        padding=[2, 2, 2, 2],
        strides=[1, 1],
        out_dtype=dtype,
        channels=64,
        kernel_size=(5, 5),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    build_run_compare(
        remote, mod, params1, {"data": input_shape}, {"data": dtype}, target, [], gpu_preprocess
    )


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_resnet50_v2_nhwc_3c(remote, target, dtype):
    input_shape = (1, 224, 224, 3)
    filter_shape = (7, 7, 3, 64)
    bias_shape = (1, 1, 1, 64)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWIO",
        padding=[3, 3, 3, 3],
        strides=[2, 2],
        out_dtype=dtype,
        channels=64,
        kernel_size=(7, 7),
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
def test_conv2d_inceptionv3_nhwc_3c(remote, target, dtype):
    input_shape = (1, 299, 299, 3)
    filter_shape = (3, 3, 3, 64)
    bias_shape = (1, 1, 1, 64)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWIO",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=64,
        kernel_size=(3, 3),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
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
def test_conv2d_1x1_16c16spatial(remote, target, dtype):
    input_shape = (1, 128, 128, 16)
    filter_shape = (4, 4, 16, 32)
    bias_shape = (1, 1, 1, 32)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWIO",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=32,
        kernel_size=(4, 4),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
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
def test_conv2d_4x4_16c16pad(remote, target, dtype):
    input_shape = (1, 256, 256, 32)
    filter_shape = (4, 4, 32, 32)
    bias_shape = (1, 1, 1, 32)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWIO",
        padding=[3, 3, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=32,
        kernel_size=(4, 4),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
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
def test_conv2d_4x4x4_16c16pad(remote, target, dtype):
    input_shape = (1, 256, 256, 32)
    filter_shape = (4, 4, 32, 4)
    bias_shape = (1, 1, 1, 4)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWIO",
        padding=[3, 3, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=4,
        kernel_size=(4, 4),
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
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
def test_conv2d_yolov3_v2_nhwc_3c(remote, target, dtype):
    input_shape = (1, 13, 13, 1024)
    filter_shape = (1, 1, 1024, 255)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)

    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWIO",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=255,
        kernel_size=(1, 1),
    )

    mod = relay.Function([A, B], conv)
    # mod, params = relay.testing.init.create_workload(func)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    initializer("weight", filter_data)
    params = {
        "weight": tvm.nd.array(filter_data),
    }

    build_run_compare(remote, mod, params, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_vgg16_winograd_4d(remote, target, dtype):
    input_shape = (1, 28, 28, 512)
    filter_shape = (3, 3, 512, 512)
    bias_shape = (1, 1, 1, 512)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWIO",
        padding=[1, 1, 1, 1],
        channels=512,
        kernel_size=[3, 3],
        out_dtype=dtype,
    )
    D = relay.op.add(conv, bias)
    D = relay.op.nn.relu(D)

    mod = relay.Function([A, B, bias], D)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    bias_data = np.zeros(bias_shape).astype(dtype)
    initializer("weight", filter_data)
    initializer("bias", bias_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
        "bias": tvm.nd.array(bias_data),
    }

    temp = utils.tempdir()
    stat_file = temp.relpath("stat.log")
    with open(stat_file, "w") as f:
        f.write(
            f'{{"input": ["opencl -keys=adreno,opencl,gpu -device=adreno -max_num_threads=256", "conv2d_nhwc_winograd.image2d", [["TENSOR", [1, 28, 28, 512], "{dtype}"], ["TENSOR", [3, 3, 512, 512], "{dtype}"], [1, 1], [1, 1, 1, 1], [1, 1], "{dtype}"], {{}}], "config": {{"index": 1591, "code_hash": null, "entity": [["auto_unroll_max_step", "ot", 4], ["tile_y", "sp", [-1, 1, 32]], ["tile_x", "sp", [-1, 4, 2]], ["tile_rc", "sp", [-1, 8]]]}}, "result": [[0.0037244], 0, 7.06374192237854, 1653898629.7427933], "version": 0.2, "tvm_version": "0.8.dev0"}}\n'
        )
    graph = build_run_compare(
        remote, mod, params1, {"data": input_shape}, {"data": dtype}, target, stat_file=stat_file
    )
    matches = re.findall("winograd", graph)
    assert len(matches) > 0


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_winograd_conv(remote, target, dtype):
    input_shape = (1, 3, 3, 4)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    filter_shape3 = (3, 3, 4, 8)
    bias_shape3 = (1, 1, 1, 8)
    B3 = relay.var("weight3", shape=filter_shape3, dtype=dtype)
    D = relay.nn.conv2d(
        A,
        B3,
        data_layout="NHWC",
        kernel_layout="HWIO",
        padding=[1, 1, 1, 1],
        channels=8,
        kernel_size=[3, 3],
        out_dtype=dtype,
    )

    filter_shape4 = (3, 3, 8, 8)
    bias_shape4 = (1, 1, 1, 8)
    B4 = relay.var("weight4", shape=filter_shape4, dtype=dtype)
    D = relay.nn.conv2d(
        D,
        B4,
        data_layout="NHWC",
        kernel_layout="HWIO",
        padding=[1, 1, 1, 1],
        channels=8,
        kernel_size=[3, 3],
        out_dtype=dtype,
    )
    mod = relay.Function([A, B3, B4], D)
    np.random.seed(1)
    initializer = relay.testing.init.Xavier()
    filter_data3 = np.zeros(filter_shape3).astype(dtype)
    bias_data3 = np.zeros(bias_shape3).astype(dtype)
    filter_data4 = np.zeros(filter_shape4).astype(dtype)
    bias_data4 = np.zeros(bias_shape4).astype(dtype)
    initializer("weight", filter_data3)
    initializer("bias", bias_data3)
    initializer("weight", filter_data4)
    initializer("bias", bias_data4)
    params1 = {
        "weight3": tvm.nd.array(filter_data3),
        "weight4": tvm.nd.array(filter_data4),
    }

    temp = utils.tempdir()
    stat_file = temp.relpath("stat.log")
    with open(stat_file, "w") as f:
        f.write(
            f'{{"input": ["opencl -keys=adreno,opencl,gpu -device=adreno -max_num_threads=256", "conv2d_nhwc_winograd.image2d", [["TENSOR", [1, 3, 3, 4], "{dtype}"], ["TENSOR", [3, 3, 4, 8], "{dtype}"], [1, 1], [1, 1, 1, 1], [1, 1], "{dtype}"], {{}}], "config": {{"index": 1591, "code_hash": null, "entity": [["auto_unroll_max_step", "ot", 4], ["tile_y", "sp", [-1, 1, 32]], ["tile_x", "sp", [-1, 4, 2]], ["tile_rc", "sp", [-1, 8]]]}}, "result": [[0.0037244], 0, 7.06374192237854, 1653898629.7427933], "version": 0.2, "tvm_version": "0.8.dev0"}}\n'
        )
    graph = build_run_compare(
        remote, mod, params1, {"data": input_shape}, {"data": dtype}, target, stat_file=stat_file
    )
    matches = re.findall("winograd", graph)
    assert len(matches) > 0


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_winograd_non_rect(remote, target, dtype):
    input_shape = (1, 36, 64, 771)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    filter_shape = (3, 3, 771, 128)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    D = relay.nn.conv2d(
        A,
        B,
        data_layout="NHWC",
        kernel_layout="HWIO",
        padding=[1, 1, 1, 1],
        channels=128,
        kernel_size=[3, 3],
        out_dtype=dtype,
    )

    mod = relay.Function([A, B], D)
    np.random.seed(1)
    initializer = relay.testing.init.Xavier()
    filter_data = np.zeros(filter_shape).astype(dtype)
    initializer("weight", filter_data)
    params1 = {
        "weight": tvm.nd.array(filter_data),
    }

    temp = utils.tempdir()
    stat_file = temp.relpath("stat.log")
    with open(stat_file, "w") as f:
        f.write(
            f'{{"input": ["opencl -keys=adreno,opencl,gpu -device=adreno -max_num_threads=256 -texture_spatial_limit=16384 -thread_warp_size=1", "conv2d_nhwc_winograd.image2d", [["TENSOR", [1, 36, 64, 771], "{dtype}"], ["TENSOR", [3, 3, 771, 128], "{dtype}"], [1, 1], [1, 1, 1, 1], [1, 1], "{dtype}"], {{}}], "config": {{"index": 5399, "code_hash": null, "entity": [["auto_unroll_max_step", "ot", 16], ["tile_y", "sp", [-1, 1, 32]], ["tile_x", "sp", [-1, 4, 8]], ["tile_rc", "sp", [-1, 193]]]}}, "result": [[0.0037244], 0, 7.06374192237854, 1653898629.7427933], "version": 0.2, "tvm_version": "0.8.dev0"}}\n'
        )
    graph = build_run_compare(
        remote, mod, params1, {"data": input_shape}, {"data": dtype}, target, stat_file=stat_file
    )
    matches = re.findall("winograd", graph)
    assert len(matches) > 0


if __name__ == "__main__":
    tvm.testing.main()
