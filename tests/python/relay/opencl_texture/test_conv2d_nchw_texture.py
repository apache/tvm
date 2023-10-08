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
from utils.adreno_utils import gpu_preprocess, build_run_compare, build_run_compare_vm
import pytest


executor_type = tvm.testing.parameter("ge", "vm")
dtype = tvm.testing.parameter("float32")


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_inceptionv3_64x35x35_96x64x3x3_nopad(remote, target, executor_type, dtype):
    input_shape = (1, 32, 42, 42)
    filter_shape = (96, 32, 3, 3)
    bias_shape = (1, 96, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
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
def test_conv2d_inceptionv3_64x35x35_96x64x3x3_nopad_pass(remote, target, executor_type, dtype):
    input_shape = (1, 32, 40, 40)
    filter_shape = (96, 32, 2, 2)
    bias_shape = (1, 96, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
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
def test_conv2d_inceptionv3_35_35_strides(remote, target, executor_type, dtype):
    input_shape = (1, 48, 35, 35)
    filter_shape = (64, 48, 5, 5)
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
def test_conv2d_resnet50_v2_nchw_3c(remote, target, executor_type, dtype):
    input_shape = (1, 3, 224, 224)
    filter_shape = (64, 3, 7, 7)
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

    if executor_type == "ge":
        build_run_compare(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_inceptionv3_nchw_3c(remote, target, executor_type, dtype):
    input_shape = (1, 3, 299, 299)
    filter_shape = (64, 3, 3, 3)
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

    if executor_type == "ge":
        build_run_compare(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_1x1_16c16spatial(remote, target, executor_type, dtype):
    input_shape = (1, 16, 256, 256)
    filter_shape = (32, 16, 4, 4)
    bias_shape = (1, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
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

    if executor_type == "ge":
        build_run_compare(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_4x4_16c16pad(remote, target, executor_type, dtype):
    input_shape = (1, 32, 256, 256)
    filter_shape = (32, 32, 4, 4)
    bias_shape = (1, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
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

    if executor_type == "ge":
        build_run_compare(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_4x4x4_16c16pad(remote, target, executor_type, dtype):
    input_shape = (1, 32, 256, 256)
    filter_shape = (4, 32, 4, 4)
    bias_shape = (1, 4, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    # C = relay.nn.relu(A)
    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
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

    if executor_type == "ge":
        build_run_compare(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, params1, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_yolov3_v2_nchw_3c(remote, target, executor_type, dtype):
    input_shape = (1, 1024, 13, 13)
    filter_shape = (255, 1024, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)

    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
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

    if executor_type == "ge":
        build_run_compare(remote, mod, params, {"data": input_shape}, {"data": dtype}, target)
    else:
        build_run_compare_vm(remote, mod, params, {"data": input_shape}, {"data": dtype}, target)


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_vgg16_winograd_4d(remote, target, executor_type, dtype):
    input_shape = (1, 512, 28, 28)
    filter_shape = (512, 512, 3, 3)
    bias_shape = (1, 512, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    bias = relay.var("bias", shape=bias_shape, dtype=dtype)

    conv = relay.nn.conv2d(
        A,
        B,
        data_layout="NCHW",
        kernel_layout="OIHW",
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
            f'{{"input": ["opencl -keys=adreno,opencl,gpu -device=adreno -max_num_threads=256", "conv2d_nchw_winograd.image2d", [["TENSOR", [1, 512, 28, 28], "{dtype}"], ["TENSOR", [512, 512, 3, 3], "{dtype}"], [1, 1], [1, 1, 1, 1], [1, 1], "{dtype}"], {{}}], "config": {{"index": 1591, "code_hash": null, "entity": [["auto_unroll_max_step", "ot", 4], ["tile_y", "sp", [-1, 1, 32]], ["tile_x", "sp", [-1, 4, 2]], ["tile_rc", "sp", [-1, 8]]]}}, "result": [[0.0037244], 0, 7.06374192237854, 1653898629.7427933], "version": 0.2, "tvm_version": "0.8.dev0"}}\n'
        )
    if executor_type == "ge":
        graph = build_run_compare(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
            stat_file=stat_file,
        )
        matches = re.findall("winograd", graph)
        assert len(matches) > 0
    else:
        vmc = build_run_compare_vm(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
            stat_file=stat_file,
        )
        matches = re.findall("winograd", vmc.primitives)
        assert len(matches) > 0


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_winograd_conv(remote, target, executor_type, dtype):
    input_shape = (1, 4, 3, 3)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    filter_shape3 = (8, 4, 3, 3)
    bias_shape3 = (8,)
    B3 = relay.var("weight3", shape=filter_shape3, dtype=dtype)
    D = relay.nn.conv2d(
        A, B3, padding=[1, 1, 1, 1], channels=8, kernel_size=[3, 3], out_dtype=dtype
    )

    filter_shape4 = (8, 8, 3, 3)
    bias_shape4 = (8,)
    B4 = relay.var("weight4", shape=filter_shape4, dtype=dtype)
    D = relay.nn.conv2d(
        D, B4, padding=[1, 1, 1, 1], channels=8, kernel_size=[3, 3], out_dtype=dtype
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
            f'{{"input": ["opencl -keys=adreno,opencl,gpu -device=adreno -max_num_threads=256", "conv2d_nchw_winograd.image2d", [["TENSOR", [1, 4, 3, 3], "{dtype}"], ["TENSOR", [8, 4, 3, 3], "{dtype}"], [1, 1], [1, 1, 1, 1], [1, 1], "{dtype}"], {{}}], "config": {{"index": 1591, "code_hash": null, "entity": [["auto_unroll_max_step", "ot", 4], ["tile_y", "sp", [-1, 1, 32]], ["tile_x", "sp", [-1, 4, 2]], ["tile_rc", "sp", [-1, 8]]]}}, "result": [[0.0037244], 0, 7.06374192237854, 1653898629.7427933], "version": 0.2, "tvm_version": "0.8.dev0"}}\n'
        )
    if executor_type == "ge":
        graph = build_run_compare(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
            stat_file=stat_file,
        )
        matches = re.findall("winograd", graph)
        assert len(matches) > 0
    else:
        vmc = build_run_compare_vm(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
            stat_file=stat_file,
        )
        matches = re.findall("winograd", vmc.primitives)
        assert len(matches) > 0


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_residual_block(remote, target, executor_type, dtype):
    """
    - some kind of residual block followed by convolution to have texture after residual block
    - scalar data type verification which should be mapped to global memory scope
        layout_transform (NCHW->NCHW4c)
                  |                      <- buffer
                conv2d (1)                  <- to get textures as output
               /         \
            conv2d (2)    |
                 \       /
                    add                     <- add should be fused into conv2d (2)
                multiply to scalar          <- buffer to the input of multiply scalar value
                    relu
                     |                      <- texture in intermediate tensor
                  conv2d (3)
                   relu
                     |                      <- buffer
               layout_transform (NCHW4c->NCHW)
    """
    input_shape = (1, 32, 40, 40)
    filter_shape1 = (32, 32, 2, 2)
    filter_shape2 = (32, 32, 1, 1)
    filter_shape3 = (32, 32, 2, 2)
    bias_shape1 = (1, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W1 = relay.var("weight1", shape=filter_shape1, dtype=dtype)
    B1 = relay.var("bias1", shape=bias_shape1, dtype=dtype)
    W2 = relay.var("weight2", shape=filter_shape2, dtype=dtype)
    W3 = relay.var("weight3", shape=filter_shape3, dtype=dtype)

    conv1 = relay.nn.conv2d(
        A,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    D = relay.op.add(conv1, B1)
    D = relay.op.nn.relu(D)

    conv2 = relay.nn.conv2d(
        D,
        W2,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )
    D = relay.op.add(conv2, D)
    D = D * relay.const(0.15, dtype)
    D = relay.op.nn.relu(D)

    conv3 = relay.nn.conv2d(
        D,
        W3,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    D = relay.op.nn.relu(conv3)

    mod = relay.Function([A, W1, B1, W2, W3], D)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data1 = np.zeros(filter_shape1).astype(dtype)
    bias_data1 = np.zeros(bias_shape1).astype(dtype)
    initializer("weight", filter_data1)
    initializer("bias", bias_data1)
    filter_data2 = np.zeros(filter_shape2).astype(dtype)
    initializer("weight", filter_data2)
    filter_data3 = np.zeros(filter_shape3).astype(dtype)
    initializer("weight", filter_data3)
    params1 = {
        "weight1": tvm.nd.array(filter_data1),
        "bias1": tvm.nd.array(bias_data1),
        "weight2": tvm.nd.array(filter_data2),
        "weight3": tvm.nd.array(filter_data3),
    }
    if dtype == "float16":
        static_memory_scope = [
            "",
            "global.texture",
            "global.texture-weight",
            "global.texture-weight",
            "global.texture",
            "global.texture-weight",
            "global",
            "global.texture",
            "global.texture-weight",
            "",
            "",
        ]
    else:
        static_memory_scope = [
            "",
            "global.texture",
            "global",
            "global.texture-weight",
            "global.texture",
            "global.texture-weight",
            "global.texture",
            "global",
            "",
            "",
        ]

    if executor_type == "ge":
        build_run_compare(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
            static_memory_scope,
        )
    else:
        build_run_compare_vm(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
        )


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_concat(remote, target, executor_type, dtype):
    """
        layout_transform (NCHW->NCHW4c)
                  |                      <- buffer
                conv2d (1)               <- to get textures as output
               /         \
            conv2d (2)    conv2d (3)
                 \       /               <- concat does not support textures, there we should have buffers
                concatenation
                     |                   <- buffer
               layout_transform (NCHW4c->NCHW)
    """
    input_shape = (1, 32, 40, 40)
    filter_shape1 = (96, 32, 2, 2)
    filter_shape2 = (32, 96, 2, 2)
    filter_shape3 = (5, 96, 2, 2)
    bias_shape1 = (1, 96, 1, 1)
    bias_shape2 = (1, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W1 = relay.var("weight1", shape=filter_shape1, dtype=dtype)
    B1 = relay.var("bias1", shape=bias_shape1, dtype=dtype)
    W2 = relay.var("weight2", shape=filter_shape2, dtype=dtype)
    W3 = relay.var("weight3", shape=filter_shape3, dtype=dtype)
    B2 = relay.var("bias2", shape=bias_shape2, dtype=dtype)

    # C = relay.nn.relu(A)
    conv1 = relay.nn.conv2d(
        A,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=96,
        kernel_size=(2, 2),
    )
    D = relay.op.add(conv1, B1)
    D = relay.op.nn.relu(D)

    conv2 = relay.nn.conv2d(
        D,
        W2,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    conv2 = relay.op.add(conv2, B2)
    conv2 = relay.op.nn.relu(conv2)

    conv3 = relay.nn.conv2d(
        D,
        W3,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[2, 2],
        out_dtype=dtype,
        channels=5,
        kernel_size=(2, 2),
    )

    t = relay.Tuple([conv2, conv3])
    c = relay.op.concatenate(t, axis=1)

    mod = relay.Function([A, W1, B1, W2, B2, W3], c)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data1 = np.zeros(filter_shape1).astype(dtype)
    bias_data1 = np.zeros(bias_shape1).astype(dtype)
    initializer("weight", filter_data1)
    initializer("bias", bias_data1)
    filter_data2 = np.zeros(filter_shape2).astype(dtype)
    bias_data2 = np.zeros(bias_shape2).astype(dtype)
    initializer("weight", filter_data2)
    initializer("bias", bias_data2)
    filter_data3 = np.zeros(filter_shape3).astype(dtype)
    initializer("weight", filter_data3)
    params1 = {
        "weight1": tvm.nd.array(filter_data1),
        "bias1": tvm.nd.array(bias_data1),
        "weight2": tvm.nd.array(filter_data2),
        "bias2": tvm.nd.array(bias_data2),
        "weight3": tvm.nd.array(filter_data3),
    }

    static_memory_scope = [
        "",
        "global.texture",
        "global",
        "global.texture-weight",
        "global",
        "global.texture-nhwc",
        "global",
        "global.texture-weight",
        "",
        "",
        "",
        "",
        "",
    ]

    if executor_type == "ge":
        build_run_compare(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
            static_memory_scope,
        )
    else:
        build_run_compare_vm(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
        )


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_pooling_branching_texture_params(remote, target, executor_type, dtype):
    """
    Verification of the pooling and many branches having textures
                layout_transform (NCHW->NCHW4c)
                         |                        <- buffer
                      conv2d (0)                  <- to get textures
                         |                        <- textures
                     pooling
               /           \           \          <- textures
            conv2d (1)    conv2d (2)    conv2d (3)
                \             /           |
                     add                  |       <- to have  the only one output, will be fused
                      \                  /
                            add                  <- to have  the only one output, will be fused
                             |                   <- buffer
                    layout_transform (NCHW4c->NCHW)
    """
    input_shape = (1, 32, 40, 40)
    filter_shape0 = (32, 32, 1, 1)
    filter_shape1 = (32, 32, 2, 2)
    filter_shape2 = (32, 32, 1, 1)
    filter_shape3 = (32, 32, 2, 2)
    bias_shape1 = (1, 32, 1, 1)
    # bias_shape2 = (1, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W0 = relay.var("weight0", shape=filter_shape0, dtype=dtype)
    W1 = relay.var("weight1", shape=filter_shape1, dtype=dtype)
    B1 = relay.var("bias1", shape=bias_shape1, dtype=dtype)
    W2 = relay.var("weight2", shape=filter_shape2, dtype=dtype)
    W3 = relay.var("weight3", shape=filter_shape3, dtype=dtype)

    conv0 = relay.nn.conv2d(
        A,
        W0,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )

    pool = relay.nn.avg_pool2d(conv0, pool_size=(2, 2), strides=(2, 2))
    conv1 = relay.nn.conv2d(
        pool,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 1, 1],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    conv1 = relay.op.add(conv1, B1)
    conv1 = relay.op.nn.relu(conv1)

    conv2 = relay.nn.conv2d(
        pool,
        W2,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )

    conv3 = relay.nn.conv2d(
        pool,
        W3,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 1, 1, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    conv3 = relay.op.nn.relu(conv3)
    res = relay.op.add(conv1, conv2)
    res = relay.op.add(res, conv3)

    mod = relay.Function([A, W0, W1, B1, W2, W3], res)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data0 = np.zeros(filter_shape0).astype(dtype)
    filter_data1 = np.zeros(filter_shape1).astype(dtype)
    bias_data1 = np.zeros(bias_shape1).astype(dtype)
    initializer("weight", filter_data1)
    initializer("bias", bias_data1)
    filter_data2 = np.zeros(filter_shape2).astype(dtype)
    initializer("weight", filter_data2)
    filter_data3 = np.zeros(filter_shape3).astype(dtype)
    initializer("weight", filter_data3)
    params1 = {
        "weight0": tvm.nd.array(filter_data0),
        "weight1": tvm.nd.array(filter_data1),
        "bias1": tvm.nd.array(bias_data1),
        "weight2": tvm.nd.array(filter_data2),
        "weight3": tvm.nd.array(filter_data3),
    }

    static_memory_scope = [
        "",
        "global.texture",
        "global.texture-weight",
        "global.texture",
        "global.texture",
        "global",
        "global.texture-weight",
        "global",
        "global.texture-weight",
        "global.texture",
        "global.texture",
        "",
        "",
    ]

    if executor_type == "ge":
        build_run_compare(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
            static_memory_scope,
        )
    else:
        build_run_compare_vm(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
        )


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_branching_texture_params(remote, target, executor_type, dtype):
    """
    Verification of passing texture to several consumers markup of relay variables in
    primary functions + on_device

                layout_transform (NCHW->NCHW4c)
                         |                      <- buffer
                      conv2d (0)                <- to get textures
             /           \           \          <- here should be textures and textures in params
          conv2d (1)    conv2d (2)    conv2d (3)
            \             /           |
                  add                 |         <- to have  the only one output
                    \                /
                           add                  <- to have  the only one output
                            |                   <- buffer
                    layout_transform (NCHW4c->NCHW)
    """
    input_shape = (1, 32, 40, 40)
    filter_shape0 = (32, 32, 1, 1)
    filter_shape1 = (32, 32, 2, 2)
    filter_shape2 = (32, 32, 1, 1)
    filter_shape3 = (32, 32, 2, 2)
    bias_shape1 = (1, 32, 1, 1)
    # bias_shape2 = (1, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W0 = relay.var("weight0", shape=filter_shape0, dtype=dtype)
    W1 = relay.var("weight1", shape=filter_shape1, dtype=dtype)
    B1 = relay.var("bias1", shape=bias_shape1, dtype=dtype)
    W2 = relay.var("weight2", shape=filter_shape2, dtype=dtype)
    W3 = relay.var("weight3", shape=filter_shape3, dtype=dtype)

    conv0 = relay.nn.conv2d(
        A,
        W0,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )

    conv1 = relay.nn.conv2d(
        conv0,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 1, 1],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    conv1 = relay.op.add(conv1, B1)
    conv1 = relay.op.nn.relu(conv1)

    conv2 = relay.nn.conv2d(
        conv0,
        W2,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )

    conv3 = relay.nn.conv2d(
        conv0,
        W3,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 1, 1, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(2, 2),
    )
    conv3 = relay.op.nn.relu(conv3)
    res = relay.op.add(conv1, conv2)
    res = relay.op.add(res, conv3)

    mod = relay.Function([A, W0, W1, B1, W2, W3], res)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data0 = np.zeros(filter_shape0).astype(dtype)
    filter_data1 = np.zeros(filter_shape1).astype(dtype)
    bias_data1 = np.zeros(bias_shape1).astype(dtype)
    initializer("weight", filter_data1)
    initializer("bias", bias_data1)
    filter_data2 = np.zeros(filter_shape2).astype(dtype)
    initializer("weight", filter_data2)
    filter_data3 = np.zeros(filter_shape3).astype(dtype)
    initializer("weight", filter_data3)
    params1 = {
        "weight0": tvm.nd.array(filter_data0),
        "weight1": tvm.nd.array(filter_data1),
        "bias1": tvm.nd.array(bias_data1),
        "weight2": tvm.nd.array(filter_data2),
        "weight3": tvm.nd.array(filter_data3),
    }

    static_memory_scope = [
        "",
        "global.texture",
        "global.texture-weight",
        "global.texture",
        "global",
        "global.texture-weight",
        "global",
        "global.texture-weight",
        "global.texture",
        "global.texture",
        "",
        "",
    ]

    if executor_type == "ge":
        build_run_compare(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
            static_memory_scope,
        )
    else:
        build_run_compare_vm(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
        )


# function repeat, params scope are different in reused functions
@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_different_lowering_same_op(remote, target, executor_type, dtype):
    """
    Use case for verification of caching compiled functions
    Three convolutions following by each other in this case should be
    compiled in three different entities and lowered differently because
    they are differ in input param memory scopes and in output memory scope

                layout_transform (NCHW->NCHW4c)
                         |                      <- buffer
                      conv2d (1)                <- buffer as input tensor and texture as output
                         |                      <- texture
                      conv2d (2)                <- texture as input and texture as output
                         |                      <- texture
                      conv2d (3)                <- texture as input and buffer as output
                         |                      <- buffer
                    layout_transform (NCHW4c->NCHW)
    """
    input_shape = (1, 32, 40, 40)
    filter_shape1 = (32, 32, 1, 1)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W1 = relay.var("weight1", shape=filter_shape1, dtype=dtype)

    conv1 = relay.nn.conv2d(
        A,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )

    conv2 = relay.nn.conv2d(
        conv1,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )

    conv3 = relay.nn.conv2d(
        conv2,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[0, 0, 0, 0],
        strides=[1, 1],
        out_dtype=dtype,
        channels=32,
        kernel_size=(1, 1),
    )

    mod = relay.Function([A, W1], conv3)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data1 = np.zeros(filter_shape1).astype(dtype)
    params1 = {
        "weight1": tvm.nd.array(filter_data1),
    }

    static_memory_scope = [
        "",
        "global.texture",
        "global.texture-weight",
        "global.texture",
        "global.texture",
        "",
        "",
    ]

    if executor_type == "ge":
        build_run_compare(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
            static_memory_scope,
        )
    else:
        build_run_compare_vm(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
        )


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_winograd_non_rect(remote, target, executor_type, dtype):
    input_shape = (1, 771, 36, 64)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    filter_shape = (128, 771, 3, 3)
    B = relay.var("weight", shape=filter_shape, dtype=dtype)
    D = relay.nn.conv2d(
        A, B, padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3], out_dtype=dtype
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
            f'{{"input": ["opencl -keys=adreno,opencl,gpu -device=adreno -max_num_threads=256 -texture_spatial_limit=16384 -thread_warp_size=1", "conv2d_nchw_winograd.image2d", [["TENSOR", [1, 771, 36, 64], "{dtype}"], ["TENSOR", [128, 771, 3, 3], "{dtype}"], [1, 1], [1, 1, 1, 1], [1, 1], "{dtype}"], {{}}], "config": {{"index": 5399, "code_hash": null, "entity": [["auto_unroll_max_step", "ot", 16], ["tile_y", "sp", [-1, 1, 32]], ["tile_x", "sp", [-1, 4, 8]], ["tile_rc", "sp", [-1, 193]]]}}, "result": [[0.0037244], 0, 7.06374192237854, 1653898629.7427933], "version": 0.2, "tvm_version": "0.8.dev0"}}\n'
        )
    if executor_type == "ge":
        graph = build_run_compare(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
            stat_file=stat_file,
        )
        matches = re.findall("winograd", graph)
        assert len(matches) > 0
    else:
        vmc = build_run_compare_vm(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
            stat_file=stat_file,
        )
        matches = re.findall("winograd", vmc.primitives)
        assert len(matches) > 0


# function repeat, params scope are different in reused functions
@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_injective_nwo_inputs1(remote, target, executor_type, dtype):
    """
    Use case for verification of stability of annotation primary functions
    having several ops accepting data outside of Primary function
    The visiting of ops during traversing of graph inside primary function
    can depend on order of relay graph creation. Thus the annotation mechanism
    should be reliable for graph traversal order
    The current decision if Prim Function support textures or not depend on
    *any* op accepting input of the function and if op support textures
                                     Input
                               /                   \
                layout_transform (NCHW->NCHW4c)    |
                         |                        /
                      conv2d (1)                 /
                         |                      /
                      conv2d (2)       mean    /
                  /         \                 /   <- Primary function several head ops
             (1)add    (2)layout_transform    |
                 |        (NCHW4c->NCHW)      |
                 |           |      \        /
                 |           |       (3) add
                 |           |         |
    layout_transform          \       /
     (NCHW4c->NCHW)             \    /
                 \                mul
                  \            /
                        add

    This test verifies a case when the latest op which is visited is (3) and does not
    support textures, but there is (1) supporting textures, thus the whole func will
    support textures
    """
    input_shape = (1, 4, 40, 40)
    filter_shape1 = (4, 4, 3, 3)
    filter_shape2 = (4, 4, 3, 3)
    filter_shape3 = (4, 4, 3, 3)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W1 = relay.var("weight1", shape=filter_shape1, dtype=dtype)
    W2 = relay.var("weight2", shape=filter_shape2, dtype=dtype)
    mean = relay.mean(A, axis=1, keepdims=True)
    conv1 = relay.nn.conv2d(
        A,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[1, 1, 1, 1],
        strides=[1, 1],
        out_dtype=dtype,
        channels=4,
        kernel_size=(3, 3),
    )

    conv2 = relay.nn.conv2d(
        conv1,
        W2,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[1, 1, 1, 1],
        strides=[1, 1],
        out_dtype=dtype,
        channels=4,
        kernel_size=(3, 3),
    )

    ad3 = relay.op.add(conv1, conv2)
    ad1 = relay.op.add(mean, conv1)
    ad2 = relay.op.multiply(ad1, conv2)
    ad4 = relay.op.add(ad3, ad2)

    mod = relay.Function([A, W1, W2], ad4)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data1 = np.zeros(filter_shape1).astype(dtype)
    filter_data2 = np.zeros(filter_shape2).astype(dtype)
    initializer("weight", filter_data1)
    initializer("weight", filter_data2)
    params1 = {
        "weight1": tvm.nd.array(filter_data1),
        "weight2": tvm.nd.array(filter_data2),
    }

    static_memory_scope = [
        "",
        "global.texture",
        "global",
        "global.texture",
        "global",
        "global.texture",
        "global",
        "global",
    ]
    if executor_type == "ge":
        build_run_compare(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
            static_memory_scope,
        )
    else:
        build_run_compare_vm(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
        )


# function repeat, params scope are different in reused functions
@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_injective_nwo_inputs2(remote, target, executor_type, dtype):
    """
    Use case for verification of stability of annotation primary functions
    having several ops accepting data outside of Primary function
    The visiting of ops during traversing of graph inside primary function
    can depend on order of relay graph creation. Thus the annotation mechanism
    should be reliable for graph traversal order
    The current decision if Prim Function support textures or not depend on
    *any* op accepting input of the function and if op support textures
                                     Input
                               /                   \
                layout_transform (NCHW->NCHW4c)    |
                         |                        /
                      conv2d (1)                 /
                         |                      /
                      conv2d (2)       mean    /
                  /         \                 /   <- Primary function several head ops
             (1)add    (2)layout_transform    |
                 |        (NCHW4c->NCHW)      |
                 |           |      \        /
                 |           |       (3) add
                 |           |         |
    layout_transform          \       /
     (NCHW4c->NCHW)             \    /
                 \                mul
                  \            /
                        add

    This test verifies a case when the latest op which is (1), it supports textures
    an whole prim function is considered as a func working with textures
    """
    input_shape = (1, 4, 40, 40)
    filter_shape1 = (4, 4, 3, 3)
    filter_shape2 = (4, 4, 3, 3)
    filter_shape3 = (4, 4, 3, 3)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W1 = relay.var("weight1", shape=filter_shape1, dtype=dtype)
    W2 = relay.var("weight2", shape=filter_shape2, dtype=dtype)
    mean = relay.mean(A, axis=1, keepdims=True)
    conv1 = relay.nn.conv2d(
        A,
        W1,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[1, 1, 1, 1],
        strides=[1, 1],
        out_dtype=dtype,
        channels=4,
        kernel_size=(3, 3),
    )

    conv2 = relay.nn.conv2d(
        conv1,
        W2,
        data_layout="NCHW",
        kernel_layout="OIHW",
        padding=[1, 1, 1, 1],
        strides=[1, 1],
        out_dtype=dtype,
        channels=4,
        kernel_size=(3, 3),
    )

    ad3 = relay.op.add(conv1, conv2)
    ad1 = relay.op.add(mean, conv1)
    ad2 = relay.op.multiply(ad1, conv2)
    ad4 = relay.op.add(ad2, ad3)

    mod = relay.Function([A, W1, W2], ad4)
    np.random.seed(0)
    initializer = relay.testing.init.Xavier()
    filter_data1 = np.zeros(filter_shape1).astype(dtype)
    filter_data2 = np.zeros(filter_shape2).astype(dtype)
    initializer("weight", filter_data1)
    initializer("weight", filter_data2)
    params1 = {
        "weight1": tvm.nd.array(filter_data1),
        "weight2": tvm.nd.array(filter_data2),
    }

    static_memory_scope = [
        "",
        "global.texture",
        "global",
        "global.texture",
        "global",
        "global",
        "global.texture",
        "global",
    ]
    if executor_type == "ge":
        build_run_compare(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
            static_memory_scope,
        )
    else:
        build_run_compare_vm(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
        )


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_to_3_channels(remote, target, executor_type, dtype):
    input_shape = (1, 256, 200, 200)
    filter_shape = (3, 256, 1, 1)
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


@tvm.testing.requires_opencl
@tvm.testing.parametrize_targets("opencl -device=adreno")
def test_conv2d_weight_on_buffers(remote, target, executor_type, dtype):
    target = "opencl -device=adreno"
    input_shape = (1, 64, 75, 75)
    filter_shape = (64, 64, 3, 3)
    bias_shape = (64,)
    A = relay.var("data", shape=input_shape, dtype=dtype)
    W = relay.var("weight", shape=filter_shape, dtype=dtype)
    BS = relay.var("bias", shape=bias_shape, dtype=dtype)
    conv = relay.nn.conv2d(A, W, padding=[1, 1, 1, 1], channels=64, kernel_size=(3, 3))
    conv = relay.nn.bias_add(conv, BS)
    conv = relay.op.nn.relu(conv)

    mod = relay.Function([A, W, BS], conv)
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

    if executor_type == "ge":
        static_memory_scope = [
            "",
            "global.texture",
            "global",
            "global.texture-weight",
            "",
            "",
        ]
        build_run_compare(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
            static_memory_scope,
        )
    else:
        static_memory_scope = """
        VM VirtualDevice[0]: device type 1, id 0 and mem_scope
        VM VirtualDevice[1]: device type 4, id 0 and mem_scope
        VM VirtualDevice[2]: device type 4, id 0 and mem_scope global.texture
        VM VirtualDevice[3]: device type 4, id 0 and mem_scope global
        VM VirtualDevice[4]: device type 4, id 0 and mem_scope global.texture-weight
        """
        build_run_compare_vm(
            remote,
            mod,
            params1,
            {"data": input_shape},
            {"data": dtype},
            target,
            static_memory_scope,
        )


if __name__ == "__main__":
    tvm.testing.main()
