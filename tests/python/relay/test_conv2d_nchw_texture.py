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


@tvm.testing.requires_opencl
def test_conv2d_inceptionv3_64x35x35_96x64x3x3_nopad():
    target = "opencl --device=adreno"
    dtype = "float16"

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

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target, gpu_preprocess)


@tvm.testing.requires_opencl
def test_conv2d_inceptionv3_64x35x35_96x64x3x3_nopad_pass():
    target = "opencl --device=adreno"
    dtype = "float16"

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

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target, gpu_preprocess)


@tvm.testing.requires_opencl
def test_conv2d_inceptionv3_35_35_strides():
    target = "opencl --device=adreno"
    dtype = "float16"

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

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target, gpu_preprocess)


@tvm.testing.requires_opencl
def test_conv2d_resnet50_v2_nchw_3c():
    target = "opencl --device=adreno"
    dtype = "float16"

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

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target)


@tvm.testing.requires_opencl
def test_conv2d_inceptionv3_nchw_3c():
    target = "opencl --device=adreno"
    dtype = "float16"

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

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target)


@tvm.testing.requires_opencl
def test_conv2d_1x1_16c16spatial():
    target = "opencl --device=adreno"
    dtype = "float16"

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

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target)


@tvm.testing.requires_opencl
def test_conv2d_4x4_16c16pad():
    target = "opencl --device=adreno"
    dtype = "float16"

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

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target)


@tvm.testing.requires_opencl
def test_conv2d_4x4x4_16c16pad():
    target = "opencl --device=adreno"
    dtype = "float16"

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

    build_run_compare(mod, params1, {"data": input_shape}, dtype, target)


@tvm.testing.requires_opencl
def test_conv2d_yolov3_v2_nchw_3c():
    target = "opencl --device=adreno"
    dtype = "float16"

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

    build_run_compare(mod, params, {"data": input_shape}, dtype, target)


@tvm.testing.requires_opencl
def test_conv2d_vgg16_winograd_4d():
    target = "opencl --device=adreno"
    dtype = "float16"

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
            '{"input": ["opencl -keys=adreno,opencl,gpu -device=adreno -max_num_threads=256", "conv2d_nchw_winograd_acc32.image2d", [["TENSOR", [1, 512, 28, 28], "float16"], ["TENSOR", [512, 512, 3, 3], "float16"], [1, 1], [1, 1, 1, 1], [1, 1], "float16"], {}], "config": {"index": 1591, "code_hash": null, "entity": [["auto_unroll_max_step", "ot", 4], ["tile_y", "sp", [-1, 1, 32]], ["tile_x", "sp", [-1, 4, 2]], ["tile_rc", "sp", [-1, 8]]]}, "result": [[0.0037244], 0, 7.06374192237854, 1653898629.7427933], "version": 0.2, "tvm_version": "0.8.dev0"}\n'
        )
    graph = build_run_compare(
        mod, params1, {"data": input_shape}, dtype, target, stat_file=stat_file
    )
    matches = re.findall("winograd", graph)
    assert len(matches) > 0


@tvm.testing.requires_opencl
def test_conv2d_winograd_conv():
    target = "opencl --device=adreno"
    dtype = "float16"

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
            '{"input": ["opencl -keys=adreno,opencl,gpu -device=adreno -max_num_threads=256", "conv2d_nchw_winograd_acc32.image2d", [["TENSOR", [1, 4, 3, 3], "float16"], ["TENSOR", [8, 4, 3, 3], "float16"], [1, 1], [1, 1, 1, 1], [1, 1], "float16"], {}], "config": {"index": 1591, "code_hash": null, "entity": [["auto_unroll_max_step", "ot", 4], ["tile_y", "sp", [-1, 1, 32]], ["tile_x", "sp", [-1, 4, 2]], ["tile_rc", "sp", [-1, 8]]]}, "result": [[0.0037244], 0, 7.06374192237854, 1653898629.7427933], "version": 0.2, "tvm_version": "0.8.dev0"}\n'
        )
    graph = build_run_compare(
        mod, params1, {"data": input_shape}, dtype, target, stat_file=stat_file
    )
    matches = re.findall("winograd", graph)
    assert len(matches) > 0
