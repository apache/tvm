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
"""Test QNN channel stripping legalization pass."""

import numpy as np
import tvm
from tvm import nd, relay

from tvm.relay import transform
from tvm.relay.testing.temp_op_attr import TempOpAttr

from tvm.topi.arm_cpu.qnn_legalize import legalize_bias_add


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def tvm_const(obj):
    return relay.Constant(nd.array(obj))


def make_test_conv_depthwise_conv():
    input_var = relay.var("x", shape=(1, 12, 12, 4), dtype="int8")

    kernel_1 = np.array(
        [[0, 1, 0, -2], [0, 3, 0, 5], [0, 5, 0, -9], [0, 2, 0, 21]], dtype="int8"
    ).reshape((1, 1, 4, 4))
    input_scale_1 = np.float32(0.5)
    output_scale_1 = np.array([0.5, 2.0, 0.25, 4.0], dtype="float32")

    out = relay.qnn.op.conv2d(
        input_var,
        tvm_const(kernel_1),
        tvm_const(np.int32(-128)),
        tvm_const(np.int32(0)),
        tvm_const(input_scale_1),
        tvm_const(output_scale_1),
        channels=4,
        kernel_size=(1, 1),
        padding=(0, 0),
        data_layout="NHWC",
        kernel_layout="HWIO",
    )

    bias_1 = np.array([198, -2, 19, 10], dtype="int32")
    out = relay.nn.bias_add(
        out,
        tvm_const(bias_1),
        axis=3,
    )

    input_scale_2 = np.float32(0.25)
    out = relay.qnn.op.requantize(
        out,
        tvm_const(input_scale_1 * output_scale_1),
        tvm_const(np.int32(0)),
        tvm_const(input_scale_2),
        tvm_const(np.int32(-128)),
        axis=3,
        out_dtype="int8",
    )
    # Outputs here will be fixed to {0: 70, 2: -118}

    kernel_2 = np.array(
        [
            [0, 6, 4, 2],
            [8, 6, -3, -1],
            [-2, -5, 3, -8],
            [-7, 5, 1, 9],
            [-4, -9, -8, -2],
            [-1, 4, -5, 3],
            [-4, -9, 2, 6],
            [9, -6, 0, 5],
            [-3, 8, 1, -7],
        ],
        dtype="int8",
    ).reshape((3, 3, 4, 1))
    output_scale_2 = np.array([0.25, 0.125, 2.0, 0.125], dtype="float32")
    out = relay.qnn.op.conv2d(
        out,
        tvm_const(kernel_2),
        tvm_const(np.int32(-128)),
        tvm_const(np.int32(0)),
        tvm_const(input_scale_2),
        tvm_const(output_scale_2),
        channels=4,
        groups=4,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWOI",
    )

    bias_2 = np.array([4582, 4, -12, 15], dtype="int32")
    out = relay.nn.bias_add(
        out,
        tvm_const(bias_2),
        axis=3,
    )

    input_scale_3 = np.float32(0.125)
    out = relay.qnn.op.requantize(
        out,
        tvm_const(input_scale_2 * output_scale_2),
        tvm_const(np.int32(0)),
        tvm_const(input_scale_3),
        tvm_const(np.int32(-128)),
        axis=3,
        out_dtype="int8",
    )
    # Outputs here will be fixed to {0: 127, 2: -128}

    kernel_3 = np.array(
        [[4, -2, 9, 9], [0, 0, 0, 0], [0, 0, 0, 0], [-1, 1, -1, 1]], dtype="int8"
    ).reshape((1, 1, 4, 4))
    output_scale_3 = np.array([0.25, 0.125, 1.0, 0.5], dtype="float32")

    out = relay.qnn.op.conv2d(
        out,
        tvm_const(kernel_3),
        tvm_const(np.int32(-128)),
        tvm_const(np.int32(0)),
        tvm_const(input_scale_3),
        tvm_const(output_scale_3),
        channels=4,
        kernel_size=(1, 1),
        padding=(0, 0),
        data_layout="NHWC",
        kernel_layout="HWIO",
    )

    bias_3 = np.array([1, -1, 4, 6], dtype="int32")
    out = relay.nn.bias_add(
        out,
        tvm_const(bias_3),
        axis=3,
    )

    return relay.Function([input_var], out)


def make_expected_conv_depthwise_conv():
    input_var = relay.var("x", shape=(1, 12, 12, 4), dtype="int8")

    kernel_1 = np.array([[1, -2], [3, 5], [5, -9], [2, 21]], dtype="int8").reshape((1, 1, 4, 2))
    input_scale_1 = np.float32(0.5)
    output_scale_1 = np.array([2.0, 4.0], dtype="float32")

    out = relay.qnn.op.conv2d(
        input_var,
        tvm_const(kernel_1),
        tvm_const(np.int32(-128)),
        tvm_const(np.int32(0)),
        tvm_const(input_scale_1),
        tvm_const(output_scale_1),
        channels=2,
        kernel_size=(1, 1),
        padding=(0, 0),
        data_layout="NHWC",
        kernel_layout="HWIO",
    )

    bias_1 = np.array([-2, 10], dtype="int32")
    out = relay.nn.bias_add(
        out,
        tvm_const(bias_1),
        axis=3,
    )

    input_scale_2 = np.float32(0.25)
    out = relay.qnn.op.requantize(
        out,
        tvm_const(input_scale_1 * output_scale_1),
        tvm_const(np.int32(0)),
        tvm_const(input_scale_2),
        tvm_const(np.int32(-128)),
        axis=3,
        out_dtype="int8",
    )
    # Outputs here will be fixed to {0: 70, 2: -118}

    kernel_2 = np.array(
        [
            [6, 2],
            [6, -1],
            [-5, -8],
            [5, 9],
            [-9, -2],
            [4, 3],
            [-9, 6],
            [-6, 5],
            [8, -7],
        ],
        dtype="int8",
    ).reshape((3, 3, 2, 1))
    output_scale_2 = np.array([0.125, 0.125], dtype="float32")
    out = relay.qnn.op.conv2d(
        out,
        tvm_const(kernel_2),
        tvm_const(np.int32(-128)),
        tvm_const(np.int32(0)),
        tvm_const(input_scale_2),
        tvm_const(output_scale_2),
        channels=2,
        groups=2,
        kernel_size=(3, 3),
        padding=(1, 1),
        data_layout="NHWC",
        kernel_layout="HWOI",
    )

    bias_2 = np.array([4, 15], dtype="int32")
    out = relay.nn.bias_add(
        out,
        tvm_const(bias_2),
        axis=3,
    )

    input_scale_3 = np.float32(0.125)
    out = relay.qnn.op.requantize(
        out,
        tvm_const(input_scale_2 * output_scale_2),
        tvm_const(np.int32(0)),
        tvm_const(input_scale_3),
        tvm_const(np.int32(-128)),
        axis=3,
        out_dtype="int8",
    )
    # Outputs here will be fixed to {0: 127, 2: -128}

    kernel_3 = np.array([[0, 0, 0, 0], [-1, 1, -1, 1]], dtype="int8").reshape((1, 1, 2, 4))
    output_scale_3 = np.array([0.25, 0.125, 1.0, 0.5], dtype="float32")

    out = relay.qnn.op.conv2d(
        out,
        tvm_const(kernel_3),
        tvm_const(np.int32(-128)),
        tvm_const(np.int32(0)),
        tvm_const(input_scale_3),
        tvm_const(output_scale_3),
        channels=4,
        kernel_size=(1, 1),
        padding=(0, 0),
        data_layout="NHWC",
        kernel_layout="HWIO",
    )

    bias_3 = np.array([1, -1, 4, 6], dtype="int32")
    out = relay.nn.bias_add(
        out,
        tvm_const(bias_3),
        axis=3,
    )

    return relay.Function([input_var], out)


def test_cdc_channel_stripping_matches_expected():
    with TempOpAttr("nn.bias_add", "FTVMLegalize", legalize_bias_add):
        before = make_test_conv_depthwise_conv()
        actual = run_opt_pass(before, transform.Legalize())
        expected = run_opt_pass(make_expected_conv_depthwise_conv(), transform.InferType())
    tvm.ir.assert_structural_equal(actual, expected)
