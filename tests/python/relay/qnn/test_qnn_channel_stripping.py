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
from tvm.testing.aot import generate_ref_data

from tvm.topi.arm_cpu.qnn_legalize import legalize_bias_add


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def execute_relay_func(relay_func, in_data):
    ref_module = tvm.IRModule.from_expr(relay_func)
    return generate_ref_data(ref_module, {"input": in_data})["output"]


def tvm_const(obj):
    return relay.Constant(nd.array(obj))


def make_test_conv_depthwise_conv():
    """Generates a convolution -> depthwise_convolution -> convolution pattern that can have
    channels stripped. The structure here mirrors MobileNetV1's layers 8-10."""

    input_var = relay.var("input", shape=(1, 12, 12, 4), dtype="int8")

    kernel_1 = np.array(
        [[0, 1, 0, -2], [0, 3, 0, 5], [0, 5, 0, -9], [0, 2, 0, 21]], dtype="int8"
    ).reshape((1, 1, 4, 4))
    input_scale_1 = np.float32(0.5)
    output_scale_1 = np.array([0.5, 2.0, 0.25, 4.0], dtype="float32")

    out = relay.qnn.conv2d(
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
    out = relay.qnn.requantize(
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
    out = relay.qnn.conv2d(
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
    out = relay.qnn.requantize(
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

    out = relay.qnn.conv2d(
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


def make_test_conv_pool_dense():
    """Generates a convolution -> pool -> dense pattern that can have channels stripped. The
    structure here mirrors MobileNetV1's final few layers."""

    input_var = relay.var("input", shape=(1, 3, 3, 4), dtype="int8")

    kernel = np.array(
        [[0, 1, 0, -2], [0, 3, 0, 5], [0, 5, 0, -9], [0, 2, 0, 21]], dtype="int8"
    ).reshape((1, 1, 4, 4))
    input_scale = np.float32(0.029626124)
    output_scale = np.array([0.5, 2.0, 0.25, 4.0], dtype="float32")

    out = relay.qnn.conv2d(
        input_var,
        tvm_const(kernel),
        tvm_const(np.int32(-128)),
        tvm_const(np.int32(0)),
        tvm_const(input_scale),
        tvm_const(output_scale),
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

    out = relay.qnn.requantize(
        out,
        tvm_const(input_scale * output_scale),
        tvm_const(np.int32(0)),
        tvm_const(np.float32(0.015656913)),
        tvm_const(np.int32(-128)),
        axis=3,
        out_dtype="int8",
    )

    out = relay.cast(out, dtype="int32")
    out = relay.nn.avg_pool2d(
        out,
        pool_size=[3, 3],
        strides=[3, 3],
        layout="NHWC",
    )

    out = relay.cast(out, dtype="int8")
    # The channel stripping logic expects two reshape operators
    out = relay.reshape(out, newshape=[-1, 4])
    out = relay.reshape(out, newshape=[-1, 4])

    dense_weights = np.array([[15, -2, -3, 11], [12, -10, 13, -10]], dtype="int8")
    out = relay.qnn.dense(
        out,
        tvm_const(dense_weights),
        tvm_const(np.int32(-128)),
        tvm_const(np.int32(0)),
        tvm_const(np.float32(0.015656913)),
        tvm_const(np.float32(0.0047202893)),
        units=2,
        out_dtype="int32",
    )

    dense_bias = np.array([1463, -1463], dtype="int32")
    out = relay.nn.bias_add(
        out,
        tvm_const(dense_bias),
        axis=1,
    )

    return relay.Function([input_var], out)


def test_conv_depthwise_conv():
    """Make sure that qnn_legalize.py is able to detect and remove empty output channels from a
    convolution -> depthwise convolution -> convolution pattern by folding into a bias_add op."""

    original = make_test_conv_depthwise_conv()

    with TempOpAttr("nn.bias_add", "FTVMLegalize", legalize_bias_add):
        unoptimized = run_opt_pass(original, transform.InferType())
        optimized = run_opt_pass(original, transform.Legalize())

    # Inputs and outputs should be unmodified by channel stripping
    assert unoptimized.checked_type == optimized.checked_type

    # Make sure 2/4 channels were removed by channel stripping
    assert tuple(unoptimized.body.args[0].args[0].checked_type.shape) == (1, 12, 12, 4)
    assert tuple(optimized.body.args[0].args[0].checked_type.shape) == (1, 12, 12, 2)

    # Make sure optimized and unoptimized versions behave identically
    np.random.seed(12402)  # Fix seed for repeatability
    input_data = np.random.randint(-128, 128, size=(1, 12, 12, 4), dtype="int8")

    unoptimized_output = execute_relay_func(unoptimized, np.copy(input_data))
    optimized_output = execute_relay_func(optimized, np.copy(input_data))
    np.testing.assert_array_equal(unoptimized_output, optimized_output)


def test_conv_pool_dense():
    """Make sure that qnn_legalize.py is able to detect and remove empty output channels from a
    convolution -> avg_pool2d -> dense pattern by folding them into a bias_add op."""

    original = make_test_conv_pool_dense()

    with TempOpAttr("nn.bias_add", "FTVMLegalize", legalize_bias_add):
        unoptimized = run_opt_pass(original, transform.InferType())
        optimized = run_opt_pass(original, transform.Legalize())

    # Inputs and outputs should be unmodified by channel stripping
    assert unoptimized.checked_type == optimized.checked_type

    # Make sure 2/4 channels were removed by channel stripping
    assert tuple(unoptimized.body.args[0].args[0].checked_type.shape) == (1, 4)
    assert tuple(optimized.body.args[0].args[0].checked_type.shape) == (1, 2)

    # Make sure optimized and unoptimized versions behave identically
    np.random.seed(12402)  # Fix seed for repeatability
    input_data = np.random.randint(-128, 128, size=(1, 3, 3, 4), dtype="int8")

    unoptimized_output = execute_relay_func(unoptimized, np.copy(input_data))
    optimized_output = execute_relay_func(optimized, np.copy(input_data))
    np.testing.assert_array_equal(unoptimized_output, optimized_output)
