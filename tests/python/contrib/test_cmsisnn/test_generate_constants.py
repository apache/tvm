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

"""CMSIS-NN integration tests: generate_constants pass"""
import math
import numpy as np
import pytest
import tvm
from tvm.testing.aot import get_dtype_range
from tvm import relay
from tvm.relay.op.contrib import cmsisnn

from .utils import (
    make_module,
    get_same_padding,
    get_conv2d_qnn_params,
    make_qnn_relu,
)

tvm._ffi._init_api("relay.ext.cmsisnn.transform", __name__)


def quantize_scale(scale):
    multiplier, shift = math.frexp(scale)
    multiplier_q31 = round(multiplier * (1 << 31))
    return multiplier_q31, shift


class CheckGeneratedConstants(tvm.relay.ExprVisitor):
    """Provides methods to compare against expected quantization parameters"""

    def __init__(self, enable_bias, multiplier, shift):
        super().__init__()
        self.num_constant_args_ = 0
        self.enable_bias_ = enable_bias
        self.multiplier_ = multiplier
        self.shift_ = shift

    def visit_call(self, call):
        """Tests if the multiplier and shift constants required by CMSIS-NN API were generated"""
        super().visit_call(call)
        if isinstance(call.op, tvm.ir.expr.GlobalVar):
            multiplier = call.args[2]
            shift = call.args[6] if self.enable_bias_ else call.args[5]
            assert isinstance(
                multiplier, relay.expr.Constant
            ), "Expected quantized multiplier at argument#3"
            assert isinstance(
                shift, relay.expr.Constant
            ), "Expected a constant while looking for quantized shift"
            multiplier = multiplier.data.numpy()
            shift = shift.data.numpy()
            tvm.testing.assert_allclose(multiplier, self.multiplier_, atol=100, rtol=1e-10)
            tvm.testing.assert_allclose(shift, self.shift_, atol=1, rtol=1e-5)


def make_model(
    shape,
    kernel_shape,
    input_zero_point,
    input_scale,
    kernel_zero_point,
    kernel_scale,
    output_zero_point,
    output_scale,
    padding,
    strides,
    dilation,
    groups,
    dtype,
    kernel_dtype,
    out_channels,
    weight_format,
    enable_bias,
    relu_type,
):
    """Return a model and any parameters it may have"""
    h_index = weight_format.index("H")
    w_index = weight_format.index("W")
    kernel_h = kernel_shape[h_index]
    kernel_w = kernel_shape[w_index]
    a = relay.var("input", shape=shape, dtype=dtype)
    p = (0, 0, 0, 0)
    if padding == "SAME":
        p = get_same_padding((shape[1], shape[2]), (kernel_h, kernel_w), dilation, strides)
        a = relay.nn.pad(
            a,
            pad_width=[(0, 0), (p[0], p[2]), (p[1], p[3]), (0, 0)],
            pad_value=input_zero_point,
            pad_mode="constant",
        )
        shape = (shape[0], shape[1] + p[0] + p[2], shape[2] + p[1] + p[3], shape[3])

    weight_shape = (kernel_h, kernel_w, shape[3] // groups, out_channels)
    rng = np.random.default_rng(12321)
    kmin, kmax = get_dtype_range(kernel_dtype)
    weight = tvm.nd.array(
        rng.integers(
            kmin,
            high=kmax,
            size=weight_shape,
            dtype=kernel_dtype,
        )
    )
    weight_const = relay.const(weight, kernel_dtype)
    conv = relay.qnn.op.conv2d(
        a,
        weight_const,
        input_zero_point=relay.const(input_zero_point, "int32"),
        kernel_zero_point=relay.const(kernel_zero_point, "int32"),
        input_scale=relay.const(input_scale, "float32"),
        kernel_scale=relay.const(kernel_scale, "float32"),
        kernel_size=(kernel_h, kernel_w),
        data_layout="NHWC",
        kernel_layout=weight_format,
        dilation=dilation,
        strides=strides,
        groups=groups,
        channels=out_channels,
        padding=p,
        out_dtype="int32",
    )
    bias = tvm.nd.array(rng.integers(0, high=10, size=(out_channels,), dtype="int32"))
    bias_const = relay.const(bias, "int32")
    last_op = relay.nn.bias_add(conv, bias_const, axis=3) if enable_bias else conv
    requant_input_sc = [sc * input_scale for sc in kernel_scale]
    last_op = relay.qnn.op.requantize(
        last_op,
        relay.const(requant_input_sc, "float32"),
        relay.const(0, "int32"),
        relay.const(output_scale, "float32"),
        relay.const(output_zero_point, "int32"),
        out_dtype=dtype,
    )
    last_op = make_qnn_relu(last_op, relu_type, output_scale, output_zero_point, dtype)
    params = {"w": weight, "b": bias}
    return last_op, params


@tvm.testing.requires_cmsisnn
@pytest.mark.parametrize("enable_bias", [True, False])
@pytest.mark.parametrize(
    "input_zero_point, input_scale, kernel_scale, out_channels",
    [(10, 0.0128, [0.11, 0.22], 2), (-64, 1, [1, 0.0256, 1.37], 3)],
)
def test_op_int8(
    enable_bias,
    input_zero_point,
    input_scale,
    kernel_scale,
    out_channels,
):
    """Tests for CMSIS-NN constants when the dtype is int8"""
    ifm_shape = (1, 28, 28, 3)
    padding = "VALID"
    strides = (1, 1)
    dilation = (1, 1)
    kernel_size = (3, 3)
    kernel_zero_point = 0
    groups = 1
    weight_format = "HWIO"
    kernel_h = kernel_size[0]
    kernel_w = kernel_size[1]
    dtype = "int8"
    relu_type = "RELU"

    weight_shape = (kernel_h, kernel_w, ifm_shape[3] // groups, out_channels)

    output_scale, output_zero_point = get_conv2d_qnn_params(
        weight_shape,
        input_scale,
        input_zero_point,
        kernel_scale,
        kernel_zero_point,
        dtype,
        dtype,
        dtype,
        False,
    )

    model, params = make_model(
        ifm_shape,
        weight_shape,
        input_zero_point,
        input_scale,
        kernel_zero_point,
        kernel_scale,
        output_zero_point,
        output_scale,
        padding,
        strides,
        dilation,
        groups,
        dtype,
        dtype,
        out_channels,
        weight_format,
        enable_bias,
        relu_type,
    )
    mod = make_module(model)

    cmsisnn_mod = cmsisnn.partition_for_cmsisnn(mod, params)
    multiplier_array = []
    shift_array = []
    for i in range(out_channels):
        multiplier, shift = quantize_scale(input_scale * kernel_scale[i] / output_scale)
        multiplier_array.append(multiplier)
        shift_array.append(shift)
    CheckGeneratedConstants(enable_bias, multiplier_array, shift_array).visit_function(
        cmsisnn_mod["main"]
    )


if __name__ == "__main__":
    tvm.testing.main()
