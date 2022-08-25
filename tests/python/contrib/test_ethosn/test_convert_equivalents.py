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

"""Unit tests for the convert equivalents pass."""

import pytest
import numpy as np

import tvm
from tvm import relay
from tvm.testing import requires_ethosn
from tvm.relay.op.contrib.ethosn import ConvertEquivalents

from . import infrastructure as tei


def _assert_structural_equal(a, b):
    """Check structural equality of two Relay expressions."""
    reason = (
        "Actual and expected relay functions are not equal. "
        "ConvertEquivalents is not correctly transforming the input "
        "graph."
    )
    assert tvm.ir.structural_equal(a, b), reason


def _create_npu_module(inputs, expr, composite_name, ext_func_name):
    """Wraps an operator as an NPU module."""
    gen_vars = lambda prefix, vars: [
        relay.var(
            prefix + var.name_hint, shape=var.type_annotation.shape, dtype=var.type_annotation.dtype
        )
        for var in vars
    ]

    mod = tvm.ir.IRModule()

    func = relay.Function(relay.analysis.free_vars(expr), expr)
    func = func.with_attr("Composite", composite_name)
    inner_vars = gen_vars("inner_", inputs)
    call = relay.Call(func, inner_vars)

    func2 = relay.Function(relay.analysis.free_vars(call), call)
    func2 = func2.with_attr("Compiler", "ethos-n")
    func2 = func2.with_attr("global_symbol", ext_func_name)
    mod[ext_func_name] = func2
    mod = relay.transform.InferType()(mod)

    outer_vars = gen_vars("outer_", inputs)
    out = relay.Call(mod.get_global_var(ext_func_name), outer_vars)
    mod["main"] = relay.Function(relay.analysis.free_vars(out), out)
    mod = relay.transform.InferType()(mod)
    return mod


@requires_ethosn
@pytest.mark.parametrize("dtype", ["uint8", "int8"])
@pytest.mark.parametrize("shape,channels", [((1, 4, 4, 8), 8), ((1, 16, 12, 4), 4)])
@pytest.mark.parametrize("reverse_inputs", [True, False])
def test_multiply_to_depthwise(dtype, shape, channels, reverse_inputs):
    """Check that multiply is correctly converted to a depthwise operation."""
    np.random.seed(0)

    iinfo = np.iinfo(dtype)
    data_min = iinfo.min
    data_max = iinfo.max
    input_zp = np.random.randint(data_min, data_max)
    input_sc = np.random.random() * 2
    input2_zp = np.random.randint(data_min, data_max)
    input2_sc = np.random.random() * 2
    output_zp, output_sc = tei.get_conv2d_qnn_params(
        dtype, input_zp, input_sc, input2_zp, input2_sc, 1, 1, shape[3]
    )
    x = relay.var("x", shape=shape, dtype=dtype)
    constant_shape = (1, 1, 1, channels)
    y_data = np.random.randint(data_min, data_max + 1, size=constant_shape, dtype=dtype)

    def before():
        y = relay.const(y_data, dtype=dtype)
        expr = relay.qnn.op.mul(
            y if reverse_inputs else x,
            x if reverse_inputs else y,
            relay.const(input_sc, "float32"),
            relay.const(input_zp, "int32"),
            relay.const(input2_sc, "float32"),
            relay.const(input2_zp, "int32"),
            relay.const(output_sc, "float32"),
            relay.const(output_zp, "int32"),
        )
        return _create_npu_module([x], expr, "ethos-n.qnn_mul", "ext_func")

    def expected():
        constant_shape_hwoi = (1, 1, channels, 1)
        y_data_hwoi = y_data.reshape(constant_shape_hwoi)
        y_hwoi = relay.const(y_data_hwoi, dtype=dtype)
        expr = relay.qnn.op.conv2d(
            x,
            y_hwoi,
            relay.const(input2_zp if reverse_inputs else input_zp, "int32"),
            relay.const(input_zp if reverse_inputs else input2_zp, "int32"),
            relay.const(input2_sc if reverse_inputs else input_sc, "float32"),
            relay.const(input_sc if reverse_inputs else input2_sc, "float32"),
            (1, 1),
            channels,
            (1, 1),
            (0, 0),
            (1, 1),
            channels,
            "NHWC",
            "HWOI",
            "NHWC",
            "int32",
        )
        expr = relay.nn.bias_add(expr, relay.const(np.zeros((channels,), dtype="int32")), axis=3)
        expr = relay.qnn.op.requantize(
            expr,
            relay.const(input2_sc if reverse_inputs else input_sc, "float32"),
            relay.const(input2_zp if reverse_inputs else input_zp, "int32"),
            relay.const(output_sc, "float32"),
            relay.const(output_zp, "int32"),
            out_dtype=dtype,
        )
        return _create_npu_module([x], expr, "ethos-n.qnn_conv2d", "ext_func")

    mod = before()
    mod = ConvertEquivalents()(mod)
    expected_mod = expected()
    _assert_structural_equal(mod["ext_func"], expected_mod["ext_func"])
