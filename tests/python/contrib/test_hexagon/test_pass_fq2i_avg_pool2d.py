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

# pylint: disable=redefined-outer-name

""" Tests for avg_pool2d fake quantization to integer """

import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relay
from tvm.contrib.hexagon.session import Session
from tvm.contrib.hexagon.pytest_plugin import HEXAGON_AOT_LLVM_TARGET

from .infrastructure import quantize_np, build_module, run_module


def _make_avgpool_conv2d():
    """Test case with avg_pool2d followed by a conv2d"""
    dtype = "int8"
    shape_x = [1, 2, 9, 9]
    shape_w = [1, 2, 3, 3]
    kernel = [3, 3]
    stride = [1, 1]
    dilation = [1, 1]
    inp = relay.var("input", shape=shape_x, dtype=dtype)
    wgt = relay.var("weight", shape=shape_w, dtype=dtype)

    x_np = np.random.random(shape_x)
    w_np = np.random.random(shape_w)

    fp_avg = tvm.topi.testing.poolnd_python(
        x_np,
        kernel,
        stride,
        dilation,
        padding_before=[0, 0],
        padding_after=[0, 0],
        pool_type="avg",
    )
    fp_output = tvm.topi.testing.conv2d_nchw_python(
        fp_avg,
        w_np,
        [1, 1],
        [0, 0],
    )

    # Computing quantization parameters
    input_quant, input_scale, input_zero_point = quantize_np(x_np, dtype)
    weight_quant, weight_scale, weight_zero_point = quantize_np(w_np, dtype)
    _, output_scale, output_zero_point = quantize_np(fp_output, dtype)

    inp_zp = relay.const(input_zero_point)
    inp_sc = relay.const(input_scale)
    wgt_zp = relay.const(weight_zero_point)
    wgt_sc = relay.const(weight_scale)
    out_zp = relay.const(output_zero_point)
    out_sc = relay.const(output_scale)

    # Tested expression.
    op0 = relay.qnn.op.dequantize(inp, inp_sc, inp_zp)
    op1 = relay.op.nn.avg_pool2d(op0, kernel)
    op2 = relay.qnn.op.dequantize(wgt, wgt_sc, wgt_zp)
    op3 = relay.op.nn.conv2d(op1, op2, kernel_size=kernel)
    expr = relay.qnn.op.quantize(op3, out_sc, out_zp, out_dtype=dtype)
    expr = relay.qnn.op.dequantize(expr, out_sc, out_zp)
    args = {"input": input_quant, "weight": weight_quant}

    # Expected graph
    op0 = relay.qnn.op.avg_pool2d(
        inp,
        input_scale=inp_sc,
        input_zero_point=inp_zp,
        output_scale=inp_sc,
        output_zero_point=inp_zp,
        pool_size=kernel,
        strides=stride,
        dilation=dilation,
        padding=[0, 0, 0, 0],
        layout="NCHW",
        count_include_pad=False,
    )
    op1 = relay.qnn.op.conv2d(
        op0,
        wgt,
        input_scale=inp_sc,
        input_zero_point=inp_zp,
        kernel_scale=wgt_sc,
        kernel_zero_point=wgt_zp,
        kernel_size=kernel,
        channels=None,
    )
    op2 = relay.qnn.op.requantize(
        op1,
        input_scale=relay.const(input_scale * weight_scale),
        input_zero_point=relay.const(0),
        output_scale=out_sc,
        output_zero_point=out_zp,
        axis=1,
        out_dtype="int8",
    )
    ref_expr = relay.qnn.op.dequantize(op2, out_sc, out_zp)

    return expr, args, ref_expr


def _make_avgpool_avgpool():
    """Test case with avg_pool2d followed by an avg_pool2d"""
    dtype = "uint8"
    shape_x = [1, 2, 9, 9]
    kernel = [3, 3]
    stride = [1, 1]
    dilation = [1, 1]
    inp = relay.var("input", shape=shape_x, dtype=dtype)
    x_np = np.random.random(shape_x)

    fp_avg = tvm.topi.testing.poolnd_python(
        x_np,
        kernel,
        stride,
        dilation,
        padding_before=[0, 0],
        padding_after=[0, 0],
        pool_type="avg",
    )
    fp_output = tvm.topi.testing.poolnd_python(
        fp_avg,
        kernel,
        stride,
        dilation,
        padding_before=[0, 0],
        padding_after=[0, 0],
        pool_type="avg",
    )

    # Computing quantization parameters
    input_quant, input_scale, input_zero_point = quantize_np(x_np, dtype)
    _, output_scale, output_zero_point = quantize_np(fp_output, dtype)

    inp_zp = relay.const(input_zero_point)
    inp_sc = relay.const(input_scale)
    out_zp = relay.const(output_zero_point)
    out_sc = relay.const(output_scale)

    # Tested expression.
    op0 = relay.qnn.op.dequantize(inp, inp_sc, inp_zp)
    op1 = relay.op.nn.avg_pool2d(op0, kernel)
    op2 = relay.op.nn.avg_pool2d(op1, kernel)
    expr = relay.qnn.op.quantize(op2, out_sc, out_zp, out_dtype=dtype)
    expr = relay.qnn.op.dequantize(expr, out_sc, out_zp)
    args = {"input": input_quant}

    # Expected graph
    op0 = relay.qnn.op.avg_pool2d(
        inp,
        input_scale=inp_sc,
        input_zero_point=inp_zp,
        output_scale=inp_sc,
        output_zero_point=inp_zp,
        pool_size=kernel,
        strides=stride,
        dilation=dilation,
        padding=[0, 0, 0, 0],
        layout="NCHW",
        count_include_pad=False,
    )
    op1 = relay.qnn.op.avg_pool2d(
        op0,
        input_scale=inp_sc,
        input_zero_point=inp_zp,
        output_scale=out_sc,
        output_zero_point=out_zp,
        pool_size=kernel,
        strides=stride,
        dilation=dilation,
        padding=[0, 0, 0, 0],
        layout="NCHW",
        count_include_pad=False,
    )
    ref_expr = relay.qnn.op.dequantize(op1, out_sc, out_zp)

    return expr, args, ref_expr


def _make_avgpool():
    dtype = "int8"
    shape_x = [1, 2, 9, 9]
    kernel = [3, 3]
    stride = [1, 1]
    dilation = [1, 1]
    inp = relay.var("input", shape=shape_x, dtype=dtype)
    x_np = np.random.random(shape_x)

    fp_output = tvm.topi.testing.poolnd_python(
        x_np,
        kernel,
        stride,
        dilation,
        padding_before=[0, 0],
        padding_after=[0, 0],
        pool_type="avg",
    )

    # Computing quantization parameters
    input_quant, input_scale, input_zero_point = quantize_np(x_np, dtype)
    _, output_scale, output_zero_point = quantize_np(fp_output, dtype)

    inp_zp = relay.const(input_zero_point)
    inp_sc = relay.const(input_scale)
    out_zp = relay.const(output_zero_point)
    out_sc = relay.const(output_scale)

    # Tested expression
    op0 = relay.qnn.op.dequantize(inp, inp_sc, inp_zp)
    op1 = relay.op.nn.avg_pool2d(op0, kernel)
    expr = relay.qnn.op.quantize(op1, out_sc, out_zp, out_dtype=dtype)
    expr = relay.qnn.op.dequantize(expr, out_sc, out_zp)
    args = {"input": input_quant}

    # Expected graph
    op = relay.qnn.op.avg_pool2d(
        inp,
        input_scale=inp_sc,
        input_zero_point=inp_zp,
        output_scale=out_sc,
        output_zero_point=out_zp,
        pool_size=kernel,
        strides=stride,
        dilation=dilation,
        padding=[0, 0, 0, 0],
        layout="NCHW",
        count_include_pad=False,
    )
    ref_expr = relay.qnn.op.dequantize(op, out_sc, out_zp)

    return expr, args, ref_expr


def compare_graphs(expr, ref_expr):
    """Compares the given graph with the expected graph"""
    mod = tvm.IRModule.from_expr(expr)
    mod = tvm.relay.transform.InferType()(mod)
    mod_int = tvm.relay.transform.FakeQuantizationToInteger()(mod)
    ref_mod = tvm.IRModule.from_expr(ref_expr)
    ref_mod = tvm.relay.transform.InferType()(ref_mod)
    tvm.ir.assert_structural_equal(mod_int["main"], ref_mod["main"], map_free_vars=True)


def compare_fq_to_int(hexagon_session, expr, inputs):
    """Compares the float module output with the integer module output"""
    mod = tvm.IRModule.from_expr(expr)
    mod = tvm.relay.transform.InferType()(mod)
    mod_int = tvm.relay.transform.FakeQuantizationToInteger()(mod)
    assert not tvm.ir.structural_equal(mod, mod_int)

    mod = build_module(
        mod, tvm.target.Target(HEXAGON_AOT_LLVM_TARGET, host=HEXAGON_AOT_LLVM_TARGET)
    )
    mod_int = build_module(
        mod_int, tvm.target.Target(HEXAGON_AOT_LLVM_TARGET, host=HEXAGON_AOT_LLVM_TARGET)
    )

    hexagon_mod = hexagon_session.get_executor_from_factory(mod)
    result = run_module(hexagon_mod, inputs)

    hexagon_mod = hexagon_session.get_executor_from_factory(mod_int)
    result_int = run_module(hexagon_mod, inputs)

    tvm.testing.assert_allclose(result, result_int, rtol=1e-02, atol=1e-02)


avgpool_test_case = tvm.testing.parameter(
    _make_avgpool,
    _make_avgpool_avgpool,
    pytest.param(
        _make_avgpool_conv2d,
        marks=pytest.mark.xfail(
            reason="Rounding differences causing mismatch of Constant, difference around 10^-7"
        ),
    ),
)


@tvm.testing.requires_hexagon
def test_execution(hexagon_session: Session, avgpool_test_case):
    expr, args, _ = avgpool_test_case()
    compare_fq_to_int(hexagon_session, expr, args)


def test_quantization(avgpool_test_case):
    expr, _, ref_expr = avgpool_test_case()
    compare_graphs(expr, ref_expr)


if __name__ == "__main__":
    tvm.testing.main()
