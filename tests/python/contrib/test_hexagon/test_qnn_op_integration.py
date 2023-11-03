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
# pylint: disable=invalid-name

"""Tests for QNN operations on Hexagon"""

import numpy as np

import tvm.testing
import tvm.topi.testing
from tvm import relay
from tvm.contrib.hexagon.session import Session
from tvm.contrib.hexagon.pytest_plugin import HEXAGON_AOT_LLVM_TARGET
from tvm.relay.backend import Executor
from tvm.relay.testing import run_opt_pass, run_infer_type

from .infrastructure import quantize_np


@tvm.testing.requires_hexagon
def test_disable_qnn_legalize_pass():
    """No QNN pass test."""
    x = relay.var("x", shape=(4, 8), dtype="float32")
    op0 = relay.qnn.quantize(x, relay.const(2.0), relay.const(10), out_dtype="uint8")
    op1 = relay.qnn.dequantize(op0, relay.const(0.5), relay.const(5))
    relay_mod = tvm.IRModule.from_expr(op1)

    target_hexagon = tvm.target.hexagon("v68")
    # Default compilation flow
    with tvm.transform.PassContext(opt_level=3):
        opt_with_legalize, _ = relay.optimize(
            relay_mod, tvm.target.Target(target_hexagon, host=target_hexagon)
        )

    # Disable QNN legalization and canonicalization passes
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["qnn.Legalize"]):
        opt_without_legalize, _ = relay.optimize(
            relay_mod, tvm.target.Target(target_hexagon, host=target_hexagon)
        )

    # Check that QNN ops are absent with default compilation flow.
    text_with_legalize = opt_with_legalize.astext(show_meta_data=False)
    assert "qnn.quantize" not in text_with_legalize and "qnn.dequantize" not in text_with_legalize

    # Check that QNN ops are present without "qnn.Legalize" passes.
    text_without_legalize = opt_without_legalize.astext(show_meta_data=False)
    assert "qnn.quantize" in text_without_legalize and "qnn.dequantize" in text_without_legalize


def build_hexagon_module(relay_mod):
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["QnnCanonicalize"]):
        exe_mod = tvm.relay.build(
            relay_mod,
            tvm.target.Target(HEXAGON_AOT_LLVM_TARGET, host=HEXAGON_AOT_LLVM_TARGET),
            executor=Executor("aot"),
        )

    return exe_mod


def build_ref_module(relay_mod):
    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        exe_mod = tvm.relay.build(
            relay_mod, tvm.target.Target(target_llvm, host=target_llvm), executor=Executor("aot")
        )
    return exe_mod


def execute(mod_executor, inputs: dict):
    for input_name, input_data in inputs.items():
        mod_executor.set_input(input_name, input_data)
    mod_executor.run()
    return [mod_executor.get_output(i).numpy() for i in range(mod_executor.get_num_outputs())]


def execute_on_hexagon(hexagon_session, exe_mod, inputs: dict):
    return execute(hexagon_session.get_executor_from_factory(exe_mod), inputs)


def execute_on_cpu(exe_mod, inputs: dict):
    return execute(tvm.runtime.executor.AotModule(exe_mod["default"](tvm.cpu(0))), inputs)


def assert_allclose(actuals, desireds, rtol=1e-07, atol=0.01):
    return [tvm.testing.assert_allclose(a, d, rtol, atol) for a, d in zip(actuals, desireds)]


def run_and_compare(hexagon_session, relay_mod, inputs, rtol=None, atol=None):
    """Compile and execute given relay module on CPU and Hexagon, and compare
    results"""
    hexagon_mod = build_hexagon_module(relay_mod)
    cpu_mod = build_ref_module(relay_mod)

    hexagon_outs = execute_on_hexagon(hexagon_session, hexagon_mod, inputs)
    cpu_outs = execute_on_cpu(cpu_mod, inputs)

    # Do not pass rtol/atol if not present to use default values from assert_allclose
    tolerances = dict()
    if rtol is not None:
        tolerances["rtol"] = rtol
    if atol is not None:
        tolerances["atol"] = atol

    assert_allclose(hexagon_outs, cpu_outs, **tolerances)


# First test basic QNN ops: quantize, dequantize, requantize
#
class TestQnnQuantize:
    """QNN Quantize test class."""

    input_shape = tvm.testing.parameter([1, 8, 8, 32], [1, 10, 10, 32], [1, 12, 12, 128])
    odtype = tvm.testing.parameter("int8", "uint8")

    @tvm.testing.requires_hexagon
    def test_qnn_quantize(self, hexagon_session: Session, odtype, input_shape):
        """Test qnn.quantize"""

        def gen_relay_expr_qnn(output_scale, output_zero_point):
            data = relay.var("data", shape=input_shape, dtype="float32")
            qnn_quantize = relay.qnn.quantize(
                data,
                output_scale=relay.const(output_scale),
                output_zero_point=relay.const(output_zero_point),
                axis=-1,
                out_dtype=odtype,
            )
            return qnn_quantize

        inputs = {"data": np.random.random(input_shape)}
        # Use quantize_np to obtain reasonable quantization parameters.
        ref_out, scale, zero_point = quantize_np(inputs["data"], odtype)

        relay_mod = tvm.IRModule.from_expr(gen_relay_expr_qnn(scale, zero_point))

        hexagon_mod = build_hexagon_module(relay_mod)
        hexagon_outs = execute_on_hexagon(hexagon_session, hexagon_mod, inputs)
        assert_allclose(hexagon_outs, [ref_out], atol=1)


class TestQnnDequantize:
    """QNN Dequantize test class."""

    input_shape = tvm.testing.parameter(
        [1, 12, 32, 128], [1, 10, 10, 32], [1, 6, 6, 2048], [1, 1000]
    )
    idtype = tvm.testing.parameter("int8", "uint8")

    @tvm.testing.requires_hexagon
    def test_qnn_dequantize(self, hexagon_session: Session, idtype, input_shape):
        """Test qnn.dequantize"""

        def gen_relay_expr_qnn(dtype, input_scale, input_zero_point):
            data = relay.var("data", shape=input_shape, dtype=dtype)
            qnn_dequantize = relay.qnn.dequantize(
                data,
                input_scale=relay.const(input_scale),
                input_zero_point=relay.const(input_zero_point),
            )
            return qnn_dequantize

        # Generate float data, then quantize it to produce input.
        ref_out = np.random.random(input_shape)
        data, scale, zero_point = quantize_np(ref_out, idtype)
        inputs = {"data": data}

        relay_mod = tvm.IRModule.from_expr(gen_relay_expr_qnn(idtype, scale, zero_point))

        hexagon_mod = build_hexagon_module(relay_mod)
        hexagon_outs = execute_on_hexagon(hexagon_session, hexagon_mod, inputs)
        # We do
        #   original -[quantize]-> input -[dequantize]-> output
        # then compare "original" with "output". Use rtol=1 because of the quantized
        # format in the middle.
        assert_allclose(hexagon_outs, [ref_out], rtol=1, atol=1e-2)  # rtol = 1


class TestQnnRequantize:
    """QNN requantize test class"""

    @tvm.testing.requires_hexagon
    def test_qnn_requantize(self, hexagon_session: Session):
        """Test qnn.requantize"""
        data_shape = [256]
        data = relay.var("data", shape=data_shape, dtype="int32")

        op = relay.qnn.requantize(
            data,
            input_scale=relay.const(0.156),
            input_zero_point=relay.const(2),
            output_scale=relay.const(0.212),
            output_zero_point=relay.const(1),
            out_dtype="int8",
        )
        relay_mod = tvm.IRModule.from_expr(op)

        inputs = {"data": np.arange(-256, 256, 2, dtype="int32")}

        run_and_compare(hexagon_session, relay_mod, inputs, rtol=0, atol=0)  # equal


class TestQnnAvgPool2d:
    """QNN AvgPool2d test class."""

    _multitest_params = [
        ([1, 12, 12, 32], "NHWC", [3, 3], [1, 1], [2, 3], [1, 2, 3, 4], False, False),
        ([1, 18, 18, 32], "NCHW", [3, 3], [2, 2], [2, 1], [1, 2, 3, 4], False, True),
    ]

    (
        input_shape,
        layout,
        kernel,
        stride,
        dilation,
        padding,
        ceil_mode,
        count_include_pad,
    ) = tvm.testing.parameters(*_multitest_params)

    idtype, odtype = tvm.testing.parameters(("uint8", "uint8"))

    @tvm.testing.requires_hexagon
    def test_qnn_avg_pool2d(
        self,
        hexagon_session: Session,
        idtype,
        odtype,
        input_shape,
        kernel,
        stride,
        dilation,
        padding,
        ceil_mode,
        count_include_pad,
        layout,
    ):
        """Test qnn.avg_pool2d"""

        def gen_relay_expr_qnn(
            dtype, input_scale, input_zero_point, output_scale, output_zero_point
        ):
            data = relay.var("data", shape=input_shape, dtype=dtype)
            qnn_avg_pool = relay.qnn.avg_pool2d(
                data,
                input_scale=relay.const(input_scale),
                input_zero_point=relay.const(input_zero_point),
                output_scale=relay.const(output_scale),
                output_zero_point=relay.const(output_zero_point),
                pool_size=kernel,
                strides=stride,
                dilation=dilation,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
                layout=layout,
            )

            return qnn_avg_pool

        # Generate inputs and reference data first.
        fp_input = np.random.random(input_shape)
        fp_output = tvm.topi.testing.poolnd_python(
            fp_input,
            kernel,
            stride,
            dilation,
            padding_before=padding[:2],
            padding_after=padding[2:],
            pool_type="avg",
            count_include_pad=count_include_pad,
            ceil_mode=ceil_mode,
            layout=layout,
        )
        input_data, input_scale, input_zero_point = quantize_np(fp_input, idtype)
        ref_out, output_scale, output_zero_point = quantize_np(fp_output, odtype)
        inputs = {"data": input_data}

        relay_mod = tvm.IRModule.from_expr(
            gen_relay_expr_qnn(
                idtype, input_scale, input_zero_point, output_scale, output_zero_point
            )
        )

        hexagon_mod = build_hexagon_module(relay_mod)
        hexagon_outs = execute_on_hexagon(hexagon_session, hexagon_mod, inputs)
        assert_allclose(hexagon_outs, [ref_out], rtol=0, atol=2)


class TestQnnBinaryOp:
    """QNN binary op test class"""

    operation = tvm.testing.parameter(relay.qnn.add, relay.qnn.subtract, relay.qnn.mul)
    dtype = tvm.testing.parameter("uint8", "int8")
    input_shape = tvm.testing.parameter([256], [4, 256])

    @tvm.testing.requires_hexagon
    def test_qnn_binary_op(self, hexagon_session: Session, operation, dtype, input_shape):
        """Test binary qnn ops"""
        lhs_shape = [4, 256]
        rhs_shape = input_shape
        lhs = relay.var("lhs", shape=lhs_shape, dtype=dtype)
        rhs = relay.var("rhs", shape=rhs_shape, dtype=dtype)
        lhs_zp = 1
        rhs_zp = 3

        op = operation(
            lhs,
            rhs,
            lhs_scale=relay.const(0.041, "float32"),
            lhs_zero_point=relay.const(lhs_zp, "int32"),
            rhs_scale=relay.const(0.017, "float32"),
            rhs_zero_point=relay.const(rhs_zp, "int32"),
            output_scale=relay.const(0.039, "float32"),
            output_zero_point=relay.const(2, "int32"),
        )
        relay_mod = tvm.IRModule.from_expr(op)

        inputs = {
            "lhs": np.random.randint(np.iinfo(dtype).min + lhs_zp, np.iinfo(dtype).max, lhs_shape),
            "rhs": np.random.randint(np.iinfo(dtype).min + rhs_zp, np.iinfo(dtype).max, rhs_shape),
        }

        run_and_compare(hexagon_session, relay_mod, inputs, atol=1)  # diff by 1 is ok

    @tvm.testing.requires_hexagon
    def test_qnn_binary_op_broadcasting(self, hexagon_session: Session, operation):
        """Test binary qnn ops (with argument broadcast)"""
        lhs_shape = [4, 256]
        lhs = relay.var("lhs", shape=lhs_shape, dtype="uint8")
        rhs = relay.const(11, dtype="uint8")

        op = operation(
            lhs,
            rhs,
            lhs_scale=relay.const(0.049, "float32"),
            lhs_zero_point=relay.const(1, "int32"),
            rhs_scale=relay.const(0.067, "float32"),
            rhs_zero_point=relay.const(3, "int32"),
            output_scale=relay.const(0.041, "float32"),
            output_zero_point=relay.const(2, "int32"),
        )
        relay_mod = tvm.IRModule.from_expr(op)

        inputs = {"lhs": np.random.randint(1, 255, size=lhs_shape)}

        run_and_compare(hexagon_session, relay_mod, inputs, atol=1)  # diff by 1 is ok


class TestQnnConcatenate:
    """QNN concatenate test class"""

    @tvm.testing.requires_hexagon
    def test_qnn_concatenate(self, hexagon_session: Session):
        """Test qnn.concatenate"""
        x_shape = [1, 64]
        y_shape = [2, 64]
        z_shape = [3, 64]
        input_x = relay.var("x", shape=x_shape, dtype="uint8")
        input_y = relay.var("y", shape=y_shape, dtype="uint8")
        input_z = relay.var("z", shape=z_shape, dtype="uint8")

        op = relay.qnn.concatenate(
            (input_x, input_y, input_z),
            input_scales=(relay.const(0.3), relay.const(0.7), relay.const(1.3)),
            input_zero_points=(relay.const(0), relay.const(1), relay.const(2)),
            output_scale=relay.const(0.8),
            output_zero_point=relay.const(5),
            axis=0,
        )
        relay_mod = tvm.IRModule.from_expr(op)

        inputs = {
            "x": np.arange(0, 64, 1, dtype="uint8").reshape(x_shape),
            "y": np.arange(0, 128, 1, dtype="uint8").reshape(y_shape),
            "z": np.arange(0, 192, 1, dtype="uint8").reshape(z_shape),
        }

        run_and_compare(hexagon_session, relay_mod, inputs, atol=1)  # diff by 1 is ok


class TestQnnConv2D:
    """QNN conv2d op test class."""

    @tvm.testing.requires_hexagon
    def test_qnn_quantize_conv2d_requantize(self, hexagon_session: Session):
        """Tast qnn.conv2d"""
        data_shape = [1, 8, 32, 32]
        weight_shape = [16, 8, 3, 3]
        data = relay.var("data", shape=data_shape, dtype="float32")
        weight = relay.var("weight", shape=weight_shape, dtype="float32")
        op0 = relay.qnn.quantize(data, relay.const(0.078), relay.const(0), out_dtype="uint8")
        op1 = relay.qnn.quantize(weight, relay.const(0.07), relay.const(0), out_dtype="int8")
        op2 = relay.qnn.conv2d(
            op0,
            op1,
            input_zero_point=relay.const(0),
            kernel_zero_point=relay.const(0),
            input_scale=relay.const(0.078),
            kernel_scale=relay.const(0.07),
            padding=[0, 0, 0, 0],
            channels=16,
            kernel_size=[3, 3],
        )
        op5 = relay.qnn.requantize(
            op2,
            input_scale=relay.const(0.05),
            input_zero_point=relay.const(0),
            output_scale=relay.const(0.21),
            output_zero_point=relay.const(61),
            out_dtype="int8",
        )
        relay_mod = tvm.IRModule.from_expr(op5)

        inputs = {
            "data": np.random.rand(*data_shape),
            "weight": np.random.rand(*weight_shape) - 0.5,
        }

        run_and_compare(hexagon_session, relay_mod, inputs, rtol=0, atol=0)  # equal


class TestQnnDense:
    """QNN dense op test class."""

    @tvm.testing.requires_hexagon
    def test_alter_layout_qnn_dense(self):
        """Test weights layout transformation of qnn.dense with int8 weights"""
        data = relay.var("data", shape=(128, 16), dtype="uint8")
        weight = relay.var("weight", shape=(64, 16), dtype="int8")
        zero = relay.const(0)
        iscale = relay.const(0.15)
        wscale = relay.const(0.37)

        def before():
            return relay.qnn.dense(data, weight, zero, zero, iscale, wscale, units=None)

        def expected():
            op0 = relay.layout_transform(weight, src_layout="NC", dst_layout="NC32n4c")
            return relay.qnn.contrib_dense_pack(data, op0, zero, zero, iscale, wscale, "NC32n4c")

        target = tvm.target.hexagon("v68")
        with tvm.target.Target(target):
            a = run_opt_pass(before(), tvm.relay.transform.AlterOpLayout())
            b = run_infer_type(expected())
            tvm.ir.assert_structural_equal(a, b)

    # Dense + bias_add + requantize
    #
    dtype = tvm.testing.parameter("uint8", "int8")
    n_dim = tvm.testing.parameter(64, 60)

    @tvm.testing.requires_hexagon
    def test_qnn_dense_biasadd_requantize(self, hexagon_session: Session, dtype, n_dim):
        """Check lowering of qnn.dense + bias_add + qnn.requantize
        dtype: type of weights
        n_dim: N dimension of weights, need to check cases when it is multiple of 32 and not.
        """
        data_shape = [128, 32]
        weight_shape = [n_dim, 32]
        bias_shape = [n_dim]
        data = relay.var("data", shape=data_shape, dtype="uint8")
        weight = relay.var("weight", shape=weight_shape, dtype=dtype)
        bias = relay.var("bias", shape=bias_shape, dtype="int32")

        op0 = relay.qnn.dense(
            data,
            weight,
            input_zero_point=relay.const(2),
            kernel_zero_point=relay.const(0),
            input_scale=relay.const(0.08),
            kernel_scale=relay.const(0.07),
            units=None,
        )
        op1 = relay.nn.bias_add(op0, bias)
        op2 = relay.qnn.requantize(
            op1,
            input_scale=relay.const(1.3),
            input_zero_point=relay.const(4),
            output_scale=relay.const(3.7),
            output_zero_point=relay.const(1),
            out_dtype="uint8",
        )
        relay_mod = tvm.IRModule.from_expr(op2)

        np.random.seed(0)

        inputs = {
            "data": np.random.randint(2, 8, size=data_shape, dtype="uint8"),
            "weight": np.random.randint(0, 8, size=weight_shape, dtype=dtype),
            "bias": np.random.randint(-10, 10, size=bias_shape, dtype="int32"),
        }

        run_and_compare(hexagon_session, relay_mod, inputs, atol=1)  # diff by 1 is ok

    # Dense + requantize
    #
    @tvm.testing.requires_hexagon
    def test_qnn_dense_requantize(self, hexagon_session: Session):
        """Check lowering of qnn.dense + qnn.requantize
        Checkint the case: data type = "uint8", weight type = "int8", input zp = 0 and kernel zp = 0
        """
        data_shape = [128, 32]
        weight_shape = [64, 32]
        data = relay.var("data", shape=data_shape, dtype="uint8")
        weight = relay.var("weight", shape=weight_shape, dtype="int8")

        op0 = relay.qnn.dense(
            data,
            weight,
            input_zero_point=relay.const(0),
            kernel_zero_point=relay.const(0),
            input_scale=relay.const(0.06),
            kernel_scale=relay.const(0.19),
            units=64,
        )
        op1 = relay.qnn.requantize(
            op0,
            input_scale=relay.const(0.1),
            input_zero_point=relay.const(0),
            output_scale=relay.const(0.24),
            output_zero_point=relay.const(64),
            out_dtype="uint8",
        )
        relay_mod = tvm.IRModule.from_expr(op1)

        np.random.seed(0)

        inputs = {
            "data": np.random.randint(0, 8, size=data_shape, dtype="uint8"),
            "weight": np.random.randint(-4, 4, size=weight_shape, dtype="int8"),
        }

        run_and_compare(hexagon_session, relay_mod, inputs, atol=1)  # diff by 1 is ok


class TestQnnTanh:
    """QNN tanh test class"""

    @tvm.testing.requires_hexagon
    def test_qnn_tanh(self, hexagon_session: Session):
        """Test qnn.tanh"""
        data_shape = [256]
        data = relay.var("data", shape=data_shape, dtype="uint8")

        op = relay.qnn.tanh(
            data,
            scale=relay.const(0.518),
            zero_point=relay.const(137),
            output_scale=relay.const(0.207),
            output_zero_point=relay.const(128),
        )
        relay_mod = tvm.IRModule.from_expr(op)

        inputs = {"data": np.arange(0, 256, 1, dtype="uint8")}

        run_and_compare(hexagon_session, relay_mod, inputs, rtol=0, atol=0)  # equal


if __name__ == "__main__":
    tvm.testing.main()
