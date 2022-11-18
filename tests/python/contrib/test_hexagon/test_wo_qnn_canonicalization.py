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
"""No QNN canonicalization tests."""

import numpy as np

import tvm.testing
from tvm import relay
from tvm.contrib.hexagon.session import Session
from tvm.contrib.hexagon.pytest_plugin import HEXAGON_AOT_LLVM_TARGET
from tvm.relay.backend import Executor


@tvm.testing.requires_hexagon
def test_no_qnn_pass():
    """No QNN pass test."""
    x = relay.var("x", shape=(4, 8), dtype="float32")
    op0 = relay.qnn.op.quantize(x, relay.const(2.0), relay.const(10), out_dtype="uint8")
    op1 = relay.qnn.op.dequantize(op0, relay.const(0.5), relay.const(5))
    mod = tvm.IRModule.from_expr(op1)

    target_hexagon = tvm.target.hexagon("v68")
    # Default compilation flow
    with tvm.transform.PassContext(opt_level=3):
        opt_mod_1, _ = relay.optimize(mod, tvm.target.Target(target_hexagon, host=target_hexagon))

    # Disable QNN legalization and canonicalization passes
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["qnn.Legalize"]):
        opt_mod_2, _ = relay.optimize(mod, tvm.target.Target(target_hexagon, host=target_hexagon))

    # Check that QNN ops are absent with default compilation flow.
    assert "qnn.quantize" not in opt_mod_1.astext(show_meta_data=False)
    assert "qnn.dequantize" not in opt_mod_1.astext(show_meta_data=False)

    # Check that QNN ops are present without "qnn.Legalize" passes.
    assert "qnn.quantize" in opt_mod_2.astext(show_meta_data=False)
    assert "qnn.dequantize" in opt_mod_2.astext(show_meta_data=False)


def execute(mod_executor, inputs: dict):
    for input_name, input_data in inputs.items():
        mod_executor.set_input(input_name, input_data)
    mod_executor.run()
    return mod_executor.get_output(0).numpy()


def build_hexagon_module(mod):
    with tvm.transform.PassContext(opt_level=3, disabled_pass=["qnn.Legalize"]):
        hexagon_lowered = tvm.relay.build(
            mod,
            tvm.target.Target(HEXAGON_AOT_LLVM_TARGET, host=HEXAGON_AOT_LLVM_TARGET),
            executor=Executor("aot"),
        )

    return hexagon_lowered


def build_ref_module(mod):
    target_llvm = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=3):
        llvm_lowered = tvm.relay.build(
            mod,
            tvm.target.Target(target_llvm, host=target_llvm),
            executor=Executor("aot"),
        )
    return llvm_lowered


@tvm.testing.requires_hexagon
def test_qnn_conv2d_rq(hexagon_session: Session):
    """QNN conv2d test."""
    data_shape = [1, 8, 32, 32]
    weight_shape = [16, 8, 3, 3]
    data = relay.var("data", shape=data_shape, dtype="float32")
    weight = relay.var("weight", shape=weight_shape, dtype="float32")
    op0 = relay.qnn.op.quantize(data, relay.const(0.078), relay.const(0), out_dtype="int8")
    op1 = relay.qnn.op.quantize(weight, relay.const(0.07), relay.const(0), out_dtype="int8")
    op2 = relay.qnn.op.conv2d(
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
    op5 = relay.qnn.op.requantize(
        op2,
        input_scale=relay.const(0.05),
        input_zero_point=relay.const(0),
        output_scale=relay.const(0.21),
        output_zero_point=relay.const(61),
        out_dtype="int8",
    )
    relay_mod = tvm.IRModule.from_expr(op5)

    # Compile for Hexagon
    hexagon_lowered = build_hexagon_module(relay_mod)

    # Reference compilation
    llvm_lowered = build_ref_module(relay_mod)

    data_np = np.random.rand(*data_shape) - 0.5
    weight_np = np.random.rand(*weight_shape) - 0.5
    inputs = {"data": data_np, "weight": weight_np}

    hx_m = hexagon_session.get_executor_from_factory(hexagon_lowered)
    hexagon_output = execute(hx_m, inputs)

    dev = tvm.cpu(0)
    llvm_m = tvm.runtime.executor.AotModule(llvm_lowered["default"](dev))
    llvm_out = execute(llvm_m, inputs)

    np.testing.assert_equal(hexagon_output, llvm_out)


@tvm.testing.requires_hexagon
def test_qnn_dense_bias_rq(hexagon_session: Session):
    """QNN dense with bias test."""
    data_shape = [8, 8]
    weight_shape = [16, 8]
    bias_shape = [16]
    data = relay.var("data", shape=data_shape, dtype="float32")
    weight = relay.var("weight", shape=weight_shape, dtype="float32")
    bias = relay.var("bias", shape=bias_shape, dtype="float32")

    op0 = relay.qnn.op.quantize(data, relay.const(0.08), relay.const(0), out_dtype="int8")
    op1 = relay.qnn.op.quantize(weight, relay.const(0.07), relay.const(0), out_dtype="int8")
    op2 = relay.qnn.op.dense(
        op0,
        op1,
        input_zero_point=relay.const(0),
        kernel_zero_point=relay.const(0),
        input_scale=relay.const(0.08),
        kernel_scale=relay.const(0.07),
        units=None,
    )
    op3 = relay.qnn.op.quantize(bias, relay.const(0.5), relay.const(0), out_dtype="int32")
    op4 = relay.nn.bias_add(op2, op3)
    op5 = relay.qnn.op.requantize(
        op4,
        input_scale=relay.const(0.05),
        input_zero_point=relay.const(0),
        output_scale=relay.const(0.212),
        output_zero_point=relay.const(10),
        out_dtype="int8",
    )
    relay_mod = tvm.IRModule.from_expr(op5)

    # Compile for Hexagon
    hexagon_lowered = build_hexagon_module(relay_mod)

    # Reference compilation
    llvm_lowered = build_ref_module(relay_mod)

    data_np = np.random.rand(*data_shape) - 0.5
    weight_np = np.random.rand(*weight_shape) - 0.5
    bias_np = np.random.rand(*bias_shape)
    inputs = {"data": data_np, "weight": weight_np, "bias": bias_np}

    hx_m = hexagon_session.get_executor_from_factory(hexagon_lowered)
    hexagon_output = execute(hx_m, inputs)

    dev = tvm.cpu(0)
    llvm_m = tvm.runtime.executor.AotModule(llvm_lowered["default"](dev))
    llvm_out = execute(llvm_m, inputs)

    np.testing.assert_equal(hexagon_output, llvm_out)


class TestQnnBinaryOp:
    """QNN binary op test class"""

    operation = tvm.testing.parameter(
        relay.qnn.op.add,
        relay.qnn.op.subtract,
        relay.qnn.op.mul,
    )
    dtype = tvm.testing.parameter("uint8", "int8")
    input_shape = tvm.testing.parameter([256], [4, 256])

    @tvm.testing.requires_hexagon
    def test_qnn_binary_op_broadcasting(
        self, hexagon_session: Session, operation, dtype, input_shape
    ):
        """qnn binary op test without QNN canonicalization."""
        lhs_shape = [4, 256]
        rhs_shape = input_shape
        lhs = relay.var("lhs", shape=lhs_shape, dtype=dtype)
        rhs = relay.var("rhs", shape=rhs_shape, dtype=dtype)
        zp_const1 = 1
        zp_const2 = 3

        op = operation(
            lhs,
            rhs,
            lhs_scale=relay.const(0.041, "float32"),
            lhs_zero_point=relay.const(zp_const1, "int32"),
            rhs_scale=relay.const(0.017, "float32"),
            rhs_zero_point=relay.const(zp_const2, "int32"),
            output_scale=relay.const(0.039, "float32"),
            output_zero_point=relay.const(2, "int32"),
        )
        mod = tvm.IRModule.from_expr(op)

        # Compile for Hexagon
        hexagon_lowered = build_hexagon_module(mod)

        # Reference compilation
        llvm_lowered = build_ref_module(mod)

        lhs_np = np.random.randint(np.iinfo(dtype).min + zp_const1, np.iinfo(dtype).max, lhs_shape)
        rhs_np = np.random.randint(np.iinfo(dtype).min + zp_const2, np.iinfo(dtype).max, rhs_shape)
        inputs = {"lhs": lhs_np, "rhs": rhs_np}

        hx_m = hexagon_session.get_executor_from_factory(hexagon_lowered)
        hexagon_output = execute(hx_m, inputs)

        dev = tvm.cpu(0)
        llvm_m = tvm.runtime.executor.AotModule(llvm_lowered["default"](dev))
        llvm_output = execute(llvm_m, inputs)

        # Diff by 1 is Ok.
        tvm.testing.assert_allclose(hexagon_output, llvm_output, atol=1)

    @tvm.testing.requires_hexagon
    def test_qnn_binary_op_scalar(self, hexagon_session: Session, operation):
        """qnn binary op test without QNN canonicalization."""
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
        mod = tvm.IRModule.from_expr(op)

        # Compile for Hexagon
        hexagon_lowered = build_hexagon_module(mod)

        # Reference compilation
        llvm_lowered = build_ref_module(mod)

        lhs_np = np.random.randint(1, 255, size=lhs_shape)
        inputs = {"lhs": lhs_np}

        hx_m = hexagon_session.get_executor_from_factory(hexagon_lowered)
        hexagon_output = execute(hx_m, inputs)

        dev = tvm.cpu(0)
        llvm_m = tvm.runtime.executor.AotModule(llvm_lowered["default"](dev))
        llvm_output = execute(llvm_m, inputs)

        # Diff by 1 is Ok.
        tvm.testing.assert_allclose(hexagon_output, llvm_output, atol=1)


if __name__ == "__main__":
    tvm.testing.main()
