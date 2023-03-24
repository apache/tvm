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
# pylint: disable=invalid-name,missing-function-docstring,redefined-outer-name

""" Test Relay integrated qnn ops
There are two types of tests for qnn ops in this file. One to verify the
correctness of the relay integration and the other one to verify
the fake quantization to integer implemented for picking up the qnn op.
The former is only executed when qnn canonicalization is disabled.
The latter is executed both with and without canonicalization.
"""
# TODO: We might want to distribute these test cases into other test cases such as
# test_wo_qnn_canonicalization and test_pass_fake_quantization_to_integer in the future.

import numpy as np

import tvm.testing
import tvm.topi.testing
from tvm import relay
from tvm.contrib.hexagon.session import Session
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.hexagon import allocate_hexagon_array
from .infrastructure import quantize_np

from .pytest_util import get_multitest_ids, create_populated_numpy_ndarray, TensorContentRandom


def compile_for_target(mod, target="hexagon", disable_canonicalization=False):
    runtime = Runtime("cpp")
    executor = Executor("graph", {"link-params": True})
    if target == "hexagon":
        target_hexagon = tvm.target.hexagon("v68")
        target = tvm.target.Target(target_hexagon, host=target_hexagon)
        print("Trying relay.build for ...", target)
        dis_passes = []
        if disable_canonicalization:
            dis_passes = ["QnnCanonicalize"]
        with tvm.transform.PassContext(opt_level=3, disabled_pass=dis_passes):
            lib = relay.build(mod, target=target, runtime=runtime, executor=executor)
            print(lib.function_metadata)
        print("Finished relay.build for...", target)
    elif target == "llvm":
        target = tvm.target.Target("llvm")
        print("Trying relay.build for ...", target)
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, runtime=runtime, executor=executor)
            print(lib.function_metadata)
        print("Finished relay.build for...", target)
    return lib


def run_model_on_hexagon(hexagon_session, mod, inputs, params=None, disable_canonicalization=True):
    hexagon_lowered = compile_for_target(mod, "hexagon", disable_canonicalization)
    graph_mod = hexagon_session.get_executor_from_factory(hexagon_lowered)
    if params is None:
        params = {}
    graph_mod.set_input(**params)
    graph_mod.run(**inputs)
    return graph_mod.get_output(0).numpy()


def run_model_on_llvm(mod, inputs, params=None):
    llvm_lowered = compile_for_target(mod, "llvm")
    llvm_graph_mod = tvm.contrib.graph_executor.GraphModule(llvm_lowered["default"](tvm.cpu(0)))
    if params is None:
        params = {}
    llvm_graph_mod.set_input(**params)
    llvm_graph_mod.run(**inputs)
    return llvm_graph_mod.get_output(0).numpy()


def compare_fq_to_int(hexagon_session, expr, input_np_quant, params=None):
    working_scope = "global"
    inputs_llvm = {"data": input_np_quant}
    input_arr = allocate_hexagon_array(
        hexagon_session.device, data=input_np_quant, mem_scope=working_scope
    )
    inputs_hex = {"data": input_arr}
    mod = tvm.IRModule.from_expr(expr)
    mod = tvm.relay.transform.InferType()(mod)
    mod_int = tvm.relay.transform.FakeQuantizationToInteger()(mod)
    assert not tvm.ir.structural_equal(mod, mod_int)

    ref_out_llvm = run_model_on_llvm(mod, inputs_llvm, params)

    # Compare the Hexagon and LLVM results with and without the qnn canonicalization
    print("Comparing Hexagon and LLVM reusults (canonicalization disabled)...")
    hexagon_output_fq_wo_qnn_can = run_model_on_hexagon(
        hexagon_session, mod_int, inputs_hex, params, True
    )
    tvm.testing.assert_allclose(ref_out_llvm, hexagon_output_fq_wo_qnn_can, rtol=0, atol=2)
    print("Comparing Hexagon and LLVM reusults (canonicalization enabled)...")
    hexagon_output_fq_w_qnn_can = run_model_on_hexagon(
        hexagon_session, mod_int, inputs_hex, params, False
    )
    assert np.all(
        np.abs(ref_out_llvm.astype("int32") - hexagon_output_fq_w_qnn_can.astype("int32")) <= 1
    )


@tvm.testing.fixture
def input_np(input_shape, idtype, input_tensor_populator):
    if idtype in ("int8", "uint8"):
        idtype = "float32"  # Use "float32" input which will be quantized later
    return create_populated_numpy_ndarray(input_shape, idtype, input_tensor_populator)


@tvm.testing.fixture
def transformed_expected_output_np(expected_output_np, odtype):
    scale = None
    zero_point = None
    if odtype in ("int8", "uint8"):
        quant_arr, scale, zero_point = quantize_np(expected_output_np, odtype)
    else:
        quant_arr = expected_output_np
    return quant_arr, scale, zero_point


@tvm.testing.fixture
def transformed_input_np(input_np, idtype):
    scale = None
    zero_point = None
    if idtype in ("int8", "uint8"):
        quant_arr, scale, zero_point = quantize_np(input_np, idtype)
    else:
        quant_arr = input_np
    return quant_arr, scale, zero_point


input_layout = tvm.testing.parameter("nhwc")
output_layout = tvm.testing.parameter("nhwc")


class TestQnnAvgPool2d:
    """QNN AvgPool2d test class."""

    _param_descs = [
        "in_shape",  # input_shape
        "layout",  # NHWC or NCHW
        "kernel",  # kernel
        "stride",  # stride
        "dil",  # dilation
        "pad",  # padding
        "ceil",  # ceil_mode
        "cnt_padded",  # count_include_pad
        None,  # input_tensor_populator
    ]

    _multitest_params = [
        (
            [1, 12, 12, 32],
            "NHWC",
            [3, 3],
            [1, 1],
            [2, 3],
            [1, 2, 3, 4],
            False,
            False,
            TensorContentRandom(),
        ),
        (
            [1, 18, 18, 32],  # output shape: [1, 16, 16, 32]
            "NCHW",
            [3, 3],
            [2, 2],
            [2, 1],
            [1, 2, 3, 4],
            False,
            True,
            TensorContentRandom(),
        ),
    ]

    _param_ids = get_multitest_ids(_multitest_params, _param_descs)
    idtype, odtype = tvm.testing.parameters(("uint8", "uint8"))

    (
        input_shape,
        layout,
        kernel,
        stride,
        dilation,
        padding,
        ceil_mode,
        count_include_pad,
        input_tensor_populator,
    ) = tvm.testing.parameters(*_multitest_params, ids=_param_ids)

    @tvm.testing.fixture
    def expected_output_np(
        self, input_np, kernel, stride, dilation, padding, ceil_mode, count_include_pad, layout
    ):
        pad_before = padding[:2]
        pad_after = padding[2:]
        ref_np = tvm.topi.testing.poolnd_python(
            input_np,
            kernel,
            stride,
            dilation,
            pad_before,
            pad_after,
            "avg",  # pool_type
            count_include_pad,
            ceil_mode,
            layout=layout,
        )

        return ref_np

    @tvm.testing.requires_hexagon
    def test_integrated_qnn_avg_pool2d(
        self,
        idtype,
        input_shape,
        kernel,
        stride,
        dilation,
        padding,
        ceil_mode,
        count_include_pad,
        layout,
        transformed_input_np,
        transformed_expected_output_np,
        hexagon_session: Session,
    ):
        working_scope = "global"

        if idtype in ("uint8"):
            input_np_quant, input_scale, input_zero_point = transformed_input_np
            golden_out_np, output_scale, output_zero_point = transformed_expected_output_np
        else:
            raise RuntimeError(f"Unsupport input dtype '{idtype}'")

        input_arr = allocate_hexagon_array(
            hexagon_session.device, data=input_np_quant, mem_scope=working_scope
        )
        inputs_hex = {"data": input_arr}

        def gen_relay_expr_qnn(dtype):
            data = relay.var("data", shape=input_shape, dtype=dtype)
            qnn_avg_pool = relay.qnn.op.avg_pool2d(
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

        op_hex = gen_relay_expr_qnn(idtype)
        mod = tvm.IRModule.from_expr(op_hex)
        mod = relay.transform.InferType()(mod)
        hexagon_out = run_model_on_hexagon(hexagon_session, mod, inputs_hex)
        np.testing.assert_allclose(hexagon_out, golden_out_np, rtol=0, atol=2)

    @tvm.testing.requires_hexagon
    def test_fake_quantize_avg_pool2d(
        self,
        idtype,
        input_shape,
        kernel,
        stride,
        dilation,
        padding,
        layout,
        ceil_mode,
        count_include_pad,
        transformed_input_np,
        transformed_expected_output_np,
        hexagon_session: Session,
    ):
        if idtype in ("uint8"):
            input_np_quant, input_scale, input_zero_point = transformed_input_np
            _, output_scale, output_zero_point = transformed_expected_output_np
        else:
            raise RuntimeError(f"Unsupport input dtype '{idtype}'")

        def gen_relay_expr(dtype):
            data = relay.var("data", shape=input_shape, dtype=dtype)
            data_deq = relay.qnn.op.dequantize(
                data, relay.const(input_scale), relay.const(input_zero_point)
            )
            op = relay.op.nn.avg_pool2d(
                data=data_deq,
                pool_size=kernel,
                strides=stride,
                dilation=dilation,
                padding=padding,
                layout=layout,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
            )
            out_quant = relay.qnn.op.quantize(
                op, relay.const(output_scale), relay.const(output_zero_point), out_dtype=dtype
            )
            return out_quant

        op_llvm = gen_relay_expr(idtype)
        compare_fq_to_int(hexagon_session, op_llvm, input_np_quant)


class TestQnnQuantize:
    """QNN Quantize test class."""

    _param_descs = ["in_shape", None]  # input_shape  # input_tensor_populator

    _multitest_params = [
        ([1, 8, 8, 32], TensorContentRandom()),
        ([1, 10, 10, 32], TensorContentRandom()),
        ([1, 12, 12, 128], TensorContentRandom()),
    ]

    _param_ids = get_multitest_ids(_multitest_params, _param_descs)

    (input_shape, input_tensor_populator) = tvm.testing.parameters(
        *_multitest_params, ids=_param_ids
    )

    idtype, odtype = tvm.testing.parameters(("float32", "int8"), ("float32", "uint8"))

    @tvm.testing.fixture
    def expected_output_np(self, input_np):
        # The expected output is of the same shape as input.
        # The only computation applied on the input is quanization.
        # Since transform_expected_output quantizes the data,
        # here, we return the orignal input array in float
        return input_np

    @tvm.testing.requires_hexagon
    def test_integrated_qnn_quantize(
        self,
        idtype,
        odtype,
        input_shape,
        input_np,
        transformed_expected_output_np,
        hexagon_session: Session,
    ):
        working_scope = "global"
        if odtype in ("int8", "uint8"):
            golden_out_np, output_scale, output_zero_point = transformed_expected_output_np
        else:
            raise RuntimeError(f"Unsupport output dtype '{odtype}'")

        input_arr = allocate_hexagon_array(
            hexagon_session.device, data=input_np, mem_scope=working_scope
        )
        inputs_hex = {"data": input_arr}

        def gen_relay_expr_qnn(dtype):
            data = relay.var("data", shape=input_shape, dtype=dtype)
            qnn_quantize = relay.qnn.op.quantize(
                data,
                output_scale=relay.const(output_scale),
                output_zero_point=relay.const(output_zero_point),
                axis=-1,
                out_dtype=odtype,
            )
            return qnn_quantize

        op_hex = gen_relay_expr_qnn(idtype)
        mod = tvm.IRModule.from_expr(op_hex)
        mod = relay.transform.InferType()(mod)
        hexagon_out = run_model_on_hexagon(hexagon_session, mod, inputs_hex)
        np.testing.assert_allclose(hexagon_out, golden_out_np, rtol=0, atol=1)


class TestQnnDequantize:
    """QNN Dequantize test class."""

    _param_descs = ["in_shape", None]  # input_shape  # input_tensor_populator

    _multitest_params = [
        ([1, 12, 32, 128], TensorContentRandom()),
        ([1, 10, 10, 32], TensorContentRandom()),
        ([1, 6, 6, 2048], TensorContentRandom()),
        ([1, 1000], TensorContentRandom()),
    ]

    _param_ids = get_multitest_ids(_multitest_params, _param_descs)

    (input_shape, input_tensor_populator) = tvm.testing.parameters(
        *_multitest_params, ids=_param_ids
    )

    idtype, odtype = tvm.testing.parameters(("int8", "float32"), ("uint8", "float32"))

    @tvm.testing.fixture
    def expected_output_np(self, input_np, idtype):
        quant_np, scale, zero_point = quantize_np(input_np, idtype)
        ref_np = (scale * (quant_np.astype("int32") - zero_point)).astype("float32")
        return ref_np

    @tvm.testing.requires_hexagon
    def test_integrated_qnn_dequantize(
        self,
        idtype,
        odtype,
        input_shape,
        transformed_input_np,
        transformed_expected_output_np,
        hexagon_session: Session,
    ):
        working_scope = "global"
        if odtype in ("float32"):
            input_np_quant, input_scale, input_zero_point = transformed_input_np
            golden_out_np, _, _ = transformed_expected_output_np
        else:
            raise RuntimeError(f"Unsupport odtype '{odtype}'")

        input_arr = allocate_hexagon_array(
            hexagon_session.device, data=input_np_quant, mem_scope=working_scope
        )
        inputs_hex = {"data": input_arr}

        def gen_relay_expr_qnn(dtype):
            data = relay.var("data", shape=input_shape, dtype=dtype)
            qnn_quantize = relay.qnn.op.dequantize(
                data,
                input_scale=relay.const(input_scale),
                input_zero_point=relay.const(input_zero_point),
            )
            return qnn_quantize

        op_hex = gen_relay_expr_qnn(idtype)
        mod = tvm.IRModule.from_expr(op_hex)
        mod = relay.transform.InferType()(mod)
        hexagon_out = run_model_on_hexagon(hexagon_session, mod, inputs_hex)
        np.testing.assert_allclose(hexagon_out, golden_out_np, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    tvm.testing.main()
