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

from typing import Callable, List

import numpy as np
import pytest
import scipy.special
import tvm
import tvm.testing
from tvm import relay
from tvm.relay.qnn.op.legalizations import hardswish_func


def dequantize(data, scale, zp):
    return scale * (np.asarray(data) - zp)


def generate_golden_output(
    floating_point_golden_func, dequantized_x, output_scale, output_zero_point, dtype
):
    output = floating_point_golden_func(dequantized_x)
    output = np.around(output / output_scale + output_zero_point)

    np_dtype = {"int8": np.int8, "uint8": np.uint8}[dtype]

    q_min = np.iinfo(np_dtype).min
    q_max = np.iinfo(np_dtype).max
    return np.clip(output, q_min, q_max)


def run_qnn_func(func: relay.Function, args: List[relay.Expr]):
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.Legalize()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(*args)
    return op_res.numpy()


def create_qnn_func(
    qnn_op: Callable[[relay.Expr, relay.Expr, relay.Expr, relay.Expr, relay.Expr], relay.Call],
    x_data: np.ndarray,
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
    input_dtype: str = "uint8",
):
    x = relay.var("x", shape=x_data.shape, dtype=input_dtype)
    y = qnn_op(
        x=x,
        scale=relay.const(input_scale, "float32"),
        zero_point=relay.const(input_zero_point, "int32"),
        output_scale=relay.const(output_scale, "float32"),
        output_zero_point=relay.const(output_zero_point, "int32"),
    )
    return relay.Function([x], y)


def run_condition(
    qnn_op: Callable[[relay.Expr, relay.Expr, relay.Expr, relay.Expr, relay.Expr], relay.Call],
    floating_point_golden_func: Callable[[np.ndarray], np.ndarray],
    x_data: np.ndarray,
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
    input_dtype: str = "uint8",
):
    func = create_qnn_func(
        qnn_op,
        x_data,
        input_scale=input_scale,
        input_zero_point=input_zero_point,
        output_scale=output_scale,
        output_zero_point=output_zero_point,
        input_dtype=input_dtype,
    )

    x_dequantized = dequantize(x_data, input_scale, input_zero_point)
    golden_output = generate_golden_output(
        floating_point_golden_func,
        x_dequantized,
        output_scale,
        output_zero_point,
        dtype=input_dtype,
    )

    op_res = run_qnn_func(func, [x_data])
    np.testing.assert_equal(op_res, golden_output.astype(input_dtype))


def generic_test(
    qnn_op: Callable[[relay.Expr, relay.Expr, relay.Expr, relay.Expr, relay.Expr], relay.Call],
    floating_point_golden_func: Callable[[np.ndarray], np.ndarray],
    input_dtype: str = "uint8",
    x_data: np.ndarray = np.arange(0, 256, dtype="uint8"),
):
    x_data = x_data.view(input_dtype)
    return run_condition(
        qnn_op,
        floating_point_golden_func,
        x_data,
        input_scale=0.125,
        input_zero_point=0,
        output_scale=0.125,
        output_zero_point=0,
        input_dtype=input_dtype,
    )


class TestRSqrt:
    def test_saturation(self):
        # Same qparams in and out
        x_data = np.array((255, 133, 0, 9)).reshape((1, 4))
        run_condition(
            relay.qnn.rsqrt,
            lambda x: 1 / np.sqrt(x),
            x_data,
            input_scale=0.125,
            input_zero_point=0,
            output_scale=0.125,
            output_zero_point=0,
            input_dtype="uint8",
        )

        # Different scale
        run_condition(
            relay.qnn.rsqrt,
            lambda x: 1 / np.sqrt(x),
            x_data,
            input_scale=0.125,
            input_zero_point=0,
            output_scale=0.25,
            output_zero_point=0,
            input_dtype="uint8",
        )

    def test_all_numbers_uint8(self):
        generic_test(relay.qnn.rsqrt, lambda x: 1 / np.sqrt(x), input_dtype="uint8")

    def test_all_numbers_int8(self):
        generic_test(
            relay.qnn.rsqrt,
            lambda x: 1 / np.sqrt(x),
            input_dtype="int8",
            x_data=np.arange(1, 128, dtype="int8"),
        )


class Sqrt:
    def test_all_numbers_uint8(self):
        generic_test(relay.qnn.sqrt, np.sqrt, input_dtype="uint8")

    def test_all_numbers_int8(self):
        generic_test(
            relay.qnn.sqrt,
            np.sqrt,
            input_dtype="int8",
            x_data=np.arange(1, 128, dtype="int8"),
        )


class TestExp:
    def test_all_numbers_uint8(self):
        generic_test(relay.qnn.exp, np.exp, input_dtype="uint8")

    def test_all_numbers_int8(self):
        generic_test(relay.qnn.exp, np.exp, input_dtype="int8")


class TestTanh:
    def test_all_numbers_uint8(self):
        generic_test(relay.qnn.tanh, np.tanh, input_dtype="uint8")

    def test_all_numbers_int8(self):
        generic_test(relay.qnn.tanh, np.tanh, input_dtype="int8")


class TestErf:
    def test_all_numbers_uint8(self):
        generic_test(relay.qnn.erf, scipy.special.erf, input_dtype="uint8")

    def test_all_numbers_int8(self):
        generic_test(relay.qnn.erf, scipy.special.erf, input_dtype="int8")


class TestSigmoid:
    def test_all_numbers_uint8(self):
        generic_test(relay.qnn.sigmoid, lambda x: 1 / (1 + np.exp(-x)), input_dtype="uint8")

    def test_all_numbers_int8(self):
        generic_test(relay.qnn.sigmoid, lambda x: 1 / (1 + np.exp(-x)), input_dtype="int8")


class TestHardswish:
    def test_all_numbers_uint8(self):
        generic_test(relay.qnn.hardswish, hardswish_func, input_dtype="uint8")

    def test_all_numbers_int8(self):
        generic_test(relay.qnn.hardswish, hardswish_func, input_dtype="int8")


if __name__ == "__main__":
    tvm.testing.main()
