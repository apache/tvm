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
import tvm
from tvm import relay


def dequantize(data, scale, zp):
    return scale * (np.asarray(data) - zp)


def generate_golden_output(
    floating_point_golden_func, dequantized_x, output_scale, output_zero_point
):
    output = floating_point_golden_func(dequantized_x)
    output = np.around(output / output_scale + output_zero_point)

    q_min = np.iinfo(np.uint8).min
    q_max = np.iinfo(np.uint8).max
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
    input_scale: float,
    input_zero_point: int,
    output_scale: float,
    output_zero_point: int,
    input_dtype: str = "uint8",
):
    x = relay.var("x", shape=(1, 4), dtype=input_dtype)
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
        input_scale=input_scale,
        input_zero_point=input_zero_point,
        output_scale=output_scale,
        output_zero_point=output_zero_point,
        input_dtype=input_dtype,
    )

    x_data = np.array((255, 133, 0, 9)).reshape((1, 4))
    x_dequantized = dequantize(x_data, input_scale, input_zero_point)
    golden_output = generate_golden_output(
        floating_point_golden_func, x_dequantized, output_scale, output_zero_point
    )

    op_res = run_qnn_func(func, [x_data])

    np.testing.assert_equal(op_res, np.uint8(golden_output))


class TestRSqrt:
    def test_saturation(self):
        # Same qparams in and out
        x_data = np.array((255, 133, 0, 9)).reshape((1, 4))
        run_condition(
            relay.qnn.op.rsqrt,
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
            relay.qnn.op.rsqrt,
            lambda x: 1 / np.sqrt(x),
            x_data,
            input_scale=0.125,
            input_zero_point=0,
            output_scale=0.25,
            output_zero_point=0,
            input_dtype="uint8",
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
