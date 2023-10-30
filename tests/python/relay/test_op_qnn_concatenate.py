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

import tvm
from tvm import te
import numpy as np
from tvm import relay
from tvm.contrib import graph_executor
import tvm.topi.testing


def test_same_io_qnn_params():
    data_dtype = "int32"
    axis = 0
    x_data = np.arange(-32, 32, 1).reshape(1, 64).astype(data_dtype)
    y_data = np.arange(-64, 64, 2).reshape(1, 64).astype(data_dtype)
    x_scale = relay.const((62 + 64) / (np.power(2, 32) - 1.0), "float32")
    y_scale = relay.const((62 + 64) / (np.power(2, 32) - 1.0), "float32")
    zero = relay.const(0, "int32")

    x = relay.var("x", shape=(1, 64), dtype=data_dtype)
    y = relay.var("y", shape=(1, 64), dtype=data_dtype)
    z = relay.qnn.concatenate(
        (x, y),
        input_scales=(x_scale, y_scale),
        input_zero_points=(zero, zero),
        output_scale=y_scale,
        output_zero_point=zero,
        axis=axis,
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    golden_output = np.concatenate((x_data, y_data), axis=axis)

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, y_data
    )
    np.testing.assert_equal(op_res.numpy(), golden_output)


def test_different_io_qnn_params():
    data_dtype = "int32"
    axis = 0
    x_data = np.arange(-32, 32, 1).reshape(1, 64).astype(data_dtype)
    y_data = np.arange(-64, 64, 2).reshape(1, 64).astype(data_dtype)

    x_scale = relay.const((62 + 64) / (np.power(2, 32) - 1.0), "float32")
    y_scale = relay.const((62 + 64) / (np.power(2, 32) - 1.0), "float32")
    x_zero_point = relay.const(3, "int32")
    y_zero_point = relay.const(4, "int32")

    x = relay.var("x", shape=(1, 64), dtype=data_dtype)
    y = relay.var("y", shape=(1, 64), dtype=data_dtype)
    z = relay.qnn.concatenate(
        (x, y),
        input_scales=(x_scale, y_scale),
        input_zero_points=(x_zero_point, y_zero_point),
        output_scale=y_scale,
        output_zero_point=relay.const(1, "int32"),
        axis=axis,
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    golden_output = np.concatenate((x_data - 2, y_data - 3), axis=axis)

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, y_data
    )
    np.testing.assert_equal(op_res.numpy(), golden_output)


def test_few_same_io_qnn_params():
    data_dtype = "int32"
    axis = 0
    x_data = np.arange(-32, 32, 1).reshape(1, 64).astype(data_dtype)
    y_data = np.arange(-64, 64, 2).reshape(1, 64).astype(data_dtype)

    x_scale = relay.const((62 + 64) / (np.power(2, 32) - 1.0), "float32")
    y_scale = relay.const((62 + 64) / (np.power(2, 32) - 1.0), "float32")
    x_zero_point = relay.const(0, "int32")
    y_zero_point = relay.const(1, "int32")

    x = relay.var("x", shape=(1, 64), dtype=data_dtype)
    y = relay.var("y", shape=(1, 64), dtype=data_dtype)
    z = relay.qnn.concatenate(
        (x, y),
        input_scales=(x_scale, y_scale),
        input_zero_points=(x_zero_point, y_zero_point),
        output_scale=y_scale,
        output_zero_point=relay.const(1, "int32"),
        axis=axis,
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    golden_output = np.concatenate((x_data + 1, y_data), axis=axis)

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, y_data
    )
    np.testing.assert_equal(op_res.numpy(), golden_output)


def test_same_i_qnn_params():
    data_dtype = "int32"
    axis = 0
    x_data = np.arange(-32, 32, 1).reshape(1, 64).astype(data_dtype)
    y_data = np.arange(-64, 64, 2).reshape(1, 64).astype(data_dtype)

    x_scale = relay.const((62 + 64) / (np.power(2, 32) - 1.0), "float32")
    y_scale = relay.const((62 + 64) / (np.power(2, 32) - 1.0), "float32")
    x_zero_point = relay.const(0, "int32")
    y_zero_point = relay.const(0, "int32")

    x = relay.var("x", shape=(1, 64), dtype=data_dtype)
    y = relay.var("y", shape=(1, 64), dtype=data_dtype)
    z = relay.qnn.concatenate(
        (x, y),
        input_scales=(x_scale, y_scale),
        input_zero_points=(x_zero_point, y_zero_point),
        output_scale=y_scale,
        output_zero_point=relay.const(1, "int32"),
        axis=axis,
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    golden_output = np.concatenate((x_data + 1, y_data + 1), axis=axis)

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, y_data
    )
    np.testing.assert_equal(op_res.numpy(), golden_output)


def test_call_input():
    # This tests the case where the input to concatenate is not explicitly a
    # tuple node but is instead a call node.
    x_data = np.ones(shape=(64,)).astype("uint8")

    x = relay.var("x", shape=(64,), dtype="uint8")
    x_scale = relay.const(1, "float32")
    y_scale = relay.const(1, "float32")
    x_zero_point = relay.const(0, "int32")
    y_zero_point = relay.const(0, "int32")

    tup = relay.split(x, 2, axis=0)
    z = relay.qnn.concatenate(
        tup,
        input_scales=(x_scale, y_scale),
        input_zero_points=(x_zero_point, y_zero_point),
        output_scale=y_scale,
        output_zero_point=relay.const(0, "int32"),
        axis=0,
    )
    func = relay.Function([x], z)

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(x_data)
    np.testing.assert_equal(op_res.numpy(), x_data)


if __name__ == "__main__":
    test_call_input()
    test_same_io_qnn_params()
    test_different_io_qnn_params()
    test_few_same_io_qnn_params()
    test_same_i_qnn_params()
