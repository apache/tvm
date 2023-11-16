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
import numpy as np
from tvm import relay


def test_tflite_same_io_qnn_params():
    data_dtype = "uint8"

    x = relay.var("x", shape=(1, 4), dtype=data_dtype)
    y = relay.var("y", shape=(1, 4), dtype=data_dtype)
    z = relay.qnn.add(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(0.00784314, "float32"),
        lhs_zero_point=relay.const(127, "int32"),
        rhs_scale=relay.const(0.00784314, "float32"),
        rhs_zero_point=relay.const(127, "int32"),
        output_scale=relay.const(0.00784314, "float32"),
        output_zero_point=relay.const(127, "int32"),
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_datas = [
        np.array((140, 153, 165, 178)).reshape((1, 4)),
        np.array((25, 153, 178, 216)).reshape((1, 4)),
        np.array((25, 153, 216, 165)).reshape((1, 4)),
    ]
    y_datas = [
        np.array((204, 178, 165, 140)).reshape((1, 4)),
        np.array((204, 178, 191, 25)).reshape((1, 4)),
        np.array((204, 178, 25, 191)).reshape((1, 4)),
    ]
    golden_outputs = [
        np.array((217, 204, 203, 191)).reshape((1, 4)),
        np.array((102, 204, 242, 114)).reshape((1, 4)),
        np.array((102, 204, 114, 229)).reshape((1, 4)),
    ]

    for i in range(0, 3):
        x_data = x_datas[i]
        y_data = y_datas[i]
        golden_output = golden_outputs[i]

        op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
            x_data, y_data
        )
        np.testing.assert_equal(op_res.numpy(), golden_output)


def test_tflite_different_io_qnn_params():
    data_dtype = "uint8"

    x = relay.var("x", shape=(1, 4), dtype=data_dtype)
    y = relay.var("y", shape=(1, 4), dtype=data_dtype)
    z = relay.qnn.add(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(0.0156863, "float32"),
        lhs_zero_point=relay.const(127, "int32"),
        rhs_scale=relay.const(0.0117647, "float32"),
        rhs_zero_point=relay.const(85, "int32"),
        output_scale=relay.const(0.0235294, "float32"),
        output_zero_point=relay.const(128, "int32"),
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_datas = [
        np.array((76, 140, 153, 172)).reshape((1, 4)),
        np.array((133, 140, 146, 153)).reshape((1, 4)),
        np.array((76, 140, 172, 146)).reshape((1, 4)),
    ]
    y_datas = [
        np.array((136, 119, 128, 17)).reshape((1, 4)),
        np.array((136, 119, 111, 94)).reshape((1, 4)),
        np.array((136, 119, 17, 128)).reshape((1, 4)),
    ]
    golden_outputs = [
        np.array((120, 154, 167, 124)).reshape((1, 4)),
        np.array((158, 154, 154, 150)).reshape((1, 4)),
        np.array((120, 154, 124, 163)).reshape((1, 4)),
    ]

    for i in range(0, 3):
        x_data = x_datas[i]
        y_data = y_datas[i]
        golden_output = golden_outputs[i]

        op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
            x_data, y_data
        )
        np.testing.assert_equal(op_res.numpy(), golden_output)


def test_saturation():
    # Same params
    data_dtype = "uint8"
    x = relay.var("x", shape=(1, 4), dtype=data_dtype)
    y = relay.var("y", shape=(1, 4), dtype=data_dtype)
    z = relay.qnn.add(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(0.125, "float32"),
        lhs_zero_point=relay.const(0, "int32"),
        rhs_scale=relay.const(0.125, "float32"),
        rhs_zero_point=relay.const(0, "int32"),
        output_scale=relay.const(0.125, "float32"),
        output_zero_point=relay.const(0, "int32"),
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]
    mod = relay.transform.InferType()(mod)

    x_data = np.array((255, 1, 1, 0)).reshape((1, 4))
    y_data = np.array((255, 255, 128, 0)).reshape((1, 4))
    golden_output = np.array((255, 255, 129, 0)).reshape((1, 4))

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, y_data
    )
    np.testing.assert_equal(op_res.numpy(), golden_output)

    # Same params, different scale
    z = relay.qnn.add(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(0.125, "float32"),
        lhs_zero_point=relay.const(0, "int32"),
        rhs_scale=relay.const(0.125, "float32"),
        rhs_zero_point=relay.const(0, "int32"),
        output_scale=relay.const(0.25, "float32"),
        output_zero_point=relay.const(0, "int32"),
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_data = np.array((255, 1, 1, 0)).reshape((1, 4))
    y_data = np.array((255, 255, 127, 0)).reshape((1, 4))
    golden_output = np.array((255, 129, 65, 0)).reshape((1, 4))

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, y_data
    )
    np.testing.assert_equal(op_res.numpy(), golden_output)

    # Same io params, different output scale
    z = relay.qnn.add(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(0.125, "float32"),
        lhs_zero_point=relay.const(0, "int32"),
        rhs_scale=relay.const(0.125, "float32"),
        rhs_zero_point=relay.const(0, "int32"),
        output_scale=relay.const(0.25, "float32"),
        output_zero_point=relay.const(0, "int32"),
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_data = np.array((255, 1, 1, 0)).reshape((1, 4))
    y_data = np.array((255, 255, 127, 0)).reshape((1, 4))
    golden_output = np.array((255, 129, 65, 0)).reshape((1, 4))

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, y_data
    )
    np.testing.assert_equal(op_res.numpy(), golden_output)

    # All params different
    z = relay.qnn.add(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(0.5, "float32"),
        lhs_zero_point=relay.const(0, "int32"),
        rhs_scale=relay.const(0.25, "float32"),
        rhs_zero_point=relay.const(0, "int32"),
        output_scale=relay.const(0.125, "float32"),
        output_zero_point=relay.const(0, "int32"),
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_data = np.array((255, 0, 1, 0)).reshape((1, 4))
    y_data = np.array((0, 128, 64, 0)).reshape((1, 4))
    golden_output = np.array((255, 255, 132, 0)).reshape((1, 4))

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, y_data
    )
    np.testing.assert_equal(op_res.numpy(), golden_output)


def test_ignore_channel_axis():
    data_dtype = "uint8"

    x = relay.var("x", shape=(4,), dtype=data_dtype)
    y = relay.var("y", shape=(4,), dtype=data_dtype)
    z = relay.qnn.add(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(0.00784314, "float32"),
        lhs_zero_point=relay.const(127, "int32"),
        rhs_scale=relay.const(0.00784314, "float32"),
        rhs_zero_point=relay.const(127, "int32"),
        output_scale=relay.const(0.00784314, "float32"),
        output_zero_point=relay.const(127, "int32"),
        lhs_axis=1,
        rhs_axis=1,
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)


if __name__ == "__main__":
    test_tflite_same_io_qnn_params()
    test_tflite_different_io_qnn_params()
    test_saturation()
    test_ignore_channel_axis()
