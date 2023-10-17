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

# "unquantize" a quantized tensor
def recover(data, scale, zp):
    return scale * (np.asarray(data) - zp)


def generate_golden_output(x_recovered, y_recovered, scale, zp):
    mul = x_recovered * y_recovered
    output = np.around(mul / scale + zp)

    q_min = np.iinfo(np.uint8).min
    q_max = np.iinfo(np.uint8).max
    return np.clip(output, q_min, q_max)


def test_tflite_same_io_qnn_params():
    data_dtype = "uint8"

    lhs_scale = rhs_scale = output_scale = 0.00784314
    lhs_zero_point = rhs_zero_point = output_zero_point = 127

    x = relay.var("x", shape=(1, 4), dtype=data_dtype)
    y = relay.var("y", shape=(1, 4), dtype=data_dtype)
    z = relay.qnn.mul(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(lhs_scale, "float32"),
        lhs_zero_point=relay.const(lhs_zero_point, "int32"),
        rhs_scale=relay.const(rhs_scale, "float32"),
        rhs_zero_point=relay.const(rhs_zero_point, "int32"),
        output_scale=relay.const(output_scale, "float32"),
        output_zero_point=relay.const(output_zero_point, "int32"),
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_datas = [
        np.array((1, 153, 2, 178)).reshape((1, 4)),
        np.array((25, 1, 178, 216)).reshape((1, 4)),
        np.array((25, 153, 1, 165)).reshape((1, 4)),
    ]
    y_datas = [
        np.array((204, 178, 1, 8)).reshape((1, 4)),
        np.array((204, 178, 191, 1)).reshape((1, 4)),
        np.array((204, 178, 1, 191)).reshape((1, 4)),
    ]

    for i in range(0, 3):
        x_data = x_datas[i]
        y_data = y_datas[i]

        x_rec = recover(x_data, lhs_scale, lhs_zero_point)
        y_rec = recover(y_data, rhs_scale, rhs_zero_point)
        golden = generate_golden_output(x_rec, y_rec, output_scale, output_zero_point)

        op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
            x_data, y_data
        )

        np.testing.assert_equal(op_res.numpy(), np.uint8(golden))


def test_tflite_different_io_qnn_params():
    data_dtype = "uint8"

    lhs_scale = 0.0156863
    lhs_zero_point = 127
    rhs_scale = 0.0117647
    rhs_zero_point = 85
    output_scale = 0.0235294
    output_zero_point = 128

    x = relay.var("x", shape=(1, 4), dtype=data_dtype)
    y = relay.var("y", shape=(1, 4), dtype=data_dtype)
    z = relay.qnn.mul(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(lhs_scale, "float32"),
        lhs_zero_point=relay.const(lhs_zero_point, "int32"),
        rhs_scale=relay.const(rhs_scale, "float32"),
        rhs_zero_point=relay.const(rhs_zero_point, "int32"),
        output_scale=relay.const(output_scale, "float32"),
        output_zero_point=relay.const(output_zero_point, "int32"),
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

    for i in range(0, 3):
        x_data = x_datas[i]
        y_data = y_datas[i]

        x_rec = recover(x_data, lhs_scale, lhs_zero_point)
        y_rec = recover(y_data, rhs_scale, rhs_zero_point)
        golden = generate_golden_output(x_rec, y_rec, output_scale, output_zero_point)

        op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
            x_data, y_data
        )
        np.testing.assert_equal(op_res.numpy(), np.uint8(golden))


def test_saturation():
    # Same params
    data_dtype = "uint8"
    lhs_scale = rhs_scale = output_scale = 0.125
    lhs_zero_point = rhs_zero_point = output_zero_point = 0

    x = relay.var("x", shape=(1, 4), dtype=data_dtype)
    y = relay.var("y", shape=(1, 4), dtype=data_dtype)
    z = relay.qnn.mul(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(lhs_scale, "float32"),
        lhs_zero_point=relay.const(lhs_zero_point, "int32"),
        rhs_scale=relay.const(rhs_scale, "float32"),
        rhs_zero_point=relay.const(rhs_zero_point, "int32"),
        output_scale=relay.const(output_scale, "float32"),
        output_zero_point=relay.const(output_zero_point, "int32"),
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_data = np.array((255, 1, 1, 0)).reshape((1, 4))
    y_data = np.array((255, 255, 128, 0)).reshape((1, 4))

    x_rec = recover(x_data, lhs_scale, lhs_zero_point)
    y_rec = recover(y_data, rhs_scale, rhs_zero_point)

    golden = generate_golden_output(x_rec, y_rec, output_scale, output_zero_point)

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, y_data
    )
    np.testing.assert_equal(op_res.numpy(), np.uint8(golden))

    # Same params, different scale

    lhs_scale = rhs_scale = 0.125
    output_scale = 0.25

    z = relay.qnn.mul(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(lhs_scale, "float32"),
        lhs_zero_point=relay.const(lhs_zero_point, "int32"),
        rhs_scale=relay.const(rhs_scale, "float32"),
        rhs_zero_point=relay.const(rhs_zero_point, "int32"),
        output_scale=relay.const(output_scale, "float32"),
        output_zero_point=relay.const(output_zero_point, "int32"),
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_data = np.array((255, 1, 1, 0)).reshape((1, 4))
    y_data = np.array((255, 255, 127, 0)).reshape((1, 4))

    x_rec = recover(x_data, lhs_scale, lhs_zero_point)
    y_rec = recover(y_data, rhs_scale, rhs_zero_point)

    golden = generate_golden_output(x_rec, y_rec, output_scale, output_zero_point)

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, y_data
    )
    np.testing.assert_equal(op_res.numpy(), np.uint8(golden))

    # All params different

    lhs_scale = 0.5
    rhs_scale = 0.25
    output_scale = 0.125

    z = relay.qnn.mul(
        lhs=x,
        rhs=y,
        lhs_scale=relay.const(lhs_scale, "float32"),
        lhs_zero_point=relay.const(lhs_zero_point, "int32"),
        rhs_scale=relay.const(rhs_scale, "float32"),
        rhs_zero_point=relay.const(rhs_zero_point, "int32"),
        output_scale=relay.const(output_scale, "float32"),
        output_zero_point=relay.const(output_zero_point, "int32"),
    )

    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_data = np.array((255, 0, 1, 0)).reshape((1, 4))
    y_data = np.array((0, 128, 64, 0)).reshape((1, 4))

    x_rec = recover(x_data, lhs_scale, lhs_zero_point)
    y_rec = recover(y_data, rhs_scale, rhs_zero_point)

    golden = generate_golden_output(x_rec, y_rec, output_scale, output_zero_point)

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(
        x_data, y_data
    )
    np.testing.assert_equal(op_res.numpy(), np.uint8(golden))


if __name__ == "__main__":
    test_tflite_same_io_qnn_params()
    test_tflite_different_io_qnn_params()
    test_saturation()
