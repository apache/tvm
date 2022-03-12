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


def dequantize(data, scale, zp):
    return scale * (np.asarray(data) - zp)


def generate_golden_output(x_data, dequantized_x, alpha, scale, zero_point):
    prod = np.multiply(dequantized_x, alpha)
    prod = np.around(prod / scale + zero_point)

    output = np.where(x_data < zero_point, prod, x_data)
    return output


def test_qnn_leaky_relu():
    data_dtype = "uint8"
    scale = 0.125
    zero_point = 60
    alpha = 0.9

    x = relay.var("x", shape=(1, 4), dtype=data_dtype)
    y = relay.qnn.op.leaky_relu(
        x=x,
        alpha=alpha,
        scale=relay.const(scale, "float32"),
        zero_point=relay.const(zero_point, "int32"),
    )

    func = relay.Function([x], y)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_data = np.array((255, 133, 0, 9)).reshape((1, 4))
    x_dequantized = dequantize(x_data, scale, zero_point)
    golden_output = generate_golden_output(x_data, x_dequantized, alpha, scale, zero_point)

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(x_data)

    np.testing.assert_equal(op_res.numpy(), golden_output)


if __name__ == "__main__":
    test_qnn_leaky_relu()
