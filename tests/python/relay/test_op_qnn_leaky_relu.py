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


def generate_golden_output(x_data, dequantized_x, alpha, o_scale, o_zero_point, i_zero_point):
    prod = np.multiply(dequantized_x, alpha)
    prod = np.around(prod / o_scale + o_zero_point)

    q_min = np.iinfo(np.uint8).min
    q_max = np.iinfo(np.uint8).max
    prod = np.clip(prod, q_min, q_max)

    requantized = np.clip(np.round(dequantized_x / o_scale + o_zero_point), q_min, q_max)

    output = np.where(x_data < i_zero_point, prod, requantized)
    return output


def test_qnn_leaky_relu():
    data_dtype = "uint8"
    input_scale = 0.125
    input_zero_point = 60
    output_scale = 0.6
    output_zero_point = 17
    alpha = 0.9

    x = relay.var("x", shape=(1, 4), dtype=data_dtype)
    y = relay.qnn.leaky_relu(
        x=x,
        alpha=alpha,
        input_scale=relay.const(input_scale, "float32"),
        input_zero_point=relay.const(input_zero_point, "int32"),
        output_scale=relay.const(output_scale, "float32"),
        output_zero_point=relay.const(output_zero_point, "int32"),
    )

    func = relay.Function([x], y)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_data = np.array((255, 133, 0, 9)).reshape((1, 4))
    x_dequantized = dequantize(x_data, input_scale, input_zero_point)
    golden_output = generate_golden_output(
        x_data, x_dequantized, alpha, output_scale, output_zero_point, input_zero_point
    )

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(x_data)

    np.testing.assert_allclose(op_res.numpy(), golden_output, atol=1)


if __name__ == "__main__":
    test_qnn_leaky_relu()
