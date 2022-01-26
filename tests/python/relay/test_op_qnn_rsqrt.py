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


def generate_golden_output(dequantized_x, output_scale, output_zero_point):
    rsqrt = 1 / np.sqrt(dequantized_x)
    output = np.around(rsqrt / output_scale + output_zero_point)

    q_min = np.iinfo(np.uint8).min
    q_max = np.iinfo(np.uint8).max
    return np.clip(output, q_min, q_max)


def test_saturation():
    # Same params
    data_dtype = "uint8"
    scale = output_scale = 0.125
    zero_point = output_zero_point = 0

    x = relay.var("x", shape=(1, 4), dtype=data_dtype)
    y = relay.qnn.op.rsqrt(
        x=x,
        scale=relay.const(scale, "float32"),
        zero_point=relay.const(zero_point, "int32"),
        output_scale=relay.const(output_scale, "float32"),
        output_zero_point=relay.const(output_zero_point, "int32"),
    )

    func = relay.Function([x], y)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_data = np.array((255, 133, 0, 9)).reshape((1, 4))
    x_dequantized = dequantize(x_data, scale, zero_point)
    golden_output = generate_golden_output(x_dequantized, output_scale, output_zero_point)

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(x_data)

    np.testing.assert_equal(op_res.numpy(), np.uint8(golden_output))

    # Different scale
    scale = 0.125
    output_scale = 0.25

    y = relay.qnn.op.rsqrt(
        x=x,
        scale=relay.const(scale, "float32"),
        zero_point=relay.const(zero_point, "int32"),
        output_scale=relay.const(output_scale, "float32"),
        output_zero_point=relay.const(output_zero_point, "int32"),
    )

    func = relay.Function([x], y)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    func = mod["main"]

    x_data = np.array((255, 133, 0, 9)).reshape((1, 4))
    x_dequantized = dequantize(x_data, scale, zero_point)
    golden_output = generate_golden_output(x_dequantized, output_scale, output_zero_point)

    op_res = relay.create_executor("graph", device=tvm.cpu(0), target="llvm").evaluate(func)(x_data)

    np.testing.assert_equal(op_res.numpy(), golden_output)


if __name__ == "__main__":
    test_saturation()
