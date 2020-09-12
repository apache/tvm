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
from tvm.contrib import graph_runtime


def dequantize_test_driver(in_dtype, quant_args, in_data, verify_output_data, axis):
    shape = in_data.shape
    input_data = relay.var("input_data", shape=shape, dtype=in_dtype)
    input_zero_point = relay.const(quant_args["in_zero_point"], "int32")
    input_scale = relay.const(quant_args["in_scale"], "float32")
    quantized_output = relay.qnn.op.dequantize(
        input_data, input_scale=input_scale, input_zero_point=input_zero_point, axis=axis
    )
    mod = relay.Function(relay.analysis.free_vars(quantized_output), quantized_output)
    mod = tvm.IRModule.from_expr(mod)
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod, "llvm", params=None)
        rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        rt_mod.set_input(input_data=in_data)
        rt_mod.set_input(**params)
        rt_mod.run()
        res = rt_mod.get_output(0).asnumpy()
        np.testing.assert_equal(res, verify_output_data)
        assert res.dtype == np.float32


def test_uint8_to_float32():
    data = np.array([0, 1, 2, 3, 4, 251, 252, 253, 254, 255]).astype("uint8").reshape((2, 5))
    output = (
        np.array([-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64])
        .astype("float32")
        .reshape((2, 5))
    )
    quant_args = {"in_zero_point": 127, "in_scale": 0.5}
    dequantize_test_driver(
        in_dtype="uint8", quant_args=quant_args, in_data=data, verify_output_data=output, axis=-1
    )


def test_int8_to_float32():
    data = (
        np.array([-128, -127, -126, -125, -124, 123, 124, 125, 126, 127])
        .astype("int8")
        .reshape((2, 5))
    )
    output = (
        np.array([-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64])
        .astype("float32")
        .reshape((2, 5))
    )
    quant_args = {"in_zero_point": -1, "in_scale": 0.5}
    dequantize_test_driver(
        in_dtype="int8", quant_args=quant_args, in_data=data, verify_output_data=output, axis=-1
    )


def test_int32_to_float32():
    data = np.array([113, 29, -1052]).astype("int32")
    output = np.array([0.6550452, 0.16810896, -6.098297]).astype("float32")
    quant_args = {"in_zero_point": 0, "in_scale": 0.0057968604}
    dequantize_test_driver(
        in_dtype="int32", quant_args=quant_args, in_data=data, verify_output_data=output, axis=-1
    )


def test_channelwise_axis_1():
    data = np.transpose(
        np.array([0, 1, 2, 3, 4, 243, 247, 249, 250, 251]).astype("uint8").reshape((2, 5))
    )
    output = np.transpose(
        np.array([-63.5, -63, -62.5, -62, -61.5, 30, 31, 31.5, 31.75, 32])
        .astype("float32")
        .reshape((2, 5))
    )
    quant_args = {
        "in_zero_point": np.array([127, 123]).astype("int32"),
        "in_scale": np.array([0.5, 0.25]).astype("float32"),
    }

    dequantize_test_driver(
        in_dtype="uint8", quant_args=quant_args, in_data=data, verify_output_data=output, axis=1
    )


if __name__ == "__main__":
    test_uint8_to_float32()
    test_int8_to_float32()
    test_int32_to_float32()
    test_channelwise_axis_1()
