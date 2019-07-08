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
from tvm.relay.testing import create_workload
from tvm.contrib import graph_runtime

# TODOs for janimesh before submitting this patch.
# TODO - Add tests for int8 input/weight dtype
# TODO - opt_level=0 fails mostly due to fusion.
# TODO - opt_level=3 fails, likely culprit kernel layout for int8
# compute. Work with Rankyung to see if this is the culprit. Handle
# it in a separate patch.

def run_infer_type(expr):
    mod = relay.Module.from_expr(expr)
    mod = relay.transform.InferType()(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body

def test_quantize_op():

    def quantize_test_driver(in_dtype, quant_args, out_dtype, in_data, verify_output_data):
        shape = in_data.shape
        input_data = relay.var("input_data", shape=shape, dtype=in_dtype)
        output_zero_point = quant_args['out_zero_point']
        output_scale = quant_args['out_scale']
        quantized_output = relay.op.qnn.quantize(input_data, output_zero_point=output_zero_point,
                                                          output_scale=output_scale, out_dtype=out_dtype)
        func = relay.Function(relay.analysis.free_vars(quantized_output), quantized_output)
        func = run_infer_type(func)
        func = relay.quantize.rewrite(func)
        func = run_infer_type(func)
        graph, lib, params = relay.build(func, "llvm", params=None)
        mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        mod.set_input(input_data=in_data)
        mod.run()
        res = mod.get_output(0).asnumpy()
        np.testing.assert_equal(res, verify_output_data)
        assert res.dtype == out_dtype

    def test_float32_to_uint8():
        data = np.array([-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64]) \
            .astype('float32') \
            .reshape((2,5))
        output = np.array([0, 1, 2, 3, 4, 251, 252, 253, 254, 255]) \
            .astype('uint8') \
            .reshape((2,5))
        quant_args = {"out_zero_point":127, "out_scale":0.5}
        quantize_test_driver(in_dtype='float32', quant_args=quant_args, out_dtype='uint8', in_data=data,
                             verify_output_data=output)

    def test_float32_to_int8():
        data = np.array([-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64]) \
            .astype('float32') \
            .reshape((2,5))
        output = np.array([-128, -127, -126, -125, -124, 123, 124, 125, 126, 127]) \
            .astype('int8') \
            .reshape((2,5))
        quant_args = {"out_zero_point":-1, "out_scale":0.5}
        quantize_test_driver(in_dtype='float32', quant_args=quant_args, out_dtype='int8', in_data=data,
                             verify_output_data=output)

    test_float32_to_uint8()
    test_float32_to_int8()

def test_dequantize_op():

    def quantize_test_driver(in_dtype, quant_args, in_data, verify_output_data):
        shape = in_data.shape
        input_data = relay.var("input_data", shape=shape, dtype=in_dtype)
        input_zero_point = quant_args['in_zero_point']
        input_scale = quant_args['in_scale']
        quantized_output = relay.op.qnn.dequantize(input_data, input_zero_point=input_zero_point,
                                                            input_scale=input_scale)
        func = relay.Function(relay.analysis.free_vars(quantized_output), quantized_output)
        func = run_infer_type(func)
        func = relay.quantize.rewrite(func)
        func = run_infer_type(func)
        graph, lib, params = relay.build(func, "llvm", params=None)
        mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        mod.set_input(input_data=in_data)
        mod.run()
        res = mod.get_output(0).asnumpy()
        np.testing.assert_allclose(res, verify_output_data)
        assert res.dtype == np.float32

    def test_uint8_to_float32():
        data = np.array([0, 1, 2, 3, 4, 251, 252, 253, 254, 255]) \
            .astype('uint8') \
            .reshape((2,5))
        output = np.array([-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64]) \
            .astype('float32') \
            .reshape((2,5))
        quant_args = {"in_zero_point":127, "in_scale":0.5}
        quantize_test_driver(in_dtype='uint8', quant_args=quant_args, in_data=data,
                             verify_output_data=output)

    def test_int8_to_float32():
        data = np.array([-128, -127, -126, -125, -124, 123, 124, 125, 126, 127]) \
            .astype('int8') \
            .reshape((2,5))
        output = np.array([-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64]) \
            .astype('float32') \
            .reshape((2,5))
        quant_args = {"in_zero_point":-1, "in_scale":0.5}
        quantize_test_driver(in_dtype='int8', quant_args=quant_args, in_data=data,
                             verify_output_data=output)

    test_uint8_to_float32()
    test_int8_to_float32()

if __name__ == "__main__":
    test_quantize_op()
    test_dequantize_op()
