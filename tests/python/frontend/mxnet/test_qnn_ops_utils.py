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
from tvm.contrib import graph_runtime
from tvm.relay.frontend.mxnet_qnn_op_utils import dequantize_mxnet_min_max, \
                                                  quantize_mxnet_min_max,   \
                                                  get_mkldnn_int8_scale,    \
                                                  get_mkldnn_uint8_scale,   \
                                                  quantize_conv_bias_mkldnn_from_var


def test_mkldnn_dequantize():

    def dequantize_test_driver(in_dtype, quant_args, in_data, verify_output_data):
        shape = in_data.shape
        input_data = relay.var("input_data", shape=shape, dtype=in_dtype)
        min_range = quant_args['min_range']
        max_range = quant_args['max_range']
        dequantized_output = dequantize_mxnet_min_max(input_data,
                                                      min_range=min_range,
                                                      max_range=max_range,
                                                      in_dtype=in_dtype)
        mod = relay.Function(relay.analysis.free_vars(dequantized_output), dequantized_output)
        mod = tvm.IRModule.from_expr(mod)
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, "llvm", params=None)
            rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
            rt_mod.set_input(input_data=in_data)
            rt_mod.set_input(**params)
            rt_mod.run()
            res = rt_mod.get_output(0).asnumpy()
            assert np.allclose(res, verify_output_data)
            assert res.dtype == np.float32

    def test_uint8_to_float32():
        data = np.array([0, 1, 2, 3, 4, 251, 252, 253, 254, 255]) \
            .astype('uint8') \
            .reshape((2, 5))
        output = np.array([0., 0.25048923, 0.50097847, 0.7514677, 1.0019569, 62.8728, 63.123287,
                           63.373775, 63.624268, 63.874756]) \
            .astype('float32') \
            .reshape((2, 5))
        quant_args = {"min_range": -63.5, "max_range": 64}
        dequantize_test_driver(in_dtype='uint8',
                               quant_args=quant_args,
                               in_data=data,
                               verify_output_data=output)

    def test_int8_to_float32():
        data = np.array([-126, -125, -124, -123, -122, 123, 124, 125, 126, 127]) \
            .astype('int8') \
            .reshape((2, 5))
        output = np.array([-63.247063, -62.745102, -62.24314, -61.74118, -61.23922,
                           61.74118, 62.24314, 62.745102, 63.247063, 63.749023]) \
            .astype('float32') \
            .reshape((2, 5))
        dequantize_args = {"min_range": -63.5, "max_range": 64}
        dequantize_test_driver(in_dtype='int8',
                               quant_args=dequantize_args,
                               in_data=data,
                               verify_output_data=output)

    test_uint8_to_float32()
    test_int8_to_float32()


def test_mkldnn_quantize():
    def quantize_test_driver(out_dtype, quant_args, in_data, verify_output_data):
        shape = in_data.shape
        input_data = relay.var("input_data", shape=shape, dtype='float32')
        min_range = quant_args['min_range']
        max_range = quant_args['max_range']
        quantized_output, _, _ = quantize_mxnet_min_max(input_data,
                                                        min_range=min_range,
                                                        max_range=max_range,
                                                        out_dtype=out_dtype)
        mod = relay.Function(relay.analysis.free_vars(quantized_output), quantized_output)
        mod = tvm.IRModule.from_expr(mod)
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, "llvm", params=None)
            rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
            rt_mod.set_input(input_data=in_data)
            rt_mod.set_input(**params)
            rt_mod.run()
            res = rt_mod.get_output(0).asnumpy()
            assert np.allclose(res, verify_output_data)
            assert res.dtype == verify_output_data.dtype

    def test_float32_to_uint8():
        data = np.array([0., 0.25048923, 0.50097847, 0.7514677, 1.0019569, 62.8728, 63.123287,
                         63.373775, 63.624268, 63.874756]) \
            .astype('float32') \
            .reshape((2, 5))
        output = np.array([0, 1, 2, 3, 4, 251, 252, 253, 254, 255]) \
            .astype('uint8') \
            .reshape((2, 5))

        quant_args = {"min_range": -63.5, "max_range": 64}
        quantize_test_driver(out_dtype='uint8',
                             quant_args=quant_args,
                             in_data=data,
                             verify_output_data=output)

    def test_float32_to_int8():
        data = np.array([-63.247063, -62.745102, -62.24314, -61.74118, -61.23922,
                         61.74118, 62.24314, 62.745102, 63.247063, 63.749023]) \
            .astype('float32') \
            .reshape((2, 5))
        output = np.array([-126, -125, -124, -123, -122, 123, 124, 125, 126, 127]) \
            .astype('int8') \
            .reshape((2, 5))

        quant_args = {"min_range": -63.5, "max_range": 64}
        quantize_test_driver(out_dtype='int8',
                             quant_args=quant_args,
                             in_data=data,
                             verify_output_data=output)

    test_float32_to_uint8()
    test_float32_to_int8()


def test_get_mkldnn_int8_scale():
    range_min = -3.904039
    range_max = 3.904039
    expected = 0.03061991354976495
    output = get_mkldnn_int8_scale(range_max=range_max,
                                   range_min=range_min)
    assert np.allclose(output, expected)


def test_get_mkldnn_uint8_scale():
    range_min = 0.0
    range_max = 55.77269
    expected = 0.21828841189047482
    output = get_mkldnn_uint8_scale(range_max=range_max,
                                    range_min=range_min)
    assert np.allclose(output, expected)


def test_quantize_conv_bias_mkldnn_from_var():
    bias_var = relay.var('bias', shape=(3,), dtype='float32')
    bias_scale = tvm.nd.array(np.array([0.5, 0.6, 0.7]))
    output = quantize_conv_bias_mkldnn_from_var(bias_var, bias_scale)
    assert isinstance(output, tvm.relay.expr.Call)
    attrs = output.attrs
    assert attrs.axis == 0
    assert attrs.out_dtype == 'int32'
    assert output.op.name == 'qnn.quantize'
    assert output.args[1].data == bias_scale


if __name__ == "__main__":
    test_mkldnn_dequantize()
    test_mkldnn_quantize()
    test_get_mkldnn_int8_scale()
    test_get_mkldnn_uint8_scale()
    test_quantize_conv_bias_mkldnn_from_var()
