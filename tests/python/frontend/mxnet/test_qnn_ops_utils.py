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


def test_mxnet_quantization():
    def quantize_test_driver(in_dtype, quant_args, out_dtype, in_data, verify_output_data):
        shape = in_data.shape
        input_data = relay.var("input_data", shape=shape, dtype=in_dtype)
        min_range = quant_args['min_range']
        max_range = quant_args['max_range']
        quantized_output, _, _ = relay.frontend.quantize_mxnet_min_max(input_data,
                                                                       min_range=min_range,
                                                                       max_range=max_range,
                                                                       out_dtype=out_dtype)
        mod = relay.Function(relay.analysis.free_vars(quantized_output), quantized_output)
        mod = relay.Module.from_expr(mod)
        mod = relay.qnn.transform.CanonicalizeOps()(mod)
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, "llvm", params=None)
            rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
            rt_mod.set_input(input_data=in_data)
            rt_mod.set_input(**params)
            rt_mod.run()
            res = rt_mod.get_output(0).asnumpy()
            np.testing.assert_equal(res, verify_output_data)
            assert res.dtype == out_dtype

    def test_float32_to_uint8():
        data = np.array([-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64]) \
            .astype('float32') \
            .reshape((2, 5))
        output = np.array([0, 1, 2, 3, 4, 251, 252, 253, 254, 255]) \
            .astype('uint8') \
            .reshape((2, 5))
        quant_args = {"min_range": -63.5, "max_range": 64}
        quantize_test_driver(in_dtype='float32', quant_args=quant_args, out_dtype='uint8',
                             in_data=data, verify_output_data=output)

    def test_float32_to_int8():
        data = np.array([-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64]) \
            .astype('float32') \
            .reshape((2, 5))
        output = np.array([-126, -125, -124, -123, -122, 123, 124, 125, 126, 127]) \
            .astype('int8') \
            .reshape((2, 5))
        quant_args = {"min_range": -63.5, "max_range": 64}
        quantize_test_driver(in_dtype='float32', quant_args=quant_args, out_dtype='int8',
                             in_data=data, verify_output_data=output)

    test_float32_to_uint8()
    test_float32_to_int8()


def test_mxnet_mkldnn_quantization():
    def quantize_test_driver(in_dtype, quant_args, out_dtype, in_data, verify_output_data):
        shape = in_data.shape
        input_data = relay.var("input_data", shape=shape, dtype=in_dtype)
        min_range = quant_args['min_range']
        max_range = quant_args['max_range']
        quantized_output, _, _ = relay.frontend.quantize_mxnet_min_max(input_data,
                                                                       min_range=min_range,
                                                                       max_range=max_range,
                                                                       out_dtype=out_dtype,
                                                                       use_mkldnn=True)
        mod = relay.Function(relay.analysis.free_vars(quantized_output), quantized_output)
        mod = relay.Module.from_expr(mod)
        mod = relay.qnn.transform.CanonicalizeOps()(mod)
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, "llvm", params=None)
            rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
            rt_mod.set_input(input_data=in_data)
            rt_mod.set_input(**params)
            rt_mod.run()
            res = rt_mod.get_output(0).asnumpy()
            np.testing.assert_equal(res, verify_output_data)
            assert res.dtype == out_dtype

    def test_float32_to_uint8():
        data = np.array([-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64]) \
            .astype('float32') \
            .reshape((2, 5))
        output = np.array([0, 0, 0, 0, 0, 247, 249, 251, 253, 255]) \
            .astype('uint8') \
            .reshape((2, 5))
        quant_args = {"min_range": -63.5, "max_range": 64}
        quantize_test_driver(in_dtype='float32', quant_args=quant_args, out_dtype='uint8',
                             in_data=data, verify_output_data=output)

    def test_float32_to_int8():
        data = np.array([-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64]) \
            .astype('float32') \
            .reshape((2, 5))
        output = np.array([-126, -125, -124, -123, -122, 123, 124, 125, 126, 127]) \
            .astype('int8') \
            .reshape((2, 5))
        quant_args = {"min_range": -63.5, "max_range": 64}
        quantize_test_driver(in_dtype='float32', quant_args=quant_args, out_dtype='int8',
                             in_data=data, verify_output_data=output)

    test_float32_to_uint8()
    test_float32_to_int8()


def test_mxnet_conv_weight_quantization_mkldnn():
    def quantize_test_driver(out_dtype, in_data, verify_output_data):
        quantized_output, _, _ = relay.frontend.quantize_conv_weights_mkldnn(in_data,
                                                                             "input_data")
        mod = relay.Function(relay.analysis.free_vars(quantized_output), quantized_output)
        mod = relay.Module.from_expr(mod)
        mod = relay.qnn.transform.CanonicalizeOps()(mod)
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, "llvm", params=None)
            rt_mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
            rt_mod.set_input(input_data=in_data)
            rt_mod.set_input(**params)
            rt_mod.run()
            res = rt_mod.get_output(0).asnumpy()
            np.testing.assert_equal(res, verify_output_data)
            assert res.dtype == out_dtype

    def test_float32_to_int8():
        data = np.array([0.0441604, 0.03017418, 0.03101145, 0.03285711, 0.0184189, 0.0333233,
                         0.02895038, 0.01649691, 0.01324425, 0.01096264, 0.01516934, 0.00323179,
                         -0.01969179, -0.02577864, -0.02674193, -0.02682905, -0.05210099, -0.05635381,
                         -0.04693264, -0.04124459]) \
            .astype('float32') \
            .reshape((5, 1, 2, 2))
        output = np.array([100, 68, 70, 74,  42, 75, 65, 37, 30, 25, 34, 7, -44, -58,
                           -60, -60, -117, -127, -106, -93]) \
            .astype('int8') \
            .reshape((5, 1, 2, 2))
        quantize_test_driver(out_dtype='int8',
                             in_data=data, verify_output_data=output)

    test_float32_to_int8()


def test_get_scale():

    def test_uint8_scale():
        scale = relay.frontend.get_mkldnn_uint8_scale(0.000407, 0.999356)
        expected_scale = 0.00391904
        assert np.allclose(expected_scale, scale)

    def test_int8_scale():
        scale = relay.frontend.get_mkldnn_int8_scale(0.000407, 0.999356)
        expected_scale = 0.007868945
        assert np.allclose(expected_scale, scale)

    test_uint8_scale()
    test_int8_scale()


if __name__ == "__main__":
    test_mxnet_quantization()
    test_mxnet_mkldnn_quantization()
    test_mxnet_conv_weight_quantization_mkldnn()
    test_get_scale()
