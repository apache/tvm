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


def test_mxnet_dequantize_op():

    def quantize_test_driver(in_dtype, quant_args, in_data, verify_output_data):
        shape = in_data.shape
        input_data = relay.var("input_data", shape=shape, dtype=in_dtype)
        min_range = quant_args['min_range']
        max_range = quant_args['max_range']
        quantized_output = \
            relay.frontend.dequantize_mxnet_min_max(input_data,
                                                    min_range=min_range,
                                                    max_range=max_range,
                                                    in_dtype=in_dtype)
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
            assert np.allclose(res, verify_output_data, )
            assert res.dtype == np.float32

    def test_uint8_to_float32():
        data = np.array([0, 1, 2, 3, 4, 251, 252, 253, 254, 255]) \
            .astype('uint8') \
            .reshape((2, 5))
        output = np.array([-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64]) \
            .astype('float32') \
            .reshape((2, 5))
        quant_args = {"min_range": -63.5, "max_range": 64}
        quantize_test_driver(in_dtype='uint8',
                             quant_args=quant_args,
                             in_data=data,
                             verify_output_data=output)

    def test_int8_to_float32():
        data = np.array([-126, -125, -124, -123, -122, 123, 124, 125, 126, 127]) \
            .astype('int8') \
            .reshape((2, 5))
        output = np.array([-63.496063, -62.992126, -62.48819, -61.984253, -61.480316,
                           61.984253, 62.48819, 62.992126, 63.496063, 64.]) \
            .astype('float32') \
            .reshape((2, 5))
        quant_args = {"min_range": -63.5, "max_range": 64}
        quantize_test_driver(in_dtype='int8',
                             quant_args=quant_args,
                             in_data=data,
                             verify_output_data=output)

    test_uint8_to_float32()
    test_int8_to_float32()


def test_mkldnn_dequantize_op():

    def quantize_test_driver(in_dtype, quant_args, in_data, verify_output_data):
        shape = in_data.shape
        input_data = relay.var("input_data", shape=shape, dtype=in_dtype)
        min_range = quant_args['min_range']
        max_range = quant_args['max_range']
        quantized_output = \
            relay.frontend.dequantize_mxnet_min_max(input_data,
                                                    min_range=min_range,
                                                    max_range=max_range,
                                                    in_dtype=in_dtype,
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
            # print(res)
            # np.testing.assert_equal(res, verify_output_data)
            assert np.allclose(res, verify_output_data, )
            assert res.dtype == np.float32

    def test_uint8_to_float32():
        data = np.array([0, 1, 2, 3, 4, 251, 252, 253, 254, 255]) \
            .astype('uint8') \
            .reshape((2, 5))
        output = np.array([0., 0.2509804, 0.5019608, 0.75294125, 1.0039216,
                           62.996082, 63.247063, 63.498043, 63.749023, 64.]) \
            .astype('float32') \
            .reshape((2, 5))
        quant_args = {"min_range": -63.5, "max_range": 64}
        quantize_test_driver(in_dtype='uint8',
                             quant_args=quant_args,
                             in_data=data,
                             verify_output_data=output)

    def test_int8_to_float32():
        data = np.array([-126, -125, -124, -123, -122, 123, 124, 125, 126, 127]) \
            .astype('int8') \
            .reshape((2, 5))
        output = np.array([-63.496063, -62.992126, -62.48819, -61.984253, -61.480316,
                           61.984253, 62.48819, 62.992126, 63.496063, 64.]) \
            .astype('float32') \
            .reshape((2, 5))
        quant_args = {"min_range": -63.5, "max_range": 64}
        quantize_test_driver(in_dtype='int8',
                             quant_args=quant_args,
                             in_data=data,
                             verify_output_data=output)

    test_uint8_to_float32()
    test_int8_to_float32()


if __name__ == "__main__":
    test_mxnet_dequantize_op()
    test_mkldnn_dequantize_op()
