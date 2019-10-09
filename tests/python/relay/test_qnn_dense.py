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


def make_requantize_params(input_scale, output_scale, output_zero_point, out_dtype):
    config = {
        'input_scale': input_scale,
        'output_scale': output_scale,
        'output_zero_point': output_zero_point,
        'out_dtype': out_dtype
    }
    return config


def make_configuration(quantized_data, 
                       quantized_kernel,
                       dtype,
                       input_shape,
                       kernel_shape,
                       input_zero_point,
                       kernel_zero_point,
                       units,
                       output,
                       out_dtype='int32',
                       bias=None,
                       requantize=None):
    if requantize is not None:
        assert bias is not None
    config = {
        'quantized_data': quantized_data,
        'quantized_kernel': quantized_kernel,
        'dtype': dtype,
        'input_shape': input_shape,
        'kernel_shape': kernel_shape,
        'input_zero_point': input_zero_point,
        'kernel_zero_point': kernel_zero_point,
        'units': units,
        'output': output,
        'out_dtype': out_dtype,
        'bias': bias,
        'requantize': requantize
    }
    return config


def make_uint_configuration(use_bias=False, requantize_output=False):
    input_shape, kernel_shape, output_shape = (2, 10), (3,10), (2, 3)
    input_zero_point, kernel_zero_point = 127, 127
    in_dtype = 'uint8'
    out_dtype = 'int32' if not requantize_output else 'uint8'
    units = 3
    quantized_data_np = np.array([129, 131, 133, 135, 137, 139, 141, 143, 109, 107,
                                  129, 131, 133, 135, 137, 139, 141, 111, 145, 107]) \
        .astype(in_dtype) \
        .reshape(input_shape)
    quantized_kernel_np = np.array([129, 131, 133, 135, 137, 139, 141, 143, 145, 147,
                                    129, 131, 133, 135, 137, 139, 141, 143, 145, 147,
                                    129, 131, 133, 135, 137, 139, 141, 143, 145, 147]) \
        .astype(in_dtype) \
        .reshape(kernel_shape)
    bias = np.array([4, 8, 12]).astype(out_dtype).reshape((units, )) if use_bias else None
    requant_params = make_requantize_params(0.25, 1.0, 127, 'uint8') if requantize_output else None

    if requantize_output:
        assert use_bias
        output = np.array([151, 152, 153, 185, 186, 187])
    elif use_bias:
        output = np.array([96, 100, 104, 232, 236, 240 ])
    else:
        output = np.array([92, 92, 92, 228, 228, 228 ])
    output = output.astype(out_dtype).reshape(output_shape)
    return make_configuration(quantized_data=quantized_data_np,
                              quantized_kernel=quantized_kernel_np,
                              dtype=in_dtype,
                              input_shape=input_shape,
                              kernel_shape=kernel_shape,
                              input_zero_point=input_zero_point,
                              kernel_zero_point=kernel_zero_point,
                              units=units,
                              output=output,
                              bias=bias,
                              requantize=requant_params)


def make_int_configuration(use_bias=False, requantize_output=False):
    input_shape, kernel_shape, output_shape = (2, 10), (3,10), (2, 3)
    input_zero_point, kernel_zero_point = -1, -1
    in_dtype = 'int8'
    out_dtype = 'int32' if not requantize_output else 'int8'
    units = 3
    quantized_data_np = np.array([1, 3, 5, 7, 9, 11, 13, 15, -19, -21,
                                  1, 3, 5, 7, 9, 11, 13, -17, 17, -21]) \
        .astype(in_dtype) \
        .reshape(input_shape)
    quantized_kernel_np = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
                                    1, 3, 5, 7, 9, 11, 13, 15, 17, 19,
                                    1, 3, 5, 7, 9, 11, 13, 15, 17, 19]) \
        .astype(in_dtype) \
        .reshape(kernel_shape)
    bias = np.array([4, 8, 12]).astype(out_dtype).reshape((units, )) if use_bias else None
    requant_params = make_requantize_params(0.25, 1.0, -1, 'int8') if requantize_output else None

    if requantize_output:
        assert use_bias
        output = np.array([23, 24, 25, 57, 58, 59])
    elif use_bias:
        output = np.array([96, 100, 104, 232, 236, 240 ])
    else:
        output = np.array([92, 92, 92, 228, 228, 228 ])
    output = output.astype(out_dtype).reshape(output_shape)
    return make_configuration(quantized_data=quantized_data_np,
                              quantized_kernel=quantized_kernel_np,
                              dtype=in_dtype,
                              input_shape=input_shape,
                              kernel_shape=kernel_shape,
                              input_zero_point=input_zero_point,
                              kernel_zero_point=kernel_zero_point,
                              units=units,
                              output=output,
                              bias=bias,
                              requantize=requant_params)


def qnn_dense_driver(test_configuration):
    in_dtype = test_configuration['dtype']
    out_dtype = test_configuration['out_dtype']
    quantized_data_name = "quantized_data"
    quantized_kernel_name = "quantized_kernel"
    expected_out_dtype = test_configuration['out_dtype']
    bias_name = 'bias'
    quantized_data = relay.var(quantized_data_name,
                               shape=test_configuration['input_shape'],
                               dtype=in_dtype)
    quantized_kernel = relay.var(quantized_kernel_name,
                                 shape=test_configuration['kernel_shape'],
                                 dtype=in_dtype)
    mod = relay.qnn.op.dense(
        quantized_data,
        quantized_kernel,
        test_configuration['input_zero_point'],
        test_configuration['kernel_zero_point'],
        test_configuration['units'])
    if test_configuration[bias_name] is not None:
        bias = relay.var(bias_name,
                         shape=test_configuration['bias'].shape,
                         dtype=out_dtype)
        mod = relay.nn.bias_add(mod, bias)
    if test_configuration['requantize'] is not None:
        requantize_config = test_configuration['requantize']
        mod = relay.qnn.op.requantize(
            mod,
            input_scale=requantize_config['input_scale'],
            input_zero_point=0,
            output_scale=requantize_config['output_scale'],
            output_zero_point=requantize_config['output_zero_point'],
            out_dtype=requantize_config['out_dtype'])
        expected_out_dtype = requantize_config['out_dtype']

    mod = relay.Function(relay.analysis.free_vars(mod), mod)
    mod = relay.Module.from_expr(mod)
    mod = relay.qnn.transform.CanonicalizeOps()(mod)
    with relay.build_config(opt_level=2):
        graph, lib, params = relay.build(mod, "llvm", params=None)
        mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        mod.set_input(quantized_data_name,test_configuration[quantized_data_name])
        mod.set_input(quantized_kernel_name,test_configuration[quantized_kernel_name])
        if test_configuration[bias_name] is not None:
            mod.set_input(bias_name, test_configuration[bias_name])
        mod.set_input(**params)
        mod.run()
        res = mod.get_output(0).asnumpy()
        np.testing.assert_equal(res, test_configuration['output'])
        assert res.dtype == expected_out_dtype


def test_qnn_dense_without_bias():
    uint32_output_without_bias_paramas = \
        make_uint_configuration(use_bias=False)
    int32_output_without_bias_params = \
        make_int_configuration(use_bias=False)
    qnn_dense_driver(uint32_output_without_bias_paramas)
    qnn_dense_driver(int32_output_without_bias_params)


def test_qnn_dense_with_bias():
    uint32_output_with_bias_params = \
        make_uint_configuration(use_bias=True)
    int32_output_with_bias_params = \
        make_int_configuration(use_bias=True)
    qnn_dense_driver(uint32_output_with_bias_params)
    qnn_dense_driver(int32_output_with_bias_params)


def test_qnn_dense_with_requantized_output():
    uint8_requantized_output_with_bias_params = \
        make_uint_configuration(use_bias=True, requantize_output=True)
    int8_requantized_output_with_bias_params = \
        make_int_configuration(use_bias=True, requantize_output=True)
    qnn_dense_driver(uint8_requantized_output_with_bias_params)
    qnn_dense_driver(int8_requantized_output_with_bias_params)


if __name__ == "__main__":
    test_qnn_dense_without_bias()
    test_qnn_dense_with_bias()
    test_qnn_dense_with_requantized_output()
