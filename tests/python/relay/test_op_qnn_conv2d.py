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
from tvm.relay import transform
from tvm.relay.testing import run_infer_type
from tvm.contrib import graph_runtime

def get_ref_func(data,
                 kernel,
                 input_zero_point,
                 kernel_zero_point,
                 kernel_size,
                 padding,
                 strides,
                 dilation,
                 data_layout,
                 kernel_layout,
                 out_dtype):
    casted_data = relay.op.cast(data, "int32")
    casted_kernel = relay.op.cast(kernel, "int32")
    shifted_data = relay.op.subtract(casted_data,
            relay.const(input_zero_point, "int32"))
    shifted_kernel = relay.op.subtract(casted_kernel,
            relay.const(kernel_zero_point, "int32"))
    func = relay.op.nn.conv2d(shifted_data,
                             shifted_kernel,
                             padding=padding,
                             strides=strides,
                             dilation=dilation,
                             kernel_size=kernel_size,
                             out_dtype=out_dtype,
                             data_layout=data_layout,
                             kernel_layout=kernel_layout)

    func = relay.Function(relay.analysis.free_vars(func), func)
    return func

def get_qnn_func(data,
                 kernel,
                 input_zero_point,
                 kernel_zero_point,
                 kernel_size,
                 padding,
                 strides,
                 dilation,
                 data_layout,
                 kernel_layout,
                 out_dtype):
    func = relay.qnn.op.conv2d(
            data, kernel,
            input_zero_point=input_zero_point,
            kernel_zero_point=kernel_zero_point,
            kernel_size=kernel_size,
            strides=strides,
            dilation=dilation,
            padding=padding,
            out_dtype=out_dtype,
            data_layout=data_layout,
            kernel_layout=kernel_layout)

    mod = relay.Function(relay.analysis.free_vars(func), func)
    mod = relay.Module.from_expr(mod)
    return mod

def get_funcs(data_shape,
              data_dtype,
              kernel_shape,
              kernel_dtype,
              input_zero_point,
              kernel_zero_point,
              kernel_size,
              padding,
              strides,
              dilation,
              data_layout,
              kernel_layout,
              out_dtype):
    data = relay.var("data", shape=data_shape,
            dtype=data_dtype)
    kernel = relay.var("kernel", shape=kernel_shape,
            dtype=kernel_dtype)
    ref_func = get_ref_func(data,
                            kernel,
                            input_zero_point,
                            kernel_zero_point,
                            kernel_size,
                            padding,
                            strides,
                            dilation,
                            data_layout,
                            kernel_layout,
                            out_dtype)
    ref_func = run_infer_type(ref_func)
    qnn_func = get_qnn_func(data,
                            kernel,
                            input_zero_point,
                            kernel_zero_point,
                            kernel_size,
                            padding,
                            strides,
                            dilation,
                            data_layout,
                            kernel_layout,
                            out_dtype)
    return (ref_func, qnn_func)

def verify(ref_func, qnn_func, data_shape, data_dtype, kernel_shape,
        kernel_dtype):
    def get_inputs(data_shape, data_dtype, kernel_shape, kernel_dtype):
        # Keeping inputs multiple of 4 because of a bug in Average Pool2d
        # https://discuss.tvm.ai/t/pool2d-gives-bad-output-for-integer-inputs/3377
        low = -128
        high = 127
        if data_dtype == "uint8":
            low = 0
            high = 255
        golden_data = np.random.random_integers(low=low, high=high,
                size=data_shape).astype(data_dtype)
        low = -128
        high = 127
        if kernel_dtype == "uint8":
            low = 0
            high = 255
        golden_weight = np.random.random_integers(low=low, high=high,
                size=kernel_shape).astype(kernel_dtype)
        return (golden_data, golden_weight)


    def get_output(func, golden_inputs):
        with relay.build_config(opt_level=2):
            golden_data, golden_weight = golden_inputs
            params = {'kernel': golden_weight}
            graph, lib, params = relay.build(func, "llvm", params=params)
            mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
            mod.set_input("data", golden_data)
            mod.set_input(**params)
            mod.run()
            res = mod.get_output(0).asnumpy()
            return res
    golden_inputs = get_inputs(data_shape, data_dtype,
            kernel_shape, kernel_dtype)
    golden_output = get_output(ref_func, golden_inputs)
    qnn_output = get_output(qnn_func, golden_inputs)
    np.testing.assert_equal(qnn_output, golden_output)

def no_zero_point_test():
    # uint8 input
    data_shape = (2, 1, 2, 4)
    data_dtype = 'uint8'
    kernel_shape = (3, 1, 2, 2)
    kernel_dtype = 'uint8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=0,
                                   kernel_zero_point=0,
                                   kernel_size=(2, 2),
                                   padding=(0, 0),
                                   strides=(1, 1),
                                   dilation=(1, 1),
                                   data_layout="NCHW",
                                   kernel_layout="OIHW",
                                   out_dtype="int32")
    verify(ref_func, qnn_func, data_shape, data_dtype,
            kernel_shape, kernel_dtype)

    # int8 input
    data_shape = (2, 1, 2, 4)
    data_dtype = 'int8'
    kernel_shape = (3, 1, 2, 2)
    kernel_dtype = 'int8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=0,
                                   kernel_zero_point=0,
                                   kernel_size=(2, 2),
                                   padding=(0, 0),
                                   strides=(1, 1),
                                   dilation=(1, 1),
                                   data_layout="NCHW",
                                   kernel_layout="OIHW",
                                   out_dtype="int32")
    verify(ref_func, qnn_func, data_shape, data_dtype,
            kernel_shape, kernel_dtype)

def kernel_zero_point_test():
    # uint8 input
    data_shape = (2, 4, 2, 4)
    data_dtype = 'uint8'
    kernel_shape = (3, 4, 2, 2)
    kernel_dtype = 'uint8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=0,
                                   kernel_zero_point=1,
                                   kernel_size=(2, 2),
                                   padding=(0, 0),
                                   strides=(1, 1),
                                   dilation=(1, 1),
                                   data_layout="NCHW",
                                   kernel_layout="OIHW",
                                   out_dtype="int32")
    verify(ref_func, qnn_func, data_shape, data_dtype,
            kernel_shape, kernel_dtype)

    # int8 input
    data_shape = (2, 1, 2, 4)
    data_dtype = 'int8'
    kernel_shape = (3, 1, 2, 2)
    kernel_dtype = 'int8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=0,
                                   kernel_zero_point=5,
                                   kernel_size=(2, 2),
                                   padding=(0, 0),
                                   strides=(1, 1),
                                   dilation=(1, 1),
                                   data_layout="NCHW",
                                   kernel_layout="OIHW",
                                   out_dtype="int32")
    verify(ref_func, qnn_func, data_shape, data_dtype,
            kernel_shape, kernel_dtype)


def input_zero_point_test():
    # uint8 input
    data_shape = (2, 4, 2, 4)
    data_dtype = 'uint8'
    kernel_shape = (3, 4, 2, 2)
    kernel_dtype = 'uint8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=5,
                                   kernel_zero_point=0,
                                   kernel_size=(2, 2),
                                   padding=(0, 0),
                                   strides=(1, 1),
                                   dilation=(1, 1),
                                   data_layout="NCHW",
                                   kernel_layout="OIHW",
                                   out_dtype="int32")
    verify(ref_func, qnn_func, data_shape, data_dtype,
            kernel_shape, kernel_dtype)

    # int8 input
    data_shape = (2, 4, 2, 4)
    data_dtype = 'int8'
    kernel_shape = (3, 4, 2, 2)
    kernel_dtype = 'int8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=5,
                                   kernel_zero_point=0,
                                   kernel_size=(2, 2),
                                   padding=(0, 0),
                                   strides=(1, 1),
                                   dilation=(1, 1),
                                   data_layout="NCHW",
                                   kernel_layout="OIHW",
                                   out_dtype="int32")
    verify(ref_func, qnn_func, data_shape, data_dtype,
            kernel_shape, kernel_dtype)

def both_zero_point_test():
    # uint8 input
    data_shape = (2, 4, 2, 4)
    data_dtype = 'uint8'
    kernel_shape = (3, 4, 2, 2)
    kernel_dtype = 'uint8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=5,
                                   kernel_zero_point=3,
                                   kernel_size=(2, 2),
                                   padding=(0, 0),
                                   strides=(1, 1),
                                   dilation=(1, 1),
                                   data_layout="NCHW",
                                   kernel_layout="OIHW",
                                   out_dtype="int32")
    verify(ref_func, qnn_func, data_shape, data_dtype,
            kernel_shape, kernel_dtype)

    # int8 input
    data_shape = (2, 4, 2, 4)
    data_dtype = 'int8'
    kernel_shape = (3, 4, 2, 2)
    kernel_dtype = 'int8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=5,
                                   kernel_zero_point=3,
                                   kernel_size=(2, 2),
                                   padding=(0, 0),
                                   strides=(1, 1),
                                   dilation=(1, 1),
                                   data_layout="NCHW",
                                   kernel_layout="OIHW",
                                   out_dtype="int32")
    verify(ref_func, qnn_func, data_shape, data_dtype,
            kernel_shape, kernel_dtype)

def layout_test():
    # uint8 input
    data_shape = (2, 2, 4, 4) # NHWC
    data_dtype = 'uint8'
    kernel_shape = (2, 2, 4, 3) # HWIO
    kernel_dtype = 'uint8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=5,
                                   kernel_zero_point=3,
                                   kernel_size=(2, 2),
                                   padding=(0, 0),
                                   strides=(1, 1),
                                   dilation=(1, 1),
                                   data_layout="NHWC",
                                   kernel_layout="HWIO",
                                   out_dtype="int32")
    verify(ref_func, qnn_func, data_shape, data_dtype,
            kernel_shape, kernel_dtype)

    # NHWC and HWIO layout. Used in depthwise conv.
    data_shape = (2, 2, 4, 1) # NHWC
    data_dtype = 'uint8'
    kernel_shape = (2, 2, 1, 1) # HWOI
    kernel_dtype = 'uint8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=5,
                                   kernel_zero_point=3,
                                   kernel_size=(2, 2),
                                   padding=(0, 0),
                                   strides=(1, 1),
                                   dilation=(1, 1),
                                   data_layout="NHWC",
                                   kernel_layout="HWOI",
                                   out_dtype="int32")
    verify(ref_func, qnn_func, data_shape, data_dtype,
            kernel_shape, kernel_dtype)



def padding_test():
    # uint8 input
    data_shape = (1, 4, 2, 2)
    data_dtype = 'uint8'
    kernel_shape = (3, 4, 2, 2)
    kernel_dtype = 'uint8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=8,
                                   kernel_zero_point=5,
                                   kernel_size=(2, 2),
                                   padding=(1, 1),
                                   strides=(1, 1),
                                   dilation=(1, 1),
                                   data_layout="NCHW",
                                   kernel_layout="OIHW",
                                   out_dtype="int32")
    verify(ref_func, qnn_func, data_shape, data_dtype,
            kernel_shape, kernel_dtype)

    # Try different layout
    data_shape = (2, 2, 4, 4) # NHWC
    data_dtype = 'uint8'
    kernel_shape = (2, 2, 4, 3) # HWIO
    kernel_dtype = 'uint8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=8,
                                   kernel_zero_point=3,
                                   kernel_size=(2, 2),
                                   padding=(1, 1),
                                   strides=(1, 1),
                                   dilation=(1, 1),
                                   data_layout="NHWC",
                                   kernel_layout="HWIO",
                                   out_dtype="int32")
    verify(ref_func, qnn_func, data_shape, data_dtype,
            kernel_shape, kernel_dtype)

def dilation_test():
    # uint8 input
    data_shape = (2, 4, 4, 4)
    data_dtype = 'uint8'
    kernel_shape = (3, 4, 2, 2)
    kernel_dtype = 'uint8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=5,
                                   kernel_zero_point=3,
                                   kernel_size=(2, 2),
                                   padding=(0, 0),
                                   strides=(1, 1),
                                   dilation=(2, 2),
                                   data_layout="NCHW",
                                   kernel_layout="OIHW",
                                   out_dtype="int32")
    verify(ref_func, qnn_func, data_shape, data_dtype,
            kernel_shape, kernel_dtype)


def const_folding_test():
    data_shape = (2, 4, 2, 4)
    data_dtype = 'uint8'
    kernel_shape = (3, 4, 2, 2)
    kernel_dtype = 'uint8'

    golden_weight = np.random.random_integers(low=0, high=255,
            size=kernel_shape).astype(kernel_dtype)
    data = relay.var("data", shape=data_shape,
            dtype=data_dtype)
    kernel = relay.const(golden_weight)
    qnn_func = get_qnn_func(data,
                            kernel,
                            input_zero_point=8,
                            kernel_zero_point=3,
                            kernel_size=(2, 2),
                            padding=(0, 0),
                            strides=(1, 1),
                            dilation=(1, 1),
                            data_layout="NCHW",
                            kernel_layout="OIHW",
                            out_dtype="int32")
    folded_mod = transform.FoldConstant()(qnn_func)
    folded_func = folded_mod["main"]
    assert "reshape" not in folded_func.astext()

def kernel_size_1x1_test():
    # uint8 input
    data_shape = (2, 4, 2, 4)
    data_dtype = 'uint8'
    kernel_shape = (3, 4, 1, 1)
    kernel_dtype = 'uint8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=5,
                                   kernel_zero_point=3,
                                   kernel_size=(1, 1),
                                   padding=(0, 0),
                                   strides=(1, 1),
                                   dilation=(1, 1),
                                   data_layout="NCHW",
                                   kernel_layout="OIHW",
                                   out_dtype="int32")
    assert 'avg_pool2d' not in qnn_func.astext()
    verify(ref_func, qnn_func, data_shape, data_dtype,
            kernel_shape, kernel_dtype)

def tflite_large_irregular_test():
    # uint8 input
    data_shape = (1, 1024, 1, 1)
    data_dtype = 'uint8'
    kernel_shape = (1001, 1024, 1, 1)
    kernel_dtype = 'uint8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=127,
                                   kernel_zero_point=127,
                                   kernel_size=(1, 1),
                                   padding=(0, 0),
                                   strides=(1, 1),
                                   dilation=(1, 1),
                                   data_layout="NCHW",
                                   kernel_layout="OIHW",
                                   out_dtype="int32")
    golden_data = np.full(data_shape, 127).astype('uint8')
    golden_weight = np.full(kernel_shape, 127).astype('uint8')

    with relay.build_config(opt_level=2):
        params = {'kernel': golden_weight}
        graph, lib, params = relay.build(qnn_func, "llvm", params=params)
        mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        mod.set_input("data", golden_data)
        mod.set_input(**params)
        mod.run()
        qnn_output = mod.get_output(0).asnumpy()
    golden_output = np.full((1, 1001, 1, 1), 0).astype('uint8')
    np.testing.assert_equal(qnn_output, golden_output)

def tflite_output_multiplier_greater_than_one():
    # uint8 input
    data_shape = (2, 1, 2, 4)
    data_dtype = 'uint8'
    kernel_shape = (3, 1, 2, 2)
    kernel_dtype = 'uint8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=128,
                                   kernel_zero_point=128,
                                   kernel_size=(2, 2),
                                   padding=(0, 0),
                                   strides=(2, 2),
                                   dilation=(1, 1),
                                   data_layout="NCHW",
                                   kernel_layout="OIHW",
                                   out_dtype="int32")
    golden_data = 128 + np.array((1, 1, 1, 1,
                                  2, 2, 2, 2,
                                  1, 2, 3, 4,
                                  1, 2, 3, 4)).reshape(data_shape).astype('uint8')
    golden_weight = 128 + np.array((1, 2, 3, 4,
                                    -1, 1, -1, 1,
                                    -1, -1, 1, 1)).reshape(kernel_shape)
    golden_weight = golden_weight.astype('uint8')

    with relay.build_config(opt_level=2):
        params = {'kernel': golden_weight}
        graph, lib, params = relay.build(qnn_func, "llvm", params=params)
        mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        mod.set_input("data", golden_data)
        mod.set_input(**params)
        mod.run()
        qnn_output = mod.get_output(0).asnumpy()
    golden_output = np.array((17, 17,
                              0, 0,
                              2, 2,
                              16, 36,
                              2, 2,
                              0, 0)).reshape(2, 3, 1, 2)
    np.testing.assert_equal(qnn_output, golden_output)

def tflite_anistropic_strides():
    # uint8 input
    data_shape = (1, 1, 3, 6)
    data_dtype = 'uint8'
    kernel_shape = (1, 1, 2, 2)
    kernel_dtype = 'uint8'
    ref_func, qnn_func = get_funcs(data_shape=data_shape,
                                   data_dtype=data_dtype,
                                   kernel_shape=kernel_shape,
                                   kernel_dtype=kernel_dtype,
                                   input_zero_point=127,
                                   kernel_zero_point=127,
                                   kernel_size=(2, 2),
                                   padding=(0, 0),
                                   strides=(1, 3),
                                   dilation=(1, 1),
                                   data_layout="NCHW",
                                   kernel_layout="OIHW",
                                   out_dtype="int32")
    golden_data = np.array((133, 131, 129, 125, 123, 121,
                            135, 133, 131, 123, 121, 119,
                            137, 135, 133, 121, 119, 117)).reshape(data_shape)
    golden_data = golden_data.astype('uint8')
    golden_weight = np.array((129, 131, 133, 135)).reshape(kernel_shape)
    golden_weight = golden_weight.astype('uint8')

    with relay.build_config(opt_level=2):
        params = {'kernel': golden_weight}
        graph, lib, params = relay.build(qnn_func, "llvm", params=params)
        mod = graph_runtime.create(graph, lib, ctx=tvm.cpu(0))
        mod.set_input("data", golden_data)
        mod.set_input(**params)
        mod.run()
        qnn_output = mod.get_output(0).asnumpy()
    golden_output = np.array((124, -92, 164, -132)).reshape(1, 1, 2, 2)
    np.testing.assert_equal(qnn_output, golden_output)

def broadcast_layout_test():
    # Test broadcast support for NHWC layout.
    data_shape = (1, 229, 229, 3) # NHWC
    data_dtype = 'uint8'
    kernel_shape = (7, 7, 3, 64) # HWIO
    kernel_dtype = 'int8'
    _, qnn_func = get_funcs(data_shape=data_shape,
                            data_dtype=data_dtype,
                            kernel_shape=kernel_shape,
                            kernel_dtype=kernel_dtype,
                            input_zero_point=8,
                            kernel_zero_point=3,
                            kernel_size=(7, 7),
                            padding=(1, 1),
                            strides=(1, 1),
                            dilation=(1, 1),
                            data_layout="NHWC",
                            kernel_layout="HWIO",
                            out_dtype="int32")
    func = qnn_func['main'].body
    bias = relay.var("bias", shape=(64,), dtype="int32")
    bias2 = relay.var("bias2", shape=(1, 225, 225, 1), dtype="int32")

    # Check broadcast support on both lhs and rhs
    func = relay.add(func, bias2)
    func = relay.add(bias2, func)
    func = relay.add(bias, func)
    func = relay.add(func, bias)
    func = relay.Function(relay.analysis.free_vars(func), func)
    mod = relay.Module.from_expr(func)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, "llvm -mcpu=skylake-avx512")

if __name__ == "__main__":
    no_zero_point_test()
    input_zero_point_test()
    kernel_zero_point_test()
    both_zero_point_test()
    layout_test()
    padding_test()
    dilation_test()
    const_folding_test()
    kernel_size_1x1_test()
    tflite_large_irregular_test()
    tflite_output_multiplier_greater_than_one()
    tflite_anistropic_strides()
    broadcast_layout_test()
