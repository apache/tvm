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
# pylint: disable=import-self, invalid-name, unused-argument
"""
TFLite testcases
================
This article is a test script to test TFLite operator with Relay.
"""
from __future__ import print_function
from functools import partial
import numpy as np
import tvm
from tvm import relay
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import variables
try:
    from tensorflow import lite as interpreter_wrapper
except ImportError:
    from tensorflow.contrib import lite as interpreter_wrapper

import tvm.relay.testing.tf as tf_testing
from packaging import version as package_version

#######################################################################
# Generic run functions for TVM & TFLite
# --------------------------------------
def convert_to_list(x):
    if not isinstance(x, list):
        x = [x]
    return x

def run_tvm_graph(tflite_model_buf, input_data, input_node, num_output=1, target='llvm',
                  out_names=None):
    """ Generic function to compile on relay and execute on tvm """
    try:
        import tflite.Model
    except ImportError:
        raise ImportError("The tflite package must be installed")

    # get TFLite model from buffer
    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)

    shape_dict = {}
    dtype_dict = {}
    for i, e in enumerate(input_node):
        shape_dict[e] = input_data[i].shape
        dtype_dict[e] = input_data[i].dtype.name

    mod, params = relay.frontend.from_tflite(tflite_model,
                                             shape_dict=shape_dict,
                                             dtype_dict=dtype_dict)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)

    ctx = tvm.context(target, 0)
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    for i, e in enumerate(input_node):
        m.set_input(e, tvm.nd.array(input_data[i].astype(input_data[i].dtype)))

    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    assert out_names is None or num_output == len(out_names), "out_names: {} num_output: {}".format(
        out_names, num_output)
    tvm_output_list = []
    for i in range(0, num_output):
        tvm_output = m.get_output(i)
        tvm_output_list.append(tvm_output.asnumpy())
    return tvm_output_list


def run_tflite_graph(tflite_model_buf, input_data):
    """ Generic function to execute TFLite """
    input_data = convert_to_list(input_data)

    interpreter = interpreter_wrapper.Interpreter(model_content=tflite_model_buf)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # set input
    assert len(input_data) == len(input_details)
    for i in range(len(input_details)):
        interpreter.set_tensor(input_details[i]['index'], input_data[i])

    # Run
    interpreter.invoke()

    # get output
    tflite_output = list()
    for i in range(len(output_details)):
        tflite_output.append(interpreter.get_tensor(output_details[i]['index']))

    return tflite_output


def compare_tflite_with_tvm(in_data, in_name, input_tensors,
                            output_tensors, init_global_variables=False, out_names=None):
    """Generic function to generate and compare TFLite and TVM output"""
    in_data = convert_to_list(in_data)
    in_name = convert_to_list(in_name)
    out_names = convert_to_list(out_names)
    in_node = [0] * len(in_name)
    for i in range(len(in_name)):
        in_node[i] = in_name[i].split(':')[0] if ":" in in_name[i] else in_name[i]

    with tf.Session() as sess:
        if init_global_variables:
            sess.run(variables.global_variables_initializer())
        # convert to tflite model
        converter = interpreter_wrapper.TFLiteConverter.from_session(
            sess, input_tensors, output_tensors)
        tflite_model_buffer = converter.convert()
        tflite_output = run_tflite_graph(tflite_model_buffer, in_data)

        for device in ["llvm"]:
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                continue

            tvm_output = run_tvm_graph(tflite_model_buffer, in_data, in_node, target=device,
                                       num_output=len(out_names), out_names=out_names)
            for i in range(len(tflite_output)):
                tvm.testing.assert_allclose(tflite_output[i], tvm_output[i], atol=1e-5, rtol=1e-5)


def with_fused_activation_function(input_tensor, fn_name):
    if fn_name is None or fn_name == "NONE":
        return input_tensor
    if fn_name == "RELU":
        return nn_ops.relu(input_tensor)
    if fn_name == "RELU6":
        return nn_ops.relu6(input_tensor)
    if fn_name == "RELU_N1_TO_1":
        return math_ops.maximum(-1, math_ops.minimum(input_tensor, 1))
    if fn_name == "TANH":
        return math_ops.tanh(input_tensor)
    raise AssertionError("Unknown fused_activation_function {}".format(fn_name))

def _test_split(in_shape, axis, num_Splits, dtype):
    '''internal split tester taking as parameters in_shape, number of tensors to split into
       and dtype (data type)'''
    np_data = np.random.uniform(-5, 5, size=in_shape).astype(dtype)
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=in_shape, dtype=dtype)
        out = array_ops.split(in_data, num_Splits, axis=axis)
        out_names = ['out_' + str(n) + ':0' for n in range(num_Splits)]
        compare_tflite_with_tvm([np_data], ['Placeholder:0'],  [in_data], out,
                                out_names=out_names)

def test_forward_split():
    '''test split layer'''
    # rank 1
    _test_split((3,), 0, 1, 'float32')
    _test_split((3,), 0, 3, 'float32')
    _test_split((6,), 0, 3, 'float32')
    # rank 2
    _test_split((6, 2), 0, 3, 'float32')
    _test_split((2, 6), 1, 6, 'float32')
    # rank 3
    if package_version.parse(tf.VERSION) >= package_version.parse('1.14.0'):
        _test_split((6, 2, 4), 0, 2, 'int32')

    _test_split((2, 6, 4), 1, 3, 'float32')
    _test_split((2, 4, 6), 2, 1, 'float32')
    # rank 4
    _test_split((6, 1, 3, 5), 0, 3, 'float32')
    _test_split((1, 6, 3, 5), 1, 3, 'float32')
    _test_split((1, 3, 6, 5), 2, 3, 'float32')
    _test_split((1, 3, 5, 6), 3, 3, 'float32')
    # split along negative axis
    _test_split((6, 1, 3, 5), -4, 3, 'float32')
    _test_split((1, 6, 3, 5), -3, 3, 'float32')
    _test_split((1, 3, 6, 5), -2, 3, 'float32')
    _test_split((1, 3, 5, 6), -1, 3, 'float32')

#######################################################################
# transpose
# ---------


def _test_forward_transpose(ishape, axes=()):
    data = np.random.uniform(size=ishape).astype(np.float32)

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)

        if not axes:
            out = array_ops.transpose(in_data)
        else:
            out = array_ops.transpose(in_data, axes)

        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])


def test_forward_transpose():
    _test_forward_transpose((2, 2))
    _test_forward_transpose((2, 3, 4))
    _test_forward_transpose((7, 8, 8, 10))
    _test_forward_transpose((2, 3, 4), (1, 2, 0))
    _test_forward_transpose((2, 3, 4), (0, 1, 2))
    _test_forward_transpose((2, 3, 4, 5), (3, 0, 1, 2))
    _test_forward_transpose((2, 3, 4, 5), ())

#######################################################################
# Cast
# --------

def _test_cast(data, cast_dtype):
    """ One iteration of CAST """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = math_ops.cast(in_data, cast_dtype)
        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])


def test_forward_cast():
    """ CAST """
    _test_cast(np.arange(6.0, dtype=np.float32).reshape((1, 6)), cast_dtype=tf.int32)
    _test_cast(np.arange(6.0, dtype=np.float32).reshape((1, 6)), cast_dtype=tf.uint8)
    _test_cast(np.arange(6.0, dtype=np.int32).reshape((1, 6)), cast_dtype=tf.int64)

#######################################################################
# tile
# ---------


def _test_forward_tile(in_shape, reps, dtype):
    data = np.random.uniform(-5, 5, size=in_shape).astype(dtype)

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)

        out = array_ops.tile(in_data, reps)

        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])


def test_forward_tile():
    _test_forward_tile((2, ), (3, ), "int32")
    _test_forward_tile((2, 2), (2, 3), "float32")

######################################################################
# BatchToSpaceND
# --------------


def _test_batch_to_space_nd(input_shape, block_shape, crops, dtype='int32'):
    data = np.random.uniform(0, 5, size=input_shape).astype(dtype)

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=input_shape, dtype=dtype)

        out = array_ops.batch_to_space_nd(in_data, block_shape, crops)

        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])


def test_forward_batch_to_space_nd():
    # test cases: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/batch-to-space-n-d
    _test_batch_to_space_nd(
        input_shape=[4, 1, 1, 1],
        block_shape=[2, 2],
        crops=[[0, 0], [0, 0]]
    )

    _test_batch_to_space_nd(
        input_shape=[4, 1, 1, 3],
        block_shape=[2, 2],
        crops=[[0, 0], [0, 0]]
    )

    _test_batch_to_space_nd(
        input_shape=[4, 2, 2, 1],
        block_shape=[2, 2],
        crops=[[0, 0], [0, 0]]
    )

######################################################################
# SpaceToBatchND
# --------------


def _test_space_to_batch_nd(input_shape, block_shape, paddings, dtype='int32'):
    data = np.random.uniform(0, 5, size=input_shape).astype(dtype)

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=input_shape, dtype=dtype)

        out = array_ops.space_to_batch_nd(in_data, block_shape, paddings)

        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])


def test_forward_space_to_batch_nd():
    # test cases: https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd
    _test_space_to_batch_nd(
        input_shape=[1, 2, 2, 1],
        block_shape=[2, 2],
        paddings=[[0, 0], [0, 0]]
    )

    _test_space_to_batch_nd(
        input_shape=[1, 2, 2, 3],
        block_shape=[2, 2],
        paddings=[[0, 0], [0, 0]]
    )

    _test_space_to_batch_nd(
        input_shape=[1, 4, 4, 1],
        block_shape=[2, 2],
        paddings=[[0, 0], [0, 0]]
    )

    _test_space_to_batch_nd(
        input_shape=[2, 2, 4, 1],
        block_shape=[2, 2],
        paddings=[[0, 0], [2, 0]]
    )

#######################################################################
# Pooling
# -------
def _test_pooling_iteration(input_shape, **kwargs):
    """ One iteration of pool operation with given shapes and attributes """

    x = -np.arange(
        np.prod(input_shape), dtype=np.float32).reshape(input_shape) - 1

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=input_shape, dtype='float32')
        out = nn_ops.pool(in_data, **kwargs)

        compare_tflite_with_tvm(x,'Placeholder:0', [in_data], [out])


def _test_pooling(input_shape, **kwargs):
    _test_pooling_iteration(input_shape, **kwargs)


def test_forward_pooling():
    """ Pooling """

    for pool_type in ['AVG', 'MAX']:
        _test_pooling(input_shape=[2, 9, 10, 2],
                      window_shape=[1, 1],
                      padding='SAME',
                      pooling_type=pool_type,
                      dilation_rate=[1, 1],
                      strides=[1, 1])

        _test_pooling(input_shape=[2, 10, 9, 2],
                      window_shape=[1, 1],
                      padding='SAME',
                      pooling_type=pool_type,
                      dilation_rate=[1, 1],
                      strides=[1, 1])

        _test_pooling(input_shape=[2, 9, 10, 2],
                      window_shape=[2, 1],
                      padding='SAME',
                      pooling_type=pool_type,
                      dilation_rate=[1, 1],
                      strides=[1, 1])

        _test_pooling(input_shape=[2, 10, 9, 2],
                      window_shape=[2, 3],
                      padding='SAME',
                      pooling_type=pool_type,
                      dilation_rate=[1, 1],
                      strides=[2, 1])


#######################################################################
# Convolution
# -----------

def _test_convolution(tensor_in_sizes, filter_in_sizes,
                      dilations, strides, padding, data_format,
                      is_depthwise=False):
    """ One iteration of convolution with given shapes and attributes """

    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
        total_size_1 *= s
    for s in filter_in_sizes:
        total_size_2 *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    data_array = [f * 1.0 for f in range(1, total_size_1 + 1)]
    filter_array = [f * 1.0 for f in range(1, total_size_2 + 1)]

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=tensor_in_sizes, dtype='float32')
        in_filter = constant_op.constant(filter_array, shape=filter_in_sizes, dtype='float32')
        strides = [1] + strides + [1]
        dilations = [1] + dilations + [1]

        if is_depthwise:
            out = nn_ops.depthwise_conv2d_native(in_data,
                                                 in_filter,
                                                 strides=strides,
                                                 padding=padding,
                                                 data_format=data_format)
        else:
            out = nn_ops.conv2d(in_data,
                                in_filter,
                                strides=strides,
                                padding=padding,
                                data_format=data_format)
        data_array = np.reshape(data_array, tensor_in_sizes).astype('float32')
        compare_tflite_with_tvm(data_array, 'Placeholder:0', [in_data], [out])


def test_forward_convolution():
    _test_convolution([4, 8, 8, 176], [1, 1, 176, 32], [1, 1], [1, 1], 'SAME', 'NHWC')
    _test_convolution([4, 17, 17, 19], [3, 3, 19, 19], [1, 1], [2, 2], 'VALID', 'NHWC')
    _test_convolution([4, 17, 17, 124], [1, 1, 124, 19], [1, 1], [1, 1], 'SAME', 'NHWC')
    _test_convolution([4, 17, 17, 12], [3, 3, 12, 32], [1, 1], [2, 2], 'VALID', 'NHWC')

    # depthwise convolution
    _test_convolution([4, 8, 8, 176], [1, 1, 176, 1], [1, 1], [1, 1], 'SAME', 'NHWC', True)
    _test_convolution([4, 17, 17, 19], [3, 3, 19, 1], [1, 1], [2, 2], 'VALID', 'NHWC', True)
    _test_convolution([4, 17, 17, 124], [1, 1, 124, 1], [1, 1], [1, 1], 'SAME', 'NHWC', True)
    _test_convolution([4, 17, 17, 12], [3, 3, 12, 1], [1, 1], [2, 2], 'VALID', 'NHWC', True)
    _test_convolution([4, 17, 17, 12], [3, 3, 12, 2], [1, 1], [2, 2], 'VALID', 'NHWC', True)


#######################################################################
# Reshape
# -------

def _test_reshape(data, out_shape):
    """ One iteration of reshape operation with given data and out shape """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = array_ops.reshape(in_data, out_shape)

        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])


def test_forward_reshape():
    _test_reshape(np.arange(6.0, dtype=np.float32), [2, 3])
    _test_reshape(np.arange(6), [-1, 2])
    _test_reshape(np.arange(6), [3, -1])
    _test_reshape(np.arange(6), [-1])


#######################################################################
# Resize
# ------

def _test_resize(tf_resize_op, data, align_corners):
    """ One iteration of Resize """

    assert len(data) == 2

    # Test with tensor and constant
    with tf.Graph().as_default():
        images_tensor = array_ops.placeholder(shape=data[0].shape, dtype=data[0].dtype, name='in')
        size = ops.convert_to_tensor(data[1], dtype=data[1].dtype)
        out_tensor = tf_resize_op(images=images_tensor, size=size, align_corners=align_corners)
        compare_tflite_with_tvm([data[0]], ['in:0'], [images_tensor], [out_tensor])


def test_all_resize():
    """ Resize """
    data = [np.random.rand(1, 16, 16, 3).astype("float32"), np.array([8, 8], dtype=np.int32)]
    ### RESIZE_BILINEAR
    _test_resize(tf.image.resize_bilinear, data, align_corners=False)
    _test_resize(tf.image.resize_bilinear, data, align_corners=True)
    ### RESIZE_NEAREST_NEIGHBOR (was added in v1.13)
    # According to topi resize.h
    # Align corners not supported for nearest neighbour
    from tflite.BuiltinOperator import BuiltinOperator
    if 'RESIZE_NEAREST_NEIGHBOR' in dir(BuiltinOperator()):
        _test_resize(tf.image.resize_nearest_neighbor, data, align_corners=False)


#######################################################################
# Concatenation
# -------------

def _test_concatenation(data, axis):
    """ One iteration of concatenation """

    assert len(data) >= 1

    with tf.Graph().as_default():
        in_data = [
            array_ops.placeholder(shape=tensor.shape, dtype=tensor.dtype, name="in_{}".format(idx))
            for idx, tensor in enumerate(data)]
        out = array_ops.concat(in_data, axis=axis)
        name = ["in_{}:0".format(idx) for idx in range(len(data))]

        compare_tflite_with_tvm(data, name, in_data, [out])


def test_forward_concatenation():

    _test_concatenation(
        [np.arange(6).reshape((1, 2, 1, 3)),
        np.arange(6).reshape((1, 2, 1, 3))], 1)

    _test_concatenation(
        [np.arange(6).reshape((3, 2)),
         np.arange(6).reshape((3, 2))], 1)

    _test_concatenation(
        [np.arange(6).reshape((2, 1, 1, 3)),
         np.arange(6).reshape((2, 1, 1, 3)),
         np.arange(6).reshape((2, 1, 1, 3))], 1)


#######################################################################
# Element-wise
# ---

def _test_elemwise(math_op, data, fused_activation_function=None):
    """ One iteration of elemwise """

    assert len(data) == 2

    # Test with two tensors
    with tf.Graph().as_default():
        in_data = [array_ops.placeholder(shape=data[0].shape, dtype=data[0].dtype, name='in_0'),
                   array_ops.placeholder(shape=data[1].shape, dtype=data[1].dtype, name='in_1')]
        out = math_op(in_data[0], in_data[1])
        out = with_fused_activation_function(out, fused_activation_function)
        compare_tflite_with_tvm(data, ['in_0:0', 'in_1:0'], in_data, [out])

    # Test with tensor and constant
    with tf.Graph().as_default():
        in_data = [array_ops.placeholder(shape=data[0].shape, dtype=data[0].dtype, name='in')]
        out = math_op(in_data[0], ops.convert_to_tensor(data[1], dtype=data[1].dtype))
        out = with_fused_activation_function(out, fused_activation_function)
        compare_tflite_with_tvm([data[0]], ['in:0'], in_data, [out])


#######################################################################
# Add
# ---

def _test_add(data, fused_activation_function=None):
    """ One iteration of add """
    return _test_elemwise(math_ops.add, data, fused_activation_function)

#######################################################################
# Subtract
# --------

def _test_sub(data, fused_activation_function=None):
    """ One iteration of subtract """
    return _test_elemwise(math_ops.subtract, data, fused_activation_function)
#######################################################################
# Mul
# ---
def _test_mul(data, fused_activation_function=None):
    """ One iteration of mul """
    return _test_elemwise(math_ops.multiply, data, fused_activation_function)

#######################################################################
# Divide
# ------

def _test_div(data, fused_activation_function=None):
    """ One iteration of divide """
    return _test_elemwise(math_ops.divide, data, fused_activation_function)
#######################################################################
# Power
# -----

def _test_pow(data):
    """ One iteration of power """
    return _test_elemwise(math_ops.pow, data)
#######################################################################
# Maximum
# -------

def _test_maximum(data):
    """ One iteration of maximum """
    return _test_elemwise(math_ops.maximum, data)
#######################################################################
# Minimum
# -------

def _test_minimum(data):
    """ One iteration of minimum """
    return _test_elemwise(math_ops.minimum, data)
#######################################################################
# Greater
# -------

def _test_greater(data):
    """ One iteration of greater """
    return _test_elemwise(math_ops.greater, data)

def _test_forward_elemwise(testop):
    """ Elewise"""
    testop([np.arange(6.0, dtype=np.float32).reshape((2, 1, 1, 3)),
               np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 1, 3))])
    testop([np.arange(6.0, dtype=np.float32).reshape((2, 1, 3)),
               np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 3))])
    testop([np.arange(3.0, dtype=np.float32).reshape((1, 3)),
               np.arange(1.0, 4.0, dtype=np.float32).reshape((1, 3))])

def test_all_elemwise():
    _test_forward_elemwise(_test_add)
    _test_forward_elemwise(partial(_test_add, fused_activation_function="RELU"))
    _test_forward_elemwise(partial(_test_add, fused_activation_function="RELU6"))
    _test_forward_elemwise(_test_sub)
    _test_forward_elemwise(partial(_test_sub, fused_activation_function="RELU"))
    _test_forward_elemwise(partial(_test_sub, fused_activation_function="RELU6"))
    _test_forward_elemwise(_test_mul)
    _test_forward_elemwise(partial(_test_mul, fused_activation_function="RELU"))
    _test_forward_elemwise(partial(_test_mul, fused_activation_function="RELU6"))
    _test_forward_elemwise(_test_div)
    _test_forward_elemwise(partial(_test_div, fused_activation_function="RELU"))
    _test_forward_elemwise(partial(_test_div, fused_activation_function="RELU6"))
    _test_forward_elemwise(_test_pow)
    _test_forward_elemwise(_test_maximum)
    _test_forward_elemwise(_test_minimum)
    _test_forward_elemwise(_test_greater)

#######################################################################
# Zeros like
# --------

def _test_zeros_like(data):
    """ One iteration of ZEROS LIKE """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = gen_array_ops.zeros_like(in_data)
        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])

def test_forward_zeros_like():
    """ ZEROS LIKE """
    _test_zeros_like(np.arange(6.0, dtype=np.float32).reshape((1, 6)))

#######################################################################
# Reduce
# ------

def _test_reduce(math_op, data, keep_dims=None):
    """ One iteration of reduce """

    assert len(data) == 2

    # Test with tensor and constant
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data[0].shape, dtype=data[0].dtype, name='in')
        out = math_op(in_data, data[1], keep_dims)
        compare_tflite_with_tvm([data[0]], ['in:0'], [in_data], [out])


#######################################################################
# Reduce_min
# ----------

def _test_reduce_min(data, keep_dims=None):
    """ One iteration of reduce_min """
    return _test_reduce(math_ops.reduce_min, data, keep_dims)

#######################################################################
# Reduce_max
# ----------

def _test_reduce_max(data, keep_dims=None):
    """ One iteration of reduce_max """
    return _test_reduce(math_ops.reduce_max, data, keep_dims)

#######################################################################
# Reduce_mean
# -----------

def _test_reduce_mean(data, keep_dims=None):
    """ One iteration of reduce_mean """
    return _test_reduce(math_ops.reduce_mean, data, keep_dims)

#######################################################################
# Reduce_prod
# -----------

def _test_reduce_prod(data, keep_dims=None):
    """ One iteration of reduce_prod """
    return _test_reduce(math_ops.reduce_prod, data, keep_dims)


def _test_forward_reduce(testop):
    """ Reduce """
    data0 = [np.random.rand(16, 16, 16, 16).astype("float32"), None]
    data1 = [np.random.rand(16, 16, 16, 16).astype("float32"), np.array([1, 2], dtype=np.int32)]
    testop(data0)
    testop(data0, keep_dims=False)
    testop(data0, keep_dims=True)
    testop(data1)
    testop(data1, keep_dims=False)
    testop(data1, keep_dims=True)


def test_all_reduce():
    _test_forward_reduce(_test_reduce_min)
    _test_forward_reduce(_test_reduce_max)
    _test_forward_reduce(_test_reduce_mean)
    _test_forward_reduce(_test_reduce_prod)


#######################################################################
# Squeeze
# -------

def _test_squeeze(data, squeeze_dims=None):
    """ One iteration of squeeze """

    if squeeze_dims is None:
        squeeze_dims = []

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)

        if squeeze_dims:
            out = array_ops.squeeze(in_data, squeeze_dims)
        else:
            out = array_ops.squeeze(in_data)

        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])


def test_forward_squeeze():
    """ Squeeze """
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3)), [0, 2])
    _test_squeeze(np.arange(6).reshape((2, 1, 3, 1)), [1, 3])


#######################################################################
# Pad
# ---

def _test_pad(data):
    """ One iteration of PAD """

    assert len(data) == 2

    # Test with tensor and constant
    with tf.Graph().as_default():
        in_data = [array_ops.placeholder(shape=data[0].shape, dtype=data[0].dtype, name='in')]
        out = array_ops.pad(in_data[0], ops.convert_to_tensor(data[1], dtype=data[1].dtype))
        compare_tflite_with_tvm([data[0]], ['in:0'], in_data, [out])


def test_forward_pad():
    """ Pad """
    _test_pad([np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 1, 3)),
               np.array([[1, 1], [2, 2], [1, 1], [2, 2]], dtype=np.int32)])
    _test_pad([np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 3)),
               np.array([[2, 2], [1, 1], [1, 1]], dtype=np.int32)])
    _test_pad([np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 3)),
               np.array([[1, 1], [2, 2]], dtype=np.int32)])
    _test_pad([np.arange(1.0, 4.0, dtype=np.float32).reshape((1, 3)),
               np.array([[1, 1], [2, 2]], dtype=np.int32)])


#######################################################################
# Pack
# -------------

def _test_pack(data, axis):
    """ One iteration of pack """

    assert len(data) >= 1

    with tf.Graph().as_default():
        in_data = [
            array_ops.placeholder(shape=tensor.shape, dtype=tensor.dtype, name="in_{}".format(idx))
            for idx, tensor in enumerate(data)]
        out = array_ops.pack(in_data, axis=axis)
        name = ["in_{}:0".format(idx) for idx in range(len(data))]

        compare_tflite_with_tvm(data, name, in_data, [out])


def test_forward_pack():
    """ Pack """
    _test_pack(
        [np.arange(6).reshape((1, 2, 1, 3)),
        np.arange(6).reshape((1, 2, 1, 3))], 1)

    _test_pack(
        [np.arange(6).reshape((3, 2)),
         np.arange(6).reshape((3, 2))], 1)

    _test_pack(
        [np.arange(6).reshape((2, 1, 1, 3)),
         np.arange(6).reshape((2, 1, 1, 3)),
         np.arange(6).reshape((2, 1, 1, 3))], 1)


#######################################################################
# Logistic
# --------

def _test_logistic(data):
    """ One iteration of LOGISTIC """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = math_ops.sigmoid(in_data)
        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])

def test_forward_logistic():
    """ LOGISTIC """
    _test_logistic(np.arange(6.0, dtype=np.float32).reshape((1, 6)))


#######################################################################
# Softmax
# -------

def _test_softmax(data):
    """ One iteration of softmax """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = nn_ops.softmax(in_data)
        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])

def test_forward_softmax():
    """ Softmax """
    _test_softmax(np.arange(6.0, dtype=np.float32).reshape((1, 6)))

#######################################################################
# Tanh
# --------

def _test_tanh(data):
    """ One iteration of TANH """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = math_ops.sigmoid(in_data)
        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])

def test_forward_tanh():
    """ TANH """
    _test_tanh(np.arange(6.0, dtype=np.float32).reshape((1, 6)))

#######################################################################
# ReLu
# --------

def _test_relu(data):
    """ One iteration of ReLU """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = nn_ops.relu(in_data)
        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])

def test_forward_relu():
    """ ReLU """
    _test_relu(np.arange(6.0, dtype=np.float32).reshape((1, 6)))

#######################################################################
# Fully Connected
# -------

def _test_fully_connected(tensor_in_sizes, filter_in_sizes, bias_in_size=None):
    """ One iteration of fully connected """

    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
        total_size_1 *= s
    for s in filter_in_sizes:
        total_size_2 *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    data_array = [f * 1.0 for f in range(1, total_size_1 + 1)]
    filter_array = [f * 1.0 for f in range(1, total_size_2 + 1)]
    assert int(total_size_1 / tensor_in_sizes[0]) == filter_in_sizes[0], \
        "input size and filter size are mismatched"

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=tensor_in_sizes, dtype='float32')
        in_filter = constant_op.constant(filter_array, shape=filter_in_sizes, dtype='float32')

        # reshape N H W C into N H*W*C
        in_data_reshape = array_ops.reshape(in_data, [tensor_in_sizes[0], -1])

        out = math_ops.mat_mul(in_data_reshape, in_filter)

        # if we have bias
        if bias_in_size:
            assert bias_in_size[0] == filter_in_sizes[1], "bias and filter size are mismatched"
            bias_array = [f * 1.0 for f in range(1, bias_in_size[0] + 1)]
            in_bias = constant_op.constant(bias_array, shape=bias_in_size, dtype='float32')
            out = nn_ops.bias_add(out, in_bias)

        data_array = np.reshape(data_array, tensor_in_sizes).astype('float32')
        compare_tflite_with_tvm(data_array, 'Placeholder:0', [in_data], [out])


def test_forward_fully_connected():
    """ Fully Connected """
    _test_fully_connected([1, 1, 1, 150], [150, 100])
    _test_fully_connected([1, 1, 1, 150], [150, 100], [100])
    _test_fully_connected([5, 1, 1, 150], [150, 100])
    _test_fully_connected([5, 1, 1, 150], [150, 100], [100])


#######################################################################
# Mobilenet
# ---------

def test_forward_mobilenet_v1():
    """Test the Mobilenet V1 TF Lite model."""
    # MobilenetV1
    tflite_model_file = tf_testing.get_workload_official(
        "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz",
        "mobilenet_v1_1.0_224.tflite")
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 224, 224, 3)).astype('float32')
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, 'input')
    tvm.testing.assert_allclose(np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]),
                                rtol=1e-5, atol=1e-5)

def test_forward_mobilenet_v2():
    """Test the Mobilenet V2 TF Lite model."""
    # MobilenetV2
    tflite_model_file = tf_testing.get_workload_official(
        "http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz",
        "mobilenet_v2_1.0_224.tflite")
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 224, 224, 3)).astype('float32')
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, 'input')
    tvm.testing.assert_allclose(np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]),
                                rtol=1e-5, atol=1e-5)

#######################################################################
# Inception
# ------------

def test_forward_inception_v3_net():
    """Test the Inception V3 TF Lite model."""
    # InceptionV3
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz",
        "inception_v3.tflite")
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 299, 299, 3)).astype('float32')
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, 'input')
    tvm.testing.assert_allclose(np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]),
                                rtol=1e-5, atol=1e-5)

def test_forward_inception_v4_net():
    """Test the Inception V4 TF Lite model."""
    # InceptionV4
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz",
        "inception_v4.tflite")
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 299, 299, 3)).astype('float32')
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, 'input')
    tvm.testing.assert_allclose(np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]),
                                rtol=1e-5, atol=1e-5)

#######################################################################
# SSD Mobilenet
# -------------

def test_forward_ssd_mobilenet_v1():
    """Test the SSD Mobilenet V1 TF Lite model."""
    # SSD MobilenetV1
    tflite_model_file = tf_testing.get_workload_official(
        "https://raw.githubusercontent.com/dmlc/web-data/master/tensorflow/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28_nopp.tgz",
        "ssd_mobilenet_v1_coco_2018_01_28_nopp.tflite")
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 300, 300, 3)).astype('float32')
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, 'normalized_input_image_tensor')
    tvm.testing.assert_allclose(np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]),
                                rtol=1e-5, atol=1e-5)

#######################################################################
# Main
# ----
if __name__ == '__main__':
    # BatchToSpaceND
    test_forward_batch_to_space_nd()

    # SpaceToBatchND
    test_forward_space_to_batch_nd()

    # Split
    test_forward_split()

    # Transpose
    test_forward_transpose()

    # Cast
    test_forward_cast()

    # Tile
    test_forward_tile()

    # Transforms
    test_forward_concatenation()
    test_forward_pad()
    test_forward_pack()
    test_forward_reshape()
    test_all_resize()
    test_forward_squeeze()

    # NN
    test_forward_convolution()
    test_forward_logistic()
    test_forward_pooling()
    test_forward_softmax()
    test_forward_tanh()
    test_forward_relu()
    test_forward_fully_connected()

    # Elemwise
    test_all_elemwise()

    # Zeros Like
    test_forward_zeros_like()

    # Reduce
    test_all_reduce()

    # End to End
    test_forward_mobilenet_v1()
    test_forward_mobilenet_v2()
    test_forward_inception_v3_net()
    test_forward_inception_v4_net()
    test_forward_ssd_mobilenet_v1()
