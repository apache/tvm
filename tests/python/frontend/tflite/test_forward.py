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
import pytest
import numpy as np
import tvm
from tvm import te
from tvm import relay
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import variables
try:
    from tensorflow import lite as interpreter_wrapper
except ImportError:
    from tensorflow.contrib import lite as interpreter_wrapper

from tvm.contrib.download import download_testdata
import tvm.relay.testing.tf as tf_testing
from packaging import version as package_version

from PIL import Image
import os

#######################################################################
# Generic run functions for TVM & TFLite
# --------------------------------------
def convert_to_list(x):
    if not isinstance(x, list):
        x = [x]
    return x


#######################################################################
# Get a real image for e2e testing
# --------------------------------
def get_real_image(im_height, im_width):
    repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/'
    img_name = 'elephant-299.jpg'
    image_url = os.path.join(repo_base, img_name)
    img_path = download_testdata(image_url, img_name, module='data')
    image = Image.open(img_path).resize((im_height, im_width))
    x = np.array(image).astype('uint8')
    data = np.reshape(x, (1, im_height, im_width, 3))
    return data

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
                            output_tensors, init_global_variables=False,
                            out_names=None, quantized=False, input_range=None):
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
        converter = tf.lite.TFLiteConverter.from_session(
            sess, input_tensors, output_tensors)

        if quantized:
            converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
            input_arrays = converter.get_input_arrays()
            input_stats = {}
            # calculate the mean and quantization scale for every input tensor,
            # with respect to its fp32 input range, defined in fake_quant.
            # s = 255/(fmax-fmin);  m = -fmin*s (the zero point)
            for i in input_arrays:
                try:
                    quant_scale = 255 / (input_range[i][1] - input_range[i][0])
                except ZeroDivisionError:
                    raise ZeroDivisionError('Min and max of the input range for tensor ' + i + ' can\'t be equal')
                mean = - input_range[i][0] * quant_scale
                input_stats[i] = (mean, quant_scale)
            converter.quantized_input_stats = input_stats

        tflite_model_buffer = converter.convert()
        tflite_output = run_tflite_graph(tflite_model_buffer, in_data)

        for device in ["llvm"]:
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                continue

            tvm_output = run_tvm_graph(tflite_model_buffer, in_data, in_node, target=device,
                                       num_output=len(out_names), out_names=out_names)

            # WARNING: the results could well be random values clipped to 0 or 255 because of badly tuned output
            # range for the specific operator. While adding test ensure that we aren't getting only clipped values
            # in output tensors that still pass the assertion. For reference see _test_elemwise_qnn_out_range()
            if quantized:
                for i in range(len(tflite_output)):
                    # allow absolute tolerance of 1 in the quantized results
                    tvm.testing.assert_allclose(tflite_output[i], tvm_output[i], atol=1, rtol=1e-5)
            else:
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
# slice
# -----

def _test_slice(data, begin, size):
    """ One iteration of SLICE """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = array_ops.slice(in_data, begin, size)
        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])

def test_forward_slice():
    """ SLICE """
    _test_slice(np.arange(4, dtype=np.float32).reshape((4, )), begin=[0], size=[2])
    _test_slice(np.arange(18, dtype=np.int32).reshape((3, 2, 3)), begin=[1, 0, 0], size=[1, 1, 3])
    # tflite 1.13 outputs nonsense values if size[i] == -1
    if package_version.parse(tf.VERSION) >= package_version.parse('1.14.0'):
        _test_slice(np.arange(8, dtype=np.int32).reshape((2, 4)), begin=[0, 1], size=[-1, -1])
        _test_slice(np.arange(5, dtype=np.int32).reshape((5, )), begin=[4], size=[-1])

#######################################################################
# Topk
# ----
def _test_topk(in_shape, k=1):
    """ One iteration of TOPK """
    data = np.random.uniform(size=in_shape).astype('float32')
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = nn_ops.top_k(in_data, k, name='TopK')
        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out[0]])

def test_forward_topk():
    """ TOPK """
    _test_topk((3,), 1)
    _test_topk((3,), 3)
    _test_topk((3, 5, 7), 3)
    _test_topk((3, 5, 7), 3)

#######################################################################
# Gather
# ------

def _test_gather(dshape, indices, axis, dtype, quantized=False, oob=False):
    """ One iteration of Gather """
    indices = np.asarray(indices).astype('int32')
    data = np.random.uniform(1, 10, size=dshape)
    data = data.astype(np.uint8) if quantized else data.astype(dtype)
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype, name="in_data")
        if axis:
            out = array_ops.gather(in_data, indices, axis=axis)
        else:
            out = array_ops.gather(in_data, indices) #tflite conversion fails for None axis
        input_range = {'in_data': (-100, 100)} if quantized else None
        try:
            compare_tflite_with_tvm([data], ['in_data:0'], [in_data], [out],
                                      quantized=quantized, input_range=input_range)
        except ValueError as e:
            if not oob:
                raise e
        except Exception as e:
            raise e

def test_forward_gather():
    """ GATHER """
    for quantized in [False, True]:
        _test_gather((4,), [1], 0, 'float32', quantized)
        _test_gather((4,), [1], None, 'int32', quantized)
        _test_gather((1, 4), [0], 0, 'int32', quantized)
        _test_gather((4,), [[[1, 0], [0, 1]]], 0, 'float32', quantized)
        _test_gather((2, 2), [[[1, 0], [0, 1]]], 1, 'int32', quantized)
        _test_gather((2, 2), [[[1, 0], [0, 1]]], None, 'float32', quantized)
        _test_gather((3, 3, 3),  [[[1, 0]]], 0, 'int32', quantized)
        _test_gather((3, 3, 3), [[[1, 0]]], 2, 'int32', quantized)
        _test_gather((4, 3, 5, 6),  [[2, 1, 0, 0]], 0, 'float32', quantized)
        _test_gather((3, 3, 3), [[[2, 1]]], -1, 'int32', quantized)
        _test_gather((4,), [16], 0, 'float32', quantized, oob=True)
        _test_gather((1, 3, 3), [12], 0, 'int32', quantized, oob=True)
        _test_gather((1, 3, 3), [20], 1, 'float32', quantized, oob=True)
        _test_gather((1, 3, 3), [20, 20], 2, 'float32', quantized, oob=True)

#######################################################################
# StridedSlice
# ------------

def _test_stridedslice(ip_shape, begin, end, stride, dtype,
                       begin_mask=0, end_mask=0, new_axis_mask=0,
                       shrink_axis_mask=0, ellipsis_mask=0, quantized=False):
    """ One iteration of a Stridedslice """
    data = np.random.uniform(size=ip_shape).astype(dtype)
    data = data.astype(np.uint8) if quantized else data.astype(dtype)
    with tf.Graph().as_default():
        in_data = tf.placeholder(dtype, ip_shape, name="in_data")
        out = array_ops.strided_slice(in_data, begin, end, stride,
                                      begin_mask=begin_mask,
                                      end_mask=end_mask,
                                      new_axis_mask=new_axis_mask,
                                      shrink_axis_mask=shrink_axis_mask,
                                      ellipsis_mask=ellipsis_mask)
        input_range = {'in_data': (-100, 100)} if quantized else None
        compare_tflite_with_tvm([data], ['in_data:0'], [in_data], [out], quantized=quantized,
                                  input_range=input_range)

def test_forward_stridedslice():
    '''test StridedSlice'''
    for quantized in [False, True]:
        _test_stridedslice((2), [1], [1], [1], 'float32', shrink_axis_mask=1, quantized=quantized)
        _test_stridedslice((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1], 'float32', quantized=quantized)
        _test_stridedslice((3, 4), [1, 0], [4, 4], [1, 1], 'float32', shrink_axis_mask=0, quantized=quantized)
        _test_stridedslice((4, 4), [1, 0], [4, 4], [1, 1], 'float32', shrink_axis_mask=2, quantized=quantized)

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
# ----

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
# Tile
# ----


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
    # dephtwise convolution with single input channel
    _test_convolution([1, 76, 64, 1], [9, 5, 1, 96], [1, 1], [1, 1], 'SAME', 'NHWC', True)


#######################################################################
# Transpose Convolution
# ---------------------

def _test_transpose_conv(tensor_in_sizes, filter_in_sizes, output_shape, strides, padding):
    """ One iteration of transpose convolution with given shapes and attributes """

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
        # in_filter layout is HWOI
        out = nn_ops.conv2d_transpose(in_data,
                                      in_filter,
                                      output_shape=output_shape,
                                      strides=strides,
                                      padding=padding)
        data_array = np.reshape(data_array, tensor_in_sizes).astype('float32')
        compare_tflite_with_tvm(data_array, 'Placeholder:0', [in_data], [out])


def test_forward_transpose_conv():
    # kernel 3x3, padding VALID
    _test_transpose_conv([4, 32, 32, 16], [3, 3, 5, 16], [4, 34, 34, 5], [1, 1], 'VALID')
    _test_transpose_conv([1, 32, 32, 16], [3, 3, 5, 16], [1, 65, 65, 5], [2, 2], 'VALID')
    _test_transpose_conv([1, 32, 32, 16], [3, 3, 5, 16], [1, 65, 34, 5], [2, 1], 'VALID')

    # kernel 2x2, padding VALID
    _test_transpose_conv([4, 32, 32, 16], [2, 2, 5, 16], [4, 33, 33, 5], [1, 1], 'VALID')
    _test_transpose_conv([1, 32, 32, 16], [2, 2, 5, 16], [1, 64, 64, 5], [2, 2], 'VALID')
    _test_transpose_conv([1, 32, 32, 16], [2, 2, 5, 16], [1, 64, 33, 5], [2, 1], 'VALID')

    # kernel 1x1, padding VALID
    _test_transpose_conv([4, 32, 32, 16], [1, 1, 5, 16], [4, 32, 32, 5], [1, 1], 'VALID')
    _test_transpose_conv([1, 32, 32, 16], [1, 1, 5, 16], [1, 63, 63, 5], [2, 2], 'VALID')
    _test_transpose_conv([1, 32, 32, 16], [1, 1, 5, 16], [1, 63, 32, 5], [2, 1], 'VALID')

    # kernel 1x1, padding SAME
    _test_transpose_conv([4, 32, 32, 16], [1, 1, 5, 16], [4, 32, 32, 5], [1, 1], 'SAME')
    _test_transpose_conv([1, 32, 32, 16], [1, 1, 5, 16], [1, 63, 63, 5], [2, 2], 'SAME')
    _test_transpose_conv([1, 32, 32, 16], [1, 1, 5, 16], [1, 63, 32, 5], [2, 1], 'SAME')


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
# Unary elemwise
# --------------

def _test_unary_elemwise(math_op, data):
    """ One iteration of unary elemwise """

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype, name='in')
        out = math_op(in_data)
        compare_tflite_with_tvm(data, ['in:0'], [in_data], [out])

#######################################################################
# Abs
# ---

def _test_abs(data):
    """ One iteration of abs """
    return _test_unary_elemwise(math_ops.abs, data)
#######################################################################
# Ceil
# ----

def _test_ceil(data):
    """ One iteration of ceil """
    return _test_unary_elemwise(math_ops.ceil, data)
#######################################################################
# Floor
# -----

def _test_floor(data):
    """ One iteration of floor """
    return _test_unary_elemwise(math_ops.floor, data)

#######################################################################
# Round
# -----

def _test_round(data):
    """ One iteration of round """
    return _test_unary_elemwise(math_ops.round, data)

#######################################################################
# Exp
# ---

def _test_exp(data):
    """ One iteration of exp """
    return _test_unary_elemwise(math_ops.exp, data)
#######################################################################
# Log
# ---

def _test_log(data):
    """ One iteration of log """
    return _test_unary_elemwise(math_ops.log, data)
#######################################################################
# Sin
# ---

def _test_sin(data):
    """ One iteration of sin """
    return _test_unary_elemwise(math_ops.sin, data)
#######################################################################
# Cos
# ---

def _test_cos(data):
    """ One iteration of cos """
    return _test_unary_elemwise(math_ops.cos, data)
#######################################################################
# Tan
# ---

def _test_tan(data):
    """ One iteration of tan """
    return _test_unary_elemwise(math_ops.tan, data)
#######################################################################
# Sqrt
# ----

def _test_sqrt(data):
    """ One iteration of sqrt """
    return _test_unary_elemwise(math_ops.sqrt, data)
#######################################################################
# Rsqrt
# -----

def _test_rsqrt(data):
    """ One iteration of rsqrt """
    return _test_unary_elemwise(math_ops.rsqrt, data)
#######################################################################
# Neg
# ---

def _test_neg(data):
    """ One iteration of neg """
    return _test_unary_elemwise(math_ops.neg, data)
#######################################################################
# Square
# ------

def _test_square(data):
    """ One iteration of square """
    return _test_unary_elemwise(math_ops.square, data)

#######################################################################
# Elu
# ---

def _test_elu(data):
    """ One iteration of elu """
    return _test_unary_elemwise(nn_ops.elu, data)

def _test_forward_unary_elemwise(test_op):
    # functions that need positive input
    if test_op.__name__ in {'_test_log', '_test_sqrt', '_test_rsqrt'}:
        test_op(np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 3)))
    else:
        test_op(np.random.uniform(-10, 10, (3, 2)).astype(np.float32))

def test_all_unary_elemwise():
    _test_forward_unary_elemwise(_test_abs)
    _test_forward_unary_elemwise(_test_floor)
    _test_forward_unary_elemwise(_test_exp)
    _test_forward_unary_elemwise(_test_log)
    _test_forward_unary_elemwise(_test_sin)
    _test_forward_unary_elemwise(_test_sqrt)
    _test_forward_unary_elemwise(_test_rsqrt)
    _test_forward_unary_elemwise(_test_neg)
    _test_forward_unary_elemwise(_test_square)
    # ceil and cos come with TFLite 1.14.0.post1 fbs schema
    if package_version.parse(tf.VERSION) >= package_version.parse('1.14.0'):
        _test_forward_unary_elemwise(_test_ceil)
        _test_forward_unary_elemwise(_test_cos)
        _test_forward_unary_elemwise(_test_round)
        # This fails with TF and Tflite 1.15.2, this could not have been tested
        # in CI or anywhere else. The failure mode is that we see a backtrace
        # from the converter that we need to provide a custom Tan operator
        # implementation.
        #_test_forward_unary_elemwise(_test_tan)
        _test_forward_unary_elemwise(_test_elu)

#######################################################################
# Element-wise
# ------------

def _test_elemwise(math_op, data, fused_activation_function=None, quantized=False, qnn_op=None):
    """ One iteration of elemwise """

    assert len(data) == 2

    # Test with two tensors
    with tf.Graph().as_default():
        in_data = [array_ops.placeholder(shape=data[0].shape, dtype='float32', name='in_0'),
                   array_ops.placeholder(shape=data[1].shape, dtype='float32', name='in_1')]

        if quantized:
            # fake_quant will keep the tensors in float32 until the conversion in the session
            inq_data = [tf.quantization.fake_quant_with_min_max_args(in_data[0], min=-100, max=100, name="inq_0"),
                        tf.quantization.fake_quant_with_min_max_args(in_data[1], min=-50, max=50, name="inq_1")]
            input_range = {'inq_0': (-100, 100), 'inq_1': (-50, 50)}
            out = math_op(inq_data[0], inq_data[1])
            out = with_fused_activation_function(out, fused_activation_function)
            # set the fp32 output range with respect to the operation
            out_min, out_max = _test_elemwise_qnn_out_range(qnn_op)
            out = tf.quantization.fake_quant_with_min_max_args(out, min=out_min, max=out_max, name="out")
            compare_tflite_with_tvm(data, ['inq_0:0', 'inq_1:0'], inq_data, [out],
                                    quantized=True, input_range=input_range)
        else:
            out = math_op(in_data[0], in_data[1])
            out = with_fused_activation_function(out, fused_activation_function)
            compare_tflite_with_tvm(data, ['in_0:0', 'in_1:0'], in_data, [out])

    # Test with tensor and constant
    with tf.Graph().as_default():
        in_data = [array_ops.placeholder(shape=data[0].shape, dtype='float32', name='in_0')]

        if quantized:
            inq_data = [tf.quantization.fake_quant_with_min_max_args(in_data[0], min=-100, max=100, name="inq_0")]
            inq_const = tf.quantization.fake_quant_with_min_max_args(data[1], min=-50, max=50, name="const_tensor")
            input_range = {'inq_0': (-100, 100)}
            # the 2nd tensor is treated as constant and directly added as part of the operation
            out = math_op(inq_data, ops.convert_to_tensor(inq_const, dtype='float32', name='inq_const'))
            out = with_fused_activation_function(out, fused_activation_function)
            out_min, out_max = _test_elemwise_qnn_out_range(qnn_op)
            out = tf.quantization.fake_quant_with_min_max_args(out, min=out_min, max=out_max, name="out")
            compare_tflite_with_tvm(data[0], ['inq_0:0'], inq_data, [out], quantized=True, input_range=input_range)
        else:
            out = math_op(in_data[0], ops.convert_to_tensor(data[1], dtype=data[1].dtype))
            out = with_fused_activation_function(out, fused_activation_function)
            compare_tflite_with_tvm(data[0], ['in_0:0'], in_data, [out])

    # Test with constant and tensor
    with tf.Graph().as_default():
        in_data = [array_ops.placeholder(shape=data[1].shape, dtype='float32', name='in_1')]

        if quantized:
            inq_const = tf.quantization.fake_quant_with_min_max_args(data[0], min=-100, max=100, name="const_tensor")
            inq_data = [tf.quantization.fake_quant_with_min_max_args(in_data[0], min=-50, max=50, name="inq_1")]
            input_range = {'inq_1': (-50, 50)}
            # the 1st tensor is treated as constant and directly added as part of the operation
            out = math_op(ops.convert_to_tensor(inq_const, dtype='float32', name='inq_const'), inq_data)
            out = with_fused_activation_function(out, fused_activation_function)
            out_min, out_max = _test_elemwise_qnn_out_range(qnn_op)
            out = tf.quantization.fake_quant_with_min_max_args(out, min=out_min, max=out_max, name="out")
            compare_tflite_with_tvm(data[1], ['inq_1:0'], inq_data, [out], quantized=True, input_range=input_range)
        else:
            out = math_op(ops.convert_to_tensor(data[0], dtype=data[0].dtype), in_data[0])
            out = with_fused_activation_function(out, fused_activation_function)
            compare_tflite_with_tvm(data[1], ['in_1:0'], in_data, [out])

#######################################################################
# Add
# ---

def _test_add(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """ One iteration of add """
    return _test_elemwise(math_ops.add, data, fused_activation_function, quantized, qnn_op)

#######################################################################
# Subtract
# --------

def _test_sub(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """ One iteration of subtract """
    return _test_elemwise(math_ops.subtract, data, fused_activation_function, quantized, qnn_op)
#######################################################################
# Mul
# ---

def _test_mul(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """ One iteration of mul """
    return _test_elemwise(math_ops.multiply, data, fused_activation_function, quantized, qnn_op)

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
#######################################################################
# Greater_equal
# -------------

def _test_greater_equal(data):
    """ One iteration of greater_equal """
    return _test_elemwise(math_ops.greater_equal, data)
#######################################################################
# Less
# ----

def _test_less(data):
    """ One iteration of less """
    return _test_elemwise(math_ops.less, data)
#######################################################################
# Less_equal
# ----------

def _test_less_equal(data):
    """ One iteration of less_equal """
    return _test_elemwise(math_ops.less_equal, data)
#######################################################################
# Equal
# -----

def _test_equal(data):
    """ One iteration of equal """
    return _test_elemwise(math_ops.equal, data)
#######################################################################
# Not_equal
# ---------

def _test_not_equal(data):
    """ One iteration of not_equal"""
    return _test_elemwise(math_ops.not_equal, data)
#######################################################################
# Squared_difference
# ------------------

def _test_squared_difference(data):
    """ One iteration of squared difference """
    return _test_elemwise(math_ops.squared_difference, data)

#######################################################################
# Floor_divide
# ------------

def _test_floor_divide(data):
    """ One iteration of floor_div"""
    return _test_elemwise(math_ops.floordiv, data)

#######################################################################
# Floor_mod
# ---------

def _test_floor_mod(data):
    """ One iteration of floor_mod"""
    return _test_elemwise(math_ops.floormod, data)

def _test_forward_elemwise(testop):
    """ Elewise"""
    testop([np.arange(6.0, dtype=np.float32).reshape((2, 1, 1, 3)),
              np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 1, 3))])
    testop([np.arange(6.0, dtype=np.float32).reshape((2, 1, 3)),
               np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 3))])
    testop([np.arange(3.0, dtype=np.float32).reshape((1, 3)),
               np.arange(1.0, 4.0, dtype=np.float32).reshape((1, 3))])

def _test_forward_elemwise_quantized(testop):
    testop([np.array(np.random.uniform(0, 255, (3, 6)), dtype=np.uint8),
            np.array(np.random.uniform(0, 255, (3, 6)), dtype=np.uint8)], quantized=True, qnn_op=testop)

def _test_elemwise_qnn_out_range(qnn_op):
    # set the fake_quant output range with respect to the input tensors float32 range
    qnn_out_range = {
        _test_add: (-150, 150),
        _test_sub: (-150, 150),
        _test_mul: (-5e+3, 5e+3),
    }

    return qnn_out_range[qnn_op]

def test_all_elemwise():
    _test_forward_elemwise(_test_add)
    _test_forward_elemwise_quantized(_test_add)
    _test_forward_elemwise(partial(_test_add, fused_activation_function="RELU"))
    # this is broken with tf upgrade 1.15.2 and hits a segfault that needs
    # further investigation.
    # _test_forward_elemwise(partial(_test_add, fused_activation_function="RELU6"))
    _test_forward_elemwise(_test_sub)
    _test_forward_elemwise_quantized(_test_sub)
    _test_forward_elemwise(partial(_test_sub, fused_activation_function="RELU"))
    _test_forward_elemwise(partial(_test_sub, fused_activation_function="RELU6"))
    _test_forward_elemwise(_test_mul)
    _test_forward_elemwise_quantized(_test_mul)
    _test_forward_elemwise(partial(_test_mul, fused_activation_function="RELU"))
    _test_forward_elemwise(partial(_test_mul, fused_activation_function="RELU6"))
    _test_forward_elemwise(_test_div)
    _test_forward_elemwise(partial(_test_div, fused_activation_function="RELU"))
    _test_forward_elemwise(partial(_test_div, fused_activation_function="RELU6"))
    _test_forward_elemwise(_test_pow)
    _test_forward_elemwise(_test_maximum)
    _test_forward_elemwise(_test_minimum)
    _test_forward_elemwise(_test_greater)
    _test_forward_elemwise(_test_squared_difference)
    _test_forward_elemwise(_test_greater_equal)
    _test_forward_elemwise(_test_less)
    _test_forward_elemwise(_test_less_equal)
    _test_forward_elemwise(_test_equal)
    _test_forward_elemwise(_test_not_equal)
    if package_version.parse(tf.VERSION) >= package_version.parse('1.14.0'):
        _test_forward_elemwise(_test_floor_divide)
        _test_forward_elemwise(_test_floor_mod)

#######################################################################
# Logical operators
# -----------------

def _test_logical_binary(logical_bin_op, data):

    with tf.Graph().as_default():
        in_data = [array_ops.placeholder(shape=data[0].shape, dtype='bool', name='in_0'),
                   array_ops.placeholder(shape=data[1].shape, dtype='bool', name='in_1')]
        out = logical_bin_op(in_data[0], in_data[1], name='out')
        compare_tflite_with_tvm(data, ['in_0:0', 'in_1:0'], in_data, [out])

def _test_forward_logical_and(data):
    """ One iteration of logical and """
    return _test_logical_binary(math_ops.logical_and, data)

def _test_forward_logical_or(data):
    """ One iteration of logical or """
    return _test_logical_binary(math_ops.logical_or, data)

def test_all_logical():
    data = [np.random.choice(a=[False, True], size=(2, 3, 4)).astype('bool'),
            np.random.choice(a=[False, True], size=(2, 3, 4)).astype('bool')]
    # boolean dtype is not supported by older versions than TFLite 1.15.0
    if package_version.parse(tf.VERSION) >= package_version.parse('1.15.0'):
        _test_forward_logical_and(data)
        _test_forward_logical_or(data)

#######################################################################
# Zeros like
# ----------

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

def _test_reduce_quantize(math_op, data, keep_dims=None):
    """ One iteration of reduce """

    assert len(data) == 2

    # Test with tensor and constant
    with tf.Graph().as_default():
        in_data = [array_ops.placeholder(shape=data[0].shape, dtype="float32", name='in')]
        inq_data = [tf.quantization.fake_quant_with_min_max_args(in_data[0], min=-100, max=100, name="inq_0")]
        input_range = {'inq_0': (-100, 100)}
        out = math_op(inq_data, data[1], keep_dims)
        out = tf.quantization.fake_quant_with_min_max_args(out, min=-200, max=200, name="out")
        compare_tflite_with_tvm([data[0]], ['inq_0:0'], [inq_data[0]], [out], quantized=True, input_range=input_range)


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

def _test_reduce_mean(data, keep_dims=None, quantized=False):
    """ One iteration of reduce_mean """
    if quantized:
        return _test_reduce_quantize(math_ops.reduce_mean, data, keep_dims)
    else:
        return _test_reduce(math_ops.reduce_mean, data, keep_dims)

#######################################################################
# Reduce_prod
# -----------

def _test_reduce_prod(data, keep_dims=None):
    """ One iteration of reduce_prod """
    return _test_reduce(math_ops.reduce_prod, data, keep_dims)

#######################################################################
# Reduce_sum
# -----------

def _test_reduce_sum(data, keep_dims=None):
    """ One iteration of reduce_sum """
    return _test_reduce(math_ops.reduce_sum, data, keep_dims)

#######################################################################
# Reduce_any
# ----------

def _test_reduce_any(data, keep_dims=None):
    """ One iteration of reduce_any """
    return _test_reduce(math_ops.reduce_any, data, keep_dims)

def _test_forward_reduce(testop, dtype="float32"):
    """ Reduce """
    if dtype == 'bool':
        data0 = [np.random.choice(a=[False, True], size=(16, 16, 16, 16)).astype(dtype),
                 None]
        data1 = [np.random.choice(a=[False, True], size=(16, 16, 16, 16)).astype(dtype),
                 np.array([1, 2], dtype=np.int32)]
    else:
        data0 = [np.random.rand(16, 16, 16, 16).astype(dtype), None]
        data1 = [np.random.rand(16, 16, 16, 16).astype(dtype), np.array([1, 2], dtype=np.int32)]
    testop(data0)
    testop(data0, keep_dims=False)
    testop(data0, keep_dims=True)
    testop(data1)
    testop(data1, keep_dims=False)
    testop(data1, keep_dims=True)

def _test_forward_reduce_quantized(testop):
    data0 = [np.array(np.random.uniform(0, 255, (3, 6)), dtype=np.uint8), np.array([1, 2], dtype=np.int32)]
    testop(data0, quantized=True)
    testop(data0, keep_dims=False, quantized=True)
    testop(data0, keep_dims=True, quantized=True)

def test_all_reduce():
    _test_forward_reduce(_test_reduce_min)
    _test_forward_reduce(_test_reduce_max)
    _test_forward_reduce(_test_reduce_mean)
    _test_forward_reduce_quantized(_test_reduce_mean)
    _test_forward_reduce(_test_reduce_prod)
    _test_forward_reduce(_test_reduce_sum)
    if package_version.parse(tf.VERSION) >= package_version.parse('1.15.0'):
        _test_forward_reduce(_test_reduce_any, dtype="bool")


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

def _test_pad(data, mode="CONSTANT", quantized=False):
    """ One iteration of PAD """

    assert len(data) == 2

    # Test with tensor and constant
    with tf.Graph().as_default():
        in_data = [array_ops.placeholder(shape=data[0].shape, dtype='float32', name='in')]

        if quantized:
            # fake_quant will keep the tensors in float32 until the conversion in the session
            input_range = {'inq_0': (-100, 100)}
            inq_data = [tf.quantization.fake_quant_with_min_max_args(in_data[0],
                                                                     min=-100,
                                                                     max=100,
                                                                     name="inq_0")]
            out = array_ops.pad(inq_data[0], ops.convert_to_tensor(data[1], dtype=data[1].dtype), mode=mode)
            compare_tflite_with_tvm([data[0]], ['inq_0:0'], inq_data, [out], quantized=True,
                                    input_range=input_range)
        else:
            out = array_ops.pad(in_data[0], ops.convert_to_tensor(data[1], dtype=data[1].dtype), mode=mode)
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
    _test_pad([np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 3)),
               np.array([[1, 1], [2, 2]], dtype=np.int32)], mode="REFLECT")
    _test_pad([np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 3)),
               np.array([[1, 1], [2, 2]], dtype=np.int32)], mode="SYMMETRIC")
    _test_pad([np.arange(0, 256, dtype=np.uint8).reshape((1, 256)),
               np.array([[1, 1], [2, 2]], dtype=np.int32)], quantized=True)


#######################################################################
# Pack
# ----

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
# Unpack
# ------

def _test_unpack(data, axis, num_unpacks):
    """ One iteration of UNPACK """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = gen_array_ops.unpack(in_data, num=num_unpacks, axis=axis, name='unpack')
        out_names = ['out_' + str(n) + ':0' for n in range(num_unpacks)]
        compare_tflite_with_tvm([data], 'Placeholder:0',  [in_data], out, out_names=out_names)

def test_forward_unpack():
    """ UNPACK """
    _test_unpack(np.array(np.random.uniform(0, 5, (3, 1)), dtype=np.int32), axis=1, num_unpacks=1)
    _test_unpack(np.array(np.random.uniform(0, 5, (3, 4)), dtype=np.float32), axis=0, num_unpacks=3)
    # tflite 1.13 doesn't accept negative axis
    if package_version.parse(tf.VERSION) >= package_version.parse('1.14.0'):
        _test_unpack(np.array(np.random.uniform(0, 5, (3, 6)), dtype=np.int32), axis=-2, num_unpacks=3)
        _test_unpack(np.array(np.random.uniform(0, 5, (2, 3, 4)), dtype=np.int32), axis=-3, num_unpacks=2)


#######################################################################
# Local response normalization
# ----------------------------

def _test_local_response_normalization(data, depth_radius, bias, alpha, beta):
    """ One iteration of LOCAL_RESPONSE_NORMALIZATION """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype='float32', name='in_0')
        out = nn_ops.local_response_normalization(in_data, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta)
        compare_tflite_with_tvm(data, 'in_0:0', [in_data], [out])

def test_forward_local_response_normalization():
    """ LOCAL_RESPONSE_NORMALIZATION """
    data = np.random.uniform(size=(1, 6, 4, 3)).astype('float32')
    # LOCAL_RESPONSE_NORMALIZATION come with TFLite >= 1.14.0 fbs schema
    if package_version.parse(tf.VERSION) >= package_version.parse('1.14.0'):
        _test_local_response_normalization(data, depth_radius=5, bias=1, alpha=1, beta=0.5)


#######################################################################
# L2 normalization
# ----------------

def _test_l2_normalization(data, axis, fused_activation_function=None):
    """ One iteration of L2_NORMALIZATION """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = nn_impl.l2_normalize(in_data, axis)
        out = with_fused_activation_function(out, fused_activation_function)
        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])

def test_forward_l2_normalization():
    """ L2_NORMALIZATION """
    data = np.random.uniform(size=(3, 6, 4)).astype('float32')
    _test_l2_normalization(data, axis=2)
    _test_l2_normalization(data, axis=2, fused_activation_function="RELU")

#######################################################################
# Logistic
# --------

def _test_logistic(data, quantized=False):
    """ One iteration of LOGISTIC """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype='float32', name='in_0')

        if quantized:
            inq_data = tf.quantization.fake_quant_with_min_max_args(in_data, min=-5, max=5, name="inq_0")
            input_range = {'inq_0': (-5, 5)}
            out = math_ops.sigmoid(inq_data)
            out = tf.quantization.fake_quant_with_min_max_args(out, min=0, max=1, name="out")
            compare_tflite_with_tvm(data, 'inq_0:0', [inq_data], [out], quantized=True, input_range=input_range)
        else:
            out = math_ops.sigmoid(in_data)
            compare_tflite_with_tvm(data, 'in_0:0', [in_data], [out])

def test_forward_logistic():
    """ LOGISTIC """
    _test_logistic(np.arange(6.0, dtype=np.float32).reshape((1, 6)))
    _test_logistic(np.random.uniform(0, 255, (3, 6)).astype(np.uint8), quantized=True)

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
# ----

def _test_tanh(data):
    """ One iteration of TANH """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = math_ops.tanh(in_data)
        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])

def test_forward_tanh():
    """ TANH """
    _test_tanh(np.arange(6.0, dtype=np.float32).reshape((1, 6)))

#######################################################################
# ReLu
# ----

def _test_relu(data):
    """ One iteration of ReLU """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = nn_ops.relu(in_data)
        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])

def test_forward_relu():
    """ ReLU """
    _test_relu(np.arange(6.0, dtype=np.float32).reshape((1, 6)))

def _test_prelu(data, alpha):
    """ One iteration of PReLU """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        # This specific pattern will be replaced into PRelu by tflite
        out = nn_ops.relu(in_data) + (-alpha * nn_ops.relu(-in_data))
        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])

def test_forward_prelu():
    """ PReLU """
    _test_prelu(np.random.uniform(-5, 5, size=(1, 32, 32, 3)).astype("float32"), np.full((3,), 0.2, dtype="float32"))
    _test_prelu(np.random.uniform(-5, 5, size=(1, 32, 32, 3)).astype("float32"), np.full((1, 1, 3), 0.2, dtype="float32"))

#######################################################################
# DepthToSpace
# ------------

def _test_depthtospace(data, block_size):
    """ One iteration of depth_to_space operation with given data and block size """

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = array_ops.depth_to_space(in_data, block_size)
        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])

def test_forward_depthtospace():
    # DEPTH_TO_SPACE comes with TFLite >= 1.15.0 fbs schema
    if package_version.parse(tf.VERSION) >= package_version.parse('1.15.0'):
        _test_depthtospace(np.random.normal(size=[1, 32, 32, 4]).astype("float32"), 2)
        _test_depthtospace(np.random.normal(size=[1, 16, 8, 32]).astype("float32"), 4)

#######################################################################
# SpaceToDepth
# ------------

def _test_spacetodepth(data, block_size):
    """ One iteration of space_to_depth operation with given data and block size """

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = array_ops.space_to_depth(in_data, block_size)
        compare_tflite_with_tvm(data, 'Placeholder:0', [in_data], [out])

def test_forward_spacetodepth():
    _test_spacetodepth(np.random.normal(size=[1, 32, 32, 4]).astype("float32"), 2)
    _test_spacetodepth(np.random.normal(size=[1, 16, 8, 32]).astype("float32"), 4)

#######################################################################
# Fully Connected
# ---------------

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
# Custom Operators
# ----------------

def test_detection_postprocess():
    tf_model_file = tf_testing.get_workload_official(
        "http://download.tensorflow.org/models/object_detection/"
        "ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz",
        "ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/tflite_graph.pb"
    )
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        tf_model_file,
        input_arrays=["raw_outputs/box_encodings", "raw_outputs/class_predictions"],
        output_arrays=[
            "TFLite_Detection_PostProcess",
            "TFLite_Detection_PostProcess:1",
            "TFLite_Detection_PostProcess:2",
            "TFLite_Detection_PostProcess:3"
        ],
        input_shapes={
            "raw_outputs/box_encodings": (1, 1917, 4),
            "raw_outputs/class_predictions": (1, 1917, 91),
        },
    )
    converter.allow_custom_ops = True
    converter.inference_type = tf.lite.constants.FLOAT
    tflite_model = converter.convert()
    np.random.seed(0)
    box_encodings = np.random.uniform(size=(1, 1917, 4)).astype('float32')
    class_predictions = np.random.uniform(size=(1, 1917, 91)).astype('float32')
    tflite_output = run_tflite_graph(tflite_model, [box_encodings, class_predictions])
    tvm_output = run_tvm_graph(tflite_model, [box_encodings, class_predictions],
                               ["raw_outputs/box_encodings", "raw_outputs/class_predictions"], num_output=4)
    # check valid count is the same
    assert tvm_output[3] == tflite_output[3]
    valid_count = tvm_output[3][0]
    tvm_boxes = tvm_output[0][0][:valid_count]
    tvm_classes = tvm_output[1][0][:valid_count]
    tvm_scores = tvm_output[2][0][:valid_count]
    # check the output data is correct
    tvm.testing.assert_allclose(np.squeeze(tvm_boxes), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-5)
    tvm.testing.assert_allclose(np.squeeze(tvm_classes), np.squeeze(tflite_output[1]), rtol=1e-5, atol=1e-5)
    tvm.testing.assert_allclose(np.squeeze(tvm_scores), np.squeeze(tflite_output[2]), rtol=1e-5, atol=1e-5)


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
# Mobilenet V3
# ------------

def test_forward_mobilenet_v3():
    """Test the Mobilenet V3 TF Lite model."""
    # In MobilenetV3, some ops are not supported before tf 1.15 fbs schema
    if package_version.parse(tf.VERSION) < package_version.parse('1.15.0'):
        return
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz",
        "v3-large_224_1.0_float/v3-large_224_1.0_float.tflite")
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 224, 224, 3)).astype('float32')
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, 'input')
    tvm.testing.assert_allclose(np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]),
                                rtol=1e-5, atol=1e-5)

#######################################################################
# Inception
# ---------

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

def test_forward_qnn_inception_v1_net():
    """Test the Quantized TFLite Inception model."""
    # InceptionV1
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_224_quant_20181026.tgz",
        "inception_v1_224_quant.tflite")
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()

    # Test image. Checking the labels because the requantize implementation is different between
    # TFLite and Relay. This cause final output numbers to mismatch. So, testing accuracy via
    # labels. Also, giving a real image, instead of random inputs.
    data = get_real_image(224, 224)

    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tflite_predictions = np.squeeze(tflite_output)
    tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
    tvm_output = run_tvm_graph(tflite_model_buf, data, 'input')
    tvm_predictions = np.squeeze(tvm_output)
    tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
    tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)

def test_forward_qnn_mobilenet_v1_net():
    """Test the Quantized TFLite Mobilenet V1 model."""
    # MobilenetV1
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz",
        "mobilenet_v1_1.0_224_quant.tflite")
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()

    # Test image. Checking the labels because the requantize implementation is different between
    # TFLite and Relay. This cause final output numbers to mismatch. So, testing accuracy via
    # labels. Also, giving a real image, instead of random inputs.
    data = get_real_image(224, 224)

    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tflite_predictions = np.squeeze(tflite_output)
    tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
    tvm_output = run_tvm_graph(tflite_model_buf, data, 'input')
    tvm_predictions = np.squeeze(tvm_output)
    tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
    tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)

def test_forward_qnn_mobilenet_v2_net():
    """Test the Quantized TFLite Mobilenet V2 model."""
    # MobilenetV2
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz",
        "mobilenet_v2_1.0_224_quant.tflite")
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()

    # Test image. Checking the labels because the requantize implementation is different between
    # TFLite and Relay. This cause final output numbers to mismatch. So, testing accuracy via
    # labels. Also, giving a real image, instead of random inputs.
    data = get_real_image(224, 224)

    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tflite_predictions = np.squeeze(tflite_output)
    tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
    tvm_output = run_tvm_graph(tflite_model_buf, data, 'input')
    tvm_predictions = np.squeeze(tvm_output)
    tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
    tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)

#######################################################################
# Mobilenet V3 Quantized
# ----------------------

def test_forward_qnn_mobilenet_v3_net():
    """Test the Quantized TFLite Mobilenet V3 model."""
    # In MobilenetV3, some ops are not supported before tf 1.15 fbs schema
    if package_version.parse(tf.VERSION) < package_version.parse('1.15.0'):
        pytest.skip("Unsupported in tflite < 1.15.0")
    else:
        pytest.skip("This segfaults with tensorflow 1.15.2 and above")

    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_uint8.tgz",
        "v3-large_224_1.0_uint8/v3-large_224_1.0_uint8.tflite")
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()

    # Test image. Checking the labels because the requantize implementation is different between
    # TFLite and Relay. This cause final output numbers to mismatch. So, testing accuracy via
    # labels. Also, giving a real image, instead of random inputs.
    data = get_real_image(224, 224)

    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tflite_predictions = np.squeeze(tflite_output)
    tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
    tvm_output = run_tvm_graph(tflite_model_buf, data, 'input')
    tvm_predictions = np.squeeze(tvm_output)
    tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
    tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)

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
    np.random.seed(0)
    data = np.random.uniform(size=(1, 300, 300, 3)).astype('float32')
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, 'normalized_input_image_tensor', num_output=2)
    for i in range(2):
        tvm.testing.assert_allclose(np.squeeze(tvm_output[i]), np.squeeze(tflite_output[i]),
                                    rtol=1e-5, atol=2e-5)

#######################################################################
# MediaPipe
# -------------

def test_forward_mediapipe_hand_landmark():
    """Test MediaPipe 2D hand landmark TF Lite model."""
    # MediaPipe 2D hand landmark TF
    tflite_model_file = download_testdata(
        "https://github.com/google/mediapipe/raw/master/mediapipe/models/hand_landmark.tflite",
        "hand_landmark.tflite")
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 256, 256, 3)).astype('float32')
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, 'input_1', num_output=2)
    for i in range(2):
        tvm.testing.assert_allclose(np.squeeze(tvm_output[i]), np.squeeze(tflite_output[i]),
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
    test_forward_unpack()
    test_forward_reshape()
    test_all_resize()
    test_forward_squeeze()
    test_forward_slice()
    test_forward_topk()
    test_forward_gather()
    test_forward_stridedslice()
    test_forward_depthtospace()
    test_forward_spacetodepth()

    # NN
    test_forward_convolution()
    test_forward_transpose_conv()
    test_forward_logistic()
    test_forward_pooling()
    test_forward_softmax()
    test_forward_tanh()
    test_forward_relu()
    test_forward_prelu()
    test_forward_fully_connected()
    test_forward_l2_normalization()
    test_forward_local_response_normalization()

    # Elemwise
    test_all_elemwise()

    # Unary elemwise
    test_all_unary_elemwise()
    # Zeros Like
    test_forward_zeros_like()

    # Reduce
    test_all_reduce()

    # Logical
    test_all_logical()

    # Detection_PostProcess
    test_detection_postprocess()

    # End to End
    test_forward_mobilenet_v1()
    test_forward_mobilenet_v2()
    test_forward_mobilenet_v3()
    test_forward_inception_v3_net()
    test_forward_inception_v4_net()
    test_forward_ssd_mobilenet_v1()
    test_forward_mediapipe_hand_landmark()

    # End to End quantized
    test_forward_qnn_inception_v1_net()
    test_forward_qnn_mobilenet_v1_net()
    test_forward_qnn_mobilenet_v2_net()
    #This also fails with a segmentation fault in my run
    #with Tflite 1.15.2
    test_forward_qnn_mobilenet_v3_net()
