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
Tensorflow testcases
====================
This article is a test script to test tensorflow operator with NNVM.
"""
from __future__ import print_function
import numpy as np
import nnvm.compiler
import tvm
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import init_ops
from tensorflow.core.framework import graph_pb2

import tvm.relay.testing.tf as tf_testing

#######################################################################
# Generic run functions for TVM & tensorflow
# ------------------------------------------
def convert_to_list(x):
    if not isinstance(x, list):
        x = [x]
    return x

def run_tvm_graph(graph_def, input_data, input_node, num_output=1, target='llvm', out_names=None):
    """ Generic function to compile on nnvm and execute on tvm """
    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)

    layout = None
    if target == "cuda":
        layout = "NCHW"
    target_host = 'llvm'

    if isinstance(input_data, list):
        shape_dict = {}
        dtype_dict = {}
        for i, e in enumerate(input_node):
            shape_dict[e] = input_data[i].shape
            dtype_dict[e] = input_data[i].dtype
    else:
        shape_dict = {input_node: input_data.shape}
        dtype_dict = {input_node: input_data.dtype}

    sym, params = nnvm.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict, outputs=out_names)
    graph, lib, params = nnvm.compiler.build(sym, target=target, target_host=target_host, shape=shape_dict,
                                             dtype=dtype_dict, params=params)

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
    assert out_names is None or num_output == len(out_names),"out_names: {} num_output: {}".format(
                                                              out_names, num_output)
    tvm_output_list = []
    for i in range(0, num_output):
        tvm_output = m.get_output(i)
        tvm_output_list.append(tvm_output.asnumpy())
    return tvm_output_list

def run_tf_graph(sess, input_data, input_node, output_node):
    """ Generic function to execute tensorflow """
    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)
    output_node = convert_to_list(output_node)

    tensor = [0] * len(output_node)
    for i in range(len(output_node)):
        tensor[i] = sess.graph.get_tensor_by_name(output_node[i])

    input_dict = {}
    for i, e in enumerate(input_node):
        input_dict[e] = input_data[i]

    output_data = sess.run(tensor, input_dict)
    return output_data


def compare_tf_with_tvm(in_data, in_name, out_name, init_global_variables=False, no_gpu=False):
    """Generic function to generate and compare tensorflow and TVM output"""

    out_name = convert_to_list(out_name)
    out_node = [0]*len(out_name)
    for i in range(len(out_name)):
        out_node[i] = out_name[i].split(':')[0] if ":" in out_name[i] else out_name[i]

    in_data = convert_to_list(in_data)
    in_name = convert_to_list(in_name)
    in_node = [0]*len(in_name)
    for i in range(len(in_name)):
        in_node[i] = in_name[i].split(':')[0] if ":" in in_name[i] else in_name[i]

    with tf.Session() as sess:
        if init_global_variables:
            sess.run(variables.global_variables_initializer())
        final_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(add_shapes=True),
            out_node,
            )
        tf_output = run_tf_graph(sess, in_data, in_name, out_name)

        for device in ["llvm", "cuda"]:
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                continue
            if no_gpu and device == 'cuda':
                continue

            tvm_output = run_tvm_graph(final_graph_def, in_data, in_node,
                                       num_output=len(out_node), target=device, out_names=out_name)
            # since the names from tensorflow and nnvm runs are not exactly same,
            # first len(tf_output) will be compared
            for i in range(len(tf_output)):
                tvm.testing.assert_allclose(tf_output[i], tvm_output[i], atol=1e-5, rtol=1e-5)

        sess.close()

def is_gpu_available():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    gpu_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
    if len(gpu_list) > 0:
        print("Tensorflow GPU:", gpu_list)
        return True
    else:
        return False

#######################################################################
# Pooling
# -------
def _test_pooling_iteration(input_shape, **kwargs):
    """ One iteration of pool operation with given shapes and attributes """

    x = -np.arange(
        np.prod(input_shape), dtype=np.float32).reshape(input_shape) - 1

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=input_shape, dtype='float32')
        nn_ops.pool(in_data, **kwargs)

        if kwargs['pooling_type'] == 'MAX':
            out_name = 'max_pool:0'
        else:
            out_name = 'avg_pool:0'

        compare_tf_with_tvm(x, 'Placeholder:0', out_name)

def _test_pooling(input_shape, **kwargs):
    _test_pooling_iteration(input_shape, **kwargs)

    if is_gpu_available():
        input_shape = [input_shape[ii] for ii in (0, 3, 1, 2)]
        kwargs['data_format'] = 'NCHW'
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
                      dilations, strides, padding, data_format):
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
        if data_format == 'NHWC':
            strides = [1] + strides + [1]
            dilations = [1] + dilations + [1]
        else:
            strides = [1, 1] + strides
            dilations = [1, 1] + dilations

        nn_ops.conv2d(in_data,
                      in_filter,
                      strides=strides,
                      padding=padding,
                      data_format=data_format)

        compare_tf_with_tvm(np.reshape(data_array, tensor_in_sizes).astype('float32'),
                            'Placeholder:0', 'Conv2D:0')

def test_forward_convolution():
    if is_gpu_available():
        _test_convolution([4, 176, 8, 8], [1, 1, 176, 32], [1, 1], [1, 1], 'SAME', 'NCHW')
        _test_convolution([4, 19, 17, 17], [3, 3, 19, 19], [1, 1], [2, 2], 'VALID', 'NCHW')
        _test_convolution([4, 124, 17, 17], [1, 1, 124, 19], [1, 1], [1, 1], 'SAME', 'NCHW')
        _test_convolution([4, 12, 17, 17], [3, 3, 12, 32], [1, 1], [2, 2], 'VALID', 'NCHW')

    _test_convolution([4, 8, 8, 176], [1, 1, 176, 32], [1, 1], [1, 1], 'SAME', 'NHWC')
    _test_convolution([4, 17, 17, 19], [3, 3, 19, 19], [1, 1], [2, 2], 'VALID', 'NHWC')
    _test_convolution([4, 17, 17, 124], [1, 1, 124, 19], [1, 1], [1, 1], 'SAME', 'NHWC')
    _test_convolution([4, 17, 17, 12], [3, 3, 12, 32], [1, 1], [2, 2], 'VALID', 'NHWC')

#######################################################################
# Reshape
# -------

def _test_reshape(data, out_shape):
    """ One iteration of reshape operation with given data and out shape """

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        array_ops.reshape(in_data, out_shape)

        compare_tf_with_tvm(data, 'Placeholder:0', 'Reshape:0')

def test_forward_reshape():
    _test_reshape(np.arange(6.0), [2, 3])
    _test_reshape(np.arange(6), [-1, 2])
    _test_reshape(np.arange(6), [3, -1])
    _test_reshape(np.arange(6), [-1])

#######################################################################
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
            array_ops.squeeze(in_data, squeeze_dims)
        else:
            array_ops.squeeze(in_data)

        compare_tf_with_tvm(data, 'Placeholder:0', 'Squeeze:0')

def test_forward_squeeze():
    """ Squeeze """

    # Nothing to squeeze.
    _test_squeeze(np.arange(2).reshape((2)))
    _test_squeeze(np.arange(6).reshape((2, 3)))

    # Squeeze the middle element away.
    _test_squeeze(np.arange(4).reshape((2, 1, 2)))

    # Squeeze on both ends.
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)))

    # Positive squeeze dim index.
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [0])
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [2, 4])
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [0, 4, 2])

    # Negative squeeze dim index.
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [-1])
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [-3, -5])
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [-3, -5, -1])

#######################################################################
# ConcatV2
# --------

def _test_concat_v2(data, dim):
    """ One iteration of ConcatV2 """

    with tf.Graph().as_default():
        gen_array_ops._concat_v2(data, dim)

        compare_tf_with_tvm(data, ['ConcatV2/values_0:0', 'ConcatV2/values_1:0'],
                            'ConcatV2:0')

def _test_forward_concat_v2():
    t1 = np.array([])
    t2 = np.array([])
    test_concat_v2([t1, t2], 0)

    t1 = np.array([[1, 2, 3], [4, 5, 6]])
    t2 = np.array([[7, 8, 9], [10, 11, 12]])

    _test_concat_v2([t1, t2], 1)

#######################################################################
# Sigmoid
# -------

def _test_sigmoid(data):
    """ One iteration of sigmoid """

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        sigmoid_out = math_ops.sigmoid(in_data)

        compare_tf_with_tvm(data, 'Placeholder:0', 'Sigmoid:0')

def test_forward_sigmoid():
    """ Sigmoid """

    _test_sigmoid(np.random.uniform(size=(3, 4, 4, 3)).astype('float32'))

#######################################################################
# Argmin/Argmax
# -------------

def _test_argx(func, data, **kwargs):

    with tf.Graph().as_default():
        inp = array_ops.placeholder(shape=data.shape, dtype=data.dtype, name="c0")
        func(inp, name="argx0", **kwargs, output_type=tf.int32)

        compare_tf_with_tvm(data, 'c0:0', 'argx0:0')

def test_forward_argminmax():
    for axis in [None,0,1,2]:
        data = np.random.uniform(size=(8,4,9)).astype('float32')
        _test_argx(tf.argmax, data=data, axis=axis)
        _test_argx(tf.argmin, data=data, axis=axis)

#######################################################################
# Reduce
# ------

def _test_reduce(func, data, **kwargs):
    """ One iteration of a reduce operation"""

    with tf.Graph().as_default():
        inp = array_ops.placeholder(shape=data.shape, dtype=data.dtype, name="c0")
        func(inp, name="reducex0", **kwargs)

        compare_tf_with_tvm(data, 'c0:0', 'reducex0:0')

def test_forward_reduce():
    data = np.random.uniform(size=(8,4,9)).astype('float32')
    _test_reduce(tf.reduce_sum, data=data)
    _test_reduce(tf.reduce_sum, data=data, axis=0)
    _test_reduce(tf.reduce_sum, data=data, axis=(0,1))


#######################################################################
# Variable
# --------

def _test_variable(data):
    """ One iteration of a variable """

    tf.reset_default_graph()
    input_op = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
    input_tensor = array_ops.reshape(input_op, data.shape)

    size = input_tensor.shape.dims[1]
    with variable_scope.variable_scope("linear", reuse=None):
        w = variable_scope.get_variable(
            "w", shape=[size, size], dtype=input_tensor.dtype)
    math_ops.matmul(input_tensor, w)

    compare_tf_with_tvm(data, 'Placeholder:0', 'MatMul:0', init_global_variables=True)

def test_forward_variable():
    """Variable type op test"""
    _test_variable(np.random.uniform(size=(32, 100)).astype('float32'))


#######################################################################
# StridedSlice
# ------------

def _test_stridedslice(ip_shape, begin, end, stride, dtype,
                             begin_mask=0, end_mask=0, new_axis_mask=0,
                             shrink_axis_mask=0, ellipsis_mask=0):
    """ One iteration of a Stridedslice """

    tf.reset_default_graph()
    in_data = tf.placeholder(dtype, ip_shape, name="in_data")
    tf.strided_slice(in_data, begin, end, stride, begin_mask=begin_mask,
                         end_mask=end_mask, new_axis_mask=new_axis_mask,
                         shrink_axis_mask=shrink_axis_mask,
                         ellipsis_mask=ellipsis_mask, name="strided_slice")
    np_data = np.random.uniform(size=ip_shape).astype(dtype)

    compare_tf_with_tvm(np_data, 'in_data:0', 'strided_slice:0')

def test_forward_stridedslice():
    '''test StridedSlice'''

    _test_stridedslice((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1], 'float32')
    _test_stridedslice((3, 4, 3), [1, 0], [4, 3], [2, 1], 'float32', ellipsis_mask=8)
    _test_stridedslice((3, 4, 3), [1, 0], [4, 2], [2, 1], 'float32', ellipsis_mask=2)
    _test_stridedslice((3, 4, 5, 3), [1, 0], [4, 2], [2, 1], 'float32', ellipsis_mask=2)
    _test_stridedslice((3, 4, 5, 3), [1, 0, 1], [4, 2, 2], [2, 1, 1], 'float32', ellipsis_mask=2)
    _test_stridedslice((3, 4, 3), [1, 1, 0], [4, 4, 2], [2, 1, 1], 'float32', new_axis_mask=5)
    _test_stridedslice((3, 4, 3), [1, 1, 1], [4, 4, 1], [2, 1, 1], 'float32', ellipsis_mask=2, new_axis_mask=4)
    _test_stridedslice((6, 4, 5), [1, 1, 1], [6, 3, 4], [2, 1, 1], 'float32', ellipsis_mask=2, new_axis_mask=5)
    _test_stridedslice((3, 4, 3), [1, 1, 2], [4, 4, 3], [2, 1, 1], 'float32', ellipsis_mask=4, new_axis_mask=2)
    _test_stridedslice((3, 4, 3), [1, 1, 2], [4, 4, 3], [2, 1, 1], 'float32', ellipsis_mask=2, new_axis_mask=3)
    _test_stridedslice((3, 4, 3), [1, 1, 0], [4, 4, 1], [2, 1, 1], 'float32', ellipsis_mask=2, new_axis_mask=3)
    _test_stridedslice((3, 4, 3), [1, 1, 2], [4, 4, 3], [2, 1, 1], 'float32', ellipsis_mask=2, new_axis_mask=2)
    _test_stridedslice((3,4), [1, 0], [4, 4], [1, 1], 'float32', shrink_axis_mask=2)
    _test_stridedslice((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], 'float32', shrink_axis_mask=2, new_axis_mask=2)
    _test_stridedslice((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], 'float32', shrink_axis_mask=1, new_axis_mask=2)
    _test_stridedslice((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], 'float32', shrink_axis_mask=2, new_axis_mask=1)
    _test_stridedslice((3, 4, 5, 4, 5, 6), [0, 0], [2, 3], [1, 1], 'float32', shrink_axis_mask=5, new_axis_mask=1)
    _test_stridedslice((3, 4, 5, 4, 5, 6), [0, 0, 1, 2, 1], [2, 3, 4, 5, 3], [1, 1, 2, 2, 1],
                       'float32', shrink_axis_mask=5, new_axis_mask=1, ellipsis_mask=2, begin_mask=8, end_mask=8)
    _test_stridedslice((3, 4, 5, 4, 5, 6), [0, 0, 1, 2, 1], [2, 3, 4, 5, 3], [1, 1, 2, 2, 1],
                       'float32', shrink_axis_mask=8, new_axis_mask=1, ellipsis_mask=2, begin_mask=5, end_mask=5)
    _test_stridedslice((3, 4, 5, 4, 5, 6), [0, 0, 1, 2, 1], [2, 3, 4, 5, 3], [1, 1, 2, 2, 1],
                       'float32', shrink_axis_mask=16, new_axis_mask=1, ellipsis_mask=2, begin_mask=5, end_mask=5)
    _test_stridedslice((3, 4, 5, 4, 5, 6), [1, 2, 0, -3], [4, 5, 3, 3], [2, 2, 1, 1],
                       'float32', shrink_axis_mask=8, new_axis_mask=1, ellipsis_mask=2, begin_mask=5,
                       end_mask=8)
    _test_stridedslice((1), [0], [1], [1], 'float32', shrink_axis_mask=1)


#######################################################################
# Gather
# ------

def _test_gather(ip_shape, indice_shape, indice_value, axis, dtype):
    """ One iteration of a Gather """

    tf.reset_default_graph()
    in_data = tf.placeholder(dtype, ip_shape, name="in_data")
    indices = tf.placeholder("int32", indice_shape, name="indices")
    tf.gather(in_data, indices, axis=axis)
    np_data = np.random.uniform(size=ip_shape).astype(dtype)

    def _fill_indices(indice_value):
        indices = np.array(ip_shape, dtype=dtype)
        if isinstance(indice_value, int):
            indices = np.array([indice_value], dtype='int32')
        else:
            indices = np.asarray(indice_value, dtype='int32')
        return indices
    np_indices = _fill_indices(indice_value)

    compare_tf_with_tvm([np_data, np_indices], ['in_data:0', 'indices:0'], 'GatherV2:0')

def test_forward_gather():
    '''test gather layer'''
    _test_gather((4,), (1,), 1, 0, 'int32')
    _test_gather((4,), (1,), 1, 0, 'float32')
    _test_gather((1,4), (1,), [0], 0, 'int32')
    _test_gather((4,), (1,2,2), [[[1,0],[0,1]]], 0, 'float32')
    _test_gather((2,2), (1,2,2), [[[1,0],[0,1]]], 0, 'int32')
    _test_gather((2,2), (1,2,2), [[[1,0],[0,1]]], 1, 'int32')
    _test_gather((2,2), (1,2,2), [[[1,0],[0,1]]], 0, 'float32')
    _test_gather((3,3,3), (1,1,2), [[[1,0]]], 0, 'int32')
    _test_gather((3,3,3), (1,1,2), [[[1,0]]], 2, 'int32')
    _test_gather((4,3,5,6), (1,4), [[2,1,0,0]], 0, 'float32')


#######################################################################
# Split
# -----

def _test_split(in_shape, axis, num_or_size_splits, dtype):
    np_data = np.random.uniform(-5, 5, size=in_shape).astype(dtype)

    """ One iteration of a Split """
    tf.reset_default_graph()
    in_data = tf.placeholder(dtype, in_shape, name="in_data")
    num_split = len(num_or_size_splits) if isinstance(num_or_size_splits, list) else num_or_size_splits
    tf.split(in_data, num_or_size_splits, axis=axis)

    compare_tf_with_tvm([np_data], ['in_data:0'], [f'split:{n}' for n in range(num_split)])

    # and now test together with concat
    tf.reset_default_graph()
    in_data = tf.placeholder(dtype, in_shape, name="in_data")
    splitted = tf.split(in_data, num_or_size_splits, axis=axis)
    tf.concat(splitted, axis)

    compare_tf_with_tvm([np_data], 'in_data:0', 'concat:0')

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
    # size_splits list
    _test_split((6,), 0, [1, 2, 3], 'int32')
    _test_split((3, 6, 4), -2, [1, 4, 1], 'float32')


#######################################################################
# Unstack
# -------

def _test_unstack(ip_shape, axis, dtype):
    np_data = np.random.uniform(-5, 5, size=ip_shape).astype(dtype)

    tf.reset_default_graph()
    in_data = tf.placeholder(dtype, ip_shape, name="in_data")
    tf.unstack(in_data, axis=axis)

    compare_tf_with_tvm([np_data], ['in_data:0'], [f'unstack:{n}' for n in range(ip_shape[axis])])

    tf.reset_default_graph()
    in_data = tf.placeholder(dtype, ip_shape, name="in_data")
    tf.stack(tf.unstack(in_data, axis=axis), axis=axis)

    compare_tf_with_tvm([np_data], ['in_data:0'], 'stack:0')

def test_forward_unstack():
    '''test unstack layer'''
    _test_unstack((6,), 0, 'int32')
    _test_unstack((2,6), 1, 'float64')
    # negative axis
    _test_unstack((1,4), -1, 'int32')
    _test_unstack((3,6,4), -2, 'float32')


#######################################################################
# Multi Input to graph
# --------------------

def test_forward_multi_input():
    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.int32, shape=[3, 3], name='in1')
        in2 = tf.placeholder(tf.int32, shape=[3, 3], name='in2')
        in3 = tf.placeholder(tf.int32, shape=[3, 3], name='in3')
        in4 = tf.placeholder(tf.int32, shape=[3, 3], name='in4')

        out1 = tf.add(in1, in2, name='out1')
        out2 = tf.subtract(in3, in4, name='out2')
        out = tf.multiply(out1, out2, name='out')
        in_data = np.arange(9, dtype='int32').reshape([3, 3])

        compare_tf_with_tvm([in_data, in_data, in_data, in_data],
                            ['in1:0', 'in2:0', 'in3:0', 'in4:0'], 'out:0')

#######################################################################
# Multi Output to Graph
# ---------------------

def test_forward_multi_output():
    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.int32, shape=[3, 3], name='in1')
        in2 = tf.placeholder(tf.int32, shape=[3, 3], name='in2')
        in3 = tf.placeholder(tf.int32, shape=[3, 3], name='in3')
        in4 = tf.placeholder(tf.int32, shape=[3, 3], name='in4')

        out1 = tf.add(in1, in2, name='out1')
        out2 = tf.subtract(in3, in4, name='out2')
        in_data = np.arange(9, dtype='int32').reshape([3, 3])
        in_data = [in_data] * 4
        in_name = ['in1:0', 'in2:0', 'in3:0', 'in4:0']
        out_name = ['out1:0', 'out2:0']
        out_node = [out.strip(':0') for out in out_name]
        in_node = [inp.strip(':0') for inp in in_name]

        with tf.Session() as sess:
            final_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph.as_graph_def(add_shapes=True), out_node,)
            tf_output = run_tf_graph(sess, in_data, in_name, out_name)
            tvm_output = run_tvm_graph(final_graph_def, in_data, in_node, target='llvm',
                                       out_names=out_node, num_output=2)
            for i in range(len(tf_output)):
                tvm.testing.assert_allclose(tf_output[i], tvm_output[i], atol=1e-5, rtol=1e-5)

#######################################################################
# Resize Bilinear
# ---------------

def _test_resize_bilinear(in_shape, to_shape, align_corners):
    """ One iteration of resize bilinear """

    data = np.random.uniform(size=in_shape).astype('float32')
    shape_data = np.array(to_shape).astype('int32')

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        shape_data = constant_op.constant(shape_data, shape=shape_data.shape, dtype=shape_data.dtype)
        tf.image.resize_bilinear(in_data, shape_data, align_corners=align_corners)

        compare_tf_with_tvm(data, 'Placeholder:0', 'ResizeBilinear:0')

def test_forward_resize_bilinear():
    """ Resize Bilinear """

    _test_resize_bilinear((4, 16, 32, 32), [50, 50], False)
    _test_resize_bilinear((6, 32, 64, 64), [20, 20], True)


#######################################################################
# Crop to bounding box
# --------------------

def _test_crop(in_shape, off_h, off_w, tar_h, tar_w):
    """ Crop to bounding box """
    data = np.random.uniform(size=in_shape).astype('float32')
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        tf.image.crop_to_bounding_box(in_data, off_h, off_w, tar_h, tar_w)
        compare_tf_with_tvm(data, 'Placeholder:0', 'crop_to_bounding_box/Slice:0')

def test_forward_crop():
    """ Crop to bounding box """
    _test_crop((1, 224, 224, 3), 20, 20, 120, 120)


#######################################################################
# LSTM
# ----

def _test_lstm_cell(batch_size, num_hidden, num_layers, forget_bias, dtype):
    """ One iteration of a LSTM cell """

    tf.reset_default_graph()
    input_size = num_hidden
    input_data = np.full((batch_size, input_size), 1., dtype=dtype)
    in_state_c = np.full((num_layers, batch_size, num_hidden), 0.1, dtype=dtype)
    in_state_h = np.full((num_layers, batch_size, num_hidden), 0.1, dtype=dtype)

    def _get_tensorflow_output():
        with tf.Session() as sess:
            with variable_scope.variable_scope(
                "root", initializer=init_ops.constant_initializer(0.5)):
                m0 = array_ops.zeros([batch_size, num_hidden])
                m1 = array_ops.zeros([batch_size, num_hidden])
                x=tf.placeholder(shape=(batch_size, input_size), dtype=dtype)
                g, ((out_m0, out_m1)) = \
                     tf.contrib.rnn.LSTMBlockCell(num_hidden,
                                                  forget_bias=forget_bias)(x, ((m0, m1)))
                sess.run([variables.global_variables_initializer()])
                res = sess.run([g, out_m0, out_m1], {
                    x.name: np.array([[1., 1.]]),
                    m0.name: 0.1 * np.ones([batch_size, num_hidden]),
                    m1.name: 0.1 * np.ones([batch_size, num_hidden]),
                })
            graph_def = sess.graph.as_graph_def(add_shapes=True)
            final_graph_def = graph_util.convert_variables_to_constants(
                sess,
                graph_def,
                ['root/lstm_cell/LSTMBlockCell'])
            return final_graph_def, res

    graph_def, tf_out = _get_tensorflow_output()
    tvm_output = run_tvm_graph(graph_def, [input_data, in_state_c, in_state_h],
                               ['root/Placeholder', 'root/lstm_cell/LSTMBlockCell_c',
                                'root/lstm_cell/LSTMBlockCell_h'], num_output=2)
    assert isinstance(tvm_output, list)

    out = tvm_output[0]
    out_state = tvm_output[1]
    out_state_tup = np.split(out_state, indices_or_sections=2, axis=1)
    out_state_c = np.reshape(out_state_tup[0], (batch_size, num_hidden))
    out_state_h = np.reshape(out_state_tup[1], (batch_size, num_hidden))
    tvm_out = [out, out_state_c, out_state_h]
    tvm.testing.assert_allclose(tf_out[0], tvm_out[0], rtol=1e-3, atol=1e-3)

def test_forward_lstm():
    '''test LSTM block cell'''
    _test_lstm_cell(1, 2, 1, 0.0, 'float32')



#######################################################################
# Pack
# ---
def _test_pack(axis, shape, **kwargs):

    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    b = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    with tf.Graph().as_default():
        tf_a = array_ops.placeholder(shape=shape, dtype='float32', name='pl_a')
        tf_b = array_ops.placeholder(shape=shape, dtype='float32', name='pl_b')
        tf_c = tf.stack([tf_a,tf_b], axis=axis, **kwargs)
        assert tf_c.op.op_def.name == 'Pack', "tf.stack() is expected to produce 'Pack' operation"

        compare_tf_with_tvm([a,b], ['pl_a:0','pl_b:0'], 'stack:0')

def test_forward_pack():
    for axis in range(-3,3):
        _test_pack(axis, [3,2,1])
    for axis in range(-1,1):
        _test_pack(axis, [3])
    _test_pack(0, [])

#######################################################################
# Pad
# ---
def _test_pad(input_shape, paddings, mode, **kwargs):
    """ One iteration of pad operation with given shape"""

    x = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=input_shape, dtype='float32')
        pad_values = constant_op.constant(paddings)
        pad = tf.pad(in_data, paddings=pad_values, mode=mode, **kwargs)

        if mode == 'CONSTANT':
            if 'constant_values' in kwargs:
                out_name = 'PadV2:0'
            else:
                out_name = 'Pad:0'

        compare_tf_with_tvm(x, 'Placeholder:0', out_name)

def test_forward_pad():
    """ Pad """
    _test_pad((2, 3), [[1,1], [2,2]], mode="CONSTANT")
    _test_pad((2, 3), [[1,1], [2,2]], mode="CONSTANT", constant_values=1.0)

#######################################################################
# Logical operators
# --------------------
def test_logical_and():
    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name='in1')
        in2 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name='in2')
        out = tf.logical_and(in1, in2, name='out')
        in_data1 = np.random.choice(a=[False, True],size=(1, 4, 4, 3)).astype('bool')
        in_data2 = np.random.choice(a=[False, True],size=(1, 4, 4, 3)).astype('bool')
        compare_tf_with_tvm([in_data1, in_data2], ['in1:0', 'in2:0'], 'out:0')

def test_logical_or():
    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name='in1')
        in2 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name='in2')
        out = tf.logical_or(in1, in2, name='out')
        in_data1 = np.random.choice(a=[False, True],size=(1, 4, 4, 3)).astype('bool')
        in_data2 = np.random.choice(a=[False, True],size=(1, 4, 4, 3)).astype('bool')
        compare_tf_with_tvm([in_data1, in_data2], ['in1:0', 'in2:0'], 'out:0')

def test_logical_xor():
    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name='in1')
        in2 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name='in2')
        out = tf.logical_xor(in1, in2, name='out')
        in_data1 = np.random.choice(a=[False, True],size=(1, 4, 4, 3)).astype('bool')
        in_data2 = np.random.choice(a=[False, True],size=(1, 4, 4, 3)).astype('bool')
        compare_tf_with_tvm([in_data1, in_data2], ['in1:0', 'in2:0'], 'out:0')

def test_logical_not():
    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name='in1')
        out = tf.logical_not(in1, name='out')
        in_data1 = np.random.choice(a=[False, True],size=(1, 4, 4, 3)).astype('bool')
        compare_tf_with_tvm(in_data1, 'in1:0', 'out:0')

def test_forward_logical():
    test_logical_and()
    test_logical_or()
    test_logical_xor()
    test_logical_not()

#######################################################################
# Inception V3
# ------------
def test_forward_inception_v3():
    '''test inception V3 model'''
    with tf.Graph().as_default():
        graph_def = tf_testing.get_workload('InceptionV3/inception_v3_2016_08_28_frozen-with_shapes.pb')
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        data = np.random.uniform(size=(1, 299, 299, 3)).astype('float32')

        with tf.Session() as sess:
            tf_output = run_tf_graph(sess, data, 'input:0', 'InceptionV3/Predictions/Reshape_1:0')
            tvm_output = run_tvm_graph(graph_def, data, 'input')
            tvm.testing.assert_allclose(tf_output[0], tvm_output[0], rtol=1e-5, atol=1e-5)

#######################################################################
# Inception V1
# ------------
def test_forward_inception_v1():
    '''test inception V1 model'''
    with tf.Graph().as_default():
        graph_def = tf_testing.get_workload("InceptionV1/classify_image_graph_def-with_shapes.pb")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        # Build an image from random data.
        from PIL import Image
        from tvm.contrib import util

        img_array = np.random.uniform(size=(1, 600, 600, 3)).astype("uint8")
        img = Image.frombuffer('RGB', (600, 600), img_array.tostring(), 'raw', 'RGB', 0, 1)
        temp = util.tempdir()
        img_path = temp.relpath("tf-test.jpg")
        img.save(img_path);

        import os.path
        if not tf.gfile.Exists(os.path.join(img_path)):
            tf.logging.fatal('File does not exist %s', image)
        data = tf.gfile.FastGFile(os.path.join(img_path), 'rb').read()

        temp.remove()

        # Extract tensorflow decoded image frame for tvm input
        with tf.Session() as sess:
            tvm_data = run_tf_graph(sess, data, 'DecodeJpeg/contents:0', 'DecodeJpeg:0')

        with tf.Session() as sess:
            tf_output = run_tf_graph(sess, data, 'DecodeJpeg/contents:0', 'softmax:0')
            tvm_output = run_tvm_graph(graph_def, tvm_data, 'DecodeJpeg/contents')
            tvm.testing.assert_allclose(tf_output[0], tvm_output[0], rtol=1e-5, atol=1e-5)

#######################################################################
# Mobilenet
# ---------
def test_forward_mobilenet():
    '''test mobilenet model'''
    # MobilenetV2
    with tf.Graph().as_default():
        graph_def = tf_testing.get_workload(
            "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz",
            "mobilenet_v2_1.4_224_frozen.pb")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        data = np.random.uniform(size=(1, 224, 224, 3)).astype('float32')
        out_node = 'MobilenetV2/Predictions/Reshape_1'

        with tf.Session() as sess:
            # Add shapes to the graph.
            graph_def = tf_testing.AddShapesToGraphDef(sess, out_node)
            tf_output = run_tf_graph(sess, data, 'input:0', out_node + ':0')
            tvm_output = run_tvm_graph(graph_def, data, 'input')
            tvm.testing.assert_allclose(np.squeeze(tvm_output[0]), np.squeeze(tf_output[0]), rtol=1e-5, atol=1e-5)

#######################################################################
# ResnetV2
# --------
def test_forward_resnetv2():
    '''test resnet model'''
    if is_gpu_available():
        with tf.Graph().as_default():
            graph_def = tf_testing.get_workload("ResnetV2/resnet-20180601_resnet_v2_imagenet-shapes.pb")
            # Call the utility to import the graph definition into default graph.
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)

            data = np.random.uniform(size=(128, 224, 224, 3)).astype('float32')
            out_node = 'ArgMax'

            with tf.Session() as sess:
                tf_output = run_tf_graph(sess, data, 'input_tensor:0', out_node + ':0')
                for device in ["llvm", "cuda"]:
                    ctx = tvm.context(device, 0)
                    if not ctx.exist:
                        print("Skip because %s is not enabled" % device)
                        continue
                    tvm_output = run_tvm_graph(graph_def, data, 'input_tensor', len(tf_output), target=device)
                    tvm.testing.assert_allclose(np.squeeze(tvm_output[0]), np.squeeze(tf_output[0]), rtol=1e-5, atol=1e-5)

#######################################################################
# Placeholder
# -----------
def test_forward_placeholder():
    '''test a simple pb with Placeholder node in the end of GraphDef'''
    with tf.Graph().as_default():
        graph_def = tf_testing.get_workload("Custom/placeholder.pb")

        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)


        data = np.random.uniform(size=(1, 224, 224, 3)).astype('float32')
        out_node = 'mul'

        with tf.Session() as sess:
            # Add shapes to the graph.
            graph_def = tf_testing.AddShapesToGraphDef(sess, out_node)
            tf_output = run_tf_graph(sess, data, 'Placeholder:0', out_node + ':0')
            tvm_output = run_tvm_graph(graph_def, data, 'Placeholder')
            print("tf_output is {}\ntvm_output is {}".format(tf_output, tvm_output))
            tvm.testing.assert_allclose(np.squeeze(tvm_output[0]), np.squeeze(tf_output[0]), rtol=1e-5, atol=1e-5)

#######################################################################
# PTB
# ---
dir(tf.contrib)
def test_forward_ptb():
    '''test ptb model'''
    config = tf_testing.get_config()
    num_steps = config.num_steps
    num_hidden = config.hidden_size
    num_layers = config.num_layers
    batch_size = config.batch_size
    vocab_size = config.vocab_size
    out_sample_shape = (batch_size, vocab_size)
    out_state_shape = (num_layers, 2, batch_size, num_hidden)
    #Sample input
    inpt = "we have no useful information on"
    cnt_sample = 20

    def _pretty_print(items, is_char_model, id2word):
        if not is_char_model:
            return ' '.join([id2word[x] for x in items])
        else:
            return ''.join([id2word[x] for x in items]).replace('_', ' ')

    def _get_tvm_graph_module(graph_def):
        sym, params = nnvm.frontend.from_tensorflow(graph_def)

        #Cell inputs 'c and 'h' consist of all layers values
        shape_dict = {'Model/Placeholder': (batch_size, num_steps),
                      'Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_c':(num_layers, batch_size, num_hidden),
                      'Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_h':(num_layers, batch_size, num_hidden)}
        dtype_dict = {'Model/Placeholder': 'int32',
                      'Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_c':'float32',
                      'Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_h':'float32'}
        target = 'llvm'
        graph, lib, params = nnvm.compiler.build(sym, target, shape_dict,
                                                 dtype=dtype_dict, params=params)
        from tvm.contrib import graph_runtime
        ctx = tvm.cpu(0)
        return params, graph_runtime.create(graph, lib, ctx)

    def _do_tvm_sample(model, data, in_states, params, num_samples):
        """Sampled from the model"""
        samples = []
        state = in_states
        sample = None
        def _get_sample(data, state):
            input_data = np.full((batch_size, num_steps), data, dtype="int32")
            in_state_tup = np.split(state, indices_or_sections=2, axis=1)
            in_state_c = np.reshape(in_state_tup[0], (num_layers, batch_size, num_hidden))
            in_state_h = np.reshape(in_state_tup[1], (num_layers, batch_size, num_hidden))

            model.set_input('Model/Placeholder', tvm.nd.array(input_data.astype("int32")))
            model.set_input('Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_c',
                        tvm.nd.array(in_state_c.astype("float32")))
            model.set_input('Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_h',
                        tvm.nd.array(in_state_h.astype("float32")))
            model.set_input(**params)
            model.run()
            tvm_output = model.get_output(0, tvm.nd.empty(out_sample_shape,
                                                      "float32")).asnumpy()
            state_output = model.get_output(1, tvm.nd.empty(out_state_shape,
                                                        "float32")).asnumpy()
            sample = tf_testing.pick_from_weight(tvm_output[0])

            return sample, state_output

        for x in data:
            sample, state = _get_sample(x, state)

        if sample is not None:
            samples.append(sample)
        else:
            samples.append(0)

        k = 1
        while k < num_samples:
            sample, state = _get_sample(samples[-1], state)
            samples.append(sample)
            k += 1
        return samples, state

    with tf.Graph().as_default():
        word_to_id, id_to_word, graph_def = tf_testing.get_workload_ptb()
        vocab_size = len(word_to_id)
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        sess = tf.Session()

    #TVM graph module creation
    params, m = _get_tvm_graph_module(graph_def)

    # Create 10 predicted statments of 20 words
    cnt_stm = 0
    while cnt_stm < 10:
        cnt_stm += 1
        in_state = np.full((num_layers, 2, batch_size, num_hidden), 0, dtype="float32")
        seed_for_sample = inpt.split()
        tvm_samples, tvm_state = _do_tvm_sample(m, [word_to_id[word] \
                                                    for word in seed_for_sample],
                                                in_state, params, cnt_sample)
        tvm_sample_str = _pretty_print(tvm_samples, False, id_to_word)
        tf_samples, tf_state = tf_testing.do_tf_sample(sess,
                                [word_to_id[word] for word in seed_for_sample],
                                in_state, cnt_sample)
        tf_sample_str = _pretty_print(tf_samples, False, id_to_word)
        inpt = tvm_sample_str
        tvm.testing.assert_allclose(tf_samples, tvm_samples, rtol=1e-5, atol=1e-5)
        assert(tvm_sample_str == tf_sample_str)

#######################################################################
# LRN (Local Response Normalization)
# ----------------------------------

def _test_lrn(ishape, size, axis, bias, alpha, beta):
    """ testing local response normalization """
    lrn_depth_radius = size / 2

    inp_array = np.random.uniform(size=ishape).astype(np.float32)

    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype, name="lrn0_data")
        nn_ops.local_response_normalization(in1,
                                            name="lrn",
                                            depth_radius=lrn_depth_radius,
                                            bias=bias,
                                            alpha=alpha,
                                            beta=beta)

        compare_tf_with_tvm(inp_array, 'lrn0_data:0', 'lrn:0')

def test_forward_lrn():
    _test_lrn((1, 3, 20, 20), 3, 1, 1.0, 1.0, 0.5)

#######################################################################
# l2_normalize
# ------------

def _test_l2_normalize(ishape, eps, axis):
    """ testing l2 normalize (uses max, sum, square, sqrt frontend operators)"""

    inp_array = np.random.uniform(size=ishape).astype(np.float32)

    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
        nn.l2_normalize(in1,
                        axis=axis,
                        epsilon=eps,
                        name=None,
                        dim=None)

        compare_tf_with_tvm(inp_array, 'Placeholder:0', 'l2_normalize:0')

def test_forward_l2_normalize():
    _test_l2_normalize((1, 3, 20, 20), 0.001, (0,))

#######################################################################
# transpose
# ---------
def _test_forward_transpose(ishape, axes=None):
    input = np.random.uniform(size=ishape).astype(np.float32)

    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=input.shape, dtype=input.dtype, name="transpose_data")

        if axes is None:
            tf.transpose(in1)
        else:
            tf.transpose(in1, perm=axes)

        compare_tf_with_tvm(input, 'transpose_data:0', 'transpose:0')

def test_forward_transpose():
    _test_forward_transpose((2, 3, 4))
    _test_forward_transpose((7, 8, 8, 10))
    _test_forward_transpose((2, 3, 4), (1, 2, 0))
    _test_forward_transpose((2, 3, 4), (0, 1, 2))
    _test_forward_transpose((2, 3, 4, 5), (3, 0, 1, 2))


def test_forward_ceil():
    ishape = (1, 3, 10, 10)
    inp_array = np.random.uniform(size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
        tf.ceil(in1)
        compare_tf_with_tvm(inp_array, 'Placeholder:0', 'Ceil:0')

def test_forward_floor():
    ishape = (1, 3, 10, 10)
    inp_array = np.random.uniform(size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
        tf.floor(in1)
        compare_tf_with_tvm(inp_array, 'Placeholder:0', 'Floor:0')

def test_forward_relu():
    ishape = (1, 3, 10, 10)
    inp_array = np.random.uniform(-5, 5, size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
        tf.nn.relu(in1)
        compare_tf_with_tvm(inp_array, 'Placeholder:0', 'Relu:0')

def test_forward_leaky_relu():
    ishape = (1, 3, 10, 10)
    inp_array = np.random.uniform(-5, 5, size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
        tf.nn.leaky_relu(in1, alpha=0.4)
        compare_tf_with_tvm(inp_array, 'Placeholder:0', 'LeakyRelu:0')

def test_forward_elu():
    ishape = (1, 3, 10, 10)
    inp_array = np.random.uniform(-5, 5, size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
        tf.nn.elu(in1)
        compare_tf_with_tvm(inp_array, 'Placeholder:0', 'Elu:0')

def test_forward_selu():
    ishape = (1, 3, 10, 10)
    inp_array = np.random.uniform(-5, 5, size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
        tf.nn.selu(in1)
        compare_tf_with_tvm(inp_array, 'Placeholder:0', 'Selu:0')

def test_forward_tanh():
    ishape = (1, 3, 10, 10)
    inp_array = np.random.uniform(-5, 5, size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
        tf.nn.tanh(in1)
        compare_tf_with_tvm(inp_array, 'Placeholder:0', 'Tanh:0')

#######################################################################
# Mean
# ----
def test_forward_mean():
    def check_mean(ishape, **kwargs):
        inp_array = np.random.uniform(size=ishape).astype(np.float32)
        with tf.Graph().as_default():
            in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
            tf.keras.backend.mean(in1, **kwargs)
            compare_tf_with_tvm(inp_array, 'Placeholder:0', 'Mean:0', no_gpu=True)

    check_mean((10, 8, 16, 32))
    check_mean((10, 8, 16, 32), axis=(2,3))
    check_mean((10, 8, 16, 32), axis=(1,2), keepdims=True)

#######################################################################
# Relational operators
# --------------------
def _test_forward_rel_op(data, func):
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=data[0].shape, dtype=data[0].dtype, name='in1')
        in2 = tf.placeholder(shape=data[1].shape, dtype=data[1].dtype, name='in2')
        op = func(in1, in2, name='op')
        out = tf.cast(op, tf.int32, name='out1')
        compare_tf_with_tvm([data[0], data[1]], ['in1:0', 'in2:0'], 'out1:0')

def test_forward_rel_ops():
    t1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    t2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    _test_forward_rel_op([t1, t2], math_ops.less)
    _test_forward_rel_op([t1, t2], math_ops.greater)
    _test_forward_rel_op([t1, t2], math_ops.less_equal)
    _test_forward_rel_op([t1, t2], math_ops.greater_equal)
    _test_forward_rel_op([t1, t2], math_ops.equal)
    _test_forward_rel_op([t1, t2], math_ops.not_equal)


#######################################################################
# Main
# ----
if __name__ == '__main__':
    # Transforms
    test_forward_transpose()
    test_forward_reshape()
    test_forward_squeeze()
    test_forward_pack()
    test_forward_resize_bilinear()
    test_forward_crop()
    test_forward_pad()
    test_forward_gather()
    test_forward_stridedslice()
    test_forward_split()
    test_forward_unstack()

    # Activations
    test_forward_sigmoid()
    test_forward_relu()
    test_forward_leaky_relu()
    test_forward_elu()
    test_forward_selu()
    test_forward_tanh()

    # Reductions
    test_forward_argminmax()
    test_forward_reduce()
    test_forward_mean()

    # NN
    test_forward_convolution()
    test_forward_pooling()
    if tf.__version__ == '1.4.1':
        _test_forward_concat_v2()
    test_forward_lrn()
    test_forward_l2_normalize()

    # General
    test_forward_multi_input()
    test_forward_multi_output()
    test_forward_variable()

    # End to End
    test_forward_inception_v3()
    test_forward_inception_v1()
    test_forward_mobilenet()
    test_forward_resnetv2()
    test_forward_placeholder()
    test_forward_ptb()

    # RNN
    test_forward_lstm()

    # Elementwise
    test_forward_ceil()
    test_forward_floor()

    # Relational ops
    test_forward_rel_ops()
    test_forward_logical()
