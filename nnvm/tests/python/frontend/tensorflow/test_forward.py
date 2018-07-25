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
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import init_ops
from tensorflow.core.framework import graph_pb2

import nnvm.testing.tf

#######################################################################
# Generic run functions for TVM & tensorflow
# ------------------------------------------
def run_tvm_graph(graph_def, input_data, input_node, output_shape, output_dtype):
    """ Generic function to compile on nnvm and execute on tvm """

    sym, params = nnvm.frontend.from_tensorflow(graph_def)
    target = 'llvm'
    if isinstance(input_data, list):
        shape_dict = {}
        dtype_dict = {}
        for i, e in enumerate(input_node):
            shape_dict[e] = input_data[i].shape
            dtype_dict[e] = input_data[i].dtype
    else:
        shape_dict = {input_node: input_data.shape}
        dtype_dict = {input_node: input_data.dtype}

    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict,
                                             dtype=dtype_dict, params=params)

    ctx = tvm.cpu(0)
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    if isinstance(input_data, list):
        for i, e in enumerate(input_node):
            m.set_input(e, tvm.nd.array(input_data[i].astype(input_data[i].dtype)))
    else:
        m.set_input(input_node, tvm.nd.array(input_data.astype(input_data.dtype)))

    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    if isinstance(output_shape, list) and isinstance(output_dtype, list):
        tvm_output_list = []
        for i, s in enumerate(output_shape):
            tvm_output = m.get_output(i, tvm.nd.empty((s), output_dtype[i]))
            tvm_output_list.append(tvm_output.asnumpy())
        return tvm_output_list
    else:
        tvm_output = m.get_output(0, tvm.nd.empty((output_shape), output_dtype))
        return tvm_output.asnumpy()

def run_tf_graph(sess, input_data, input_node, output_node):
    """ Generic function to execute tensorflow """

    tensor = sess.graph.get_tensor_by_name(output_node)

    if isinstance(input_data, list):
        input_dict = {}
        for i, e in enumerate(input_node):
            input_dict[e] = input_data[i]
    else:
        input_dict = {input_node: input_data}

    output_data = sess.run(tensor, input_dict)
    return output_data

#######################################################################
# Pooling
# -------
def _test_pooling(input_shape, **kwargs):
    """ One iteration of pool operation with given shapes and attributes """

    x = -np.arange(
        np.prod(input_shape), dtype=np.float32).reshape(input_shape) - 1

    with tf.Graph().as_default():
        in_data = constant_op.constant(x, shape=input_shape, dtype='float32')
        # pylint: disable=unused-variable
        pool = nn_ops.pool(in_data, **kwargs)
        # pylint: enable=unused-variable

        if kwargs['pooling_type'] == 'MAX':
            out_node = 'max_pool'
            out_name = 'max_pool:0'
        else:
            out_node = 'avg_pool'
            out_name = 'avg_pool:0'

        with tf.Session() as sess:
            graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(add_shapes=True),
                [out_node],
                )

            tf_output = run_tf_graph(sess, x, 'Const:0', out_name)
            tvm_output = run_tvm_graph(graph_def, x.astype('float32'),
                                       "Const", tf_output.shape, 'float32')
            np.testing.assert_allclose(tf_output, tvm_output, atol=1e-3, rtol=1e-3)

            sess.close()

def test_forward_pooling():
    """ Pooling """

    _test_pooling(input_shape=[2, 9, 10, 2],
                 window_shape=[1, 1],
                 padding='SAME',
                 pooling_type='MAX',
                 dilation_rate=[1, 1],
                 strides=[1, 1])
    _test_pooling(input_shape=[2, 9, 10, 2],
                 window_shape=[1, 1],
                 padding='SAME',
                 pooling_type='AVG',
                 dilation_rate=[1, 1],
                 strides=[1, 1])

    _test_pooling(input_shape=[2, 10, 9, 2],
                 window_shape=[1, 1],
                 padding='SAME',
                 pooling_type='MAX',
                 dilation_rate=[1, 1],
                 strides=[1, 1])
    _test_pooling(input_shape=[2, 10, 9, 2],
                 window_shape=[1, 1],
                 padding='SAME',
                 pooling_type='AVG',
                 dilation_rate=[1, 1],
                 strides=[1, 1])

    _test_pooling(input_shape=[2, 9, 10, 2],
                 window_shape=[2, 1],
                 padding='SAME',
                 pooling_type='MAX',
                 dilation_rate=[1, 1],
                 strides=[1, 1])
    _test_pooling(input_shape=[2, 9, 10, 2],
                 window_shape=[2, 1],
                 padding='SAME',
                 pooling_type='AVG',
                 dilation_rate=[1, 1],
                 strides=[2, 1])

    _test_pooling(input_shape=[2, 10, 9, 2],
                 window_shape=[2, 3],
                 padding='SAME',
                 pooling_type='MAX',
                 dilation_rate=[1, 1],
                 strides=[2, 1])
    _test_pooling(input_shape=[2, 10, 9, 2],
                 window_shape=[2, 3],
                 padding='SAME',
                 pooling_type='AVG',
                 dilation_rate=[1, 1],
                 strides=[1, 2])


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
        in_data = constant_op.constant(data_array, shape=tensor_in_sizes, dtype='float32')
        in_filter = constant_op.constant(filter_array, shape=filter_in_sizes, dtype='float32')
        strides = [1] + strides + [1]
        dilations = [1] + dilations + [1]

        # pylint: disable=unused-variable
        conv = nn_ops.conv2d(in_data,
                             in_filter,
                             strides=strides,
                             padding=padding,
                             data_format=data_format)
        # pylint: enable=unused-variable

        with tf.Session() as sess:
            graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(add_shapes=True),
                ['Conv2D'],
                )

            tf_output = run_tf_graph(sess, np.reshape(data_array, tensor_in_sizes),
                                     'Const:0', 'Conv2D:0')
            tvm_output = run_tvm_graph(graph_def,
                                       np.reshape(data_array, tensor_in_sizes).astype('float32'),
                                       "Const", tf_output.shape, 'float32')

            np.testing.assert_allclose(tf_output, tvm_output, atol=1e-3, rtol=1e-3)

            sess.close()

def test_forward_convolution():
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
        in_data = constant_op.constant(data, shape=data.shape, dtype=data.dtype)

        # pylint: disable=unused-variable
        reshape_out = array_ops.reshape(in_data, out_shape)
        # pylint: enable=unused-variable

        with tf.Session() as sess:
            graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(add_shapes=True),
                ['Reshape'],
                )

            tf_output = run_tf_graph(sess, data,
                                     'Const:0', 'Reshape:0')
            tvm_output = run_tvm_graph(graph_def,
                                       data,
                                       "Const", tf_output.shape, data.dtype)

            np.testing.assert_allclose(tf_output, tvm_output)

            sess.close()

def test_forward_reshape():
    _test_reshape(np.arange(6.0), [2, 3])
    _test_reshape(np.arange(6), [-1, 2])
    _test_reshape(np.arange(6), [3, -1])
    _test_reshape(np.arange(6), [-1])

#######################################################################
# Squeeze
# -------

def _test_squeeze(data, squeeze_dims=None):
    """ One iteration of squeeze """

    if squeeze_dims is None:
        squeeze_dims = []

    with tf.Graph().as_default():
        in_data = constant_op.constant(data, shape=data.shape, dtype=data.dtype)

        # pylint: disable=unused-variable
        if squeeze_dims:
            squeeze_out = array_ops.squeeze(in_data, squeeze_dims)
        else:
            squeeze_out = array_ops.squeeze(in_data)
        # pylint: enable=unused-variable

        with tf.Session() as sess:
            graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(add_shapes=True),
                ['Squeeze'],
                )

            tf_output = run_tf_graph(sess, data,
                                     'Const:0', 'Squeeze:0')
            tvm_output = run_tvm_graph(graph_def,
                                       data,
                                       "Const", tf_output.shape, data.dtype)

            np.testing.assert_allclose(tf_output, tvm_output)

            sess.close()

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

        # pylint: disable=unused-variable
        concat_out = gen_array_ops._concat_v2(data, dim)
        # pylint: enable=unused-variable

        with tf.Session() as sess:
            graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(add_shapes=True),
                ['ConcatV2'],
                )

            tf_output = run_tf_graph(sess, data,
                                     ['ConcatV2/values_0:0', 'ConcatV2/values_1:0'], 'ConcatV2:0')
            tvm_output = run_tvm_graph(graph_def,
                                       data,
                                       ["ConcatV2/values_0", 'ConcatV2/values_1'],
                                       tf_output.shape, tf_output.dtype)

            np.testing.assert_allclose(tf_output, tvm_output)

            sess.close()

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
        in_data = constant_op.constant(data, shape=data.shape, dtype=data.dtype)

        # pylint: disable=unused-variable
        sigmoid_out = math_ops.sigmoid(in_data)
        # pylint: enable=unused-variable

        with tf.Session() as sess:
            graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(add_shapes=True),
                ['Sigmoid'],
                )

            tf_output = run_tf_graph(sess, data,
                                     'Const:0', 'Sigmoid:0')
            tvm_output = run_tvm_graph(graph_def,
                                       data,
                                       "Const", tf_output.shape, data.dtype)

            np.testing.assert_allclose(tf_output, tvm_output, atol=1e-5, rtol=1e-5)

            sess.close()

def test_forward_sigmoid():
    """ Sigmoid """

    _test_sigmoid(np.random.uniform(size=(3, 4, 4, 3)).astype('float32'))


#######################################################################
# Variable
# --------

def _test_variable(data):
    tf.reset_default_graph()
    input_op = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
    input_tensor = array_ops.reshape(input_op, data.shape)

    size = input_tensor.shape.dims[1]
    with variable_scope.variable_scope("linear", reuse=None):
        w = variable_scope.get_variable(
            "w", shape=[size, size], dtype=input_tensor.dtype)
    # pylint: disable=unused-variable
    output_op = math_ops.matmul(input_tensor, w)
    # pylint: enable=unused-variable

    with tf.Session() as sess:
        sess.run(variables.global_variables_initializer())
        final_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(add_shapes=True),
            ['MatMul'],
            )

        tf_output = run_tf_graph(sess, data, 'Placeholder:0', 'MatMul:0')
        tvm_output = run_tvm_graph(final_graph_def, data,
                                   "Placeholder", tf_output.shape, data.dtype)

        np.testing.assert_allclose(tf_output, tvm_output, atol=1e-5, rtol=1e-5)
        sess.close()

def test_forward_variable():
    """Variable type op test"""
    _test_variable(np.random.uniform(size=(32, 100)).astype('float32'))


#######################################################################
# LSTM
# ----
def _test_lstm_cell(batch_size, num_hidden, num_layers, forget_bias, dtype):
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
                                'root/lstm_cell/LSTMBlockCell_h'],
                               [tf_out[0].shape, (2, batch_size, num_hidden)],
                               [tf_out[0].dtype, tf_out[1].dtype])

    if isinstance(tvm_output, list):
        out = tvm_output[0]
        out_state = tvm_output[1]
        out_state_tup = np.split(out_state, indices_or_sections=2, axis=0)
        out_state_c = np.reshape(out_state_tup[0], (batch_size, num_hidden))
        out_state_h = np.reshape(out_state_tup[1], (batch_size, num_hidden))
        tvm_out = [out, out_state_c, out_state_h]
        np.testing.assert_allclose(tf_out, tvm_out, rtol=1e-3, atol=1e-3)

def test_forward_lstm():
    '''test LSTM block cell'''
    _test_lstm_cell(1, 2, 1, 0.0, 'float32')


#######################################################################
# StridedSlice
# ------------

def _test_stridedslice(ip_shape, begin, end, stride, dtype,
                             begin_mask=0, end_mask=0, new_axis_mask=0,
                             shrink_axis_mask=0, ellipsis_mask=0):
    tf.reset_default_graph()
    in_data = tf.placeholder(dtype, ip_shape, name="in_data")
    tf.strided_slice(in_data, begin, end, stride, begin_mask=begin_mask,
                         end_mask=end_mask, new_axis_mask=new_axis_mask,
                         shrink_axis_mask=shrink_axis_mask,
                         ellipsis_mask=ellipsis_mask, name="strided_slice")
    np_data = np.random.uniform(size=ip_shape).astype(dtype)

    with tf.Session() as sess:
        final_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(add_shapes=True),
            ['strided_slice'])
        tf_output = run_tf_graph(sess, np_data,
                                 'in_data:0', 'strided_slice:0')
        tvm_output = run_tvm_graph(final_graph_def, np_data,
                                   "in_data", tf_output.shape, np_data.dtype)
        np.testing.assert_allclose(tf_output, tvm_output, atol=1e-5, rtol=1e-5)
        sess.close()

def test_forward_stridedslice():
    '''test StridedSlice'''
    _test_stridedslice((3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1], 'float32')
    _test_stridedslice((3, 4, 3), [1, 0], [4, 3], [2, 1], 'float32', ellipsis_mask=8)
    _test_stridedslice((3, 4, 3), [1, 1, 0], [4, 4, 2], [2, 1, 1], 'float32', new_axis_mask=5)
    _test_stridedslice((3, 4, 3), [1, 1, 1], [4, 4, 1], [2, 1, 1], 'float32', ellipsis_mask=2, new_axis_mask=4)
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


#######################################################################
# Gather
# ------

def _test_gather(ip_shape, indice_shape, indice_value, axis, dtype):
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

    with tf.Session() as sess:
        final_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(add_shapes=True),
            ['GatherV2'])
        tf_output = run_tf_graph(sess, [np_data, np_indices], ['in_data:0',
                                 'indices:0'], 'GatherV2:0')
        tvm_output = run_tvm_graph(final_graph_def, [np_data, np_indices],
                                   ['in_data', 'indices'], tf_output.shape, dtype)
        np.testing.assert_allclose(tf_output, tvm_output, atol=1e-5, rtol=1e-5)
        sess.close()

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

        with tf.Session() as sess:
            graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(add_shapes=True),
                ['out'],
                )

            in_data = np.arange(9, dtype='int32').reshape([3, 3])

            tf_output = run_tf_graph(sess, [in_data, in_data, in_data, in_data ],
                                     ['in1:0', 'in2:0', 'in3:0', 'in4:0'], 'out:0')
            tvm_output = run_tvm_graph(graph_def,
                                       [in_data, in_data, in_data, in_data ],
                                       ['in1', 'in2', 'in3', 'in4'],
                                       tf_output.shape, tf_output.dtype)

            np.testing.assert_allclose(tf_output, tvm_output)

            sess.close()

#######################################################################
# Resize Bilinear
# ---------------

def _test_resize_bilinear(in_shape, to_shape, align_corners):
    """ One iteration of resize bilinear """

    data = np.random.uniform(size=in_shape).astype('float32')
    shape_data = np.array(to_shape).astype('int32')

    with tf.Graph().as_default():
        in_data = constant_op.constant(data, shape=data.shape, dtype=data.dtype)
        shape_data = constant_op.constant(shape_data, shape=shape_data.shape, dtype=shape_data.dtype)

        # pylint: disable=unused-variable
        resize_out = tf.image.resize_bilinear(in_data, shape_data, align_corners=align_corners)
        # pylint: enable=unused-variable

        with tf.Session() as sess:
            graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                sess.graph.as_graph_def(add_shapes=True),
                ['ResizeBilinear'],
                )

            tf_output = run_tf_graph(sess, data,
                    'Const:0', 'ResizeBilinear:0')

            tvm_output = run_tvm_graph(graph_def,
                                       data,
                                       "Const", tf_output.shape, data.dtype)

            np.testing.assert_allclose(tf_output, tvm_output, atol=1e-3, rtol=1e-3)

            sess.close()

def test_forward_resize_bilinear():
    """ Resize Bilinear """

    _test_resize_bilinear((4, 16, 32, 32), [50, 50], False)
    _test_resize_bilinear((6, 32, 64, 64), [20, 20], True)


#######################################################################
# Inception V3
# ------------
def test_forward_inception_v3():
    '''test inception V3 model'''
    with tf.Graph().as_default():
        graph_def = nnvm.testing.tf.get_workload('InceptionV3/inception_v3_2016_08_28_frozen-with_shapes.pb')
        # Call the utility to import the graph definition into default graph.
        graph_def = nnvm.testing.tf.ProcessGraphDefParam(graph_def)

        data = np.random.uniform(size=(1, 299, 299, 3)).astype('float32')

        with tf.Session() as sess:
            tf_output = run_tf_graph(sess, data, 'input:0', 'InceptionV3/Predictions/Reshape_1:0')
            tvm_output = run_tvm_graph(graph_def, data, 'input', tf_output.shape, 'float32')
            np.testing.assert_allclose(tf_output, tvm_output, rtol=1e-5, atol=1e-5)

#######################################################################
# Inception V1
# ------------
def test_forward_inception_v1():
    '''test inception V1 model'''
    with tf.Graph().as_default():
        graph_def = nnvm.testing.tf.get_workload("InceptionV1/classify_image_graph_def-with_shapes.pb")
        # Call the utility to import the graph definition into default graph.
        graph_def = nnvm.testing.tf.ProcessGraphDefParam(graph_def)

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
            tvm_output = run_tvm_graph(graph_def, tvm_data, 'DecodeJpeg/contents', tf_output.shape, 'float32')
            np.testing.assert_allclose(tf_output, tvm_output, rtol=1e-5, atol=1e-5)

#######################################################################
# Mobilenet
# ---------
def test_forward_mobilenet():
    '''test mobilenet model'''
    with tf.Graph().as_default():
        graph_def = nnvm.testing.tf.get_workload("MobilenetV1/mobilenet_v1_1.0_224_frozen-with-shapes.pb")
        # Call the utility to import the graph definition into default graph.
        graph_def = nnvm.testing.tf.ProcessGraphDefParam(graph_def)

        data = np.random.uniform(size=(1, 224, 224, 3)).astype('float32')
        out_node = 'MobilenetV1/Predictions/Reshape_1'

        with tf.Session() as sess:
            tf_output = run_tf_graph(sess, data, 'input:0', out_node + ':0')
            tvm_output = run_tvm_graph(graph_def, data, 'input', tf_output.shape, 'float32')
            np.testing.assert_allclose(np.squeeze(tvm_output), np.squeeze(tf_output), rtol=1e-5, atol=1e-5)

#######################################################################
# PTB
# ---
dir(tf.contrib)
def test_forward_ptb():
    '''test ptb model'''
    config = nnvm.testing.tf.get_config()
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
            sample = nnvm.testing.tf.pick_from_weight(tvm_output[0])
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
        word_to_id, id_to_word, graph_def = nnvm.testing.tf.get_workload_ptb()
        vocab_size = len(word_to_id)
        # Call the utility to import the graph definition into default graph.
        graph_def = nnvm.testing.tf.ProcessGraphDefParam(graph_def)
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
        tf_samples, tf_state = nnvm.testing.tf.do_tf_sample(sess,
                                [word_to_id[word] for word in seed_for_sample],
                                in_state, cnt_sample)
        tf_sample_str = _pretty_print(tf_samples, False, id_to_word)
        inpt = tvm_sample_str
        np.testing.assert_allclose(tf_samples, tvm_samples, rtol=1e-5, atol=1e-5)
        assert(tvm_sample_str == tf_sample_str)

#######################################################################
# Main
# ----
if __name__ == '__main__':
    test_forward_convolution()
    test_forward_pooling()
    test_forward_reshape()
    test_forward_squeeze()
    test_forward_sigmoid()
    if tf.__version__ == '1.4.1':
        _test_forward_concat_v2()
    test_forward_multi_input()
    test_forward_inception_v3()
    test_forward_inception_v1()
    test_forward_mobilenet()
    test_forward_variable()
    test_forward_resize_bilinear()
    test_forward_lstm()
    test_forward_stridedslice()
    test_forward_gather()
    test_forward_ptb()
