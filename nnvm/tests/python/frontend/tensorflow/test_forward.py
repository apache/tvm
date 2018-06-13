# pylint: disable=import-self, invalid-name, unused-argument
"""
Tensorflow testcases
=====================
This article is a test script to test tensorflow operator with NNVM.
"""
from __future__ import print_function
import numpy as np
import nnvm.compiler
import tvm
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops

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
    tvm_output = m.get_output(0, tvm.nd.empty((output_shape), output_dtype))
    return tvm_output.asnumpy()

def run_tf_graph(sess, input_data, input_node, output_node):
    """ Generic function to execute tensor flow """

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
def test_pooling(input_shape, **kwargs):
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

    test_pooling(input_shape=[2, 9, 10, 2],
                 window_shape=[1, 1],
                 padding='SAME',
                 pooling_type='MAX',
                 dilation_rate=[1, 1],
                 strides=[1, 1])

    test_pooling(input_shape=[2, 9, 10, 2],
                 window_shape=[1, 1],
                 padding='SAME',
                 pooling_type='AVG',
                 dilation_rate=[1, 1],
                 strides=[1, 1])
    test_pooling(input_shape=[2, 10, 9, 2],
                 window_shape=[1, 1],
                 padding='SAME',
                 pooling_type='MAX',
                 dilation_rate=[1, 1],
                 strides=[1, 1])
    test_pooling(input_shape=[2, 10, 9, 2],
                 window_shape=[1, 1],
                 padding='SAME',
                 pooling_type='AVG',
                 dilation_rate=[1, 1],
                 strides=[1, 1])

#######################################################################
# Convolution
# -----------

def test_convolution(tensor_in_sizes, filter_in_sizes,
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
    test_convolution([4, 8, 8, 176], [1, 1, 176, 32], [1, 1], [1, 1], 'SAME', 'NHWC')
    test_convolution([4, 17, 17, 19], [3, 3, 19, 19], [1, 1], [2, 2], 'VALID', 'NHWC')
    test_convolution([4, 17, 17, 124], [1, 1, 124, 19], [1, 1], [1, 1], 'SAME', 'NHWC')
    test_convolution([4, 17, 17, 12], [3, 3, 12, 32], [1, 1], [2, 2], 'VALID', 'NHWC')

#######################################################################
# Reshape
# -----------

def test_reshape(data, out_shape):
    """ One iteration of reshape operation with given sata and out shape """

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
    test_reshape(np.arange(6.0), [2, 3])
    test_reshape(np.arange(6), [-1, 2])
    test_reshape(np.arange(6), [3, -1])
    test_reshape(np.arange(6), [-1])

#######################################################################
# Squeeze
# -----------

def test_squeeze(data, squeeze_dims=None):
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
    test_squeeze(np.arange(2).reshape((2)))
    test_squeeze(np.arange(6).reshape((2, 3)))

    # Squeeze the middle element away.
    test_squeeze(np.arange(4).reshape((2, 1, 2)))

    # Squeeze on both ends.
    test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)))

    # Positive squeeze dim index.
    test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [0])
    test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [2, 4])
    test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [0, 4, 2])

    # Negative squeeze dim index.
    test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [-1])
    test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [-3, -5])
    test_squeeze(np.arange(6).reshape((1, 2, 1, 3, 1)), [-3, -5, -1])

#######################################################################
# ConcatV2
# -----------

def test_concat_v2(data, dim):
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

def test_forward_concat_v2():
    t1 = np.array([])
    t2 = np.array([])
    test_concat_v2([t1, t2], 0)

    t1 = np.array([[1, 2, 3], [4, 5, 6]])
    t2 = np.array([[7, 8, 9], [10, 11, 12]])

    test_concat_v2([t1, t2], 1)

#######################################################################
# Main
# ----
if __name__ == '__main__':
    test_forward_convolution()
    test_forward_pooling()
    test_forward_reshape()
    test_forward_squeeze()
    test_forward_concat_v2()
