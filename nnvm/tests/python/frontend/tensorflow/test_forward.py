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
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
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
        img_array = np.random.uniform(size=(1, 600, 600, 3)).astype("uint8")
        img = Image.frombuffer('RGB', (600, 600), img_array.tostring(), 'raw', 'RGB', 0, 1)
        img.save('tf_test.jpg');

        import os.path
        if not tf.gfile.Exists(os.path.join('./tf_test.jpg')):
            tf.logging.fatal('File does not exist %s', image)
        data = tf.gfile.FastGFile(os.path.join("./tf_test.jpg"), 'rb').read()

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
