# pylint: disable=import-self, invalid-name, unused-argument
"""
TFLite testcases
================
This article is a test script to test TFLite operator with Relay.
"""
from __future__ import print_function
import numpy as np
import tvm
from tvm import relay
from tvm.contrib import util
import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.contrib import lite as interpreter_wrapper

import tvm.relay.testing.tf as tf_testing

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

    func, params = relay.frontend.from_tflite(tflite_model,
                                              shape_dict=shape_dict,
                                              dtype_dict=dtype_dict)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(func, target, params=params)

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


def compare_tflite_with_tvm(tflite_in_data, tvm_in_data, in_name, input_tensors,
                            output_tensors, output_need_transpose_nchw=False,
                            init_global_variables=False):
    """Generic function to generate and compare TFLite and TVM output"""
    tflite_in_data = convert_to_list(tflite_in_data)
    tvm_in_data = convert_to_list(tvm_in_data)
    in_name = convert_to_list(in_name)
    in_node = [0] * len(in_name)
    for i in range(len(in_name)):
        in_node[i] = in_name[i].split(':')[0] if ":" in in_name[i] else in_name[i]

    with tf.Session() as sess:
        if init_global_variables:
            sess.run(variables.global_variables_initializer())
        # convert to tflite model
        converter = tf.contrib.lite.TFLiteConverter.from_session(
            sess, input_tensors, output_tensors)
        tflite_model_buffer = converter.convert()
        tflite_output = run_tflite_graph(tflite_model_buffer, tflite_in_data)

        for device in ["llvm"]:
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                continue

            tvm_output = run_tvm_graph(tflite_model_buffer, tvm_in_data, in_node, target=device)
            for i in range(len(tflite_output)):
                if output_need_transpose_nchw:
                    tvm.testing.assert_allclose(tflite_output[i],
                                                np.transpose(tvm_output[i], axes=(0, 2, 3, 1)),
                                                atol=1e-5, rtol=1e-5)
                else:
                    tvm.testing.assert_allclose(tflite_output[i], tvm_output[i],
                                                atol=1e-5, rtol=1e-5)

        sess.close()


#######################################################################
# Pooling
# -------
def _test_pooling_iteration(input_shape, **kwargs):
    """ One iteration of pool operation with given shapes and attributes """

    x = -np.arange(
        np.prod(input_shape), dtype=np.float32).reshape(input_shape) - 1
    tvm_data = np.transpose(x, axes=(0, 3, 1, 2))

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=input_shape, dtype='float32')
        out = nn_ops.pool(in_data, **kwargs)

        compare_tflite_with_tvm(x, tvm_data, 'Placeholder:0', [in_data], [out],
                                output_need_transpose_nchw=True)


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
        # TFLite is NHWC, TVM is NCHW
        tflite_data_array = np.reshape(data_array, tensor_in_sizes).astype('float32')
        tvm_data_array = np.transpose(tflite_data_array, axes=(0, 3, 1, 2))
        # TFLite output is NHWC, TVM is NCHW, we need transpose
        compare_tflite_with_tvm(tflite_data_array, tvm_data_array,
                                'Placeholder:0', [in_data], [out],
                                output_need_transpose_nchw=True)


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


#######################################################################
# Reshape
# -------

def _test_reshape(data, out_shape):
    """ One iteration of reshape operation with given data and out shape """
    # see relay/frontend/tflite.py convert_reshape more detail of channel first rule
    if len(data.shape) == 1 or len(data.shape) == 2:
        tvm_data = data
    elif len(data.shape) == 3:
        tvm_data = np.transpose(data, axes=(0, 2, 1))
    elif len(data.shape) == 4:
        tvm_data = np.transpose(data, axes=(0, 3, 1, 2))
    else:
        raise NotImplementedError("Not support input shape {} of reshape : ".
                                  format(str(len(data))))

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = array_ops.reshape(in_data, out_shape)

        compare_tflite_with_tvm(data, tvm_data, 'Placeholder:0', [in_data], [out])


def test_forward_reshape():
    _test_reshape(np.arange(6.0, dtype=np.float32), [2, 3])
    _test_reshape(np.arange(6), [-1, 2])
    _test_reshape(np.arange(6), [3, -1])
    _test_reshape(np.arange(6), [-1])


#######################################################################
# Concatenation
# -------------

def _test_concatenation(data, axis):
    """ One iteration of concatenation """

    assert len(data) >= 1
    need_transpose = False
    if len(data[0].shape) == 1 or len(data[0].shape) == 2:
        tvm_data = data
    elif len(data[0].shape) == 3:
        #need_transpose = True
        tvm_data = [np.transpose(d, axes=(0, 2, 1)) for d in data]
    elif len(data[0].shape) == 4:
        need_transpose = True
        tvm_data = [np.transpose(d, axes=(0, 3, 1, 2)) for d in data]
    else:
        raise NotImplementedError("Not support input shape {} of reshape : ".
                                  format(str(len(data))))

    with tf.Graph().as_default():
        in_data = [
            array_ops.placeholder(shape=tensor.shape, dtype=tensor.dtype, name="in_{}".format(idx))
            for idx, tensor in enumerate(data)]
        out = array_ops.concat(in_data, axis=axis)
        name = ["in_{}:0".format(idx) for idx in range(len(data))]

        compare_tflite_with_tvm(data, tvm_data, name, in_data, [out], need_transpose)


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
# Squeeze
# -------

def _test_squeeze(data, squeeze_dims=None):
    """ One iteration of squeeze """

    if squeeze_dims is None:
        squeeze_dims = []

    # see relay/frontend/tflite.py convert_squeeze more detail of channel first rule
    if len(data.shape) == 1 or len(data.shape) == 2:
        tvm_data = data
    elif len(data.shape) == 3:
        tvm_data = np.transpose(data, axes=(0, 2, 1))
    elif len(data.shape) == 4:
        tvm_data = np.transpose(data, axes=(0, 3, 1, 2))
    else:
        raise NotImplementedError("Not support input shape {} of reshape : ".
                                  format(str(len(data.shape))))

    tvm_data = np.transpose(data, axes=(0, 3, 1, 2))

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)

        if squeeze_dims:
            out = array_ops.squeeze(in_data, squeeze_dims)
        else:
            out = array_ops.squeeze(in_data)

        compare_tflite_with_tvm(data, tvm_data, 'Placeholder:0', [in_data], [out])


def test_forward_squeeze():
    """ Squeeze """
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3)), [0, 2])
    _test_squeeze(np.arange(6).reshape((2, 1, 3, 1)), [1, 3])

#######################################################################
# Softmax
# -------

def _test_softmax(data):
    """ One iteration of softmax """
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = nn_ops.softmax(in_data)
        compare_tflite_with_tvm(data, data, 'Placeholder:0', [in_data], [out])

def test_forward_softmax():
    """ Softmax """
    _test_softmax(np.arange(6.0, dtype=np.float32).reshape((1, 6)))

#######################################################################
# Mobilenet
# ---------

def test_forward_mobilenet():
    '''test mobilenet v1 tflite model'''
    # MobilenetV1
    temp = util.tempdir()
    tflite_model_file = tf_testing.get_workload_official(
        "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz",
        "mobilenet_v1_1.0_224.tflite", temp)
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 224, 224, 3)).astype('float32')
    tvm_data = np.transpose(data, axes=(0, 3, 1, 2))
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, tvm_data, 'input')
    tvm.testing.assert_allclose(np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]),
                                rtol=1e-5, atol=1e-5)
    temp.remove()

#######################################################################
# Inception V3
# ------------

def test_forward_inception_v3_net():
    '''test inception v3 tflite model'''
    # InceptionV3
    temp = util.tempdir()
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz",
        "inception_v3.tflite", temp)
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 299, 299, 3)).astype('float32')
    tvm_data = np.transpose(data, axes=(0, 3, 1, 2))
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, tvm_data, 'input')
    tvm.testing.assert_allclose(np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]),
                                rtol=1e-5, atol=1e-5)
    temp.remove()

#######################################################################
# Main
# ----
if __name__ == '__main__':
    # Transforms
    test_forward_concatenation()
    test_forward_reshape()
    test_forward_squeeze()

    # NN
    test_forward_convolution()
    test_forward_pooling()
    test_forward_softmax()

    # End to End
    test_forward_mobilenet()
    test_forward_inception_v3_net()
