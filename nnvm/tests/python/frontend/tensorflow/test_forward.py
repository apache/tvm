# pylint: disable=import-self, invalid-name, unused-argument
"""
Tensorflow testcases
=====================
This article is a test script to test tensorflow models with NNVM.
All the required dependency files will be downloaded from the internet
by the script.
"""
from __future__ import print_function
import os
import sys
import urllib
import requests
import numpy as np
import nnvm.compiler
import tvm
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops


if sys.version_info >= (3,):
    import urllib.request as urllib2
else:
    import urllib2

#######################################################################
# Some tensorflow helper functions to handle models
# -------------------------------------------------
def process_graph_default(graph_def):
    """Type-checks and possibly canonicalizes `graph_def`."""
    if not isinstance(graph_def, graph_pb2.GraphDef):
        # `graph_def` could be a dynamically-created message, so try a duck-typed
        # approach
        try:
            old_graph_def = graph_def
            graph_def = graph_pb2.GraphDef()
            graph_def.MergeFrom(old_graph_def)
        except TypeError:
            raise TypeError('graph_def must be a GraphDef proto.')
    return graph_def


def load_graph(model_name):
    with tf.gfile.FastGFile(model_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        # pylint: disable=unused-variable
        graph = tf.import_graph_def(graph_def, name='')
        # pylint: enable=unused-variable
        graph_def = process_graph_default(graph_def)
    return graph_def

#######################################################################
# File download helper function
# -----------------------------
def _download(url, path, overwrite=False, sizecompare=False):
    ''' Download from internet'''
    if os.path.isfile(path) and not overwrite:
        if sizecompare:
            file_size = os.path.getsize(path)
            res_head = requests.head(url)
            res_get = requests.get(url, stream=True)
            if 'Content-Length' not in res_head.headers:
                res_get = urllib2.urlopen(url)
            urlfile_size = int(res_get.headers['Content-Length'])
            if urlfile_size != file_size:
                print("exist file got corrupted, downloading", path, " file freshly")
                _download(url, path, True, False)
                return
        print('File {} exists, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    # pylint: disable=bare-except
    try:
        urllib.request.urlretrieve(url, path)
        print('')
    except:
        urllib.urlretrieve(url, path)
    # pylint: enable=bare-except

#######################################################################
# Generic run functions for TVM & tensorflow
# ------------------------------------------
def run_tvm_graph(graph_def, input_data, input_node, output_shape, output_dtype):
    """ Generic function to compile on nnvm and execute on tvm """
    sym, params = nnvm.frontend.from_tensorflow(graph_def)
    target = 'llvm'
    shape_dict = {input_node: input_data.shape}
    dtype_dict = {input_node: input_data.dtype}
    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict,
                                             dtype=dtype_dict, params=params)

    ctx = tvm.cpu(0)
    from tvm.contrib import graph_runtime
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input(input_node, tvm.nd.array(input_data.astype(input_data.dtype)))
    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0, tvm.nd.empty((output_shape), output_dtype))
    return tvm_output.asnumpy()

def run_tf_graph(sess, input_data, input_node, output_node):
    tensor = sess.graph.get_tensor_by_name(output_node)
    output_data = sess.run(tensor, {input_node: input_data})
    return output_data

#######################################################################
# Inception V1
# ------------
def inception_v1_tvm(graph_def, image_name):
    from PIL import Image
    image = Image.open(image_name).resize((299, 299))
    image = np.array(image)

    output = run_tvm_graph(graph_def, image, 'DecodeJpeg/contents', (1, 1008), 'float32')
    return np.squeeze(output)


def inception_v1_tf(graph_def, image_name):
    if not tf.gfile.Exists(image_name):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image_name, 'rb').read()

    with tf.Session() as sess:
        output = run_tf_graph(sess, image_data, 'DecodeJpeg/contents:0', 'softmax:0')
        return np.squeeze(output)

def test_forward_inception_v1():
    '''test inception V1 model'''
    model_name = 'inception_v1'

    repo = 'https://github.com/srkreddy1238/dmlc_data/raw/master/models/tensorflow/InceptionV1/'
    model_name = 'classify_image_graph_def-with_shapes.pb'

    model_url = repo + model_name
    _download(model_url, model_name)

    graph_def = load_graph(model_name)

    image_name = 'elephant-299.jpg'
    image_url = repo + image_name
    _download(image_url, image_name)

    tf_output = inception_v1_tf(graph_def, image_name)
    tvm_output = inception_v1_tvm(graph_def, image_name)

    np.testing.assert_allclose(tf_output, tvm_output, rtol=2e-2, atol=2e-2)

#######################################################################
# Convolution
# -----------

# Borrowed from tensorflow for test cases.
def get_shrunk_inception_shapes(shrink=10):
    """Iterator for smaller versions of convolution shapes in 2015 Inception.

    Relative to inception, each depth value is `depth // shrink`.

    Args:
        shrink: Factor to shrink each depth value by relative to Inception.

    Yields:
        Tuple (input_size, filter_size, out_size, stride, padding), the convolution
        parameters of Inception layers.
    """
    input_sizes = [[4, 5, 5, 1248], [4, 8, 8, 384], [4, 8, 8, 384],
                   [4, 8, 8, 2048], [4, 8, 8, 448], [4, 8, 8, 2048],
                   [4, 8, 8, 2048], [4, 8, 8, 2048], [4, 8, 8, 1760],
                   [4, 8, 8, 1760], [4, 8, 8, 1760], [4, 8, 8, 1760],
                   [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 1248],
                   [4, 17, 17, 128], [4, 17, 17, 1248], [4, 17, 17, 224],
                   [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 1216],
                   [4, 17, 17, 1216], [4, 17, 17, 224], [4, 17, 17, 192],
                   [4, 17, 17, 192], [4, 17, 17, 1152], [4, 17, 17, 1152],
                   [4, 17, 17, 192], [4, 17, 17, 160], [4, 17, 17, 1152],
                   [4, 17, 17, 1024], [4, 17, 17, 128], [4, 17, 17, 1024],
                   [4, 17, 17, 128], [4, 17, 17, 1024], [4, 17, 17, 128],
                   [4, 17, 17, 768], [4, 17, 17, 128], [4, 17, 17, 128],
                   [4, 17, 17, 768], [4, 17, 17, 768], [4, 35, 35, 96],
                   [4, 35, 35, 288], [4, 35, 35, 64], [4, 35, 35, 288],
                   [4, 35, 35, 256], [4, 35, 35, 48], [4, 35, 35, 256],
                   [4, 35, 35, 96], [4, 35, 35, 192], [4, 35, 35, 192],
                   [4, 35, 35, 192], [4, 73, 73, 64], [4, 73, 73, 64],
                   [4, 147, 147, 24]]
    filter_sizes = [[1, 1, 1248, 128], [1, 3, 384, 384], [3, 1, 384, 384],
                    [1, 1, 2048, 192], [3, 3, 448, 384], [1, 1, 2048, 320],
                    [1, 1, 2048, 448], [1, 1, 2048, 384], [1, 1, 1760, 384],
                    [1, 1, 1760, 192], [1, 1, 1760, 448], [1, 1, 1760, 320],
                    [3, 3, 192, 192], [3, 3, 192, 192], [1, 1, 1248, 192],
                    [3, 3, 128, 320], [1, 1, 1248, 128], [1, 3, 224, 224],
                    [3, 1, 192, 256], [1, 3, 192, 256], [1, 1, 1216, 192],
                    [1, 1, 1216, 96], [3, 1, 224, 224], [3, 3, 192, 224],
                    [1, 3, 192, 192], [1, 1, 1152, 192], [1, 1, 1152, 128],
                    [3, 1, 192, 192], [3, 3, 160, 192], [1, 1, 1152, 160],
                    [1, 1, 1024, 128], [1, 3, 128, 192], [1, 1, 1024, 160],
                    [3, 1, 128, 192], [1, 1, 1024, 256], [3, 1, 128, 128],
                    [1, 1, 768, 192], [1, 3, 128, 128], [3, 3, 128, 128],
                    [1, 1, 768, 128], [1, 1, 768, 320], [3, 3, 96, 96],
                    [3, 3, 288, 384], [3, 3, 64, 96], [1, 1, 288, 64],
                    [1, 1, 256, 64], [5, 5, 48, 64], [1, 1, 256, 48],
                    [3, 3, 96, 96], [1, 1, 192, 32], [1, 1, 192, 64],
                    [1, 1, 192, 48], [3, 3, 64, 192], [1, 1, 64, 64],
                    [1, 1, 24, 64]]
    out_sizes = [[4, 5, 5, 128], [4, 8, 8, 384], [4, 8, 8, 384],
                 [4, 8, 8, 192], [4, 8, 8, 384], [4, 8, 8, 320],
                 [4, 8, 8, 448], [4, 8, 8, 384], [4, 8, 8, 384],
                 [4, 8, 8, 192], [4, 8, 8, 448], [4, 8, 8, 320],
                 [4, 8, 8, 192], [4, 17, 17, 192], [4, 17, 17, 192],
                 [4, 8, 8, 320], [4, 17, 17, 128], [4, 17, 17, 224],
                 [4, 17, 17, 256], [4, 17, 17, 256], [4, 17, 17, 192],
                 [4, 17, 17, 96], [4, 17, 17, 224], [4, 17, 17, 224],
                 [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 128],
                 [4, 17, 17, 192], [4, 17, 17, 192], [4, 17, 17, 160],
                 [4, 17, 17, 128], [4, 17, 17, 192], [4, 17, 17, 160],
                 [4, 17, 17, 192], [4, 17, 17, 256], [4, 17, 17, 128],
                 [4, 17, 17, 192], [4, 17, 17, 128], [4, 17, 17, 128],
                 [4, 17, 17, 128], [4, 17, 17, 320], [4, 17, 17, 96],
                 [4, 17, 17, 384], [4, 35, 35, 96], [4, 35, 35, 64],
                 [4, 35, 35, 64], [4, 35, 35, 64], [4, 35, 35, 48],
                 [4, 35, 35, 96], [4, 35, 35, 32], [4, 35, 35, 64],
                 [4, 35, 35, 48], [4, 71, 71, 192], [4, 73, 73, 64],
                 [4, 147, 147, 64]]
    strides = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ]
    # Shrink sizes to make the test faster
    # pylint: disable=invalid-name
    for i in input_sizes:
        i[3] //= shrink
    for f in filter_sizes:
        f[2] //= shrink
        f[3] //= shrink
    for o in out_sizes:
        o[3] //= shrink

    VALID = "VALID"
    SAME = "SAME"
    paddings = [
        SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME,
        VALID, SAME, SAME, VALID, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME,
        SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME, SAME,
        SAME, SAME, SAME, SAME, SAME, VALID, VALID, SAME, SAME, SAME, SAME, SAME,
        SAME, SAME, SAME, SAME, VALID, VALID, VALID
    ]
    for i, f, o, s, p in zip(input_sizes, filter_sizes, out_sizes, strides,
                             paddings):
        yield i, f, o, s, p
    # pylint: enable=invalid-name

def test_convolution_iteration(tensor_in_sizes, filter_in_sizes,
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
    # pylint: disable=unused-variable
    for index, (input_size_, filter_size_, output_size_, stride_,
                padding_) in enumerate(get_shrunk_inception_shapes()):
        with tf.Graph().as_default():
            test_convolution_iteration(input_size_, filter_size_, [1, 1],
                                       [stride_, stride_], padding_, 'NHWC')
    # pylint: enable=unused-variable

#######################################################################
# Main
# ----
if __name__ == '__main__':
    test_forward_inception_v1()
    test_forward_convolution()
