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
This article is a test script to test tensorflow operator with Relay.
"""
from __future__ import print_function
import numpy as np
import pytest
try:
    import tensorflow.compat.v1 as tf
except ImportError:
    import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import graph_util
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import init_ops
from distutils.version import LooseVersion
import tvm
from tvm import te
from tvm import relay
import tvm.relay.testing.tf as tf_testing
from packaging import version as package_version

#######################################################################
# Generic run functions for TVM & tensorflow
# ------------------------------------------


def convert_to_list(x):
    if not isinstance(x, list):
        x = [x]
    return x

tf_dtypes = {
    'float32': tf.float32,
    'float16': tf.float16,
    'float64': tf.float64,
    'int32': tf.int32,
    'uint8' : tf.uint8,
    'int8': tf.int8,
    'int16': tf.int16,
    'uint16': tf.uint16,
    'int64': tf.int64,
}

def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        return [o.asnumpy().tolist()]
    elif isinstance(o, tvm.runtime.container.ADT):
        result = []
        for f in o:
            result.extend(vmobj_to_list(f))
        return result
    elif isinstance(o, tvm.relay.backend.interpreter.ConstructorValue):
        if o.constructor.name_hint == 'Cons':
            tl = vmobj_to_list(o.fields[1])
            hd = vmobj_to_list(o.fields[0])
            hd.extend(tl)
            return hd
        elif o.constructor.name_hint == 'Nil':
            return []
        elif 'tensor_nil' in o.constructor.name_hint:
            return [0]
        elif 'tensor' in o.constructor.name_hint:
            return [o.fields[0].asnumpy()]
        else:
            raise RuntimeError("Unknown object type: %s" %
                               o.constructor.name_hint)
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


def run_tvm_graph(graph_def, input_data, input_node, num_output=1,
                  target='llvm', out_names=None, opt_level=3, mode='graph_runtime',
                  cuda_layout="NCHW"):
    """ Generic function to compile on relay and execute on tvm """
    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)
    layout = None
    if target == "cuda":
        layout = cuda_layout
    target_host = None
    shape_dict = {e: i.shape for e, i in zip(input_node, input_data)}
    mod, params = relay.frontend.from_tensorflow(graph_def,
                                                 layout=layout,
                                                 shape=shape_dict,
                                                 outputs=out_names)
    if mode in ['debug', 'vm']:
        ex = relay.create_executor(mode, mod=mod, ctx=tvm.cpu(), target="llvm")
        inputs = []
        for param in mod['main'].params:
            found = False
            for i, n in enumerate(input_node):
                if n == param.name_hint:
                    found = True
                    inputs.append(tvm.nd.array(input_data[i]))
                    break
            # Interpreter doesn't bind constants, so still need to find in params
            if not found:
                inputs.append(tvm.nd.array(params[param.name_hint]))
        result = ex.evaluate()(*inputs)
        return vmobj_to_list(result)
    else:
        with relay.build_config(opt_level=opt_level):
            graph, lib, params = relay.build(mod, target, target_host, params)

        ctx = tvm.context(target, 0)
        from tvm.contrib import graph_runtime
        m = graph_runtime.create(graph, lib, ctx)
        # set inputs
        for e, i in zip(input_node, input_data):
            m.set_input(e, tvm.nd.array(i))

        m.set_input(**params)
        # execute
        m.run()
        # get outputs
        assert out_names is None or num_output == len(out_names), (
            "out_names: {} num_output: {}".format(out_names, num_output))
        tvm_output_list = [m.get_output(i).asnumpy()
                           for i in range(num_output)]
        return tvm_output_list


def run_tf_graph(sess, input_data, input_node, output_node):
    """ Generic function to execute tensorflow """
    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)
    output_node = convert_to_list(output_node)

    tensor = [sess.graph.get_tensor_by_name(
        output_name) for output_name in output_node]

    input_dict = {e: input_data[i] for i, e in enumerate(input_node)}

    output_data = sess.run(tensor, input_dict)
    return output_data


def compare_tf_with_tvm(in_data, in_name, out_name, init_global_variables=False,
                        no_gpu=False, opt_level=3, mode='graph_runtime',
                        cuda_layout="NCHW"):
    """Generic function to generate and compare tensorflow and TVM output"""
    def name_without_num(name):
        return name.split(':')[0] if ":" in name else name

    out_name = convert_to_list(out_name)
    out_node = [name_without_num(name) for name in out_name]

    in_data = convert_to_list(in_data)
    in_name = convert_to_list(in_name)
    in_node = [name_without_num(name) for name in in_name]
    with tf.Session() as sess:
        if init_global_variables:
            sess.run(variables.global_variables_initializer())
        final_graph_def = tf_testing.AddShapesToGraphDef(sess, out_node)
        tf_output = run_tf_graph(sess, in_data, in_name, out_name)

        for device in ["llvm", "cuda"]:
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                continue
            if no_gpu and device == 'cuda':
                continue

            tvm_output = run_tvm_graph(final_graph_def, in_data, in_node,
                                       target=device, out_names=out_name,
                                       num_output=len(out_name), opt_level=opt_level, mode=mode,
                                       cuda_layout=cuda_layout)
            # since the names from tensorflow and relay runs are not exactly same,
            # first len(tf_output) will be compared
            for i in range(len(tf_output)):
                tvm.testing.assert_allclose(
                    tf_output[i], tvm_output[i], atol=1e-5, rtol=1e-5)

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
        if len(input_shape) == 4:
            input_shape = [input_shape[ii] for ii in (0, 3, 1, 2)]
            kwargs['data_format'] = 'NCHW'
            _test_pooling_iteration(input_shape, **kwargs)


def test_forward_pooling():
    """ Pooling """
    # TensorFlow only supports NDHWC for max_pool3d on CPU
    for pool_type in ['AVG', 'MAX']:
        # NDHWC is the default layout for max_pool3d and avg_pool3d in TensorFlow
        _test_pooling(input_shape=[1, 3, 32, 32, 32],
                      window_shape=[2, 2, 2],
                      padding='VALID',
                      pooling_type=pool_type,
                      dilation_rate=[1, 1, 1],
                      strides=[2, 2, 2])

        _test_pooling(input_shape=[1, 3, 32, 32, 32],
                      window_shape=[1, 1, 1],
                      padding='SAME',
                      pooling_type=pool_type,
                      dilation_rate=[1, 1, 1],
                      strides=[1, 1, 1])

        _test_pooling(input_shape=[1, 3, 32, 32, 32],
                      window_shape=[2, 2, 2],
                      padding='SAME',
                      pooling_type=pool_type,
                      dilation_rate=[1, 1, 1],
                      strides=[2, 2, 2])

        # test cases for max_pool3d & avg_pool3d with layout NCDHW
        # TensorFlow pool3d  doesn't support NCDHW on cpu
        if is_gpu_available():
            _test_pooling(input_shape=[1, 3, 32, 32, 32],
                          window_shape=[1, 1, 1],
                          padding='SAME',
                          pooling_type=pool_type,
                          dilation_rate=[1, 1, 1],
                          strides=[1, 1, 1],
                          data_format='NCDHW')

            _test_pooling(input_shape=[1, 3, 32, 32, 32],
                          window_shape=[2, 2, 2],
                          padding='VALID',
                          pooling_type=pool_type,
                          dilation_rate=[1, 1, 1],
                          strides=[2, 2, 2],
                          data_format='NCDHW')

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

        # Tests involving SpaceToBatchND
        _test_pooling(input_shape=[1, 1, 2, 1],
                      window_shape=[1, 1],
                      padding='VALID',
                      pooling_type=pool_type,
                      dilation_rate=[1, 2])

        _test_pooling(input_shape=[1, 2, 1],
                      window_shape=[1],
                      padding='VALID',
                      pooling_type=pool_type,
                      dilation_rate=[2])

#######################################################################
# Convolution
# -----------


def _test_convolution(opname, tensor_in_sizes, filter_in_sizes,
                      dilations, strides, padding, data_format,
                      deconv_output_shape=[]):
    """ One iteration of convolution with given shapes and attributes """

    total_size_1 = np.prod(tensor_in_sizes)
    total_size_2 = np.prod(filter_in_sizes)
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    data_array = [f * 1.0 for f in range(1, total_size_1 + 1)]
    filter_array = [f * 1.0 for f in range(1, total_size_2 + 1)]

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=tensor_in_sizes, dtype='float32')
        in_filter = constant_op.constant(
            filter_array, shape=filter_in_sizes, dtype='float32')
        if data_format == 'NHWC':
            strides = [1] + strides + [1]
            dilations = [1] + dilations + [1]
        else:
            strides = [1, 1] + strides
            dilations = [1, 1] + dilations

        if opname == 'conv':
            nn_ops.conv2d(in_data,
                          in_filter,
                          strides=strides,
                          dilations=dilations,
                          padding=padding,
                          data_format=data_format)

            compare_tf_with_tvm(np.reshape(data_array, tensor_in_sizes).astype('float32'),
                                'Placeholder:0', 'Conv2D:0')
        elif opname == 'conv_transpose':
            nn_ops.conv2d_transpose(in_data,
                                    in_filter,
                                    output_shape=deconv_output_shape,
                                    strides=strides,
                                    padding=padding,
                                    data_format=data_format)

            compare_tf_with_tvm(np.reshape(data_array, tensor_in_sizes).astype('float32'),
                                'Placeholder:0', 'conv2d_transpose:0')
        else:
            nn_ops.depthwise_conv2d_native(in_data,
                                           in_filter,
                                           strides=strides,
                                           dilations=dilations,
                                           padding=padding,
                                           data_format=data_format)

            compare_tf_with_tvm(np.reshape(data_array, tensor_in_sizes).astype('float32'),
                                'Placeholder:0', 'DepthwiseConv2dNative:0')


def test_forward_convolution():
    if is_gpu_available():
        _test_convolution('conv', [4, 176, 8, 8], [1, 1, 176, 32], [1, 1], [1, 1], 'SAME', 'NCHW')
        _test_convolution('conv', [4, 19, 17, 17], [3, 3, 19, 19], [1, 1], [2, 2], 'VALID', 'NCHW')
        _test_convolution('conv', [4, 124, 17, 17], [1, 1, 124, 19], [1, 1], [1, 1], 'SAME', 'NCHW')
        _test_convolution('conv', [4, 12, 17, 17], [3, 3, 12, 32], [1, 1], [2, 2], 'VALID', 'NCHW')
        _test_convolution('depthwise', [4, 176, 8, 8], [1, 1, 176, 1], [1, 1], [1, 1], 'SAME', 'NCHW')
        _test_convolution('depthwise', [4, 19, 17, 17], [3, 3, 19, 1], [1, 1], [2, 2], 'VALID', 'NCHW')
        _test_convolution('depthwise', [4, 124, 17, 17], [1, 1, 124, 1], [1, 1], [1, 1], 'SAME', 'NCHW')
        _test_convolution('depthwise', [4, 12, 17, 17], [3, 3, 12, 1], [1, 1], [2, 2], 'VALID', 'NCHW')
        _test_convolution('depthwise', [4, 12, 17, 17], [3, 3, 12, 2], [1, 1], [2, 2], 'VALID', 'NCHW')
        _test_convolution('conv_transpose', [4, 32, 8, 8], [1, 1, 176, 32], [1, 1], [1, 1], 'SAME',
                          'NCHW', [4, 176, 8, 8])
        _test_convolution('conv_transpose', [4, 32, 8, 8], [2, 2, 176, 32], [1, 1], [1, 1], 'SAME',
                          'NCHW', [4, 176, 8, 8])
        _test_convolution('conv_transpose', [4, 32, 8, 8], [2, 2, 176, 32], [1, 1], [2, 2], 'SAME',
                          'NCHW', [4, 176, 15, 15])
        _test_convolution('conv_transpose', [4, 32, 8, 8], [3, 3, 176, 32], [1, 1], [1, 1], 'SAME',
                          'NCHW', [4, 176, 8, 8])
        _test_convolution('conv_transpose', [4, 32, 8, 8], [3, 3, 176, 32], [1, 1], [2, 2], 'SAME',
                          'NCHW', [4, 176, 15, 15])
        _test_convolution('conv_transpose', [4, 32, 8, 8], [3, 3, 176, 32], [1, 1], [2, 2], 'SAME',
                          'NCHW', [4, 176, 16, 16])
        _test_convolution('conv_transpose', [4, 19, 8, 8], [3, 3, 19, 19], [1, 1], [2, 2], 'VALID',
                          'NCHW', [4, 19, 17, 17])
        _test_convolution('conv_transpose', [4, 19, 17, 17], [1, 1, 124, 19], [1, 1], [1, 1], 'SAME',
                          'NCHW', [4, 124, 17, 17])
        _test_convolution('conv_transpose', [4, 19, 17, 17], [3, 3, 124, 19], [1, 1], [1, 1], 'SAME',
                          'NCHW', [4, 124, 17, 17])
        _test_convolution('conv_transpose', [4, 32, 8, 8], [3, 3, 12, 32], [1, 1], [2, 2], 'VALID',
                          'NCHW', [4, 12, 17, 17])
        # kernel 2x2, strides (2,2)
        _test_convolution('conv_transpose', [4, 19, 8, 8], [2, 2, 19, 19], [1, 1], [2, 2], 'VALID',
                          'NCHW', [4, 19, 16, 16])
        _test_convolution('conv_transpose', [4, 32, 8, 8], [2, 2, 12, 32], [1, 1], [2, 2], 'VALID',
                          'NCHW', [4, 12, 16, 16])
        # output channel is 1
        _test_convolution('conv_transpose', [1, 19, 8, 8], [1, 1, 1, 19], [1, 1], [1, 1], 'VALID',
                          'NCHW', [1, 1, 8, 8])

    _test_convolution('conv', [4, 8, 8, 176], [1, 1, 176, 32], [1, 1], [1, 1], 'SAME', 'NHWC')
    _test_convolution('conv', [4, 17, 17, 19], [3, 3, 19, 19], [1, 1], [2, 2], 'VALID', 'NHWC')
    _test_convolution('conv', [4, 17, 17, 124], [1, 1, 124, 19], [1, 1], [1, 1], 'SAME', 'NHWC')
    _test_convolution('conv', [4, 17, 17, 12], [3, 3, 12, 32], [1, 1], [2, 2], 'VALID', 'NHWC')
    _test_convolution('depthwise', [4, 8, 8, 176], [1, 1, 176, 1], [1, 1], [1, 1], 'SAME', 'NHWC')
    _test_convolution('depthwise', [4, 17, 17, 19], [3, 3, 19, 1], [1, 1], [2, 2], 'VALID', 'NHWC')
    _test_convolution('depthwise', [4, 17, 17, 124], [1, 1, 124, 1], [1, 1], [1, 1], 'SAME', 'NHWC')
    _test_convolution('depthwise', [4, 17, 17, 12], [3, 3, 12, 1], [1, 1], [2, 2], 'VALID', 'NHWC')
    _test_convolution('depthwise', [4, 17, 17, 12], [3, 3, 12, 2], [1, 1], [2, 2], 'VALID', 'NHWC')
    _test_convolution('conv_transpose', [4, 8, 8, 32], [1, 1, 176, 32], [1, 1], [1, 1], 'SAME',
                      'NHWC', [4, 8, 8, 176])
    _test_convolution('conv_transpose', [4, 8, 8, 32], [2, 2, 176, 32], [1, 1], [1, 1], 'SAME',
                      'NHWC', [4, 8, 8, 176])
    _test_convolution('conv_transpose', [4, 8, 8, 32], [2, 2, 176, 32], [1, 1], [2, 2], 'SAME',
                      'NHWC', [4, 15, 15, 176])
    _test_convolution('conv_transpose', [4, 8, 8, 32], [3, 3, 176, 32], [1, 1], [1, 1], 'SAME',
                      'NHWC', [4, 8, 8, 176])
    _test_convolution('conv_transpose', [4, 8, 8, 32], [3, 3, 176, 32], [1, 1], [2, 2], 'SAME',
                      'NHWC', [4, 15, 15, 176])
    _test_convolution('conv_transpose', [4, 8, 8, 32], [3, 3, 176, 32], [1, 1], [2, 2], 'SAME',
                      'NHWC', [4, 16, 16, 176])
    _test_convolution('conv_transpose', [4, 8, 8, 19], [3, 3, 19, 19], [1, 1], [2, 2], 'VALID',
                      'NHWC', [4, 17, 17, 19])
    _test_convolution('conv_transpose', [4, 17, 17, 19], [1, 1, 124, 19], [1, 1], [1, 1], 'SAME',
                      'NHWC', [4, 17, 17, 124])
    _test_convolution('conv_transpose', [4, 17, 17, 19], [3, 3, 124, 19], [1, 1], [1, 1], 'SAME',
                      'NHWC', [4, 17, 17, 124])
    _test_convolution('conv_transpose', [4, 8, 8, 32], [3, 3, 12, 32], [1, 1], [2, 2], 'VALID',
                      'NHWC', [4, 17, 17, 12])
    # kernel 2x2, strides (2,2)
    _test_convolution('conv_transpose', [4, 8, 8, 19], [2, 2, 19, 19], [1, 1], [2, 2], 'VALID',
                      'NHWC', [4, 16, 16, 19])
    _test_convolution('conv_transpose', [4, 8, 8, 32], [2, 2, 12, 32], [1, 1], [2, 2], 'VALID',
                      'NHWC', [4, 16, 16, 12])
    # output channel is 1
    _test_convolution('conv_transpose', [1, 8, 8, 19], [1, 1, 1, 19], [1, 1], [1, 1], 'VALID',
                      'NHWC', [1, 8, 8, 1])


#######################################################################
# Convolution3D
# -----------


def _test_convolution3d(opname, tensor_in_sizes, filter_in_sizes,
                        dilations, strides, padding, data_format,
                        deconv_output_shape=[]):
    """ One iteration of 3D convolution with given shapes and attributes """

    total_size_1 = np.prod(tensor_in_sizes)
    total_size_2 = np.prod(filter_in_sizes)
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    data_array = [f * 1.0 for f in range(1, total_size_1 + 1)]
    filter_array = [f * 1.0 for f in range(1, total_size_2 + 1)]

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=tensor_in_sizes, dtype='float32')
        in_filter = constant_op.constant(
            filter_array, shape=filter_in_sizes, dtype='float32')
        if data_format == 'NDHWC':
            strides = [1] + strides + [1]
            dilations = [1] + dilations + [1]
        else:
            strides = [1, 1] + strides
            dilations = [1, 1] + dilations

        if opname == 'conv':
            nn_ops.conv3d(in_data,
                          in_filter,
                          strides=strides,
                          dilations=dilations,
                          padding=padding,
                          data_format=data_format)

            compare_tf_with_tvm(np.reshape(data_array, tensor_in_sizes).astype('float32'),
                                'Placeholder:0', 'Conv3D:0', cuda_layout="NCDHW")

def test_forward_convolution3d():
    if is_gpu_available():
        _test_convolution3d('conv', [4, 176, 8, 8, 8], [1, 1, 1, 176, 32], [1, 1, 1], [1, 1, 1], 'SAME', 'NCDHW')
        _test_convolution3d('conv', [4, 19, 17, 17, 17], [3, 3, 3, 19, 19], [1, 1, 1], [2, 2, 2], 'VALID', 'NCDHW')
        _test_convolution3d('conv', [4, 124, 17, 17, 17], [1, 1, 1, 124, 19], [1, 1, 1], [1, 1, 1], 'SAME', 'NCDHW')
        _test_convolution3d('conv', [4, 12, 17, 17, 17], [3, 3, 3, 12, 32], [1, 1, 1], [2, 2, 2], 'VALID', 'NCDHW')
    _test_convolution3d('conv', [4, 8, 8, 8, 176], [1, 1, 1, 176, 32], [1, 1, 1], [1, 1, 1], 'SAME', 'NDHWC')
    _test_convolution3d('conv', [4, 17, 17, 17, 19], [3, 3, 3, 19, 19], [1, 1, 1], [2, 2, 2], 'VALID', 'NDHWC')
    _test_convolution3d('conv', [4, 17, 17, 17, 124], [1, 1, 1, 124, 19], [1, 1, 1], [1, 1, 1], 'SAME', 'NDHWC')
    _test_convolution3d('conv', [4, 17, 17, 17, 12], [3, 3, 3, 12, 32], [1, 1, 1], [2, 2, 2], 'VALID', 'NDHWC')


#######################################################################
# BiasAdd
# -----------


def _test_biasadd(tensor_in_sizes, data_format):
    """ One iteration of biasadd with given shapes and attributes """

    total_size_1 = 1
    for s in tensor_in_sizes:
        total_size_1 *= s
    tensor_bias_sizes = [tensor_in_sizes[1]
                         ] if data_format == 'NCHW' else [tensor_in_sizes[3]]
    total_size_2 = tensor_bias_sizes[0]
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    data_array = [f * 1.0 for f in range(1, total_size_1 + 1)]
    bias_array = [f * 1.0 for f in range(1, total_size_2 + 1)]

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=tensor_in_sizes, dtype='float32')
        in_bias = constant_op.constant(
            bias_array, shape=tensor_bias_sizes, dtype='float32')
        nn_ops.bias_add(in_data,
                        in_bias,
                        data_format=data_format)

        compare_tf_with_tvm(np.reshape(data_array, tensor_in_sizes).astype('float32'),
                            'Placeholder:0', 'BiasAdd:0')


def test_forward_biasadd():
    if is_gpu_available():
        _test_biasadd([4, 176, 8, 8], 'NCHW')
        _test_biasadd([1, 100, 1, 1], 'NCHW')
        _test_biasadd([4, 19, 17, 17], 'NCHW')
        _test_biasadd([4, 124, 3, 3], 'NCHW')

    _test_biasadd([4, 8, 8, 176], 'NHWC')
    _test_biasadd([1, 1, 1, 100], 'NHWC')
    _test_biasadd([4, 17, 17, 19], 'NHWC')
    _test_biasadd([4, 3, 3, 124], 'NHWC')


def _test_forward_where(input_shape):
    with tf.Graph().as_default():
        dtype = tf.float32
        t = tf.constant(np.random.choice([0, 1, -2, 3, -1, 0.1, -0.2],
                                         size=input_shape).astype(dtype.name))
        out = tf.where(t)
        compare_tf_with_tvm([], [], out.name, mode='debug')
        compare_tf_with_tvm([], [], out.name, mode='vm')


def test_forward_argwhere():
    _test_forward_where((5,))
    _test_forward_where((5, 5))
    _test_forward_where((5, 5, 5))
    _test_forward_where((5, 5, 5, 5))
    _test_forward_where((5, 5, 5, 5, 5))

#######################################################################
# SpaceToBatchND
# --------------


def _test_space_to_batch_nd(input_shape, block_shape, paddings, dtype='int32'):
    data = np.random.uniform(0, 5, size=input_shape).astype(dtype)

    with tf.Graph().as_default():
        in_data = tf.placeholder(shape=input_shape, dtype=dtype)
        out = tf.space_to_batch_nd(in_data, block_shape, paddings)

        compare_tf_with_tvm(data, in_data.name, out.name)

def _test_space_to_batch_nd_infer_paddings(input_shape, block_shape, dtype='int32'):
    data = np.random.uniform(0, 5, size=input_shape).astype(dtype)
    padding_np = np.array([0, 1]).astype(np.int32).reshape((1, 2))
    with tf.Graph().as_default():
        in_data = tf.placeholder(shape=input_shape, dtype=dtype)
        const1 = tf.constant(padding_np, dtype=tf.int32)
        # make paddings an input to tf.transpose, but not an input to the graph,
        # so it can be extracted with infer_value_simulated
        paddings = tf.reverse(const1, axis=[-1])
        out = tf.space_to_batch_nd(in_data, block_shape, paddings)
        compare_tf_with_tvm(data, in_data.name, out.name)

def test_forward_space_to_batch_nd():
    # test cases: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/space-to-batch-n-d
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
        paddings=[[0, 0], [2, 0]],
        dtype='int64'
    )

    # pylint: disable=line-too-long
    # https://github.com/tensorflow/tensorflow/blob/24f578/tensorflow/python/kernel_tests/spacetobatch_op_test.py
    _test_space_to_batch_nd(
        input_shape=[2, 3],
        block_shape=[2],
        paddings=[[1, 0]],
        dtype='float32'
    )

    _test_space_to_batch_nd(
        input_shape=[2, 3, 2],
        block_shape=[2],
        paddings=[[1, 0]],
        dtype='float64'
    )

    _test_space_to_batch_nd_infer_paddings(
        input_shape=[2, 3, 2],
        block_shape=[2]
    )

#######################################################################
# BatchToSpaceND
# --------------


def _test_batch_to_space_nd(input_shape, block_shape, crops, dtype='int32'):
    data = np.random.uniform(0, 5, size=input_shape).astype(dtype)

    with tf.Graph().as_default():
        in_data = tf.placeholder(shape=input_shape, dtype=dtype)
        out = tf.batch_to_space_nd(in_data, block_shape, crops)

        compare_tf_with_tvm(data, in_data.name, out.name)


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

    _test_batch_to_space_nd(
        input_shape=[8, 1, 3, 1],
        block_shape=[2, 2],
        crops=[[0, 0], [2, 0]],
        dtype='int64'
    )

    # pylint: disable=line-too-long
    # https://github.com/tensorflow/tensorflow/blob/24f578/tensorflow/python/kernel_tests/batchtospace_op_test.py
    _test_batch_to_space_nd(
        input_shape=[18, 2, 1, 2],
        block_shape=[2, 3],
        crops=[[1, 1], [0, 0]],
        dtype='float32'
    )

    _test_batch_to_space_nd(
        input_shape=[20, 5, 8, 7],
        block_shape=[2, 2],
        crops=[[1, 1], [1, 1]],
        dtype='float64'
    )

#######################################################################
# Reshape
# -------


def _test_reshape(data, out_shape):
    """ One iteration of reshape operation with given data and out shape """

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        array_ops.reshape(in_data, out_shape)

        compare_tf_with_tvm(data, 'Placeholder:0', 'Reshape:0')

def _test_reshape_with_call():
    """ relay.expr.Call as shape """
    data = np.zeros((6, 4, 2))
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out_shape = tf.constant([1, 2, 3], dtype="int32")
        out_shape = tf.multiply(out_shape, 2)
        array_ops.reshape(in_data, out_shape)

        compare_tf_with_tvm(data, 'Placeholder:0', 'Reshape:0')

def _test_reshape_like(data, shape_like):
    """ A special case for reshape. """

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        in_shape_like = array_ops.placeholder(shape=shape_like.shape, dtype=data.dtype)
        out_shape = array_ops.shape(in_shape_like)
        array_ops.reshape(in_data, out_shape)

        compare_tf_with_tvm(data, 'Placeholder:0', 'Reshape:0')

def test_forward_reshape():
    _test_reshape(np.arange(6.0), [2, 3])
    _test_reshape(np.arange(6), [-1, 2])
    _test_reshape(np.arange(6), [3, -1])
    _test_reshape(np.arange(6), [-1])
    _test_reshape_with_call()
    _test_reshape_like(np.zeros((3, 6)), np.zeros((9, 2)))

#######################################################################
# DepthToSpace
# ------------


def _test_depthtospace(data, block_size):
    """ One iteration of depth_to_space operation with given data and block size """

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        array_ops.depth_to_space(in_data, block_size)

        compare_tf_with_tvm(data, 'Placeholder:0', 'DepthToSpace:0')


def test_forward_depthtospace():
    _test_depthtospace(np.random.normal(size=[1, 32, 32, 4]), 2)
    _test_depthtospace(np.random.normal(size=[1, 16, 8, 32]), 4)

#######################################################################
# SpaceToDepth
# ------------


def _test_spacetodepth(data, block_size):
    """ One iteration of space_to_depth operation with given data and block size """

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        array_ops.space_to_depth(in_data, block_size)

        compare_tf_with_tvm(data, 'Placeholder:0', 'SpaceToDepth:0')


def test_forward_spacetodepth():
    _test_spacetodepth(np.random.normal(size=[1, 32, 32, 4]), 2)
    _test_spacetodepth(np.random.normal(size=[1, 16, 8, 32]), 4)

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
# TensorArray
# -----------
def test_tensor_array_write_read():
    def run(dtype_str, infer_shape, element_shape):
        with tf.Graph().as_default():
            dtype = tf_dtypes[dtype_str]
            np_data = np.array([[1.0, 2.0], [3.0, 4.0]]).astype(dtype_str)
            in_data = [np_data, np_data]
            t1 = tf.constant(np_data, dtype=dtype)
            t2 = tf.constant(np_data, dtype=dtype)
            ta1 = tf.TensorArray(dtype=dtype, size=2, infer_shape=infer_shape,
                                 element_shape=element_shape)
            ta2 = ta1.write(0, t1)
            ta3 = ta2.write(1, t2)
            out = ta3.read(0)
            g = tf.get_default_graph()
            compare_tf_with_tvm([], [], 'TensorArrayReadV3:0', mode='vm')

    for dtype in ["float32", "int8"]:
        run(dtype, False, None)
        run(dtype, False, tf.TensorShape([None, 2]))
        run(dtype, True, None)


def test_tensor_array_scatter():
    def run(dtype_str, infer_shape):
        with tf.Graph().as_default():
            dtype =  tf_dtypes[dtype_str]
            t = tf.constant(np.array([[1.0], [2.0], [3.0]]).astype(dtype_str), dtype=dtype)
            indices = tf.constant([2, 1, 0])
            ta1 = tf.TensorArray(dtype=dtype, size=3,
                                 infer_shape=infer_shape)
            ta2 = ta1.scatter(indices, t)
            out0 = ta2.read(0)
            out1 = ta2.read(1)
            out2 = ta2.read(2)
            g = tf.get_default_graph()
            compare_tf_with_tvm([], [], ['TensorArrayReadV3:0'], mode='vm')
            compare_tf_with_tvm([], [], ['TensorArrayReadV3_1:0'], mode='vm')
            compare_tf_with_tvm([], [], ['TensorArrayReadV3_2:0'], mode='vm')
    for dtype in ["float32", "int8"]:
        run(dtype, False)
        run(dtype, True)


def test_tensor_array_gather():
    def run(dtype_str, infer_shape):
        with tf.Graph().as_default():
            dtype =  tf_dtypes[dtype_str]
            t = tf.constant(np.array([[1.0], [2.0], [3.0]]).astype(dtype_str))
            scatter_indices = tf.constant([2, 1, 0])
            gather_indices = tf.constant([1, 2])
            ta1 = tf.TensorArray(dtype=dtype, size=3, infer_shape=infer_shape)
            ta2 = ta1.scatter(scatter_indices, t)
            t1 = ta2.gather(gather_indices)
            g = tf.get_default_graph()
            compare_tf_with_tvm([], [], ['TensorArrayGatherV3:0'], mode='vm')
    for dtype in ["float32", "int8"]:
        run(dtype, True)


def test_tensor_array_split():
    def run(dtype_str, infer_shape):
        with tf.Graph().as_default():
            dtype =  tf_dtypes[dtype_str]
            t = tf.constant(np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]).astype(dtype_str), dtype=dtype)
            split_length = tf.constant([2, 2, 2, 2], dtype=tf.int32)
            ta1 = tf.TensorArray(dtype=dtype, size=4, infer_shape=infer_shape)
            ta2 = ta1.split(t, split_length)
            out0 = ta2.read(0)
            out1 = ta2.read(1)
            out2 = ta2.read(2)
            out3 = ta2.read(3)
            g = tf.get_default_graph()
            compare_tf_with_tvm([], [], ['TensorArrayReadV3:0'], mode='debug')
            compare_tf_with_tvm([], [], ['TensorArrayReadV3_1:0'], mode='debug')
            compare_tf_with_tvm([], [], ['TensorArrayReadV3_2:0'], mode='debug')
            compare_tf_with_tvm([], [], ['TensorArrayReadV3_3:0'], mode='debug')
    for dtype in ["float32", "int8"]:
        run(dtype, False)
        run(dtype, True)


def test_tensor_array_concat():
    def run(dtype_str, infer_shape):
        with tf.Graph().as_default():
            dtype = tf_dtypes[dtype_str]
            t = tf.constant(np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]).astype(dtype_str), dtype=dtype)
            split_length = tf.constant([2, 2, 2, 2], dtype=tf.int32)
            ta1 = tf.TensorArray(dtype=dtype, size=4,
                                 infer_shape=infer_shape)
            ta2 = ta1.split(t, split_length)
            t = ta2.concat()
            out = tf.identity(t)
            compare_tf_with_tvm([], [], ['Identity:0'], mode='debug')
    for dtype in ["float32", "int8"]:
        run(dtype, False)
        run(dtype, True)


def test_tensor_array_size():
    def run(dtype_str, infer_shape):
        with tf.Graph().as_default():
            dtype =  tf_dtypes[dtype_str]
            ta1 = tf.TensorArray(dtype=dtype, size=2, infer_shape=infer_shape)
            out = ta1.size()
            g = tf.get_default_graph()
            compare_tf_with_tvm([], [], 'TensorArraySizeV3:0', mode='debug')
    for dtype in ["float32", "int8"]:
        run(dtype, False)
        run(dtype, True)


def test_tensor_array_stack():
    def run(dtype_str, infer_shape):
        with tf.Graph().as_default():
            dtype =  tf_dtypes[dtype_str]
            t = tf.constant(np.array([[1.0], [2.0], [3.0]]).astype(dtype_str))
            scatter_indices = tf.constant([2, 1, 0])
            ta1 = tf.TensorArray(dtype=dtype, size=3, infer_shape=infer_shape)
            ta2 = ta1.scatter(scatter_indices, t)
            t1 = ta2.stack()
            print(t1)
            g = tf.get_default_graph()

            compare_tf_with_tvm([], [], ['TensorArrayStack/TensorArrayGatherV3:0'], mode='vm')
    for dtype in ["float32", "int8"]:
        run(dtype, True)


def test_tensor_array_unstack():
    def run(dtype_str, input_shape, infer_shape):
        with tf.Graph().as_default():
            dtype = tf_dtypes[dtype_str]
            t = tf.constant(np.random.choice([0, 1, 2, 3],
                                             size=input_shape).astype(dtype.name))
            ta1 = tf.TensorArray(dtype=dtype, infer_shape=infer_shape, size=input_shape[0])
            ta2 = ta1.unstack(t)
            out0 = ta2.size()
            out1 = ta2.read(0)
            compare_tf_with_tvm([], [], 'TensorArraySizeV3:0', mode='debug')
            compare_tf_with_tvm([], [], 'TensorArrayReadV3:0', mode='debug')
    for dtype in ["float32", "int8"]:
        run(dtype, (5,), False)
        run(dtype, (5, 5), True)
        run(dtype, (5, 5, 5), False)
        run(dtype, (5, 5, 5, 5), True)


#######################################################################
# ConcatV2
# --------


def _test_concat_v2(shape1, shape2, dim):
    """ One iteration of ConcatV2 """

    with tf.Graph().as_default():
        dtype = 'float32'
        in1 = tf.placeholder(shape=shape1, dtype=dtype, name='in1')
        in2 = tf.placeholder(shape=shape2, dtype=dtype, name='in2')
        array_ops.concat_v2([in1, in2], dim)

        np_data1 = np.random.uniform(size=shape1).astype(dtype)
        np_data2 = np.random.uniform(size=shape2).astype(dtype)

        compare_tf_with_tvm([np_data1, np_data2], [
                            'in1:0', 'in2:0'], 'ConcatV2:0')


def test_forward_concat_v2():
    if tf.__version__ < LooseVersion('1.4.1'):
        return

    _test_concat_v2([2, 3], [2, 3], 0)
    _test_concat_v2([10, 3, 5], [2, 3, 5], 0)
    _test_concat_v2([2, 3], [2, 3], 1)
    _test_concat_v2([5, 8], [5, 4], 1)
    _test_concat_v2([2, 8, 5], [2, 8, 6], -1)

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
        inp = array_ops.placeholder(
            shape=data.shape, dtype=data.dtype, name="c0")
        func(inp, name="argx0", output_type=tf.int32, **kwargs)

        compare_tf_with_tvm(data, 'c0:0', 'argx0:0')


def test_forward_argminmax():
    for axis in [None, 0, 1, 2]:
        data = np.random.uniform(size=(8, 4, 9)).astype('float32')
        _test_argx(tf.argmax, data=data, axis=axis)
        _test_argx(tf.argmin, data=data, axis=axis)


#######################################################################
# Variable
# --------

def _test_variable(data):
    """ One iteration of a variable """

    tf.reset_default_graph()
    with tf.Graph().as_default():
        input_op = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        input_tensor = array_ops.reshape(input_op, data.shape)

        size = input_tensor.shape.dims[1]
        with variable_scope.variable_scope("linear", reuse=None):
            w = variable_scope.get_variable(
                "w", shape=[size, size], dtype=input_tensor.dtype)
        math_ops.matmul(input_tensor, w)

        compare_tf_with_tvm(data, 'Placeholder:0', 'MatMul:0',
                            init_global_variables=True)


def test_forward_variable():
    """Variable type op test"""
    _test_variable(np.random.uniform(size=(32, 100)).astype('float32'))


def test_read_variable_op():
    """ Read Variable op test """

    tf.reset_default_graph()
    data = np.random.uniform(size=(32, 100)).astype('float32')
    input_tensor = array_ops.placeholder(shape=data.shape, dtype=data.dtype)

    size = input_tensor.shape.dims[1]
    var_data = np.random.uniform(-5, 5, size=[size, size]).astype(np.float32)
    input_var = tf.Variable(var_data, name='var1', use_resource=True)
    math_ops.matmul(input_tensor, input_var)

    out_name = ['MatMul:0']
    out_node = ['MatMul']
    in_name = ['Placeholder:0']
    in_node = ['Placeholder']
    in_data = [data]

    with tf.Session() as sess:
        sess.run(variables.global_variables_initializer())

        final_graph_def = sess.graph.as_graph_def(add_shapes=True)
        tf_output = run_tf_graph(sess, in_data, in_name, out_name)

        shape_dict = {e: i.shape for e, i in zip(in_name, in_data)}
        with pytest.raises(Exception) as exexcinfo:
            mod, params = relay.frontend.from_tensorflow(final_graph_def,
                                                         layout=None,
                                                         shape=shape_dict,
                                                         outputs=None)

        assert exexcinfo.value.args[0].startswith("Graph is not frozen. Provide a frozen graph.")

        # Now convert the variables to constant and run inference on the converted graph
        final_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            sess.graph.as_graph_def(add_shapes=True),
            out_node,
        )

        for device in ["llvm", "cuda"]:
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                continue

            tvm_output = run_tvm_graph(final_graph_def, in_data, in_node,
                                       target=device, out_names=out_name,
                                       num_output=len(out_name))
            for i in range(len(tf_output)):
                tvm.testing.assert_allclose(
                    tf_output[i], tvm_output[i], atol=1e-4, rtol=1e-5)

        sess.close()


#######################################################################
# MatMul, BatchMatMul, BatchMatMulV2
# ----------------------------------

def _test_matmul(i, j, k, dtype, outer=None):
    """ One iteration of matmul """

    A_shape_init = [i, j]
    B_shape_init = [j, k]

    for transpose_a in [False, True]:
        for transpose_b in [False, True]:
            outer = outer or []
            A_shape = outer + \
                (A_shape_init[::-1] if transpose_a else A_shape_init)
            B_shape = outer + \
                (B_shape_init[::-1] if transpose_b else B_shape_init)

            with tf.Graph().as_default():
                A = tf.placeholder(shape=A_shape, dtype=dtype, name='A')
                B = tf.placeholder(shape=B_shape, dtype=dtype, name='B')
                result = tf.matmul(
                    A, B, transpose_a=transpose_a, transpose_b=transpose_b)

                A_np = np.random.uniform(high=5.0, size=A_shape).astype(dtype)
                B_np = np.random.uniform(high=5.0, size=B_shape).astype(dtype)
                compare_tf_with_tvm(
                    [A_np, B_np], [A.name, B.name], result.name)


def test_forward_matmul():
    """ MatMul op test"""
    _test_matmul(1, 3, 6, 'int32')
    _test_matmul(5, 3, 1, 'float64')


def _test_batch_matmul(A_shape, B_shape, dtype, adjoint_a=False, adjoint_b=False):

    with tf.Graph().as_default():
        A = tf.placeholder(shape=A_shape, dtype=dtype, name='A')
        B = tf.placeholder(shape=B_shape, dtype=dtype, name='B')
        result = tf.matmul(A, B, adjoint_a=adjoint_a,
                           adjoint_b=adjoint_b, name='batchmatmul')

        A_np = np.random.uniform(high=5.0, size=A_shape).astype(dtype)
        B_np = np.random.uniform(high=5.0, size=B_shape).astype(dtype)
        compare_tf_with_tvm([A_np, B_np], [A.name, B.name], result.name)


def test_forward_batch_matmul():
    """ TF op BatchMatMul, BatchMatMulV2 test"""
    _test_batch_matmul((3, 5, 4), (3, 4, 5), 'int32')
    _test_batch_matmul((3, 5, 4), (3, 4, 5), 'float32', True, True)
    _test_batch_matmul((3, 5, 4), (3, 5, 4), 'int32', True, False)
    _test_batch_matmul((3, 5, 4), (3, 5, 4), 'float32', False, True)
    _test_batch_matmul((2, 3, 4, 5, 6), (2, 3, 4, 6, 5), 'int32')
    _test_batch_matmul((1, 2, 3, 4, 5, 6),
                       (1, 2, 3, 4, 6, 5), 'float32', True, True)
    _test_batch_matmul((3, 4, 5, 6), (3, 4, 5, 6), 'int32', True, False)
    _test_batch_matmul((2, 3, 4, 2, 3, 4, 5, 6),
                       (2, 3, 4, 2, 3, 4, 5, 6), 'float32', False, True)


#######################################################################
# StridedSlice
# ------------

def _test_stridedslice(ip_shape, begin, end, stride, dtype,
                       begin_mask=0, end_mask=0, new_axis_mask=0,
                       shrink_axis_mask=0, ellipsis_mask=0):
    """ One iteration of a Stridedslice """

    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(dtype, ip_shape, name="in_data")
        tf.strided_slice(in_data, begin, end, stride, begin_mask=begin_mask,
                         end_mask=end_mask, new_axis_mask=new_axis_mask,
                         shrink_axis_mask=shrink_axis_mask,
                         ellipsis_mask=ellipsis_mask, name="strided_slice")
        np_data = np.random.uniform(size=ip_shape).astype(dtype)

        compare_tf_with_tvm(np_data, 'in_data:0', 'strided_slice:0')


def test_forward_stridedslice():
    '''test StridedSlice'''

    _test_stridedslice((2), [1], [1], [1], 'float32', shrink_axis_mask=1)
    _test_stridedslice((2, 1), [0], [1], [1], 'float32', shrink_axis_mask=1)
    _test_stridedslice((2, 3, 4), [0], [1], [1], 'float32', shrink_axis_mask=8)
    _test_stridedslice((3, 4, 3), [1, -1, 0],
                       [4, -5, 3], [2, -1, 1], 'float32')
    _test_stridedslice((3, 4, 3), [1, 0], [4, 3], [
                       2, 1], 'float32', ellipsis_mask=8)
    _test_stridedslice((3, 4, 3), [1, 0], [4, 2], [
                       2, 1], 'float32', ellipsis_mask=2)
    _test_stridedslice((3, 4, 5, 3), [1, 0], [4, 2], [
                       2, 1], 'float32', ellipsis_mask=2)
    _test_stridedslice((3, 4, 5, 3), [1, 0, 1], [4, 2, 2], [
                       2, 1, 1], 'float32', ellipsis_mask=2)
    _test_stridedslice((3, 4, 3), [1, 1, 0], [4, 4, 2], [
                       2, 1, 1], 'float32', new_axis_mask=5)
    _test_stridedslice((3, 4, 3), [1, 1, 1], [4, 4, 1], [2, 1, 1], 'float32', ellipsis_mask=2,
                       new_axis_mask=4)
    _test_stridedslice((6, 4, 5), [1, 1, 1], [6, 3, 4], [2, 1, 1], 'float32', ellipsis_mask=2,
                       new_axis_mask=5)
    _test_stridedslice((3, 4, 3), [1, 1, 2], [4, 4, 3], [2, 1, 1], 'float32', ellipsis_mask=4,
                       new_axis_mask=2)
    _test_stridedslice((3, 4, 3), [1, 1, 2], [4, 4, 3], [2, 1, 1], 'float32', ellipsis_mask=2,
                       new_axis_mask=3)
    _test_stridedslice((3, 4, 3), [1, 1, 0], [4, 4, 1], [2, 1, 1], 'float32', ellipsis_mask=2,
                       new_axis_mask=3)
    _test_stridedslice((3, 4, 3), [1, 1, 2], [4, 4, 3], [2, 1, 1], 'float32', ellipsis_mask=2,
                       new_axis_mask=2)
    _test_stridedslice((3, 4), [1, 0], [4, 4], [
                       1, 1], 'float32', shrink_axis_mask=2)
    _test_stridedslice((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], 'float32', shrink_axis_mask=2,
                       new_axis_mask=2)
    _test_stridedslice((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], 'float32', shrink_axis_mask=1,
                       new_axis_mask=2)
    _test_stridedslice((3, 4, 3), [1, 1, 0], [4, 4, 3], [2, 1, 1], 'float32', shrink_axis_mask=2,
                       new_axis_mask=1)
    _test_stridedslice((3, 4, 5, 4, 5, 6), [0, 0], [2, 3], [1, 1], 'float32', shrink_axis_mask=5,
                       new_axis_mask=1)
    _test_stridedslice((3, 4, 5, 4, 5, 6), [0, 0, 1, 2, 1], [2, 3, 4, 5, 3], [1, 1, 2, 2, 1],
                       'float32', shrink_axis_mask=5, new_axis_mask=1, ellipsis_mask=2,
                       begin_mask=8, end_mask=8)
    _test_stridedslice((3, 4, 5, 4, 5, 6), [0, 0, 1, 2, 1], [2, 3, 4, 5, 3], [1, 1, 2, 2, 1],
                       'float32', shrink_axis_mask=8, new_axis_mask=1, ellipsis_mask=2,
                       begin_mask=5, end_mask=5)
    _test_stridedslice((3, 4, 5, 4, 5, 6), [0, 0, 1, 2, 1], [2, 3, 4, 5, 3], [1, 1, 2, 2, 1],
                       'float32', shrink_axis_mask=16, new_axis_mask=1, ellipsis_mask=2,
                       begin_mask=5, end_mask=5)
    _test_stridedslice((3, 4, 5, 4, 5, 6), [1, 2, 0, -3], [4, 5, 3, 3], [2, 2, 1, 1],
                       'float32', shrink_axis_mask=8, new_axis_mask=1, ellipsis_mask=2,
                       begin_mask=5, end_mask=8)

#######################################################################
# FloorDiv, RealDiv
# -----------------
def _test_forward_divide(ip_shape, dtype):
    np_numer = np.random.uniform(-100, 100, size=ip_shape).astype(dtype)
    np_denomin = np.random.uniform(1, 100, size=ip_shape).astype(dtype)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        numerator = tf.placeholder(dtype, ip_shape, name="numer")
        denominator = tf.placeholder(dtype, ip_shape, name="denomin")
        tf.math.divide(numerator, denominator, name='RealDiv')
        compare_tf_with_tvm([np_numer, np_denomin], [
                            'numer:0', 'denomin:0'], 'RealDiv:0')


def _test_forward_floordiv(ip_shape, dtype):
    np_numer = np.random.uniform(1, 100, size=ip_shape).astype(dtype)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        numerator = tf.placeholder(dtype, ip_shape, name="numer")
        tf.math.floordiv(numerator, tf.constant(5, dtype=dtype), name='FloorDiv')
        compare_tf_with_tvm([np_numer], ['numer:0'], 'FloorDiv:0')


def test_forward_divide():
    '''test FloorDiv, RealDiv'''
    _test_forward_divide((4,), 'int32')
    _test_forward_divide((4, 3, 7), 'float32')
    _test_forward_floordiv((4, 3, 7), 'float32')
    _test_forward_floordiv((4, 3, 7), 'int32')

#######################################################################
# FloorMod
# --------
def _test_forward_floormod(in_shape, if_shape, dtype):
    np_numer = np.random.uniform(1, 100, size=in_shape).astype(dtype)
    np_factor = np.random.uniform(1, 100, size=if_shape).astype(dtype)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        numerator = tf.placeholder(dtype, in_shape, name="numer")
        factor = tf.placeholder(dtype, if_shape, name="factor")
        tf.floormod(numerator, factor, name='FloorMod')
        compare_tf_with_tvm([np_numer, np_factor], ['numer:0', 'factor:0'], 'FloorMod:0')

def test_forward_floormod():
    '''test FloorMod'''
    _test_forward_floormod((10,), (10,), 'float32')
    _test_forward_floormod((8, 2), (1,), 'float32')
    _test_forward_floormod((4, 3, 7), (4, 3, 7), 'float32')
    _test_forward_floormod((4, 3, 7), (4, 3, 7), 'int32')


#######################################################################
# TruncateMod
# -----------
def _test_forward_truncatemod(ip_shape, dtype):
    np_data_1 = np.random.uniform(-100, 100, size=ip_shape).astype(dtype)
    np_data_2 = np.random.uniform(1, 10, size=ip_shape).astype(dtype)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data_1 = tf.placeholder(dtype, ip_shape, name="in_data_1")
        in_data_2 = tf.placeholder(dtype, ip_shape, name="in_data_2")
        tf.truncatemod(in_data_1, in_data_2, name='truncatemod')
        compare_tf_with_tvm([np_data_1, np_data_2], [
                            'in_data_1:0', 'in_data_2:0'], 'truncatemod:0')


def test_forward_truncatemod():
    '''test TruncateMod'''
    _test_forward_truncatemod((4, 3, 7), 'int32')


#######################################################################
# Gather, GatherV2, GatherNd
# --------------------------

def _test_gather(ip_shape, indice_shape, indice_value, axis, dtype):
    """ One iteration of a GatherV2 """

    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(dtype, ip_shape, name="in_data")
        indices = tf.placeholder("int32", indice_shape, name="indices")
        out = tf.gather(in_data, indices, axis=axis)
        np_data = np.random.uniform(1, 10, size=ip_shape).astype(dtype)

        def _fill_indices(indice_value):
            indices = np.array(ip_shape, dtype=dtype)
            if isinstance(indice_value, int):
                indices = np.array([indice_value], dtype='int32')
            else:
                indices = np.asarray(indice_value, dtype='int32')
            return indices
        np_indices = _fill_indices(indice_value)

        compare_tf_with_tvm([np_data, np_indices], [
                            'in_data:0', 'indices:0'], out.name)


def test_forward_gather():
    '''test Gather/GatherV2 layer'''
    _test_gather((4,), (1,), 1, 0, 'int32')
    _test_gather((4,), (1,), 1, 0, 'float32')
    _test_gather((1, 4), (1,), [0], 0, 'int32')
    _test_gather((4,), (1, 2, 2), [[[1, 0], [0, 1]]], 0, 'float32')
    _test_gather((2, 2), (1, 2, 2), [[[1, 0], [0, 1]]], 0, 'int32')
    _test_gather((2, 2), (1, 2, 2), [[[1, 0], [0, 1]]], 1, 'int32')
    _test_gather((2, 2), (1, 2, 2), [[[1, 0], [0, 1]]], 0, 'float32')
    _test_gather((3, 3, 3), (1, 1, 2), [[[1, 0]]], 0, 'int32')
    _test_gather((3, 3, 3), (1, 1, 2), [[[1, 0]]], 2, 'int32')
    _test_gather((4, 3, 5, 6), (1, 4), [[2, 1, 0, 0]], 0, 'float32')


def test_forward_gather_nd():
    """test operator GatherNd"""
    np_data = np.random.uniform(1, 100, size=(2, 2, 2)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (2, 2, 2), name="in_data")
        tf.gather_nd(in_data, indices=[[1, 0, 0], [0, 0, 0]], name="gather_nd")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'gather_nd:0')


#######################################################################
# BiasAdd
# -------
def test_forward_bias_add():
    """test Op BiasAdd"""
    def check_bias_add(lh_shpae, rh_shape, dtype):
        tf.reset_default_graph()
        lh_data = np.random.uniform(size=lh_shpae).astype(dtype)
        rh_data = np.random.uniform(size=rh_shape).astype(dtype)
        with tf.Graph().as_default():
            lft_data = tf.placeholder(dtype, name="lft_data")
            rgt_data = tf.placeholder(dtype, name="rgt_data")
            tf.nn.bias_add(lft_data, rgt_data, name="BiasAdd")
            compare_tf_with_tvm([lh_data, rh_data], [
                                'lft_data:0', 'rgt_data:0'], 'BiasAdd:0')

    check_bias_add((10, 8, 16, 32), (32,), dtype="int32")
    check_bias_add((10, 20), (20,), dtype="float32")


#######################################################################
# Split
# -----

def _test_split(in_shape, axis, num_or_size_splits, dtype):
    np_data = np.random.uniform(-5, 5, size=in_shape).astype(dtype)

    """ One iteration of a Split """
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(dtype, in_shape, name="in_data")
        num_split = len(num_or_size_splits) if isinstance(num_or_size_splits, list)\
            else num_or_size_splits
        split = tf.split(in_data, num_or_size_splits, axis=axis)
        relu = [tf.nn.relu(i) for i in split]

        compare_tf_with_tvm([np_data], ['in_data:0'], [n.name for n in relu])

    # and now test together with concat
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(dtype, in_shape, name="in_data")
        splitted = tf.split(in_data, num_or_size_splits, axis=axis)
        concat = tf.concat(splitted, axis)
        compare_tf_with_tvm([np_data], 'in_data:0', concat.name)


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


######################################################################
# TopKV2
# ------

def _test_forward_top_k_v2(in_shape, k):
    np_data = np.random.uniform(-100, 100, size=in_shape).astype("float32")
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder("float32", in_shape, name="in_data")
        tf.math.top_k(in_data, k, name='TopK')
        compare_tf_with_tvm([np_data], ['in_data:0'], 'TopK:0')


def test_forward_top_k_v2():
    _test_forward_top_k_v2((3,), 1)
    _test_forward_top_k_v2((3,), 3)
    _test_forward_top_k_v2((3, 5, 7), 3)
    _test_forward_top_k_v2((3, 5, 7), 3)


#######################################################################
# Unstack
# -------

def _test_unstack(ip_shape, axis, dtype):
    np_data = np.random.uniform(-5, 5, size=ip_shape).astype(dtype)

    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(dtype, ip_shape, name="in_data")
        unstack = tf.unstack(in_data, axis=axis)

        compare_tf_with_tvm([np_data], ['in_data:0'], [n.name for n in unstack])

    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(dtype, ip_shape, name="in_data")
        tf.stack(tf.unstack(in_data, axis=axis), axis=axis)

        compare_tf_with_tvm([np_data], ['in_data:0'], 'stack:0')


def test_forward_unstack():
    '''test unstack layer'''
    _test_unstack((6,), 0, 'int32')
    _test_unstack((2, 6), 1, 'float64')
    # negative axis
    _test_unstack((1, 4), -1, 'int32')
    _test_unstack((3, 6, 4), -2, 'float32')


#######################################################################
# Tile
# ----

def _test_tile(in_shape, multiples, dtype):
    np_data = np.random.uniform(-5, 5, size=in_shape).astype(dtype)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(dtype, in_shape, name="in_data")
        tf.tile(in_data, multiples=multiples, name="tile")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'tile:0')


def test_forward_tile():
    '''test Tile'''
    _test_tile((2, ), (3, ), "int32")
    _test_tile((2, 2), (2, 3), "float32")
    _test_tile((2, 4, 6), (6, 7, 8), "float64")


#######################################################################
# ClipByValue
# -----------

def _test_forward_clip_by_value(ip_shape, clip_value_min, clip_value_max, dtype):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(dtype, ip_shape, name="in_data")
        tf.clip_by_value(in_data, clip_value_min,
                         clip_value_max, name="ClipByValue")
        np_data = np.random.uniform(-100, 100, size=ip_shape).astype(dtype)
        compare_tf_with_tvm([np_data], ['in_data:0'], 'ClipByValue:0')


def test_forward_clip_by_value():
    '''test ClipByValue op'''
    if tf.__version__ < LooseVersion('1.9'):
        _test_forward_clip_by_value((4,), .1, 5., 'float32')
        _test_forward_clip_by_value((4, 4), 1, 5, 'int32')

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
                tvm.testing.assert_allclose(
                    tf_output[i], tvm_output[i], atol=1e-5, rtol=1e-5)

#######################################################################
# Resize Bilinear, Nearest_Neighbor
# ---------------------------------


def _test_resize_bilinear(in_shape, to_shape, align_corners):
    """ One iteration of resize bilinear """

    data = np.random.uniform(size=in_shape).astype('float32')
    shape_data = np.array(to_shape).astype('int32')

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        shape_data = constant_op.constant(
            shape_data, shape=shape_data.shape, dtype=shape_data.dtype)
        tf.image.resize_bilinear(
            in_data, shape_data, align_corners=align_corners)

        compare_tf_with_tvm(data, 'Placeholder:0', 'ResizeBilinear:0')


def _test_resize_bilinear_from_tensor(in_shape, align_corners):
    """ One iteration of resize bilinear with non-constant output shape, requires
        value inference to get proper output shape."""

    data = np.random.uniform(size=in_shape).astype('float32')

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(
            shape=[in_shape[0], None, None, in_shape[3]], dtype=data.dtype)
        to_shape = tf.shape(in_data)[1:3]
        tf.image.resize_bilinear(
            in_data, to_shape, align_corners=align_corners)

        compare_tf_with_tvm(data, 'Placeholder:0', 'ResizeBilinear:0')


def _test_resize_nearest_neighbor(in_shape, to_shape):
    """ One iteration of resize nearest neighbor """

    data = np.random.uniform(size=in_shape).astype('float32')
    shape_data = np.array(to_shape).astype('int32')

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        shape_data = constant_op.constant(
            shape_data, shape=shape_data.shape, dtype=shape_data.dtype)
        tf.image.resize_nearest_neighbor(
            in_data, shape_data, name='resize_nearest_neighbor')

        compare_tf_with_tvm(data, 'Placeholder:0', 'resize_nearest_neighbor:0')


def _test_resize_nearest_neighbor_dynamic_shape(in_shape, scale):
    """ One iteration of resize nearest neighbor for graph with dynamic input shape """

    data = np.random.uniform(size=in_shape).astype('float32')
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=None, dtype=data.dtype)
        # multiply input shape by scale factor
        new_shape = tf.shape(in_data)[1:3] * tf.constant(scale, dtype=tf.int32)
        tf.image.resize_nearest_neighbor(
            in_data, new_shape, name='resize_nearest_neighbor')

        compare_tf_with_tvm(data, 'Placeholder:0', 'resize_nearest_neighbor:0')


def test_forward_resize():
    """ Resize Bilinear, Nearest_Neighbor """
    # TF default layout is NHWC
    _test_resize_bilinear((4, 32, 32, 3), [50, 50], False)
    _test_resize_bilinear((6, 32, 32, 3), [20, 20], True)
    _test_resize_bilinear_from_tensor((4, 32, 32, 3), False)
    _test_resize_bilinear_from_tensor((6, 50, 50, 3), True)
    _test_resize_nearest_neighbor((6, 32, 32, 3), [20, 20])
    _test_resize_nearest_neighbor_dynamic_shape((1, 16, 16, 3), scale=[2, 2])


#######################################################################
# BroadcastTo
# -----------

def _test_broadcast_to(in_shape, to_shape):
    """ One iteration of broadcast_to"""

    data = np.random.uniform(size=in_shape).astype('float32')
    shape_data = np.array(to_shape).astype('int32')

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        shape_data = constant_op.constant(
            shape_data, shape=shape_data.shape, dtype=shape_data.dtype)
        tf.broadcast_to(in_data, shape_data)

        compare_tf_with_tvm(data, 'Placeholder:0',
                            'BroadcastTo:0', opt_level=0)


def _test_broadcast_to_from_tensor(in_shape):
    """ One iteration of broadcast_to with unknown shape at graph build"""

    data = np.random.uniform(size=in_shape).astype('float32')

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(
            shape=[None], dtype=data.dtype)

        shape_data = tf.multiply(tf.shape(in_data), 32)
        tf.broadcast_to(in_data, shape_data)

        compare_tf_with_tvm(data, 'Placeholder:0', 'BroadcastTo:0')


def test_forward_broadcast_to():
    """ Resize Bilinear """

    _test_broadcast_to((4, 1, 32, 32), [4, 8, 32, 32])
    _test_broadcast_to((6, 32, 32, 1), [6, 32, 32, 16])
    _test_broadcast_to_from_tensor((1))


#######################################################################
# Fill
# ----

def _test_fill(in_shape):
    """ Use the fill op to create a tensor of ones with non-constant shape."""

    with tf.Graph().as_default():
        tf.ones(shape=in_shape, dtype='float32')
        compare_tf_with_tvm(in_shape, [], 'ones:0', opt_level=1)


def _test_fill_from_tensor(in_shape):
    """ Use the fill op to create a tensor of ones with non-constant shape.
        Some extra ops need to be added here to prevent the graph from
        being fully constant and folded away."""

    data = np.random.uniform(size=in_shape).astype('float32')

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(
            shape=[in_shape[0], in_shape[1], None, None], dtype=data.dtype)

        x = tf.ones(shape=2*tf.shape(in_data), dtype=data.dtype)
        y = tf.math.add(in_data, tf.reduce_mean(x), name='out1')
        compare_tf_with_tvm(data, 'Placeholder:0', 'out1:0')


def test_forward_fill():
    """ Resize Bilinear """

    _test_fill((32))
    _test_fill((6, 32, 64, 64))
    _test_fill_from_tensor((6, 32, 64, 64))

#######################################################################
# Crop to bounding box
# --------------------


def _test_crop(in_shape, off_h, off_w, tar_h, tar_w):
    """ Crop to bounding box """
    data = np.random.uniform(size=in_shape).astype('float32')
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        tf.image.crop_to_bounding_box(in_data, off_h, off_w, tar_h, tar_w)
        compare_tf_with_tvm(data, 'Placeholder:0',
                            'crop_to_bounding_box/Slice:0')


def test_forward_crop():
    """ Crop to bounding box """
    _test_crop((1, 224, 224, 3), 20, 20, 120, 120)


#######################################################################
# CropAndResize
# -------------

def _test_forward_crop_and_resize(img_shape, boxes, box_idx, crop_size,
                                  extrapolation_value=0.0, method='bilinear', dtype="float32"):
    image = np.random.uniform(0, 10, size=img_shape).astype(dtype)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(dtype, image.shape, name="in_data")
        tf.image.crop_and_resize(in_data, boxes=boxes, box_ind=box_idx,
                                 crop_size=crop_size, method=method,
                                 extrapolation_value=extrapolation_value,
                                 name="crop_and_resize")
        compare_tf_with_tvm([image], ['in_data:0'], 'crop_and_resize:0')


def test_forward_crop_and_resize():
    """ CropAndResize """
    _test_forward_crop_and_resize([1, 6, 6, 3], [[0, 0, 1, 1]], [0], [3, 3])
    _test_forward_crop_and_resize([1, 6, 6, 3], [[0, 0, 1, 1]], [0], [3, 3], 0.2)
    _test_forward_crop_and_resize([1, 6, 6, 3], [[0, 0, 1, 1]], [0], [3, 3], 0.2, 'nearest')
    _test_forward_crop_and_resize([1, 11, 11, 3], [[.3, .3,  1,  1]], [0], [21, 21])
    _test_forward_crop_and_resize([1, 41, 41, 3], [[.2, .4, .8, .8]], [0], [21, 11])
    _test_forward_crop_and_resize([1, 100, 100, 3], [[ 0,  0, .9, .9]], [0], [30, 30])
    _test_forward_crop_and_resize([1, 224, 224, 3], [[.1, .2,  1,  1]], [0], [9, 9])
    _test_forward_crop_and_resize([1, 249, 249, 3], [[ 0,  0,  1,  1]], [0], [9, 9])
    _test_forward_crop_and_resize([1, 201, 301, 3], [[.2, .3, .7, .8]], [0], [51, 51])
    _test_forward_crop_and_resize(img_shape=[10, 11, 11, 3],
                                  boxes=[[ 0,  0, .9, .9],
                                         [.2, .2, .8, .8]],
                                  box_idx=[0, 1], crop_size=[5, 5])
    _test_forward_crop_and_resize(img_shape=[20, 576, 576, 3],
                                  boxes=[[ 0,  0,  1,  1],
                                         [ 0,  0, .8, .8],
                                         [.1, .2, .9,  1],
                                         [.2,  0,  1,  1]],
                                  box_idx=[1, 0, 2, 3], crop_size=[24, 24],
                                  extrapolation_value=0.3)
    _test_forward_crop_and_resize(img_shape=[20, 229, 229, 3],
                                  boxes=[[ 0,  0, .9, .9],
                                         [.3, .3,  1,  1],
                                         [.2, .1, .7, .8],
                                         [ 0,  0,  1,  1]],
                                  box_idx=[3, 0, 2, 1], crop_size=[58, 58],
                                  extrapolation_value=0.2, method='nearest')


#######################################################################
# LSTM
# ----

def _test_lstm_cell(batch_size, num_hidden, num_layers, forget_bias, dtype):
    """ One iteration of a LSTM cell """

    tf.reset_default_graph()
    input_size = num_hidden
    input_data = np.full((batch_size, input_size), 1., dtype=dtype)
    in_state_c = np.full(
        (num_layers, batch_size, num_hidden), 0.1, dtype=dtype)
    in_state_h = np.full(
        (num_layers, batch_size, num_hidden), 0.1, dtype=dtype)

    def _get_tensorflow_output():
        with tf.Session() as sess:
            with variable_scope.variable_scope(
                    "root", initializer=init_ops.constant_initializer(0.5)):
                m0 = array_ops.zeros([batch_size, num_hidden])
                m1 = array_ops.zeros([batch_size, num_hidden])
                x = tf.placeholder(shape=(batch_size, input_size), dtype=dtype)
                g, ((out_m0, out_m1)) = \
                    tensorflow.contrib.rnn.LSTMBlockCell(num_hidden,
                                                         forget_bias=forget_bias)(x, (m0, m1))
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
    if package_version.parse(tf.VERSION) < package_version.parse('2.0.0'):
        #in 2.0, tf.contrib.rnn.LSTMBlockCell is removed
        _test_lstm_cell(1, 2, 1, 0.5, 'float32')


#######################################################################
# Pack
# ---
def _test_pack(axis, shape, **kwargs):

    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    b = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    with tf.Graph().as_default():
        tf_a = array_ops.placeholder(shape=shape, dtype='float32', name='pl_a')
        tf_b = array_ops.placeholder(shape=shape, dtype='float32', name='pl_b')
        tf_c = tf.stack([tf_a, tf_b], axis=axis, **kwargs)
        assert tf_c.op.op_def.name == 'Pack', "tf.stack() is expected to produce 'Pack' operation"

        compare_tf_with_tvm([a, b], ['pl_a:0', 'pl_b:0'], 'stack:0')


def test_forward_pack():
    for axis in range(-3, 3):
        _test_pack(axis, [3, 2, 1])
    for axis in range(-1, 1):
        _test_pack(axis, [3])
    _test_pack(0, [])


#######################################################################
# Unpack
# ------
def _test_forward_unpack(in_shape, axis, dtype):
    """test operator Unpack"""
    np_data = np.random.uniform(-100, 100, size=in_shape).astype(dtype)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(dtype, in_shape, name="in_data")
        tf.unstack(in_data, axis=axis, name="Unpack")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'Unpack:0')


def test_forward_unpack():
    _test_forward_unpack((3,), 0, 'int32')
    _test_forward_unpack((3,), -1, 'int16')
    _test_forward_unpack((21, 23, 3), 2, 'float32')

#######################################################################
# Range
# -----


def test_forward_range():
    """test operator Range"""
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.range(1, 18, 3, name="range")
        compare_tf_with_tvm([], [], 'range:0')

    """test type assignment for operator Range"""
    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.range(1, 256 + 1, 1, dtype=tf.float32)
        compare_tf_with_tvm([], [], 'range:0')

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
        else:
            out_name = 'MirrorPad:0'

        compare_tf_with_tvm(x, 'Placeholder:0', out_name)


def test_forward_pad():
    """ Pad """
    _test_pad((2, 3), [[1, 1], [2, 2]], mode="CONSTANT")
    _test_pad((2, 3), [[1, 1], [2, 2]], mode="CONSTANT", constant_values=1.0)
    _test_pad((2, 3), [[1, 1], [2, 2]], mode="SYMMETRIC")
    _test_pad((2, 3), [[1, 1], [2, 2]], mode="REFLECT")

#######################################################################
# Logical operators
# --------------------


def test_logical_and():
    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name='in1')
        in2 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name='in2')
        out = tf.logical_and(in1, in2, name='out')
        in_data1 = np.random.choice(
            a=[False, True], size=(1, 4, 4, 3)).astype('bool')
        in_data2 = np.random.choice(
            a=[False, True], size=(1, 4, 4, 3)).astype('bool')
        compare_tf_with_tvm([in_data1, in_data2], ['in1:0', 'in2:0'], 'out:0')


def test_logical_or():
    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name='in1')
        in2 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name='in2')
        out = tf.logical_or(in1, in2, name='out')
        in_data1 = np.random.choice(
            a=[False, True], size=(1, 4, 4, 3)).astype('bool')
        in_data2 = np.random.choice(
            a=[False, True], size=(1, 4, 4, 3)).astype('bool')
        compare_tf_with_tvm([in_data1, in_data2], ['in1:0', 'in2:0'], 'out:0')


def test_logical_xor():
    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name='in1')
        in2 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name='in2')
        out = tf.logical_xor(in1, in2, name='out')
        in_data1 = np.random.choice(
            a=[False, True], size=(1, 4, 4, 3)).astype('bool')
        in_data2 = np.random.choice(
            a=[False, True], size=(1, 4, 4, 3)).astype('bool')
        compare_tf_with_tvm([in_data1, in_data2], ['in1:0', 'in2:0'], 'out:0')


def test_logical_not():
    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name='in1')
        out = tf.logical_not(in1, name='out')
        in_data1 = np.random.choice(
            a=[False, True], size=(1, 4, 4, 3)).astype('bool')
        compare_tf_with_tvm(in_data1, 'in1:0', 'out:0')


def test_forward_logical():
    test_logical_and()
    test_logical_or()
    test_logical_xor()
    test_logical_not()


#######################################################################
# Where, Select
# -------------
def test_forward_where():
    ''' Where: return elements depending on conditions'''
    with tf.Graph().as_default():
        with tf.Session() as sess:
            input1 = tf.placeholder(
                tf.int32, shape=[1, 4, 4, 3], name='input1')
            input2 = tf.placeholder(
                tf.int32, shape=[1, 4, 4, 3], name='input2')
            mask = input1 > input2
            tf.where(mask, input1 + 1, input2 * 2)
            in_data1 = np.random.uniform(
                0, 10, size=(1, 4, 4, 3)).astype("uint32")
            in_data2 = np.random.uniform(
                0, 10, size=(1, 4, 4, 3)).astype("uint32")
            compare_tf_with_tvm([in_data1, in_data2], [
                                'input1:0', 'input2:0'], 'Select:0')


#######################################################################
# Inception V3
# ------------
def test_forward_inception_v3():
    '''test inception V3 model'''
    with tf.Graph().as_default():
        graph_def = tf_testing.get_workload(
            'InceptionV3/inception_v3_2016_08_28_frozen-with_shapes.pb')
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        data = np.random.uniform(size=(1, 299, 299, 3)).astype('float32')

        with tf.Session() as sess:
            tf_output = run_tf_graph(
                sess, data, 'input:0', 'InceptionV3/Predictions/Reshape_1:0')
            tvm_output = run_tvm_graph(graph_def, data, 'input')
            tvm.testing.assert_allclose(
                tf_output[0], tvm_output[0], rtol=1e-5, atol=1e-5)

#######################################################################
# Inception V1
# ------------


def test_forward_inception_v1():
    '''test inception V1 model'''
    with tf.Graph().as_default():
        graph_def = tf_testing.get_workload(
            "InceptionV1/classify_image_graph_def-with_shapes.pb")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)

        # Build an image from random data.
        from PIL import Image
        from tvm.contrib import util

        img_array = np.random.uniform(size=(1, 600, 600, 3)).astype("uint8")
        img = Image.frombuffer(
            'RGB', (600, 600), img_array.tostring(), 'raw', 'RGB', 0, 1)
        temp = util.tempdir()
        img_path = temp.relpath("tf-test.jpg")
        img.save(img_path)

        import os.path
        if not tf.gfile.Exists(os.path.join(img_path)):
            tf.logging.fatal('File does not exist %s', img_path)
        data = tf.gfile.FastGFile(os.path.join(img_path), 'rb').read()

        temp.remove()

        # Extract tensorflow decoded image frame for tvm input
        with tf.Session() as sess:
            tvm_data = run_tf_graph(
                sess, data, 'DecodeJpeg/contents:0', 'DecodeJpeg:0')

        with tf.Session() as sess:
            tf_output = run_tf_graph(
                sess, data, 'DecodeJpeg/contents:0', 'softmax:0')
            tvm_output = run_tvm_graph(
                graph_def, tvm_data, 'DecodeJpeg/contents')
            tvm.testing.assert_allclose(
                tf_output[0], tvm_output[0], rtol=1e-5, atol=1e-5)

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
            tvm.testing.assert_allclose(np.squeeze(tvm_output[0]), np.squeeze(tf_output[0]),
                                        rtol=1e-5, atol=1e-5)

#######################################################################
# ResnetV2
# --------


def test_forward_resnetv2():
    '''test resnet model'''
    if is_gpu_available():
        with tf.Graph().as_default():
            graph_def = tf_testing.get_workload(
                "ResnetV2/resnet-20180601_resnet_v2_imagenet-shapes.pb")
            # Call the utility to import the graph definition into default graph.
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)

            data = np.random.uniform(size=(128, 224, 224, 3)).astype('float32')
            out_node = 'ArgMax'

            with tf.Session() as sess:
                tf_output = run_tf_graph(
                    sess, data, 'input_tensor:0', out_node + ':0')
                for device in ["llvm", "cuda"]:
                    ctx = tvm.context(device, 0)
                    if not ctx.exist:
                        print("Skip because %s is not enabled" % device)
                        continue
                    tvm_output = run_tvm_graph(graph_def, data, 'input_tensor', len(tf_output),
                                               target=device)
                    tvm.testing.assert_allclose(np.squeeze(tvm_output[0]), np.squeeze(tf_output[0]),
                                                rtol=1e-5, atol=1e-5)

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
            tf_output = run_tf_graph(
                sess, data, 'Placeholder:0', out_node + ':0')
            tvm_output = run_tvm_graph(graph_def, data, 'Placeholder')
            tvm.testing.assert_allclose(np.squeeze(tvm_output[0]), np.squeeze(tf_output[0]),
                                        rtol=1e-5, atol=1e-5)


#######################################################################
# PTB
# ---
try:
    #Load contrib for running ptb model in tf version before 2.0
    import tensorflow.contrib
except:
    pass

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
    # Sample input
    inpt = "we have no useful information on"
    cnt_sample = 20

    def _pretty_print(items, is_char_model, id2word):
        if not is_char_model:
            return ' '.join([id2word[x] for x in items])
        else:
            return ''.join([id2word[x] for x in items]).replace('_', ' ')

    def _get_tvm_graph_module(graph_def):
        # Cell inputs 'c and 'h' consist of all layers values
        shape_dict = {'Model/Placeholder': (batch_size, num_steps),
                      'Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_c':
                      (num_layers, batch_size, num_hidden),
                      'Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_h':
                      (num_layers, batch_size, num_hidden)}

        mod, params = relay.frontend.from_tensorflow(
            graph_def, shape=shape_dict)

        dtype_dict = {'Model/Placeholder': 'int32',
                      'Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_c': 'float32',
                      'Model/RNN/RNN/multi_rnn_cell/cell_0/lstm_cell/LSTMBlockCell_h': 'float32'}
        target = 'llvm'
        with relay.build_config(opt_level=0):
            graph, lib, params = relay.build(mod,
                                             target,
                                             params=params)
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
            in_state_c = np.reshape(
                in_state_tup[0], (num_layers, batch_size, num_hidden))
            in_state_h = np.reshape(
                in_state_tup[1], (num_layers, batch_size, num_hidden))

            model.set_input('Model/Placeholder',
                            tvm.nd.array(input_data.astype("int32")))
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

    # TVM graph module creation
    params, m = _get_tvm_graph_module(graph_def)

    # Create 10 predicted statments of 20 words
    cnt_stm = 0
    while cnt_stm < 10:
        cnt_stm += 1
        in_state = np.full(
            (num_layers, 2, batch_size, num_hidden), 0, dtype="float32")
        seed_for_sample = inpt.split()
        tvm_samples, tvm_state = _do_tvm_sample(m, [word_to_id[word]
                                                    for word in seed_for_sample],
                                                in_state, params, cnt_sample)
        tvm_sample_str = _pretty_print(tvm_samples, False, id_to_word)
        tf_samples, tf_state = tf_testing.do_tf_sample(
            sess,
            [word_to_id[word] for word in seed_for_sample],
            in_state, cnt_sample)
        tf_sample_str = _pretty_print(tf_samples, False, id_to_word)
        inpt = tvm_sample_str
        tvm.testing.assert_allclose(
            tf_samples, tvm_samples, rtol=1e-5, atol=1e-5)
        assert tvm_sample_str == tf_sample_str

#######################################################################
# LRN (Local Response Normalization)
# ----------------------------------


def _test_lrn(ishape, size, axis, bias, alpha, beta):
    """ testing local response normalization """
    lrn_depth_radius = size / 2

    inp_array = np.random.uniform(size=ishape).astype(np.float32)

    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape,
                             dtype=inp_array.dtype, name="lrn0_data")
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
    data = np.random.uniform(size=ishape).astype(np.float32)

    with tf.Graph().as_default():
        in1 = tf.placeholder(
            shape=data.shape, dtype=data.dtype, name="transpose_data")

        if axes is None:
            tf.transpose(in1)
        else:
            tf.transpose(in1, perm=axes)

        compare_tf_with_tvm(data, 'transpose_data:0', 'transpose:0')

def _test_forward_tranapose_axes_input(ishape, axes):
    data = np.random.uniform(size=ishape).astype(np.float32)
    axes_np = np.array(axes).astype(np.int32)

    with tf.Graph().as_default():
        in1 = tf.placeholder(
            shape=data.shape, dtype=data.dtype, name="transpose_data")

        const1 = tf.constant(axes_np, dtype=tf.int32)

        # make axes an input to tf.transpose, but not an input to the graph,
        # so it can be extracted with infer_value_simulated
        axes = tf.reverse(const1, axis=[-1])
        tf.transpose(in1, axes)

        compare_tf_with_tvm([data], ['transpose_data:0'], 'transpose:0')

def test_forward_transpose():
    _test_forward_transpose((2, 3, 4), (1, 2, 0))
    _test_forward_transpose((2, 3, 4))
    _test_forward_transpose((7, 8, 8, 10))
    _test_forward_transpose((2, 3, 4), (1, 2, 0))
    _test_forward_transpose((2, 3, 4), (0, 1, 2))
    _test_forward_transpose((2, 3, 4, 5), (3, 0, 1, 2))
    _test_forward_tranapose_axes_input((2, 3, 4), (1, 2, 0))
    _test_forward_tranapose_axes_input((2, 3, 4, 5), (3, 0, 1, 2))


def _test_forward_slice_operation_input(input_value, begin_value, size_value):
    input_data = np.array(input_value, dtype=np.float32)
    with tf.Graph().as_default():
        input_tensor = tf.placeholder(
            shape=input_data.shape, dtype=input_data.dtype, name="input")
        tf.slice(input_tensor, begin_value, size_value, name='slice_output')
        compare_tf_with_tvm([input_data], ['input:0'], 'slice_output:0')


def test_forward_slice():
    _test_forward_slice_operation_input([1, 1], [0], [2])
    _test_forward_slice_operation_input([0, 1, 2, 3], [3], [-1])
    _test_forward_slice_operation_input([[0, 1, 2, 3], [4, 5, 6, 7]],
                                        begin_value=[0, 1], size_value=[-1, -1])

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
    for mode in ['graph_runtime', 'vm']:
        with tf.Graph().as_default():
            in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
            tf.nn.relu(in1)
            compare_tf_with_tvm(inp_array, 'Placeholder:0', 'Relu:0', mode=mode)

def test_forward_leaky_relu():
    ishape = (1, 3, 10, 10)
    inp_array = np.random.uniform(-5, 5, size=ishape).astype(np.float32)
    for mode in ['graph_runtime', 'vm']:
        with tf.Graph().as_default():
            in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
            tf.nn.leaky_relu(in1, alpha=0.4)
            compare_tf_with_tvm(inp_array, 'Placeholder:0', 'LeakyRelu:0', mode=mode)


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
# Softmax
# -------
def test_forward_softmax():
    """test operator Softmax """
    def check_softmax(in_shape, axis, dtype):
        np_data = np.random.uniform(-100, 100, size=in_shape).astype(dtype)
        tf.reset_default_graph()
        with tf.Graph().as_default():
            in_data = tf.placeholder(dtype, in_shape, name="in_data")
            tf.nn.softmax(in_data, axis=axis, name="Softmax")
            compare_tf_with_tvm([np_data], ['in_data:0'], 'Softmax:0')
    check_softmax((2, 3, 5), 2, "float32")
    check_softmax((2, 3, 5), -1, "float32")


#######################################################################
# Tensor
# ------

def test_forward_round():
    """test Round"""
    np_data = np.random.uniform(-10, 10, size=(5, 7)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (5, 7), name="in_data")
        tf.round(in_data, name="round")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'round:0')


def test_forward_abs():
    """test operator Abs"""
    np_data = np.random.uniform(1, 100, size=(9, 11)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (9, 11), name="in_data")
        tf.math.abs(in_data, name="abs")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'abs:0')


def _test_forward_zeros_like(in_shape, dtype):
    np_data = np.random.uniform(-10, 10, size=in_shape).astype(dtype)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(dtype, in_shape, name="in_data")
        tf.zeros_like(in_data, name="zeros_like")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'zeros_like:0')


def test_forward_zeros_like():
    if tf.__version__ < LooseVersion('1.2'):
        _test_forward_zeros_like((2, 3), "int32")
        _test_forward_zeros_like((2, 3, 5), "int8")
        _test_forward_zeros_like((2, 3, 5, 7), "uint16")
        _test_forward_zeros_like((2, 3, 11), "float32")
        _test_forward_zeros_like((2, 3, 11), "float64")


def test_forward_erf():
    ishape = (1, 3, 10, 10)
    inp_array = np.random.uniform(-5, 5, size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
        tf.math.erf(in1)
        compare_tf_with_tvm(inp_array, 'Placeholder:0', 'Erf:0')


def test_forward_squared_difference():
    ishape = (1, 3, 10, 14)
    inp_array_a = np.random.uniform(-5, 5, size=ishape).astype(np.float32)
    inp_array_b = np.random.uniform(-5, 5, size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array_a.shape,
                             dtype=inp_array_a.dtype, name="in1")
        in2 = tf.placeholder(shape=inp_array_b.shape,
                             dtype=inp_array_b.dtype, name="in2")
        out = tf.math.squared_difference(in1, in2)
        compare_tf_with_tvm([inp_array_a, inp_array_b], [
                            in1.name, in2.name], out.name)


def _test_forward_reverse_v2(in_shape, axis, dtype):
    np_data = np.random.uniform(-10, 10, size=in_shape).astype(dtype)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(dtype, in_shape, name="in_data")
        tf.reverse(in_data, axis=[axis], name="reverse")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'reverse:0')


def test_forward_reverse_v2():
    """test ReverseV2"""
    _test_forward_reverse_v2((2, 3), 0, "int32")
    _test_forward_reverse_v2((2, 3, 5), 2, "float32")
    _test_forward_reverse_v2((2, 3, 5, 7), 1, "float32")
    _test_forward_reverse_v2((2, 3, 5), -1, "float64")
    _test_forward_reverse_v2((2, 3, 5), -3, "float64")


def test_forward_sign():
    """test Sign"""
    np_data = np.random.uniform(-10, 10, size=(5, 7, 11)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (5, 7, 11), name="in_data")
        tf.sign(in_data, name="sign")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'sign:0')


def test_forward_square():
    """test operator Square """
    np_data = np.random.uniform(1, 100, size=(2, 3, 5)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (2, 3, 5), name="in_data")
        tf.square(in_data, name="square")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'square:0')


def test_forward_pow_exp():
    """test Pow and Exp """
    np_in1 = np.random.uniform(-2, 2, size=(5, 7, 11)).astype(np.float32)
    np_in2 = np.random.uniform(-2, 2, size=(5, 7, 11)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.float32, (5, 7, 11), name="in1")
        in2 = tf.placeholder(tf.float32, (5, 7, 11), name="in2")
        out1 = tf.pow(in1, in2, name="pow")
        out = tf.exp(in1, name='exp')
        compare_tf_with_tvm([np_in1, np_in2], ['in1:0', 'in2:0'], 'pow:0')
        compare_tf_with_tvm([np_in1], ['in1:0'], 'exp:0')


def test_forward_log():
    """test operator Log """
    np_data = np.random.uniform(1, 100, size=(2, 3, 5)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (2, 3, 5), name="in_data")
        tf.log(in_data, name="log")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'log:0')


def test_forward_log1p():
    """test operator Log1p """
    np_data = np.random.uniform(1, 100, size=(2, 3, 5)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (2, 3, 5), name="in_data")
        tf.log1p(in_data, name="log1p")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'log1p:0')


def test_forward_cos():
    """test operator cos """
    np_data = np.random.uniform(1, 100, size=(2, 3, 5)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (2, 3, 5), name="in_data")
        tf.cos(in_data, name="cos")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'cos:0')


def test_forward_tan():
    """test operator tan """
    np_data = np.random.uniform(1, 100, size=(2, 3, 5)).astype(np.float32)
    tf.reset_default_graph()
    in_data = tf.placeholder(tf.float32, (2, 3, 5), name="in_data")
    tf.tan(in_data, name="tan")
    compare_tf_with_tvm([np_data], ['in_data:0'], 'tan:0')

def test_forward_atan():
    """test operator tan """
    tf.disable_eager_execution()
    np_data = np.random.uniform(1, 100, size=(2, 3, 5)).astype(np.float32)
    tf.reset_default_graph()
    in_data = tf.placeholder(tf.float32, (2, 3, 5), name="in_data")
    tf.atan(in_data, name="atan")
    compare_tf_with_tvm([np_data], ['in_data:0'], 'atan:0')

def test_forward_atan2():
    """test operator tan """
    tf.disable_eager_execution()
    np_data_1 = np.random.uniform(1, 100, size=(2, 3, 5)).astype(np.float32)
    np_data_2 = np.random.uniform(1, 100, size=(2, 3, 5)).astype(np.float32)
    tf.reset_default_graph()
    in_data_1 = tf.placeholder(tf.float32, (2, 3, 5), name="in_data_1")
    in_data_2 = tf.placeholder(tf.float32, (2, 3, 5), name="in_data_2")
    tf.atan2(in_data_1, in_data_2, name="atan2")
    compare_tf_with_tvm([np_data_1, np_data_2], ['in_data_1:0', 'in_data_2:0'], 'atan2:0')


def test_forward_sin():
    """test operator sin """
    np_data = np.random.uniform(1, 100, size=(2, 3, 5)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (2, 3, 5), name="in_data")
        tf.sin(in_data, name="sin")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'sin:0')


def test_forward_negative():
    """test tf operator Neg """
    np_data = np.random.uniform(-100, 255,
                                size=(224, 224, 3)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (224, 224, 3), name="in_data")
        tf.negative(in_data, name="negative")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'negative:0')


def test_forward_log_softmax():
    """test operator LogSoftmax"""
    np_data = np.random.uniform(1, 100, size=(9, 11)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (9, 11), name="in_data")
        tf.math.log_softmax(in_data, name="LogSoftmax")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'LogSoftmax:0')


def test_forward_softplus():
    """test operator Softplus"""
    np_data = np.random.uniform(1, 10, size=(2, 3, 5)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (2, 3, 5), name="in_data")
        tf.nn.softplus(in_data, name="softplus")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'softplus:0')


def test_forward_rsqrt():
    """test Rsqrt """
    np_data = np.random.uniform(1, 100, size=(5, 7, 11)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (5, 7, 11), name="in_data")
        tf.rsqrt(in_data, name="rsqrt")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'rsqrt:0')


def test_forward_sqrt():
    """test Sqrt """
    np_data = np.random.uniform(1, 100, size=(5, 7, 11)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (5, 7, 11), name="in_data")
        tf.sqrt(in_data, name="sqrt")
        compare_tf_with_tvm([np_data], ['in_data:0'], 'sqrt:0')


def _test_forward_right_shift(in_shape, dtype):
    """test operator RightShift"""
    lh_data = np.random.randint(1, 3, size=in_shape).astype(dtype)
    rh_data = np.random.randint(1, 8, size=in_shape).astype(dtype)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        lft_data = tf.placeholder(dtype, in_shape, name="lft_data")
        rgt_data = tf.placeholder(dtype, in_shape, name="rgt_data")
        tf.bitwise.right_shift(lft_data, rgt_data, name="RightShift")
        compare_tf_with_tvm([lh_data, rh_data], [
                            'lft_data:0', 'rgt_data:0'], 'RightShift:0')


def test_forward_right_shift():
    _test_forward_right_shift((7,), 'int32')
    _test_forward_right_shift((3, 11), 'int16')


def _test_forward_left_shift(in_shape, dtype):
    """test operator LeftShift"""
    lh_data = np.random.randint(100, 1000000, size=in_shape).astype(dtype)
    rh_data = np.random.randint(1, 3, size=in_shape).astype(dtype)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        lft_data = tf.placeholder(dtype, in_shape, name="lft_data")
        rgt_data = tf.placeholder(dtype, in_shape, name="rgt_data")
        tf.bitwise.left_shift(lft_data, rgt_data, name="LeftShift")
        compare_tf_with_tvm([lh_data, rh_data], [
                            'lft_data:0', 'rgt_data:0'], 'LeftShift:0')


def test_forward_left_shift():
    _test_forward_left_shift((10,), 'int32')
    _test_forward_left_shift((224, 224, 3), 'int16')

#######################################################################
# Mean
# ----


def test_forward_mean():
    def check_mean(ishape, **kwargs):
        inp_array = np.random.uniform(size=ishape).astype(np.float32)
        with tf.Graph().as_default():
            in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
            tf.keras.backend.mean(in1, **kwargs)
            compare_tf_with_tvm(inp_array, 'Placeholder:0',
                                'Mean:0', no_gpu=True)

    check_mean((10, 8, 16, 32))
    check_mean((10, 8, 16, 32), axis=(2, 3))
    check_mean((10, 8, 16, 32), axis=(1, 2), keepdims=True)

#######################################################################
# Size
# ----


def test_forward_size():
    def check_size(ishape):
        np_input = np.random.uniform(size=ishape).astype(np.float32)

        # if all dimensions are constant, TF will optimize away size operator into constant
        tf_input_shape = list(np_input.shape)
        tf_input_shape[0] = None

        with tf.Graph().as_default():
            input = tf.placeholder(shape=tf_input_shape,
                                   dtype=np_input.dtype, name='input')
            tf.size(input, name='size')
            compare_tf_with_tvm([np_input], ['input:0'], 'size:0')

    check_size((10, 8, 16, 32))
    check_size((10,))

#######################################################################
# All, Any, Max, Min, Prod, variance, std, logsumexp, euclidean_norm
# ------------------------------------------------------------------

def test_forward_reduce():
    def _check_op(tf_op, ishape, axis, keepdims, dtype="float32"):
        tf.reset_default_graph()
        if dtype == "bool":
            np_data = np.random.choice([True, False], size=ishape)
        else:
            np_data = np.random.uniform(size=ishape).astype(dtype)
        if tf_op == tf.math.reduce_prod:
            axis = 1
            np_data = np_data.reshape(1, -1)
        with tf.Graph().as_default():
            in_data = tf.placeholder(dtype, name="in_data")
            reduce_op = tf_op(in_data, axis=axis,
                               keepdims=keepdims, name="reduce_std")
            compare_tf_with_tvm([np_data], ['in_data:0'], reduce_op.name)

    def _test_math_op(op, dtypes=["int32", "float32"]):
        for dtype in dtypes:
            _check_op(op, (3, 10), axis=(-1), keepdims=False, dtype=dtype)
            _check_op(op, (8, 16, 32), axis=(-1), keepdims=False, dtype=dtype)
            _check_op(op, (1, 8, 8, 3), axis=(2, 3), keepdims=True, dtype=dtype)
            _check_op(op, (2, 3, 10, 10), axis=(1, 2), keepdims=True, dtype=dtype)

    _test_math_op(tf.math.reduce_all, dtypes=["bool"])
    _test_math_op(tf.math.reduce_any, dtypes=["bool"])
    _test_math_op(tf.math.reduce_max)
    _test_math_op(tf.math.reduce_min)
    _test_math_op(tf.math.reduce_prod)
    _test_math_op(tf.math.reduce_variance)
    _test_math_op(tf.math.reduce_std, dtypes=["float32"])
    _test_math_op(tf.math.reduce_logsumexp, dtypes=["float32"])
    if package_version.parse(tf.VERSION) >= package_version.parse('1.15.0'):
        _test_math_op(tf.math.reduce_euclidean_norm)

#######################################################################
# Relational operators
# --------------------


def _test_forward_rel_op(data, func):
    with tf.Graph().as_default():
        in1 = tf.placeholder(
            shape=data[0].shape, dtype=data[0].dtype, name='in1')
        in2 = tf.placeholder(
            shape=data[1].shape, dtype=data[1].dtype, name='in2')
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
# ExpandDims
# ----------


def _test_forward_expand_dims(data, axis):
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=data.shape, dtype=data.dtype, name='in1')
        out = tf.expand_dims(in1, axis)
        compare_tf_with_tvm([data], [in1.name], out.name)


def test_forward_expand_dims():
    _test_forward_expand_dims(np.int32(1), 0)
    _test_forward_expand_dims(np.array([1]), 0)
    _test_forward_expand_dims(np.array([1]), -1)
    _test_forward_expand_dims(np.array([[1], [2]]), 0)
    _test_forward_expand_dims(np.array([[1], [2]]), 1)
    _test_forward_expand_dims(np.array([[1], [2]]), -1)


#######################################################################
# Maximum, Minimum
# ----------------
def test_forward_maximum():
    """test Op Maximum"""
    def check_maximum(lh_shape, rh_shape, dtype):
        tf.reset_default_graph()
        lh_data = np.random.uniform(size=lh_shape).astype(dtype)
        rh_data = np.random.uniform(size=rh_shape).astype(dtype)
        with tf.Graph().as_default():
            lft_data = tf.placeholder(dtype, name="lft_data")
            rgt_data = tf.placeholder(dtype, name="rgt_data")
            tf.math.maximum(lft_data, rgt_data, name="maximum")
            compare_tf_with_tvm([lh_data, rh_data], [
                                'lft_data:0', 'rgt_data:0'], 'maximum:0')

    check_maximum((10, 8, 16, 32), (1,), dtype="int32")
    check_maximum((10, 8, 16, 32), (10, 8, 16, 32), dtype="float32")


def test_forward_minimum():
    """test Op Minimum"""
    def check_minimum(lh_shape, rh_shape, dtype):
        tf.reset_default_graph()
        lh_data = np.random.uniform(size=lh_shape).astype(dtype)
        rh_data = np.random.uniform(size=rh_shape).astype(dtype)
        with tf.Graph().as_default():
            lft_data = tf.placeholder(dtype, name="lft_data")
            rgt_data = tf.placeholder(dtype, name="rgt_data")
            tf.math.minimum(lft_data, rgt_data, name="minimum")
            compare_tf_with_tvm([lh_data, rh_data], [
                                'lft_data:0', 'rgt_data:0'], 'minimum:0')

    check_minimum((10, 8, 16, 32), (1,), dtype="int32")
    check_minimum((10, 8, 16, 32), (10, 8, 16, 32), dtype="float32")


#######################################################################
# PlaceholderWithDefault
# ----------------------
def test_placeholder():
    with tf.Graph().as_default():
        in_data1 = np.random.uniform(-5, 5, size=(3, 4, 5)).astype(np.float32)
        var1 = tf.Variable(in_data1, name='in1')
        var2 = array_ops.placeholder_with_default(var1, None, name='place1')

        in_data2 = np.random.uniform(-5, 5, size=(3, 4, 5)).astype(np.float32)
        place1 = array_ops.placeholder(
            shape=in_data1.shape, dtype=in_data1.dtype, name='in2')

        out1 = tf.math.add(var1, var2, name='out1')
        out2 = tf.math.add(out1, place1, name='out2')

        compare_tf_with_tvm([in_data1, in_data2], ['place1:0', 'in2:0'], 'out2:0',
                            init_global_variables=True)

#######################################################################
# OneHot
# ----------------------


def _test_forward_one_hot(indices_shape, depth, on_value, off_value, axis, out_dtype):
    inp_array1 = np.random.randint(0, 5, size=indices_shape)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array1.shape, dtype=inp_array1.dtype)
        out = tf.one_hot(in1, depth, on_value, off_value,
                         axis, dtype=out_dtype)
        compare_tf_with_tvm(inp_array1, in1.name, out.name)


def test_forward_one_hot():
    _test_forward_one_hot((3,), 3, 1, 0, -1, "int32")
    _test_forward_one_hot((3,), 3, 1.0, 0.0, -1, "float32")
    _test_forward_one_hot((2, 2), 5, 2, -2, 0, "int32")
    _test_forward_one_hot((2, 2), 5, 0.5, -0.5, 1, "float32")
    _test_forward_one_hot((3, 2, 4, 5), 6, 1, 0, 1, "int32")
    _test_forward_one_hot((3, 2, 4, 5), 6, 1.0, 0.0, 0, "float32")

#######################################################################
# AddN
# ----------------------


def _test_forward_add_n(inputs):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        temp = []
        for each in inputs:
            temp.append(tf.placeholder(shape=each.shape, dtype=each.dtype))
        output = tf.add_n(temp)
        compare_tf_with_tvm([each for each in inputs], [
                            each.name for each in temp], output.name)


def test_forward_add_n():
    x = np.random.randint(1, 100, size=(3, 3, 3), dtype=np.int32)
    y = np.random.randint(1, 100, size=(3, 3, 3), dtype=np.int32)
    z = np.random.randint(1, 100, size=(3, 3, 3), dtype=np.int32)
    m, n, o = x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)
    in0 = x
    in1 = [x, y]
    in2 = (x, y, z)
    in3 = m
    in4 = [m, n]
    in5 = (m, n, o)
    _test_forward_add_n(in0)
    _test_forward_add_n(in1)
    _test_forward_add_n(in2)
    _test_forward_add_n(in3)
    _test_forward_add_n(in4)
    _test_forward_add_n(in5)

#######################################################################
# Sharing params case
# ----------------------


def test_sharing_node():
    """Test the sharing params case."""
    np_data = np.random.uniform(size=(2,2,2)).astype('float32')
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, shape=(2, 2, 2), name='in_data')
        axis = tf.constant([-1], dtype=tf.int32, name='axis')
        mean0 = tf.reduce_mean(in_data, axis=axis, keepdims=False, name='mean0')
        mean1 = tf.reduce_mean(in_data, axis=axis, keepdims=False, name='mean1')
        out = tf.add(mean0, mean1, name='out')
        compare_tf_with_tvm([np_data], ['in_data:0'], 'out:0')

#######################################################################
# Unravel Index
# ----------------------
def _test_forward_unravel_index(inputs):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        temp = []
        for each in inputs:
            temp.append(tf.placeholder(shape=each.shape, dtype=each.dtype))
        output = tf.unravel_index(temp[0], temp[1])
        compare_tf_with_tvm([each for each in inputs], [
            each.name for each in temp], output.name)


def _test_forward_unravel_index_scalar(x, y, dtype="int32"):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        indices_1 = constant_op.constant(x, dtype=dtype)
        dims_1 = constant_op.constant(y, dtype=dtype)
        out_1 = array_ops.unravel_index(indices_1, dims_1)
        compare_tf_with_tvm([], [], out_1.name)


def test_forward_unravel_index():
    x = np.array([0, 1, 2, 3])
    y = np.array([2, 2])
    _test_forward_unravel_index([x, y])

    x = np.array([0, 1, 2, 5])
    y = np.array([2, 2])
    _test_forward_unravel_index([x, y])

    x = np.array([0, 1, 2, 5])
    y = np.array([2])
    _test_forward_unravel_index([x, y])

    x = np.array([102, 300, 16])
    y = np.array([10, 10, 9, 6])
    _test_forward_unravel_index([x, y])

    x = np.array([100])
    y = np.array([10, 10, 9, 6])
    _test_forward_unravel_index([x, y])

    # Test scalar input
    _test_forward_unravel_index_scalar(13, [1, 4, 5, 2])


#######################################################################
# Dilation2d
# ----------------------
def _test_dilation2d(tensor_in_sizes, filter_in_sizes,
                     strides, dilations, padding):
    """ One iteration of dilation2d with given shapes and attributes """

    total_size_1 = np.prod(tensor_in_sizes)
    total_size_2 = np.prod(filter_in_sizes)
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    data_array = [f * 1.0 for f in range(1, total_size_1 + 1)]
    filter_array = [f * 1.0 for f in range(1, total_size_2 + 1)]

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=tensor_in_sizes, dtype='float32')
        in_filter = constant_op.constant(
            filter_array, shape=filter_in_sizes, dtype='float32')

        nn_ops.dilation2d(in_data,
                          in_filter,
                          strides=strides,
                          rates=dilations,
                          padding=padding)

        compare_tf_with_tvm(np.reshape(data_array, tensor_in_sizes).astype('float32'),
                            'Placeholder:0', 'Dilation2D:0', no_gpu=True)


def test_forward_dilation():
    _test_dilation2d([1, 18, 18, 32], [4, 4, 32], [1, 1, 1, 1], [1, 2, 1, 1], "VALID")
    _test_dilation2d([1, 15, 15, 32], [4, 4, 32], [1, 1, 1, 1], [1, 2, 1, 1], "SAME")
    _test_dilation2d([1, 5, 5, 1], [2, 2, 1], [1, 1, 1, 1], [1, 1, 1, 1], "VALID")
    _test_dilation2d([1, 5, 5, 1], [3, 3, 1], [1, 1, 1, 1], [1, 2, 2, 1], "VALID")
    _test_dilation2d([1, 5, 5, 3], [3, 3, 3], [1, 1, 1, 1], [1, 1, 1, 1], "SAME")
    _test_dilation2d([1, 28, 28, 3], [5, 5, 3], [1, 2, 2, 1], [1, 1, 1, 1], "VALID")
    _test_dilation2d([1, 224, 224, 10], [8, 8, 10], [1, 1, 1, 1], [1, 1, 1, 1], "VALID")
    _test_dilation2d([1, 18, 18, 32], [4, 4, 32], [1, 1, 1, 1], [1, 2, 1, 1], "SAME")
    _test_dilation2d([1, 15, 15, 32], [4, 4, 32], [1, 1, 1, 1], [1, 2, 1, 1], "VALID")
    _test_dilation2d([1, 5, 5, 1], [7, 2, 1], [1, 3, 1, 1], [1, 1, 1, 1], "SAME")
    _test_dilation2d([1, 5, 5, 1], [3, 4, 1], [1, 2, 1, 1], [1, 2, 2, 1], "SAME")
    _test_dilation2d([1, 5, 5, 3], [3, 3, 3], [1, 1, 4, 1], [1, 1, 1, 1], "VALID")
    _test_dilation2d([1, 28, 28, 3], [5, 6, 3], [1, 1, 2, 1], [1, 1, 1, 1], "SAME")
    _test_dilation2d([1, 224, 224, 10], [8, 8, 10], [1, 3, 1, 1], [1, 1, 1, 1], "SAME")
    _test_dilation2d([1, 3, 3, 1], [2, 2, 1], [1, 1, 1, 1], [1, 2, 2, 1], "SAME")
    _test_dilation2d([1, 3, 3, 1], [2, 2, 1], [1, 1, 1, 1], [1, 1, 2, 1], "VALID")


#######################################################################
# infinity ops
# ------------
def _verify_infiniteness_ops(tf_op, name):
    """test operator infinity ops"""

    # Only float types are allowed in Tensorflow for isfinite and isinf
    # float16 is failing on cuda
    tf_dtypes = ["float32", "float64"]
    for tf_dtype in tf_dtypes:
        shape = (8, 8)
        data = np.random.uniform(size=shape).astype(tf_dtype)
        data.ravel()[np.random.choice(data.size, int(data.size * 0.5), replace=False)] = np.infty
        data.ravel()[np.random.choice(data.size, int(data.size * 0.5), replace=False)] = np.nan

        tf.reset_default_graph()
        in_data = tf.placeholder(tf_dtype, shape, name="in_data")
        tf_op(in_data, name=name)
        compare_tf_with_tvm([data], ['in_data:0'], '{}:0'.format(name))


def test_forward_isinf():
    _verify_infiniteness_ops(tf.is_inf, "isinf")


def test_forward_isfinite():
    _verify_infiniteness_ops(tf.is_finite, "isfinite")


#######################################################################
# Main
# ----
if __name__ == '__main__':
    # Transforms
    test_forward_slice()
    test_forward_transpose()
    test_forward_reshape()
    test_forward_depthtospace()
    test_forward_spacetodepth()
    test_forward_squeeze()
    test_forward_pack()
    test_forward_size()
    test_forward_broadcast_to()
    test_forward_fill()
    test_forward_crop()
    test_forward_resize()
    test_forward_crop_and_resize()
    test_forward_pad()
    test_forward_unpack()
    test_forward_gather()
    test_forward_gather_nd()
    test_forward_stridedslice()
    test_forward_split()
    test_forward_unstack()
    test_forward_tile()
    test_forward_top_k_v2()
    test_forward_clip_by_value()
    test_forward_maximum()
    test_forward_minimum()
    test_forward_range()
    test_forward_right_shift()
    test_forward_left_shift()
    test_forward_truncatemod()
    test_forward_one_hot()
    test_forward_atan()
    test_forward_atan2()

    # Activations
    test_forward_sigmoid()
    test_forward_relu()
    test_forward_leaky_relu()
    test_forward_elu()
    test_forward_selu()
    test_forward_tanh()

    # Tensor
    test_forward_round()
    test_forward_reverse_v2()
    test_forward_pow_exp()
    test_forward_sign()
    test_forward_log()
    test_forward_log1p()
    test_forward_tan()
    test_forward_cos()
    test_forward_sin()
    test_forward_negative()
    test_forward_divide()
    test_forward_abs()
    test_forward_softplus()
    test_forward_sqrt()
    test_forward_rsqrt()
    test_forward_expand_dims()
    test_forward_square()
    test_forward_softmax()
    test_forward_log_softmax()
    test_forward_bias_add()
    test_forward_zeros_like()
    test_forward_erf()
    test_forward_squared_difference()
    test_forward_add_n()
    test_forward_floormod()
    test_forward_isfinite()
    test_forward_isinf()
    test_forward_unravel_index()

    # Reductions
    test_forward_argminmax()
    test_forward_reduce()
    test_forward_mean()

    # TensorArray
    test_tensor_array_write_read()
    test_tensor_array_concat()
    test_tensor_array_scatter()
    test_tensor_array_gather()
    test_tensor_array_size()
    test_tensor_array_split()
    test_tensor_array_stack()
    test_tensor_array_unstack()

    # General
    test_forward_multi_input()
    test_forward_multi_output()
    test_forward_variable()
    test_placeholder()

    # NN
    test_forward_convolution()
    test_forward_pooling()
    test_forward_concat_v2()
    test_forward_lrn()
    test_forward_l2_normalize()
    test_forward_space_to_batch_nd()
    test_forward_batch_to_space_nd()
    test_forward_dilation()

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
    test_forward_where()
    test_forward_matmul()
    test_forward_batch_matmul()

    # Internal misc. ops
    test_read_variable_op()

    # Sharing params case using Mean ops
    test_sharing_node()
