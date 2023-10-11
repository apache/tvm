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
# pylint: disable=deprecated-module

""" Test translate from tensorflow. """

from packaging import version as package_version
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables


try:
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
except ImportError:
    import tensorflow as tf

import tvm
import tvm.testing
import tvm.relay.testing.tf as tf_testing
from tvm.contrib.msc.framework.tensorflow.frontend import translate
from tvm.contrib.msc.framework.tensorflow import codegen


# Only allow TF to run on half the GPU RAM to save the other half
# For TVM
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
gpu_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
gpu_sess.close()


def convert_to_list(x):
    if not isinstance(x, list):
        x = [x]
    return x


def run_tf_graph(sess, input_data, input_node, output_node):
    """Generic function to execute tensorflow"""

    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)
    output_node = convert_to_list(output_node)

    tensor = [sess.graph.get_tensor_by_name(output_name) for output_name in output_node]

    input_dict = {e: input_data[i] for i, e in enumerate(input_node)}
    if len(input_node) == 1 and input_node[0] == "":
        output_data = sess.run(tensor)
    else:
        output_data = sess.run(tensor, input_dict)
    return output_data


def get_graph_def(in_data, in_name, out_name):
    """Get tf.GraphDef for translate"""

    def name_without_num(name):
        return name.split(":")[0] if ":" in name else name

    out_name = convert_to_list(out_name)
    out_node = [name_without_num(name) for name in out_name]
    in_data = convert_to_list(in_data)
    in_name = convert_to_list(in_name)

    with tf.Session() as sess:
        sess.run(variables.global_variables_initializer())
        final_graph_def = tf_testing.AddShapesToGraphDef(sess, out_node)
        golden = run_tf_graph(sess, in_data, in_name, out_name)
    return final_graph_def, golden


def verify_model(graph_def, golden, in_data, in_name, out_name, use_out_name=True):
    """Generic function to generate and compare tensorflow and MSC-TFV1 output"""

    out_name = convert_to_list(out_name)
    in_data = convert_to_list(in_data)
    in_name = convert_to_list(in_name)
    shape_dict = {i: d.shape for i, d in zip(in_name, in_data)}
    graph, weights = translate.from_tensorflow(graph_def, shape_dict, out_name)
    with tf.Graph().as_default():
        outputs = codegen.to_tensorflow(graph, weights)
        with tf.Session() as sess:
            sess.run(variables.global_variables_initializer())
            if not use_out_name:
                out_name = [o.name for o in convert_to_list(outputs)]
            result = run_tf_graph(sess, in_data, in_name, out_name)

    golden = convert_to_list(golden)
    result = convert_to_list(result)
    assert len(golden) == len(result), "golden {} mismatch with result {}".format(
        len(golden), len(result)
    )
    for gol_r, new_r in zip(golden, result):
        if isinstance(gol_r, np.ndarray):
            tvm.testing.assert_allclose(gol_r, new_r, atol=1e-5, rtol=1e-5)
        else:
            assert gol_r == new_r


def _test_pooling(input_shape, **kwargs):
    """One iteration of pool operation with given shapes and attributes"""

    x = -np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) - 1

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=input_shape, dtype="float32")
        nn_ops.pool(in_data, **kwargs)
        out_name = "max_pool:0" if kwargs["pooling_type"] == "MAX" else "avg_pool:0"
        io_info = {"in_data": x, "in_name": "Placeholder:0", "out_name": out_name}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_pooling():
    """test tensorflow translator for pooling"""

    for pool_type in ["AVG", "MAX"]:
        _test_pooling(
            input_shape=[2, 9, 10, 2],
            window_shape=[2, 1],
            padding="SAME",
            pooling_type=pool_type,
            dilation_rate=[1, 1],
            strides=[1, 1],
        )

        _test_pooling(
            input_shape=[2, 9, 10, 2],
            window_shape=[2, 1],
            padding="VALID",
            pooling_type=pool_type,
            dilation_rate=[1, 1],
            strides=[1, 1],
        )

        _test_pooling(
            input_shape=[1, 2, 1],
            window_shape=[1],
            padding="VALID",
            pooling_type=pool_type,
            dilation_rate=[1],
        )

    # Explicit padding
    if package_version.parse(tf.VERSION) >= package_version.parse("2.4.1"):
        _test_pooling(
            input_shape=[2, 9, 10, 2],
            window_shape=[4, 4],
            padding=[[0, 0], [0, 1], [2, 3], [0, 0]],
            pooling_type="MAX",
            dilation_rate=[1, 1],
            strides=[1, 1],
        )


def _test_convolution(
    opname,
    tensor_in_sizes,
    filter_in_sizes,
    dilations,
    strides,
    padding,
    data_format,
):
    """One iteration of convolution with given shapes and attributes"""
    total_size_1 = np.prod(tensor_in_sizes)
    total_size_2 = np.prod(filter_in_sizes)
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    data_array = [f * 1.0 for f in range(1, total_size_1 + 1)]
    filter_array = [f * 1.0 for f in range(1, total_size_2 + 1)]

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=tensor_in_sizes, dtype="float32")
        in_filter = constant_op.constant(filter_array, shape=filter_in_sizes, dtype="float32")
        if data_format == "NHWC":
            strides = [1] + strides + [1]
            dilations = [1] + dilations + [1]
        else:
            strides = [1, 1] + strides
            dilations = [1, 1] + dilations

        if opname == "conv":
            nn_ops.conv2d(
                in_data,
                in_filter,
                strides=strides,
                dilations=dilations,
                padding=padding,
                data_format=data_format,
            )
            io_info = {
                "in_data": np.reshape(data_array, tensor_in_sizes).astype("float32"),
                "in_name": "Placeholder:0",
                "out_name": "Conv2D:0",
            }
            graph_def, golden = get_graph_def(**io_info)
        else:
            nn_ops.depthwise_conv2d_native(
                in_data,
                in_filter,
                strides=strides,
                dilations=dilations,
                padding=padding,
                data_format=data_format,
            )
            io_info = {
                "in_data": np.reshape(data_array, tensor_in_sizes).astype("float32"),
                "in_name": "Placeholder:0",
                "out_name": "DepthwiseConv2dNative:0",
            }
            graph_def, golden = get_graph_def(**io_info)
        verify_model(graph_def, golden, **io_info)


def test_convolution():
    """test tensorflow translator for convolution"""

    _test_convolution("conv", [4, 8, 8, 176], [1, 1, 176, 32], [1, 1], [1, 1], "SAME", "NHWC")
    _test_convolution("conv", [4, 17, 17, 19], [3, 3, 19, 19], [1, 1], [2, 2], "VALID", "NHWC")
    _test_convolution("depthwise", [4, 8, 8, 176], [1, 1, 176, 1], [1, 1], [1, 1], "SAME", "NHWC")
    _test_convolution("depthwise", [4, 17, 17, 19], [3, 3, 19, 1], [1, 1], [2, 2], "VALID", "NHWC")

    # Explicit padding
    if package_version.parse(tf.VERSION) >= package_version.parse("2.4.1"):
        _test_convolution(
            "conv",
            [4, 8, 8, 16],
            [1, 1, 16, 32],
            [1, 1],
            [1, 1],
            [[0, 0], [2, 3], [0, 1], [0, 0]],
            "NHWC",
        )
        _test_convolution(
            "depthwise",
            [4, 8, 8, 16],
            [1, 1, 16, 1],
            [1, 1],
            [1, 1],
            [[0, 0], [2, 3], [0, 1], [0, 0]],
            "NHWC",
        )


def _test_biasadd(tensor_in_sizes, data_format):
    """One iteration of biasadd with given shapes and attributes"""

    total_size_1 = 1
    for s in tensor_in_sizes:
        total_size_1 *= s
    tensor_bias_sizes = [tensor_in_sizes[1]] if data_format == "NCHW" else [tensor_in_sizes[3]]
    total_size_2 = tensor_bias_sizes[0]
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    data_array = [f * 1.0 for f in range(1, total_size_1 + 1)]
    bias_array = [f * 1.0 for f in range(1, total_size_2 + 1)]

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=tensor_in_sizes, dtype="float32")
        in_bias = constant_op.constant(bias_array, shape=tensor_bias_sizes, dtype="float32")
        nn_ops.bias_add(in_data, in_bias, data_format=data_format)
        io_info = {
            "in_data": np.reshape(data_array, tensor_in_sizes).astype("float32"),
            "in_name": "Placeholder:0",
            "out_name": "BiasAdd:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_biasadd():
    """test tensorflow translator for bias_add"""

    _test_biasadd([4, 8, 8, 176], "NHWC")


def _test_where_with_broadcast(in_shape, cond_shape):
    choice_list = list(np.arange(10).astype("float32"))
    t_1 = np.random.choice(choice_list, size=cond_shape)
    t_2 = np.random.choice(choice_list, size=cond_shape)
    x = np.random.choice(choice_list, size=in_shape)
    y = np.random.choice(choice_list, size=in_shape)

    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=cond_shape, dtype="float32", name="in1")
        in2 = tf.placeholder(shape=cond_shape, dtype="float32", name="in2")
        condition = math_ops.less(in1, in2, name="less")
        lhs = tf.placeholder(shape=in_shape, dtype="float32", name="x")
        rhs = tf.placeholder(shape=in_shape, dtype="float32", name="y")
        out = tf.where(condition, lhs, rhs)
        io_info = {
            "in_data": [t_1, t_2, x, y],
            "in_name": ["in1:0", "in2:0", "x:0", "y:0"],
            "out_name": out.name,
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_where_with_broadcast():
    """test tensorflow translator for where"""

    _test_where_with_broadcast((5, 2), (5,))
    _test_where_with_broadcast((3, 2, 5), (3,))


def _test_reshape(data, out_shape):
    """One iteration of reshape operation with given data and out shape"""

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        array_ops.reshape(in_data, out_shape)
        io_info = {"in_data": data, "in_name": "Placeholder:0", "out_name": "Reshape:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def _test_reshape_with_call():
    """relay.expr.Call as shape"""
    data = np.zeros((6, 4, 2))
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out_shape = tf.constant([1, 2, 3], dtype="int32")
        out_shape = tf.multiply(out_shape, 2)
        array_ops.reshape(in_data, out_shape)
        io_info = {"in_data": data, "in_name": "Placeholder:0", "out_name": "Reshape:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def _test_reshape_like(data, shape_like):
    """A special case for reshape."""

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        in_shape_like = array_ops.placeholder(shape=shape_like.shape, dtype=data.dtype)
        out_shape = array_ops.shape(in_shape_like)
        array_ops.reshape(in_data, out_shape)
        io_info = {"in_data": data, "in_name": "Placeholder:0", "out_name": "Reshape:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_reshape():
    """test tensorflow translator for reshape"""

    _test_reshape(np.arange(6.0), [2, 3])
    _test_reshape(np.arange(6), [-1, 2])
    _test_reshape_with_call()
    _test_reshape_like(np.zeros((3, 6)), np.zeros((9, 2)))


def _test_sigmoid(data):
    """One iteration of sigmoid"""

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        _ = math_ops.sigmoid(in_data)
        io_info = {"in_data": data, "in_name": "Placeholder:0", "out_name": "Sigmoid:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_sigmoid():
    """test tensorflow translator for concat"""

    _test_sigmoid(np.random.uniform(size=(3, 4, 4, 3)).astype("float32"))


def _test_argx(func, data, **kwargs):

    with tf.Graph().as_default():
        inp = array_ops.placeholder(shape=data.shape, dtype=data.dtype, name="c0")
        func(inp, name="argx", **kwargs)
        io_info = {"in_data": data, "in_name": "c0:0", "out_name": "argx:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_argx():
    """test tensorflow translator for argmax/argmin"""

    data = np.random.uniform(size=(8, 4, 9)).astype("float32")
    for output_type in [tf.int64, tf.int32]:
        _test_argx(tf.argmax, data=data, axis=1, output_type=output_type)
        _test_argx(tf.argmin, data=data, axis=1, output_type=output_type)


def _test_matmul(i, j, k, transpose_a=False, transpose_b=False):
    """One iteration of matmul"""

    a_shape_init = [i, j]
    b_shape_init = [j, k]
    a_shape = [] + (a_shape_init[::-1] if transpose_a else a_shape_init)
    b_shape = [] + (b_shape_init[::-1] if transpose_b else b_shape_init)

    with tf.Graph().as_default():
        a_in = tf.placeholder(shape=a_shape, dtype="float32", name="A")
        b_in = tf.placeholder(shape=b_shape, dtype="float32", name="B")
        result = tf.matmul(a_in, b_in, transpose_a=transpose_a, transpose_b=transpose_b)

        a_np = np.random.uniform(high=5.0, size=a_shape).astype("float32")
        b_np = np.random.uniform(high=5.0, size=b_shape).astype("float32")
        io_info = {
            "in_data": [a_np, b_np],
            "in_name": [a_in.name, b_in.name],
            "out_name": result.name,
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info, use_out_name=False)


def test_matmul():
    """test tensorflow translator for matmul"""

    _test_matmul(1, 3, 6)
    _test_matmul(1, 3, 6, True, True)
    _test_matmul(1, 3, 6, True, False)
    _test_matmul(1, 3, 6, False, True)


def _test_batch_matmul(a_shape, b_shape, adjoint_a=False, adjoint_b=False):

    with tf.Graph().as_default():
        a_in = tf.placeholder(shape=a_shape, dtype="float32", name="A")
        b_in = tf.placeholder(shape=b_shape, dtype="float32", name="B")
        result = tf.matmul(a_in, b_in, adjoint_a=adjoint_a, adjoint_b=adjoint_b, name="batchmatmul")

        a_np = np.random.uniform(high=5.0, size=a_shape).astype("float32")
        b_np = np.random.uniform(high=5.0, size=b_shape).astype("float32")
        io_info = {
            "in_data": [a_np, b_np],
            "in_name": [a_in.name, b_in.name],
            "out_name": result.name,
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_batch_matmul():
    """test tensorflow translator for batch_matmul"""

    _test_batch_matmul((3, 5, 4), (3, 4, 5))
    _test_batch_matmul((3, 5, 4), (3, 4, 5), True, True)
    _test_batch_matmul((3, 5, 4), (3, 5, 4), True, False)
    _test_batch_matmul((3, 5, 4), (3, 5, 4), False, True)


def _test_stridedslice(
    ip_shape,
    begin,
    end,
    stride,
    begin_mask=0,
    end_mask=0,
    new_axis_mask=0,
    shrink_axis_mask=0,
    ellipsis_mask=0,
):
    """One iteration of a Stridedslice"""

    tf.reset_default_graph()
    np_data = np.random.uniform(size=ip_shape).astype("float32")
    with tf.Graph().as_default():
        in_data = tf.placeholder("float32", ip_shape, name="in_data")
        tf.strided_slice(
            in_data,
            begin,
            end,
            stride,
            begin_mask=begin_mask,
            end_mask=end_mask,
            new_axis_mask=new_axis_mask,
            shrink_axis_mask=shrink_axis_mask,
            ellipsis_mask=ellipsis_mask,
            name="strided_slice",
        )
        io_info = {"in_data": np_data, "in_name": "in_data:0", "out_name": "strided_slice:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_stridedslice():
    """test tensorflow translator for stridedslice"""

    _test_stridedslice([2, 3, 4], [0], [1], [1], shrink_axis_mask=8)
    _test_stridedslice([3, 4, 3], [1, -1, 0], [4, -5, 3], [2, -1, 1])
    _test_stridedslice([3, 4, 3], [1, 0], [4, 3], [2, 1], ellipsis_mask=8)
    _test_stridedslice([3, 4, 3], [1, 1, 0], [4, 4, 2], [2, 1, 1], new_axis_mask=5)
    _test_stridedslice(
        [3, 4, 5, 4, 5, 6],
        [0, 0, 1, 2, 1],
        [2, 3, 4, 5, 3],
        [1, 1, 2, 2, 1],
        shrink_axis_mask=5,
        new_axis_mask=1,
        ellipsis_mask=2,
        begin_mask=8,
        end_mask=8,
    )


def _test_divide(ip_shape, dtype):
    np_numer = np.random.uniform(-100, 100, size=ip_shape).astype(dtype)
    np_denomin = np.random.uniform(1, 100, size=ip_shape).astype(dtype)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        numerator = tf.placeholder(dtype, ip_shape, name="numer")
        denominator = tf.placeholder(dtype, ip_shape, name="denomin")
        tf.math.divide(numerator, denominator, name="RealDiv")
        io_info = {
            "in_data": [np_numer, np_denomin],
            "in_name": ["numer:0", "denomin:0"],
            "out_name": "RealDiv:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def _test_floordiv(ip_shape, dtype):
    np_numer = np.random.uniform(1, 100, size=ip_shape).astype(dtype)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        numerator = tf.placeholder(dtype, ip_shape, name="numer")
        tf.math.floordiv(numerator, tf.constant(5, dtype=dtype), name="FloorDiv")
        io_info = {"in_data": [np_numer], "in_name": ["numer:0"], "out_name": "FloorDiv:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_divide():
    """test tensorflow translator for div"""

    _test_divide((4, 3, 7), "float32")
    _test_divide((4, 3, 7), "int32")
    _test_floordiv((4, 3, 7), "float32")
    _test_floordiv((4, 3, 7), "int32")


def _test_gather(ip_shape, indice_shape, indice_value, axis, batch_dims):
    """One iteration of a GatherV2"""

    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder("float32", ip_shape, name="in_data")
        indices = tf.placeholder("int32", indice_shape, name="indices")
        out = tf.gather(in_data, indices, axis=axis, batch_dims=batch_dims)
        np_data = np.random.uniform(1, 10, size=ip_shape).astype("float32")

        def _fill_indices(indice_value):
            indices = np.array(ip_shape, dtype="float32")
            if isinstance(indice_value, int):
                indices = np.array([indice_value], dtype="int32")
            else:
                indices = np.asarray(indice_value, dtype="int32")
            return indices

        np_indices = _fill_indices(indice_value)
        io_info = {
            "in_data": [np_data, np_indices],
            "in_name": ["in_data:0", "indices:0"],
            "out_name": out.name,
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_gather():
    """test tensorflow translator for gather"""

    _test_gather((4,), (1,), 1, 0, 0)
    _test_gather((2, 2), (1, 2, 2), [[[1, 0], [0, 1]]], 0, 0)
    _test_gather((4, 3, 5, 6), (1, 4), [[2, 1, 0, 0]], 0, 0)


def _test_split(in_shape, axis, num_or_size_splits):
    """One iteration of a Split"""
    np_data = np.random.uniform(-5, 5, size=in_shape).astype("float32")

    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder("float32", in_shape, name="in_data")
        _ = len(num_or_size_splits) if isinstance(num_or_size_splits, list) else num_or_size_splits
        split = tf.split(in_data, num_or_size_splits, axis=axis)
        relu = [tf.nn.relu(i) for i in split]
        io_info = {
            "in_data": [np_data],
            "in_name": ["in_data:0"],
            "out_name": [n.name for n in relu],
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)

    # and now test together with concat
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder("float32", in_shape, name="in_data")
        splitted = tf.split(in_data, num_or_size_splits, axis=axis)
        concat = tf.concat(splitted, axis)
        io_info = {
            "in_data": [np_data],
            "in_name": ["in_data:0"],
            "out_name": concat.name,
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_split():
    """test tensorflow translator for split"""

    _test_split((6, 1, 3, 5), 0, 3)
    _test_split((6, 1, 3, 5), -4, 3)
    _test_split((3, 6, 4), -2, [1, 4, 1])


def _test_tile(in_shape, multiples):
    np_data = np.random.uniform(-5, 5, size=in_shape).astype("float32")
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder("float32", in_shape, name="in_data")
        tf.tile(in_data, multiples=multiples, name="tile")
        io_info = {"in_data": np_data, "in_name": "in_data:0", "out_name": "tile:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_tile():
    """test tensorflow translator for tile"""

    _test_tile((2, 2), (2, 3))


def _test_clip_by_value(ip_shape, clip_value_min, clip_value_max):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder("float32", ip_shape, name="in_data")
        tf.clip_by_value(in_data, clip_value_min, clip_value_max, name="ClipByValue")
        np_data = np.random.uniform(-100, 100, size=ip_shape).astype("float32")
        io_info = {"in_data": np_data, "in_name": "in_data:0", "out_name": "ClipByValue:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_clip_by_value():
    """test tensorflow translator for clip"""

    _test_clip_by_value((4,), 0.1, 5.0)


def test_multi_input():
    """test tensorflow translator for multi input"""

    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.int32, shape=[3, 3], name="in1")
        in2 = tf.placeholder(tf.int32, shape=[3, 3], name="in2")
        in3 = tf.placeholder(tf.int32, shape=[3, 3], name="in3")
        in4 = tf.placeholder(tf.int32, shape=[3, 3], name="in4")

        out1 = tf.add(in1, in2, name="out1")
        out2 = tf.subtract(in3, in4, name="out2")
        _ = tf.multiply(out1, out2, name="out")
        in_data = np.arange(9, dtype="int32").reshape([3, 3])
        io_info = {
            "in_data": [in_data, in_data, in_data, in_data],
            "in_name": ["in1:0", "in2:0", "in3:0", "in4:0"],
            "out_name": "out:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_multi_output():
    """test tensorflow translator for multi output"""

    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.int32, shape=[3, 3], name="in1")
        in2 = tf.placeholder(tf.int32, shape=[3, 3], name="in2")
        in3 = tf.placeholder(tf.int32, shape=[3, 3], name="in3")
        in4 = tf.placeholder(tf.int32, shape=[3, 3], name="in4")

        _ = tf.add(in1, in2, name="out1")
        _ = tf.subtract(in3, in4, name="out2")
        in_data = np.arange(9, dtype="int32").reshape([3, 3])
        io_info = {
            "in_data": [in_data] * 4,
            "in_name": ["in1:0", "in2:0", "in3:0", "in4:0"],
            "out_name": ["out1:0", "out2:0"],
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def _test_resize_bilinear(in_shape, to_shape, align_corners):
    """One iteration of resize bilinear"""

    data = np.random.uniform(size=in_shape).astype("float32")
    shape_data = np.array(to_shape).astype("int32")

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        shape_data = constant_op.constant(
            shape_data, shape=shape_data.shape, dtype=shape_data.dtype
        )
        tf.image.resize_bilinear(in_data, shape_data, align_corners=align_corners)
        io_info = {"in_data": data, "in_name": "Placeholder:0", "out_name": "ResizeBilinear:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def _test_resize_nearest_neighbor(in_shape, to_shape):
    """One iteration of resize nearest neighbor"""

    data = np.random.uniform(size=in_shape).astype("float32")
    shape_data = np.array(to_shape).astype("int32")

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        shape_data = constant_op.constant(
            shape_data, shape=shape_data.shape, dtype=shape_data.dtype
        )
        tf.image.resize_nearest_neighbor(in_data, shape_data, name="resize_nearest_neighbor")
        io_info = {
            "in_data": data,
            "in_name": "Placeholder:0",
            "out_name": "resize_nearest_neighbor:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_resize():
    """test tensorflow translator for resize"""

    _test_resize_bilinear((4, 32, 32, 3), [50, 50], False)
    _test_resize_bilinear((6, 32, 32, 3), [20, 20], True)
    _test_resize_nearest_neighbor((6, 32, 32, 3), [20, 20])


def _test_broadcast_to(in_shape, to_shape):
    """One iteration of broadcast_to"""

    data = np.random.uniform(size=in_shape).astype("float32")
    shape_data = np.array(to_shape).astype("int32")

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        shape_data = constant_op.constant(
            shape_data, shape=shape_data.shape, dtype=shape_data.dtype
        )
        tf.broadcast_to(in_data, shape_data)
        io_info = {"in_data": data, "in_name": "Placeholder:0", "out_name": "BroadcastTo:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_broadcast_to():
    """test tensorflow translator for broadcast_to"""

    _test_broadcast_to((4, 1, 32, 32), [4, 8, 32, 32])


def _test_fill(in_shape):
    """Use the fill op to create a tensor of ones with non-constant shape."""

    with tf.Graph().as_default():
        tf.ones(shape=in_shape, dtype="float32")
        io_info = {"in_data": in_shape, "in_name": [], "out_name": "ones:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info, use_out_name=False)


def test_fill():
    """test tensorflow translator for fill"""

    _test_fill((6, 32, 64, 64))


def _test_pack(axis, shape, **kwargs):

    a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    b = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)

    with tf.Graph().as_default():
        tf_a = array_ops.placeholder(shape=shape, dtype="float32", name="pl_a")
        tf_b = array_ops.placeholder(shape=shape, dtype="float32", name="pl_b")
        tf_c = tf.stack([tf_a, tf_b], axis=axis, **kwargs)
        assert tf_c.op.op_def.name == "Pack", "tf.stack() is expected to produce 'Pack' operation"
        io_info = {"in_data": [a, b], "in_name": ["pl_a:0", "pl_b:0"], "out_name": "stack:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_pack():
    """test tensorflow translator for pack"""

    _test_pack(3, [3, 2, 1])


def _test_unpack(in_shape, axis):
    """test operator Unpack"""
    np_data = np.random.uniform(-100, 100, size=in_shape).astype("float32")
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder("float32", in_shape, name="in_data")
        tf.unstack(in_data, axis=axis, name="Unpack")
        io_info = {"in_data": [np_data], "in_name": ["in_data:0"], "out_name": "Unpack:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_unpack():
    """test tensorflow translator for unpack"""

    _test_unpack((21, 23, 3), 2)


def _test_einsum(equation, *shape_of_input_tensors):
    """Test Einsum Op"""

    with tf.Graph().as_default():
        inputs_placeholders = []
        input_data = []
        for idx, shape in enumerate(shape_of_input_tensors):
            input_name = f"input_{idx}"
            inputs_placeholders.append(
                tf.placeholder(shape=shape, dtype="float32", name=input_name)
            )
            input_data.append(np.random.normal(size=shape).astype("float32"))

        result = tf.einsum(equation, *inputs_placeholders)
        io_info = {
            "in_data": input_data,
            "in_name": [ph.name for ph in inputs_placeholders],
            "out_name": result.name,
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info, use_out_name=False)


def test_einsum():
    """test tensorflow translator for einsum"""

    _test_einsum("ij,jk->ik", [2, 3], [3, 5])  # Matmul
    _test_einsum("ij,jk", [2, 3], [3, 5])  # Matmul
    _test_einsum("i,i->", [2], [2])  # Dot product
    _test_einsum("i,j->ij", [3], [5])  # Outer produce
    _test_einsum("ij->ji", [2, 3])  # Transpose
    _test_einsum("ii->i", [3, 3])  # Diag
    _test_einsum("ii", [3, 3])  # Trace of a square matrix
    _test_einsum("bij,bjk->bik", [7, 5, 3], [7, 3, 2])  # Batch matmul


def _test_pad(input_shape, paddings, mode, **kwargs):
    """One iteration of pad operation with given shape"""

    x = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape)

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=input_shape, dtype="float32")
        pad_values = constant_op.constant(paddings)
        _ = tf.pad(in_data, paddings=pad_values, mode=mode, **kwargs)

        if mode == "CONSTANT":
            if "constant_values" in kwargs:
                out_name = "PadV2:0"
            else:
                out_name = "Pad:0"
        else:
            out_name = "MirrorPad:0"

        io_info = {
            "in_data": x,
            "in_name": "Placeholder:0",
            "out_name": out_name,
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_pad():
    """test tensorflow translator for pad"""

    _test_pad((2, 3), [[1, 1], [2, 2]], mode="CONSTANT")
    _test_pad((2, 3), [[1, 1], [2, 2]], mode="CONSTANT", constant_values=1.0)


def test_logical_and():
    """test tensorflow translator for logical_and"""

    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name="in1")
        in2 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name="in2")
        _ = tf.logical_and(in1, in2, name="out")
        in_data1 = np.random.choice(a=[False, True], size=(1, 4, 4, 3)).astype("bool")
        in_data2 = np.random.choice(a=[False, True], size=(1, 4, 4, 3)).astype("bool")
        io_info = {
            "in_data": [in_data1, in_data2],
            "in_name": ["in1:0", "in2:0"],
            "out_name": "out:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_logical_or():
    """test tensorflow translator for logical_or"""

    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name="in1")
        in2 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name="in2")
        _ = tf.logical_or(in1, in2, name="out")
        in_data1 = np.random.choice(a=[False, True], size=(1, 4, 4, 3)).astype("bool")
        in_data2 = np.random.choice(a=[False, True], size=(1, 4, 4, 3)).astype("bool")
        io_info = {
            "in_data": [in_data1, in_data2],
            "in_name": ["in1:0", "in2:0"],
            "out_name": "out:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_logical_xor():
    """test tensorflow translator for logical_xor"""

    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name="in1")
        in2 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name="in2")
        _ = tf.logical_xor(in1, in2, name="out")
        in_data1 = np.random.choice(a=[False, True], size=(1, 4, 4, 3)).astype("bool")
        in_data2 = np.random.choice(a=[False, True], size=(1, 4, 4, 3)).astype("bool")
        io_info = {
            "in_data": [in_data1, in_data2],
            "in_name": ["in1:0", "in2:0"],
            "out_name": "out:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_logical_not():
    """test tensorflow translator for logical_not"""

    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.bool, shape=[1, 4, 4, 3], name="in1")
        _ = tf.logical_not(in1, name="out")
        in_data1 = np.random.choice(a=[False, True], size=(1, 4, 4, 3)).astype("bool")
        io_info = {
            "in_data": [in_data1],
            "in_name": ["in1:0"],
            "out_name": "out:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_where():
    """test tensorflow translator for where"""

    with tf.Graph().as_default():
        with tf.Session() as _:
            input1 = tf.placeholder(tf.int32, shape=[1, 4, 4, 3], name="input1")
            input2 = tf.placeholder(tf.int32, shape=[1, 4, 4, 3], name="input2")
            mask = input1 > input2
            tf.where(mask, input1 + 1, input2 * 2)
            in_data1 = np.random.uniform(0, 10, size=(1, 4, 4, 3)).astype("uint32")
            in_data2 = np.random.uniform(0, 10, size=(1, 4, 4, 3)).astype("uint32")
            io_info = {
                "in_data": [in_data1, in_data2],
                "in_name": ["input1:0", "input2:0"],
                "out_name": "Select:0",
            }
            graph_def, golden = get_graph_def(**io_info)
        verify_model(graph_def, golden, **io_info)


def _test_transpose(ishape, axes=None):
    data = np.random.uniform(size=ishape).astype(np.float32)

    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=data.shape, dtype=data.dtype, name="transpose_data")

        if axes is None:
            tf.transpose(in1)
        else:
            tf.transpose(in1, perm=axes)

        io_info = {
            "in_data": data,
            "in_name": "transpose_data:0",
            "out_name": "transpose:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def _test_tranapose_axes_input(ishape, axes):
    data = np.random.uniform(size=ishape).astype(np.float32)
    axes_np = np.array(axes).astype(np.int32)

    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=data.shape, dtype=data.dtype, name="transpose_data")

        const1 = tf.constant(axes_np, dtype=tf.int32)

        # make axes an input to tf.transpose, but not an input to the graph,
        # so it can be extracted with infer_value_simulated
        axes = tf.reverse(const1, axis=[-1])
        tf.transpose(in1, axes)
        io_info = {
            "in_data": data,
            "in_name": "transpose_data:0",
            "out_name": "transpose:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_transpose():
    """test tensorflow translator for transpose"""

    _test_transpose((2, 3, 4), (1, 2, 0))
    _test_transpose((2, 3, 4))
    _test_tranapose_axes_input((2, 3, 4), (1, 2, 0))
    _test_tranapose_axes_input((2, 3, 4, 5), (3, 0, 1, 2))


def _test_slice_operation_input(input_value, begin_value, size_value):
    input_data = np.array(input_value, dtype=np.float32)
    with tf.Graph().as_default():
        input_tensor = tf.placeholder(shape=input_data.shape, dtype=input_data.dtype, name="input")
        tf.slice(input_tensor, begin_value, size_value, name="slice_output")
        io_info = {
            "in_data": input_data,
            "in_name": "input:0",
            "out_name": "slice_output:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_slice():
    """test tensorflow translator for slice"""

    _test_slice_operation_input([1, 1], [0], [2])


def test_ceil():
    """test tensorflow translator for ceil"""

    ishape = (1, 3, 10, 10)
    inp_array = np.random.uniform(size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
        tf.ceil(in1)
        io_info = {
            "in_data": inp_array,
            "in_name": "Placeholder:0",
            "out_name": "Ceil:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_floor():
    """test tensorflow translator for floor"""

    ishape = (1, 3, 10, 10)
    inp_array = np.random.uniform(size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
        tf.floor(in1)
        io_info = {
            "in_data": inp_array,
            "in_name": "Placeholder:0",
            "out_name": "Floor:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_relu():
    """test tensorflow translator for relu"""

    ishape = (1, 3, 10, 10)
    inp_array = np.random.uniform(-5, 5, size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
        tf.nn.relu(in1)
        io_info = {
            "in_data": inp_array,
            "in_name": "Placeholder:0",
            "out_name": "Relu:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_elu():
    """test tensorflow translator for elu"""

    ishape = (1, 3, 10, 10)
    inp_array = np.random.uniform(-5, 5, size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
        tf.nn.elu(in1)
        io_info = {
            "in_data": inp_array,
            "in_name": "Placeholder:0",
            "out_name": "Elu:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_selu():
    """test tensorflow translator for selu"""

    ishape = (1, 3, 10, 10)
    inp_array = np.random.uniform(-5, 5, size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
        tf.nn.selu(in1)
        io_info = {
            "in_data": inp_array,
            "in_name": "Placeholder:0",
            "out_name": "Selu:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_tanh():
    """test tensorflow translator for tanh"""

    ishape = (1, 3, 10, 10)
    inp_array = np.random.uniform(-5, 5, size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
        tf.nn.tanh(in1)
        io_info = {
            "in_data": inp_array,
            "in_name": "Placeholder:0",
            "out_name": "Tanh:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_softmax():
    """test tensorflow translator for softmax"""

    def check_softmax(in_shape, axis, dtype):
        np_data = np.random.uniform(-100, 100, size=in_shape).astype(dtype)
        tf.reset_default_graph()
        with tf.Graph().as_default():
            in_data = tf.placeholder(dtype, in_shape, name="in_data")
            tf.nn.softmax(in_data, axis=axis, name="Softmax")
            io_info = {
                "in_data": np_data,
                "in_name": "in_data:0",
                "out_name": "Softmax:0",
            }
            graph_def, golden = get_graph_def(**io_info)
        verify_model(graph_def, golden, **io_info)

    check_softmax((2, 3, 5), 2, "float32")
    check_softmax((2, 3, 5), -1, "float32")


def test_round():
    """test tensorflow translator for round"""

    np_data = np.random.uniform(-10, 10, size=(5, 7)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (5, 7), name="in_data")
        tf.round(in_data, name="round")
        io_info = {
            "in_data": np_data,
            "in_name": "in_data:0",
            "out_name": "round:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_abs():
    """test tensorflow translator for abs"""

    np_data = np.random.uniform(1, 100, size=(9, 11)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (9, 11), name="in_data")
        tf.math.abs(in_data, name="abs")
        io_info = {
            "in_data": np_data,
            "in_name": "in_data:0",
            "out_name": "abs:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_squared_difference():
    """test tensorflow translator for squared_difference"""

    ishape = (1, 3, 10, 14)
    inp_array_a = np.random.uniform(-5, 5, size=ishape).astype(np.float32)
    inp_array_b = np.random.uniform(-5, 5, size=ishape).astype(np.float32)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array_a.shape, dtype=inp_array_a.dtype, name="in1")
        in2 = tf.placeholder(shape=inp_array_b.shape, dtype=inp_array_b.dtype, name="in2")
        out = tf.math.squared_difference(in1, in2)
        io_info = {
            "in_data": [inp_array_a, inp_array_b],
            "in_name": [in1.name, in2.name],
            "out_name": out.name,
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_sign():
    """test tensorflow translator for sign"""

    np_data = np.random.uniform(-10, 10, size=(5, 7, 11)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (5, 7, 11), name="in_data")
        tf.sign(in_data, name="sign")
        io_info = {"in_data": [np_data], "in_name": ["in_data:0"], "out_name": "sign:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_square():
    """test tensorflow translator for square"""

    np_data = np.random.uniform(1, 100, size=(2, 3, 5)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (2, 3, 5), name="in_data")
        tf.square(in_data, name="square")
        io_info = {"in_data": [np_data], "in_name": ["in_data:0"], "out_name": "square:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_pow_exp():
    """test tensorflow translator for pow && exp"""

    np_in1 = np.random.uniform(-2, 2, size=(5, 7, 11)).astype(np.float32)
    np_in2 = np.random.uniform(-2, 2, size=(5, 7, 11)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in1 = tf.placeholder(tf.float32, (5, 7, 11), name="in1")
        in2 = tf.placeholder(tf.float32, (5, 7, 11), name="in2")
        in3 = tf.pow(in1, in2, name="pow")
        _ = tf.exp(in3, name="exp")
        io_info = {"in_data": [np_in1, np_in2], "in_name": ["in1:0", "in2:0"], "out_name": "exp:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_unary():
    """test tensorflow translator for unary"""

    def _test_unary(op, a_min=1, a_max=5, dtype=np.float32):
        """test unary operators"""
        np_data = np.random.uniform(a_min, a_max, size=(2, 3, 5)).astype(dtype)
        tf.reset_default_graph()
        with tf.Graph().as_default():
            in_data = tf.placeholder(dtype, (2, 3, 5), name="in_data")
            out = op(in_data)
            io_info = {"in_data": [np_data], "in_name": ["in_data:0"], "out_name": out.name}
            graph_def, golden = get_graph_def(**io_info)
        verify_model(graph_def, golden, **io_info)

    _test_unary(tf.acos, -1, 1)
    _test_unary(tf.asin, -1, 1)
    _test_unary(tf.atanh, -1, 1)
    _test_unary(tf.sinh)
    _test_unary(tf.cosh)
    _test_unary(tf.acosh)
    _test_unary(tf.asinh)
    _test_unary(tf.atan)
    _test_unary(tf.sin)
    _test_unary(tf.cos)
    _test_unary(tf.tan)
    _test_unary(tf.tanh)
    _test_unary(tf.erf)
    _test_unary(tf.log)


def test_atan2():
    """test tensorflow translator for atan2"""

    tf.disable_eager_execution()
    np_data_1 = np.random.uniform(1, 100, size=(2, 3, 5)).astype(np.float32)
    np_data_2 = np.random.uniform(1, 100, size=(2, 3, 5)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data_1 = tf.placeholder(tf.float32, (2, 3, 5), name="in_data_1")
        in_data_2 = tf.placeholder(tf.float32, (2, 3, 5), name="in_data_2")
        tf.atan2(in_data_1, in_data_2, name="atan2")
        io_info = {
            "in_data": [np_data_1, np_data_2],
            "in_name": ["in_data_1:0", "in_data_2:0"],
            "out_name": "atan2:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_expm1():
    """test tensorflow translator for expm1"""

    def _test_expm1(shape):
        tf.disable_eager_execution()
        np_data = np.random.uniform(1, 10, size=shape).astype(np.float32)
        tf.reset_default_graph()
        with tf.Graph().as_default():
            in_data = tf.placeholder(tf.float32, shape, name="in_data")
            tf.expm1(in_data, name="expm1")
            io_info = {"in_data": [np_data], "in_name": ["in_data:0"], "out_name": "expm1:0"}
            graph_def, golden = get_graph_def(**io_info)
        verify_model(graph_def, golden, **io_info)

    _test_expm1([2, 5, 2, 5])


def test_softsign():
    """test tensorflow translator for softsign"""

    def _test_softsign(shape):
        tf.disable_eager_execution()
        np_data = np.random.uniform(1, 100, size=shape).astype(np.float32)
        tf.reset_default_graph()
        with tf.Graph().as_default():
            in_data = tf.placeholder(tf.float32, shape, name="in_data")
            tf.nn.softsign(in_data, name="softsign")
            io_info = {"in_data": [np_data], "in_name": ["in_data:0"], "out_name": "softsign:0"}
            graph_def, golden = get_graph_def(**io_info)
        verify_model(graph_def, golden, **io_info)

    _test_softsign([2, 5, 2, 5])


def test_rint():
    """test tensorflow translator for rint"""

    def _test_rint(shape):
        tf.disable_eager_execution()
        np_data = np.random.uniform(-100, 100, size=shape).astype(np.float32)
        tf.reset_default_graph()
        with tf.Graph().as_default():
            in_data = tf.placeholder(tf.float32, shape, name="in_data")
            tf.math.rint(in_data, name="rint")
            io_info = {"in_data": [np_data], "in_name": ["in_data:0"], "out_name": "rint:0"}
            graph_def, golden = get_graph_def(**io_info)
        verify_model(graph_def, golden, **io_info)

    _test_rint([2, 5, 2, 5])


def test_negative():
    """test tensorflow translator for neg"""

    np_data = np.random.uniform(-100, 255, size=(224, 224, 3)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (224, 224, 3), name="in_data")
        tf.negative(in_data, name="negative")
        io_info = {"in_data": [np_data], "in_name": ["in_data:0"], "out_name": "negative:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_log_softmax():
    """test tensorflow translator for log_softmax"""

    np_data = np.random.uniform(1, 100, size=(9, 11)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (9, 11), name="in_data")
        tf.math.log_softmax(in_data, name="LogSoftmax")
        io_info = {"in_data": [np_data], "in_name": ["in_data:0"], "out_name": "LogSoftmax:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_softplus():
    """test tensorflow translator for softplus"""

    np_data = np.random.uniform(1, 10, size=(2, 3, 5)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (2, 3, 5), name="in_data")
        tf.nn.softplus(in_data, name="softplus")
        io_info = {"in_data": [np_data], "in_name": ["in_data:0"], "out_name": "softplus:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_rsqrt():
    """test tensorflow translator for rsqrt"""

    np_data = np.random.uniform(1, 100, size=(5, 7, 11)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (5, 7, 11), name="in_data")
        tf.rsqrt(in_data, name="rsqrt")
        io_info = {"in_data": [np_data], "in_name": ["in_data:0"], "out_name": "rsqrt:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_sqrt():
    """test tensorflow translator for sqrt"""

    np_data = np.random.uniform(1, 100, size=(5, 7, 11)).astype(np.float32)
    tf.reset_default_graph()
    with tf.Graph().as_default():
        in_data = tf.placeholder(tf.float32, (5, 7, 11), name="in_data")
        tf.sqrt(in_data, name="sqrt")
        io_info = {"in_data": [np_data], "in_name": ["in_data:0"], "out_name": "sqrt:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_mean():
    """test tensorflow translator for mean"""

    def check_mean(ishape, **kwargs):
        inp_array = np.random.uniform(size=ishape).astype(np.float32)
        with tf.Graph().as_default():
            in1 = tf.placeholder(shape=inp_array.shape, dtype=inp_array.dtype)
            tf.keras.backend.mean(in1, **kwargs)
            io_info = {"in_data": inp_array, "in_name": "Placeholder:0", "out_name": "Mean:0"}
            graph_def, golden = get_graph_def(**io_info)
        verify_model(graph_def, golden, **io_info)

    check_mean((10, 8, 16, 32))
    check_mean((10, 8, 16, 32), axis=(2, 3))
    check_mean((10, 8, 16, 32), axis=(1, 2), keepdims=True)


def test_reduce():
    """test tensorflow translator for reduce"""

    def _check_op(tf_op, ishape, axis, keepdims):
        tf.reset_default_graph()
        np_data = np.random.uniform(size=ishape).astype("float32")
        if tf_op == tf.math.reduce_prod:
            axis = 1
            np_data = np_data.reshape(1, -1)
        with tf.Graph().as_default():
            in_data = tf.placeholder(shape=np_data.shape, dtype="float32", name="in_data")
            reduce_op = tf_op(in_data, axis=axis, keepdims=keepdims, name="reduce_op")
            io_info = {"in_data": np_data, "in_name": "in_data:0", "out_name": reduce_op.name}
            graph_def, golden = get_graph_def(**io_info)
        verify_model(graph_def, golden, **io_info)

    def _test_math_op(op):
        _check_op(op, (8, 16, 32), axis=(-1), keepdims=False)
        _check_op(op, (1, 8, 8, 3), axis=(2, 3), keepdims=True)

    _test_math_op(tf.math.reduce_max)
    _test_math_op(tf.math.reduce_min)
    _test_math_op(tf.math.reduce_prod)
    _test_math_op(tf.math.reduce_variance)
    _test_math_op(tf.math.reduce_std)
    _test_math_op(tf.math.reduce_logsumexp)
    if package_version.parse(tf.VERSION) >= package_version.parse("1.15.0"):
        _test_math_op(tf.math.reduce_euclidean_norm)


def _test_rel_op(data, func):
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=data[0].shape, dtype=data[0].dtype, name="in1")
        in2 = tf.placeholder(shape=data[1].shape, dtype=data[1].dtype, name="in2")
        op = func(in1, in2, name="op")
        _ = tf.cast(op, tf.int32, name="out1")
        io_info = {
            "in_data": [data[0], data[1]],
            "in_name": ["in1:0", "in2:0"],
            "out_name": "out1:0",
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_rel_ops():
    """test tensorflow translator for relation"""

    t_1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    t_2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    _test_rel_op([t_1, t_2], math_ops.less)
    _test_rel_op([t_1, t_2], math_ops.greater)
    _test_rel_op([t_1, t_2], math_ops.less_equal)
    _test_rel_op([t_1, t_2], math_ops.greater_equal)
    _test_rel_op([t_1, t_2], math_ops.equal)
    _test_rel_op([t_1, t_2], math_ops.not_equal)


def _test_expand_dims(data, axis):
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=data.shape, dtype=data.dtype, name="in1")
        out = tf.expand_dims(in1, axis)
        io_info = {"in_data": data, "in_name": in1.name, "out_name": out.name}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_expand_dims():
    """test tensorflow translator for expand_dims"""

    _test_expand_dims(np.array([1]), -1)
    _test_expand_dims(np.array([[1], [2]]), 1)


def test_maximum():
    """test tensorflow translator for maximum"""

    def check_maximum(lh_shape, rh_shape, dtype):
        tf.reset_default_graph()
        lh_data = np.random.uniform(size=lh_shape).astype(dtype)
        rh_data = np.random.uniform(size=rh_shape).astype(dtype)
        with tf.Graph().as_default():
            lft_data = tf.placeholder(shape=lh_data.shape, dtype=dtype, name="lft_data")
            rgt_data = tf.placeholder(shape=rh_data.shape, dtype=dtype, name="rgt_data")
            tf.math.maximum(lft_data, rgt_data, name="maximum")
            io_info = {
                "in_data": [lh_data, rh_data],
                "in_name": ["lft_data:0", "rgt_data:0"],
                "out_name": "maximum:0",
            }
            graph_def, golden = get_graph_def(**io_info)
        verify_model(graph_def, golden, **io_info)

    check_maximum((10, 8, 16, 32), (10, 8, 16, 32), dtype="float32")


def test_minimum():
    """test tensorflow translator for minimum"""

    def check_minimum(lh_shape, rh_shape, dtype):
        tf.reset_default_graph()
        lh_data = np.random.uniform(size=lh_shape).astype(dtype)
        rh_data = np.random.uniform(size=rh_shape).astype(dtype)
        with tf.Graph().as_default():
            lft_data = tf.placeholder(shape=lh_data.shape, dtype=dtype, name="lft_data")
            rgt_data = tf.placeholder(shape=rh_data.shape, dtype=dtype, name="rgt_data")
            tf.math.minimum(lft_data, rgt_data, name="minimum")
            io_info = {
                "in_data": [lh_data, rh_data],
                "in_name": ["lft_data:0", "rgt_data:0"],
                "out_name": "minimum:0",
            }
            graph_def, golden = get_graph_def(**io_info)
        verify_model(graph_def, golden, **io_info)

    check_minimum((10, 8, 16, 32), (10, 8, 16, 32), dtype="float32")


def _test_add_n(inputs):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        temp = []
        for each in inputs:
            temp.append(tf.placeholder(shape=each.shape, dtype=each.dtype))
        output = tf.add_n(temp)
        io_info = {
            "in_data": list(inputs),
            "in_name": [each.name for each in temp],
            "out_name": output.name,
        }
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_add_n():
    """test tensorflow translator for add_n"""

    x_in = np.random.randint(1, 100, size=(3, 3, 3), dtype=np.int32)
    y_in = np.random.randint(1, 100, size=(3, 3, 3), dtype=np.int32)
    z_in = np.random.randint(1, 100, size=(3, 3, 3), dtype=np.int32)
    m_dim, n_dim, o_dim = x_in.astype(np.float32), y_in.astype(np.float32), z_in.astype(np.float32)
    in0 = x_in
    in1 = [x_in, y_in]
    in2 = (x_in, y_in, z_in)
    in3 = m_dim
    in4 = [m_dim, n_dim]
    in5 = (m_dim, n_dim, o_dim)
    _test_add_n(in0)
    _test_add_n(in1)
    _test_add_n(in2)
    _test_add_n(in3)
    _test_add_n(in4)
    _test_add_n(in5)


def _test_identityn(data_np_list):
    with tf.Graph().as_default():
        data_tensors = []
        data_tensors_name = []
        for index, data_np in enumerate(data_np_list):
            tensor_name = f"data_{index}"
            data_tensors_name.append(tensor_name + ":0")
            data_tensors.append(
                tf.placeholder(shape=data_np.shape, dtype=str(data_np.dtype), name=tensor_name)
            )

        output = tf.identity_n(data_tensors)
        output_names = [out.name for out in output]
        io_info = {"in_data": data_np_list, "in_name": data_tensors_name, "out_name": output_names}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info, use_out_name=False)


def test_identityn():
    """test tensorflow translator for identityn"""

    data_np_list = [
        np.array([[1, 1], [0, 3], [0, 1], [2, 0], [3, 1]], dtype=np.int64),
        np.array([1, 2, 3, 4, 5], dtype=np.int64),
        np.array([5, 6], dtype=np.int64),
    ]
    _test_identityn(data_np_list)
    data_np_list = [
        np.array([[1, 1], [0, 3], [2, 0], [3, 1]], dtype=np.int64),
        np.array([1, 2, 3, 4], dtype=np.int64),
        np.array([5, 6], dtype=np.int64),
        np.array([True, False, True]),
    ]
    _test_identityn(data_np_list)


def _test_infinity(tf_op, name):
    """test operator infinity ops"""

    # Only float types are allowed in Tensorflow for isfinite and isinf
    # float16 is failing on cuda
    tf_dtypes = ["float32", "float64"]  # pylint: disable=redefined-outer-name
    for tf_dtype in tf_dtypes:
        shape = (8, 8)
        data = np.random.uniform(size=shape).astype(tf_dtype)
        data.ravel()[np.random.choice(data.size, int(data.size * 0.5), replace=False)] = np.infty
        data.ravel()[np.random.choice(data.size, int(data.size * 0.5), replace=False)] = np.nan

        tf.reset_default_graph()
        in_data = tf.placeholder(tf_dtype, shape, name="in_data")
        tf_op(in_data, name=name)
        io_info = {"in_data": data, "in_name": "in_data:0", "out_name": f"{name}:0"}
        graph_def, golden = get_graph_def(**io_info)
    verify_model(graph_def, golden, **io_info)


def test_infinity():
    """test tensorflow translator for infinity"""

    _test_infinity(tf.is_inf, "isinf")
    _test_infinity(tf.is_finite, "isfinite")
    _test_infinity(tf.is_nan, "isnan")


if __name__ == "__main__":
    tvm.testing.main()
