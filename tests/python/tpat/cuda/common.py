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

import ctypes
import os
import sys

import numpy as np
import onnx
import pycuda.autoinit
import pycuda.driver as cuda
import pytest
import tensorflow as tf
import tensorflow.compat.v1 as tf
import tensorrt as trt
from onnx import TensorProto, helper, mapping, numpy_helper
from onnx.backend.test.case.node import _extract_value_info

from tvm import tpat

from .trt import allocate_buffers, build_engine, do_inference, load_plugin

tf.disable_v2_behavior()

I_GPU = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(I_GPU)
np.random.seed(0)
ITERATIONS = 10
INPUT_MODEL_FILE = "test_op_plugin.onnx"
OUTPUT_MODEL_FILE = "test_op_trt.onnx"

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
BATCH_SIZE = 1


# Simple helper data class that's a little nicer to use than a 2-tuple.


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
    # if len(input_node) == 1 and input_node[0] == "":
    #     output_data = sess.run(tensor)
    # else:
    output_data = sess.run(tensor, input_dict)
    return output_data


def verify_tf_with_trt_result(in_data, in_name, out_name, op_name):
    def name_without_num(name):
        return name.split(":")[0] if ":" in name else name

    out_name = convert_to_list(out_name)
    out_node = [name_without_num(name) for name in out_name]
    in_data = convert_to_list(in_data)
    in_name = convert_to_list(in_name)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf_result = run_tf_graph(sess, in_data, in_name, out_name)
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, out_node)
        with open("./test_op_{}.pb".format(op_name), "wb") as ofile:
            ofile.write(frozen_graph.SerializeToString())
    os.system(
        "python3 -m tf2onnx.convert --input ./test_op_{}.pb --inputs {} --outputs {} --output {} --opset 11".format(
            op_name, str(",").join(in_name), str(",").join(out_name), INPUT_MODEL_FILE
        )
    )
    ops_name = [op_name]

    _, trt_plugin_names = tpat.cuda.pipeline(
        INPUT_MODEL_FILE, ops_name, False, "./log_db", OUTPUT_MODEL_FILE
    )

    load_plugin(trt_plugin_names)
    engine = build_engine(OUTPUT_MODEL_FILE, trt_engine_datatype=trt.DataType.HALF)

    inputs, outputs, bindings, stream = allocate_buffers(engine)
    with engine.create_execution_context() as context:
        for i in range(len(inputs)):
            input_data = in_data[i].ravel()
            np.copyto(inputs[i].host, input_data)

        trt_result = do_inference(
            context,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
        )

    ret = True
    if len(trt_result) == 1:
        ret = compare_tf_trt_result(tf_result, trt_result)
    else:
        for i in range(len(trt_result)):
            ret &= compare_tf_trt_result(tf_result[i], trt_result[i])
    assert ret, "result check False"
    return ret


def compare_tf_trt_result(tf_result, trt_result):
    print(tf_result)
    print("================")
    print(trt_result)
    tf_reshape = np.array(tf_result).reshape(-1)
    trt_reshape = np.array(trt_result).reshape(-1)

    if (
        isinstance(tf_result, list)
        and isinstance(trt_result, list)
        and len(tf_result) > 0
        and len(trt_result) > 0
        and np.isnan(tf_result[0]).any()
        and np.isnan(trt_result[0]).any()
    ):
        return True
    elif (
        isinstance(tf_result, list)
        and isinstance(trt_result, list)
        and len(tf_result) > 0
        and len(trt_result) > 0
        and np.isinf(tf_result[0]).any()
        and np.isinf(trt_result[0]).any()
    ):
        return True
    elif np.isnan(tf_reshape).any() and np.isnan(trt_reshape).any():
        return True
    print(
        "trt cross_check output ",
        str(np.allclose(tf_reshape.flatten(), trt_reshape.flatten(), atol=1e-5)),
        flush=True,
    )
    return bool(np.allclose(tf_reshape.flatten(), trt_reshape.flatten(), atol=1e-5))


def get_onnxruntime_output(model, inputs):
    import onnxruntime.backend

    rep = onnxruntime.backend.prepare(model, "GPU")
    if isinstance(inputs, list) and len(inputs) == 1:
        inp = inputs[0]
    else:
        inp = inputs
    output = rep.run(inp)
    # Unpack output if there's only a single value.
    if len(output) == 1:
        output = output[0]
    return output


def verify_with_ort_with_trt(
    model,
    inputs,
    op_name,
    opset=None,
    dtype="float32",
    opt_level=1,
    np_result=None,
    use_vm=False,
    layout=0,
):
    if opset is not None:
        model.opset_import[0].version = opset
    onnx.save(model, INPUT_MODEL_FILE)
    if np_result is None:
        ort_result = get_onnxruntime_output(model, inputs)
    else:
        ort_result = np_result

    in_data = convert_to_list(inputs)
    ops_name = [op_name]

    _, trt_plugin_names = tpat.cuda.pipeline(
        INPUT_MODEL_FILE, ops_name, False, "./log_db", OUTPUT_MODEL_FILE
    )

    load_plugin(trt_plugin_names)
    engine = build_engine(OUTPUT_MODEL_FILE, trt_engine_datatype=trt.DataType.HALF)

    inputs, outputs, bindings, stream = allocate_buffers(engine)
    with engine.create_execution_context() as context:
        for i in range(len(inputs)):
            input_data = in_data[i].ravel()
            np.copyto(inputs[i].host, input_data)

        trt_result = do_inference(
            context,
            bindings=bindings,
            inputs=inputs,
            outputs=outputs,
            stream=stream,
        )

    ret = True
    if len(trt_result) == 1:
        ret = compare_tf_trt_result(ort_result, trt_result)
    else:
        # ret &= compare_tf_trt_result(ort_result[0], trt_result[0])
        for i in range(len(trt_result)):
            ret &= compare_tf_trt_result(ort_result[i], trt_result[i])
    assert ret, "result check False"
    return ret


def make_constant_node(name, data_type, dims, vals):
    return helper.make_node(
        "Constant",
        inputs=[],
        outputs=[name],
        value=helper.make_tensor(name=name, data_type=data_type, dims=dims, vals=vals),
    )


def make_onnx_model(node, inputs, outputs, name, **kwargs):
    present_inputs = [x for x in node.input if (x != "")]
    present_outputs = [x for x in node.output if (x != "")]
    input_type_protos = [None] * len(inputs)
    if "input_type_protos" in kwargs:
        input_type_protos = kwargs[str("input_type_protos")]
        del kwargs[str("input_type_protos")]
    output_type_protos = [None] * len(outputs)
    if "output_type_protos" in kwargs:
        output_type_protos = kwargs[str("output_type_protos")]
        del kwargs[str("output_type_protos")]
    inputs_vi = [
        _extract_value_info(arr, arr_name, input_type)
        for arr, arr_name, input_type in zip(inputs, present_inputs, input_type_protos)
    ]
    outputs_vi = [
        _extract_value_info(arr, arr_name, output_type)
        for arr, arr_name, output_type in zip(outputs, present_outputs, output_type_protos)
    ]
    graph = helper.make_graph(nodes=[node], name=name, inputs=inputs_vi, outputs=outputs_vi)
    kwargs[str("producer_name")] = "TRTPluginAutoGen-test"
    model = onnx.helper.make_model(graph, **kwargs)
    return model


def op_expect(node, inputs, outputs, op_type, op_name, np_result=None):
    model = make_onnx_model(node, inputs=inputs, outputs=outputs, name="test_{}".format(op_type))
    verify_with_ort_with_trt(model, inputs, op_name, np_result=np_result)


# ====================================================================================
# ---UnitTest
# ====================================================================================


def test_abs():
    op_name = "abs_0"
    op_type = "Abs"
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = abs(x)
    node = helper.make_node(op_type, inputs=["x"], outputs=["y"], name=op_name)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_acos():
    op_name = "acos_0"
    op_type = "Acos"
    node = onnx.helper.make_node("Acos", inputs=["x"], outputs=["y"], name=op_name)
    x = np.array([-0.5, 0, 0.5]).astype(np.float32)
    y = np.arccos(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "acos_1"
    op_type = "Acos"
    node = onnx.helper.make_node("Acos", inputs=["x"], outputs=["y"], name=op_name)
    x = np.random.rand(3, 4, 5).astype(np.float32)
    y = np.arccos(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


@pytest.mark.skip(reason="TensorRT segmentfault")
def test_and():
    op_name = "and_0"
    op_type = "And"
    node = onnx.helper.make_node("And", inputs=["x", "y"], outputs=["and"], name=op_name)
    # 2d
    x = (np.random.randn(3, 4) > 0).astype(bool)
    y = (np.random.randn(3, 4) > 0).astype(bool)
    z = np.logical_and(x, y)
    op_expect(node, inputs=[x, y], outputs=[z], op_type=op_type, op_name=op_name)

    op_name = "and_1"
    op_type = "And"
    node = onnx.helper.make_node("And", inputs=["x", "y"], outputs=["and"], name=op_name)
    x = (np.random.randn(3, 4, 5) > 0).astype(bool)
    y = (np.random.randn(3, 4, 5) > 0).astype(bool)
    z = np.logical_and(x, y)
    op_expect(node, inputs=[x, y], outputs=[z], op_type=op_type, op_name=op_name)

    op_name = "and_2"
    op_type = "And"
    node = onnx.helper.make_node("And", inputs=["x", "y"], outputs=["and"], name=op_name)
    x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
    y = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
    z = np.logical_and(x, y)
    op_expect(node, inputs=[x, y], outputs=[z], op_type=op_type, op_name=op_name)


def test_add():
    op_name = "add_0"
    op_type = "Add"
    node = onnx.helper.make_node("Add", inputs=["x", "y"], outputs=["sum"], name=op_name)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    op_expect(node, inputs=[x, y], outputs=[x + y], op_type=op_type, op_name=op_name)

    op_name = "add_1"
    op_type = "Add"
    node = onnx.helper.make_node("Add", inputs=["x", "y"], outputs=["sum"], name=op_name)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    op_expect(node, inputs=[x, y], outputs=[x + y], op_type=op_type, op_name=op_name)


def test_argmax():
    op_type = "ArgMax"
    op_name = "argmax_0"
    data = np.array([[2, 1, 3, 10], [3, 4, 5, 6]], dtype=np.float32)
    keepdims = 1
    axis = -1
    node = onnx.helper.make_node(
        "ArgMax",
        inputs=["data"],
        outputs=["result"],
        keepdims=keepdims,
        axis=axis,
        name=op_name,
    )

    # result: [[1], [1]]
    from onnx.backend.test.case.node.argmax import argmax_use_numpy

    result = argmax_use_numpy(data, keepdims=keepdims, axis=axis)
    op_expect(node, inputs=[data], outputs=[result], op_type=op_type, op_name=op_name)

    op_name = "argmax_1"
    node = onnx.helper.make_node(
        "ArgMax",
        inputs=["data"],
        outputs=["result"],
        keepdims=keepdims,
        axis=axis,
        name=op_name,
    )

    data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
    # result's shape: [1, 3, 4]
    result = argmax_use_numpy(data, keepdims=keepdims, axis=axis)
    op_expect(node, inputs=[data], outputs=[result], op_type=op_type, op_name=op_name)


def test_argmin():
    op_type = "ArgMin"
    op_name = "argmin_0"
    data = np.array([[2, 1], [3, 10]], dtype=np.float32)
    keepdims = 1
    axis = 1
    node = onnx.helper.make_node(
        "ArgMin",
        inputs=["data"],
        outputs=["result"],
        keepdims=keepdims,
        axis=axis,
        name=op_name,
    )

    # result: [[1], [1]]
    from onnx.backend.test.case.node.argmin import argmin_use_numpy

    result = argmin_use_numpy(data, keepdims=keepdims, axis=axis)
    op_expect(node, inputs=[data], outputs=[result], op_type=op_type, op_name=op_name)


def test_asin():
    op_name = "asin_0"
    op_type = "Asin"
    node = onnx.helper.make_node("Asin", inputs=["x"], outputs=["y"], name=op_name)

    x = np.array([-0.5, 0, 0.5]).astype(np.float32)
    y = np.arcsin(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "asin_1"
    op_type = "Asin"
    node = onnx.helper.make_node("Asin", inputs=["x"], outputs=["y"], name=op_name)

    x = np.random.rand(3, 4, 5).astype(np.float32)
    y = np.arcsin(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_asinh():
    op_name = "asinh_0"
    op_type = "Asinh"
    node = onnx.helper.make_node("Asinh", inputs=["x"], outputs=["y"], name=op_name)

    x = np.array([-1, 0, 1]).astype(np.float32)
    y = np.arcsinh(x)  # expected output [-0.88137358,  0.,  0.88137358]
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "asinh_1"
    op_type = "Asinh"
    node = onnx.helper.make_node("Asinh", inputs=["x"], outputs=["y"], name=op_name)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.arcsinh(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_atan():
    op_type = "Atan"
    op_name = "atan_0"
    node = onnx.helper.make_node("Atan", inputs=["x"], outputs=["y"], name=op_name)

    x = np.array([-1, 0, 1]).astype(np.float32)
    y = np.arctan(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_type = "Atan"
    op_name = "atan_1"
    node = onnx.helper.make_node("Atan", inputs=["x"], outputs=["y"], name=op_name)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.arctan(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_atanh():
    op_name = "atanh_0"
    op_type = "Atanh"
    node = onnx.helper.make_node("Atanh", inputs=["x"], outputs=["y"], name=op_name)

    x = np.array([-0.5, 0, 0.5]).astype(np.float32)
    y = np.arctanh(x)  # expected output [-0.54930615,  0.,  0.54930615]
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "atanh_1"
    op_type = "Atanh"
    node = onnx.helper.make_node("Atanh", inputs=["x"], outputs=["y"], name=op_name)
    x = np.random.uniform(0.0, 1.0, (3, 4, 5)).astype(np.float32)
    y = np.arctanh(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_averagepool():
    op_name = "averagepool_1d_default"
    op_type = "AveragePool"
    """
    input_shape: [1, 3, 32]
    output_shape: [1, 3, 31]
    """
    node = onnx.helper.make_node(
        "AveragePool", inputs=["x"], outputs=["y"], kernel_shape=[2], name=op_name
    )
    x = np.random.randn(1, 3, 32).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = [2]
    strides = [1]
    from onnx.backend.test.case.node.pool_op_common import get_output_shape, pool

    out_shape = get_output_shape("VALID", x_shape[2:], kernel_shape, strides)
    padded = x
    y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0], "AVG")
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "averagepool_2d_ceil"
    op_type = "AveragePool"
    node = onnx.helper.make_node(
        "AveragePool",
        inputs=["x"],
        outputs=["y"],
        kernel_shape=[3, 3],
        strides=[2, 2],
        ceil_mode=True,
        name=op_name,
    )
    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ]
    ).astype(np.float32)
    y = np.array([[[[6, 7.5], [12, 13.5]]]]).astype(np.float32)

    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


@pytest.mark.skip(reason="TensorRT segmentfault")
def test_batchnormalization():
    op_name = "batchnormalization_0"
    op_type = "BatchNormalization"
    # input size: (2, 3, 4, 5)
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    mean = np.random.randn(3).astype(np.float32)
    var = np.random.rand(3).astype(np.float32)
    from onnx.backend.test.case.node.batchnorm import _batchnorm_test_mode

    y = _batchnorm_test_mode(x, s, bias, mean, var).astype(np.float32)

    node = onnx.helper.make_node(
        "BatchNormalization",
        inputs=["x", "s", "bias", "mean", "var"],
        outputs=["y"],
        name=op_name,
    )

    # output size: (2, 3, 4, 5)
    op_expect(
        node,
        inputs=[x, s, bias, mean, var],
        outputs=[y],
        op_type=op_type,
        op_name=op_name,
    )


def test_ceil():
    op_name = "ceil_0"
    op_type = "Ceil"
    node = onnx.helper.make_node("Ceil", inputs=["x"], outputs=["y"], name=op_name)

    x = np.array([-1.5, 1.2]).astype(np.float32)
    y = np.ceil(x)  # expected output [-1., 2.]
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "ceil_1"
    op_type = "Ceil"
    node = onnx.helper.make_node("Ceil", inputs=["x"], outputs=["y"], name=op_name)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.ceil(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_celu():
    op_name = "celu_0"
    op_type = "Celu"
    alpha = 2.0
    node = onnx.helper.make_node("Celu", inputs=["X"], outputs=["Y"], alpha=alpha, name=op_name)

    input_data = np.array(
        [
            [
                [[0.8439683], [0.5665144], [0.05836735]],
                [[0.02916367], [0.12964272], [0.5060197]],
                [[0.79538304], [0.9411346], [0.9546573]],
            ],
            [
                [[0.17730942], [0.46192095], [0.26480448]],
                [[0.6746842], [0.01665257], [0.62473077]],
                [[0.9240844], [0.9722341], [0.11965699]],
            ],
            [
                [[0.41356155], [0.9129373], [0.59330076]],
                [[0.81929934], [0.7862604], [0.11799799]],
                [[0.69248444], [0.54119414], [0.07513223]],
            ],
        ],
        dtype=np.float32,
    )

    # Calculate expected output data
    positive_input = np.maximum(0, input_data)
    negative_input = np.minimum(0, alpha * (np.exp(input_data / alpha) - 1))
    expected_output = positive_input + negative_input

    op_expect(
        node,
        inputs=[input_data],
        outputs=[expected_output],
        op_type=op_type,
        op_name=op_name,
    )


def test_clip():
    op_name = "Clip_0"
    op_type = "Clip"
    node = onnx.helper.make_node("Clip", inputs=["x", "min", "max"], outputs=["y"], name=op_name)
    x = np.array([-2, 0, 2]).astype(np.float32)
    min_val = np.array([-1.0]).astype(np.float32)  # .float32(-1.0)
    max_val = np.array([1.0]).astype(np.float32)  # .float32(1.0)
    y = np.clip(x, min_val, max_val)  # expected output [-1., 0., 1.]
    op_expect(
        node,
        inputs=[x, min_val, max_val],
        outputs=[y],
        op_type=op_type,
        op_name=op_name,
    )

    op_name = "Clip_1"
    op_type = "Clip"
    node = onnx.helper.make_node("Clip", inputs=["x", "min", "max"], outputs=["y"], name=op_name)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, min_val, max_val)
    op_expect(
        node,
        inputs=[x, min_val, max_val],
        outputs=[y],
        op_type=op_type,
        op_name=op_name,
    )

    op_name = "Clip_2"
    op_type = "Clip"
    node = onnx.helper.make_node("Clip", inputs=["x", "min", "max"], outputs=["y"], name=op_name)
    min_val = np.array([-5.0]).astype(np.float32)  # .float32(-1.0)
    max_val = np.array([5.0]).astype(np.float32)  # .float32(1.0)
    op_name = "Clip_3"
    op_type = "Clip"
    node = onnx.helper.make_node("Clip", inputs=["x", "min", "max"], outputs=["y"], name=op_name)

    x = np.array([-1, 0, 1]).astype(np.float32)
    y = np.array([-1, 0, 1]).astype(np.float32)
    op_expect(
        node,
        inputs=[x, min_val, max_val],
        outputs=[y],
        op_type=op_type,
        op_name=op_name,
    )

    op_name = "Clip_4"
    op_type = "Clip"
    node = onnx.helper.make_node("Clip", inputs=["x", "min", "max"], outputs=["y"], name=op_name)
    x = np.array([-6, 0, 6]).astype(np.float32)
    y = np.array([-5, 0, 5]).astype(np.float32)
    op_expect(
        node,
        inputs=[x, min_val, max_val],
        outputs=[y],
        op_type=op_type,
        op_name=op_name,
    )

    op_name = "Clip_5"
    op_type = "Clip"
    node = onnx.helper.make_node("Clip", inputs=["x", "min", "max"], outputs=["y"], name=op_name)
    x = np.array([-1, 0, 6]).astype(np.float32)
    y = np.array([-1, 0, 5]).astype(np.float32)
    op_expect(
        node,
        inputs=[x, min_val, max_val],
        outputs=[y],
        op_type=op_type,
        op_name=op_name,
    )


def test_concat():
    test_cases = {
        "1d": ([1, 2], [3, 4]),
        "2d": ([[1, 2], [3, 4]], [[5, 6], [7, 8]]),
        "3d": (
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]],
        ),
    }  # type: Dict[Text, Sequence[Any]]

    for test_case, values_ in test_cases.items():
        values = [np.asarray(v, dtype=np.float32) for v in values_]
        for i in range(len(values[0].shape)):
            op_name = "concat_{}_{}".format(test_case, i)
            op_type = "Concat"
            in_args = ["value" + str(k) for k in range(len(values))]
            node = onnx.helper.make_node(
                "Concat",
                inputs=[s for s in in_args],
                outputs=["output"],
                axis=i,
                name=op_name,
            )
            output = np.concatenate(values, i)
            op_expect(
                node,
                inputs=[v for v in values],
                outputs=[output],
                op_type=op_type,
                op_name=op_name,
            )

        for i in range(-len(values[0].shape), 0):
            op_name = "concat_{}_1_{}".format(test_case, abs(i))
            op_type = "Concat"
            in_args = ["value" + str(k) for k in range(len(values))]
            node = onnx.helper.make_node(
                "Concat",
                inputs=[s for s in in_args],
                outputs=["output"],
                axis=i,
                name=op_name,
            )
            output = np.concatenate(values, i)
            op_expect(
                node,
                inputs=[v for v in values],
                outputs=[output],
                op_type=op_type,
                op_name=op_name,
            )


def test_conv():
    # ------Conv
    op_name, op_type = "test_basic_conv_with_padding", "Conv"
    x = np.array(
        [
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 5, 5) input tensor
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ]
            ]
        ]
    ).astype(np.float32)
    # NOCC:invalid-name(其他:onnx example)
    W = np.array(
        [
            [
                [
                    [1.0, 1.0, 1.0],  # (1, 1, 3, 3) tensor for convolution weights
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ]
        ]
    ).astype(np.float32)

    # Convolution with padding
    node_with_padding = onnx.helper.make_node(
        "Conv",
        inputs=["x", "W"],
        outputs=["y"],
        kernel_shape=[3, 3],
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=[1, 1, 1, 1],
        name=op_name,
    )
    y_with_padding = np.array(
        [
            [
                [
                    [12.0, 21.0, 27.0, 33.0, 24.0],  # (1, 1, 5, 5) output tensor
                    [33.0, 54.0, 63.0, 72.0, 51.0],
                    [63.0, 99.0, 108.0, 117.0, 81.0],
                    [93.0, 144.0, 153.0, 162.0, 111.0],
                    [72.0, 111.0, 117.0, 123.0, 84.0],
                ]
            ]
        ]
    ).astype(np.float32)
    op_expect(
        node_with_padding,
        inputs=[x, W],
        outputs=[y_with_padding],
        op_type=op_type,
        op_name=op_name,
    )

    op_name, op_type = "test_basic_conv_without_padding", "Conv"
    # Convolution without padding
    node_without_padding = onnx.helper.make_node(
        "Conv",
        inputs=["x", "W"],
        outputs=["y"],
        kernel_shape=[3, 3],
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=[0, 0, 0, 0],
        name=op_name,
    )
    y_without_padding = np.array(
        [
            [
                [
                    [54.0, 63.0, 72.0],  # (1, 1, 3, 3) output tensor
                    [99.0, 108.0, 117.0],
                    [144.0, 153.0, 162.0],
                ]
            ]
        ]
    ).astype(np.float32)
    op_expect(
        node_without_padding,
        inputs=[x, W],
        outputs=[y_without_padding],
        op_type=op_type,
        op_name=op_name,
    )

    # conv_with_autopad_same
    op_name, op_type = "test_conv_with_autopad_same", "Conv"
    x = np.array(
        [
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 5, 5) input tensor
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                ]
            ]
        ]
    ).astype(np.float32)
    # NOCC:invalid-name(其他:onnx example)
    W = np.array(
        [
            [
                [
                    [1.0, 1.0, 1.0],  # (1, 1, 3, 3) tensor for convolution weights
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ]
        ]
    ).astype(np.float32)

    # Convolution with auto_pad='SAME_LOWER' and strides=2
    node = onnx.helper.make_node(
        "Conv",
        inputs=["x", "W"],
        outputs=["y"],
        auto_pad="SAME_LOWER",
        kernel_shape=[3, 3],
        strides=[2, 2],
        name=op_name,
    )
    y = np.array([[[[12.0, 27.0, 24.0], [63.0, 108.0, 81.0], [72.0, 117.0, 84.0]]]]).astype(
        np.float32
    )
    op_expect(node, inputs=[x, W], outputs=[y], op_type=op_type, op_name=op_name)

    # conv_with_strides
    op_name, op_type = "test_conv_with_strides_padding", "Conv"
    x = np.array(
        [
            [
                [
                    [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 7, 5) input tensor
                    [5.0, 6.0, 7.0, 8.0, 9.0],
                    [10.0, 11.0, 12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0, 18.0, 19.0],
                    [20.0, 21.0, 22.0, 23.0, 24.0],
                    [25.0, 26.0, 27.0, 28.0, 29.0],
                    [30.0, 31.0, 32.0, 33.0, 34.0],
                ]
            ]
        ]
    ).astype(np.float32)
    # NOCC:invalid-name(其他:onnx example)
    W = np.array(
        [
            [
                [
                    [1.0, 1.0, 1.0],  # (1, 1, 3, 3) tensor for convolution weights
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                ]
            ]
        ]
    ).astype(np.float32)

    # Convolution with strides=2 and padding
    node_with_padding = onnx.helper.make_node(
        "Conv",
        inputs=["x", "W"],
        outputs=["y"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[
            2,
            2,
        ],  # Default values for other attributes: dilations=[1, 1], groups=1
        name=op_name,
    )
    y_with_padding = np.array(
        [
            [
                [
                    [12.0, 27.0, 24.0],  # (1, 1, 4, 3) output tensor
                    [63.0, 108.0, 81.0],
                    [123.0, 198.0, 141.0],
                    [112.0, 177.0, 124.0],
                ]
            ]
        ]
    ).astype(np.float32)
    op_expect(
        node_with_padding,
        inputs=[x, W],
        outputs=[y_with_padding],
        op_type=op_type,
        op_name=op_name,
    )

    op_name = "test_conv_with_strides_no_padding"
    # Convolution with strides=2 and no padding
    node_without_padding = onnx.helper.make_node(
        "Conv",
        inputs=["x", "W"],
        outputs=["y"],
        kernel_shape=[3, 3],
        pads=[0, 0, 0, 0],
        strides=[
            2,
            2,
        ],  # Default values for other attributes: dilations=[1, 1], groups=1
        name=op_name,
    )
    y_without_padding = np.array(
        [[[[54.0, 72.0], [144.0, 162.0], [234.0, 252.0]]]]  # (1, 1, 3, 2) output tensor
    ).astype(np.float32)
    op_expect(
        node_without_padding,
        inputs=[x, W],
        outputs=[y_without_padding],
        op_type=op_type,
        op_name=op_name,
    )

    op_name = "test_conv_with_strides_and_asymmetric_padding"
    # Convolution with strides=2 and padding only along one dimension (the H dimension in NxCxHxW tensor)
    node_with_asymmetric_padding = onnx.helper.make_node(
        "Conv",
        inputs=["x", "W"],
        outputs=["y"],
        kernel_shape=[3, 3],
        pads=[1, 0, 1, 0],
        strides=[
            2,
            2,
        ],  # Default values for other attributes: dilations=[1, 1], groups=1
        name=op_name,
    )
    y_with_asymmetric_padding = np.array(
        [
            [
                [
                    [21.0, 33.0],  # (1, 1, 4, 2) output tensor
                    [99.0, 117.0],
                    [189.0, 207.0],
                    [171.0, 183.0],
                ]
            ]
        ]
    ).astype(np.float32)
    op_expect(
        node_with_asymmetric_padding,
        inputs=[x, W],
        outputs=[y_with_asymmetric_padding],
        op_type=op_type,
        op_name=op_name,
    )


def test_convtranspose():
    op_name, op_type = "test_convtranspose", "ConvTranspose"
    x = np.array([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]).astype(  # (1, 1, 3, 3)
        np.float32
    )

    # NOCC:invalid-name(其他:onnx example)
    W = np.array(
        [
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],  # (1, 2, 3, 3)
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ]
        ]
    ).astype(np.float32)

    node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"], name=op_name)

    y = np.array(
        [
            [
                [
                    [0.0, 1.0, 3.0, 3.0, 2.0],  # (1, 2, 5, 5)
                    [3.0, 8.0, 15.0, 12.0, 7.0],
                    [9.0, 21.0, 36.0, 27.0, 15.0],
                    [9.0, 20.0, 33.0, 24.0, 13.0],
                    [6.0, 13.0, 21.0, 15.0, 8.0],
                ],
                [
                    [0.0, 1.0, 3.0, 3.0, 2.0],
                    [3.0, 8.0, 15.0, 12.0, 7.0],
                    [9.0, 21.0, 36.0, 27.0, 15.0],
                    [9.0, 20.0, 33.0, 24.0, 13.0],
                    [6.0, 13.0, 21.0, 15.0, 8.0],
                ],
            ]
        ]
    ).astype(np.float32)

    op_expect(node, inputs=[x, W], outputs=[y], op_type=op_type, op_name=op_name)

    op_name, op_type = "test_convtranspose_1d", "ConvTranspose"

    x = np.array([[[0.0, 1.0, 2.0]]]).astype(np.float32)  # (1, 1, 3)

    # NOCC:invalid-name(其他:onnx example)
    W = np.array([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]).astype(np.float32)  # (1, 2, 3)

    node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"], name=op_name)

    y = np.array([[[0.0, 1.0, 3.0, 3.0, 2.0], [0.0, 1.0, 3.0, 3.0, 2.0]]]).astype(  # (1, 2, 5)
        np.float32
    )

    op_expect(node, inputs=[x, W], outputs=[y], op_type=op_type, op_name=op_name)

    op_name, op_type = "test_convtranspose_3d", "ConvTranspose"
    x = np.array(
        [
            [
                [
                    [
                        [0.0, 1.0, 2.0, 3.0, 4.0],  # (1, 1, 3, 4, 5)
                        [5.0, 6.0, 7.0, 8.0, 9.0],
                        [10.0, 11.0, 12.0, 13.0, 14.0],
                        [15.0, 16.0, 17.0, 18.0, 19.0],
                    ],
                    [
                        [20.0, 21.0, 22.0, 23.0, 24.0],
                        [25.0, 26.0, 27.0, 28.0, 29.0],
                        [30.0, 31.0, 32.0, 33.0, 34.0],
                        [35.0, 36.0, 37.0, 38.0, 39.0],
                    ],
                    [
                        [40.0, 41.0, 42.0, 43.0, 44.0],
                        [45.0, 46.0, 47.0, 48.0, 49.0],
                        [50.0, 51.0, 52.0, 53.0, 54.0],
                        [55.0, 56.0, 57.0, 58.0, 59.0],
                    ],
                ]
            ]
        ]
    ).astype(np.float32)

    # NOCC:invalid-name(其他:onnx example)
    W = np.array(
        [
            [
                [
                    [
                        [1.0, 1.0, 1.0],  # (1, 2, 3, 3, 3)
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ],
                [
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ],
            ]
        ]
    ).astype(np.float32)

    node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"], name=op_name)

    y = np.array(
        [
            [
                [
                    [
                        [0.0, 1.0, 3.0, 6.0, 9.0, 7.0, 4.0],  # (1, 2, 5, 6, 7)
                        [5.0, 12.0, 21.0, 27.0, 33.0, 24.0, 13.0],
                        [15.0, 33.0, 54.0, 63.0, 72.0, 51.0, 27.0],
                        [30.0, 63.0, 99.0, 108.0, 117.0, 81.0, 42.0],
                        [25.0, 52.0, 81.0, 87.0, 93.0, 64.0, 33.0],
                        [15.0, 31.0, 48.0, 51.0, 54.0, 37.0, 19.0],
                    ],
                    [
                        [20.0, 42.0, 66.0, 72.0, 78.0, 54.0, 28.0],
                        [50.0, 104.0, 162.0, 174.0, 186.0, 128.0, 66.0],
                        [90.0, 186.0, 288.0, 306.0, 324.0, 222.0, 114.0],
                        [120.0, 246.0, 378.0, 396.0, 414.0, 282.0, 144.0],
                        [90.0, 184.0, 282.0, 294.0, 306.0, 208.0, 106.0],
                        [50.0, 102.0, 156.0, 162.0, 168.0, 114.0, 58.0],
                    ],
                    [
                        [60.0, 123.0, 189.0, 198.0, 207.0, 141.0, 72.0],
                        [135.0, 276.0, 423.0, 441.0, 459.0, 312.0, 159.0],
                        [225.0, 459.0, 702.0, 729.0, 756.0, 513.0, 261.0],
                        [270.0, 549.0, 837.0, 864.0, 891.0, 603.0, 306.0],
                        [195.0, 396.0, 603.0, 621.0, 639.0, 432.0, 219.0],
                        [105.0, 213.0, 324.0, 333.0, 342.0, 231.0, 117.0],
                    ],
                    [
                        [60.0, 122.0, 186.0, 192.0, 198.0, 134.0, 68.0],
                        [130.0, 264.0, 402.0, 414.0, 426.0, 288.0, 146.0],
                        [210.0, 426.0, 648.0, 666.0, 684.0, 462.0, 234.0],
                        [240.0, 486.0, 738.0, 756.0, 774.0, 522.0, 264.0],
                        [170.0, 344.0, 522.0, 534.0, 546.0, 368.0, 186.0],
                        [90.0, 182.0, 276.0, 282.0, 288.0, 194.0, 98.0],
                    ],
                    [
                        [40.0, 81.0, 123.0, 126.0, 129.0, 87.0, 44.0],
                        [85.0, 172.0, 261.0, 267.0, 273.0, 184.0, 93.0],
                        [135.0, 273.0, 414.0, 423.0, 432.0, 291.0, 147.0],
                        [150.0, 303.0, 459.0, 468.0, 477.0, 321.0, 162.0],
                        [105.0, 212.0, 321.0, 327.0, 333.0, 224.0, 113.0],
                        [55.0, 111.0, 168.0, 171.0, 174.0, 117.0, 59.0],
                    ],
                ],
                [
                    [
                        [0.0, 1.0, 3.0, 6.0, 9.0, 7.0, 4.0],
                        [5.0, 12.0, 21.0, 27.0, 33.0, 24.0, 13.0],
                        [15.0, 33.0, 54.0, 63.0, 72.0, 51.0, 27.0],
                        [30.0, 63.0, 99.0, 108.0, 117.0, 81.0, 42.0],
                        [25.0, 52.0, 81.0, 87.0, 93.0, 64.0, 33.0],
                        [15.0, 31.0, 48.0, 51.0, 54.0, 37.0, 19.0],
                    ],
                    [
                        [20.0, 42.0, 66.0, 72.0, 78.0, 54.0, 28.0],
                        [50.0, 104.0, 162.0, 174.0, 186.0, 128.0, 66.0],
                        [90.0, 186.0, 288.0, 306.0, 324.0, 222.0, 114.0],
                        [120.0, 246.0, 378.0, 396.0, 414.0, 282.0, 144.0],
                        [90.0, 184.0, 282.0, 294.0, 306.0, 208.0, 106.0],
                        [50.0, 102.0, 156.0, 162.0, 168.0, 114.0, 58.0],
                    ],
                    [
                        [60.0, 123.0, 189.0, 198.0, 207.0, 141.0, 72.0],
                        [135.0, 276.0, 423.0, 441.0, 459.0, 312.0, 159.0],
                        [225.0, 459.0, 702.0, 729.0, 756.0, 513.0, 261.0],
                        [270.0, 549.0, 837.0, 864.0, 891.0, 603.0, 306.0],
                        [195.0, 396.0, 603.0, 621.0, 639.0, 432.0, 219.0],
                        [105.0, 213.0, 324.0, 333.0, 342.0, 231.0, 117.0],
                    ],
                    [
                        [60.0, 122.0, 186.0, 192.0, 198.0, 134.0, 68.0],
                        [130.0, 264.0, 402.0, 414.0, 426.0, 288.0, 146.0],
                        [210.0, 426.0, 648.0, 666.0, 684.0, 462.0, 234.0],
                        [240.0, 486.0, 738.0, 756.0, 774.0, 522.0, 264.0],
                        [170.0, 344.0, 522.0, 534.0, 546.0, 368.0, 186.0],
                        [90.0, 182.0, 276.0, 282.0, 288.0, 194.0, 98.0],
                    ],
                    [
                        [40.0, 81.0, 123.0, 126.0, 129.0, 87.0, 44.0],
                        [85.0, 172.0, 261.0, 267.0, 273.0, 184.0, 93.0],
                        [135.0, 273.0, 414.0, 423.0, 432.0, 291.0, 147.0],
                        [150.0, 303.0, 459.0, 468.0, 477.0, 321.0, 162.0],
                        [105.0, 212.0, 321.0, 327.0, 333.0, 224.0, 113.0],
                        [55.0, 111.0, 168.0, 171.0, 174.0, 117.0, 59.0],
                    ],
                ],
            ]
        ]
    ).astype(np.float32)

    op_expect(node, inputs=[x, W], outputs=[y], op_type=op_type, op_name=op_name)

    op_name, op_type = "test_convtranspose_pads", "ConvTranspose"

    x = np.array([[[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]]]).astype(  # (1, 1, 3, 3)
        np.float32
    )

    # NOCC:invalid-name(其他:onnx example)
    W = np.array(
        [
            [
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],  # (1, 2, 3, 3)
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            ]
        ]
    ).astype(np.float32)

    node = onnx.helper.make_node(
        "ConvTranspose",
        ["X", "W"],
        ["Y"],
        strides=[3, 2],
        pads=[1, 2, 1, 2],
        name=op_name,
    )

    y = np.array(
        [
            [
                [
                    [1.0, 1.0, 3.0],  # (1, 2, 7, 3)
                    [1.0, 1.0, 3.0],
                    [7.0, 4.0, 9.0],
                    [7.0, 4.0, 9.0],
                    [7.0, 4.0, 9.0],
                    [13.0, 7.0, 15.0],
                    [13.0, 7.0, 15.0],
                ],
                [
                    [1.0, 1.0, 3.0],
                    [1.0, 1.0, 3.0],
                    [7.0, 4.0, 9.0],
                    [7.0, 4.0, 9.0],
                    [7.0, 4.0, 9.0],
                    [13.0, 7.0, 15.0],
                    [13.0, 7.0, 15.0],
                ],
            ]
        ]
    ).astype(np.float32)

    op_expect(node, inputs=[x, W], outputs=[y], op_type=op_type, op_name=op_name)


def test_cos():
    op_name, op_type = "test_cos_example", "Cos"
    node = onnx.helper.make_node("Cos", inputs=["x"], outputs=["y"], name=op_name)

    x = np.array([-1, 0, 1]).astype(np.float32)
    y = np.cos(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name, op_type = "test_cos", "Cos"
    node = onnx.helper.make_node("Cos", inputs=["x"], outputs=["y"], name=op_name)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.cos(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_cosh():
    op_name, op_type = "test_cosh_example", "Cosh"
    node = onnx.helper.make_node("Cosh", inputs=["x"], outputs=["y"], name=op_name)

    x = np.array([-1, 0, 1]).astype(np.float32)
    y = np.cosh(x)  # expected output [1.54308069,  1.,  1.54308069]
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name, op_type = "test_cosh", "Cosh"
    node = onnx.helper.make_node("Cosh", inputs=["x"], outputs=["y"], name=op_name)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.cosh(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_depthtospace():
    op_name, op_type = "test_depthtospace_crd_mode_example", "DepthToSpace"
    node = onnx.helper.make_node(
        "DepthToSpace",
        inputs=["x"],
        outputs=["y"],
        blocksize=2,
        mode="CRD",
        name=op_name,
    )

    # (1, 8, 2, 3) input tensor
    x = np.array(
        [
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
                [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
                [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0]],
                [[36.0, 37.0, 38.0], [39.0, 40.0, 41.0]],
                [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0]],
                [[54.0, 55.0, 56.0], [57.0, 58.0, 59.0]],
                [[63.0, 64.0, 65.0], [66.0, 67.0, 68.0]],
            ]
        ]
    ).astype(np.float32)

    # (1, 2, 4, 6) output tensor
    y = np.array(
        [
            [
                [
                    [0.0, 9.0, 1.0, 10.0, 2.0, 11.0],
                    [18.0, 27.0, 19.0, 28.0, 20.0, 29.0],
                    [3.0, 12.0, 4.0, 13.0, 5.0, 14.0],
                    [21.0, 30.0, 22.0, 31.0, 23.0, 32.0],
                ],
                [
                    [36.0, 45.0, 37.0, 46.0, 38.0, 47.0],
                    [54.0, 63.0, 55.0, 64.0, 56.0, 65.0],
                    [39.0, 48.0, 40.0, 49.0, 41.0, 50.0],
                    [57.0, 66.0, 58.0, 67.0, 59.0, 68.0],
                ],
            ]
        ]
    ).astype(np.float32)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_depthtospace_example"
    node = onnx.helper.make_node(
        "DepthToSpace",
        inputs=["x"],
        outputs=["y"],
        blocksize=2,
        mode="DCR",
        name=op_name,
    )

    # (1, 8, 2, 3) input tensor
    x = np.array(
        [
            [
                [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
                [[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]],
                [[18.0, 19.0, 20.0], [21.0, 22.0, 23.0]],
                [[27.0, 28.0, 29.0], [30.0, 31.0, 32.0]],
                [[36.0, 37.0, 38.0], [39.0, 40.0, 41.0]],
                [[45.0, 46.0, 47.0], [48.0, 49.0, 50.0]],
                [[54.0, 55.0, 56.0], [57.0, 58.0, 59.0]],
                [[63.0, 64.0, 65.0], [66.0, 67.0, 68.0]],
            ]
        ]
    ).astype(np.float32)

    # (1, 2, 4, 6) output tensor
    y = np.array(
        [
            [
                [
                    [0.0, 18.0, 1.0, 19.0, 2.0, 20.0],
                    [36.0, 54.0, 37.0, 55.0, 38.0, 56.0],
                    [3.0, 21.0, 4.0, 22.0, 5.0, 23.0],
                    [39.0, 57.0, 40.0, 58.0, 41.0, 59.0],
                ],
                [
                    [9.0, 27.0, 10.0, 28.0, 11.0, 29.0],
                    [45.0, 63.0, 46.0, 64.0, 47.0, 65.0],
                    [12.0, 30.0, 13.0, 31.0, 14.0, 32.0],
                    [48.0, 66.0, 49.0, 67.0, 50.0, 68.0],
                ],
            ]
        ]
    ).astype(np.float32)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_div():
    op_name, op_type = "test_div_example", "Div"
    node = onnx.helper.make_node("Div", inputs=["x", "y"], outputs=["z"], name=op_name)

    x = np.array([3, 4]).astype(np.float32)
    y = np.array([1, 2]).astype(np.float32)
    z = x / y  # expected output [3., 2.]
    op_expect(node, inputs=[x, y], outputs=[z], op_type=op_type, op_name=op_name)

    op_name, op_type = "test_div", "Div"
    node = onnx.helper.make_node("Div", inputs=["x", "y"], outputs=["z"], name=op_name)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.rand(3, 4, 5).astype(np.float32) + 1.0
    z = x / y
    op_expect(node, inputs=[x, y], outputs=[z], op_type=op_type, op_name=op_name)

    op_name, op_type = "test_div_bcast", "Div"
    node = onnx.helper.make_node("Div", inputs=["x", "y"], outputs=["z"], name=op_name)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.rand(5).astype(np.float32) + 1.0
    z = x / y
    op_expect(node, inputs=[x, y], outputs=[z], op_type=op_type, op_name=op_name)


@pytest.mark.skip(reason="TensorRT segmentfault")
def test_einsum():
    op_name, op_type = "test_einsum_batch_diagonal", "Einsum"
    eqn = "...ii ->...i"
    node = onnx.helper.make_node("Einsum", inputs=["x"], outputs=["y"], equation=eqn, name=op_name)

    # NOCC:invalid-name(其他:onnx example)
    X = np.random.randn(3, 5, 5).astype(np.float32)
    from onnx.backend.test.case.node.einsum import einsum_reference_implementation

    # NOCC:invalid-name(其他:onnx example)
    Z = einsum_reference_implementation(eqn, (X,))
    op_expect(node, inputs=[X], outputs=[Z], op_type=op_type, op_name=op_name)


def test_elu():
    op_name, op_type = "test_elu_example", "Elu"
    node = onnx.helper.make_node("Elu", inputs=["x"], outputs=["y"], alpha=2.0, name=op_name)

    x = np.array([-1, 0, 1]).astype(np.float32)
    # expected output [-1.2642411, 0., 1.]
    y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name, op_type = "test_elu", "Elu"
    node = onnx.helper.make_node("Elu", inputs=["x"], outputs=["y"], alpha=2.0, name=op_name)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name, op_type = "test_elu_default", "Elu"
    default_alpha = 1.0
    node = onnx.helper.make_node("Elu", inputs=["x"], outputs=["y"], name=op_name)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_erf():
    op_name, op_type = "test_erf", "Erf"
    node = onnx.helper.make_node("Erf", inputs=["x"], outputs=["y"], name=op_name)

    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    import math

    y = np.vectorize(math.erf)(x).astype(np.float32)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_exp():
    op_name, op_type = "test_exp_example", "Exp"
    node = onnx.helper.make_node("Exp", inputs=["x"], outputs=["y"], name=op_name)

    x = np.array([-1, 0, 1]).astype(np.float32)
    y = np.exp(x)  # expected output [0.36787945, 1., 2.71828175]
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name, op_type = "test_exp", "Exp"
    node = onnx.helper.make_node("Exp", inputs=["x"], outputs=["y"], name=op_name)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.exp(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_eyelike():
    op_name, op_type = "test_eyelike_populate_off_main_diagonal", "EyeLike"
    shape = (4, 5)
    off_diagonal_offset = 1
    node = onnx.helper.make_node(
        "EyeLike",
        inputs=["x"],
        outputs=["y"],
        k=off_diagonal_offset,
        dtype=onnx.TensorProto.FLOAT,
        name=op_name,
    )

    x = np.random.randint(0, 100, size=shape, dtype=np.int32)
    y = np.eye(shape[0], shape[1], k=off_diagonal_offset, dtype=np.float32)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_eyelike_with_dtype"
    shape = (3, 4)
    node = onnx.helper.make_node(
        "EyeLike",
        inputs=["x"],
        outputs=["y"],
        dtype=onnx.TensorProto.FLOAT,
        name=op_name,
    )

    x = np.random.randint(0, 100, size=shape, dtype=np.int32)
    y = np.eye(shape[0], shape[1], dtype=np.float32)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_eyelike_without_dtype"
    shape = (4, 4)
    node = onnx.helper.make_node("EyeLike", inputs=["x"], outputs=["y"], name=op_name)

    x = np.random.randint(0, 100, size=shape, dtype=np.int32)
    y = np.eye(shape[0], shape[1], dtype=np.int32)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_floor():
    op_name, op_type = "test_floor_example", "Floor"
    node = onnx.helper.make_node("Floor", inputs=["x"], outputs=["y"], name=op_name)

    x = np.array([-1.5, 1.2, 2]).astype(np.float32)
    y = np.floor(x)  # expected output [-2., 1., 2.]
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name, op_type = "test_floor", "Floor"
    node = onnx.helper.make_node("Floor", inputs=["x"], outputs=["y"], name=op_name)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.floor(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def verify_rnn(
    seq_length,
    batch_size,
    input_size,
    hidden_size,
    rnn_type="LSTM",
    use_bias=False,
    activations=None,
    alphas=None,
    betas=None,
    use_initial_state=False,
    use_peep=False,
    linear_before_reset=False,
    op_name=None,
    layout=0,
):
    if rnn_type == "LSTM":
        multiplier = 4
    elif rnn_type == "GRU":
        multiplier = 3
    else:
        raise NotImplementedError("%s RNNs not yet supported." % rnn_type)

    x_np = np.random.uniform(size=(seq_length, batch_size, input_size)).astype("float32")
    w_np = np.random.uniform(size=(1, multiplier * hidden_size, input_size)).astype("float32")
    r_np = np.random.uniform(size=(1, multiplier * hidden_size, hidden_size)).astype("float32")
    input_names = ["X", "W", "R"]

    input_tensors = [
        helper.make_tensor_value_info("X", TensorProto.FLOAT, list(x_np.shape)),
        helper.make_tensor_value_info("W", TensorProto.FLOAT, list(w_np.shape)),
        helper.make_tensor_value_info("R", TensorProto.FLOAT, list(r_np.shape)),
    ]

    input_values = [x_np, w_np, r_np]

    if use_bias:
        b_np = np.random.uniform(size=(1, multiplier * 2 * hidden_size)).astype("float32")
        input_names.append("B")
        input_tensors.append(
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [1, multiplier * 2 * hidden_size])
        )
        input_values.append(b_np)

    if use_initial_state:
        assert use_bias is True, "Initial states must have bias specified."
        sequence_np = np.repeat(seq_length, batch_size).astype("int32")
        input_names.append("sequence_lens")
        input_tensors.append(
            helper.make_tensor_value_info("sequence_lens", TensorProto.INT32, [batch_size])
        )
        input_values.append(sequence_np)

        initial_h_np = np.random.uniform(size=(1, batch_size, hidden_size)).astype("float32")
        input_names.append("initial_h")
        input_tensors.append(
            helper.make_tensor_value_info(
                "initial_h", TensorProto.FLOAT, [1, batch_size, hidden_size]
            )
        )
        input_values.append(initial_h_np)

        if rnn_type == "LSTM":
            initial_c_np = np.random.uniform(size=(1, batch_size, hidden_size)).astype("float32")
            input_names.append("initial_c")
            input_tensors.append(
                helper.make_tensor_value_info(
                    "initial_c", TensorProto.FLOAT, [1, batch_size, hidden_size]
                )
            )
            input_values.append(initial_c_np)

    if use_peep and rnn_type == "LSTM":
        assert use_initial_state is True, "Peepholes require initial state to be specified."
        p_np = np.random.uniform(size=(1, 3 * hidden_size)).astype("float32")
        input_names.append("P")
        input_tensors.append(
            helper.make_tensor_value_info("P", TensorProto.FLOAT, [1, 3 * hidden_size])
        )
        input_values.append(p_np)

    Y_shape = [seq_length, 1, batch_size, hidden_size]
    Y_h_shape = [1, batch_size, hidden_size]
    outputs = ["Y", "Y_h"]

    graph_outputs = [
        helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(Y_shape)),
        helper.make_tensor_value_info("Y_h", TensorProto.FLOAT, list(Y_h_shape)),
    ]
    output_shapes = [Y_shape, Y_h_shape]

    if rnn_type == "LSTM":
        Y_c_shape_0 = [1, batch_size, hidden_size]
        outputs.append("Y_c")
        graph_outputs.append(
            helper.make_tensor_value_info("Y_c", TensorProto.FLOAT, list(Y_c_shape_0))
        )
        output_shapes.append(Y_c_shape_0)

    rnn_node = helper.make_node(
        rnn_type,
        inputs=input_names,
        outputs=outputs,
        hidden_size=hidden_size,
        layout=0,
        name=op_name,
    )
    if activations is not None:
        activations_attr = helper.make_attribute("activations", activations)
        rnn_node.attribute.append(activations_attr)
    if alphas is not None:
        alphas_attr = helper.make_attribute("activation_alpha", alphas)
        rnn_node.attribute.append(alphas_attr)
    if betas is not None:
        betas_attr = helper.make_attribute("activation_beta", betas)
        rnn_node.attribute.append(betas_attr)
    if linear_before_reset and rnn_type == "GRU":
        lbr_attr = helper.make_attribute("linear_before_reset", 1)
        rnn_node.attribute.append(lbr_attr)

    graph = helper.make_graph([rnn_node], "rnn_test", inputs=input_tensors, outputs=graph_outputs)

    model = helper.make_model(graph, producer_name="rnn_test")

    verify_with_ort_with_trt(model, input_values, op_name, layout=layout)


def test_gather():
    op_name, op_type = "test_gather_0", "Gather"
    node = onnx.helper.make_node(
        "Gather", inputs=["data", "indices"], outputs=["y"], axis=0, name=op_name
    )
    data = np.random.randn(5, 4, 3, 2).astype(np.float32)
    indices = np.array([0, 1, 3])
    y = np.take(data, indices, axis=0)

    op_expect(
        node,
        inputs=[data, indices.astype(np.int64)],
        outputs=[y],
        op_type=op_type,
        op_name=op_name,
    )

    op_name = "test_gather_1"
    node = onnx.helper.make_node(
        "Gather", inputs=["data", "indices"], outputs=["y"], axis=1, name=op_name
    )
    data = np.random.randn(5, 4, 3, 2).astype(np.float32)
    indices = np.array([0, 1, 3])
    y = np.take(data, indices, axis=1)

    op_expect(
        node,
        inputs=[data, indices.astype(np.int64)],
        outputs=[y],
        op_type=op_type,
        op_name=op_name,
    )

    op_name = "test_gather_2d_indices"
    node = onnx.helper.make_node(
        "Gather", inputs=["data", "indices"], outputs=["y"], axis=1, name=op_name
    )
    data = np.random.randn(3, 3).astype(np.float32)
    indices = np.array([[0, 2]])
    y = np.take(data, indices, axis=1)

    op_expect(
        node,
        inputs=[data, indices.astype(np.int64)],
        outputs=[y],
        op_type=op_type,
        op_name=op_name,
    )

    op_name = "test_gather_negative_indices"
    node = onnx.helper.make_node(
        "Gather", inputs=["data", "indices"], outputs=["y"], axis=0, name=op_name
    )
    data = np.arange(10).astype(np.float32)
    indices = np.array([0, -9, -10])
    y = np.take(data, indices, axis=0)

    # print(y)
    # [0. 1. 0.]

    op_expect(
        node,
        inputs=[data, indices.astype(np.int64)],
        outputs=[y],
        op_type=op_type,
        op_name=op_name,
    )


def test_gatherelement():
    op_name, op_type = "test_gather_elements_0", "GatherElements"
    axis = 1
    node = onnx.helper.make_node(
        "GatherElements",
        inputs=["data", "indices"],
        outputs=["y"],
        axis=axis,
        name=op_name,
    )
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    indices = np.array([[0, 0], [1, 0]], dtype=np.int32)

    from onnx.backend.test.case.node.gatherelements import gather_elements

    y = gather_elements(data, indices, axis)
    # print(y) produces
    # [[1, 1],
    #  [4, 3]]

    op_expect(
        node,
        inputs=[data, indices.astype(np.int64)],
        outputs=[y],
        op_type=op_type,
        op_name=op_name,
    )

    op_name = "test_gather_elements_1"
    axis = 0
    node = onnx.helper.make_node(
        "GatherElements",
        inputs=["data", "indices"],
        outputs=["y"],
        axis=axis,
        name=op_name,
    )
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    indices = np.array([[1, 2, 0], [2, 0, 0]], dtype=np.int32)

    y = gather_elements(data, indices, axis)
    # print(y) produces
    # [[4, 8, 3],
    #  [7, 2, 3]]
    op_expect(
        node,
        inputs=[data, indices.astype(np.int64)],
        outputs=[y],
        op_type=op_type,
        op_name=op_name,
    )

    op_name = "test_gather_elements_negative_indices"
    axis = 0
    node = onnx.helper.make_node(
        "GatherElements",
        inputs=["data", "indices"],
        outputs=["y"],
        axis=axis,
        name=op_name,
    )
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
    indices = np.array([[-1, -2, 0], [-2, 0, 0]], dtype=np.int32)

    y = gather_elements(data, indices, axis)
    # print(y) produces
    # [[7, 5, 3],
    #  [4, 2, 3]]
    op_expect(
        node,
        inputs=[data, indices.astype(np.int64)],
        outputs=[y],
        op_type=op_type,
        op_name=op_name,
    )


def test_gathernd():
    op_name, op_type = "test_gathernd_example_float32", "GatherND"
    node = onnx.helper.make_node(
        "GatherND", inputs=["data", "indices"], outputs=["output"], name=op_name
    )

    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
    indices = np.array([[[0, 1]], [[1, 0]]], dtype=np.int64)
    from onnx.backend.test.case.node.gathernd import gather_nd_impl

    output = gather_nd_impl(data, indices, 0)
    expected_output = np.array([[[2, 3]], [[4, 5]]], dtype=np.float32)
    assert np.array_equal(output, expected_output)
    op_expect(node, inputs=[data, indices], outputs=[output], op_type=op_type, op_name=op_name)

    op_name = "test_gathernd_example_int32"
    node = onnx.helper.make_node(
        "GatherND", inputs=["data", "indices"], outputs=["output"], name=op_name
    )

    data = np.array([[0, 1], [2, 3]], dtype=np.int32)
    indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
    output = gather_nd_impl(data, indices, 0)
    expected_output = np.array([0, 3], dtype=np.int32)
    assert np.array_equal(output, expected_output)
    op_expect(node, inputs=[data, indices], outputs=[output], op_type=op_type, op_name=op_name)

    op_name = "test_gathernd_example_int32_batch_dim1"
    node = onnx.helper.make_node(
        "GatherND",
        inputs=["data", "indices"],
        outputs=["output"],
        batch_dims=1,
        name=op_name,
    )

    data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
    indices = np.array([[1], [0]], dtype=np.int64)
    output = gather_nd_impl(data, indices, 1)
    expected_output = np.array([[2, 3], [4, 5]], dtype=np.int32)
    assert np.array_equal(output, expected_output)
    op_expect(node, inputs=[data, indices], outputs=[output], op_type=op_type, op_name=op_name)


def test_gemm():
    op_name, op_type = "test_gemm_all_attributes", "Gemm"
    node = onnx.helper.make_node(
        "Gemm",
        inputs=["a", "b", "c"],
        outputs=["y"],
        alpha=0.25,
        beta=0.35,
        transA=1,
        transB=1,
        name=op_name,
    )
    a = np.random.ranf([4, 3]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.random.ranf([1, 5]).astype(np.float32)
    from onnx.backend.test.case.node.gemm import gemm_reference_implementation

    y = gemm_reference_implementation(a, b, c, transA=1, transB=1, alpha=0.25, beta=0.35)
    op_expect(node, inputs=[a, b, c], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_gemm_alpha"
    node = onnx.helper.make_node(
        "Gemm", inputs=["a", "b", "c"], outputs=["y"], alpha=0.5, name=op_name
    )
    a = np.random.ranf([3, 5]).astype(np.float32)
    b = np.random.ranf([5, 4]).astype(np.float32)
    c = np.zeros([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, alpha=0.5)
    op_expect(node, inputs=[a, b, c], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_gemm_beta"
    node = onnx.helper.make_node(
        "Gemm", inputs=["a", "b", "c"], outputs=["y"], beta=0.5, name=op_name
    )
    a = np.random.ranf([2, 7]).astype(np.float32)
    b = np.random.ranf([7, 4]).astype(np.float32)
    c = np.random.ranf([1, 4]).astype(np.float32)
    y = gemm_reference_implementation(a, b, c, beta=0.5)
    op_expect(node, inputs=[a, b, c], outputs=[y], op_type=op_type, op_name=op_name)


def test_globalaveragepool():
    op_name, op_type = "test_globalaveragepool", "GlobalAveragePool"
    node = onnx.helper.make_node("GlobalAveragePool", inputs=["x"], outputs=["y"], name=op_name)
    x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    y = np.mean(x, axis=tuple(range(2, np.ndim(x))), keepdims=True)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_globalaveragepool_precomputed"
    node = onnx.helper.make_node("GlobalAveragePool", inputs=["x"], outputs=["y"], name=op_name)
    x = np.array(
        [
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ]
        ]
    ).astype(np.float32)
    y = np.array([[[[5]]]]).astype(np.float32)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_globalmaxpool():
    op_name = "test_globalmaxpool"
    op_type = "GlobalMaxPool"
    node = onnx.helper.make_node("GlobalMaxPool", inputs=["x"], outputs=["y"], name=op_name)
    x = np.random.randn(1, 3, 5, 5).astype(np.float32)
    y = np.max(x, axis=tuple(range(2, np.ndim(x))), keepdims=True)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_globalmaxpool_precomputed"
    node = onnx.helper.make_node("GlobalMaxPool", inputs=["x"], outputs=["y"], name=op_name)
    x = np.array(
        [
            [
                [
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ]
            ]
        ]
    ).astype(np.float32)
    y = np.array([[[[9]]]]).astype(np.float32)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_hardsigmoid():
    op_name, op_type = "test_hardsigmoid_example", "HardSigmoid"
    node = onnx.helper.make_node(
        "HardSigmoid", inputs=["x"], outputs=["y"], alpha=0.5, beta=0.6, name=op_name
    )

    x = np.array([-1, 0, 1]).astype(np.float32)
    y = np.clip(x * 0.5 + 0.6, 0, 1)  # expected output [0.1, 0.6, 1.]
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_hardsigmoid"
    node = onnx.helper.make_node(
        "HardSigmoid", inputs=["x"], outputs=["y"], alpha=0.5, beta=0.6, name=op_name
    )
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x * 0.5 + 0.6, 0, 1)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_hardsigmoid_default"

    default_alpha = 0.2
    default_beta = 0.5
    node = onnx.helper.make_node("HardSigmoid", inputs=["x"], outputs=["y"], name=op_name)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x * default_alpha + default_beta, 0, 1)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_hardswish():
    op_name, op_type = "test_hardswish", "HardSwish"
    node = onnx.helper.make_node("HardSwish", inputs=["x"], outputs=["y"], name=op_name)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    from onnx.backend.test.case.node.hardswish import hardswish

    y = hardswish(x)

    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_hardmax():
    op_name, op_type = "test_hardmax_example", "Hardmax"
    node = onnx.helper.make_node("Hardmax", inputs=["x"], outputs=["y"], name=op_name)

    x = np.array([[3, 0, 1, 2], [2, 5, 1, 0], [0, 1, 3, 2], [0, 1, 2, 3]]).astype(np.float32)
    # expect result:
    # [[1. 0. 0. 0.]
    # [0. 1. 0. 0.]
    # [0. 0. 1. 0.]
    # [0. 0. 0. 1.]]
    from onnx.backend.test.case.node.hardmax import hardmax

    y = hardmax(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_identity():
    op_name, op_type = "test_identity", "Identity"
    node = onnx.helper.make_node("Identity", inputs=["x"], outputs=["y"], name=op_name)

    data = np.array(
        [
            [
                [
                    [1, 2],
                    [3, 4],
                ]
            ]
        ],
        dtype=np.float32,
    )

    op_expect(node, inputs=[data], outputs=[data], op_type=op_type, op_name=op_name)


def test_instancenormalization():
    op_name, op_type = "test_instancenorm_example", "InstanceNormalization"

    def _instancenorm_test_mode(x, s, bias, epsilon=1e-5):  # type: ignore
        dims_x = len(x.shape)
        axis = tuple(range(2, dims_x))
        mean = np.mean(x, axis=axis, keepdims=True)
        var = np.var(x, axis=axis, keepdims=True)
        dim_ones = (1,) * (dims_x - 2)
        s = s.reshape(-1, *dim_ones)
        bias = bias.reshape(-1, *dim_ones)
        return s * (x - mean) / np.sqrt(var + epsilon) + bias

    # input size: (1, 2, 1, 3)
    x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
    s = np.array([1.0, 1.5]).astype(np.float32)
    bias = np.array([0, 1]).astype(np.float32)
    y = _instancenorm_test_mode(x, s, bias).astype(np.float32)

    node = onnx.helper.make_node(
        "InstanceNormalization", inputs=["x", "s", "bias"], outputs=["y"], name=op_name
    )

    # output size: (1, 2, 1, 3)
    op_expect(node, inputs=[x, s, bias], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_instancenorm_epsilon"
    # input size: (2, 3, 4, 5)
    x = np.random.randn(2, 3, 4, 5).astype(np.float32)
    s = np.random.randn(3).astype(np.float32)
    bias = np.random.randn(3).astype(np.float32)
    epsilon = 1e-2
    y = _instancenorm_test_mode(x, s, bias, epsilon).astype(np.float32)

    node = onnx.helper.make_node(
        "InstanceNormalization",
        inputs=["x", "s", "bias"],
        outputs=["y"],
        epsilon=epsilon,
        name=op_name,
    )

    # output size: (2, 3, 4, 5)
    op_expect(node, inputs=[x, s, bias], outputs=[y], op_type=op_type, op_name=op_name)


def test_leakyrelu():
    op_name, op_type = "test_leakyrelu_example", "LeakyRelu"
    node = onnx.helper.make_node("LeakyRelu", inputs=["x"], outputs=["y"], alpha=0.1, name=op_name)

    x = np.array([-1, 0, 1]).astype(np.float32)
    # expected output [-0.1, 0., 1.]
    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_leakyrelu"
    node = onnx.helper.make_node("LeakyRelu", inputs=["x"], outputs=["y"], alpha=0.1, name=op_name)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_leakyrelu_default"
    default_alpha = 0.01
    node = onnx.helper.make_node("LeakyRelu", inputs=["x"], outputs=["y"], name=op_name)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * default_alpha
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_log():
    op_name = "test_log_example"
    op_type = "Log"
    node = onnx.helper.make_node("Log", inputs=["x"], outputs=["y"], name=op_name)

    x = np.array([1, 10]).astype(np.float32)
    y = np.log(x)  # expected output [0., 2.30258512]
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_log"
    node = onnx.helper.make_node("Log", inputs=["x"], outputs=["y"], name=op_name)

    x = np.exp(np.random.randn(3, 4, 5).astype(np.float32))
    y = np.log(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


@pytest.mark.skip(reason="Wrong answer, at axis 1")
def test_logsoftmax():
    op_name, op_type = "test_logsoftmax_example_1", "LogSoftmax"
    node = onnx.helper.make_node("LogSoftmax", inputs=["x"], outputs=["y"], name=op_name)
    x = np.array([[-1, 0, 1]]).astype(np.float32)
    # expected output
    # [[-2.4076061 -1.407606  -0.407606 ]]
    from onnx.backend.test.case.node.logsoftmax import logsoftmax

    y = logsoftmax(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
    axis_order = [0, 1, -1]
    for axis in axis_order:
        op_name = "test_logsoftmax_axis_{}".format(str(axis + 1))
        node = onnx.helper.make_node(
            "LogSoftmax", inputs=["x"], outputs=["y"], axis=axis, name=op_name
        )
        y = logsoftmax(x, axis=axis)
        op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_matmul():
    op_name, op_type = "test_matmul_2d", "MatMul"
    node = onnx.helper.make_node("MatMul", inputs=["a", "b"], outputs=["c"], name=op_name)

    # 2d
    a = np.random.randn(3, 4).astype(np.float32)
    b = np.random.randn(4, 3).astype(np.float32)
    c = np.matmul(a, b)
    op_expect(node, inputs=[a, b], outputs=[c], op_type=op_type, op_name=op_name)


def test_max():
    op_name = "test_max_example"
    op_type = "Max"
    data_0 = np.array([3, 2, 1]).astype(np.float32)
    data_1 = np.array([1, 4, 4]).astype(np.float32)
    data_2 = np.array([2, 5, 3]).astype(np.float32)
    result = np.array([3, 5, 4]).astype(np.float32)
    node = onnx.helper.make_node(
        "Max", inputs=["data_0", "data_1", "data_2"], outputs=["result"], name=op_name
    )
    op_expect(
        node,
        inputs=[data_0, data_1, data_2],
        outputs=[result],
        op_type=op_type,
        op_name=op_name,
    )

    op_name = "test_max_two_inputs"
    result = np.maximum(data_0, data_1)
    node = onnx.helper.make_node(
        "Max", inputs=["data_0", "data_1"], outputs=["result"], name=op_name
    )
    op_expect(
        node,
        inputs=[data_0, data_1],
        outputs=[result],
        op_type=op_type,
        op_name=op_name,
    )


def _test_maxpool_2d_ceil():
    op_name, op_type = "test_maxpool_2d_ceil", "MaxPool"
    node = onnx.helper.make_node(
        "MaxPool",
        inputs=["x"],
        outputs=["y"],
        kernel_shape=[3, 3],
        strides=[2, 2],
        ceil_mode=True,
        name=op_name,
    )
    x = np.array(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ]
    ).astype(np.float32)
    y = np.array([[[[11, 12], [15, 16]]]]).astype(np.float32)

    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def _test_maxpool_1d_default():
    op_name, op_type = "test_maxpool_1d_default", "MaxPool"
    node = onnx.helper.make_node(
        "MaxPool", inputs=["x"], outputs=["y"], kernel_shape=[2], name=op_name
    )
    x = np.random.randn(1, 3, 32).astype(np.float32)
    x_shape = np.shape(x)
    kernel_shape = [2]
    strides = [1]
    from onnx.backend.test.case.node.pool_op_common import get_output_shape, pool

    out_shape = get_output_shape("VALID", x_shape[2:], kernel_shape, strides)
    padded = x
    y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0], "MAX")
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_maxpool():
    _test_maxpool_2d_ceil()
    _test_maxpool_1d_default()


def test_mean():
    op_name, op_type = "test_mean_example", "Mean"
    data_0 = np.array([3, 0, 2]).astype(np.float32)
    data_1 = np.array([1, 3, 4]).astype(np.float32)
    data_2 = np.array([2, 6, 6]).astype(np.float32)
    result = np.array([2, 3, 4]).astype(np.float32)
    node = onnx.helper.make_node(
        "Mean", inputs=["data_0", "data_1", "data_2"], outputs=["result"], name=op_name
    )
    op_expect(
        node,
        inputs=[data_0, data_1, data_2],
        outputs=[result],
        op_type=op_type,
        op_name=op_name,
    )

    op_name = "test_mean_two_inputs"
    result = np.divide(np.add(data_0, data_1), 2.0)
    node = onnx.helper.make_node(
        "Mean", inputs=["data_0", "data_1"], outputs=["result"], name=op_name
    )
    op_expect(
        node,
        inputs=[data_0, data_1],
        outputs=[result],
        op_type=op_type,
        op_name=op_name,
    )


def test_min():
    op_name, op_type = "test_min_example", "Min"
    data_0 = np.array([3, 2, 1]).astype(np.float32)
    data_1 = np.array([1, 4, 4]).astype(np.float32)
    data_2 = np.array([2, 5, 0]).astype(np.float32)
    result = np.array([1, 2, 0]).astype(np.float32)
    node = onnx.helper.make_node(
        "Min", inputs=["data_0", "data_1", "data_2"], outputs=["result"], name=op_name
    )
    op_expect(
        node,
        inputs=[data_0, data_1, data_2],
        outputs=[result],
        op_type=op_type,
        op_name=op_name,
    )

    op_name = "test_min_two_inputs"
    result = np.minimum(data_0, data_1)
    node = onnx.helper.make_node(
        "Min", inputs=["data_0", "data_1"], outputs=["result"], name=op_name
    )
    op_expect(
        node,
        inputs=[data_0, data_1],
        outputs=[result],
        op_type=op_type,
        op_name=op_name,
    )


def test_mul():
    op_name, op_type = "test_mul_example", "Mul"
    node = onnx.helper.make_node("Mul", inputs=["x", "y"], outputs=["z"], name=op_name)

    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([4, 5, 6]).astype(np.float32)
    z = x * y  # expected output [4., 10., 18.]
    op_expect(node, inputs=[x, y], outputs=[z], op_type=op_type, op_name=op_name)

    op_name = "test_mul"
    node = onnx.helper.make_node("Mul", inputs=["x", "y"], outputs=["z"], name=op_name)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    z = x * y
    op_expect(node, inputs=[x, y], outputs=[z], op_type=op_type, op_name=op_name)

    op_name = "test_mul_bcast"
    node = onnx.helper.make_node("Mul", inputs=["x", "y"], outputs=["z"], name=op_name)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.random.randn(5).astype(np.float32)
    z = x * y
    op_expect(node, inputs=[x, y], outputs=[z], op_type=op_type, op_name=op_name)


def test_neg():
    op_name, op_type = "test_neg_example", "Neg"
    node = onnx.helper.make_node("Neg", inputs=["x"], outputs=["y"], name=op_name)

    x = np.array([-4, 2]).astype(np.float32)
    y = np.negative(x)  # expected output [4., -2.],
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_neg"
    node = onnx.helper.make_node("Neg", inputs=["x"], outputs=["y"], name=op_name)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.negative(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_negativeloglikelihoodloss():
    op_name, op_type = "test_nllloss_NC", "NegativeLogLikelihoodLoss"
    reduction = "none"
    node = onnx.helper.make_node(
        "NegativeLogLikelihoodLoss",
        inputs=["input", "target"],
        outputs=["loss"],
        reduction=reduction,
        name=op_name,
    )

    # NOCC:invalid-name(其他:onnx example)
    N, C = 3, 5
    np.random.seed(0)
    input = np.random.rand(N, C).astype(np.float32)
    target = np.random.randint(0, high=C, size=(N,)).astype(np.int64)
    from onnx.backend.test.case.node.negativeloglikelihoodloss import (
        compute_negative_log_likelihood_loss,
    )

    negative_log_likelihood_loss = compute_negative_log_likelihood_loss(
        input, target, weight=None, reduction=reduction
    )

    op_expect(
        node,
        inputs=[input, target],
        outputs=[negative_log_likelihood_loss],
        op_type=op_type,
        op_name=op_name,
    )


def test_prelu():
    op_name, op_type = "test_prelu_example", "PRelu"
    node = onnx.helper.make_node("PRelu", inputs=["x", "slope"], outputs=["y"], name=op_name)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    slope = np.random.randn(3, 4, 5).astype(np.float32)
    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

    op_expect(node, inputs=[x, slope], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_prelu_broadcast"
    node = onnx.helper.make_node("PRelu", inputs=["x", "slope"], outputs=["y"], name=op_name)

    x = np.random.randn(3, 4, 5).astype(np.float32)
    slope = np.random.randn(5).astype(np.float32)
    y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

    op_expect(node, inputs=[x, slope], outputs=[y], op_type=op_type, op_name=op_name)


def test_pow():
    op_name, op_type = "test_pow_example", "Pow"
    node = onnx.helper.make_node("Pow", inputs=["x", "y"], outputs=["z"], name=op_name)

    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([4, 5, 6]).astype(np.float32)
    z = pow(x, y)  # expected output [1., 32., 729.]
    op_expect(node, inputs=[x, y], outputs=[z], op_type=op_type, op_name=op_name)

    op_name = "test_pow"
    node = onnx.helper.make_node("Pow", inputs=["x", "y"], outputs=["z"], name=op_name)
    x = np.arange(60).reshape(3, 4, 5).astype(np.float32)
    y = np.random.randn(3, 4, 5).astype(np.float32)
    z = pow(x, y)
    op_expect(node, inputs=[x, y], outputs=[z], op_type=op_type, op_name=op_name)

    op_name = "test_pow_bcast_scalar"
    node = onnx.helper.make_node("Pow", inputs=["x", "y"], outputs=["z"], name=op_name)

    x = np.array([1, 2, 3]).astype(np.float32)
    y = np.array([2]).astype(np.float32)
    z = pow(x, y)  # expected output [1., 4., 9.]
    op_expect(node, inputs=[x, y], outputs=[z], op_type=op_type, op_name=op_name)

    op_name = "test_pow_bcast_array"
    node = onnx.helper.make_node("Pow", inputs=["x", "y"], outputs=["z"], name=op_name)
    x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
    y = np.array([[1, 2, 3]]).astype(np.float32)
    # expected output [[1, 4, 27], [4, 25, 216]]
    z = pow(x, y)
    op_expect(node, inputs=[x, y], outputs=[z], op_type=op_type, op_name=op_name)


def test_reciprocal():
    op_name, op_type = "test_reciprocal_example", "Reciprocal"
    node = onnx.helper.make_node("Reciprocal", inputs=["x"], outputs=["y"], name=op_name)

    x = np.array([-4, 2]).astype(np.float32)
    y = np.reciprocal(x)  # expected output [-0.25, 0.5],
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)

    op_name = "test_reciprocal"
    node = onnx.helper.make_node("Reciprocal", inputs=["x"], outputs=["y"], name=op_name)
    x = np.random.rand(3, 4, 5).astype(np.float32) + 0.5
    y = np.reciprocal(x)
    op_expect(node, inputs=[x], outputs=[y], op_type=op_type, op_name=op_name)


def test_reducel1():
    op_name, op_type = "test_reduce_l1_default_axes_keepdims_example", "ReduceL1"
    shape = [3, 2, 2]
    axes = None
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceL1",
        inputs=["data"],
        outputs=["reduced"],
        keepdims=keepdims,
        name=op_name,
    )

    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    # print(data)
    # [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

    reduced = np.sum(a=np.abs(data), axis=axes, keepdims=keepdims == 1)
    # print(reduced)
    # [[[78.]]]

    op_expect(node, inputs=[data], outputs=[reduced], op_type=op_type, op_name=op_name)

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sum(a=np.abs(data), axis=axes, keepdims=keepdims == 1)

    op_name = "test_reduce_l1_default_axes_keepdims_random"
    node = onnx.helper.make_node(
        "ReduceL1",
        inputs=["data"],
        outputs=["reduced"],
        keepdims=keepdims,
        name=op_name,
    )
    op_expect(node, inputs=[data], outputs=[reduced], op_type=op_type, op_name=op_name)


def test_reducel2():
    op_name, op_type = "test_reduce_l2_default_axes_keepdims_example", "ReduceL2"
    shape = [3, 2, 2]
    axes = None
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceL2",
        inputs=["data"],
        outputs=["reduced"],
        keepdims=keepdims,
        name=op_name,
    )

    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    # print(data)
    # [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

    reduced = np.sqrt(np.sum(a=np.square(data), axis=axes, keepdims=keepdims == 1))
    # print(reduced)
    # [[[25.49509757]]]

    op_expect(node, inputs=[data], outputs=[reduced], op_type=op_type, op_name=op_name)

    op_name = "test_reduce_l2_default_axes_keepdims_random"
    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sqrt(np.sum(a=np.square(data), axis=axes, keepdims=keepdims == 1))
    node = onnx.helper.make_node(
        "ReduceL2",
        inputs=["data"],
        outputs=["reduced"],
        keepdims=keepdims,
        name=op_name,
    )

    op_expect(node, inputs=[data], outputs=[reduced], op_type=op_type, op_name=op_name)


@pytest.mark.skip(reason="ORT: Unrecognized attribute: axes for operator ReduceLogSu")
def test_reducelogsum():
    op_name, op_type = "test_reduce_log_sum_default", "ReduceLogSum"
    node = onnx.helper.make_node("ReduceLogSum", inputs=["data"], outputs=["reduced"], name=op_name)
    data = np.random.ranf([3, 4, 5]).astype(np.float32)
    reduced = np.log(np.sum(data, keepdims=True))
    op_expect(node, inputs=[data], outputs=[reduced], op_type=op_type, op_name=op_name)

    op_name = "test_reduce_log_sum_negative_axes"
    node = onnx.helper.make_node(
        "ReduceLogSum", inputs=["data"], outputs=["reduced"], axes=[-2], name=op_name
    )
    data = np.random.ranf([3, 4, 5]).astype(np.float32)
    reduced = np.log(np.sum(data, axis=(-2), keepdims=True))
    # print(reduced)
    op_expect(node, inputs=[data], outputs=[reduced], op_type=op_type, op_name=op_name)

    op_name = "test_reduce_log_sum_desc_axes"
    node = onnx.helper.make_node(
        "ReduceLogSum",
        inputs=["data"],
        outputs=["reduced"],
        axes=[2, 1],
        keepdims=0,
        name=op_name,
    )
    data = np.random.ranf([3, 4, 5]).astype(np.float32)
    reduced = np.log(np.sum(data, axis=(2, 1), keepdims=False))
    op_expect(node, inputs=[data], outputs=[reduced], op_type=op_type, op_name=op_name)

    op_name = "test_reduce_log_sum_asc_axes"
    node = onnx.helper.make_node(
        "ReduceLogSum",
        inputs=["data"],
        outputs=["reduced"],
        axes=[0, 1],
        keepdims=0,
        name=op_name,
    )
    data = np.random.ranf([3, 4, 5]).astype(np.float32)
    reduced = np.log(np.sum(data, axis=(0, 1), keepdims=False))
    op_expect(node, inputs=[data], outputs=[reduced], op_type=op_type, op_name=op_name)


def test_reducelogsumexp():
    op_name, op_type = (
        "test_reduce_log_sum_exp_default_axes_keepdims_example",
        "ReduceLogSumExp",
    )
    shape = [3, 2, 2]
    axes = None
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceLogSumExp",
        inputs=["data"],
        outputs=["reduced"],
        keepdims=keepdims,
        name=op_name,
    )

    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
    reduced = np.log(np.sum(np.exp(data), axis=axes, keepdims=keepdims == 1))
    # print(reduced)
    # [[[60.00671387]]]

    op_expect(node, inputs=[data], outputs=[reduced], op_type=op_type, op_name=op_name)

    op_name = "test_reduce_log_sum_exp_default_axes_keepdims_random"
    node = onnx.helper.make_node(
        "ReduceLogSumExp",
        inputs=["data"],
        outputs=["reduced"],
        keepdims=keepdims,
        name=op_name,
    )

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.log(np.sum(np.exp(data), axis=axes, keepdims=keepdims == 1))
    op_expect(node, inputs=[data], outputs=[reduced], op_type=op_type, op_name=op_name)


def test_reducemax():
    op_name, op_type = "test_reduce_max_default_axes_keepdim_example", "ReduceMax"
    shape = [3, 2, 2]
    axes = None
    keepdims = 1
    node = onnx.helper.make_node(
        "ReduceMax",
        inputs=["data"],
        outputs=["reduced"],
        keepdims=keepdims,
        name=op_name,
    )

    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
    reduced = np.maximum.reduce(data, axis=axes, keepdims=keepdims == 1)
    # print(reduced)
    # [[[60.]]]

    op_expect(node, inputs=[data], outputs=[reduced], op_type=op_type, op_name=op_name)

    op_name = "test_reduce_max_default_axes_keepdims_random"
    node = onnx.helper.make_node(
        "ReduceMax",
        inputs=["data"],
        outputs=["reduced"],
        keepdims=keepdims,
        name=op_name,
    )
    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.maximum.reduce(data, axis=axes, keepdims=keepdims == 1)

    op_expect(node, inputs=[data], outputs=[reduced], op_type=op_type, op_name=op_name)


def test_reducemean():
    op_name, op_type = "test_reduce_mean_default_axes_keepdims_example", "ReduceMean"
    shape = [3, 2, 2]
    axes = None
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceMean",
        inputs=["data"],
        outputs=["reduced"],
        keepdims=keepdims,
        name=op_name,
    )

    data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
    reduced = np.mean(data, axis=axes, keepdims=keepdims == 1)
    # print(reduced)
    # [[[18.25]]]

    op_expect(node, inputs=[data], outputs=[reduced], op_type=op_type, op_name=op_name)

    op_name = "test_reduce_mean_default_axes_keepdims_random"

    node = onnx.helper.make_node(
        "ReduceMean",
        inputs=["data"],
        outputs=["reduced"],
        keepdims=keepdims,
        name=op_name,
    )
    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.mean(data, axis=axes, keepdims=keepdims == 1)

    op_expect(node, inputs=[data], outputs=[reduced], op_type=op_type, op_name=op_name)


def test_reducesum():
    batch_size = 32
    op_name = "reduce_sum_1"
    with tf.Graph().as_default():
        input_ph = tf.placeholder(
            dtype=tf.float32, shape=[batch_size, 256], name="input"
        )  # [batchsize, 10]
        input_data = np.random.rand(batch_size, 256).astype(np.float32)
        x = tf.math.reduce_sum(input_ph, axis=1, name=op_name)
        _ = tf.identity(x, name="output")
        verify_tf_with_trt_result([input_data], ["input:0"], ["output:0"], op_name=op_name)


def test_maxunpool():
    def verify_maxunpool(
        data, indices, kernel_shape, strides, output_shape=None, pads=None, op_name=None
    ):
        input_names = ["xT", "xI"]
        input_info = [
            helper.make_tensor_value_info("xT", TensorProto.FLOAT, list(data.shape)),
            helper.make_tensor_value_info("xI", TensorProto.INT64, list(indices.shape)),
        ]
        input_values = [data, indices]
        # input_values = [data ]
        if output_shape is not None:
            input_names.append("output_shape")
            input_info.append(
                helper.make_tensor_value_info(
                    "output_shape", TensorProto.INT64, list(output_shape.shape)
                )
            )
            input_values.append(output_shape)
        else:
            # Compute expected output shape
            output_shape = np.asarray(([1, 1] + list(strides))) * np.asarray(list(data.shape))
            output_shape += np.asarray(([0, 0] + list(kernel_shape))) - np.asarray(
                ([0, 0] + list(strides))
            )
            if pads is not None:
                output_shape -= np.asarray(
                    [0, 0] + list(np.sum(np.reshape(list(pads), [-1, 2]), axis=-1))
                )
        output_shape = [int(i) for i in output_shape]

        node = helper.make_node(
            "MaxUnpool",
            inputs=input_names,
            outputs=["y"],
            kernel_shape=kernel_shape,
            name=op_name,
        )

        if pads is not None:
            pad_attr = helper.make_attribute("pads", pads)
            node.attribute.append(pad_attr)

        if strides is not None:
            strides_attr = helper.make_attribute("strides", strides)
            node.attribute.append(strides_attr)

        graph = helper.make_graph(
            [node],
            "maxunpool_test",
            inputs=input_info,
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
        )

        model = helper.make_model(graph, producer_name="size_test")
        verify_with_ort_with_trt(model, input_values, op_name=op_name, opset=11)

    # NOCC:invalid-name(其他:onnx example)
    xT = np.array([[[[5, 6], [7, 8]]]], dtype=np.float32)
    # NOCC:invalid-name(其他:onnx example)
    xI = np.array([[[[0, 7], [13, 15]]]], dtype=np.int64)
    verify_maxunpool(xT, xI, [2, 2], strides=[2, 2], op_name="max_unpool_1")


def _test_forward_one_hot(indices_shape, depth, on_value, off_value, axis, out_dtype, op_name):
    inp_array1 = np.random.randint(0, 5, size=indices_shape)
    with tf.Graph().as_default():
        in1 = tf.placeholder(shape=inp_array1.shape, dtype=inp_array1.dtype, name="input")
        out = tf.one_hot(in1, depth, on_value, off_value, axis, dtype=out_dtype, name=op_name)
        out = tf.identity(out, "output")
        verify_tf_with_trt_result([inp_array1], ["input:0"], ["output:0"], op_name)
        # compare_tf_with_tvm(inp_array1, in1.name, out.name)


def test_forward_one_hot():
    _test_forward_one_hot((3,), 3, 1.0, 0.0, -1, "float32", "onehot_2")


def test_where():
    op_name, op_type = "test_where", "Where"
    node = onnx.helper.make_node(
        "Where", inputs=["condition", "x", "y"], outputs=["z"], name=op_name
    )
    condition = np.array([[1, 0], [1, 1]], dtype=bool)
    x = np.array([[1, 2], [3, 4]], dtype=np.int64)
    y = np.array([[9, 8], [7, 6]], dtype=np.int64)
    z = np.where(condition, x, y)  # expected output [[1, 8], [3, 4]]
    op_expect(node, inputs=[condition, x, y], outputs=[z], op_type=op_type, op_name=op_name)


def _test_slice_iteration_v1(indata, outdata, starts, ends, axes=None):
    op_name = "slice_0"
    if axes:
        y = helper.make_node(
            "Slice", ["in"], ["out"], axes=axes, starts=starts, ends=ends, name=op_name
        )
    else:
        y = helper.make_node("Slice", ["in"], ["out"], starts=starts, ends=ends, name=op_name)

    graph = helper.make_graph(
        [y],
        "slice_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outdata.shape))],
    )

    model = helper.make_model(graph, producer_name="slice_test")
    # verify_with_ort_with_trt(model, [indata], [outdata.shape], op_name=op_name, opset=1)
    verify_with_ort_with_trt(model, [indata], op_name=op_name, opset=1)


def test_slice():
    x = np.random.randn(20, 10, 5).astype(np.float32)
    _test_slice_iteration_v1(x, x[0:3, 0:10], starts=(0, 0), ends=(3, 10), axes=(0, 1))


def verify_pad_v11(indata, pads, mode="constant", value=0.0):
    op_name = "pad_001"
    indata = np.array(indata).astype(np.float32)
    #  numpy expect result
    len_dim = len(pads) // 2
    np_pads = [(pads[i], pads[i + len_dim]) for i in range(len_dim)]
    pads = np.array(pads)
    #  onnx graph
    if mode in ["edge", "reflect"]:
        inputs = [indata]
        outdata = np.pad(indata, pad_width=np_pads, mode=mode)
        node = helper.make_node(
            "Pad", inputs=["input", "pads"], outputs=["output"], mode=mode, name=op_name
        )
        graph = helper.make_graph(
            [node],
            "pad_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape)),
                helper.make_tensor_value_info("pads", TensorProto.INT64, (len(pads),)),
            ],
            initializer=[helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads)],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
            ],
        )
    else:
        inputs = [indata]
        outdata = np.pad(indata, pad_width=np_pads, mode="constant", constant_values=value)
        node = helper.make_node(
            "Pad",
            inputs=["input", "pads", "constant_value"],
            outputs=["output"],
            mode="constant",
            name=op_name,
        )
        graph = helper.make_graph(
            [node],
            "pad_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape)),
                helper.make_tensor_value_info("pads", TensorProto.INT64, (len(pads),)),
                helper.make_tensor_value_info("constant_value", TensorProto.FLOAT, (1,)),
            ],
            initializer=[
                helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads),
                helper.make_tensor("constant_value", TensorProto.FLOAT, (1,), [value]),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
            ],
        )
    model = helper.make_model(graph, producer_name="pad_test")
    verify_with_ort_with_trt(model, inputs, op_name, opset=11)


@pytest.mark.skip(reason="TensorRT segmentfault")
def test_pad():
    verify_pad_v11(np.random.randn(2, 2).astype(np.float32), [0, 1, 0, 0], "constant", 0.0)


@pytest.mark.skip(reason="TensorRT segmentfault")
def test_batch_norm():
    def verify_batch_norm(in_shape):
        op_name = "batchNorm_{}".format(sum(in_shape))
        batchnorm = onnx.helper.make_node(
            "BatchNormalization",
            inputs=["x", "scale", "B", "mean", "var"],
            outputs=["Y"],
            name=op_name,
        )

        graph = helper.make_graph(
            [batchnorm],
            "batchnorm_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, list(in_shape)),
                helper.make_tensor_value_info("scale", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("B", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("mean", TensorProto.FLOAT, [in_shape[1]]),
                helper.make_tensor_value_info("var", TensorProto.FLOAT, [in_shape[1]]),
            ],
            outputs=[helper.make_tensor_value_info("Y", TensorProto.FLOAT, list(in_shape))],
        )

        model = helper.make_model(graph, producer_name="batchnorm_test")
        # X, scale, b, mean, var
        inshapes = [in_shape, in_shape[1], in_shape[1], in_shape[1], in_shape[1]]
        inputs = [np.random.uniform(size=ishape).astype("float32") for ishape in inshapes]

        verify_with_ort_with_trt(model, inputs, op_name=op_name)

    verify_batch_norm([1, 3, 224, 224])
    verify_batch_norm([1, 3, 24, 24])
    verify_batch_norm([16, 3, 24, 24])
    verify_batch_norm([16, 16, 24, 24])
    verify_batch_norm([16, 16, 10, 10])


def verify_softmax(inshape, axis, op_name):
    indata = np.random.uniform(size=inshape).astype(np.float32)
    outshape = inshape
    y = helper.make_node("Softmax", ["in"], ["out"], name=op_name)
    if axis is not None:
        axis_attr = helper.make_attribute("axis", axis)
        y.attribute.append(axis_attr)

    graph = helper.make_graph(
        [y],
        "Softmax_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(indata.shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(outshape))],
    )

    model = helper.make_model(graph, producer_name="Softmax_test")
    verify_with_ort_with_trt(model, [indata], op_name=op_name)


def test_softmax():
    verify_softmax((1, 10), None, op_name="softmax_0")
    # verify_softmax((1, 10), 1, op_name='softmax_1')


def verify_mod(x_shape, y_shape, fmod, out_shape, dtype="float32", op_name=""):
    x_np = np.random.uniform(-100.0, 100.0, x_shape).astype(dtype)
    y_np = np.random.uniform(-100.0, 100.0, y_shape).astype(dtype)
    y_np = np.where(y_np == 0, 1, y_np)  # remove 0's to avoid division by zero error

    mod_node = helper.make_node("Mod", inputs=["x", "y"], outputs=["z"], fmod=fmod, name=op_name)

    onnx_dtype = TensorProto.FLOAT if dtype == "float32" else TensorProto.INT32
    graph = helper.make_graph(
        [mod_node],
        "mod_test",
        inputs=[
            helper.make_tensor_value_info("x", onnx_dtype, list(x_shape)),
            helper.make_tensor_value_info("y", onnx_dtype, list(y_shape)),
        ],
        outputs=[helper.make_tensor_value_info("z", onnx_dtype, list(out_shape))],
    )
    model = helper.make_model(graph, producer_name="mod_test")
    # verify_with_ort_with_trt(model, [x_np, y_np], [out_shape], op_name=op_name)
    verify_with_ort_with_trt(model, [x_np, y_np], op_name=op_name)


def test_mod():
    # Mod
    verify_mod(
        x_shape=[1, 32, 32],
        y_shape=[1, 1, 32],
        fmod=0,
        out_shape=(1, 32, 32),
        dtype="int32",
        op_name="tvm_mod",
    )


def verify_mean(input_dim, op_name):
    dtype = "float32"
    a_np1 = np.random.uniform(size=input_dim).astype(dtype)
    a_np2 = np.random.uniform(size=input_dim).astype(dtype)
    a_np3 = np.random.uniform(size=input_dim).astype(dtype)

    mean_node = helper.make_node("Mean", ["a_np1", "a_np2", "a_np3"], ["out"], name=op_name)

    graph = helper.make_graph(
        [mean_node],
        "Mean_test",
        inputs=[
            helper.make_tensor_value_info("a_np1", TensorProto.FLOAT, list(input_dim)),
            helper.make_tensor_value_info("a_np2", TensorProto.FLOAT, list(input_dim)),
            helper.make_tensor_value_info("a_np3", TensorProto.FLOAT, list(input_dim)),
        ],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(input_dim))],
    )

    model = helper.make_model(graph, producer_name="Mean_test")
    verify_with_ort_with_trt(model, [a_np1, a_np2, a_np3], op_name=op_name)


def test_forward_mean():
    verify_mean((1, 3, 20, 20), op_name="mean_111")
    verify_mean((20, 20), op_name="mean_222")


def verify_instance_norm(shape, axis=1, op_name="default"):
    x = np.random.randn(*shape).astype(np.float32)
    gamma = np.random.randn(shape[1]).astype(np.float32)
    beta = np.random.randn(shape[1]).astype(np.float32)
    epsilon = 1e-5

    node = onnx.helper.make_node(
        "InstanceNormalization",
        inputs=["x", "gamma", "beta"],
        outputs=["y"],
        epsilon=epsilon,
        name=op_name,
    )
    graph = helper.make_graph(
        [node],
        "instance_norm_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(shape)),
            helper.make_tensor_value_info("gamma", TensorProto.FLOAT, (shape[1],)),
            helper.make_tensor_value_info("beta", TensorProto.FLOAT, (shape[1],)),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(shape))],
    )
    model = helper.make_model(graph, producer_name="instance_norm_test")
    verify_with_ort_with_trt(model, [x, gamma, beta], op_name=op_name)


def test_instance_norm():
    verify_instance_norm((2, 3, 4, 5), op_name="instance_norm")
    # verify_instance_norm((32, 64, 80, 64))
    # verify_instance_norm((8, 6, 5))
    # verify_instance_norm((8, 7, 6, 5, 4))


def verify_lrn(shape, nsize, dtype, alpha=None, beta=None, bias=None, op_name=None):
    in_array = np.random.uniform(size=shape).astype(dtype)

    if alpha is None and beta is None and bias is None:
        alpha = 0.0001
        beta = 0.75
        bias = 1.0
        node = onnx.helper.make_node(
            "LRN", inputs=["in"], outputs=["out"], size=nsize, name=op_name
        )
    else:
        node = onnx.helper.make_node(
            "LRN",
            inputs=["in"],
            outputs=["out"],
            alpha=alpha,
            beta=beta,
            bias=bias,
            size=nsize,
            name=op_name,
        )

    graph = helper.make_graph(
        [node],
        "lrn_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, list(shape))],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, list(shape))],
    )
    model = helper.make_model(graph, producer_name="lrn_test")
    verify_with_ort_with_trt(model, [in_array], op_name=op_name)


def test_lrn():
    verify_lrn((5, 5, 5, 5), 3, "float32", op_name="test_lrn_1")
    verify_lrn(
        (5, 5, 5, 5),
        3,
        "float32",
        alpha=0.0002,
        beta=0.5,
        bias=2.0,
        op_name="test_lrn_2",
    )


def test_lstm():
    # # Different activation testing.
    # # Default value hardsigmoid.
    verify_rnn(
        seq_length=2,
        batch_size=1,
        input_size=16,
        hidden_size=32,
        use_bias=False,
        activations=["HardSigmoid", "Tanh", "Tanh"],
        rnn_type="LSTM",
        op_name="test_lstm_without_bias",
        layout=1,
    )


def test_binary_ops():
    in_shape = (1, 2, 3, 3)
    dtype = "float32"
    out_shape = in_shape

    def verify_binary_ops(op, x, y, out_type="float32", op_name=None):
        z = helper.make_node(op, ["in1", "in2"], ["out"], name=op_name)
        graph = helper.make_graph(
            [z],
            "_test",
            inputs=[
                helper.make_tensor_value_info("in1", TensorProto.FLOAT, x.shape),
                helper.make_tensor_value_info("in2", TensorProto.FLOAT, y.shape),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "out",
                    mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(out_type)],
                    list(out_shape),
                )
            ],
        )
        model = helper.make_model(graph, producer_name="_test")
        verify_with_ort_with_trt(model, [x, y], op_name=op_name)

    x = np.random.uniform(size=in_shape).astype(dtype)
    y = np.random.uniform(size=in_shape).astype(dtype)
    z = np.random.uniform(size=(3,)).astype(dtype)
    verify_binary_ops("Sub", x, y, op_name="sub_1")
    verify_binary_ops("Sub", x, z, op_name="sub_2")


def verify_reduce_func(func, data, axis, keepdims, op_name=None):
    inshape = data.shape
    outshape = np.sum(data, axis=axis, keepdims=keepdims == 1).shape

    if axis:
        node = onnx.helper.make_node(
            func,
            inputs=["x"],
            outputs=["y"],
            axes=axis,
            keepdims=keepdims,
            name=op_name,
        )
    else:
        node = onnx.helper.make_node(
            func, inputs=["x"], outputs=["y"], keepdims=keepdims, name=op_name
        )

    graph = helper.make_graph(
        [node],
        "reduce_test",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, list(inshape))],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, list(outshape))],
    )

    model = helper.make_model(graph, producer_name="reduce_test")

    verify_with_ort_with_trt(model, [data], opset=11, op_name=op_name)


def test_all_reduce_funcs():
    funcs = [
        # "ReduceMax",
        # "ReduceMean",
        # "ReduceMin",
        # "ReduceProd",
        # "ReduceSum",
        # "ReduceSumSquare",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceL1",
        "ReduceL2",
    ]

    for func in funcs:
        for keepdims in [True, False]:
            verify_reduce_func(
                func,
                np.random.randn(3, 2, 2).astype(np.float32),
                axis=None,
                keepdims=keepdims,
                op_name=func + str(int(keepdims)) + "1",
            )

            verify_reduce_func(
                func,
                np.random.randn(3, 2, 3).astype(np.float32),
                axis=None,
                keepdims=keepdims,
                op_name=func + str(int(keepdims)) + "2",
            )

            verify_reduce_func(
                func,
                np.random.randn(3, 3, 3).astype(np.float32),
                axis=(1,),
                keepdims=keepdims,
                op_name=func + str(int(keepdims)) + "3",
            )

            verify_reduce_func(
                func,
                np.random.randn(3, 3, 3, 1).astype(np.float32),
                axis=(1, 2),
                keepdims=keepdims,
                op_name=func + str(int(keepdims)) + "4",
            )

            verify_reduce_func(
                func,
                np.random.randn(3, 3, 3, 1).astype(np.float32),
                axis=(1,),
                keepdims=keepdims,
                op_name=func + str(int(keepdims)) + "5",
            )

            verify_reduce_func(
                func,
                np.random.randn(1, 3, 4, 1).astype(np.float32),
                axis=(1,),
                keepdims=keepdims,
                op_name=func + str(int(keepdims)) + "6",
            )


def verify_split(indata, outdatas, split, axis=0, pass_split=True, opset=11, op_name=None):
    indata = np.array(indata).astype(np.float32)
    outdatas = [np.array(o).astype(np.float32) for o in outdatas]
    inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape))]
    input_names = ["input"]
    initializer = []

    if split:
        split_index = range(len(split))
    else:
        split_index = range(len(outdatas))

    if pass_split:
        if opset >= 13:
            input_names.append("split")
            np_split = np.array(split).astype(np.int64)
            inputs.append(
                helper.make_tensor_value_info("split", TensorProto.INT64, list(np_split.shape))
            )
            indata = [indata, np_split]
            initializer.append(
                helper.make_tensor("split", TensorProto.INT64, list(np_split.shape), np_split)
            )
    node = helper.make_node(
        "Split",
        inputs=input_names,
        outputs=["output_{}".format(i) for i in range(len(split_index))],
        axis=axis,
        name=op_name,
    )

    if pass_split and opset < 13:
        split_attr = helper.make_attribute("split", split)
        node.attribute.append(split_attr)

    graph = helper.make_graph(
        [node],
        "split_test",
        inputs=inputs,
        initializer=initializer,
        outputs=[
            helper.make_tensor_value_info(
                "output_{}".format(i), TensorProto.FLOAT, list(outdatas[i].shape)
            )
            for i in range(len(split_index))
        ],
    )
    model = helper.make_model(graph, producer_name="split_test")
    verify_with_ort_with_trt(model, indata, opset=opset, op_name=op_name)


def test_split():
    # 1D
    verify_split(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [2, 2, 2],
        0,
        op_name="split_1",
    )
    verify_split(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
        [2, 2, 2],
        0,
        False,
        op_name="split_2",
    )
    # 2D
    verify_split(
        [[1.0, 2.0, 3.0, 4.0], [7.0, 8.0, 9.0, 10.0]],
        [[[1.0, 2.0], [7.0, 8.0]], [[3.0, 4.0], [9.0, 10.0]]],
        [2, 2],
        1,
        op_name="split_4",
    )
    # Split evenly (unstack)
    verify_split([1, 2, 3], [[1], [2], [3]], False, 0, False, op_name="split_5")
    # Split a single value to a single value
    verify_split([1], [[1]], [1], pass_split=True, op_name="split_6")


def verify_xor(x_shape, y_shape, op_name=None):
    x_np = np.random.choice(a=[False, True], size=x_shape).astype("bool")
    y_np = np.random.choice(a=[False, True], size=y_shape).astype("bool")

    np_out = np.logical_xor(x_np, y_np)
    out_shape = np_out.shape

    xor_node = helper.make_node("Xor", inputs=["x", "y"], outputs=["z"], name=op_name)

    onnx_dtype = TensorProto.BOOL
    graph = helper.make_graph(
        [xor_node],
        "xor_test",
        inputs=[
            helper.make_tensor_value_info("x", onnx_dtype, list(x_shape)),
            helper.make_tensor_value_info("y", onnx_dtype, list(y_shape)),
        ],
        outputs=[helper.make_tensor_value_info("z", onnx_dtype, list(out_shape))],
    )
    model = helper.make_model(graph, producer_name="xor_test")
    verify_with_ort_with_trt(model, [x_np, y_np], op_name=op_name)


@pytest.mark.skip(reason="TensorRT segmentfault")
def test_xor():
    # XOR
    verify_xor(x_shape=[1, 32, 32], y_shape=[1, 32, 32], op_name="test_xor_1")

    # Xor broadcast
    verify_xor(x_shape=[1, 32, 32], y_shape=[1, 1, 32], op_name="test_xor_2")


def verify_if(cond_array, op_name):
    # Given a bool scalar input cond.
    # return constant tensor x if cond is True, otherwise return constant tensor y.
    then_out = onnx.helper.make_tensor_value_info("then_out", onnx.TensorProto.FLOAT, [5])
    else_out = onnx.helper.make_tensor_value_info("else_out", onnx.TensorProto.FLOAT, [5])

    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

    then_const_node = onnx.helper.make_node(
        "Constant", inputs=[], outputs=["then_out"], value=numpy_helper.from_array(x)
    )

    else_const_node = onnx.helper.make_node(
        "Constant", inputs=[], outputs=["else_out"], value=numpy_helper.from_array(y)
    )

    then_body = onnx.helper.make_graph([then_const_node], "then_body", [], [then_out])

    else_body = onnx.helper.make_graph([else_const_node], "else_body", [], [else_out])

    if_node = onnx.helper.make_node(
        "If",
        inputs=["cond"],
        outputs=["res"],
        then_branch=then_body,
        else_branch=else_body,
        name=op_name,
    )

    if_graph = onnx.helper.make_graph(
        [if_node],
        "if_outer",
        inputs=[
            onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, []),
        ],
        outputs=[
            onnx.helper.make_tensor_value_info("res", onnx.TensorProto.FLOAT, [5]),
        ],
    )

    if_model = onnx.helper.make_model(if_graph)
    if cond_array:
        cond = np.array([1]).astype("bool")
    else:
        cond = np.array(1).astype("bool")
    verify_with_ort_with_trt(if_model, [cond], op_name=op_name)


@pytest.mark.skip(
    reason="ORT: NOT_IMPLEMENTED : Could not find an implementation for If(19) node with name 'if_test_1'"
)
def test_if():
    # Confirm that if works with cond as an array or scalar.
    verify_if(cond_array=False, op_name="if_test_1")
    verify_if(cond_array=True, op_name="if_test_2")


def test_softmax_cross_entropyloss():
    op_name = "test_SoftmaxCrossEntropyLoss"
    reduction = "mean"
    ignore_index = np.int64(-1)

    node = onnx.helper.make_node(
        "SoftmaxCrossEntropyLoss",
        inputs=["x", "y", "w"],
        outputs=["z"],
        reduction=reduction,
        ignore_index=ignore_index,
        name=op_name,
    )
    # NOCC:invalid-name(其他:onnx example)
    N, C, dim1 = 3, 5, 6
    np.random.seed(0)
    x = np.random.rand(N, C, dim1).astype(np.float32)
    labels = np.random.randint(0, high=C, size=(N, dim1)).astype(np.int64)
    labels[0][0] = -1
    weight = np.random.rand(C).astype(np.float32)
    from onnx.backend.test.case.node.softmaxcrossentropy import softmaxcrossentropy

    sce = softmaxcrossentropy(
        x, labels, weight=weight, reduction=reduction, ignore_index=ignore_index
    )

    op_expect(
        node,
        inputs=[x, labels, weight],
        outputs=[sce],
        op_name=op_name,
        op_type="float32",
    )


def _test_logical(method, op_name):
    batch_size = 128
    input_data = (2 * np.random.rand(batch_size, 256) - 1).astype(np.float32)
    with tf.Graph().as_default():
        input_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, 256], name="input")
        x = tf.nn.relu(input_ph)
        mask = tf.cast(x, tf.bool)
        x = tf.nn.relu(tf.layers.dense(x, 256))
        y = x
        x = tf.cast(x, tf.bool)
        if method == "or":
            x = tf.math.logical_or(x, mask, name=op_name)
        elif method == "and":
            x = tf.math.logical_and(x, mask, name=op_name)
        elif method == "not":
            x = tf.math.logical_not(x, name=op_name)
        elif method == "equal":
            x = tf.math.equal(x, mask, name=op_name)
        elif method == "greater":
            x = tf.math.greater(y, input_ph, name=op_name)
        elif method == "xor":
            x = tf.math.logical_xor(x, mask, name=op_name)
        elif method == "is_inf":
            x = tf.math.is_inf(input_ph, name=op_name)
        elif method == "is_nan":
            x = tf.math.is_nan(input_ph, name=op_name)
        _ = tf.identity(x, name="output")
        verify_tf_with_trt_result([input_data], ["input:0"], ["output:0"], op_name)


@pytest.mark.skip(reason="TensorRT segmentfault")
def test_logical():
    _test_logical("or", "test_logical_or")
    _test_logical("and", "test_logical_and")
    _test_logical("not", "test_logical_not")
    _test_logical("equal", "test_logical_equal")
    _test_logical("greater", "test_logical_greater")
    _test_logical("xor", "test_logical_xor")
    _test_logical("is_inf", "test_logical_inf")
    _test_logical("is_nan", "test_logical_nan")


@pytest.mark.skip(reason="TensorRT segmentfault")
def test_scatternd():
    batch_size = 32
    op_name = "scatternd"
    with tf.Graph().as_default():
        input_ph = tf.placeholder(
            dtype=tf.float32, shape=[batch_size, 10], name="input"
        )  # [batchsize, 10]
        input_data = np.random.rand(batch_size, 10).astype(np.float32)
        x = tf.layers.dense(input_ph, 1)
        # duplicated indices case (undefined)
        # test ScatterND (32, 128, 128, 256) (32, 600, 3) (32, 600, 256)
        data = tf.tile(tf.reshape(tf.layers.dense(x, 128 * 128), [-1, 128, 128, 1]), [1, 1, 1, 256])
        x = tf.add(x, 1)
        idx = tf.reshape(tf.layers.dense(x, 600 * 3), [-1, 600, 3])
        idx = tf.cast(tf.clip_by_value(idx, 0, 1), tf.int32)
        indices = idx
        # indices = tf.zeros([32, 600, 3], dtype=tf.dtypes.int32)
        # indices = tf.stack([tf.range(tf.shape(x)[0]), idx], axis=1)
        x = tf.add(x, 2)
        updates = tf.reshape(tf.layers.dense(x, 600 * 256), [-1, 600, 256])
        # updates = tf.ones([32, 600, 256])
        x = tf.tensor_scatter_nd_update(data, indices, updates, name=op_name)
        # x = tf.scatter_nd(indices, updates, data.shape)
        _ = tf.identity(x, name="output")
        verify_tf_with_trt_result([input_data], ["input:0"], ["output:0"], op_name)
