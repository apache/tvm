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
# pylint: disable=unused-argument, import-outside-toplevel, inconsistent-return-statements
"""
TFLite testcases
================
This article is a test script to test TFLite operator with Relay.
"""
from __future__ import print_function
from functools import partial
from distutils.version import LooseVersion

import os
import tempfile
import typing
from packaging import version as package_version
import pytest
import numpy as np

from PIL import Image

from tflite.BuiltinOperator import BuiltinOperator

try:
    import tensorflow.compat.v1 as tf

    # tensorflow.python.framework.ops module itself is not part of
    # TensorFlow's public API: the precise contents of that module
    # may vary from one version to the next
    import tensorflow.compat.v1 as ops
except ImportError:
    import tensorflow as tf
    import tensorflow as ops
from tensorflow.python.framework import constant_op

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import variables
from tensorflow import raw_ops

try:
    from tensorflow import lite as interpreter_wrapper
except ImportError:
    from tensorflow.contrib import lite as interpreter_wrapper

import tvm
import tvm.relay.testing.tf as tf_testing
from tvm.contrib.download import download_testdata
from tvm import relay, ir
from tvm.contrib import graph_executor
from relay.utils.tag_span import _set_span, _create_span, _verify_structural_equal_with_span


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
def get_real_image(im_height, im_width, quantized=True):
    repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"
    img_name = "elephant-299.jpg"
    image_url = os.path.join(repo_base, img_name)
    img_path = download_testdata(image_url, img_name, module="data")
    image = Image.open(img_path).resize((im_height, im_width))
    x = np.array(image).astype("uint8") if quantized else np.array(image).astype("float32")
    data = np.reshape(x, (1, im_height, im_width, 3))
    return data


def pre_processed_image(height, width):
    """Image preprocessed"""
    repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"
    img_name = "elephant-299.jpg"
    image_url = os.path.join(repo_base, img_name)
    img_path = download_testdata(image_url, img_name, module="data")
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    with tf.name_scope("eval_image"):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.central_crop(image, central_fraction=0.875)
    # Resize the image to the specified height and width.
    image = tf.image.resize(image, [height, width], align_corners=False)
    image = tf.expand_dims(image, axis=0)
    return image


def get_real_image_object_detection(im_height, im_width):
    repo_base = "https://github.com/dmlc/web-data/raw/main/gluoncv/detection/"
    img_name = "street_small.jpg"
    image_url = os.path.join(repo_base, img_name)
    img_path = download_testdata(image_url, img_name, module="data")
    image = Image.open(img_path).resize((im_height, im_width))
    x = np.array(image).astype("uint8")
    data = np.reshape(x, (1, im_height, im_width, 3))
    return data


def vmobj_to_list(obj):
    """Converts TVM objects returned by VM execution to Python List."""
    if isinstance(obj, tvm.nd.NDArray):
        return [obj.numpy().tolist()]
    elif isinstance(obj, tvm.runtime.container.ADT):
        result = []
        for f in obj:
            result.extend(vmobj_to_list(f))
        return result
    elif isinstance(obj, tvm.relay.backend.interpreter.ConstructorValue):
        if obj.constructor.name_hint == "Cons":
            t_l = vmobj_to_list(obj.fields[1])
            h_d = vmobj_to_list(obj.fields[0])
            h_d.extend(t_l)
            return h_d
        elif obj.constructor.name_hint == "Nil":
            return []
        elif "tensor_nil" in obj.constructor.name_hint:
            return [0]
        elif "tensor" in obj.constructor.name_hint:
            return [obj.fields[0].numpy()]
        else:
            raise RuntimeError(f"Unknown object type: {obj.constructor.name_hint}")
    else:
        raise RuntimeError(f"Unknown object type: {type(obj)}")


def _quantize_keras_model(
    keras_model,
    representative_data_gen,
    is_float_input=False,
    is_float_output=False,
    int_quant_dtype=tf.int8,
):
    """Utility function to quantize a Keras model using TFLite converter."""
    converter = interpreter_wrapper.TFLiteConverter.from_keras_model(keras_model)
    if int_quant_dtype == tf.int8:
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        inference_dtype = tf.uint8
    elif int_quant_dtype == tf.int16:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]
        inference_dtype = tf.uint16
    else:
        raise RuntimeError(
            f"Invalid quantized dtype {int_quant_dtype}. Supported types: int8, int16."
        )

    # NOTE: If representative dataset is provided, and inference input type is not set,
    #       then converter will self add quant & dequant Op accordingly.
    if not is_float_input:
        converter.inference_input_type = inference_dtype
    if not is_float_output:
        converter.inference_output_type = inference_dtype

    return converter.convert()


def run_tvm_graph(
    tflite_model_buf,
    input_data,
    input_node,
    num_output=1,
    target="llvm",
    out_names=None,
    mode="graph_executor",
    op_converter=relay.frontend.tflite.OperatorConverter,
):
    """Generic function to compile on relay and execute on tvm"""
    # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
    try:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except ImportError as exc:
        raise ImportError("The tflite package must be installed") from exc

    input_data = convert_to_list(input_data)
    input_node = convert_to_list(input_node)

    shape_dict = {}
    dtype_dict = {}
    for i, node in enumerate(input_node):
        shape_dict[node] = input_data[i].shape
        dtype_dict[node] = input_data[i].dtype.name

    with tvm.testing.disable_span_filling():
        mod, params = relay.frontend.from_tflite(
            tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict, op_converter=op_converter
        )
    with tvm.testing.enable_span_filling():
        mod_with_span, _ = relay.frontend.from_tflite(
            tflite_model, shape_dict=shape_dict, dtype_dict=dtype_dict, op_converter=op_converter
        )
    assert tvm.ir.structural_equal(mod["main"], mod_with_span["main"])

    if mode in ["debug", "vm"]:
        inputs = []
        for param in mod["main"].params:
            found = False
            for i, n in enumerate(input_node):
                if n == param.name_hint:
                    found = True
                    inputs.append(tvm.nd.array(input_data[i]))
                    break
            # Interpreter doesn't bind constants, so still need to find in params
            if not found:
                inputs.append(tvm.nd.array(params[param.name_hint]))
        result = relay.create_executor(mode, mod=mod, device=tvm.cpu(), target="llvm").evaluate()(
            *inputs
        )
        return vmobj_to_list(result)
    else:
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target, params=params)

        dev = tvm.device(target, 0)

        m = graph_executor.GraphModule(lib["default"](dev))
        # set inputs
        for i, node in enumerate(input_node):
            m.set_input(node, tvm.nd.array(input_data[i].astype(input_data[i].dtype)))
        # execute
        m.run()
        # get outputs
        assert out_names is None or num_output == len(
            out_names
        ), f"out_names: {out_names} num_output: {num_output}"
        tvm_output_list = []
        for i in range(0, num_output):
            tvm_output = m.get_output(i)
            tvm_output_list.append(tvm_output.numpy())
        return tvm_output_list


def run_tflite_graph(tflite_model_buf, input_data):
    """Generic function to execute TFLite"""
    input_data = convert_to_list(input_data)

    interpreter = interpreter_wrapper.Interpreter(model_content=tflite_model_buf)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    for i, input_detail in enumerate(input_details):
        interpreter.resize_tensor_input(input_detail["index"], input_data[i].shape)
    interpreter.allocate_tensors()

    # set input
    assert len(input_data) == len(input_details)
    for i, input_detail in enumerate(input_details):
        interpreter.set_tensor(input_detail["index"], input_data[i])

    # Run
    interpreter.invoke()

    # get output
    tflite_output = []
    for _, output_detail in enumerate(output_details):
        tflite_output.append(interpreter.get_tensor(output_detail["index"]))

    return tflite_output


def compare_tflite_with_tvm(
    in_data: typing.List[np.ndarray],
    in_name: typing.List[str],
    input_tensors: typing.List,
    output_tensors: typing.List,
    init_global_variables: bool = False,
    out_names=None,
    quantized=False,
    input_range=None,
    mode="graph_executor",
    experimental_new_converter=False,
    fp16_quantized=False,
    int_quant_dtype=tf.uint8,
):
    """Generic function to generate and compare TFLite and TVM output"""
    in_data = convert_to_list(in_data)
    in_name = convert_to_list(in_name)
    out_names = convert_to_list(out_names)
    in_node = [0] * len(in_name)
    for i, _ in enumerate(in_name):
        in_node[i] = in_name[i].split(":")[0] if ":" in in_name[i] else in_name[i]

    with tf.Session() as sess:
        if init_global_variables:
            sess.run(variables.global_variables_initializer())
        # convert to tflite model
        converter = tf.lite.TFLiteConverter.from_session(sess, input_tensors, output_tensors)

        if len(input_tensors) > 1:
            if len(input_tensors[0].shape) <= 4 and len(input_tensors[1].shape) <= 4:
                converter._experimental_disable_batchmatmul_unfold = True
            else:
                converter._experimental_disable_batchmatmul_unfold = False

        converter.experimental_new_converter = experimental_new_converter
        if quantized:
            if int_quant_dtype == tf.int16:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.target_spec.supported_ops = [
                    tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
                ]
            elif int_quant_dtype == tf.int8:
                converter.inference_type = tf.lite.constants.INT8
            else:
                # default to int8 quantization
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
                    print("Min and max of the input range for tensor " + i + " can't be equal")
                mean = -input_range[i][0] * quant_scale
                input_stats[i] = (mean, quant_scale)
            converter.quantized_input_stats = input_stats
        elif fp16_quantized:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        tflite_model_buffer = converter.convert()
        tflite_output = run_tflite_graph(tflite_model_buffer, in_data)

        for device in ["llvm"]:
            _ = tvm.device(device, 0)
            if not tvm.testing.device_enabled(device):
                print(f"Skip because {device} is not enabled")
                continue

            tvm_output = run_tvm_graph(
                tflite_model_buffer,
                in_data,
                in_node,
                target=device,
                num_output=len(out_names),
                out_names=out_names,
                mode=mode,
            )
            # WARNING: the results could well be random values clipped to 0 or 255 because of badly
            # tuned output range for the specific operator. While adding test ensure that we aren't
            # getting only clipped values in output tensors that still pass the assertion.
            # For reference see _test_elemwise_qnn_out_range()
            if quantized and not fp16_quantized:
                for i, _ in enumerate(tflite_output):
                    # allow absolute tolerance of 1 in the quantized results
                    tvm.testing.assert_allclose(
                        tflite_output[i],  # pylint: disable=unnecessary-list-index-lookup
                        tvm_output[i],
                        atol=1,
                        rtol=1e-5,
                    )
            else:
                for i, _ in enumerate(tflite_output):
                    tvm.testing.assert_allclose(
                        tflite_output[i],  # pylint: disable=unnecessary-list-index-lookup
                        tvm_output[i],
                        atol=1e-5,
                        rtol=1e-5,
                    )


def with_fused_activation_function(input_tensor, fn_name):
    """Fused activation function"""
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
    raise AssertionError(f"Unknown fused_activation_function {fn_name}")


def _test_split(in_shape, axis, num_splits, dtype):
    """internal split tester taking as parameters in_shape, number of tensors to split into
    and dtype (data type)"""

    np_data = np.random.uniform(-5, 5, size=in_shape).astype(dtype)
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=in_shape, dtype=dtype, name="in_data")
        out = array_ops.split(in_data, num_splits, axis=axis)
        num_splits = len(num_splits) if isinstance(num_splits, list) else num_splits
        out_names = ["out_" + str(n) + ":0" for n in range(num_splits)]
        compare_tflite_with_tvm([np_data], ["in_data"], [in_data], out, out_names=out_names)


def test_forward_split():
    """test split layer"""
    # rank 1
    _test_split((3,), 0, 1, "float32")
    _test_split((3,), 0, 3, "float32")
    _test_split((6,), 0, 3, "float32")
    # rank 2
    _test_split((6, 2), 0, 3, "float32")
    _test_split((2, 6), 1, 6, "float32")
    # rank 3
    if package_version.parse(tf.VERSION) >= package_version.parse("1.14.0"):
        _test_split((6, 2, 4), 0, 2, "int32")

    _test_split((2, 6, 4), 1, 3, "float32")
    _test_split((2, 4, 6), 2, 1, "float32")
    # rank 4
    _test_split((6, 1, 3, 5), 0, 3, "float32")
    _test_split((1, 6, 3, 5), 1, 3, "float32")
    _test_split((1, 3, 6, 5), 2, 3, "float32")
    _test_split((1, 3, 5, 6), 3, 3, "float32")
    # split along negative axis
    _test_split((6, 1, 3, 5), -4, 3, "float32")
    _test_split((1, 6, 3, 5), -3, 3, "float32")
    _test_split((1, 3, 6, 5), -2, 3, "float32")
    _test_split((1, 3, 5, 6), -1, 3, "float32")
    # size_splits split
    _test_split((6,), 0, [1, 2, 3], "float32")
    _test_split((3, 6, 4), -2, [1, 4, 1], "float32")


#######################################################################
# slice
# -----


def _test_slice(data, begin, size):
    """One iteration of SLICE"""
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = array_ops.slice(in_data, begin, size)
        compare_tflite_with_tvm(data, "Placeholder:0", [in_data], [out])


def test_forward_slice():
    """SLICE"""
    _test_slice(np.arange(4, dtype=np.float32).reshape((4,)), begin=[0], size=[2])
    _test_slice(np.arange(18, dtype=np.int32).reshape((3, 2, 3)), begin=[1, 0, 0], size=[1, 1, 3])
    # tflite 1.13 outputs nonsense values if size[i] == -1
    if package_version.parse(tf.VERSION) >= package_version.parse("1.14.0"):
        _test_slice(np.arange(8, dtype=np.int32).reshape((2, 4)), begin=[0, 1], size=[-1, -1])
        _test_slice(np.arange(5, dtype=np.int32).reshape((5,)), begin=[4], size=[-1])


#######################################################################
# Topk
# ----
def _test_topk(in_shape, k=1):
    """One iteration of TOPK"""
    data = np.random.uniform(size=in_shape).astype("float32")
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = nn_ops.top_k(in_data, k, name="TopK")
        compare_tflite_with_tvm(data, "Placeholder:0", [in_data], [out[0]])


def test_forward_topk():
    """TOPK"""
    _test_topk((3,), 1)
    _test_topk((3,), 3)
    _test_topk((3, 5, 7), 3)
    _test_topk((3, 5, 7), 3)


#######################################################################
# Gather
# ------


def _test_gather(dshape, indices, axis, dtype, quantized=False, oob=False, wrap_idx=False):
    """One iteration of Gather"""
    indices = np.asarray(indices).astype("int32")
    data = np.random.uniform(1, 10, size=dshape)
    data = data.astype(np.uint8) if quantized else data.astype(dtype)
    with tf.Graph().as_default():
        if wrap_idx:
            in_name = "in_indices"
            indices_expr = array_ops.placeholder(
                shape=indices.shape, dtype=indices.dtype, name=in_name
            )
            in_tensor_name = [in_name + ":0"]
            in_indices = [indices_expr]
        else:
            indices_expr = indices
            indices = []
            in_tensor_name = []
            in_indices = []

        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype, name="in_data")
        if axis:
            out = array_ops.gather(in_data, indices_expr, axis=axis)
        else:
            out = array_ops.gather(in_data, indices_expr)  # tflite conversion fails for None axis
        input_range = {"in_data": (-100, 100)} if quantized else None
        try:
            compare_tflite_with_tvm(
                [data] + indices,
                ["in_data:0"] + in_tensor_name,
                [in_data] + in_indices,
                [out],
                quantized=quantized,
                input_range=input_range,
            )
        except ValueError as exc:
            if not oob:
                raise exc
        except Exception as exc:
            raise exc


def test_forward_gather():
    """GATHER"""
    for quantized in [False, True]:
        for wrap_idx in [False, True]:
            _test_gather((4,), [1], 0, "float32", quantized, wrap_idx)
            _test_gather((4,), [1], None, "int32", quantized, wrap_idx)
            _test_gather((1, 4), [0], 0, "int32", quantized, wrap_idx)
            _test_gather((4,), [[[1, 0], [0, 1]]], 0, "float32", quantized, wrap_idx)
            _test_gather((2, 2), [[[1, 0], [0, 1]]], 1, "int32", quantized, wrap_idx)
            _test_gather((2, 2), [[[1, 0], [0, 1]]], None, "float32", quantized, wrap_idx)
            _test_gather((3, 3, 3), [[[1, 0]]], 0, "int32", quantized, wrap_idx)
            _test_gather((3, 3, 3), [[[1, 0]]], 2, "int32", quantized, wrap_idx)
            _test_gather((4, 3, 5, 6), [[2, 1, 0, 0]], 0, "float32", quantized, wrap_idx)
            _test_gather((3, 3, 3), [[[2, 1]]], -1, "int32", quantized, wrap_idx)
        # Out of boundary error cannot be tested with wrapped index
        _test_gather((4,), [16], 0, "float32", quantized, oob=True)
        _test_gather((1, 3, 3), [12], 0, "int32", quantized, oob=True)
        _test_gather((1, 3, 3), [20], 1, "float32", quantized, oob=True)
        _test_gather((1, 3, 3), [20, 20], 2, "float32", quantized, oob=True)


#######################################################################
# Gather_ND
# ---------


def _test_gather_nd(data, indices):
    """One iteration of GATHER_ND"""
    with tf.Graph().as_default():
        in_data = tf.placeholder(shape=data.shape, dtype=data.dtype, name="data")
        indices_data = tf.placeholder(shape=indices.shape, dtype=indices.dtype, name="indices")
        out = tf.gather_nd(in_data, indices_data)

        compare_tflite_with_tvm(
            [data, indices], ["data:0", "indices:0"], [in_data, indices_data], [out]
        )


def test_forward_gather_nd():
    """GATHER_ND"""
    _test_gather_nd(
        np.array([[[1.2, 2.0], [3.1, 4.1]], [[5.1, 6.1], [7.1, 8.1]]]).astype("float32"),
        np.asarray([[0, 1], [1, 0]]).astype("int32"),
    )
    _test_gather_nd(
        np.reshape(np.arange(30), [5, 6]).astype("int32"), np.asarray([[1, 2]]).astype("int32")
    )
    _test_gather_nd(
        np.reshape(np.arange(12), [2, 3, 2]).astype("int32"),
        np.asarray([[[0, 0], [0, 1]], [[1, 0], [1, 1]]]).astype("int32"),
    )
    _test_gather_nd(
        np.reshape(np.arange(4), [4]).astype("float32"), np.asarray([1]).astype("int32")
    )
    _test_gather_nd(
        np.reshape(np.arange(4), [1, 4]).astype("float32"), np.asarray([0]).astype("int32")
    )
    _test_gather_nd(
        np.reshape(np.arange(4), [1, 4]).astype("float32"), np.asarray([0, 3]).astype("int32")
    )


#######################################################################
# StridedSlice
# ------------


def _test_stridedslice(
    ip_shape,
    begin,
    end,
    stride,
    dtype,
    begin_mask=0,
    end_mask=0,
    new_axis_mask=0,
    shrink_axis_mask=0,
    ellipsis_mask=0,
    quantized=False,
):
    """One iteration of a Stridedslice"""
    data = np.random.uniform(size=ip_shape).astype(dtype)
    data = data.astype(np.uint8) if quantized else data.astype(dtype)
    with tf.Graph().as_default():
        in_data = tf.placeholder(dtype, ip_shape, name="in_data")
        out = array_ops.strided_slice(
            in_data,
            begin,
            end,
            stride,
            begin_mask=begin_mask,
            end_mask=end_mask,
            new_axis_mask=new_axis_mask,
            shrink_axis_mask=shrink_axis_mask,
            ellipsis_mask=ellipsis_mask,
        )
        input_range = {"in_data": (-100, 100)} if quantized else None
        compare_tflite_with_tvm(
            [data], ["in_data:0"], [in_data], [out], quantized=quantized, input_range=input_range
        )


def test_forward_stridedslice():
    """test StridedSlice"""
    for quantized in [False, True]:
        _test_stridedslice(
            (1, 3, 3),
            [0, 0, 0],
            [3, 3, 3],
            [1, 1, 1],
            "float32",
            shrink_axis_mask=7,
            quantized=quantized,
        )
        _test_stridedslice(
            (1, 3, 3),
            [0, 0, 0],
            [3, 3, 3],
            [1, 1, 1],
            "float32",
            shrink_axis_mask=5,
            quantized=quantized,
        )
        _test_stridedslice((2), [1], [1], [1], "float32", shrink_axis_mask=1, quantized=quantized)
        _test_stridedslice(
            (3, 4, 3), [1, -1, 0], [4, -5, 3], [2, -1, 1], "float32", quantized=quantized
        )
        _test_stridedslice(
            (3, 4), [1, 0], [4, 4], [1, 1], "float32", shrink_axis_mask=0, quantized=quantized
        )
        _test_stridedslice(
            (4, 4), [1, 0], [4, 4], [1, 1], "float32", shrink_axis_mask=2, quantized=quantized
        )
        _test_stridedslice(
            (3, 4), [-1, 0], [0, 3], [1, 1], "float32", shrink_axis_mask=1, quantized=quantized
        )


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

        compare_tflite_with_tvm(data, "Placeholder:0", [in_data], [out])


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


def _test_cast(data, cast_dtype, use_mlir=False):
    """One iteration of CAST"""
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = math_ops.cast(in_data, cast_dtype)
        compare_tflite_with_tvm(
            data, "Placeholder:0", [in_data], [out], experimental_new_converter=use_mlir
        )


def test_forward_cast():
    """CAST"""
    for use_mlir in [False, True]:
        _test_cast(
            np.arange(6.0, dtype=np.float32).reshape((1, 6)), cast_dtype=tf.int32, use_mlir=use_mlir
        )
        _test_cast(
            np.arange(6.0, dtype=np.float32).reshape((1, 6)), cast_dtype=tf.uint8, use_mlir=use_mlir
        )
        _test_cast(
            np.arange(6.0, dtype=np.int32).reshape((1, 6)), cast_dtype=tf.int64, use_mlir=use_mlir
        )


#######################################################################
# Batch Mat Mul
# ----
def _test_batch_matmul(
    a_shape, b_shape, dtype, out_dtype, adjoint_a=False, adjoint_b=False, quantized=False
):
    with tf.Graph().as_default():
        a = array_ops.placeholder(shape=a_shape, dtype=dtype, name="A")
        b = array_ops.placeholder(shape=b_shape, dtype=dtype, name="B")
        print(tf.__version__)

        result = raw_ops.BatchMatMulV3(
            x=a, y=b, Tout=out_dtype, adj_x=adjoint_a, adj_y=adjoint_b, name="batchmatmul"
        )
        input_range = {"A": (-100, 100), "B": (-100, 100)} if quantized else None

        a_np = np.random.uniform(high=5.0, size=a_shape).astype(dtype)
        b_np = np.random.uniform(high=5.0, size=b_shape).astype(dtype)
        compare_tflite_with_tvm(
            [a_np, b_np],
            [a.name, b.name],
            [a, b],
            [result],
            experimental_new_converter=True,
            quantized=quantized,
            input_range=input_range,
        )


@pytest.mark.parametrize("config", [("int8", "int32", True), ("float32", "float32", False)])
def test_forward_batch_matmul(config):
    """BATCH_MAT_MUL"""
    _test_batch_matmul(
        (3, 5, 4), (3, 4, 5), dtype=config[0], out_dtype=config[1], quantized=config[2]
    )
    _test_batch_matmul(
        (3, 5, 4),
        (3, 4, 5),
        dtype=config[0],
        out_dtype=config[1],
        adjoint_a=True,
        adjoint_b=True,
        quantized=config[2],
    )
    _test_batch_matmul(
        (3, 5, 4),
        (3, 5, 4),
        dtype=config[0],
        out_dtype=config[1],
        adjoint_a=True,
        adjoint_b=False,
        quantized=config[2],
    )
    _test_batch_matmul(
        (2, 3, 5, 4),
        (1, 3, 5, 4),
        dtype=config[0],
        out_dtype=config[1],
        adjoint_a=True,
        adjoint_b=False,
        quantized=config[2],
    )
    _test_batch_matmul(
        (3, 5, 4),
        (3, 5, 4),
        dtype=config[0],
        out_dtype=config[1],
        adjoint_a=False,
        adjoint_b=True,
        quantized=config[2],
    )
    _test_batch_matmul(
        (2, 3, 5, 4),
        (1, 3, 5, 4),
        dtype=config[0],
        out_dtype=config[1],
        adjoint_a=False,
        adjoint_b=True,
        quantized=config[2],
    )
    _test_batch_matmul(
        (3, 4, 5, 6), (3, 4, 6, 5), dtype=config[0], out_dtype=config[1], quantized=config[2]
    )
    # BatchMatMul doesn't support larger than 4D tensors
    # _test_batch_matmul(
    #    (2, 3, 4, 5, 6), (2, 3, 4, 6, 5), dtype=config[0], out_dtype=config[1], quantized=config[2]
    # )


#######################################################################
# Tile
# ----


def _test_forward_tile(in_shape, reps, dtype):
    data = np.random.uniform(-5, 5, size=in_shape).astype(dtype)

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)

        out = array_ops.tile(in_data, reps)

        compare_tflite_with_tvm(data, "Placeholder:0", [in_data], [out])


def test_forward_tile():
    _test_forward_tile((2,), (3,), "int32")
    _test_forward_tile((2, 2), (2, 3), "float32")


######################################################################
# BatchToSpaceND
# --------------


def _test_batch_to_space_nd(input_shape, block_shape, crops, dtype="int32"):
    data = np.random.uniform(0, 5, size=input_shape).astype(dtype)

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=input_shape, dtype=dtype)

        out = array_ops.batch_to_space_nd(in_data, block_shape, crops)

        compare_tflite_with_tvm(data, "Placeholder:0", [in_data], [out])


def test_forward_batch_to_space_nd():
    # test cases: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/batch-to-space-n-d
    _test_batch_to_space_nd(input_shape=[4, 1, 1, 1], block_shape=[2, 2], crops=[[0, 0], [0, 0]])

    _test_batch_to_space_nd(input_shape=[4, 1, 1, 3], block_shape=[2, 2], crops=[[0, 0], [0, 0]])

    _test_batch_to_space_nd(input_shape=[4, 2, 2, 1], block_shape=[2, 2], crops=[[0, 0], [0, 0]])

    _test_batch_to_space_nd(input_shape=[4, 3, 3, 1], block_shape=[2, 2], crops=[[0, 1], [0, 1]])


######################################################################
# SpaceToBatchND
# --------------


def _test_space_to_batch_nd(input_shape, block_shape, paddings, dtype="int32"):
    data = np.random.uniform(0, 5, size=input_shape).astype(dtype)

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=input_shape, dtype=dtype)

        out = array_ops.space_to_batch_nd(in_data, block_shape, paddings)

        compare_tflite_with_tvm(data, "Placeholder:0", [in_data], [out])


def test_forward_space_to_batch_nd():
    # test cases: https://www.tensorflow.org/api_docs/python/tf/space_to_batch_nd
    _test_space_to_batch_nd(input_shape=[1, 2, 2, 1], block_shape=[2, 2], paddings=[[0, 0], [0, 0]])

    _test_space_to_batch_nd(input_shape=[1, 2, 2, 3], block_shape=[2, 2], paddings=[[0, 0], [0, 0]])

    _test_space_to_batch_nd(input_shape=[1, 4, 4, 1], block_shape=[2, 2], paddings=[[0, 0], [0, 0]])

    _test_space_to_batch_nd(input_shape=[2, 2, 4, 1], block_shape=[2, 2], paddings=[[0, 0], [2, 0]])


#######################################################################
# Pooling
# -------
def _test_pooling_iteration(input_shape, **kwargs):
    """One iteration of pool operation with given shapes and attributes"""

    x = -np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) - 1

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=input_shape, dtype="float32")
        out = nn_ops.pool(in_data, **kwargs)

        compare_tflite_with_tvm(x, "Placeholder:0", [in_data], [out])


def _test_pooling(input_shape, **kwargs):
    _test_pooling_iteration(input_shape, **kwargs)


def test_forward_pooling():
    """Pooling"""

    for pool_type in ["AVG", "MAX"]:
        _test_pooling(
            input_shape=[2, 9, 10, 2],
            window_shape=[1, 1],
            padding="SAME",
            pooling_type=pool_type,
            dilation_rate=[1, 1],
            strides=[1, 1],
        )

        _test_pooling(
            input_shape=[2, 10, 9, 2],
            window_shape=[1, 1],
            padding="SAME",
            pooling_type=pool_type,
            dilation_rate=[1, 1],
            strides=[1, 1],
        )

        _test_pooling(
            input_shape=[2, 9, 10, 2],
            window_shape=[2, 1],
            padding="SAME",
            pooling_type=pool_type,
            dilation_rate=[1, 1],
            strides=[1, 1],
        )

        _test_pooling(
            input_shape=[2, 10, 9, 2],
            window_shape=[2, 3],
            padding="SAME",
            pooling_type=pool_type,
            dilation_rate=[1, 1],
            strides=[2, 1],
        )


def _test_l2_pool2d(input_shape, ksize, strides, padding, data_format, fused_func_name=None):
    x = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) - 1

    with tf.Graph().as_default():
        in_data = tf.placeholder(dtype=tf.float32, name="input", shape=input_shape)
        out = tf.sqrt(
            tf.nn.avg_pool(
                tf.square(in_data),
                ksize=ksize,
                strides=strides,
                padding=padding,
                data_format=data_format,
            )
        )
        out = with_fused_activation_function(out, fused_func_name)

        compare_tflite_with_tvm(x, "input", [in_data], [out])


def test_forward_l2_pool2d():
    _test_l2_pool2d([1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], "SAME", "NHWC", "RELU6")
    _test_l2_pool2d([2, 9, 10, 2], [1, 1, 1, 1], [1, 1, 1, 1], "SAME", "NHWC", "RELU6")
    _test_l2_pool2d([2, 9, 10, 2], [1, 2, 1, 1], [1, 1, 1, 1], "SAME", "NHWC")
    _test_l2_pool2d([2, 9, 10, 2], [1, 2, 1, 1], [1, 1, 2, 1], "SAME", "NHWC")
    _test_l2_pool2d([1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], "VALID", "NHWC", "RELU")
    _test_l2_pool2d([2, 9, 10, 2], [1, 1, 1, 1], [1, 1, 1, 1], "VALID", "NHWC")
    _test_l2_pool2d([2, 9, 10, 2], [1, 2, 1, 1], [1, 1, 1, 1], "VALID", "NHWC")
    _test_l2_pool2d([2, 9, 10, 2], [1, 2, 1, 1], [1, 1, 2, 1], "VALID", "NHWC", "RELU6")


#######################################################################
# Convolution
# -----------


def _test_tflite2_quantized_convolution(
    input_shape,
    kernel_shape,
    filters,
    padding="valid",
    data_format=None,
    int_quant_dtype=tf.int8,
    groups=1,
):
    """One iteration of TFLite2 quantized convolution with given shapes and attributes"""
    data_format = "channels_last" if data_format == "NHWC" else "channels_first"
    data = np.random.uniform(0, 1, input_shape).astype("float32")
    _ = np.random.uniform(0, 1, kernel_shape).astype("float32")

    data_in = tf.keras.layers.Input(shape=data.shape[1:])
    conv = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=(kernel_shape[0], kernel_shape[1]),
        activation=tf.nn.relu,
        padding=padding,
        data_format=data_format,
        groups=groups,
    )(data_in)
    keras_model = tf.keras.models.Model(data_in, conv)

    # To create quantized values with dynamic range of activations, needs representative dataset
    def representative_data_gen():
        for _ in range(1):
            yield [data]

    tflite_model_quant = _quantize_keras_model(
        keras_model,
        representative_data_gen,
        is_float_input=True,
        is_float_output=True,
        int_quant_dtype=int_quant_dtype,
    )
    # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
    try:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_quant, 0)
    except AttributeError:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_quant, 0)
    except ImportError as exc:
        raise ImportError("The tflite package must be installed") from exc

    subgraph = tflite_model.Subgraphs(0)
    model_input = subgraph.InputsAsNumpy()
    input_node = subgraph.Tensors(model_input).Name().decode("utf-8")

    tflite_output = run_tflite_graph(tflite_model_quant, data)
    if tf.__version__ < LooseVersion("2.9"):
        input_node = data_in.name.replace(":0", "")
    else:
        input_node = "serving_default_" + data_in.name + ":0"
    tvm_output = run_tvm_graph(tflite_model_quant, data, input_node)
    tvm.testing.assert_allclose(
        np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-2, atol=1e-2
    )


def test_forward_quantized_convolution():
    """Quantized convolution"""
    for int_quant_dtype in [tf.int8, tf.int16]:
        _test_tflite2_quantized_convolution(
            (1, 28, 28, 1),
            (1, 1),
            12,
            data_format="NHWC",
            int_quant_dtype=int_quant_dtype,
        )

        _test_tflite2_quantized_convolution(
            (1, 1, 28, 28),
            (1, 1),
            12,
            data_format="NCWH",
            int_quant_dtype=int_quant_dtype,
        )

        _test_tflite2_quantized_convolution(
            (64, 2, 28, 28),
            (1, 1),
            12,
            data_format="NCWH",
            int_quant_dtype=int_quant_dtype,
        )

        _test_tflite2_quantized_convolution(
            (1, 16, 10, 10),
            (3, 3),
            2,
            data_format="NCWH",
            int_quant_dtype=int_quant_dtype,
            groups=2,
        )

        _test_tflite2_quantized_convolution(
            (2, 32, 28, 28),
            (1, 1),
            16,
            data_format="NCWH",
            int_quant_dtype=int_quant_dtype,
            groups=8,
        )


def test_forward_quantized_depthwise_convolution():
    for int_quant_dtype in [tf.int8, tf.int16]:
        _test_tflite2_quantized_depthwise_convolution(
            [1, 8, 8, 128], [1, 1, 128, 1], [1, 1], [1, 1], "SAME", "NHWC", 1, int_quant_dtype
        )
        _test_tflite2_quantized_depthwise_convolution(
            [1, 17, 17, 12], [3, 3, 12, 1], [1, 1], [2, 2], "VALID", "NHWC", 1, int_quant_dtype
        )
        _test_tflite2_quantized_depthwise_convolution(
            [1, 24, 24, 3], [7, 7, 3, 8], [1, 1], [2, 2], "SAME", "NHWC", 8, int_quant_dtype
        )


def _test_tflite2_quantized_depthwise_convolution(
    input_shape,
    kernel_shape,
    dilations,
    strides,
    padding,
    data_format,
    depth_multiplier,
    int_quant_dtype=tf.int8,
):
    """One iteration of TFLite2 quantized depthwise convolution with given shapes and attributes"""

    data_format = "channels_last" if data_format == "NHWC" else "channels_first"
    data = np.random.uniform(0, 1, input_shape).astype("float32")
    kernel = np.random.uniform(0, 1, kernel_shape).astype("float32")

    data_in = tf.keras.layers.Input(shape=data.shape[1:])
    conv = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(kernel_shape[0], kernel_shape[1]),
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation="relu",
        use_bias=False,
        depth_multiplier=depth_multiplier,
    )(data_in)
    keras_model = tf.keras.models.Model(data_in, conv)
    keras_model.layers[1].set_weights([kernel])

    # To create quantized values with dynamic range of activations, needs representative dataset
    def representative_data_gen():
        for _ in range(1):
            yield [data]

    tflite_model_quant = _quantize_keras_model(
        keras_model,
        representative_data_gen,
        is_float_input=True,
        is_float_output=True,
        int_quant_dtype=int_quant_dtype,
    )

    # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
    try:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_quant, 0)
    except AttributeError:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_quant, 0)
    except ImportError as exc:
        raise ImportError("The tflite package must be installed") from exc

    subgraph = tflite_model.Subgraphs(0)
    model_input = subgraph.InputsAsNumpy()
    input_node = subgraph.Tensors(model_input).Name().decode("utf-8")

    tflite_output = run_tflite_graph(tflite_model_quant, data)
    tvm_output = run_tvm_graph(tflite_model_quant, data, input_node)
    tvm.testing.assert_allclose(
        np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-2, atol=1e-2
    )


def _test_convolution(
    tensor_in_sizes,
    filter_in_sizes,
    dilations,
    strides,
    padding,
    data_format,
    is_depthwise=False,
    quantized=False,
    fp16_quantized=False,
):
    """One iteration of convolution with given shapes and attributes"""

    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
        total_size_1 *= s
    for s in filter_in_sizes:
        total_size_2 *= s
    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    if quantized:
        data_array = np.random.uniform(0, 255, tensor_in_sizes).astype("uint8")
        filter_array = np.random.uniform(0, 255, filter_in_sizes).astype("uint8")
    else:
        data_array = [f * 1.0 for f in range(1, total_size_1 + 1)]
        filter_array = [f * 1.0 for f in range(1, total_size_2 + 1)]

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=tensor_in_sizes, dtype="float32", name="in_data")
        in_filter = constant_op.constant(
            filter_array, shape=filter_in_sizes, dtype="float32", name="in_filter"
        )
        strides = [1] + strides + [1]
        dilations = [1] + dilations + [1]

        if is_depthwise:
            out = nn_ops.depthwise_conv2d_native(
                in_data, in_filter, strides=strides, padding=padding, data_format=data_format
            )
        else:
            out = nn_ops.conv2d(
                in_data, in_filter, strides=strides, padding=padding, data_format=data_format
            )

        if quantized and not fp16_quantized:
            if is_depthwise:
                # Quantized the inputs and feed them to the convolution
                inq_data = tf.quantization.fake_quant_with_min_max_args(
                    in_data, min=-100, max=100, name="inq_data"
                )
                inq_filter = tf.quantization.fake_quant_with_min_max_args(
                    in_filter, min=-100, max=100, name="inq_filter"
                )
                out = nn_ops.depthwise_conv2d_native(
                    inq_data, inq_filter, strides=strides, padding=padding, data_format=data_format
                )
                out = tf.quantization.fake_quant_with_min_max_args(
                    out, min=-200, max=200, name="out"
                )

                # Set the input quantization range
                input_range = {"in_data": (-100, 100)} if quantized else None

                # Compare
                compare_tflite_with_tvm(
                    data_array,
                    "in_data",
                    [in_data],
                    [out],
                    quantized=quantized,
                    input_range=input_range,
                    experimental_new_converter=True,
                )
            else:
                # Quantized the inputs and feed them to the convolution
                inq_data = tf.quantization.fake_quant_with_min_max_args(
                    in_data, min=-100, max=100, name="inq_data"
                )
                inq_filter = tf.quantization.fake_quant_with_min_max_args(
                    in_filter, min=-100, max=100, name="inq_filter"
                )
                out = nn_ops.conv2d(
                    inq_data, inq_filter, strides=strides, padding=padding, data_format=data_format
                )
                out = tf.quantization.fake_quant_with_min_max_args(
                    out, min=-200, max=200, name="out"
                )

                # Set the input quantization range
                input_range = {"in_data": (-100, 100)} if quantized else None

                # Compare
                compare_tflite_with_tvm(
                    data_array,
                    "in_data",
                    [in_data],
                    [out],
                    quantized=quantized,
                    input_range=input_range,
                    experimental_new_converter=True,
                )
        else:
            data_array = np.reshape(data_array, tensor_in_sizes).astype("float32")
            compare_tflite_with_tvm(data_array, "in_data", [in_data], [out])


def test_forward_convolution():
    """Convolution"""
    for quantized in [False, True]:
        for fp16_quantized in [False, True]:
            _test_convolution(
                [4, 8, 8, 176],
                [1, 1, 176, 32],
                [1, 1],
                [1, 1],
                "SAME",
                "NHWC",
                quantized=quantized,
                fp16_quantized=fp16_quantized,
            )
            _test_convolution(
                [4, 17, 17, 19],
                [3, 3, 19, 19],
                [1, 1],
                [2, 2],
                "VALID",
                "NHWC",
                quantized=quantized,
                fp16_quantized=fp16_quantized,
            )
            _test_convolution(
                [4, 17, 17, 124],
                [1, 1, 124, 19],
                [1, 1],
                [1, 1],
                "SAME",
                "NHWC",
                quantized=quantized,
                fp16_quantized=fp16_quantized,
            )
            _test_convolution(
                [4, 17, 17, 12],
                [3, 3, 12, 32],
                [1, 1],
                [2, 2],
                "VALID",
                "NHWC",
                quantized=quantized,
                fp16_quantized=fp16_quantized,
            )

            # depthwise convolution
            _test_convolution(
                [4, 8, 8, 176],
                [1, 1, 176, 1],
                [1, 1],
                [1, 1],
                "SAME",
                "NHWC",
                True,
                quantized=quantized,
                fp16_quantized=fp16_quantized,
            )
            _test_convolution(
                [4, 17, 17, 19],
                [3, 3, 19, 1],
                [1, 1],
                [2, 2],
                "VALID",
                "NHWC",
                True,
                quantized=quantized,
                fp16_quantized=fp16_quantized,
            )
            _test_convolution(
                [4, 17, 17, 124],
                [1, 1, 124, 1],
                [1, 1],
                [1, 1],
                "SAME",
                "NHWC",
                True,
                quantized=quantized,
                fp16_quantized=fp16_quantized,
            )
            _test_convolution(
                [4, 17, 17, 12],
                [3, 3, 12, 1],
                [1, 1],
                [2, 2],
                "VALID",
                "NHWC",
                True,
                quantized=quantized,
                fp16_quantized=fp16_quantized,
            )
            _test_convolution(
                [4, 17, 17, 12],
                [3, 3, 12, 2],
                [1, 1],
                [2, 2],
                "VALID",
                "NHWC",
                True,
                quantized=quantized,
                fp16_quantized=fp16_quantized,
            )
            # depthwise convolution with single input channel
            _test_convolution(
                [1, 76, 64, 1],
                [9, 5, 1, 96],
                [1, 1],
                [1, 1],
                "SAME",
                "NHWC",
                True,
                quantized=quantized,
                fp16_quantized=fp16_quantized,
            )

    # TFLite2 quantized convolution testing
    if package_version.parse(tf.VERSION) >= package_version.parse("2.3.0"):
        _test_convolution(
            [1, 8, 8, 176], [1, 1, 176, 32], [1, 1], [1, 1], "SAME", "NHWC", quantized=True
        )
        _test_convolution(
            [1, 17, 17, 12], [3, 3, 12, 32], [1, 1], [2, 2], "VALID", "NHWC", quantized=True
        )
        _test_convolution(
            [1, 17, 17, 19], [3, 3, 19, 19], [1, 1], [2, 2], "VALID", "NHWC", quantized=True
        )
        _test_convolution(
            [1, 17, 17, 124], [1, 1, 124, 19], [1, 1], [1, 1], "SAME", "NHWC", quantized=True
        )


#######################################################################
# Transpose Convolution
# ---------------------


def _test_transpose_conv(
    tensor_in_sizes,
    filter_in_sizes,
    output_shape,
    strides,
    padding,
    quantized=False,
    fp16_quantized=False,
):
    """One iteration of transpose convolution with given shapes and attributes"""

    total_size_1 = 1
    total_size_2 = 1
    for s in tensor_in_sizes:
        total_size_1 *= s
    for s in filter_in_sizes:
        total_size_2 *= s

    with tf.Graph().as_default():
        if quantized and not fp16_quantized:
            # Initializes the input tensor with array containing incrementing
            # numbers from 1.
            data_array = [max(f, 255) for f in range(1, total_size_1 + 1)]
            filter_array = [max(f, 255) for f in range(1, total_size_2 + 1)]
            data_array = np.reshape(data_array, tensor_in_sizes).astype("uint8")
            filter_array = np.reshape(filter_array, filter_in_sizes).astype("uint8")

            in_data = array_ops.placeholder(shape=tensor_in_sizes, dtype="float32", name="in_data")
            inq_data = tf.quantization.fake_quant_with_min_max_args(
                in_data, min=-100, max=100, name="q_data"
            )
            input_range = {"q_data": (-100, 100)}

            in_filter = constant_op.constant(
                filter_array, shape=filter_in_sizes, dtype="float32", name="in_filter"
            )
            inq_filter = tf.quantization.fake_quant_with_min_max_args(
                in_filter, min=-100, max=100, name="q_filter"
            )

            strides = [1] + strides + [1]

            out = nn_ops.conv2d_transpose(
                inq_data, inq_filter, output_shape=output_shape, strides=strides, padding=padding
            )
            out = tf.quantization.fake_quant_with_min_max_args(out, min=-100, max=100, name="out")
            compare_tflite_with_tvm(
                [data_array], ["q_data"], [inq_data], [out], quantized=True, input_range=input_range
            )
        else:
            # Initializes the input tensor with array containing incrementing
            # numbers from 1.
            data_array = [f * 1.0 for f in range(1, total_size_1 + 1)]
            filter_array = [f * 1.0 for f in range(1, total_size_2 + 1)]

            in_data = array_ops.placeholder(shape=tensor_in_sizes, dtype="float32", name="in_data")
            in_filter = constant_op.constant(
                filter_array, shape=filter_in_sizes, dtype="float32", name="in_filter"
            )
            strides = [1] + strides + [1]
            # in_filter layout is HWOI
            out = nn_ops.conv2d_transpose(
                in_data, in_filter, output_shape=output_shape, strides=strides, padding=padding
            )
            data_array = np.reshape(data_array, tensor_in_sizes).astype("float32")
            compare_tflite_with_tvm(
                [data_array], ["in_data"], [in_data], [out], fp16_quantized=fp16_quantized
            )


def test_forward_transpose_conv():
    """Transpose convolution"""
    for quantized in [True, False]:
        for fp16_quantized in [True, False]:
            # odd size input, padding VALID
            _test_transpose_conv(
                [1, 5, 6, 16],
                [2, 2, 16, 16],
                [1, 10, 12, 16],
                [2, 2],
                "VALID",
                quantized,
                fp16_quantized,
            )
            # odd size input, padding SAME
            _test_transpose_conv(
                [1, 5, 6, 16],
                [2, 2, 16, 16],
                [1, 10, 12, 16],
                [2, 2],
                "SAME",
                quantized,
                fp16_quantized,
            )
            # kernel 3x3, padding VALID
            _test_transpose_conv(
                [4, 32, 32, 16],
                [3, 3, 5, 16],
                [4, 34, 34, 5],
                [1, 1],
                "VALID",
                quantized,
                fp16_quantized,
            )
            _test_transpose_conv(
                [1, 32, 32, 16],
                [3, 3, 5, 16],
                [1, 65, 65, 5],
                [2, 2],
                "VALID",
                quantized,
                fp16_quantized,
            )
            _test_transpose_conv(
                [1, 32, 32, 16],
                [3, 3, 5, 16],
                [1, 65, 34, 5],
                [2, 1],
                "VALID",
                quantized,
                fp16_quantized,
            )

            # kernel 3x3, padding SAME
            _test_transpose_conv(
                [4, 32, 32, 16],
                [3, 3, 5, 16],
                [4, 32, 32, 5],
                [1, 1],
                "SAME",
                quantized,
                fp16_quantized,
            )
            _test_transpose_conv(
                [1, 32, 32, 16],
                [3, 3, 5, 16],
                [1, 64, 64, 5],
                [2, 2],
                "SAME",
                quantized,
                fp16_quantized,
            )
            _test_transpose_conv(
                [1, 32, 32, 16],
                [3, 3, 5, 16],
                [1, 64, 32, 5],
                [2, 1],
                "SAME",
                quantized,
                fp16_quantized,
            )

            # kernel 2x2, padding VALID
            _test_transpose_conv(
                [4, 32, 32, 16],
                [2, 2, 5, 16],
                [4, 33, 33, 5],
                [1, 1],
                "VALID",
                quantized,
                fp16_quantized,
            )
            _test_transpose_conv(
                [1, 32, 32, 16],
                [2, 2, 5, 16],
                [1, 64, 64, 5],
                [2, 2],
                "VALID",
                quantized,
                fp16_quantized,
            )
            _test_transpose_conv(
                [1, 32, 32, 16],
                [2, 2, 5, 16],
                [1, 64, 33, 5],
                [2, 1],
                "VALID",
                quantized,
                fp16_quantized,
            )

            # kernel 2x2, padding SAME
            _test_transpose_conv(
                [4, 32, 32, 16],
                [2, 2, 5, 16],
                [4, 32, 32, 5],
                [1, 1],
                "SAME",
                quantized,
                fp16_quantized,
            )
            _test_transpose_conv(
                [1, 32, 32, 16],
                [2, 2, 5, 16],
                [1, 64, 64, 5],
                [2, 2],
                "SAME",
                quantized,
                fp16_quantized,
            )
            _test_transpose_conv(
                [1, 32, 32, 16],
                [2, 2, 5, 16],
                [1, 64, 32, 5],
                [2, 1],
                "SAME",
                quantized,
                fp16_quantized,
            )

            # kernel 1x1, padding VALID
            _test_transpose_conv(
                [4, 32, 32, 16],
                [1, 1, 5, 16],
                [4, 32, 32, 5],
                [1, 1],
                "VALID",
                quantized,
                fp16_quantized,
            )
            _test_transpose_conv(
                [1, 32, 32, 16],
                [1, 1, 5, 16],
                [1, 63, 63, 5],
                [2, 2],
                "VALID",
                quantized,
                fp16_quantized,
            )
            _test_transpose_conv(
                [1, 32, 32, 16],
                [1, 1, 5, 16],
                [1, 63, 32, 5],
                [2, 1],
                "VALID",
                quantized,
                fp16_quantized,
            )

            # kernel 1x1, padding SAME
            _test_transpose_conv(
                [4, 32, 32, 16],
                [1, 1, 5, 16],
                [4, 32, 32, 5],
                [1, 1],
                "SAME",
                quantized,
                fp16_quantized,
            )
            _test_transpose_conv(
                [1, 32, 32, 16],
                [1, 1, 5, 16],
                [1, 63, 63, 5],
                [2, 2],
                "SAME",
                quantized,
                fp16_quantized,
            )
            _test_transpose_conv(
                [1, 32, 32, 16],
                [1, 1, 5, 16],
                [1, 63, 32, 5],
                [2, 1],
                "SAME",
                quantized,
                fp16_quantized,
            )


def _test_tflite2_quantized_transpose_conv(
    input_shape,
    kernel_shape,
    filters,
    padding="valid",
    strides=(1, 1),
    data_format=None,
    int_quant_dtype=tf.int8,
):
    """One iteration of TFLite2 quantized tranpose conv with given shapes and attributes"""
    data_format = "channels_last" if data_format == "NHWC" else "channels_first"
    data = np.random.uniform(0, 1, input_shape).astype("float32")
    _ = np.random.uniform(0, 1, kernel_shape).astype("float32")

    data_in = tf.keras.layers.Input(shape=data.shape[1:], batch_size=1)
    transpose_conv = tf.keras.layers.Conv2DTranspose(
        filters=filters,
        kernel_size=(kernel_shape[0], kernel_shape[1]),
        padding=padding,
        strides=strides,
        use_bias=True,
    )(data_in)
    keras_model = tf.keras.models.Model(data_in, transpose_conv)

    # To create quantized values with dynamic range of activations, needs representative dataset
    def representative_data_gen():
        for _ in range(1):
            yield [data]

    tflite_model_quant = _quantize_keras_model(
        keras_model,
        representative_data_gen,
        is_float_input=True,
        is_float_output=True,
        int_quant_dtype=int_quant_dtype,
    )

    # TFLite.Model.Model has changed to TFLite.Model from 1.14 to 2.1
    try:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_quant, 0)
    except AttributeError:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_quant, 0)
    except ImportError as exc:
        raise ImportError("The tflite package must be installed") from exc

    subgraph = tflite_model.Subgraphs(0)
    model_input = subgraph.InputsAsNumpy()
    input_node = subgraph.Tensors(model_input).Name().decode("utf-8")

    tflite_output = run_tflite_graph(tflite_model_quant, data)

    if tf.__version__ < LooseVersion("2.9"):
        input_node = data_in.name.replace(":0", "")
    else:
        input_node = "serving_default_" + data_in.name + ":0"

    tvm_output = run_tvm_graph(tflite_model_quant, data, input_node)
    tvm.testing.assert_allclose(
        np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-2, atol=1e-2
    )


def test_forward_quantized_transpose_conv():
    """Quantized convolution"""
    for int_quant_dtype in [tf.int8, tf.int16]:
        _test_tflite2_quantized_transpose_conv(
            (1, 1, 5, 64),
            (3, 3),
            64,
            padding="same",
            strides=(1, 2),
            data_format="NHWC",
            int_quant_dtype=int_quant_dtype,
        )


#######################################################################
# Reshape
# -------


def _test_reshape(data, out_shape, wrap_shape, quantized=False):
    """One iteration of reshape operation with given data and out shape"""
    if quantized:
        with tf.Graph().as_default():
            in_data = array_ops.placeholder(shape=data.shape, dtype="float32", name="in")
            inq_data = tf.quantization.fake_quant_with_min_max_args(
                in_data, min=-100, max=100, name="inq_0"
            )

            input_range = {"inq_0": (-100, 100)}
            out_shape = out_shape if not wrap_shape else np.array(out_shape, dtype=np.int32)

            in_shape = (
                out_shape
                if not wrap_shape
                else array_ops.placeholder(
                    shape=out_shape.shape, dtype=out_shape.dtype, name="Newshape"
                )
            )

            out = array_ops.reshape(inq_data, in_shape)
            out = tf.quantization.fake_quant_with_min_max_args(out, min=-200, max=200, name="out")
            compare_tflite_with_tvm(
                [data, out_shape] if wrap_shape else [data],
                ["inq_0:0", "Newshape:0"] if wrap_shape else ["inq_0:0"],
                [inq_data, in_shape] if wrap_shape else [inq_data],
                [out],
                quantized=True,
                input_range=input_range,
                mode="vm",
            )
    else:
        # Test with tensor and constant
        with tf.Graph().as_default():
            in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)

            out_shape = out_shape if not wrap_shape else np.array(out_shape, dtype=np.int32)

            in_shape = (
                out_shape
                if not wrap_shape
                else array_ops.placeholder(
                    shape=out_shape.shape, dtype=out_shape.dtype, name="Newshape"
                )
            )

            out = array_ops.reshape(in_data, in_shape)

            compare_tflite_with_tvm(
                [data, out_shape] if wrap_shape else [data],
                ["Placeholder:0", "Newshape:0"] if wrap_shape else ["Placeholder:0"],
                [in_data, in_shape] if wrap_shape else [in_data],
                [out],
                mode="vm",
            )


def test_forward_reshape():
    for wrap in [True, False]:
        _test_reshape(np.arange(6.0, dtype=np.float32), [2, 3], wrap)
        _test_reshape(np.arange(6), [-1, 2], wrap)
        _test_reshape(np.arange(6), [3, -1], wrap)
        _test_reshape(np.arange(6), [-1], wrap)

    _test_reshape(np.arange(6, dtype=np.uint8), [2, 3], False, True)
    _test_reshape(np.arange(6, dtype=np.uint8), [-1, 2], False, True)


#######################################################################
# Resize
# ------


def _test_resize(
    tf_resize_op, images_data, size_data, align_corners, half_pixel_centers, quantized=False
):
    """One iteration of Resize"""
    # Test with tensor and constant
    with tf.Graph().as_default():
        images_tensor = array_ops.placeholder(shape=images_data.shape, dtype="float32", name="in")
        size = ops.convert_to_tensor(size_data, dtype=size_data.dtype)

        if quantized:
            images_tensor_q = tf.quantization.fake_quant_with_min_max_args(
                images_tensor, min=-3, max=2, name="in"
            )
            input_range = {"in": (-3, 2)}
            out_tensor = tf_resize_op(
                images=images_tensor_q,
                size=size,
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers,
            )
            out_tensor = tf.quantization.fake_quant_with_min_max_args(
                out_tensor, min=-3, max=2, name="out_tensor"
            )

            compare_tflite_with_tvm(
                [images_data],
                ["in:0"],
                [images_tensor],
                [out_tensor],
                quantized=True,
                input_range=input_range,
            )
        else:
            out_tensor = tf_resize_op(
                images=images_tensor,
                size=size,
                align_corners=align_corners,
                half_pixel_centers=half_pixel_centers,
            )
            compare_tflite_with_tvm([images_data], ["in:0"], [images_tensor], [out_tensor])


def test_all_resize():
    """Resize"""
    images_data = np.random.uniform(0, 255, (1, 16, 16, 3))
    images_data_float32 = images_data.astype(np.float32)
    images_data_uint8 = images_data.astype(np.uint8)
    size_data = np.array([8, 8]).astype("int32")
    ### RESIZE_BILINEAR
    _test_resize(
        tf.image.resize_bilinear,
        images_data_float32,
        size_data,
        align_corners=False,
        half_pixel_centers=False,
        quantized=False,
    )
    _test_resize(
        tf.image.resize_bilinear,
        images_data_float32,
        size_data,
        align_corners=True,
        half_pixel_centers=False,
        quantized=False,
    )
    _test_resize(
        tf.image.resize_bilinear,
        images_data_uint8,
        size_data,
        align_corners=False,
        half_pixel_centers=False,
        quantized=True,
    )
    _test_resize(
        tf.image.resize_bilinear,
        images_data_uint8,
        size_data,
        align_corners=True,
        half_pixel_centers=False,
        quantized=True,
    )
    _test_resize(
        tf.image.resize_bilinear,
        images_data_uint8,
        size_data,
        align_corners=False,
        half_pixel_centers=True,
        quantized=True,
    )
    ### RESIZE_NEAREST_NEIGHBOR (was added in v1.13)
    # According to topi resize.h
    # Align corners not supported for nearest neighbour

    if "RESIZE_NEAREST_NEIGHBOR" in dir(BuiltinOperator()):
        _test_resize(
            tf.image.resize_nearest_neighbor,
            images_data_float32,
            size_data,
            align_corners=False,
            half_pixel_centers=False,
        )
        _test_resize(
            tf.image.resize_nearest_neighbor,
            images_data_float32,
            size_data,
            align_corners=True,
            half_pixel_centers=False,
        )
        _test_resize(
            tf.image.resize_nearest_neighbor,
            images_data_float32,
            size_data,
            align_corners=False,
            half_pixel_centers=True,
        )


#######################################################################
# Range
# -----
def _test_range(start, limit, delta):
    # tflite 1.13 convert method does not accept empty shapes
    if package_version.parse(tf.VERSION) >= package_version.parse("1.14.0"):
        tf.reset_default_graph()
        with tf.Graph().as_default():
            start_scalar, limit_scalar, delta_scalar = (
                tf.placeholder(dtype=start.dtype, shape=(), name="start"),
                tf.placeholder(dtype=limit.dtype, shape=(), name="limit"),
                tf.placeholder(dtype=delta.dtype, shape=(), name="delta"),
            )

            out = tf.range(start_scalar, limit_scalar, delta_scalar, name="range")

            compare_tflite_with_tvm(
                [start, limit, delta],
                ["start", "limit", "delta"],
                [start_scalar, limit_scalar, delta_scalar],
                [out],
                mode="vm",
                quantized=False,
            )


def _test_range_default():
    # tflite 1.13 convert method does not accept empty shapes
    if package_version.parse(tf.VERSION) >= package_version.parse("1.14.0"):
        tf.reset_default_graph()
        with tf.Graph().as_default():
            inputs = [
                tf.placeholder(dtype=tf.int32, shape=(), name="p1"),
                tf.placeholder(dtype=tf.int32, shape=(), name="p2"),
            ]
            outputs = [
                tf.range(start=inputs[0], limit=inputs[1]),  # use default delta
                tf.range(
                    start=inputs[1]
                ),  # use start as limit with 0 as the first item in the range
            ]

            compare_tflite_with_tvm(
                [np.int32(1), np.int32(18)], ["p1", "p2"], inputs, outputs, mode="vm"
            )


def test_forward_range():
    _test_range(np.int32(1), np.int32(18), np.int32(3))
    _test_range(np.int32(1), np.int32(18), np.float32(3.1))  # increment is of type float
    _test_range(np.float32(1.0), np.int32(18), np.int32(3.1))  # start is of type float
    _test_range_default()


#######################################################################
# Shape
# -----


def _test_shape(dtype):
    # tflite 1.13 convert method does not accept empty shapes
    if package_version.parse(tf.VERSION) >= package_version.parse("1.14.0"):
        tf.reset_default_graph()
        with tf.Graph().as_default():
            data = np.array([1, 18, 3], dtype=np.int32)
            start = tf.placeholder(dtype=tf.int32, shape=[], name="start")
            limit = tf.placeholder(dtype=tf.int32, shape=[], name="limit")
            delta = tf.placeholder(dtype=tf.int32, shape=[], name="delta")
            tf_range = tf.range(start, limit, delta, tf.int32, name="range")
            out = tf.shape(tf_range, out_type=dtype)
            out = tf.add(out, tf.constant([1], dtype=dtype))
            compare_tflite_with_tvm(
                list(np.nditer(data)),
                ["start", "limit", "delta"],
                [start, limit, delta],
                [out],
                mode="vm",
            )


def test_forward_shape():
    _test_shape(tf.int32)
    _test_shape(tf.int64)


#######################################################################
# Concatenation
# -------------


def _test_concatenation(data, axis):
    """One iteration of concatenation"""

    assert len(data) >= 1

    with tf.Graph().as_default():
        in_data = [
            array_ops.placeholder(shape=tensor.shape, dtype=tensor.dtype, name=f"in_{idx}")
            for idx, tensor in enumerate(data)
        ]
        out = array_ops.concat(in_data, axis)
        name = [f"in_{idx}:0" for idx in range(len(data))]

        compare_tflite_with_tvm(data, name, in_data, [out])


def test_forward_concatenation():

    _test_concatenation([np.arange(6).reshape((1, 2, 1, 3)), np.arange(6).reshape((1, 2, 1, 3))], 1)

    _test_concatenation([np.arange(6).reshape((3, 2)), np.arange(6).reshape((3, 2))], 1)

    _test_concatenation(
        [
            np.arange(6).reshape((2, 1, 1, 3)),
            np.arange(6).reshape((2, 1, 1, 3)),
            np.arange(6).reshape((2, 1, 1, 3)),
        ],
        1,
    )


#######################################################################
# Unary elemwise
# --------------


def _test_unary_elemwise(math_op, data, quantized, quant_range=(-6, 6), int_quant_dtype=tf.int8):
    """One iteration of unary elemwise"""
    if quantized:
        with tf.Graph().as_default():
            quant_min, quant_max = quant_range
            in_data = array_ops.placeholder(shape=data.shape, dtype="float32", name="in_0")
            inq_data = tf.quantization.fake_quant_with_min_max_args(
                in_data, min=quant_min, max=quant_max, name="inq_0"
            )
            input_range = {"inq_0": (quant_min, quant_max)}
            out = math_op(inq_data)
            out = tf.quantization.fake_quant_with_min_max_args(
                out, min=quant_min, max=quant_max, name="out"
            )
            compare_tflite_with_tvm(
                data,
                "inq_0:0",
                [inq_data],
                [out],
                quantized=True,
                input_range=input_range,
                experimental_new_converter=True,
                int_quant_dtype=int_quant_dtype,
            )
    else:
        with tf.Graph().as_default():
            in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype, name="in")
            out = math_op(in_data)
            compare_tflite_with_tvm(data, ["in:0"], [in_data], [out])


def _unary_elewise_create_model(math_op, data, offset=0, int_quant_dtype=tf.int8):
    class Model(tf.Module):
        @tf.function
        def tf_function(self, x):
            op = math_op(x)
            return op

    if int_quant_dtype in (tf.int8, tf.uint8):
        _ = "int8"
    elif int_quant_dtype in (tf.int16, tf.uint16):
        _ = "int16"
    else:
        raise Exception(f"Unsupported dtype '{int_quant_dtype}' for unary elementwise test.")

    model = Model()

    # Save the model
    export_dir = tempfile.gettempdir() + "/tf_model"
    tf.saved_model.save(
        model,
        export_dir,
        signatures=model.tf_function.get_concrete_function(
            tf.TensorSpec(data.shape, tf.float32, name="input")
        ),
    )

    # Convert the model
    def representative_dataset():
        for _ in range(100):
            tmp_data = np.random.rand(*tuple(data.shape))
            yield [tmp_data.astype(np.float32) * 2 - offset]

    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    if int_quant_dtype in (tf.int16, tf.uint16):
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        ]
    else:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = int_quant_dtype
    converter.inference_output_type = int_quant_dtype

    tflite_model = converter.convert()
    return tflite_model


#######################################################################
# Abs
# ----


def _test_abs(data, quantized, int_quant_dtype=tf.int8):
    """One iteration of abs"""
    if quantized:
        tflite_model_quant = _unary_elewise_create_model(
            tf.math.abs, data, offset=1, int_quant_dtype=int_quant_dtype
        )
        tflite_output = run_tflite_graph(tflite_model_quant, data)

        # TFLite 2.6.x upgrade support
        if tf.__version__ < LooseVersion("2.6.1"):
            in_node = ["serving_default_input_int8"]
        elif tf.__version__ < LooseVersion("2.9"):
            in_node = (
                ["serving_default_input_int16"] if int_quant_dtype == tf.int16 else ["tfl.quantize"]
            )
        else:
            in_node = "serving_default_input"

        tvm_output = run_tvm_graph(tflite_model_quant, data, in_node)
        tvm.testing.assert_allclose(
            np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-2
        )
    else:
        return _test_unary_elemwise(math_ops.abs, data, quantized, int_quant_dtype=int_quant_dtype)


#######################################################################
# Rsqrt
# ----


def _test_rsqrt(data, quantized, int_quant_dtype=tf.int8):
    """One iteration of rsqrt"""

    # tensorflow version upgrade support
    if tf.__version__ < LooseVersion("2.6.1") or not quantized:
        return _test_unary_elemwise(
            math_ops.rsqrt, data, quantized, quant_range=[1, 6], int_quant_dtype=int_quant_dtype
        )
    else:
        tflite_model_quant = _unary_elewise_create_model(
            tf.math.rsqrt, data, int_quant_dtype=int_quant_dtype
        )
        tflite_output = run_tflite_graph(tflite_model_quant, data)
        if tf.__version__ < LooseVersion("2.9"):
            in_node = ["tfl.quantize"]
        else:
            in_node = "serving_default_input"
        tvm_output = run_tvm_graph(tflite_model_quant, data, in_node)
        tvm.testing.assert_allclose(
            np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-2
        )


#######################################################################
# Ceil
# ----


def _test_ceil(data, quantized, int_quant_dtype=tf.int8):
    """One iteration of ceil"""
    return _test_unary_elemwise(math_ops.ceil, data, quantized, int_quant_dtype=int_quant_dtype)


#######################################################################
# Floor
# -----


def _test_floor(data, quantized, int_quant_dtype=tf.int8):
    """One iteration of floor"""
    return _test_unary_elemwise(math_ops.floor, data, quantized, int_quant_dtype=int_quant_dtype)


#######################################################################
# Round
# -----


def _test_round(data, quantized, int_quant_dtype=tf.int8):
    """One iteration of round"""
    return _test_unary_elemwise(math_ops.round, data, quantized, int_quant_dtype=int_quant_dtype)


#######################################################################
# Exp
# ---


def _test_exp(data, quantized, int_quant_dtype=tf.int8):
    """One iteration of exp"""
    return _test_unary_elemwise(math_ops.exp, data, quantized, int_quant_dtype=int_quant_dtype)


#######################################################################
# Log
# ---


def _test_log(data, quantized, int_quant_dtype=tf.int8):
    """One iteration of log"""
    return _test_unary_elemwise(
        math_ops.log, data, quantized, quant_range=[1, 6], int_quant_dtype=int_quant_dtype
    )


#######################################################################
# Sin
# ---


def _test_sin(data, quantized, int_quant_dtype=tf.int8):
    """One iteration of sin"""
    return _test_unary_elemwise(math_ops.sin, data, quantized, int_quant_dtype=int_quant_dtype)


#######################################################################
# Cos
# ---


def _test_cos(data, quantized, int_quant_dtype=tf.int8):
    """One iteration of cos"""
    if quantized:
        tflite_model_quant = _unary_elewise_create_model(
            tf.math.cos, data, int_quant_dtype=int_quant_dtype
        )
        tflite_output = run_tflite_graph(tflite_model_quant, data)
        if tf.__version__ < LooseVersion("2.9"):
            in_node = ["tfl.quantize"]
        else:
            in_node = "serving_default_input"
        tvm_output = run_tvm_graph(tflite_model_quant, data, in_node)
        tvm.testing.assert_allclose(
            np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-2
        )
    else:
        return _test_unary_elemwise(math_ops.cos, data, quantized)


#######################################################################
# Tan
# ---


def _test_tan(data, quantized, int_quant_dtype=tf.int8):
    """One iteration of tan"""
    return _test_unary_elemwise(math_ops.tan, data, quantized, int_quant_dtype=int_quant_dtype)


#######################################################################
# Square
# ------


def _test_square(data, quantized, int_quant_dtype=tf.int8):
    """One iteration of square"""
    return _test_unary_elemwise(math_ops.square, data, quantized, int_quant_dtype=int_quant_dtype)


#######################################################################
# Neg
# ------


def _test_neg(data, quantized, int_quant_dtype=tf.int8):
    """One iteration of neg"""
    return _test_unary_elemwise(math_ops.neg, data, quantized, int_quant_dtype=int_quant_dtype)


#######################################################################
# Sqrt
# ------


def _test_sqrt(data, quantized, int_quant_dtype=tf.int8):
    """One iteration of sqrt"""
    return _test_unary_elemwise(
        math_ops.sqrt, data, quantized, quant_range=[1, 6], int_quant_dtype=int_quant_dtype
    )


#######################################################################
# Elu
# ---


def _test_elu(data, quantized, int_quant_dtype=tf.int8):
    """One iteration of elu"""
    return _test_unary_elemwise(nn_ops.elu, data, quantized, int_quant_dtype=int_quant_dtype)


def _test_forward_unary_elemwise(test_op, int_quant_dtype=None, quantized=True, negative=True):
    # input data
    in_data, inq_data = [], []

    np_dtype = int_quant_dtype.as_numpy_dtype if int_quant_dtype else np.uint8

    # quantized input data
    if quantized:
        inq_data.append(np.arange(1, 240, 40, dtype=np_dtype))
        inq_data.append(np.arange(1, 240, 40, dtype=np_dtype).reshape((2, 1, 3)))
        if int_quant_dtype == np.int8:
            inq_data.append(np.arange(-128, 127, 45, dtype=np.int8))

    for data in inq_data:
        test_op(data, quantized=True, int_quant_dtype=int_quant_dtype)

    # normal input data
    if negative:
        in_data.append(np.arange(-2.0, 4.0, dtype=np.float32))
        in_data.append(np.arange(-2.0, 4.0, dtype=np.float32).reshape((2, 1, 3)))
    else:
        in_data.append(np.arange(1.0, 7.0, dtype=np.float32))
        in_data.append(np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 3)))

    for data in in_data:
        test_op(data, quantized=False, int_quant_dtype=int_quant_dtype)


def test_all_unary_elemwise():
    """All unary elemwise"""
    _test_forward_unary_elemwise(_test_abs, int_quant_dtype=tf.int8)
    _test_forward_unary_elemwise(_test_abs, int_quant_dtype=tf.int16)
    _test_forward_unary_elemwise(_test_floor)
    _test_forward_unary_elemwise(_test_exp)
    _test_forward_unary_elemwise(_test_log, negative=False)
    _test_forward_unary_elemwise(_test_square, int_quant_dtype=tf.int8)
    _test_forward_unary_elemwise(_test_sin)
    _test_forward_unary_elemwise(_test_neg)
    _test_forward_unary_elemwise(_test_sqrt, negative=False)
    # tensorflow version upgrade support
    if tf.__version__ < LooseVersion("2.6.1"):
        _test_forward_unary_elemwise(_test_rsqrt, negative=False, int_quant_dtype=tf.uint8)
    else:
        _test_forward_unary_elemwise(_test_rsqrt, negative=False, int_quant_dtype=tf.int8)
    # ceil and cos come with TFLite 1.14.0.post1 fbs schema
    if package_version.parse(tf.VERSION) >= package_version.parse("1.14.0"):
        _test_forward_unary_elemwise(_test_ceil)
        if tf.__version__ < LooseVersion("2.6.1"):
            _test_forward_unary_elemwise(_test_cos, quantized=False)
        else:
            _test_forward_unary_elemwise(_test_cos, int_quant_dtype=tf.int8)
        _test_forward_unary_elemwise(_test_round)
        # This fails with TF and Tflite 1.15.2, this could not have been tested
        # in CI or anywhere else. The failure mode is that we see a backtrace
        # from the converter that we need to provide a custom Tan operator
        # implementation.
        # _test_forward_unary_elemwise(_test_tan)
        _test_forward_unary_elemwise(_test_elu)


#######################################################################
# Element-wise
# ------------


def _test_elemwise(
    math_op,
    data,
    fused_activation_function=None,
    quantized=False,
    qnn_op=None,
    same_qnn_params=False,
    comparison_op=False,
    exclude_zero_point=False,
):
    """One iteration of elemwise"""

    assert len(data) == 2

    def __test_elemwise(in_data):
        assert len(in_data) == 2
        if quantized:
            int_quant_dtype = None
            if data[0].dtype == "int8":
                int_quant_dtype = tf.int8
            elif data[0].dtype == "uint8":
                int_quant_dtype = tf.uint8
            elif data[0].dtype == "int16":
                int_quant_dtype = tf.int16
            else:
                assert False, "Unsupported conversion from numpy to tflite dtype!"

            # set the fp32 output range with respect to the operation
            out_min, out_max = _test_elemwise_qnn_out_range(qnn_op)
            inq0_min, inq0_max = (-100, 100)
            inq1_min, inq1_max = (-50, 50)

            # if requested use same quantization parameters provided by _test_elemwise_qnn_out_range
            if same_qnn_params:
                inq0_min, inq0_max = (out_min, out_max)
                inq1_min, inq1_max = (out_min, out_max)

            if exclude_zero_point:
                if inq1_max == inq1_min:
                    raise ZeroDivisionError("Input range is 0.")

                # only compute for rhs.
                quant_scale = 255 / (inq1_max - inq1_min)
                zero_point = int(round(-inq1_min * quant_scale))
                data[1][data[1] == zero_point] += 1
                data[1][data[1] == 0] += 1

            # fake_quant will keep the tensors in float32 until the conversion in the session
            inq_data = [
                tf.quantization.fake_quant_with_min_max_args(
                    in_data[0], min=out_min, max=out_max, name="inq_0"
                )
                if in_data[0] is not None
                else tf.quantization.fake_quant_with_min_max_args(
                    data[0], min=out_min, max=out_max, name="const_tensor0"
                ),
                tf.quantization.fake_quant_with_min_max_args(
                    in_data[1], min=out_min, max=out_max, name="inq_1"
                )
                if in_data[1] is not None
                else tf.quantization.fake_quant_with_min_max_args(
                    data[1], min=out_min, max=out_max, name="const_tensor1"
                ),
            ]

            input_range = {
                x[1][0]: x[1][1]
                for x in zip(
                    in_data, (("inq_0", (inq0_min, inq0_max)), ("inq_1", (inq1_min, inq1_max)))
                )
                if x[0] is not None
            }

            if comparison_op:
                out = math_op(inq_data[0], inq_data[1])
                out = with_fused_activation_function(out, fused_activation_function)

                compare_tflite_with_tvm(
                    [x[1] for x in zip(in_data, data) if x[0] is not None],
                    [x + ":0" for x in input_range.keys()],
                    [x[1] for x in zip(in_data, inq_data) if x[0] is not None],
                    [out],
                    quantized=True,
                    input_range=input_range,
                    experimental_new_converter=same_qnn_params,
                    int_quant_dtype=int_quant_dtype,
                )
            else:
                out = math_op(inq_data[0], inq_data[1])
                out = with_fused_activation_function(out, fused_activation_function)
                out = tf.quantization.fake_quant_with_min_max_args(
                    out, min=out_min, max=out_max, name="out"
                )

                # Note same_qnn_params uses experimental_new_converter as toco failed
                compare_tflite_with_tvm(
                    [x[1] for x in zip(in_data, data) if x[0] is not None],
                    [x + ":0" for x in input_range.keys()],
                    [x[1] for x in zip(in_data, inq_data) if x[0] is not None],
                    [out],
                    quantized=True,
                    input_range=input_range,
                    experimental_new_converter=same_qnn_params,
                    int_quant_dtype=int_quant_dtype,
                )
        else:
            out = math_op(
                in_data[0]
                if in_data[0] is not None
                else ops.convert_to_tensor(data[0], dtype=data[0].dtype),
                in_data[1]
                if in_data[1] is not None
                else ops.convert_to_tensor(data[1], dtype=data[1].dtype),
            )
            out = with_fused_activation_function(out, fused_activation_function)
            compare_tflite_with_tvm(
                [x[1] for x in zip(in_data, data) if x[0] is not None],
                [x[1] for x in zip(in_data, ("in_0:0", "in_1:0")) if x[0] is not None],
                [x for x in in_data if x is not None],
                [out],
            )

    # Test with two tensors
    with tf.Graph().as_default():
        __test_elemwise(
            in_data=[
                array_ops.placeholder(shape=data[0].shape, dtype="float32", name="in_0"),
                array_ops.placeholder(shape=data[1].shape, dtype="float32", name="in_1"),
            ]
        )
    # Test with tensor and constant
    with tf.Graph().as_default():
        __test_elemwise(
            in_data=[array_ops.placeholder(shape=data[0].shape, dtype="float32", name="in_0"), None]
        )
    # Test with constant and tensor
    with tf.Graph().as_default():
        __test_elemwise(
            in_data=[None, array_ops.placeholder(shape=data[1].shape, dtype="float32", name="in_1")]
        )


#######################################################################
# Add
# ---


def _test_add(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of add"""
    return _test_elemwise(math_ops.add, data, fused_activation_function, quantized, qnn_op)


#######################################################################
# Subtract
# --------


def _test_sub(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of subtract"""
    return _test_elemwise(math_ops.subtract, data, fused_activation_function, quantized, qnn_op)


#######################################################################
# Mul
# ---


def _test_mul(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of mul"""
    return _test_elemwise(math_ops.multiply, data, fused_activation_function, quantized, qnn_op)


#######################################################################
# Divide
# ------


def _test_div(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of divide"""
    return _test_elemwise(
        math_ops.divide,
        data,
        fused_activation_function,
        quantized,
        qnn_op,
        same_qnn_params=True,
        exclude_zero_point=True,
    )


#######################################################################
# Power
# -----


def _test_pow(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of power"""
    return _test_elemwise(
        math_ops.pow,
        data,
        fused_activation_function,
        quantized,
        qnn_op,
        same_qnn_params=True,
    )


#######################################################################
# Maximum
# -------


def _test_maximum(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of maximum"""
    return _test_elemwise(
        math_ops.maximum, data, fused_activation_function, quantized, qnn_op, same_qnn_params=True
    )


#######################################################################
# Minimum
# -------


def _test_minimum(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of minimum"""
    return _test_elemwise(
        math_ops.minimum, data, fused_activation_function, quantized, qnn_op, same_qnn_params=True
    )


#######################################################################
# Greater
# -------


def _test_greater(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of greater"""
    return _test_elemwise(
        math_ops.greater,
        data,
        fused_activation_function,
        quantized,
        qnn_op,
        same_qnn_params=True,
        comparison_op=True,
    )


#######################################################################
# Greater_equal
# -------------


def _test_greater_equal(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of greater_equal"""
    return _test_elemwise(
        math_ops.greater_equal,
        data,
        fused_activation_function,
        quantized,
        qnn_op,
        same_qnn_params=True,
        comparison_op=True,
    )


#######################################################################
# Less
# ----


def _test_less(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of less"""
    return _test_elemwise(
        math_ops.less,
        data,
        fused_activation_function,
        quantized,
        qnn_op,
        same_qnn_params=True,
        comparison_op=True,
    )


#######################################################################
# Less_equal
# ----------


def _test_less_equal(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of less_equal"""
    return _test_elemwise(
        math_ops.less_equal,
        data,
        fused_activation_function,
        quantized,
        qnn_op,
        same_qnn_params=True,
        comparison_op=True,
    )


#######################################################################
# Equal
# -----


def _test_equal(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of equal"""
    return _test_elemwise(
        math_ops.equal,
        data,
        fused_activation_function,
        quantized,
        qnn_op,
        same_qnn_params=True,
        comparison_op=True,
    )


#######################################################################
# Not_equal
# ---------


def _test_not_equal(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of not_equal"""
    return _test_elemwise(
        math_ops.not_equal,
        data,
        fused_activation_function,
        quantized,
        qnn_op,
        same_qnn_params=True,
        comparison_op=True,
    )


#######################################################################
# Squared_difference
# ------------------


def _test_squared_difference(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of squared difference"""
    return _test_elemwise(
        math_ops.squared_difference,
        data,
        fused_activation_function,
        quantized,
        qnn_op,
        same_qnn_params=True,
    )


#######################################################################
# Floor_divide
# ------------


def _test_floor_divide(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of floor_div"""
    return _test_elemwise(
        math_ops.floordiv,
        data,
        fused_activation_function,
        quantized,
        qnn_op,
        same_qnn_params=True,
        exclude_zero_point=True,
    )


#######################################################################
# Floor_mod
# ---------


def _test_floor_mod(data, fused_activation_function=None, quantized=False, qnn_op=None):
    """One iteration of floor_mod"""
    return _test_elemwise(
        math_ops.floormod,
        data,
        fused_activation_function,
        quantized,
        qnn_op,
        same_qnn_params=True,
    )


def _test_forward_elemwise(testop):
    """Elewise"""
    testop(
        [
            np.arange(6.0, dtype=np.float32).reshape((2, 1, 1, 3)),
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 1, 3)),
        ]
    )
    testop(
        [
            np.arange(6.0, dtype=np.float32).reshape((2, 1, 3)),
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 3)),
        ]
    )
    testop(
        [
            np.arange(3.0, dtype=np.float32).reshape((1, 3)),
            np.arange(1.0, 4.0, dtype=np.float32).reshape((1, 3)),
        ]
    )


def _test_forward_elemwise_quantized(testop, dtype=np.uint8):
    type_info = np.iinfo(dtype)
    _min, _max = type_info.min, type_info.max
    testop(
        [
            np.array(np.random.uniform(_min, _max, (3, 6)), dtype=dtype),
            np.array(np.random.uniform(_min, _max, (3, 6)), dtype=dtype),
        ],
        quantized=True,
        qnn_op=testop,
    )


def _test_elemwise_qnn_out_range(qnn_op):
    # set the fake_quant output range with respect to the input tensors float32 range
    qnn_out_range = {
        _test_add: (-150, 150),
        _test_sub: (-150, 150),
        _test_mul: (-5e3, 5e3),
        _test_div: (-150, 150),
        _test_maximum: (-112, 111),
        _test_minimum: (-128, 127),
        _test_equal: (-150, 150),
        _test_greater: (-150, 150),
        _test_squared_difference: (0, 65025),
        _test_floor_divide: (-150, 150),
        _test_less: (-150, 150),
        _test_floor_mod: (-150, 150),
        _test_not_equal: (-150, 150),
        _test_pow: (0, 3),
        _test_less_equal: (-150, 150),
        _test_greater_equal: (-150, 150),
    }

    return qnn_out_range[qnn_op]


def test_all_elemwise():
    """All_elemwise"""
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
    _test_forward_elemwise_quantized(_test_div)
    _test_forward_elemwise(_test_pow)
    _test_forward_elemwise_quantized(_test_pow)
    _test_forward_elemwise(_test_maximum)
    _test_forward_elemwise_quantized(_test_maximum)
    _test_forward_elemwise(_test_minimum)
    _test_forward_elemwise_quantized(_test_minimum)
    _test_forward_elemwise(_test_greater)
    _test_forward_elemwise_quantized(_test_greater)
    _test_forward_elemwise(_test_squared_difference)
    _test_forward_elemwise_quantized(_test_squared_difference, np.int8)
    _test_forward_elemwise(_test_greater_equal)
    _test_forward_elemwise_quantized(_test_greater_equal)
    _test_forward_elemwise(_test_less)
    _test_forward_elemwise_quantized(_test_less)
    _test_forward_elemwise(_test_less_equal)
    _test_forward_elemwise_quantized(_test_less_equal)
    _test_forward_elemwise(_test_equal)
    _test_forward_elemwise_quantized(_test_equal)
    _test_forward_elemwise(_test_not_equal)
    _test_forward_elemwise_quantized(_test_not_equal)
    if package_version.parse(tf.VERSION) >= package_version.parse("1.14.0"):
        _test_forward_elemwise(_test_floor_divide)
        _test_forward_elemwise_quantized(_test_floor_divide)
        _test_forward_elemwise(_test_floor_mod)
        # This test of quantized floor mod is currently disabled due
        # to flaky CI failures in main, failing approximately 45% of
        # the time.
        #
        # _test_forward_elemwise_quantized(_test_floor_mod)


#######################################################################
# AddN
# ----


def _test_forward_add_n(inputs):
    tf.reset_default_graph()
    with tf.Graph().as_default():
        temp = []
        for each in inputs:
            temp.append(tf.placeholder(shape=each.shape, dtype=each.dtype))
        output = tf.add_n(temp)
        compare_tflite_with_tvm(
            list(inputs),
            [each.name for each in temp],
            list(temp),
            [output],
        )


def test_forward_add_n():
    """Add n"""
    if package_version.parse(tf.VERSION) >= package_version.parse("1.14.0"):
        x = np.random.randint(1, 100, size=(3, 3, 3), dtype=np.int32)
        y = np.random.randint(1, 100, size=(3, 3, 3), dtype=np.int32)
        z_1 = np.random.randint(1, 100, size=(3, 3, 3), dtype=np.int32)
        x_1, x_2, z_2 = x.astype(np.float32), y.astype(np.float32), z_1.astype(np.float32)
        in0 = x
        in1 = [x, y]
        in2 = (x, y, z_1)
        in3 = x_1
        in4 = [x_1, x_2]
        in5 = (x_1, x_2, z_2)
        _test_forward_add_n(in0)
        _test_forward_add_n(in1)
        _test_forward_add_n(in2)
        _test_forward_add_n(in3)
        _test_forward_add_n(in4)
        _test_forward_add_n(in5)


#######################################################################
# Logical operators
# -----------------


def _test_logical_binary(logical_bin_op, data):

    with tf.Graph().as_default():
        in_data = [
            array_ops.placeholder(shape=data[0].shape, dtype="bool", name="in_0"),
            array_ops.placeholder(shape=data[1].shape, dtype="bool", name="in_1"),
        ]
        if logical_bin_op is math_ops.logical_not:
            out = math_ops.logical_or(in_data[0], in_data[1], name="out1")
            out = logical_bin_op(out, name="out")
        else:
            out = logical_bin_op(in_data[0], in_data[1], name="out")

        compare_tflite_with_tvm(data, ["in_0:0", "in_1:0"], in_data, [out])


def _test_forward_logical_and(data):
    """One iteration of logical and"""
    return _test_logical_binary(math_ops.logical_and, data)


def _test_forward_logical_or(data):
    """One iteration of logical or"""
    return _test_logical_binary(math_ops.logical_or, data)


def _test_forward_logical_not(data):
    """One iteration of logical not"""
    return _test_logical_binary(math_ops.logical_not, data)


def test_all_logical():
    data = [
        np.random.choice(a=[False, True], size=(2, 3, 4)).astype("bool"),
        np.random.choice(a=[False, True], size=(2, 3, 4)).astype("bool"),
    ]
    # boolean dtype is not supported by older versions than TFLite 1.15.0
    if package_version.parse(tf.VERSION) >= package_version.parse("1.15.0"):
        _test_forward_logical_and(data)
        _test_forward_logical_or(data)
        _test_forward_logical_not(data)


#######################################################################
# Zeros like
# ----------


def _test_zeros_like(data):
    """One iteration of ZEROS LIKE"""
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = gen_array_ops.zeros_like(in_data)
        compare_tflite_with_tvm(data, "Placeholder:0", [in_data], [out])


def test_forward_zeros_like():
    """ZEROS LIKE"""
    _test_zeros_like(np.arange(6.0, dtype=np.float32).reshape((1, 6)))


#######################################################################
# Fill
# ----


def _test_fill(dims, value_data, value_dtype):
    """Use the fill op to create a tensor of value_data with constant dims."""

    value_data = np.array(value_data, dtype=value_dtype)
    # TF 1.13 TFLite convert method does not accept empty shapes
    if package_version.parse(tf.VERSION) >= package_version.parse("1.14.0"):
        with tf.Graph().as_default():
            value = array_ops.placeholder(dtype=value_dtype, name="value", shape=[])
            out = tf.fill(dims, value)
            compare_tflite_with_tvm([value_data], ["value"], [value], [out])

    with tf.Graph().as_default():
        input1 = array_ops.placeholder(dtype=value_dtype, name="input1", shape=dims)
        # Fill op gets converted to static tensor during conversion
        out = tf.fill(dims, value_data)
        out1 = tf.add(out, input1)
        input1_data = np.random.uniform(0, 5, size=dims).astype(value_dtype)
        compare_tflite_with_tvm([input1_data], ["input1"], [input1], [out1])


def test_forward_fill():
    """Test FILL op"""

    _test_fill((1, 2, 2, 4), 5, "int32")
    _test_fill((1, 2, 2, 4), 5, "float32")
    _test_fill((5,), 5, "int32")


#######################################################################
# Reduce
# ------


def _test_reduce(math_op, data, keep_dims=None):
    """One iteration of reduce"""

    assert len(data) == 2

    # Test with tensor and constant
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data[0].shape, dtype=data[0].dtype, name="in")
        out = math_op(in_data, data[1], keep_dims)
        compare_tflite_with_tvm([data[0]], ["in:0"], [in_data], [out])


def _test_reduce_quantize(math_op, data, keep_dims=None):
    """One iteration of reduce"""

    assert len(data) == 2

    # Test with tensor and constant
    with tf.Graph().as_default():
        in_data = [array_ops.placeholder(shape=data[0].shape, dtype="float32", name="in")]
        inq_data = [
            tf.quantization.fake_quant_with_min_max_args(
                in_data[0], min=-100, max=100, name="inq_0"
            )
        ]
        input_range = {"inq_0": (-100, 100)}
        out = math_op(inq_data, data[1], keep_dims)
        out = tf.quantization.fake_quant_with_min_max_args(out, min=-200, max=200, name="out")
        compare_tflite_with_tvm(
            [data[0]], ["inq_0:0"], [inq_data[0]], [out], quantized=True, input_range=input_range
        )


#######################################################################
# Reduce_min
# ----------


def _test_reduce_min(data, keep_dims=None):
    """One iteration of reduce_min"""
    return _test_reduce(math_ops.reduce_min, data, keep_dims)


#######################################################################
# Reduce_max
# ----------


def _test_reduce_max(data, keep_dims=None):
    """One iteration of reduce_max"""
    return _test_reduce(math_ops.reduce_max, data, keep_dims)


#######################################################################
# Reduce_mean
# -----------


def _test_reduce_mean(data, keep_dims=None, quantized=False):
    """One iteration of reduce_mean"""
    if quantized:
        return _test_reduce_quantize(math_ops.reduce_mean, data, keep_dims)
    else:
        return _test_reduce(math_ops.reduce_mean, data, keep_dims)


#######################################################################
# Reduce_prod
# -----------


def _test_reduce_prod(data, keep_dims=None):
    """One iteration of reduce_prod"""
    return _test_reduce(math_ops.reduce_prod, data, keep_dims)


#######################################################################
# Reduce_sum
# -----------


def _test_reduce_sum(data, keep_dims=None):
    """One iteration of reduce_sum"""
    return _test_reduce(math_ops.reduce_sum, data, keep_dims)


#######################################################################
# Reduce_any
# ----------


def _test_reduce_any(data, keep_dims=None):
    """One iteration of reduce_any"""
    return _test_reduce(math_ops.reduce_any, data, keep_dims)


def _test_forward_reduce(testop, dtype="float32"):
    """Reduce"""
    if dtype == "bool":
        data0 = [np.random.choice(a=[False, True], size=(16, 16, 16, 16)).astype(dtype), None]
        data1 = [
            np.random.choice(a=[False, True], size=(16, 16, 16, 16)).astype(dtype),
            np.array(1, dtype=np.int32),
        ]
        data2 = [
            np.random.choice(a=[False, True], size=(16, 16, 16, 16)).astype(dtype),
            np.array([1, 2], dtype=np.int32),
        ]
    else:
        data0 = [np.random.rand(16, 16, 16, 16).astype(dtype), None]
        data1 = [np.random.rand(16, 16, 16, 16).astype(dtype), np.array(1, dtype=np.int32)]
        data2 = [np.random.rand(16, 16, 16, 16).astype(dtype), np.array([1, 2], dtype=np.int32)]

    for data in [data0, data1, data2]:
        testop(data)
        testop(data, keep_dims=False)
        testop(data, keep_dims=True)


def _test_forward_reduce_quantized(testop):
    data0 = [
        np.array(np.random.uniform(0, 255, (3, 6)), dtype=np.uint8),
        np.array([1, 2], dtype=np.int32),
    ]
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
    if package_version.parse(tf.VERSION) >= package_version.parse("1.15.0"):
        _test_forward_reduce(_test_reduce_any, dtype="bool")


#######################################################################
# Arg_min_max
# -----------


def _test_arg_min_max(math_op, data, axis, quantized=False):
    """One iteration of arg_min_max"""

    with tf.Graph().as_default():
        t_name = "in"
        in_data = array_ops.placeholder(shape=data.shape, dtype=np.float32, name=t_name)
        input_range = None
        qmin, qmax = -100, 102
        if quantized:
            inq_data = tf.quantization.fake_quant_with_min_max_args(
                in_data, min=qmin, max=qmax, name="q" + t_name
            )
            input_range = {inq_data.name.split(":")[0]: (qmin, qmax)}
            out = math_op(input=inq_data, axis=axis)
            compare_tflite_with_tvm(
                [data], [inq_data.name], [inq_data], [out], quantized=True, input_range=input_range
            )
        else:
            out = math_op(input=in_data, axis=axis)
            compare_tflite_with_tvm([data], [in_data.name], [in_data], [out])


def test_forward_arg_min_max():
    """Arg min max"""
    # test quantized
    for data in [np.array(np.random.uniform(-100, 100, (3, 4)), dtype=np.uint8)]:
        # There is no quantized version of ArgMin
        for axis in [None, 0, 1, -1]:
            _test_arg_min_max(math_ops.argmax, data, axis, True)

    for data in [np.array(np.random.uniform(-100, 100, (3, 4)), dtype=np.float32)]:
        for axis in [None, 0, 1, -1]:
            _test_arg_min_max(math_ops.argmax, data, axis)
            _test_arg_min_max(math_ops.argmin, data, axis)


#######################################################################
# Select, Where
# -------------


def test_forward_select():
    """Select"""
    with tf.Graph().as_default():
        with tf.Session() as _:
            input1 = tf.placeholder(tf.int32, shape=[1, 4, 4, 3], name="input1")
            input2 = tf.placeholder(tf.int32, shape=[1, 4, 4, 3], name="input2")
            mask = input1 > input2
            out = tf.where(mask, input1 + 1, input2 * 2)
            in_data1 = np.random.uniform(0, 10, size=(1, 4, 4, 3)).astype("int32")
            in_data2 = np.random.uniform(0, 10, size=(1, 4, 4, 3)).astype("int32")

            compare_tflite_with_tvm(
                [in_data1, in_data2], ["input1:0", "input2:0"], [input1, input2], [out]
            )


@pytest.mark.parametrize("quant_bits", [2, 4, 8, 16])
@pytest.mark.parametrize(
    "value, min_value, max_value",
    [[-10.11, -6, 6], [-3.55, -6, 6], [0, -6, 6], [3.55, -6, 6], [10.11, -6, 6]],
)
def test_forward_fake_quant(value, min_value, max_value, quant_bits):
    """Fake quant"""
    with tf.Graph().as_default():
        with tf.Session() as _:
            input_placeholder = tf.placeholder(tf.float32, shape=[1], name="input")
            out = tf.quantization.fake_quant_with_min_max_args(
                input_placeholder, min=min_value, max=max_value, num_bits=quant_bits, name=None
            )

            in_data = np.float32(value)
            compare_tflite_with_tvm([in_data], ["input:0"], [input_placeholder], [out])


# Squeeze
# -------


def _test_squeeze(data, squeeze_dims=None):
    """One iteration of squeeze"""

    if squeeze_dims is None:
        squeeze_dims = []

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)

        if squeeze_dims:
            out = array_ops.squeeze(in_data, squeeze_dims)
        else:
            out = array_ops.squeeze(in_data)

        compare_tflite_with_tvm(data, "Placeholder:0", [in_data], [out])


def test_forward_squeeze():
    """Squeeze"""
    _test_squeeze(np.arange(6).reshape((1, 2, 1, 3)), [0, 2])
    _test_squeeze(np.arange(6).reshape((2, 1, 3, 1)), [1, 3])


#######################################################################
# Quantize/DeQuantize
# -------------------


def _test_quantize_dequantize(data):
    """One iteration of quantize and dequantize"""

    # Keras model to force TFLite converter to insert 2 TFLite quantize ops.
    # First TFLite quantize op converts float32 tensor to int8 tensor - Qnn quantize.
    # Second TFLite quantize op converts int8 tensor to int8 tensor - Qnn requantize.
    data_in = tf.keras.layers.Input(shape=data.shape[1:])
    relu = tf.keras.layers.ReLU()(data_in)
    add = tf.keras.layers.Add()([data_in, relu])
    concat = tf.keras.layers.Concatenate(axis=0)([relu, add])
    keras_model = tf.keras.models.Model(inputs=data_in, outputs=concat)

    # To create quantized values with dynamic range of activations, needs representative dataset
    def representative_data_gen():
        for _ in range(1):
            yield [data]

    tflite_model_quant = _quantize_keras_model(keras_model, representative_data_gen, True, True)

    tflite_output = run_tflite_graph(tflite_model_quant, data)
    if tf.__version__ < LooseVersion("2.9"):
        in_node = data_in.name.split(":")[0]
    else:
        in_node = "serving_default_" + data_in.name + ":0"
    tvm_output = run_tvm_graph(tflite_model_quant, data, in_node)
    tvm.testing.assert_allclose(
        np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-2
    )


def _test_quantize_dequantize_const(data):
    """One iteration of quantize and dequantize"""

    # Keras model to force TFLite converter to insert 2 TFLite quantize ops.
    # First TFLite quantize op converts float32 tensor to int8 tensor - Qnn quantize.
    # Second TFLite quantize op converts int8 tensor to int8 tensor - Qnn requantize.
    data_in = tf.keras.layers.Input(shape=data.shape[1:])
    relu = tf.keras.layers.ReLU()(data_in)
    add = tf.keras.layers.Add()([data, relu])
    concat = tf.keras.layers.Concatenate(axis=0)([relu, add])
    keras_model = tf.keras.models.Model(inputs=data_in, outputs=concat)

    # To create quantized values with dynamic range of activations, needs representative dataset
    def representative_data_gen():
        for _ in range(1):
            yield [data]

    tflite_model_quant = _quantize_keras_model(keras_model, representative_data_gen, True, True)

    tflite_output = run_tflite_graph(tflite_model_quant, data)
    if tf.__version__ < LooseVersion("2.9"):
        in_node = data_in.name.split(":")[0]
    else:
        in_node = "serving_default_" + data_in.name + ":0"
    tvm_output = run_tvm_graph(tflite_model_quant, data, in_node)
    tvm.testing.assert_allclose(
        np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-2
    )


def test_forward_quantize_dequantize():
    """Quantize Dequantize"""
    data = np.random.uniform(0, 1, (1, 4, 4, 3)).astype("float32")
    if package_version.parse(tf.VERSION) >= package_version.parse("2.1.0"):
        _test_quantize_dequantize(data)
        _test_quantize_dequantize_const(data)


#######################################################################
# Pad
# ---


def _test_pad(data, mode="CONSTANT", quantized=False):
    """One iteration of PAD"""

    assert len(data) == 2

    # Test with tensor and constant
    with tf.Graph().as_default():
        in_data = [array_ops.placeholder(shape=data[0].shape, dtype="float32", name="in")]

        if quantized:
            # fake_quant will keep the tensors in float32 until the conversion in the session
            input_range = {"inq_0": (-100, 100)}
            inq_data = [
                tf.quantization.fake_quant_with_min_max_args(
                    in_data[0], min=-100, max=100, name="inq_0"
                )
            ]
            out = array_ops.pad(
                inq_data[0], ops.convert_to_tensor(data[1], dtype=data[1].dtype), mode=mode
            )
            compare_tflite_with_tvm(
                [data[0]],
                ["inq_0:0"],
                inq_data,
                [out],
                quantized=True,
                input_range=input_range,
                experimental_new_converter=True,
            )
        else:
            out = array_ops.pad(
                in_data[0], ops.convert_to_tensor(data[1], dtype=data[1].dtype), mode=mode
            )
            compare_tflite_with_tvm([data[0]], ["in:0"], in_data, [out])


def test_forward_pad():
    """Pad"""
    _test_pad(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 1, 3)),
            np.array([[1, 1], [2, 2], [1, 1], [2, 2]], dtype=np.int32),
        ]
    )
    _test_pad(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 3)),
            np.array([[2, 2], [1, 1], [1, 1]], dtype=np.int32),
        ]
    )
    _test_pad(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 3)),
            np.array([[1, 1], [2, 2]], dtype=np.int32),
        ]
    )
    _test_pad(
        [
            np.arange(1.0, 4.0, dtype=np.float32).reshape((1, 3)),
            np.array([[1, 1], [2, 2]], dtype=np.int32),
        ]
    )
    _test_pad(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 3)),
            np.array([[1, 1], [2, 2]], dtype=np.int32),
        ],
        mode="REFLECT",
    )
    _test_pad(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 3)),
            np.array([[1, 1], [2, 2]], dtype=np.int32),
        ],
        mode="SYMMETRIC",
    )
    _test_pad(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 3)),
            np.array([[1, 1], [2, 2]], dtype=np.int64),
        ],
        mode="REFLECT",
    )
    _test_pad(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 3)),
            np.array([[1, 1], [2, 2]], dtype=np.int64),
        ],
        mode="SYMMETRIC",
    )
    _test_pad(
        [
            np.arange(0, 256, dtype=np.uint8).reshape((1, 256)),
            np.array([[1, 1], [2, 2]], dtype=np.int32),
        ],
        quantized=True,
    )
    _test_pad(
        [
            np.arange(0, 256, dtype=np.uint8).reshape((1, 256)),
            np.array([[1, 1], [2, 2]], dtype=np.int32),
        ],
        quantized=True,
        mode="SYMMETRIC",
    )
    _test_pad(
        [
            np.arange(0, 256, dtype=np.uint8).reshape((1, 256)),
            np.array([[0, 0], [2, 2]], dtype=np.int32),
        ],
        quantized=True,
        mode="REFLECT",
    )


#######################################################################
# PADV2
# -----


def _test_padv2(data, mode="CONSTANT", quantized=False):
    """One iteration of PADV2"""

    assert len(data) == 2 or len(data) == 3

    with_constant_values = len(data) == 3

    # Test with tensor and constant
    with tf.Graph().as_default():
        in_data = [array_ops.placeholder(shape=data[0].shape, dtype="float32", name="in")]

        if quantized:
            # fake_quant will keep the tensors in float32 until the conversion in the session
            input_range = {"inq_0": (-100, 100)}
            inq_data = [
                tf.quantization.fake_quant_with_min_max_args(
                    in_data[0], min=-100, max=100, name="inq_0"
                )
            ]
            if with_constant_values:
                in_constant_values = constant_op.constant(
                    data[2], shape=data[2].shape, dtype="float32", name="in_constant_values"
                )
                inq_constant_values = tf.quantization.fake_quant_with_min_max_args(
                    in_constant_values, min=-100, max=100, name="inq_constant_values"
                )
                out = array_ops.pad_v2(
                    inq_data[0],
                    ops.convert_to_tensor(data[1], dtype=data[1].dtype),
                    constant_values=inq_constant_values,
                    mode=mode,
                )
                out = tf.quantization.fake_quant_with_min_max_args(
                    out, min=-100, max=100, name="out"
                )
            else:
                out = array_ops.pad_v2(
                    inq_data[0], ops.convert_to_tensor(data[1], dtype=data[1].dtype), mode=mode
                )
            compare_tflite_with_tvm(
                [data[0]], ["inq_0:0"], inq_data, [out], quantized=True, input_range=input_range
            )
        else:
            if with_constant_values:
                out = array_ops.pad_v2(
                    in_data[0],
                    ops.convert_to_tensor(data[1], dtype=data[1].dtype),
                    constant_values=ops.convert_to_tensor(data[2], dtype=data[2].dtype),
                    mode=mode,
                )
            else:
                out = array_ops.pad_v2(
                    in_data[0], ops.convert_to_tensor(data[1], dtype=data[1].dtype), mode=mode
                )
            compare_tflite_with_tvm([data[0]], ["in:0"], in_data, [out])


def test_forward_padv2():
    """PADV2"""
    # Tests without Constant_values
    _test_padv2(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 1, 3)),
            np.array([[1, 1], [2, 2], [1, 1], [2, 2]], dtype=np.int32),
        ]
    )
    _test_padv2(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 3)),
            np.array([[2, 2], [1, 1], [1, 1]], dtype=np.int32),
        ]
    )
    _test_padv2(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 3)),
            np.array([[1, 1], [2, 2]], dtype=np.int32),
        ]
    )
    _test_padv2(
        [
            np.arange(1.0, 4.0, dtype=np.float32).reshape((1, 3)),
            np.array([[1, 1], [2, 2]], dtype=np.int32),
        ]
    )
    _test_padv2(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 3)),
            np.array([[1, 1], [2, 2]], dtype=np.int32),
        ],
        mode="REFLECT",
    )
    _test_padv2(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 3)),
            np.array([[1, 1], [2, 2]], dtype=np.int32),
        ],
        mode="SYMMETRIC",
    )
    _test_padv2(
        [
            np.arange(0, 256, dtype=np.uint8).reshape((1, 256)),
            np.array([[1, 1], [2, 2]], dtype=np.int32),
        ],
        quantized=True,
    )

    # Tests with Constant_values
    _test_padv2(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 1, 3)),
            np.array([[1, 1], [2, 2], [1, 1], [2, 2]], dtype=np.int32),
            np.array([2], dtype=np.float32),
        ]
    )
    _test_padv2(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 3)),
            np.array([[2, 2], [1, 1], [1, 1]], dtype=np.int32),
            np.array([1], dtype=np.float32),
        ]
    )
    _test_padv2(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 3)),
            np.array([[1, 1], [2, 2]], dtype=np.int32),
            np.array([-1], dtype=np.float32),
        ]
    )
    _test_padv2(
        [
            np.arange(1.0, 4.0, dtype=np.float32).reshape((1, 3)),
            np.array([[1, 1], [2, 2]], dtype=np.int32),
            np.array([2], dtype=np.float32),
        ]
    )
    # NOTE: In versions > 2.1.0, there is a bug in Tensorflow package for this scenario.
    #       Hence, it is disabled temporarily for TF version > 2.1.0 .
    if package_version.parse(tf.VERSION) <= package_version.parse("2.1.0"):
        _test_padv2(
            [
                np.arange(0, 256, dtype=np.uint8).reshape((1, 256)),
                np.array([[1, 1], [2, 2]], dtype=np.int32),
                np.array([2], dtype=np.float32),
            ],
            quantized=True,
        )

    # Constant Values input can be scalar
    _test_padv2(
        [
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 1, 3)),
            np.array([[1, 1], [2, 2], [1, 1], [2, 2]], dtype=np.int32),
            np.float32(2),
        ]
    )
    # NOTE: In versions > 2.1.0, there is a bug in Tensorflow package for this scenario.
    #       Hence, it is disabled temporarily for TF versions > 2.1.0.
    if package_version.parse(tf.VERSION) <= package_version.parse("2.1.0"):
        _test_padv2(
            [
                np.arange(0, 256, dtype=np.uint8).reshape((1, 256)),
                np.array([[1, 1], [2, 2]], dtype=np.int32),
                np.uint8(10),
            ],
            quantized=True,
        )


#######################################################################
# EXPAND_DIMS
# -----------


def _test_expand_dims(input_shape, input_type, axis, quantized=False):
    """One iteration of EXPAND_DIMS"""
    with tf.Graph().as_default():
        axis = ops.convert_to_tensor(axis, dtype=axis.dtype)

        if quantized:
            # ignoring input_type as quantized requires uint8
            input_array = np.random.uniform(0, 256, input_shape).astype("uint8")
            in_input = tf.placeholder(dtype="float32", shape=input_array.shape, name="input")

            input_range = {"q_input": (-100, 100)}
            inq_input = tf.quantization.fake_quant_with_min_max_args(
                in_input, min=-100, max=100, name="q_input"
            )

            out = array_ops.expand_dims(inq_input, axis=axis)
            out = tf.quantization.fake_quant_with_min_max_args(out, min=-100, max=100, name="out")

            compare_tflite_with_tvm(
                [input_array],
                ["q_input"],
                [inq_input],
                [out],
                quantized=True,
                input_range=input_range,
            )
        else:
            input_array = np.random.uniform(-100, 100, input_shape).astype(input_type)
            in_input = tf.placeholder(
                dtype=input_array.dtype, shape=input_array.shape, name="input"
            )

            out = array_ops.expand_dims(in_input, axis=axis)

            compare_tflite_with_tvm([input_array], ["input"], [in_input], [out])


def test_forward_expand_dims():
    """EXPAND_DIMS"""
    for quantized in [False, True]:
        _test_expand_dims((6, 2, 7, 5), "float32", np.int32(0), quantized=quantized)
        _test_expand_dims((1, 2, 3), "int32", np.int32(-2), quantized=quantized)
        _test_expand_dims((2, 4, 5), "float32", np.array([1], dtype=np.int32), quantized=quantized)


#######################################################################
# ONE_HOT
# -------


def _test_one_hot(indices, depth, on_value, off_value, axis=None):
    """One iteration of One_Hot"""
    with tf.Graph().as_default():
        in_indices = tf.placeholder(dtype=indices.dtype, shape=indices.shape, name="indices")
        in_depth = ops.convert_to_tensor(depth, dtype=depth.dtype)
        in_on_value = tf.placeholder(dtype=on_value.dtype, shape=on_value.shape, name="on_value")
        in_off_value = tf.placeholder(
            dtype=off_value.dtype, shape=off_value.shape, name="off_value"
        )
        if axis is not None:
            out = array_ops.one_hot(in_indices, in_depth, in_on_value, in_off_value, axis=axis)
        else:
            out = array_ops.one_hot(in_indices, in_depth, in_on_value, in_off_value)
        compare_tflite_with_tvm(
            [indices, on_value, off_value],
            ["indices", "on_value", "off_value"],
            [in_indices, in_on_value, in_off_value],
            [out],
        )


def test_forward_one_hot():
    """One_Hot"""
    _test_one_hot(np.int32(2), np.int32(8), np.int32(1), np.int32(0))
    _test_one_hot(np.int32(4), np.int32(8), np.float32(1), np.float32(0))
    _test_one_hot(np.array([1, 2, 3], dtype=np.int32), np.int32(8), np.int32(3), np.int32(-1))
    _test_one_hot(
        np.array([1, 2, 3], dtype=np.int32), np.int32(8), np.int32(3), np.int32(-1), axis=0
    )


#######################################################################
# Pack
# ----


def _test_pack(data, is_var, axis, quantized=False):
    """One iteration of pack"""

    assert len(data) >= 1
    assert len(data) == len(is_var)
    if quantized:
        with tf.Graph().as_default():
            in_data = [
                array_ops.placeholder(shape=d.shape, dtype="float32", name="in_" + str(idx))
                if is_var[idx]
                else constant_op.constant(
                    d, shape=d.shape, dtype="float32", name="in_constant_" + str(idx)
                )
                for idx, d in enumerate(data)
            ]
            inq_data = [
                tf.quantization.fake_quant_with_min_max_args(
                    i_data, min=-100, max=100, name=f"inq_{idx}"
                )
                for idx, i_data in enumerate(in_data)
            ]
            input_range = {}
            for i in range(len(data)):
                input_range[f"inq_{i}"] = (-100, 100)

            out = array_ops.pack(inq_data, axis=axis)
            out = tf.quantization.fake_quant_with_min_max_args(out, min=-100, max=100, name="out")
            name = [f"inq_{idx}:0" for idx in range(len(data))]
            compare_tflite_with_tvm(
                data, name, inq_data, [out], quantized=True, input_range=input_range
            )
    else:
        with tf.Graph().as_default():
            in_data = [
                array_ops.placeholder(shape=d.shape, dtype=d.dtype, name="in_" + str(idx))
                if is_var[idx]
                else constant_op.constant(
                    d, shape=d.shape, dtype=d.dtype, name="in_constant_" + str(idx)
                )
                for idx, d in enumerate(data)
            ]

            out = array_ops.pack(in_data, axis=axis)
            name = [_.name for _ in in_data]
            compare_tflite_with_tvm(data, name, in_data, [out], experimental_new_converter=True)


def test_forward_pack():
    """Pack"""
    _test_pack([np.int32(1), np.int32(5)], [False, False], 0)
    _test_pack([np.array([1, 4]), np.array([2, 5]), np.array([3, 6])], [True, False, False], 0)
    _test_pack(
        [np.arange(6).reshape((1, 2, 1, 3)), np.arange(6).reshape((1, 2, 1, 3))], [True, True], 1
    )

    _test_pack([np.arange(6).reshape((3, 2)), np.arange(6).reshape((3, 2))], [True, True], 1)

    _test_pack(
        [
            np.arange(6).reshape((2, 1, 1, 3)),
            np.arange(6).reshape((2, 1, 1, 3)),
            np.arange(6).reshape((2, 1, 1, 3)),
        ],
        [True, True, True],
        1,
    )

    _test_pack(
        [
            np.arange(6, dtype=np.uint8).reshape((2, 1, 1, 3)),
            np.arange(6, dtype=np.uint8).reshape((2, 1, 1, 3)),
            np.arange(6, dtype=np.uint8).reshape((2, 1, 1, 3)),
        ],
        [True, True, True],
        1,
        quantized=True,
    )


#######################################################################
# Unpack
# ------


def _test_unpack(data, axis, num_unpacks):
    """One iteration of UNPACK"""
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = gen_array_ops.unpack(in_data, num=num_unpacks, axis=axis, name="unpack")
        out_names = ["out_" + str(n) + ":0" for n in range(num_unpacks)]
        compare_tflite_with_tvm([data], "Placeholder:0", [in_data], out, out_names=out_names)


def test_forward_unpack():
    """UNPACK"""
    _test_unpack(np.array(np.random.uniform(0, 5, (3, 1)), dtype=np.int32), axis=1, num_unpacks=1)
    _test_unpack(np.array(np.random.uniform(0, 5, (3, 4)), dtype=np.float32), axis=0, num_unpacks=3)
    _test_unpack(
        np.array(np.random.uniform(0, 5, (3, 1, 2)), dtype=np.float32), axis=0, num_unpacks=3
    )
    # tflite 1.13 doesn't accept negative axis
    if package_version.parse(tf.VERSION) >= package_version.parse("1.14.0"):
        _test_unpack(
            np.array(np.random.uniform(0, 5, (3, 6)), dtype=np.int32), axis=-2, num_unpacks=3
        )
        _test_unpack(
            np.array(np.random.uniform(0, 5, (2, 3, 4)), dtype=np.int32), axis=-3, num_unpacks=2
        )


#######################################################################
# Local response normalization
# ----------------------------


def _test_local_response_normalization(data, depth_radius, bias, alpha, beta):
    """One iteration of LOCAL_RESPONSE_NORMALIZATION"""
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype="float32", name="in_0")
        out = nn_ops.local_response_normalization(
            in_data, depth_radius=depth_radius, bias=bias, alpha=alpha, beta=beta
        )
        compare_tflite_with_tvm(data, "in_0:0", [in_data], [out])


def test_forward_local_response_normalization():
    """LOCAL_RESPONSE_NORMALIZATION"""
    data = np.random.uniform(size=(1, 6, 4, 3)).astype("float32")
    # LOCAL_RESPONSE_NORMALIZATION come with TFLite >= 1.14.0 fbs schema
    if package_version.parse(tf.VERSION) >= package_version.parse("1.14.0"):
        _test_local_response_normalization(data, depth_radius=5, bias=1, alpha=1, beta=0.5)


#######################################################################
# L2 normalization
# ----------------


def _test_l2_normalization(data, axis, fused_activation_function=None):
    """One iteration of L2_NORMALIZATION"""
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = nn_impl.l2_normalize(in_data, axis)
        out = with_fused_activation_function(out, fused_activation_function)
        compare_tflite_with_tvm(data, "Placeholder:0", [in_data], [out])


def test_forward_l2_normalization():
    """L2_NORMALIZATION"""
    data = np.random.uniform(size=(3, 6, 4)).astype("float32")
    _test_l2_normalization(data, axis=2)
    _test_l2_normalization(data, axis=2, fused_activation_function="RELU")


#######################################################################
# Logistic
# --------


def _test_logistic(data, quantized=False):
    """One iteration of LOGISTIC"""
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype="float32", name="in_0")

        if quantized:
            inq_data = tf.quantization.fake_quant_with_min_max_args(
                in_data, min=-5, max=5, name="inq_0"
            )
            input_range = {"inq_0": (-5, 5)}
            out = math_ops.sigmoid(inq_data)
            out = tf.quantization.fake_quant_with_min_max_args(out, min=0, max=1, name="out")
            compare_tflite_with_tvm(
                data, "inq_0:0", [inq_data], [out], quantized=True, input_range=input_range
            )
        else:
            out = math_ops.sigmoid(in_data)
            compare_tflite_with_tvm(data, "in_0:0", [in_data], [out])


def test_forward_logistic():
    """LOGISTIC"""
    _test_logistic(np.arange(6.0, dtype=np.float32).reshape((1, 6)))
    _test_logistic(np.random.uniform(0, 255, (3, 6)).astype(np.uint8), quantized=True)


#######################################################################
# Softmax
# -------


def _test_softmax(data):
    """One iteration of softmax"""
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = nn_ops.softmax(in_data)
        compare_tflite_with_tvm(data, "Placeholder:0", [in_data], [out])


def test_forward_softmax():
    """Softmax"""
    _test_softmax(np.arange(6.0, dtype=np.float32).reshape((1, 6)))
    _test_softmax(np.arange(6.0, dtype=np.float32).reshape((1, 2, 3)))


######################################################################
# Log_softmax
# -----------


def _test_log_softmax(data, quantized=False):
    """One iteration of log_softmax"""
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype="float32", name="in_0")

        if quantized:
            inq_data = tf.quantization.fake_quant_with_min_max_args(
                in_data, min=-10, max=10, name="inq_0"
            )
            input_range = {"inq_0": (-10, 10)}
            # tflite log_softmax supports only the case when axis is not specified
            out = nn_ops.log_softmax(inq_data)
            out = tf.quantization.fake_quant_with_min_max_args(out, min=-20, max=0, name="out")
            compare_tflite_with_tvm(
                data, "inq_0:0", [inq_data], [out], quantized=True, input_range=input_range
            )
        else:
            out = nn_ops.log_softmax(in_data)
            compare_tflite_with_tvm(data, "in_0:0", [in_data], [out])


def test_forward_log_softmax():
    """Log_softmax"""
    _test_log_softmax(np.random.uniform(-10, 10, size=(3, 6)).astype(np.float32))
    _test_log_softmax(np.random.uniform(0, 255, (3, 6)).astype(np.uint8), quantized=True)


#######################################################################
# Tanh
# ----


def _test_tanh(data, quantized=False):
    """One iteration of TANH"""
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype="float32", name="in_0")

        if quantized:
            inq_data = tf.quantization.fake_quant_with_min_max_args(
                in_data, min=-3, max=3, name="inq_0"
            )
            input_range = {"inq_0": (-3, 3)}
            out = math_ops.tanh(inq_data)
            out = tf.quantization.fake_quant_with_min_max_args(out, min=-1, max=1, name="out")
            compare_tflite_with_tvm(
                data, "inq_0:0", [inq_data], [out], quantized=True, input_range=input_range
            )
        else:
            out = math_ops.tanh(in_data)
            compare_tflite_with_tvm(data, "in_0:0", [in_data], [out])


def test_forward_tanh():
    """TANH"""
    _test_tanh(np.arange(6.0, dtype=np.float32).reshape((1, 6)), quantized=False)
    _test_tanh(np.arange(0, 256, 30, dtype=np.uint8), quantized=True)


#######################################################################
# ReLu
# ----


def _test_relu(data, quantized=False):
    """One iteration of ReLU"""

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype="float32", name="in_0")

        if quantized:
            inq_data = tf.quantization.fake_quant_with_min_max_args(
                in_data, min=-10, max=10, name="inq_0"
            )
            input_range = {"inq_0": (-10, 10)}
            out = nn_ops.relu(inq_data)
            out = tf.quantization.fake_quant_with_min_max_args(out, min=0, max=6, name="out")
            compare_tflite_with_tvm(
                data, "inq_0:0", [inq_data], [out], quantized=True, input_range=input_range
            )
        else:
            out = nn_ops.relu(in_data)
            compare_tflite_with_tvm(data, "in_0:0", [in_data], [out])


def test_forward_relu():
    """ReLU"""
    _test_relu(np.arange(6.0, dtype=np.float32).reshape((1, 6)))
    _test_relu(np.random.uniform(0, 255, (3, 6)).astype(np.uint8), quantized=True)


#######################################################################
# ReLU6
# -----


def _test_relu6(data, quantized=False):
    """One iteration of ReLU6"""
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype="float32", name="in_0")

        if quantized:
            inq_data = tf.quantization.fake_quant_with_min_max_args(
                in_data, min=-10, max=10, name="inq_0"
            )
            input_range = {"inq_0": (-10, 10)}
            out = nn_ops.relu6(inq_data)
            out = tf.quantization.fake_quant_with_min_max_args(out, min=0, max=6, name="out")
            compare_tflite_with_tvm(
                data, "inq_0:0", [inq_data], [out], quantized=True, input_range=input_range
            )
        else:
            out = nn_ops.relu6(in_data)
            compare_tflite_with_tvm(data, "in_0:0", [in_data], [out])


def test_forward_relu6():
    """ReLU6"""
    _test_relu6(np.random.uniform(-10, 10, size=(3, 6)).astype(np.float32))
    _test_relu6(np.random.uniform(0, 255, (3, 6)).astype(np.uint8), quantized=True)


#######################################################################
# Leaky_ReLU
# ----------


def _test_leaky_relu(data, alpha, quantized=False):
    """One iteration of Leaky_ReLU"""
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype="float32", name="in_0")

        if quantized:
            inq_data = tf.quantization.fake_quant_with_min_max_args(
                in_data, min=-3, max=2, name="inq_0"
            )
            input_range = {"inq_0": (-3, 2)}
            out = nn_ops.leaky_relu(inq_data, alpha)
            out = tf.quantization.fake_quant_with_min_max_args(out, min=-3, max=2, name="out")
            compare_tflite_with_tvm(
                data, "inq_0:0", [inq_data], [out], quantized=True, input_range=input_range
            )
        else:
            out = nn_ops.leaky_relu(in_data, alpha)
            compare_tflite_with_tvm(data, "in_0:0", [in_data], [out])


def test_forward_leaky_relu():
    """Leaky_ReLU"""
    _test_leaky_relu(np.random.uniform(-5, 5, (1, 6)).astype(np.float32), alpha=0.2)
    if package_version.parse(tf.VERSION) >= package_version.parse("1.14.0"):
        _test_leaky_relu(
            np.random.uniform(0, 255, (2, 3)).astype(np.uint8), alpha=0.3, quantized=True
        )


#######################################################################
# ReLU_n1_to_1
# ------------


def _test_relu_n1_to_1(data, quantized=False):
    """One iteration of ReLU_n1_to_1"""
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype="float32", name="in_0")

        if quantized:
            inq_data = tf.quantization.fake_quant_with_min_max_args(
                in_data, min=-3, max=3, name="inq_0"
            )
            input_range = {"inq_0": (-3, 3)}
            # There is no such tf operation.
            # The specific pattern will be replaced into RELU_N1_TO_1 by tflite
            out = math_ops.maximum(-1.0, math_ops.minimum(inq_data, 1.0))
            out = tf.quantization.fake_quant_with_min_max_args(out, min=-1, max=1, name="out")
            compare_tflite_with_tvm(
                data, "inq_0:0", [inq_data], [out], quantized=True, input_range=input_range
            )
        else:
            out = math_ops.maximum(-1.0, math_ops.minimum(in_data, 1.0))
            compare_tflite_with_tvm(data, "in_0:0", [in_data], [out])


def test_forward_relu_n1_to_1():
    """ReLU_n1_to_1"""
    _test_relu_n1_to_1(np.random.uniform(-3, 3, (1, 6)).astype(np.float32))
    if package_version.parse(tf.VERSION) >= package_version.parse("1.14.0"):
        _test_relu_n1_to_1(np.random.uniform(0, 255, (3, 6)).astype(np.uint8), quantized=True)


#######################################################################
# PReLU
# -----


def _test_prelu(data, alpha):
    """One iteration of PReLU"""
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        # This specific pattern will be replaced into PRelu by tflite
        out = nn_ops.relu(in_data) + (-alpha * nn_ops.relu(-in_data))
        compare_tflite_with_tvm(data, "Placeholder:0", [in_data], [out])


def test_forward_prelu():
    """PReLU"""
    _test_prelu(
        np.random.uniform(-5, 5, size=(1, 32, 32, 3)).astype("float32"),
        np.full((3,), 0.2, dtype="float32"),
    )
    _test_prelu(
        np.random.uniform(-5, 5, size=(1, 32, 32, 3)).astype("float32"),
        np.full((1, 3), 0.2, dtype="float32"),
    )
    _test_prelu(
        np.random.uniform(-5, 5, size=(1, 32, 32, 3)).astype("float32"),
        np.full((1, 1, 3), 0.2, dtype="float32"),
    )
    _test_prelu(
        np.random.uniform(-5, 5, size=(1, 32, 32, 3)).astype("float32"),
        np.full((1, 1, 1, 3), 0.2, dtype="float32"),
    )
    #
    _test_prelu(
        np.random.uniform(-5, 5, size=(1, 32, 32, 3)).astype("float32"),
        np.full((32, 3), 0.2, dtype="float32"),
    )
    _test_prelu(
        np.random.uniform(-5, 5, size=(1, 32, 32, 3)).astype("float32"),
        np.full((32, 32, 3), 0.2, dtype="float32"),
    )
    _test_prelu(
        np.random.uniform(-5, 5, size=(1, 32, 32, 3)).astype("float32"),
        np.full((1, 32, 1, 3), 0.2, dtype="float32"),
    )
    #
    _test_prelu(
        np.random.uniform(-5, 5, size=(1, 1, 3)).astype("float32"),
        np.full((3,), 0.2, dtype="float32"),
    )
    _test_prelu(
        np.random.uniform(-5, 5, size=(1, 32, 3)).astype("float32"),
        np.full((32, 3), 0.2, dtype="float32"),
    )
    _test_prelu(
        np.random.uniform(-5, 5, size=(32, 3)).astype("float32"), np.full((3), 0.2, dtype="float32")
    )


#######################################################################
# DepthToSpace
# ------------


def _test_depthtospace(data, block_size):
    """One iteration of depth_to_space operation with given data and block size"""

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = array_ops.depth_to_space(in_data, block_size)
        compare_tflite_with_tvm(data, "Placeholder:0", [in_data], [out])


def test_forward_depthtospace():
    # DEPTH_TO_SPACE comes with TFLite >= 1.15.0 fbs schema
    if package_version.parse(tf.VERSION) >= package_version.parse("1.15.0"):
        _test_depthtospace(np.random.normal(size=[1, 32, 32, 4]).astype("float32"), 2)
        _test_depthtospace(np.random.normal(size=[1, 16, 8, 32]).astype("float32"), 4)


#######################################################################
# SpaceToDepth
# ------------


def _test_spacetodepth(data, block_size):
    """One iteration of space_to_depth operation with given data and block size"""

    with tf.Graph().as_default():
        in_data = array_ops.placeholder(shape=data.shape, dtype=data.dtype)
        out = array_ops.space_to_depth(in_data, block_size)
        compare_tflite_with_tvm(data, "Placeholder:0", [in_data], [out])


def test_forward_spacetodepth():
    _test_spacetodepth(np.random.normal(size=[1, 32, 32, 4]).astype("float32"), 2)
    _test_spacetodepth(np.random.normal(size=[1, 16, 8, 32]).astype("float32"), 4)


#######################################################################
# ReverseSequence
# ---------------


def _test_reverse_sequence(shape, dtype, seq_lengths, batch_axis, seq_axis):
    """One iteration of reverse_sequence operation with given data and attributes"""

    data = np.random.uniform(0, 100, size=shape).astype(dtype)
    with tf.Graph().as_default():
        in_data = array_ops.placeholder(dtype=dtype, name="input", shape=shape)
        out = tf.reverse_sequence(
            in_data, seq_lengths=seq_lengths, batch_axis=batch_axis, seq_axis=seq_axis
        )

        compare_tflite_with_tvm(data, "input", [in_data], [out])


def test_forward_reverse_sequence():
    if package_version.parse(tf.VERSION) >= package_version.parse("1.14.0"):
        _test_reverse_sequence([4, 3], "float32", [3, 2, 1], 1, 0)
        _test_reverse_sequence([4, 3], "float32", [3, 2, 1, 3], 0, 1)
        _test_reverse_sequence([2, 3, 3, 3], "float32", [2, 3, 2], 2, 1)
        _test_reverse_sequence([2, 4, 6, 4, 5], "float32", [5, 3], 0, 2)
        _test_reverse_sequence([2, 4, 6, 4, 5], "float32", [5, 3, 1, 4], 3, 2)


#######################################################################
# Sparse To Dense
# ---------------
def _test_sparse_to_dense(sparse_indices, sparse_values, default_value, output_shape):
    # tflite 1.13 convert method does not accept empty shapes
    if package_version.parse(tf.VERSION) >= package_version.parse("1.14.0"):
        with tf.Graph().as_default():
            indices = tf.placeholder(
                shape=sparse_indices.shape, dtype=str(sparse_indices.dtype), name="indices"
            )
            values = tf.placeholder(
                shape=sparse_values.shape, dtype=str(sparse_values.dtype), name="values"
            )
            oshape = tf.constant(
                output_shape, shape=output_shape.shape, dtype=str(output_shape.dtype)
            )

            if default_value is None:
                output = tf.sparse_to_dense(indices, oshape, values)
                compare_tflite_with_tvm(
                    [sparse_indices, sparse_values],
                    ["indices", "values"],
                    [indices, values],
                    [output],
                )
            else:
                dv_placeholder = tf.placeholder(
                    shape=(), dtype=str(default_value.dtype), name="default_value"
                )
                output = tf.sparse_to_dense(indices, oshape, values, dv_placeholder)
                compare_tflite_with_tvm(
                    [sparse_indices, sparse_values, default_value],
                    ["indices", "values", "default_value"],
                    [indices, values, dv_placeholder],
                    [output],
                )


def test_forward_sparse_to_dense():
    """
    Works in tvm/topi/tensorflow. But tflite converter breaks this test case
    _test_sparse_to_dense(
        np.int32(1),
        np.int32(3),
        np.int32(0),
        np.array([5]).astype("int32")
    )
    """
    # vector
    _test_sparse_to_dense(
        np.array([0, 1, 4]).astype("int32"),
        np.array([3, 3, 3]).astype("int32"),
        np.int32(0),
        np.array([5]).astype("int32"),
    )
    # vector nXd
    _test_sparse_to_dense(
        np.array([[0, 0], [1, 2]]).astype("int32"),
        np.array([1, 2]).astype("int32"),
        np.int32(0),
        np.array([3, 4]).astype("int32"),
    )
    _test_sparse_to_dense(
        np.array([[0, 0, 0], [1, 2, 3]]).astype("int32"),
        np.array([1, 2]).astype("int32"),
        np.int32(4),
        np.array([2, 3, 4]).astype("int32"),
    )
    # floats
    _test_sparse_to_dense(
        np.array([0, 1, 4]).astype("int32"),
        np.array([3.1, 3.1, 3.1]).astype("float32"),
        np.float32(3.5),
        np.array([5]).astype("int32"),
    )
    # default value not specified
    _test_sparse_to_dense(
        np.array([0, 1, 4]).astype("int32"),
        np.array([3.1, 3.1, 3.1]).astype("float32"),
        None,
        np.array([5]).astype("int32"),
    )


#######################################################################
# Fully Connected
# ---------------
def _test_fully_connected(
    tensor_in_sizes,
    const_input,
    filter_in_sizes,
    bias_in_size=None,
    quantized=False,
    fp16_quantized=False,
):
    """One iteration of fully connected"""

    total_size_1 = np.prod(tensor_in_sizes)
    total_size_2 = np.prod(filter_in_sizes)

    assert (
        int(total_size_1 / tensor_in_sizes[0]) == filter_in_sizes[0]
    ), "input size and filter size are mismatched"

    # Initializes the input tensor with array containing incrementing
    # numbers from 1.
    data_array = np.arange(
        1, total_size_1 + 1, dtype=np.uint8 if quantized and not fp16_quantized else np.float32
    )
    filter_array = np.arange(
        1, total_size_2 + 1, dtype=np.uint8 if quantized and not fp16_quantized else np.float32
    )
    in_name = "input"

    with tf.Graph().as_default():
        in_data = (
            constant_op.constant(data_array, shape=tensor_in_sizes, dtype=np.float32, name=in_name)
            if const_input
            else array_ops.placeholder(shape=tensor_in_sizes, dtype=np.float32, name=in_name)
        )

        in_filter = constant_op.constant(filter_array, shape=filter_in_sizes, dtype=np.float32)
        data_array = np.reshape(data_array, tensor_in_sizes)

        # if we have bias
        if bias_in_size:
            assert bias_in_size[0] == filter_in_sizes[1], "bias and filter size are mismatched"
            bias_array = np.arange(
                1, bias_in_size[0] + 1, dtype=np.uint8 if quantized else np.float32
            )
            in_bias = constant_op.constant(bias_array, shape=bias_in_size, dtype=np.float32)

        if quantized and not fp16_quantized:
            inq_data = tf.quantization.fake_quant_with_min_max_args(
                in_data, min=-100, max=100, name="inq_0"
            )
            input_range = {"inq_0": (-100, 100)}
            inq_filter = tf.quantization.fake_quant_with_min_max_args(
                in_filter, min=-100, max=100, name="inq_1"
            )
            input_range = {"inq_0": (-100, 100), "inq_1": (-100, 100)}
            # reshape N H W C into N H*W*C
            inq_data_reshape = array_ops.reshape(inq_data, [tensor_in_sizes[0], -1])
            out = math_ops.mat_mul(inq_data_reshape, inq_filter)
            out = tf.quantization.fake_quant_with_min_max_args(out, min=-100, max=100, name="out")

            # if we have bias
            if bias_in_size:
                out = nn_ops.bias_add(out, in_bias)

            compare_tflite_with_tvm(
                data_array,
                inq_data.name,
                [inq_data],
                [out],
                quantized=True,
                input_range=input_range,
                experimental_new_converter=True,
            )
        else:
            # reshape N H W C into N H*W*C
            in_data_reshape = array_ops.reshape(in_data, [tensor_in_sizes[0], -1])
            out = math_ops.mat_mul(in_data_reshape, in_filter)
            # TODO : Need to construct a fc op with (keep_num_dims == True)

            # if we have bias
            if bias_in_size:
                out = nn_ops.bias_add(out, in_bias)

            compare_tflite_with_tvm(
                data_array,
                in_data.name,
                [in_data],
                [out],
                experimental_new_converter=True,
                fp16_quantized=fp16_quantized,
            )


def test_forward_fully_connected():
    """Fully Connected"""
    for input_shape, weight_shape, bias_shape in [
        ([1, 4], [4, 4], None),
        ([1, 4], [4, 4], [4]),
        ([1, 1, 1, 5], [5, 5], None),
        ([1, 1, 10], [10, 103], None),
        ([1, 1, 1, 150], [150, 100], None),
        ([1, 1, 1, 150], [150, 100], None),
        ([1, 1, 1, 150], [150, 100], [100]),
        ([5, 1, 1, 150], [150, 100], None),
        ([5, 1, 1, 150], [150, 100], [100]),
    ]:
        for const_input in [False, True]:
            for quantized in [False, True]:
                for fp16_quantized in [False, True]:
                    _test_fully_connected(
                        input_shape,
                        const_input,
                        weight_shape,
                        bias_shape,
                        quantized,
                        fp16_quantized,
                    )


#######################################################################
# REVERSE_V2
# ----------


def _test_reverse_v2(input_shape, axis, dtype):
    """One iteration of REVERSE_V2"""
    with tf.Graph().as_default():
        input_array = np.random.randint(0, 100, size=input_shape).astype(dtype)
        in_input = tf.placeholder(dtype=input_array.dtype, shape=input_array.shape, name="input")
        in_axis = ops.convert_to_tensor(axis, dtype=axis.dtype)

        out = array_ops.reverse(in_input, in_axis)

        compare_tflite_with_tvm([input_array], ["input"], [in_input], [out])


def test_forward_reverse_v2():
    """REVERSE_V2"""
    for dtype in ["float32", "int32"]:
        _test_reverse_v2((5), np.array([0], dtype="int32"), dtype)
        _test_reverse_v2((5, 6, 4, 2), np.array([2], dtype="int32"), dtype)


#######################################################################
# MATRIX_SET_DIAG
# ---------------


def _test_matrix_set_diag(input_shape, input_type, quantized=False):
    """One iteration of MATRIX_SET_DIAG"""
    with tf.Graph().as_default():
        diagonal_shape = list(input_shape[:-2])
        diagonal_shape.append(min(input_shape[-2], input_shape[-1]))

        if quantized:
            # ignoring input_type as quantized requires uint8
            input_array = np.random.uniform(0, 256, input_shape).astype("uint8")
            in_input = tf.placeholder(dtype="float32", shape=input_array.shape, name="input")
            inq_input = tf.quantization.fake_quant_with_min_max_args(
                in_input, min=-100, max=100, name="q_input"
            )

            diagonal = np.random.uniform(0, 256, diagonal_shape).astype("uint8")
            in_diagonal = tf.placeholder(dtype="float32", shape=diagonal.shape, name="diagonal")
            inq_diagonal = tf.quantization.fake_quant_with_min_max_args(
                in_diagonal, min=-100, max=100, name="q_diagonal"
            )

            input_range = {"q_input": (-100, 100), "q_diagonal": (-100, 100)}

            out = array_ops.matrix_set_diag(inq_input, inq_diagonal)
            out = tf.quantization.fake_quant_with_min_max_args(out, min=-100, max=100, name="out")

            compare_tflite_with_tvm(
                [input_array, diagonal],
                ["q_input", "q_diagonal"],
                [inq_input, inq_diagonal],
                [out],
                quantized=True,
                input_range=input_range,
            )
        else:
            input_array = np.random.uniform(0, 100, input_shape).astype(input_type)
            diagonal = np.random.uniform(0, 100, diagonal_shape).astype(input_type)

            in_input = tf.placeholder(
                dtype=input_array.dtype, shape=input_array.shape, name="input"
            )
            in_diagonal = tf.placeholder(
                dtype=diagonal.dtype, shape=diagonal.shape, name="diagonal"
            )

            out = array_ops.matrix_set_diag(in_input, in_diagonal)

            compare_tflite_with_tvm(
                [input_array, diagonal], ["input", "diagonal"], [in_input, in_diagonal], [out]
            )


def test_forward_matrix_set_diag():
    """MATRIX_SET_DIAG"""
    for dtype in [np.float32, np.int32]:
        _test_matrix_set_diag((4, 4), dtype)
        _test_matrix_set_diag((5, 4, 3, 4), dtype)
        _test_matrix_set_diag((4, 4, 2), dtype)

    _test_matrix_set_diag((4, 4), np.uint8, quantized=True)
    _test_matrix_set_diag((5, 4, 3, 4), np.uint8, quantized=True)
    _test_matrix_set_diag((4, 4, 2), np.uint8, quantized=True)


#######################################################################
# MATRIX_DIAG
# -----------


def _test_matrix_diag(diagonal_shape, dtype):
    """One iteration of MATRIX_DIAG"""
    with tf.Graph().as_default():
        diagonal = np.random.uniform(0, 100, diagonal_shape).astype(dtype)
        in_diagonal = tf.placeholder(dtype=diagonal.dtype, shape=diagonal.shape, name="diagonal")

        out = array_ops.matrix_diag(in_diagonal)

        compare_tflite_with_tvm(
            [diagonal], ["diagonal"], [in_diagonal], [out], experimental_new_converter=True
        )


def test_forward_matrix_diag():
    """MATRIX_DIAG"""
    for dtype in [np.float32, np.int32]:
        _test_matrix_diag((4), dtype)
        _test_matrix_diag((5, 4, 3), dtype)
        _test_matrix_diag((2, 3), dtype)


#######################################################################
# Custom Operators
# ----------------


def _test_detection_postprocess(tf_model_file, box_encodings_size, class_predictions_size):
    """One iteration of detection postProcess with given model and shapes"""
    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        tf_model_file,
        input_arrays=["raw_outputs/box_encodings", "raw_outputs/class_predictions"],
        output_arrays=[
            "TFLite_Detection_PostProcess",
            "TFLite_Detection_PostProcess:1",
            "TFLite_Detection_PostProcess:2",
            "TFLite_Detection_PostProcess:3",
        ],
        input_shapes={
            "raw_outputs/box_encodings": box_encodings_size,
            "raw_outputs/class_predictions": class_predictions_size,
        },
    )
    converter.allow_custom_ops = True
    converter.inference_type = tf.lite.constants.FLOAT
    tflite_model = converter.convert()
    np.random.seed(0)
    box_encodings = np.random.uniform(size=box_encodings_size).astype("float32")
    class_predictions = np.random.uniform(size=class_predictions_size).astype("float32")
    tflite_output = run_tflite_graph(tflite_model, [box_encodings, class_predictions])
    tvm_output = run_tvm_graph(
        tflite_model,
        [box_encodings, class_predictions],
        ["raw_outputs/box_encodings", "raw_outputs/class_predictions"],
        num_output=4,
    )

    # Check all output shapes are equal
    assert all(
        list(
            tvm_tensor.shape == tflite_tensor.shape
            for (tvm_tensor, tflite_tensor) in zip(tvm_output, tflite_output)
        )
    )

    # Check valid count is the same
    assert tvm_output[3] == tflite_output[3]
    valid_count = tvm_output[3][0]

    # For boxes that do not have any detections, TFLite puts random values. Therefore, we compare
    # tflite and tvm tensors for only valid boxes.
    for i in range(0, valid_count):
        # Check bounding box co-ords
        tvm.testing.assert_allclose(
            np.squeeze(tvm_output[0][0][i]),
            np.squeeze(tflite_output[0][0][i]),
            rtol=1e-5,
            atol=1e-5,
        )

        # Check the class
        # Stricter check to ensure class remains same
        np.testing.assert_equal(np.squeeze(tvm_output[1][0][i]), np.squeeze(tflite_output[1][0][i]))

        # Check the score
        tvm.testing.assert_allclose(
            np.squeeze(tvm_output[2][0][i]),
            np.squeeze(tflite_output[2][0][i]),
            rtol=1e-5,
            atol=1e-5,
        )


def test_detection_postprocess():
    """Detection PostProcess"""

    # Fast-NMS
    box_encodings_size = (1, 1917, 4)
    class_predictions_size = (1, 1917, 91)
    tf_model_file = tf_testing.get_workload_official(
        "http://download.tensorflow.org/models/object_detection/"
        "ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03.tar.gz",
        "ssd_mobilenet_v2_quantized_300x300_coco_2019_01_03/tflite_graph.pb",
    )
    _test_detection_postprocess(tf_model_file, box_encodings_size, class_predictions_size)

    # Fast-NMS
    box_encodings_size = (1, 2034, 4)
    class_predictions_size = (1, 2034, 91)
    tf_model_file = download_testdata(
        "https://github.com/czh978/models_for_tvm_test/raw/main/tflite_graph_with_postprocess.pb",
        "tflite_graph_with_postprocess.pb",
    )
    _test_detection_postprocess(tf_model_file, box_encodings_size, class_predictions_size)

    # Regular NMS
    box_encodings_size = (1, 1917, 4)
    class_predictions_size = (1, 1917, 91)
    tf_model_file = download_testdata(
        (
            "https://github.com/Grovety/ModelZoo/raw/52fb82156ae8c8e3f62c7d7caf6867b25261dda4/"
            "models/object_detection/ssd_mobilenet_v1/tflite_int8/tflite_graph_with_regular_nms.pb"
        ),
        "tflite_graph_with_regular_nms.pb",
    )
    _test_detection_postprocess(tf_model_file, box_encodings_size, class_predictions_size)


#######################################################################
# Custom Converter
# ----------------


def test_custom_op_converter():
    """Test case for user-defined operator converter in TFLite frontend"""

    class DummyOperatorConverter(relay.frontend.tflite.OperatorConverter):
        """Operator Converter for converting TFLite ops to relay ops"""

        def __init__(self, model, subgraph, exp_tab):
            super().__init__(model, subgraph, exp_tab)
            self.allow_custom_ops = True

            convert_map_overwrite = {"SUB": self.convert_sub_dummy}

            self.convert_map.update(convert_map_overwrite)

        def convert_sub_dummy(self, op):
            """Convert TFLite SUB"""
            input_tensors = self.get_input_tensors(op)
            assert len(input_tensors) == 2, "input tensors length should be 2"

            lhs_tensor = input_tensors[0]
            rhs_tensor = input_tensors[1]

            lhs_expr = self.get_expr(lhs_tensor.tensor_idx)
            rhs_expr = self.get_expr(rhs_tensor.tensor_idx)

            temp_expr = relay.op.negative(rhs_expr)
            out = relay.op.add(lhs_expr, temp_expr)

            return out

    with tf.Graph().as_default():
        # Generate TFLite model for single addition
        data = [
            np.arange(6.0, dtype=np.float32).reshape((2, 1, 1, 3)),
            np.arange(1.0, 7.0, dtype=np.float32).reshape((2, 1, 1, 3)),
        ]
        in_data = [
            array_ops.placeholder(shape=data[0].shape, dtype="float32", name="in_0"),
            array_ops.placeholder(shape=data[1].shape, dtype="float32", name="in_1"),
        ]
        out = math_ops.subtract(in_data[0], in_data[1])
        in_name = [x[1] for x in zip(in_data, ("in_0:0", "in_1:0"))]
        input_tensors = in_data
        output_tensors = [out]
        in_node = [0] * len(in_name)
        for i, _ in enumerate(in_name):
            in_node[i] = in_name[i].split(":")[0]

        with tf.Session() as sess:
            converter = tf.lite.TFLiteConverter.from_session(sess, input_tensors, output_tensors)
            tflite_model_buf = converter.convert()
    in_data = [x[1] for x in zip(in_data, data)]
    tvm_output_orig = run_tvm_graph(tflite_model_buf, in_data, in_node)
    tvm_output_dummy = run_tvm_graph(
        tflite_model_buf, in_data, in_node, op_converter=DummyOperatorConverter
    )
    tvm.testing.assert_allclose(
        np.squeeze(tvm_output_orig[0]), np.squeeze(tvm_output_dummy[0]), rtol=1e-5, atol=1e-5
    )


#######################################################################
# Mobilenet
# ---------


def test_forward_mobilenet_v1():
    """Test the Mobilenet V1 TF Lite model."""
    # MobilenetV1
    tflite_model_file = tf_testing.get_workload_official(
        "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz",
        "mobilenet_v1_1.0_224.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 224, 224, 3)).astype("float32")
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, "input")
    tvm.testing.assert_allclose(
        np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-5
    )


def test_forward_mobilenet_v2():
    """Test the Mobilenet V2 TF Lite model."""
    # MobilenetV2
    tflite_model_file = tf_testing.get_workload_official(
        "http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz",
        "mobilenet_v2_1.0_224.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 224, 224, 3)).astype("float32")
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, "input")
    tvm.testing.assert_allclose(
        np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-5
    )


#######################################################################
# Mobilenet V3
# ------------


def test_forward_mobilenet_v3():
    """Test the Mobilenet V3 TF Lite model."""
    # In MobilenetV3, some ops are not supported before tf 1.15 fbs schema
    if package_version.parse(tf.VERSION) < package_version.parse("1.15.0"):
        return
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_float.tgz",
        "v3-large_224_1.0_float/v3-large_224_1.0_float.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 224, 224, 3)).astype("float32")
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, "input")
    tvm.testing.assert_allclose(
        np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-5
    )


#######################################################################
# Mobilenet V1 Sparse
# -----------------


def test_forward_sparse_mobilenet_v1():
    """Test the Sparse version of Mobilenet V1 TF Lite model."""
    # MobilenetV1
    tflite_model_file = download_testdata(
        "https://storage.googleapis.com/fast-convnets/tflite-models/mbv1_140_90_12b4_720.tflite",
        "mbv1_140_90_12b4_720.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 224, 224, 3)).astype("float32")
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, "float_image_input")
    tvm.testing.assert_allclose(
        np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-5
    )


#######################################################################
# Mobilenet V2 Sparse
# -----------------


def test_forward_sparse_mobilenet_v2():
    """Test the Sparse version of Mobilenet V2 TF Lite model."""
    # MobilenetV1
    tflite_model_file = download_testdata(
        "https://storage.googleapis.com/fast-convnets/tflite-models/mbv2_200_85_11-16b2_744.tflite",
        "mbv2_200_85_11-16b2_744.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 224, 224, 3)).astype("float32")
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, "float_image_input")
    tvm.testing.assert_allclose(
        np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-5
    )


#######################################################################
# Inception
# ---------


def test_forward_inception_v3_net():
    """Test the Inception V3 TF Lite model."""
    # InceptionV3
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/"
        "upload_20180427/inception_v3_2018_04_27.tgz",
        "inception_v3.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 299, 299, 3)).astype("float32")
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, "input")
    tvm.testing.assert_allclose(
        np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-5
    )


def test_forward_inception_v4_net():
    """Test the Inception V4 TF Lite model."""
    # InceptionV4
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/"
        "tflite/model_zoo/upload_20180427/"
        "inception_v4_2018_04_27.tgz",
        "inception_v4.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 299, 299, 3)).astype("float32")
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, "input")
    tvm.testing.assert_allclose(
        np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-5
    )


def test_forward_inception_v4_net_batched():
    """Test the Inception V4 TF Lite model."""
    # InceptionV4
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/"
        "tflite/model_zoo/upload_20180427/"
        "inception_v4_2018_04_27.tgz",
        "inception_v4.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(4, 299, 299, 3)).astype("float32")
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, "input")
    tvm.testing.assert_allclose(
        np.squeeze(tvm_output[0]), np.squeeze(tflite_output[0]), rtol=1e-5, atol=1e-5
    )


def test_forward_qnn_inception_v1_net():
    """Test the Quantized TFLite Inception model."""
    # InceptionV1
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/"
        "inception_v1_224_quant_20181026.tgz",
        "inception_v1_224_quant.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()

    # Test image. Checking the labels because the requantize implementation is different between
    # TFLite and Relay. This cause final output numbers to mismatch. So, testing accuracy via
    # labels. Also, giving a real image, instead of random inputs.
    data = get_real_image(224, 224)

    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tflite_predictions = np.squeeze(tflite_output)
    tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
    tvm_output = run_tvm_graph(tflite_model_buf, data, "input")
    tvm_predictions = np.squeeze(tvm_output)
    tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
    tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)


def test_forward_qnn_mobilenet_v1_net():
    """Test the Quantized TFLite Mobilenet V1 model."""
    # MobilenetV1
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/"
        "mobilenet_v1_1.0_224_quant.tgz",
        "mobilenet_v1_1.0_224_quant.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()

    # Test image. Checking the labels because the requantize implementation is different between
    # TFLite and Relay. This cause final output numbers to mismatch. So, testing accuracy via
    # labels. Also, giving a real image, instead of random inputs.
    data = get_real_image(224, 224)

    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tflite_predictions = np.squeeze(tflite_output)
    tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
    tvm_output = run_tvm_graph(tflite_model_buf, data, "input")
    tvm_predictions = np.squeeze(tvm_output)
    tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
    tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)


def test_forward_qnn_mobilenet_v2_net():
    """Test the Quantized TFLite Mobilenet V2 model."""
    # MobilenetV2
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/"
        "mobilenet_v2_1.0_224_quant.tgz",
        "mobilenet_v2_1.0_224_quant.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()

    # Test image. Checking the labels because the requantize implementation is different between
    # TFLite and Relay. This cause final output numbers to mismatch. So, testing accuracy via
    # labels. Also, giving a real image, instead of random inputs.
    data = get_real_image(224, 224)

    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tflite_predictions = np.squeeze(tflite_output)
    tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
    tvm_output = run_tvm_graph(tflite_model_buf, data, "input")
    tvm_predictions = np.squeeze(tvm_output)
    tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
    tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)


#######################################################################
# Mobilenet V3 Quantized
# ----------------------


def test_forward_qnn_mobilenet_v3_net():
    """Test the Quantized TFLite Mobilenet V3 model."""
    # In MobilenetV3, some ops are not supported before tf 1.15 fbs schema
    if package_version.parse(tf.VERSION) < package_version.parse("1.15.0"):
        pytest.skip("Unsupported in tflite < 1.15.0")
    else:
        pytest.skip("This segfaults with tensorflow 1.15.2 and above")

    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/mobilenet_v3/checkpoints/v3-large_224_1.0_uint8.tgz",
        "v3-large_224_1.0_uint8/v3-large_224_1.0_uint8.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()

    # Test image. Checking the labels because the requantize implementation is different between
    # TFLite and Relay. This cause final output numbers to mismatch. So, testing accuracy via
    # labels. Also, giving a real image, instead of random inputs.
    data = get_real_image(224, 224)

    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tflite_predictions = np.squeeze(tflite_output)
    tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
    tvm_output = run_tvm_graph(tflite_model_buf, data, "input")
    tvm_predictions = np.squeeze(tvm_output)
    tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
    tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)


def test_forward_tflite2_qnn_resnet50():
    """Test the Quantized TFLite version 2.1.0 Resnet50 model."""
    if package_version.parse(tf.VERSION) >= package_version.parse("2.1.0"):
        tflite_model_file = download_testdata(
            "https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/models/Quantized/"
            "resnet_50_quantized.tflite",
            "resnet_50_quantized.tflite",
        )
        with open(tflite_model_file, "rb") as f:
            tflite_model_buf = f.read()

        data = pre_processed_image(224, 224)

        tflite_output = run_tflite_graph(tflite_model_buf, data)
        tflite_predictions = np.squeeze(tflite_output)
        tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
        tvm_output = run_tvm_graph(tflite_model_buf, np.array(data), "input_1")
        tvm_predictions = np.squeeze(tvm_output)
        tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
        tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)


def test_forward_tflite2_qnn_inception_v1():
    """Test the Quantized TFLite version 2.1.0 Inception V1 model."""
    if package_version.parse(tf.VERSION) >= package_version.parse("2.1.0"):
        tflite_model_file = download_testdata(
            "https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/models/Quantized/"
            "inception_v1_quantized.tflite",
            "inception_v1_quantized.tflite",
        )
        with open(tflite_model_file, "rb") as f:
            tflite_model_buf = f.read()

        data = pre_processed_image(224, 224)

        tflite_output = run_tflite_graph(tflite_model_buf, data)
        tflite_predictions = np.squeeze(tflite_output)
        tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
        tvm_output = run_tvm_graph(tflite_model_buf, np.array(data), "input_1")
        tvm_predictions = np.squeeze(tvm_output)
        tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
        tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)


def test_forward_tflite2_qnn_mobilenet_v2():
    """Test the Quantized TFLite version 2.1.0 Mobilenet V2 model."""
    if package_version.parse(tf.VERSION) >= package_version.parse("2.1.0"):
        tflite_model_file = download_testdata(
            "https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/models/Quantized/"
            "mobilenet_v2_quantized.tflite",
            "mobilenet_v2_quantized.tflite",
        )
        with open(tflite_model_file, "rb") as f:
            tflite_model_buf = f.read()

        data = pre_processed_image(224, 224)

        tflite_output = run_tflite_graph(tflite_model_buf, data)
        tflite_predictions = np.squeeze(tflite_output)
        tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
        tvm_output = run_tvm_graph(tflite_model_buf, np.array(data), "input_1")
        tvm_predictions = np.squeeze(tvm_output)
        tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
        tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)


def test_forward_tflite_float16():
    """Test float16 quantized model"""
    # MobilenetV2
    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/"
        "mobilenet_v1_0.25_128.tgz",
        "mobilenet_v1_0.25_128_frozen.pb",
    )

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        tflite_model_file, ["input"], ["MobilenetV1/Predictions/Reshape_1"]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model_buf = converter.convert()

    # Test image. Checking the labels because the requantize implementation is different between
    # TFLite and Relay. This cause final output numbers to mismatch. So, testing accuracy via
    # labels. Also, giving a real image, instead of random inputs.
    data = get_real_image(128, 128, quantized=False)

    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tflite_predictions = np.squeeze(tflite_output)
    tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
    tvm_output = run_tvm_graph(tflite_model_buf, data, "input")
    tvm_predictions = np.squeeze(tvm_output)
    tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
    tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)


def test_forward_mobilenet_int16():
    """Test int16 quantized model"""
    # MobilenetV2
    model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/"
        "mobilenet_v1_0.25_128.tgz",
        "mobilenet_v1_0.25_128_frozen.pb",
    )

    # Test image. Checking the labels because the requantize implementation is different between
    # TFLite and Relay. This cause final output numbers to mismatch. So, testing accuracy via
    # labels. Also, giving a real image, instead of random inputs.
    #
    # According to TFLite documentation, despite the quantization being done to make this model
    # use int16 types, inputs and outputs are kept float32 by default.
    # https://www.tensorflow.org/lite/performance/post_training_integer_quant_16x8
    data = get_real_image(128, 128, quantized=False)

    converter = tf.lite.TFLiteConverter.from_frozen_graph(
        model_file, ["input"], ["MobilenetV1/Predictions/Reshape_1"]
    )

    def representative_dataset():
        for _ in range(1):
            yield [data]

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
    ]
    converter.representative_dataset = representative_dataset
    tflite_model_buf = converter.convert()

    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tflite_predictions = np.squeeze(tflite_output)
    tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
    tvm_output = run_tvm_graph(tflite_model_buf, data, "input")
    tvm_predictions = np.squeeze(tvm_output)
    tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
    tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)


def test_forward_ds_cnn_int16():
    """Test DS_CNN int16 quantized model"""
    tflite_model_file = download_testdata(
        "https://github.com/ARM-software/ML-zoo/blob/48f458af1e9065d9aad2ad94d24b58d6e7c00817/"
        "models/keyword_spotting/ds_cnn_small/tflite_int16/ds_cnn_quantized.tflite?raw=true",
        "ds_cnn_quantized_int16.tflite",
    )

    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()

    data = np.random.uniform(size=(1, 490)).astype("int16")

    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tflite_predictions = np.squeeze(tflite_output)
    tflite_sorted_labels = tflite_predictions.argsort()[-3:][::-1]
    tvm_output = run_tvm_graph(tflite_model_buf, data, "serving_default_input:0")
    tvm_predictions = np.squeeze(tvm_output)
    tvm_sorted_labels = tvm_predictions.argsort()[-3:][::-1]
    tvm.testing.assert_allclose(tvm_sorted_labels, tflite_sorted_labels)


#######################################################################
# Unidirectional Sequence LSTM
# ---------------------
def test_forward_unidirectional_sequence_lstm():
    """Test the UnidirectionalSequenceLSTM TFLite"""
    if package_version.parse(tf.VERSION) >= package_version.parse("2.1.0"):
        tflite_model_file = download_testdata(
            "https://github.com/SebastianBoblestETAS/nn_models/blob/"
            "ce49c5de64889493161ca4194a20e0fd5eb707e6/lstm_1_in_3_out_2_ts_4.tflite?raw=true",
            "lstm_1_in_3_out_2_ts_4.tflite",
        )
        with open(tflite_model_file, "rb") as f:
            tflite_model_buf = f.read()

        data = np.array(
            [
                [
                    [0.5488135, 0.71518934, 0.60276335],
                    [0.5448832, 0.4236548, 0.6458941],
                    [0.4375872, 0.891773, 0.96366274],
                    [0.3834415, 0.79172504, 0.5288949],
                ]
            ],
            dtype="float32",
        )

        tflite_output = run_tflite_graph(tflite_model_buf, data)
        tvm_output = run_tvm_graph(tflite_model_buf, data, "serving_default_input_1:0")
        tvm.testing.assert_allclose(tflite_output, tvm_output)


#######################################################################
# Quantized SSD Mobilenet
# -----------------------


def test_forward_qnn_coco_ssd_mobilenet_v1():
    """Test the quantized Coco SSD Mobilenet V1 TF Lite model."""
    pytest.skip(
        "LLVM bug - getExtendedVectorNumElements - "
        + "https://discuss.tvm.apache.org/t/segfault-in-llvm/3567. The workaround is to use a "
        + "specific target, for example, llvm -mpcu=core-avx2"
    )

    tflite_model_file = tf_testing.get_workload_official(
        "https://storage.googleapis.com/download.tensorflow.org/models/tflite/"
        "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip",
        "detect.tflite",
    )

    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()

    data = get_real_image_object_detection(300, 300)
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(
        tflite_model_buf, data, "normalized_input_image_tensor", num_output=4
    )

    # Check all output shapes are equal
    assert all(
        list(
            tvm_tensor.shape == tflite_tensor.shape
            for (tvm_tensor, tflite_tensor) in zip(tvm_output, tflite_output)
        )
    )

    # Check valid count is the same
    assert tvm_output[3] == tflite_output[3]
    valid_count = tvm_output[3][0]

    # For boxes that do not have any detections, TFLite puts random values. Therefore, we compare
    # tflite and tvm tensors for only valid boxes.
    for i in range(0, valid_count):
        # We compare the bounding boxes whose prediction score is above 60%. This is typical in end
        # to end application where a low prediction score is discarded. This is also needed because
        # multiple low score bounding boxes can have same score and TFlite and TVM can have
        # different orderings for same score bounding boxes. Another reason for minor differences in
        # low score bounding boxes is the difference between TVM and TFLite for requantize operator.
        if tvm_output[2][0][i] > 0.6:
            # Check bounding box co-ords. The tolerances have to be adjusted, from 1e-5 to 1e-2,
            # because of differences between for requantiize operator in TFLite and TVM.
            tvm.testing.assert_allclose(
                np.squeeze(tvm_output[0][0][i]),
                np.squeeze(tflite_output[0][0][i]),
                rtol=1e-2,
                atol=1e-2,
            )

            # Check the class
            # Stricter check to ensure class remains same
            np.testing.assert_equal(
                np.squeeze(tvm_output[1][0][i]), np.squeeze(tflite_output[1][0][i])
            )

            # Check the score
            tvm.testing.assert_allclose(
                np.squeeze(tvm_output[2][0][i]),
                np.squeeze(tflite_output[2][0][i]),
                rtol=1e-5,
                atol=1e-5,
            )


#######################################################################
# SSD Mobilenet
# -------------


def test_forward_coco_ssd_mobilenet_v1():
    """Test the FP32 Coco SSD Mobilenet V1 TF Lite model."""
    tflite_model_file = tf_testing.get_workload_official(
        "https://raw.githubusercontent.com/dmlc/web-data/main/tensorflow/models/object_detection/"
        "ssd_mobilenet_v1_coco_2018_01_28.tgz",
        "ssd_mobilenet_v1_coco_2018_01_28.tflite",
    )

    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()

    np.random.seed(0)
    data = np.random.uniform(size=(1, 300, 300, 3)).astype("float32")
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(
        tflite_model_buf, data, "normalized_input_image_tensor", num_output=4
    )

    # Check all output shapes are equal
    assert all(
        list(
            tvm_tensor.shape == tflite_tensor.shape
            for (tvm_tensor, tflite_tensor) in zip(tvm_output, tflite_output)
        )
    )

    # Check valid count is the same
    assert tvm_output[3] == tflite_output[3]
    valid_count = tvm_output[3][0]

    # For boxes that do not have any detections, TFLite puts random values. Therefore, we compare
    # tflite and tvm tensors for only valid boxes.
    for i in range(0, valid_count):
        # Check bounding box co-ords
        tvm.testing.assert_allclose(
            np.squeeze(tvm_output[0][0][i]),
            np.squeeze(tflite_output[0][0][i]),
            rtol=1e-5,
            atol=1e-5,
        )
        # Check the class
        np.testing.assert_equal(np.squeeze(tvm_output[1][0][i]), np.squeeze(tflite_output[1][0][i]))

        # Check the score
        tvm.testing.assert_allclose(
            np.squeeze(tvm_output[2][0][i]),
            np.squeeze(tflite_output[2][0][i]),
            rtol=1e-5,
            atol=1e-5,
        )


#######################################################################
# MediaPipe
# -------------
def test_forward_mediapipe_hand_landmark():
    """Test MediaPipe 2D hand landmark TF Lite model."""
    # MediaPipe 2D hand landmark TF
    tflite_model_file = download_testdata(
        "https://github.com/google/mediapipe/raw/v0.7.4/mediapipe/models/hand_landmark.tflite",
        "hand_landmark.tflite",
    )
    with open(tflite_model_file, "rb") as f:
        tflite_model_buf = f.read()
    data = np.random.uniform(size=(1, 256, 256, 3)).astype("float32")
    tflite_output = run_tflite_graph(tflite_model_buf, data)
    tvm_output = run_tvm_graph(tflite_model_buf, data, "input_1", num_output=2)
    for i in range(2):
        tvm.testing.assert_allclose(
            np.squeeze(tvm_output[i]), np.squeeze(tflite_output[i]), rtol=1e-5, atol=1e-5
        )


#######################################################################
# Test check for Tensorflow "dynamic range quantization" optimization
# --------------
def test_prevent_tensorflow_dynamic_range():
    """
    Should prevent running "dynamic range quantization" optimized TFLite graph
    """
    data_array = np.random.randint(0, 2, (1, 1024, 1024)).astype(dtype=np.float32)
    filter_array = np.random.randint(0, 2, (1024, 1024)).astype(dtype=np.float32)
    data_in = tf.keras.layers.Input(shape=data_array.shape[1:])
    dense = tf.keras.layers.Dense(units=filter_array.shape[-1], use_bias=False)(data_in)
    keras_model = tf.keras.models.Model(data_in, dense)
    keras_model.layers[1].set_weights([filter_array])

    converter = interpreter_wrapper.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with pytest.raises(tvm.error.OpNotImplemented):
        _ = run_tvm_graph(tflite_model, data_array, data_in.name.replace(":0", ""))


def _test_nms_v5(
    bx_shape, score_shape, iou_threshold, score_threshold, max_output_size, dtype="float32"
):
    """One iteration of nms_v5 with given attributes"""
    boxes = np.random.uniform(0, 10, size=bx_shape).astype(dtype)
    scores = np.random.uniform(size=score_shape).astype(dtype)

    tf.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    in_data_1 = array_ops.placeholder(dtype, boxes.shape, name="in_data_1")
    in_data_2 = array_ops.placeholder(dtype, scores.shape, name="in_data_2")
    out = image_ops.non_max_suppression_with_scores(
        boxes=in_data_1,
        scores=in_data_2,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        name="nms",
    )

    compare_tflite_with_tvm(
        [boxes, scores],
        ["in_data_1:0", "in_data_2:0"],
        [in_data_1, in_data_2],
        [out[0], out[1]],
        out_names=[out[0].name, out[1].name],
        experimental_new_converter=True,
    )


def test_forward_nms_v5():
    """test nms_v5"""
    _test_nms_v5((10000, 4), (10000,), 0.5, 0.4, 100)
    _test_nms_v5((1000, 4), (1000,), 0.7, 0.3, 50)


#######################################################################
# Test structural_equal and span of a model
# --------------------------------------
def test_structure_and_span():
    """Test Structure and span of frequently-used models"""

    def _verify(res_fptr, golden_fptr):
        with tvm.testing.enable_span_filling():
            with_span = res_fptr()
        with tvm.testing.disable_span_filling():
            without_span = res_fptr()
        assert tvm.ir.structural_equal(with_span, without_span)
        _verify_structural_equal_with_span(with_span, golden_fptr())

    def _tf_to_tflite(
        input_tensors, output_tensors, init_global_variables=False, experimental_new_converter=False
    ):
        with tf.Session() as sess:
            if init_global_variables:
                sess.run(variables.global_variables_initializer())
            converter = tf.lite.TFLiteConverter.from_session(sess, input_tensors, output_tensors)
            converter.experimental_new_converter = experimental_new_converter

            tflite_model_buffer = converter.convert()

        try:
            import tflite.Model

            tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buffer, 0)
        except AttributeError:
            import tflite

            tflite_model = tflite.Model.GetRootAsModel(tflite_model_buffer, 0)
        except ImportError:
            raise ImportError("The tflite package must be installed")
        return tflite_model

    def _test_conv2d_bias_add_span():
        def _res():
            in_shape = (1, 5, 5, 1)
            kernel_shpae = (2, 2, 1, 2)
            kernel_in = np.ones(kernel_shpae)

            with tf.Graph().as_default():
                x = array_ops.placeholder(shape=in_shape, dtype="float32", name="input")
                kernel = tf.constant(kernel_in, dtype=tf.float32, name="filter_weight")
                tf_model = tf.nn.conv2d(
                    x, kernel, strides=[1, 1, 1, 1], padding="VALID", name="conv2d"
                )
                tflite_model = _tf_to_tflite([x], [tf_model])

            mod, _ = relay.frontend.from_tflite(
                tflite_model,
                shape_dict={"input": in_shape},
                dtype_dict={"input": "float32"},
                op_converter=relay.frontend.tflite.OperatorConverter,
            )
            return mod["main"]

        def _golden():
            in_input = relay.var(
                "input", relay.TensorType([1, 5, 5, 1]), span=_create_span("input")
            )
            weight = relay.var(
                "_param_1", relay.TensorType([2, 2, 1, 2]), span=_create_span("filter_weight")
            )
            bias = relay.var("_param_2", relay.TensorType([2]), span=_create_span("conv2d_bias"))
            conv2d = _set_span(
                relay.nn.conv2d(
                    in_input,
                    weight,
                    channels=2,
                    kernel_size=[2, 2],
                    data_layout="NHWC",
                    kernel_layout="HWIO",
                ),
                "conv2d",
            )
            bias_add = _set_span(relay.nn.bias_add(conv2d, bias, axis=3), "conv2d")
            attrs = ir.make_node("DictAttrs", **{"output_tensor_names": ["conv2d"]})
            func = relay.Function([in_input, weight, bias], bias_add, attrs=attrs)
            mod = ir.IRModule.from_expr(func)
            return mod["main"]

        _verify(_res, _golden)

    def _test_fully_connected_bias_add_span():
        def _res():
            in_shape = (1, 10)
            kernel_shpae = (10, 10)
            kernel_in = np.ones(kernel_shpae)

            with tf.Graph().as_default():
                x = array_ops.placeholder(shape=in_shape, dtype="float32", name="input")
                weight = tf.constant(kernel_in, dtype=tf.float32, name="filter_weight")
                tf_model = math_ops.mat_mul(x, weight, name="dense")
                tflite_model = _tf_to_tflite([x], [tf_model])

            mod, _ = relay.frontend.from_tflite(
                tflite_model,
                shape_dict={"input": in_shape},
                dtype_dict={"input": "float32"},
                op_converter=relay.frontend.tflite.OperatorConverter,
            )
            return mod["main"]

        def _golden():
            in_input = relay.var("input", relay.TensorType([1, 10]), span=_create_span("input"))
            weight = relay.var(
                "_param_1", relay.TensorType([10, 10]), span=_create_span("filter_weight/transpose")
            )
            bias = relay.var("_param_2", relay.TensorType([10]), span=_create_span("dense_bias"))
            reshape = _set_span(relay.reshape(in_input, [-1, 10]), "dense")
            dense = _set_span(relay.nn.dense(reshape, weight, units=10), "dense")
            bias_add = _set_span(relay.nn.bias_add(dense, bias), "dense")
            attrs = ir.make_node("DictAttrs", **{"output_tensor_names": ["dense"]})
            func = relay.Function([in_input, weight, bias], bias_add, attrs=attrs)
            mod = ir.IRModule.from_expr(func)
            return mod["main"]

        _verify(_res, _golden)

    def _test_reshape_span():
        def _res():
            in_shape = (1, 10)
            output_shape = (2, 5)

            with tf.Graph().as_default():
                x = array_ops.placeholder(shape=in_shape, dtype="float32", name="input")
                tf_model = array_ops.reshape(x, output_shape, "reshape")
                tflite_model = _tf_to_tflite([x], [tf_model])

            mod, _ = relay.frontend.from_tflite(
                tflite_model,
                shape_dict={"input": in_shape},
                dtype_dict={"input": "float32"},
                op_converter=relay.frontend.tflite.OperatorConverter,
            )
            return mod["main"]

        def _golden():
            in_input = relay.var("input", relay.TensorType([1, 10]), span=_create_span("input"))
            reshape = _set_span(relay.reshape(in_input, [2, 5]), "reshape")
            attrs = ir.make_node("DictAttrs", **{"output_tensor_names": ["reshape"]})
            func = relay.Function([in_input], reshape, attrs=attrs)
            mod = ir.IRModule.from_expr(func)
            return mod["main"]

        _verify(_res, _golden)

    _test_conv2d_bias_add_span()
    _test_fully_connected_bias_add_span()
    _test_reshape_span()


class TestConv2d:
    """Import Conv2d operator from TFLite, build with Relay and test."""

    input_shape, kernel_shape, padding = tvm.testing.parameters(
        ((1, 128, 256, 6), (5, 5, 6, 10), "SAME"),
        ((1, 128, 256, 6), (5, 5, 6, 10), "VALID"),
        # conv2d_group cases
        ((1, 30, 40, 6), (5, 5, 1, 6), "SAME"),
        ((1, 30, 40, 6), (5, 5, 1, 6), "VALID"),
    )

    def test_conv2d(self, input_shape: tuple, kernel_shape: tuple, padding: str):
        dtype = tf.float32
        kernel_in = np.ones(kernel_shape)
        with tf.Graph().as_default():
            x = array_ops.placeholder(shape=input_shape, dtype=dtype.name, name="input")
            kernel = tf.constant(kernel_in, dtype=dtype, name="filter_weight")
            out = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding=padding, name="conv2d")
            input_data = np.random.randn(*input_shape).astype(dtype.name)
            compare_tflite_with_tvm(
                [input_data],
                ["input"],
                [x],
                [out],
            )


if __name__ == "__main__":
    tvm.testing.main()
