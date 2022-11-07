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
"""microTVM cares a lot about the convolution + bias + requantize + fused ReLU use case. There have
been some accuracy issues in the past, so this test steps through a model (MobileNetV1) layer by
layer and ensures there is 1-1 correspondance at each step. This test would run way faster if we ran
the model all at once, but then we wouldn't know which layers had issues."""

import numpy as np
from PIL import Image
import pytest
import tensorflow as tf
import time

import tvm
import tvm.testing
from tvm import meta_schedule, relay
from tvm.testing.aot import AOTTestModel, run_and_check, AOTCompiledTestModel
from tvm.relay.backend import Executor, Runtime
from tvm.micro.testing.aot_test_utils import AOT_CORSTONE300_RUNNER
from tvm.contrib.download import download_testdata
from test_generalized_conv2d import change_ndarray_layout

MODEL_URL = "https://github.com/mlcommons/tiny/raw/master/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite"
SAMPLE_URL = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/elephant-299.jpg"

@pytest.fixture(scope="module")
def interpreter(request):
    # Download the reference model
    rel_model_path = f"model_microtvm_mobilenetv1.tflite"
    file = download_testdata(MODEL_URL, rel_model_path, overwrite=False)

    # Load it into TensorFlow and allocate memory
    interpreter = tf.lite.Interpreter(file, experimental_preserve_all_tensors=True)
    interpreter.allocate_tensors()

    # Download an image. The neuron activations are very strange if we use random data or ones,
    # so downloading an image is necessary.
    rel_image_path = f"image_microtvm_mobilenetv1.jpg"
    img_path = download_testdata(SAMPLE_URL, rel_image_path, overwrite=False)
    image = Image.open(img_path).resize((96, 96))
    image_data_hwc_uint8 = np.asarray(image)
    assert image_data_hwc_uint8.shape == (96, 96, 3)
    assert image_data_hwc_uint8.dtype == "uint8"
    image_data_nhwc_int8 = (image_data_hwc_uint8 + 128).view("int8").reshape((1, 96, 96, 3))

    # Load the image into the TFLite interpreter and compute all intermediate tensor values
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], image_data_nhwc_int8)
    interpreter.invoke()
    return interpreter


def _get_layer_attributes(layer_num):
    """Returns the relevant schedule, data type, padding, and stride for a given layer in a
    MobileNetV1 model. It's a huge headache to read this data from TensorFlow, as it is not user
    accessible via the interpreter. If we really wanted to, we would have to parse the .tflite file
    ourselves. This function is a bit of a hack, but lets us skip that."""

    if layer_num == 0: # Regular conv2d
        return (None, "int16", (1, 1, 0, 0), (2, 2))
    elif layer_num % 2 == 0: # 1x1 conv2d
        return (None, "int16", (1, 1, 1, 1), (1, 1))
    elif layer_num in [3, 7, 11, 23]: # Downsizing depthwise_conv2d layers
        return (None, "int16", (1, 1, 0, 0), (2, 2))
    else: # Depthwise conv2d
        return (None, "int16", (1, 1, 1, 1), (1, 1))


def _get_relu_activation_prefix(layer_num):
    if layer_num == 0:
        return "model/activation/Relu;"
    else:
        return f"model/activation_{layer_num}/Relu;"


def _get_main_path_tensor_details(details, tensor_num):
    """A "main path" tensor is a fused layer input/output. Gets the tensor details from the tensor
    index, where 0 gives the original input tensor, 1 gives the output of the first fused
    convolution layer, and so on. TFLite names are a little wack, so we get this information by
    finding the SECOND tensor (which has the suffix "1") for each ReLU activation (the first tensor
    is the bias)."""

    if tensor_num == 0:
        return details[0]
    prefix = _get_relu_activation_prefix(tensor_num - 1)
    detail = next(d for d in details if d["name"].startswith(prefix) and d["name"].endswith("1"))
    assert len(detail["shape"]) == 4
    assert detail["dtype"] == np.int8
    return detail


def _get_bias_details(details, layer_num):
    """Gets the tensor details for the bias tensor for the corresponding convolution layer. The
    bias tensors always appear before the main path tensors, so we don't have to check the ending to
    make sure we have the right one."""
    prefix = _get_relu_activation_prefix(layer_num)
    detail = next(d for d in details if d["name"].startswith(prefix))
    assert len(detail["shape"]) == 1
    assert detail["dtype"] == np.int32
    return detail


def _get_kernel_details(details, layer_num):
    """Gets the tensor details for the kernel tensor for the corresponding convolution layer. These
    have a different naming scheme from the main path and bias tensors, as they are converted before
    activation function fusion. Note that regular vs depthwise conv2ds have different prefixes."""

    if layer_num == 0:
        prefix = "model/conv2d/Conv2D"
    elif layer_num % 2 == 0:
        prefix = f"model/conv2d_{layer_num // 2}/"
    else:
        prefix = f"model/batch_normalization_{layer_num}/"

    detail = next(d for d in details if d["name"].startswith(prefix))
    assert len(detail["shape"]) == 4
    assert detail["dtype"] == np.int8
    return detail


def _get_quant_scale_const(quantization_dict, as_scalar=False):
    scales = quantization_dict["scales"]
    if as_scalar:
        assert len(scales) == 1
        scales = scales[0]
    return relay.const(scales, "float32")


def _get_quant_zp_const(quantization_dict, as_scalar = False):
    zero_points = quantization_dict["zero_points"]
    if as_scalar:
        assert len(zero_points) == 1
        zero_points = zero_points[0]
    return relay.const(zero_points, "int32")





@pytest.mark.parametrize("layer", range(2, 3))
@tvm.testing.requires_corstone300
def test_qnn_conv2d_mobilenetv1_layer(layer, interpreter):
    schedule_name, dtype, padding, strides = _get_layer_attributes(layer)
    """Load the input, kernel, bias, and generated output from each layer when it was run by the
    TensorFlow TFLite interpreter. The tensor values are quantized (though note that biases_tensor
    is an int32), while the quantization data is not. Note the zero points are zero everywhere
    except between layers."""
    tensor_details = interpreter.get_tensor_details()
    def lookup(detail):
        return interpreter.get_tensor(detail["index"]), detail["quantization_parameters"]
    inputs_tensor, inputs_quant = lookup(_get_main_path_tensor_details(tensor_details, layer))
    kernel_tensor, kernel_quant = lookup(_get_kernel_details(tensor_details, layer))
    biases_tensor, biases_quant = lookup(_get_bias_details(tensor_details, layer))
    output_tensor, output_quant = lookup(_get_main_path_tensor_details(tensor_details, layer + 1))
    out_channel_multiplier, kernel_h, kernel_w, in_channels = kernel_tensor.shape

    # Reshape tensors to match the layouts we will see after legalization
    if layer % 2 == 0: # Regular conv2d
        new_inputs_layout, new_kernel_layout, new_output_layout = "NHWC", "OHWI", "NHWC"
    else: # Depthwise conv2d
        new_inputs_layout, new_kernel_layout, new_output_layout = "NCHW", "OIHW", "NCHW"
    inputs_ndarr = change_ndarray_layout(inputs_tensor, "NHWC", new_inputs_layout).astype(dtype)
    kernel_ndarr = change_ndarray_layout(kernel_tensor, "OHWI", new_kernel_layout).astype(dtype)
    output_ndarr = change_ndarray_layout(output_tensor, "NHWC", new_output_layout).astype(dtype)

    """Construct our Relay function out of a qnn.conv2d, bias_add, and qnn.requantize. These will be
    fused into a single schedule by te_compiler_cache.cc."""
    input_var = relay.var("input", relay.TensorType(inputs_ndarr.shape, dtype))
    convolution = relay.qnn.op.conv2d(
        input_var,
        relay.const(kernel_ndarr, dtype),
        input_zero_point=_get_quant_zp_const(inputs_quant, as_scalar=True),
        kernel_zero_point=_get_quant_zp_const(kernel_quant),
        input_scale=_get_quant_scale_const(inputs_quant, as_scalar=True),
        kernel_scale=_get_quant_scale_const(kernel_quant),
        kernel_size=(kernel_h, kernel_w),
        data_layout=new_inputs_layout,
        kernel_layout=new_kernel_layout,

        dilation=(1, 1),
        strides=strides,
        padding=padding,
        groups=(1 if layer % 2 == 0 else in_channels),
        channels=(out_channel_multiplier if layer % 2 == 0 else in_channels),
        out_dtype="int32",
    )

    biased_convolution = relay.op.nn.bias_add(
        convolution,
        relay.const(biases_tensor, "int32"),
        axis=3,
    )

    output = relay.qnn.op.requantize(
        biased_convolution,
        _get_quant_scale_const(biases_quant),
        _get_quant_zp_const(biases_quant),
        _get_quant_scale_const(output_quant, as_scalar=True),
        _get_quant_zp_const(output_quant, as_scalar=True),
        axis=3,
        compute_dtype="int64",
        out_dtype=dtype,
    )

    test_function = relay.Function([input_var], output)
    test_model = AOTTestModel(
        module=tvm.IRModule.from_expr(test_function),
        inputs={"input": inputs_ndarr},
        outputs={"output": output_ndarr},
    )
    print(test_model.params)

    target = tvm.target.Target("c -keys=arm_cpu -mcpu=cortex-m7")
    runtime = Runtime("crt")
    executor = Executor(
        "aot",
        {
            "workspace-byte-alignment": 8,
            "constant-byte-alignment": 8,
            "interface-api": "c",
            "unpacked-api": True,
        },
    )


    # There should only be one operator
    def schedule_fn(sch):
        print(sch.mod.attrs["task_name"])
        assert "fused_qnn_conv2d_add_qnn_requantize" in sch.mod.attrs["task_name"]
        tvm.topi.arm_cpu.schedule_qnn_conv2d(sch)
        #import pdb
        #pdb.set_trace()

        return True


    with tvm.transform.PassContext(
        opt_level=3,
        config={
            "tir.disable_vectorize": True,
            "relay.backend.use_meta_schedule": True,
            "relay.backend.tir_converter": "allow_extern",
        },
        disabled_pass=["qnn.Legalize"]
    ), meta_schedule.database.ScheduleFnDatabase(schedule_fn):
        executor_factory = tvm.relay.build(
            test_model.module,
            target,
            executor=executor,
            runtime=runtime,
            params=test_model.params,
            mod_name=test_model.name,
        )
        compiled = AOTCompiledTestModel(model=test_model, executor_factory=executor_factory)

    run_and_check(
        models=[compiled],
        runner=AOT_CORSTONE300_RUNNER,
        interface_api="c",
        workspace_byte_alignment=8,
        constant_byte_alignment=8,
    )
