#!/usr/bin/env python3

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

"""Compile TFLite model for the STM32 target."""

import os
import shutil
import sys
import argparse
import tarfile

#
# Disable GPU usage information:
#
#
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np

import re

import tensorflow as tf

import tvm
import tvm.relay as relay
from tvm.micro.contrib import stm32

import logging

# ========================================================
#   get_tflite_model
# ========================================================
def get_tflite_model(model_path):

    #
    # Load TFLite model and allocate tensors.
    #
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    #
    # Get input and output tensors.
    #
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #
    # Figure out shapes and
    #

    shape_dict = {}
    dtype_dict = {}

    for input in input_details:
        input_name = input["name"]
        input_shape = input["shape"].tolist()
        input_dtype = str(np.dtype(input["dtype"]))
        shape_dict[input_name] = input_shape
        dtype_dict[input_name] = input_dtype

    #
    # Load the TFLite Model for TVM:
    #
    # https://docs.tvm.ai/tutorials/frontend/from_tflite.html
    # https://jackwish.net/tflite/docs/

    model_buf = open(model_path, "rb").read()

    #
    # Get TFLite model from buffer
    #
    try:
        import tflite

        model = tflite.Model.GetRootAsModel(model_buf, 0)
        assert isinstance(model, tflite.Model)
    except AttributeError:
        import tflite.Model

        model = tflite.Model.Model.GetRootAsModel(model_buf, 0)
        assert isinstance(model, tflite.Model.Model)

    print("TVM: Importing a TFLite model ...")

    return model, shape_dict, dtype_dict


# ========================================================
#   extract_tflite_quantization
# ========================================================


def _make_qnn_params(quantization):
    qnn_params = {}
    qnn_params["min"] = quantization.MinAsNumpy()
    qnn_params["max"] = quantization.MaxAsNumpy()
    qnn_params["scale"] = quantization.ScaleAsNumpy()
    qnn_params["zero_point"] = quantization.ZeroPointAsNumpy()
    qnn_params["dim"] = quantization.QuantizedDimension()
    # print("  Quantization: ({}, {}), s={}, z={}, dim={}".format(min, max, scale, zero_point, dim))
    return qnn_params


def extract_tflite_quantization(model):

    assert model.SubgraphsLength() == 1, "only support one subgraph (main subgraph)"

    subgraph = model.Subgraphs(0)

    quantization_info = {}

    # model inputs / outputs
    model_inputs = subgraph.InputsAsNumpy()
    model_outputs = subgraph.OutputsAsNumpy()

    for node_id in model_inputs:
        tensor = subgraph.Tensors(node_id)
        tensor_name = tensor.Name().decode("utf-8")
        tensor_type = tensor.Type()
        # print("== Input[{}]: {} shape={} type={}".format(node_id, tensor_name, tensor.ShapeAsNumpy(), tensor_type))
        dl_tensor_name = stm32.get_input_tensor_name(tensor_name)

        quantization = tensor.Quantization()
        if quantization is not None:
            qnn_params = _make_qnn_params(quantization)
            quantization_info[dl_tensor_name] = qnn_params

    for node_id in model_outputs:
        tensor = subgraph.Tensors(node_id)
        tensor_name = tensor.Name().decode("utf-8")
        tensor_type = tensor.Type()
        # print("== Output[{}]: {} shape={} type={}".format(node_id, tensor_name, tensor.ShapeAsNumpy(), tensor_type))
        #
        # TODO: TVM does not preserve the output tensor names.
        #       Eventually, we should be able to form a valid name.
        #
        dl_tensor_name = stm32.get_output_tensor_name(tensor_name, 0)

        quantization = tensor.Quantization()
        if quantization is not None:
            qnn_params = _make_qnn_params(quantization)
            quantization_info[dl_tensor_name] = qnn_params

    return quantization_info


# ==================================================================
#   __main__
# ==================================================================
if __name__ == "__main__":

    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

    parser = argparse.ArgumentParser()

    parser.add_argument("-model", type=str, required=True, help="The TFLite model to compile")
    parser.add_argument("-name", type=str, help="The name for the generated implementation")
    parser.add_argument(
        "-target-dir", type=str, help="The directory for storing the generated implementation"
    )

    args = parser.parse_args()

    model_path = args.model
    target_dir = args.target_dir

    #
    # Extract the model name
    #
    model_file = os.path.basename(model_path)
    print("=== TVM: Model name: {}".format(model_file))
    model_file_ext = os.path.splitext(model_file)
    assert model_file_ext[1] == ".tflite"

    if not args.name:
        model_name = model_file_ext[0]
    else:
        model_name = args.name

    if not target_dir:
        target_dir = model_file_ext[0] + "_gen"

    if os.path.exists(target_dir):
        print(f'Removing existing "{target_dir}" directory')
        try:
            shutil.rmtree(target_dir)
        except OSError as err:
            raise ValueError(f"emit_code.Error: {target_dir} : {err.strerror}")

    # Make a new one
    os.makedirs(target_dir)

    model, shape_dict, dtype_dict = get_tflite_model(model_path)

    #
    # Import the model with relay
    #

    print("=== TVM: Importing the model.")

    mod, params = relay.frontend.from_tflite(model, shape_dict, dtype_dict)

    print("=== TVM: Compiling the TFLite model ...")

    #
    # Build a TVM C module for the ARM CPU (without compiling the kernels
    # library to the object code form):
    #
    target = "c -device=arm_cpu"
    opt_level = 3

    with tvm.transform.PassContext(opt_level=opt_level, config={"tir.disable_vectorize": True}):
        rt_module = relay.build(mod, target=target, params=params)
        #
        # Export model library format
        #
        mlf_tar_path = os.path.join(target_dir, model_name + "_lib.tar")
        import tvm.micro as micro

        micro.export_model_library_format(rt_module, mlf_tar_path)

    #
    # Instantiate the STM32 code emitter.
    #  May take 3 optional arguments:
    #   - include_activations : activations area is statically allocated with *.nn_data_act section (default: True)
    #   - include_inputs      : input area is statically allocated with *.nn_data_act section sharing memory with activation buffers (default: True)
    #   - include_outputs     : output area is statically allocated with *.nn_data_act section sharing memory with activation buffers (default: True)
    #
    emitter = stm32.CodeEmitter()
    #
    # Extract model's inputs/outputs quantization info.
    #  NOTE: TVM does not provide this -- workaround.
    #  Prepare the quantization info if network is quantized
    #  builds a dictionary of the form:
    #
    #  {
    #   "tensor-name>": {'min':fp, 'max':fp, 'scale':fp, 'zero_point':int, 'dim':int}
    #   "tensor-name>": {'min':fp, 'max':fp, 'scale':fp, 'zero_point':int, 'dim':int}
    #   ...
    #   "tensor-name>": {'min':fp, 'max':fp, 'scale':fp, 'zero_point':int, 'dim':int}
    #  }
    #
    quantization = extract_tflite_quantization(model)

    print("== Quantization: {}".format(quantization))

    #
    # Initialize the emiiter: use the LibraryModuleFormat
    #
    # emitter.parse_model (rt_module, quantization)
    emitter.parse_library_format(mlf_tar_path, quantization)

    #
    # Emit the C code
    #
    emitter.emit_code(target_dir, model_name)

    print("=== TVM: TFLite model compiled.")
