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
"""
.. _tutorial-micro-mlperftiny:

8. Creating Your MLPerfTiny Submission with microTVM
====================================================
**Authors**:
`Mehrdad Hessar <https://github.com/mehrdadh>`_

This tutorial is showcasing building an MLPerfTiny submission using microTVM. This
tutorial shows the steps to import a TFLite model from MLPerfTiny benchmark models,
compile it with TVM and generate a Zephyr project which can be flashed to a Zephyr
supported board to benchmark the model using EEMBC runner.
"""

######################################################################
#
#     .. include:: ../../../../gallery/how_to/work_with_microtvm/install_dependencies.rst
#

import os
import pathlib
import tarfile
import tempfile
import shutil

######################################################################
#
#     .. include:: ../../../../gallery/how_to/work_with_microtvm/install_zephyr.rst
#


######################################################################
#
# **Note:** Install CMSIS-NN only if you are interested to generate this submission
# using CMSIS-NN code generator.
#

######################################################################
#
#     .. include:: ../../../../gallery/how_to/work_with_microtvm/install_cmsis.rst
#

######################################################################
# Import Python dependencies
# -------------------------------
#
import tensorflow as tf
import numpy as np

import tvm
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.download import download_testdata
from tvm.micro import export_model_library_format
import tvm.micro.testing
from tvm.micro.testing.utils import (
    create_header_file,
    mlf_extract_workspace_size_bytes,
)

######################################################################
# Import Visual Wake Word Model
# --------------------------------------------------------------------
#
# To begin with, download and import the Visual Wake Word (VWW) TFLite model from MLPerfTiny.
# This model is originally from `MLPerf Tiny repository <https://github.com/mlcommons/tiny>`_.
# We also capture metadata information from the TFLite model such as input/output name,
# quantization parameters, etc. which will be used in following steps.
#
# We use indexing for various models to build the submission. The indices are defined as follows:
# To build another model, you need to update the model URL, the short name and index number.
#
#   * Keyword Spotting(KWS) 1
#   * Visual Wake Word(VWW) 2
#   * Anomaly Detection(AD) 3
#   * Image Classification(IC) 4
#
# If you would like to build the submission with CMSIS-NN, modify USE_CMSIS environment variable.
#
#   .. code-block:: bash
#
#     export USE_CMSIS=1
#

MODEL_URL = "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite"
MODEL_PATH = download_testdata(MODEL_URL, "vww_96_int8.tflite", module="model")

MODEL_SHORT_NAME = "VWW"
MODEL_INDEX = 2

USE_CMSIS = os.environ.get("TVM_USE_CMSIS", False)

tflite_model_buf = open(MODEL_PATH, "rb").read()
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_name = input_details[0]["name"]
input_shape = tuple(input_details[0]["shape"])
input_dtype = np.dtype(input_details[0]["dtype"]).name
output_name = output_details[0]["name"]
output_shape = tuple(output_details[0]["shape"])
output_dtype = np.dtype(output_details[0]["dtype"]).name

# We extract quantization information from TFLite model.
# This is required for all models except Anomaly Detection,
# because for other models we send quantized data to interpreter
# from host, however, for AD model we send floating data and quantization
# happens on the microcontroller.
if MODEL_SHORT_NAME != "AD":
    quant_output_scale = output_details[0]["quantization_parameters"]["scales"][0]
    quant_output_zero_point = output_details[0]["quantization_parameters"]["zero_points"][0]

relay_mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_name: input_shape}, dtype_dict={input_name: input_dtype}
)

######################################################################
# Defining Target, Runtime and Executor
# --------------------------------------------------------------------
#
# Now we need to define the target, runtime and executor to compile this model. In this tutorial,
# we use Ahead-of-Time (AoT) compilation and we build a standalone project. This is different
# than using AoT with host-driven mode where the target would communicate with host using host-driven
# AoT executor to run inference.
#

# Use the C runtime (crt)
RUNTIME = Runtime("crt")

# Use the AoT executor with `unpacked-api=True` and `interface-api=c`. `interface-api=c` forces
# the compiler to generate C type function APIs and `unpacked-api=True` forces the compiler
# to generate minimal unpacked format inputs which reduces the stack memory usage on calling
# inference layers of the model.
EXECUTOR = Executor(
    "aot",
    {"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 8},
)

# Select a Zephyr board
BOARD = os.getenv("TVM_MICRO_BOARD", default="nucleo_l4r5zi")

# Get the full target description using the BOARD
TARGET = tvm.micro.testing.get_target("zephyr", BOARD)

######################################################################
# Compile the model and export model library format
# --------------------------------------------------------------------
#
# Now, we compile the model for the target. Then, we generate model
# library format for the compiled model. We also need to calculate the
# workspace size that is required for the compiled model.
#
#

config = {"tir.disable_vectorize": True}
if USE_CMSIS:
    from tvm.relay.op.contrib import cmsisnn

    config["relay.ext.cmsisnn.options"] = {"mcpu": TARGET.mcpu}
    relay_mod = cmsisnn.partition_for_cmsisnn(relay_mod, params, mcpu=TARGET.mcpu)

with tvm.transform.PassContext(opt_level=3, config=config):
    module = tvm.relay.build(
        relay_mod, target=TARGET, params=params, runtime=RUNTIME, executor=EXECUTOR
    )

temp_dir = tvm.contrib.utils.tempdir()
model_tar_path = temp_dir / "model.tar"
export_model_library_format(module, model_tar_path)
workspace_size = mlf_extract_workspace_size_bytes(model_tar_path)

######################################################################
# Generate input/output header files
# --------------------------------------------------------------------
#
# To create a microTVM standalone project with AoT, we need to generate
# input and output header files. These header files are used to connect
# the input and output API from generated code to the rest of the
# standalone project. For this specific submission, we only need to generate
# output header file since the input API call is handled differently.
#

extra_tar_dir = tvm.contrib.utils.tempdir()
extra_tar_file = extra_tar_dir / "extra.tar"

with tarfile.open(extra_tar_file, "w:gz") as tf:
    create_header_file(
        "output_data",
        np.zeros(
            shape=output_shape,
            dtype=output_dtype,
        ),
        "include/tvm",
        tf,
    )

######################################################################
# Create the project, build and prepare the project tar file
# --------------------------------------------------------------------
#
# Now that we have the compiled model as a model library format,
# we can generate the full project using Zephyr template project. First,
# we prepare the project options, then build the project. Finally, we
# cleanup the temporary files and move the submission project to the
# current working directory which could be downloaded and used on
# your development kit.
#

input_total_size = 1
for i in range(len(input_shape)):
    input_total_size *= input_shape[i]

template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr"))
project_options = {
    "extra_files_tar": str(extra_tar_file),
    "project_type": "mlperftiny",
    "board": BOARD,
    "compile_definitions": [
        f"-DWORKSPACE_SIZE={workspace_size + 512}",  # Memory workspace size, 512 is a temporary offset
        # since the memory calculation is not accurate.
        f"-DTARGET_MODEL={MODEL_INDEX}",  # Sets the model index for project compilation.
        f"-DTH_MODEL_VERSION=EE_MODEL_VERSION_{MODEL_SHORT_NAME}01",  # Sets model version. This is required by MLPerfTiny API.
        f"-DMAX_DB_INPUT_SIZE={input_total_size}",  # Max size of the input data array.
    ],
}

if MODEL_SHORT_NAME != "AD":
    project_options["compile_definitions"].append(f"-DOUT_QUANT_SCALE={quant_output_scale}")
    project_options["compile_definitions"].append(f"-DOUT_QUANT_ZERO={quant_output_zero_point}")

if USE_CMSIS:
    project_options["compile_definitions"].append(f"-DCOMPILE_WITH_CMSISNN=1")

# Note: You might need to adjust this based on the board that you are using.
project_options["config_main_stack_size"] = 4000

if USE_CMSIS:
    project_options["cmsis_path"] = os.environ.get("CMSIS_PATH", "/content/cmsis")

generated_project_dir = temp_dir / "project"

project = tvm.micro.project.generate_project_from_mlf(
    template_project_path, generated_project_dir, model_tar_path, project_options
)
project.build()

# Cleanup the build directory and extra artifacts
shutil.rmtree(generated_project_dir / "build")
(generated_project_dir / "model.tar").unlink()

project_tar_path = pathlib.Path(os.getcwd()) / "project.tar"
with tarfile.open(project_tar_path, "w:tar") as tar:
    tar.add(generated_project_dir, arcname=os.path.basename("project"))

print(f"The generated project is located here: {project_tar_path}")

######################################################################
# Use this project with your board
# --------------------------------------------------------------------
#
# Now that we have the generated project, you can use this project locally
# to flash your board and prepare it for EEMBC runner software.
# To do this follow these steps:
#
#   .. code-block:: bash
#
#     tar -xf project.tar
#     cd project
#     mkdir build
#     cmake ..
#     make -j2
#     west flash
#
# Now you can connect your board to EEMBC runner using this
# `instructions <https://github.com/eembc/energyrunner>`_
# and benchmark this model on your board.
#
