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
Micro TVM with TFLite Models
============================
**Author**: `Tom Gall <https://github.com/tom-gall>`_

This tutorial is an introduction to working with MicroTVM and a TFLite
model with Relay.
"""

# %%
# Setup
# -----
#
# To get started, TFLite package needs to be installed as prerequisite.
#
# install tflite
#
# .. code-block:: bash
#
#   pip install tflite=2.1.0 --user
#
# or you could generate TFLite package yourself. The steps are the following:
#
#   Get the flatc compiler.
#   Please refer to https://github.com/google/flatbuffers for details
#   and make sure it is properly installed.
#
# .. code-block:: bash
#
#   flatc --version
#
# Get the TFLite schema.
#
# .. code-block:: bash
#
#   wget https://raw.githubusercontent.com/tensorflow/tensorflow/r1.13/tensorflow/lite/schema/schema.fbs
#
# Generate TFLite package.
#
# .. code-block:: bash
#
#   flatc --python schema.fbs
#
# Add the current folder (which contains generated tflite module) to PYTHONPATH.
#
# .. code-block:: bash
#
#   export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$(pwd)
#
# To validate that the TFLite package was installed successfully, ``python -c "import tflite"``
#
# CMSIS needs to be downloaded and the CMSIS_ST_PATH environment variable setup
# This tutorial only supports the STM32F7xx series of boards.
# Download from : https://www.st.com/en/embedded-software/stm32cubef7.html
# After you've expanded the zip file
#
# .. code-block:: bash
#
#   export CMSIS_ST_PATH=/path/to/STM32Cube_FW_F7_V1.16.0/Drivers/CMSIS

# %%
# Recreating your own Pre-Trained TFLite model
# --------------------------------------------
#
# The tutorial downloads a pretrained TFLite model. When working with microcontrollers
# you need to be mindful these are highly resource constrained devices as such standard
# models like MobileNet may not fit into their modest memory.
#
# For this tutorial, we'll make use of one of the TF Micro example models.
#
# If you wish to replicate the training steps see:
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/hello_world/train
#
#   .. note::
#
#     If you accidentally download the example pretrained model from:
#     wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/micro/hello_world_2020_04_13.zip
#     this will fail due to an unimplemented opcode (114)

import os
import numpy as np
import tvm
import tvm.micro as micro
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_runtime, util
from tvm import relay

# %%
# Load and prepare the Pre-Trained Model
# --------------------------------------
#
# Load the pretrained TFLite model from a file in your current
# directory into a buffer

model_url = "https://people.linaro.org/~tom.gall/sine_model.tflite"
model_file = "sine_model.tflite"
model_path = download_testdata(model_url, model_file, module="data")

tflite_model_buf = open(model_path, "rb").read()

######################################################################
# Using the buffer, transform into a tflite model python object
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

######################################################################
# Print out the version of the model
version = tflite_model.Version()
print("Model Version: " + str(version))

######################################################################
# Parse the python model object to convert it into a relay module
# and weights.
# It is important to note that the input tensor name must match what
# is contained in the model.
#
# If you are unsure what that might be, this can be discovered by using
# the visualize.py script within the Tensorflow project.
# See : How do I inspect a .tflite file? `<https://www.tensorflow.org/lite/guide/faq>`_

input_tensor = "dense_4_input"
input_shape = (1,)
input_dtype = "float32"

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)

######################################################################
# Now we create a build config for relay. turning off two options
# and then calling relay.build which will result in a C source
# file.
#
# .. code-block:: python
#
TARGET = tvm.target.target.micro("host")

with tvm.transform.PassContext(
    opt_level=3, config={"tir.disable_vectorize": True}, disabled_pass=["FuseOps"]
):
    graph, c_mod, c_params = relay.build(mod, target=TARGET, params=params)


# %%
# Running on simulated device
# ----------------------------------------------
#
# First, compile a static microTVM runtime for the targeted device. In this case, the host simulated
# device is used.
workspace = tvm.micro.Workspace()

compiler = tvm.micro.DefaultCompiler(target=TARGET)
opts = tvm.micro.default_options(os.path.join(tvm.micro.CRT_ROOT_DIR, "host"))

micro_binary = tvm.micro.build_static_runtime(
    # the x86 compiler *expects* you to give the exact same dictionary for both
    # lib_opts and bin_opts. so the library compiler is mutating lib_opts and
    # the binary compiler is expecting those mutations to be in bin_opts.
    # TODO(weberlo) fix this very bizarre behavior
    workspace,
    compiler,
    c_mod,
    lib_opts=opts["bin_opts"],
    bin_opts=opts["bin_opts"],
)


######################################################################
# Next, establish a session with the simulated device and run the
# computation. The `with session` line would typically flash an attached
# microcontroller, but in this tutorial, it simply launches a subprocess
# to stand in for an attached microcontroller.
#
# .. code-block:: python
#
flasher = compiler.flasher()
with tvm.micro.Session(binary=micro_binary, flasher=flasher) as session:
    graph_mod = tvm.micro.create_local_graph_runtime(
        graph, session.get_system_lib(), session.context
    )

    # Set the model parameters using the lowered parameters produced by `relay.build`.
    graph_mod.set_input(**c_params)

    # The model consumes a single float32 value and returns a predicted sine value.  To pass the
    # input value we construct a tvm.nd.array object with a single contrived number as input. For
    # this model values of 0 to 2Pi are acceptable.
    graph_mod.set_input(input_tensor, tvm.nd.array(np.array([0.5], dtype="float32")))
    graph_mod.run()

    tvm_output = graph_mod.get_output(0).asnumpy()
    print("result is: " + str(tvm_output))
