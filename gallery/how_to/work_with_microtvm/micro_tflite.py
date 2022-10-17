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
.. _microTVM-with-TFLite:

microTVM with TFLite Models
===========================
**Author**: `Tom Gall <https://github.com/tom-gall>`_

This tutorial is an introduction to working with microTVM and a TFLite
model with Relay.
"""

# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore

######################################################################
# .. note::
#     If you want to run this tutorial on the microTVM Reference VM, download the Jupyter
#     notebook using the link at the bottom of this page and save it into the TVM directory. Then:
#
#     #. Login to the reference VM with a modified ``vagrant ssh`` command:
#
#         ``$ vagrant ssh -- -L8888:localhost:8888``
#
#     #. Install jupyter:  ``pip install jupyterlab``
#     #. ``cd`` to the TVM directory.
#     #. Install tflite: poetry install -E importer-tflite
#     #. Launch Jupyter Notebook: ``jupyter notebook``
#     #. Copy the localhost URL displayed, and paste it into your browser.
#     #. Navigate to saved Jupyter Notebook (``.ipynb`` file).
#
#
# Setup
# -----
#
# Install TFLite
# ^^^^^^^^^^^^^^
#
# To get started, TFLite package needs to be installed as prerequisite. You can do this in two ways:
#
# 1. Install tflite with ``pip``
#
#     .. code-block:: bash
#
#       pip install tflite=2.1.0 --user
#
# 2. Generate the TFLite package yourself. The steps are the following:
#
#     Get the flatc compiler.
#     Please refer to https://github.com/google/flatbuffers for details
#     and make sure it is properly installed.
#
#     .. code-block:: bash
#
#       flatc --version
#
#     Get the TFLite schema.
#
#     .. code-block:: bash
#
#       wget https://raw.githubusercontent.com/tensorflow/tensorflow/r1.13/tensorflow/lite/schema/schema.fbs
#
#     Generate TFLite package.
#
#     .. code-block:: bash
#
#       flatc --python schema.fbs
#
#     Add the current folder (which contains generated tflite module) to PYTHONPATH.
#
#     .. code-block:: bash
#
#       export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$(pwd)
#
# To validate that the TFLite package was installed successfully, ``python -c "import tflite"``
#
# Install Zephyr (physical hardware only)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# When running this tutorial with a host simulation (the default), you can use the host ``gcc`` to
# build a firmware image that simulates the device. When compiling to run on physical hardware, you
# need to install a *toolchain* plus some target-specific dependencies. microTVM allows you to
# supply any compiler and runtime that can launch the TVM RPC server, but to get started, this
# tutorial relies on the Zephyr RTOS to provide these pieces.
#
# You can install Zephyr by following the
# `Installation Instructions <https://docs.zephyrproject.org/latest/getting_started/index.html>`_.
#
# Aside: Recreating your own Pre-Trained TFLite model
#  The tutorial downloads a pretrained TFLite model. When working with microcontrollers
#  you need to be mindful these are highly resource constrained devices as such standard
#  models like MobileNet may not fit into their modest memory.
#
#  For this tutorial, we'll make use of one of the TF Micro example models.
#
#  If you wish to replicate the training steps see:
#  https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/hello_world/train
#
#    .. note::
#
#      If you accidentally download the example pretrained model from:
#
#      ``wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/micro/hello_world_2020_04_13.zip``
#
#      this will fail due to an unimplemented opcode (114)
#
# Load and prepare the Pre-Trained Model
# --------------------------------------
#
# Load the pretrained TFLite model from a file in your current
# directory into a buffer

import os
import json
import tarfile
import pathlib
import tempfile
import numpy as np

import tvm
from tvm import relay
import tvm.contrib.utils
from tvm.contrib.download import download_testdata

use_physical_hw = bool(os.getenv("TVM_MICRO_USE_HW"))
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
# the ``visualize.py`` script within the Tensorflow project.
# See `How do I inspect a .tflite file? <https://www.tensorflow.org/lite/guide/faq>`_

input_tensor = "dense_4_input"
input_shape = (1,)
input_dtype = "float32"

mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype}
)

######################################################################
# Defining the target
# -------------------
#
# Now we create a build config for relay, turning off two options and then calling relay.build which
# will result in a C source file for the selected TARGET. When running on a simulated target of the
# same architecture as the host (where this Python script is executed) choose "host" below for the
# TARGET, the C Runtime as the RUNTIME and a proper board/VM to run it (Zephyr will create the right
# QEMU VM based on BOARD. In the example below the x86 arch is selected and a x86 VM is picked up accordingly:
#
RUNTIME = tvm.relay.backend.Runtime("crt", {"system-lib": True})
TARGET = tvm.target.target.micro("host")

#
# Compiling for physical hardware
#  When running on physical hardware, choose a TARGET and a BOARD that describe the hardware. The
#  STM32F746 Nucleo target and board is chosen in the example below. Another option would be to
#  choose the STM32F746 Discovery board instead. Since that board has the same MCU as the Nucleo
#  board but a couple of wirings and configs differ, it's necessary to select the "stm32f746g_disco"
#  board to generated the right firmware image.
#

if use_physical_hw:
    boards_file = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr")) / "boards.json"
    with open(boards_file) as f:
        boards = json.load(f)

    BOARD = os.getenv("TVM_MICRO_BOARD", default="nucleo_f746zg")
    TARGET = tvm.target.target.micro(boards[BOARD]["model"])

#
#  For some boards, Zephyr runs them emulated by default, using QEMU. For example, below is the
#  TARGET and BOARD used to build a microTVM firmware for the mps2-an521 board. Since that board
#  runs emulated by default on Zephyr the suffix "-qemu" is added to the board name to inform
#  microTVM that the QEMU transporter must be used to communicate with the board. If the board name
#  already has the prefix "qemu_", like "qemu_x86", then it's not necessary to add that suffix.
#
#  TARGET = tvm.target.target.micro("mps2_an521")
#  BOARD = "mps2_an521-qemu"

######################################################################
# Now, compile the model for the target:

with tvm.transform.PassContext(
    opt_level=3, config={"tir.disable_vectorize": True}, disabled_pass=["AlterOpLayout"]
):
    module = relay.build(mod, target=TARGET, runtime=RUNTIME, params=params)


# Inspecting the compilation output
# ---------------------------------
#
# The compilation process has produced some C code implementing the operators in this graph. We
# can inspect it by printing the CSourceModule contents (for the purposes of this tutorial, let's
# just print the first 10 lines):

c_source_module = module.get_lib().imported_modules[0]
assert c_source_module.type_key == "c", "tutorial is broken"

c_source_code = c_source_module.get_source()
first_few_lines = c_source_code.split("\n")[:10]
assert any(
    l.startswith("TVM_DLL int32_t tvmgen_default_") for l in first_few_lines
), f"tutorial is broken: {first_few_lines!r}"
print("\n".join(first_few_lines))


# Compiling the generated code
# ----------------------------
#
# Now we need to incorporate the generated C code into a project that allows us to run inference on the
# device. The simplest way to do this is to integrate it yourself, using microTVM's standard output format
# (:doc:`Model Library Format` </dev/model_library_format>`). This is a tarball with a standard layout:

# Get a temporary path where we can store the tarball (since this is running as a tutorial).

fd, model_library_format_tar_path = tempfile.mkstemp()
os.close(fd)
os.unlink(model_library_format_tar_path)
tvm.micro.export_model_library_format(module, model_library_format_tar_path)

with tarfile.open(model_library_format_tar_path, "r:*") as tar_f:
    print("\n".join(f" - {m.name}" for m in tar_f.getmembers()))

# Cleanup for tutorial:
os.unlink(model_library_format_tar_path)


# TVM also provides a standard way for embedded platforms to automatically generate a standalone
# project, compile and flash it to a target, and communicate with it using the standard TVM RPC
# protocol. The Model Library Format serves as the model input to this process. When embedded
# platforms provide such an integration, they can be used directly by TVM for both host-driven
# inference and autotuning . This integration is provided by the
# `microTVM Project API` <https://github.com/apache/tvm-rfcs/blob/main/rfcs/0008-microtvm-project-api.md>_,
#
# Embedded platforms need to provide a Template Project containing a microTVM API Server (typically,
# this lives in a file ``microtvm_api_server.py`` in the root directory). Let's use the example ``host``
# project in this tutorial, which simulates the device using a POSIX subprocess and pipes:

template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))
project_options = {}  # You can use options to provide platform-specific options through TVM.

# Compiling for physical hardware (or an emulated board, like the mps_an521)
# --------------------------------------------------------------------------
#  For physical hardware, you can try out the Zephyr platform by using a different template project
#  and options:
#

if use_physical_hw:
    template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr"))
    project_options = {"project_type": "host_driven", "board": BOARD}

# Create a temporary directory

temp_dir = tvm.contrib.utils.tempdir()
generated_project_dir = temp_dir / "generated-project"
generated_project = tvm.micro.generate_project(
    template_project_path, module, generated_project_dir, project_options
)

# Build and flash the project
generated_project.build()
generated_project.flash()


######################################################################
# Next, establish a session with the simulated device and run the
# computation. The `with session` line would typically flash an attached
# microcontroller, but in this tutorial, it simply launches a subprocess
# to stand in for an attached microcontroller.

with tvm.micro.Session(transport_context_manager=generated_project.transport()) as session:
    graph_mod = tvm.micro.create_local_graph_executor(
        module.get_graph_json(), session.get_system_lib(), session.device
    )

    # Set the model parameters using the lowered parameters produced by `relay.build`.
    graph_mod.set_input(**module.get_params())

    # The model consumes a single float32 value and returns a predicted sine value.  To pass the
    # input value we construct a tvm.nd.array object with a single contrived number as input. For
    # this model values of 0 to 2Pi are acceptable.
    graph_mod.set_input(input_tensor, tvm.nd.array(np.array([0.5], dtype="float32")))
    graph_mod.run()

    tvm_output = graph_mod.get_output(0).numpy()
    print("result is: " + str(tvm_output))
