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

"""
microTVM with ONNX models
=========================
**Author**: `Matt Welsh <https://www.mdw.la/>`

This tutorial is an introduction to compiling and
running an ONNX model on a device using microTVM.
"""

# Setup
# -----
#
# Build TVM wth ``USE_MICRO``
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Download the TVM sources, and build TVM from source following the
# instructions `on this page
# <https://tvm.apache.org/docs/install/from_source.html#install-from-source>`.
# When bulding TVM, ensure that the following lines are present in
# your ``config.cmake`` file:
# .. code-block:: bash
#
#    # Whether enable MicroTVM runtime
#    set(USE_MICRO ON)
#
# Install Python dependencies
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# microTVM and this tutorial require a number of Python dependencies --
# in addition to TVM itself -- to be installed in your Python environment.
# For convenience, we have provided a `Poetry <https://python-poetry.org/>`
# ``pyproject.toml`` configuration file in the directory ``apps/microtvm``.
# You can use this as follows:
#
# .. code-block:: bash
#
#    $ cd $TVM_HOME/apps/microtvm
#    $ poetry lock && poetry install
#    $ poetry shell
#
# You should now be in a Python virtual environment with all of the appropriate
# dependencies installed.
#
# Install Zephyr
# ^^^^^^^^^^^^^^
#
# microTVM currently uses the `Zephyr <https://zephyrproject.org>` RTOS as
# the basis for the device-side runtime. To get started, install Zephyr
# and an appropriate device-specific toolchain using the
# `Zephyr installation instructions <https://docs.zephyrproject.org/latest/getting_started/index.html>`_.
#
# Be sure you are able to build and flash a sample Zephyr program
# (e.g., the "Blinky" demo) to your device before you proceed with this tutorial.
#
# Instead of building and installing the Zephyr toolchain yourself, you can use the
# `tutorial-micro-reference-vm` for a quick setup which includes all of the necessary
# tools.

import datetime
import io
import os
import sys

import onnx
import tvm
import tvm.micro
from tvm import autotvm
from tvm import relay
from tvm.contrib import graph_runtime as runtime
from tvm.micro.contrib import zephyr
from PIL import Image
import numpy as np

######################################################################
# For this tutorial, we use a pretrained ONNX model implementing
# the MNIST handwritten digit recognition on 28x28 px input images.

MODEL_FILE = "../../tests/micro/zephyr/testdata/mnist-8.onnx"
MODEL_SHAPE = (1, 1, 28, 28)
INPUT_TENSOR_NAME = "Input3"

onnx_model = onnx.load(MODEL_FILE)
print(f"Loaded ONNX model: {MODEL_FILE}")

######################################################################
# Next, we convert the model to Relay format.
relay_mod, params = relay.frontend.from_onnx(onnx_model, shape=MODEL_SHAPE, freeze_params=True)
relay_mod = relay.transform.DynamicToStatic()(relay_mod)

######################################################################
# Next, we lower the Relay model to the specific device we are
# targeting. In this case, we are using the Nordic Semiconductor
# `nRF5340DK development board <https://www.nordicsemi.com/Software-and-tools/Development-Kits/nRF5340-DK>`.

# This is the device target name used by microTVM. It is used to select default
# options for the device when we use ``tvm.target.target.micro()`` below.
UTVM_TARGET = "nrf5340dk" # or stm32f746xx

# This is the board designation used by Zephyr, and required for the compilation process.
UTVM_ZEPHYR_BOARD = "nrf5340dk_nrf5340_cpuapp" # or nucleo_f746zg

######################################################################
# If you wish to run against an emulated Zephyr device using QEMU,
# you can uncomment these lines instead:
# UTVM_TARGET = "host"
# UTVM_ZEPHYR_BOARD = "qemu_x86"


######################################################################
# We define the TVM target here.
# We add -link-params=1 option here, so that the model parameters are
# included in the resulting binary image.
target = tvm.target.target.micro(UTVM_TARGET, options=["-link-params=1"])

######################################################################
# Now, we do the Relay build.
TVM_OPT_LEVEL = 3
with tvm.transform.PassContext(opt_level=TVM_OPT_LEVEL, config={"tir.disable_vectorize": True}):
    lowered = relay.build(relay_mod, target, params=params)
    graph_json_str = lowered.get_json()

######################################################################
# Next, create a uTVM Workspace. This is a location where the
# generated Zephyr project will be compiled.
workspace_root = os.path.abspath(
    f'workspace/{datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")}'
)
workspace_parent = os.path.dirname(workspace_root)
if not os.path.exists(workspace_parent):
    os.makedirs(workspace_parent)
workspace = tvm.micro.Workspace(debug=True, root=workspace_root)
print(f"Using workspace: {workspace_root}")

######################################################################
# Now we create the ``ZephyrCompiler`` object, which generates the
# Zephyr project for hosting the device runtime code, as well as
# generating the device-specific binary from our model.

# You are welcome to implement your own Zephyr-based runtime environment
# for your project. This runtime is a demo which provides basic capabilities
# for interfacing to the microTVM device code via the device's serial
# port.
UTVM_ZEPHYR_RUNTIME_DIR = "../../apps/microtvm/zephyr/demo_runtime"

# The ``west`` command is used by Zephyr for compilation and
# flashing devices.
UTVM_WEST_CMD = "west"

compiler = zephyr.ZephyrCompiler(
    project_dir=UTVM_ZEPHYR_RUNTIME_DIR,
    board=UTVM_ZEPHYR_BOARD,
    zephyr_toolchain_variant="zephyr",
    west_cmd=UTVM_WEST_CMD,
)

######################################################################
# Do the actual build.
opts = tvm.micro.default_options(f"{UTVM_ZEPHYR_RUNTIME_DIR}/crt")

micro_bin = tvm.micro.build_static_runtime(workspace, compiler, lowered.lib, opts)

######################################################################
# Next, we create a ``tvm.micro.Session`` which handles the details
# of flashing the binary to the device and opening a serial-port
# RPC session with the device for controlling it.

flasher = compiler.flasher()
with tvm.micro.Session(binary=micro_bin, flasher=flasher) as sess:
    mod = tvm.micro.create_local_graph_runtime(graph_json_str, sess.get_system_lib(), sess.context)

    # Load test images.
    DIGIT_2_IMAGE = "../../tests/micro/zephyr/testdata/digit-2.jpg"
    DIGIT_9_IMAGE = "../../tests/micro/zephyr/testdata/digit-9.jpg"

    digit_2 = Image.open(DIGIT_2_IMAGE).resize((28, 28))
    digit_9 = Image.open(DIGIT_9_IMAGE).resize((28, 28))
    digit_2 = np.asarray(digit_2).astype("float32")
    digit_9 = np.asarray(digit_9).astype("float32")
    digit_2 = np.expand_dims(digit_2, axis=0)
    digit_9 = np.expand_dims(digit_9, axis=0)

    # Set the input tensor of the model to the digit-2 test image.
    mod.set_input(INPUT_TENSOR_NAME, tvm.nd.array(digit_2))

    # Run inference and get the result.
    mod.run()
    output = mod.get_output(0).asnumpy()
    print(f"Top result for digit-2 is: {np.argmax(output)}")

    # Do likewise for the digit-9 image.
    mod.set_input(INPUT_TENSOR_NAME, tvm.nd.array(digit_9))
    mod.run()
    output = mod.get_output(0).asnumpy()
    print(f"Top result for digit-9 is: {np.argmax(output)}")
