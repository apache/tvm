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
.. _tutorial-micro-aot:

3. microTVM Ahead-of-Time (AOT) Compilation
===========================================
**Authors**:
`Mehrdad Hessar <https://github.com/mehrdadh>`_,
`Alan MacDonald <https://github.com/alanmacd>`_

This tutorial is showcasing microTVM host-driven AoT compilation with
a TFLite model. AoTExecutor reduces the overhead of parsing graph at runtime
compared to GraphExecutor. Also, we can have better memory management using ahead
of time compilation. This tutorial can be executed on a x86 CPU using C runtime (CRT)
or on Zephyr platform on a microcontroller/board supported by Zephyr.
"""

######################################################################
#
#     .. include:: ../../../../gallery/how_to/work_with_microtvm/install_dependencies.rst
#


import os

# By default, this tutorial runs on x86 CPU using TVM's C runtime. If you would like
# to run on real Zephyr hardware, you must export the `TVM_MICRO_USE_HW` environment
# variable. Otherwise (if you are using the C runtime), you can skip installing
# Zephyr. It takes ~20 minutes to install Zephyr.
use_physical_hw = bool(os.getenv("TVM_MICRO_USE_HW"))

######################################################################
#
#     .. include:: ../../../../gallery/how_to/work_with_microtvm/install_zephyr.rst
#

######################################################################
# Import Python dependencies
# -------------------------------
#
import numpy as np
import pathlib
import json

import tvm
from tvm import relay
import tvm.micro.testing
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.download import download_testdata

######################################################################
# Import a TFLite model
# ---------------------
#
# To begin with, download and import a Keyword Spotting TFLite model.
# This model is originally from `MLPerf Tiny repository <https://github.com/mlcommons/tiny>`_.
# To test this model, we use samples from `KWS dataset provided by Google <https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html>`_.
#
# **Note:** By default this tutorial runs on x86 CPU using CRT, if you would like to run on Zephyr platform
# you need to export `TVM_MICRO_USE_HW` environment variable.
#
MODEL_URL = "https://github.com/mlcommons/tiny/raw/bceb91c5ad2e2deb295547d81505721d3a87d578/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite"
MODEL_PATH = download_testdata(MODEL_URL, "kws_ref_model.tflite", module="model")
SAMPLE_URL = "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/data/keyword_spotting_int8_6.pyc.npy"
SAMPLE_PATH = download_testdata(SAMPLE_URL, "keyword_spotting_int8_6.pyc.npy", module="data")

tflite_model_buf = open(MODEL_PATH, "rb").read()
try:
    import tflite

    tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
except AttributeError:
    import tflite.Model

    tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

input_shape = (1, 49, 10, 1)
INPUT_NAME = "input_1"
relay_mod, params = relay.frontend.from_tflite(
    tflite_model, shape_dict={INPUT_NAME: input_shape}, dtype_dict={INPUT_NAME: "int8"}
)

######################################################################
# Defining the target
# -------------------
#
# Now we need to define the target, runtime and executor. In this tutorial, we focused on
# using AOT host driven executor. We use the host micro target which is for running a model
# on x86 CPU using CRT runtime or running a model with Zephyr platform on qemu_x86 simulator
# board. In the case of a physical microcontroller, we get the target model for the physical
# board (E.g. nucleo_l4r5zi) and change `BOARD` to supported Zephyr board.
#

# Use the C runtime (crt) and enable static linking by setting system-lib to True
RUNTIME = Runtime("crt", {"system-lib": True})

# Simulate a microcontroller on the host machine. Uses the main() from `src/runtime/crt/host/main.cc`.
# To use physical hardware, replace "host" with something matching your hardware.
TARGET = tvm.micro.testing.get_target("crt")

# Use the AOT executor rather than graph or vm executors. Don't use unpacked API or C calling style.
EXECUTOR = Executor("aot")

if use_physical_hw:
    BOARD = os.getenv("TVM_MICRO_BOARD", default="nucleo_l4r5zi")
    SERIAL = os.getenv("TVM_MICRO_SERIAL", default=None)
    TARGET = tvm.micro.testing.get_target("zephyr", BOARD)

######################################################################
# Compile the model
# -----------------
#
# Now, we compile the model for the target:
#
with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
    module = tvm.relay.build(
        relay_mod, target=TARGET, params=params, runtime=RUNTIME, executor=EXECUTOR
    )

######################################################################
# Create a microTVM project
# -------------------------
#
# Now that we have the compiled model as an IRModule, we need to create a firmware project
# to use the compiled model with microTVM. To do this, we use Project API. We have defined
# CRT and Zephyr microTVM template projects which are used for x86 CPU and Zephyr boards
# respectively.
#
template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))
project_options = {}  # You can use options to provide platform-specific options through TVM.

if use_physical_hw:
    template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr"))
    project_options = {
        "project_type": "host_driven",
        "board": BOARD,
        "serial_number": SERIAL,
        "config_main_stack_size": 4096,
        "zephyr_base": os.getenv("ZEPHYR_BASE", default="/content/zephyrproject/zephyr"),
    }

temp_dir = tvm.contrib.utils.tempdir()
generated_project_dir = temp_dir / "project"
project = tvm.micro.generate_project(
    template_project_path, module, generated_project_dir, project_options
)

######################################################################
# Build, flash and execute the model
# ----------------------------------
# Next, we build the microTVM project and flash it. Flash step is specific to
# physical microcontrollers and it is skipped if it is simulating a microcontroller
# via the host main.cc or if a Zephyr emulated board is selected as the target.
# Next, we define the labels for the model output and execute the model with a
# sample with expected value of 6 (label: left).
#
project.build()
project.flash()

labels = [
    "_silence_",
    "_unknown_",
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
]
with tvm.micro.Session(project.transport()) as session:
    aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
    sample = np.load(SAMPLE_PATH)
    aot_executor.get_input(INPUT_NAME).copyfrom(sample)
    aot_executor.run()
    result = aot_executor.get_output(0).numpy()
    print(f"Label is `{labels[np.argmax(result)]}` with index `{np.argmax(result)}`")
