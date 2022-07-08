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
.. _tutorial-micro-AoT:

microTVM Host-Driven AoT
===========================
**Authors**:
`Mehrdad Hessar <https://github.com/mehrdadh>`_,
`Alan MacDonald <https://github.com/alanmacd>`_

This tutorial is showcasing microTVM host-driven AoT compilation with
a TFLite model. This tutorial can be executed on a X86 CPU using C runtime (CRT)
or on Zephyr plarform on a microcontroller that supports Zephyr platform.
"""

import numpy as np
import pathlib
import json
import os

import tvm
from tvm import relay
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.download import download_testdata

######################################################################
# Import a TFLite model
# ---------------------
#
# To begin with, download and import a TFLite model from TinyMLPerf models.
#
# **Note:** By default this tutorial runs on X86 CPU using CRT, if you would like to run on Zephyr platform
# you need to export `TVM_MICRO_USE_HW` environment variable.
#
use_physical_hw = bool(os.getenv("TVM_MICRO_USE_HW"))
MODEL_URL = "https://github.com/tlc-pack/web-data/raw/main/testdata/microTVM/model/keyword_spotting_quant.tflite"
MODEL_PATH = download_testdata(MODEL_URL, "keyword_spotting_quant.tflite", module="model")
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
# on X86 CPU using CRT runtime or running a model with Zephyr platform on qemu_x86 simulator
# board. In the case of a physical microcontoller, we get the target model for the physical
# board (E.g. nucleo_f746zg) and pass it to `tvm.target.target.micro` to create a full
# micro target.
#
RUNTIME = Runtime("crt", {"system-lib": True})
TARGET = tvm.target.target.micro("host")
EXECUTOR = Executor("aot")

if use_physical_hw:
    boards_file = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr")) / "boards.json"
    with open(boards_file) as f:
        boards = json.load(f)
    BOARD = os.getenv("TVM_MICRO_BOARD", default="nucleo_f746zg")
    TARGET = tvm.target.target.micro(boards[BOARD]["model"])

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
# -----------------------
#
# Now that we have the comipled model as an IRModule, we need to create a project
# with the compiled model in microTVM. To do this, we use Project API. We have defined
# CRT and Zephyr microTVM template projects which are used for X86 CPU and Zephyr platforms
# respectively.
#
template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))
project_options = {}  # You can use options to provide platform-specific options through TVM.

if use_physical_hw:
    template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("zephyr"))
    project_options = {"project_type": "host_driven", "zephyr_board": BOARD}

temp_dir = tvm.contrib.utils.tempdir()
generated_project_dir = temp_dir / "project"
project = tvm.micro.generate_project(
    template_project_path, module, generated_project_dir, project_options
)

######################################################################
# Build, flash and execute the model
# -----------------------
# Next, we build the microTVM project and flash it. Flash step is specific to
# physical microcontrollers and it is skipped if it is using CRT runtime or running
# on Zephyr simulator. Next, we define the labels for the model output and execute
# the model with a sample with expected value of 6 (label: left).
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
#
# Output:
# Label is `left` with index `6`
#
