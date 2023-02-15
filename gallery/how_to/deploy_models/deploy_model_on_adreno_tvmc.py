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
.. _tutorial-deploy-model-on-adreno-tvmc:

Deploy the Pretrained Model on Adreno™ with tvmc Interface
==========================================================
**Author**: Siva Rama Krishna

This article is a step-by-step tutorial to deploy pretrained Keras resnet50 model on Adreno™.

Besides that, you should have TVM built for Android.
See the following instructions on how to build it and setup RPC environment.

`Deploy to Adreno GPU <https://tvm.apache.org/docs/how_to/deploy/adreno.html>`_

"""

import os
import tvm
import numpy as np
from tvm import relay
from tvm.driver import tvmc
from tvm.driver.tvmc.model import TVMCPackage
from tvm.contrib import utils

#################################################################
# Configuration
# -------------
# Specify Adreno target before compiling to generate texture
# leveraging kernels and get all the benefits of textures
# Note: This generated example running on our x86 server for demonstration.
# If running it on the Android device, we need to
# specify its instruction set. Set :code:`local_demo` to False if you want
# to run this tutorial with a real device over rpc.
local_demo = True

# by default on CPU target will execute.
# select 'llvm', 'opencl' and 'opencl -device=adreno'
target = "llvm"

# Change target configuration.
# Run `adb shell cat /proc/cpuinfo` to find the arch.
arch = "arm64"
target_host = "llvm -mtriple=%s-linux-android" % arch

# Auto tuning is compute and time taking task, hence disabling for default run. Please enable it if required.
is_tuning = False
tune_log = "adreno-resnet50.log"

# To enable OpenCLML accelerated operator library.
enable_clml = False
cross_compiler = (
    os.getenv("ANDROID_NDK_HOME", "")
    + "/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android28-clang"
)

#######################################################################
# Make a Keras Resnet50 Model
# ---------------------------

from tensorflow.keras.applications.resnet50 import ResNet50

tmp_path = utils.tempdir()
model_file_name = tmp_path.relpath("resnet50.h5")

model = ResNet50(include_top=True, weights="imagenet", input_shape=(224, 224, 3), classes=1000)
model.save(model_file_name)


#######################################################################
# Load Model
# ----------
# Convert a model from any framework to a tvm relay module.
# tvmc.load supports models from any framework (like tensorflow saves_model, onnx, tflite ..etc) and auto detects the filetype.

tvmc_model = tvmc.load(model_file_name)

print(tvmc_model.mod)

# tvmc_model consists of tvmc_mode.mod which is relay module and tvmc_model.params which parms of the module.

#######################################################################
# AutoTuning
# ----------
# Now, the below api can be used for autotuning the model for any target.
# Tuning required RPC setup and please refer to
# `Deploy to Adreno GPU <https://tvm.apache.org/docs/how_to/deploy/adreno.html>`_

rpc_tracker_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")
rpc_tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
rpc_key = "android"
rpc_tracker = rpc_tracker_host + ":" + str(rpc_tracker_port)

# Auto tuning is compute intensive and time taking task.
# It is set to False in above configuration as this script runs in x86 for demonstration.
# Please to set :code:`is_tuning` to True to enable auto tuning.

# Also, :code:`test_target` is set to :code:`llvm` as this example to make compatible for x86 demonstration.
# Please change it to :code:`opencl` or :code:`opencl -device=adreno` for RPC target in configuration above.

if is_tuning:
    tvmc.tune(
        tvmc_model,
        target=target,
        tuning_records=tune_log,
        target_host=target_host,
        hostname=rpc_tracker_host,
        port=rpc_tracker_port,
        rpc_key=rpc_key,
        tuner="xgb",
        repeat=30,
        trials=3,
        early_stopping=0,
    )

#######################################################################
# Compilation
# -----------
# Compilation to produce tvm artifacts

# This generated example running on our x86 server for demonstration.
# To deply and tun on real target over RPC please set :code:`local_demo` to False in above configuration sestion.

# OpenCLML offloading will try to accelerate supported operators by using OpenCLML proprietory operator library.
# By default :code:`enable_clml` is set to False in above configuration section.

if not enable_clml:
    if local_demo:
        tvmc_package = tvmc.compile(
            tvmc_model,
            target=target,
        )
    else:
        tvmc_package = tvmc.compile(
            tvmc_model,
            target=target,
            target_host=target_host,
            cross=cross_compiler,
            tuning_records=tune_log,
        )
else:
    # Altrernatively, we can save the compilation output and save it as a TVMCPackage.
    # This way avoids loading of compiled module without compiling again.
    target = target + ", clml"
    pkg_path = tmp_path.relpath("keras-resnet50.tar")
    tvmc.compile(
        tvmc_model,
        target=target,
        target_host=target_host,
        cross=cross_compiler,
        tuning_records=tune_log,
        package_path=pkg_path,
    )

    # Load the compiled package
    tvmc_package = TVMCPackage(package_path=pkg_path)

# tvmc_package consists of tvmc_package.lib_path, tvmc_package.graph, tvmc_package.params
# Saved TVMPackage is nothing but tar archive with mod.so, mod.json and mod.params.


#######################################################################
# Deploy & Run
# ------------
# Deploy and run the compiled model on RPC
# Let tvmc fill inputs using random

# Run on RPC setup
if local_demo:
    result = tvmc.run(tvmc_package, device="cpu", fill_mode="random")
else:
    result = tvmc.run(
        tvmc_package,
        device="cl",
        rpc_key=rpc_key,
        hostname=rpc_tracker_host,
        port=rpc_tracker_port,
        fill_mode="random",
    )

# result is a dictionary of outputs.
print("Result:", result)
