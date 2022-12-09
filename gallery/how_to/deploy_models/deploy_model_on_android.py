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
.. _tutorial-deploy-model-on-android:

Deploy the Pretrained Model on Android
=======================================
**Author**: `Tomohiro Kato <https://tkat0.github.io/>`_

This is an example of using Relay to compile a keras model and deploy it on Android device.
"""

# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore

import os
import numpy as np
from PIL import Image
import keras
from keras.applications.mobilenet_v2 import MobileNetV2
import tvm
from tvm import te
import tvm.relay as relay
from tvm import rpc
from tvm.contrib import utils, ndk, graph_executor as runtime
from tvm.contrib.download import download_testdata


######################################################################
# Setup Environment
# -----------------
# Since there are many required packages for Android, it is recommended to use the official Docker Image.
#
# First, to build and run Docker Image, we can run the following command.
#
# .. code-block:: bash
#
#   git clone --recursive https://github.com/apache/tvm tvm
#   cd tvm
#   docker build -t tvm.demo_android -f docker/Dockerfile.demo_android ./docker
#   docker run --pid=host -h tvm -v $PWD:/workspace \
#          -w /workspace -p 9190:9190 --name tvm -it tvm.demo_android bash
#
# You are now inside the container. The cloned TVM directory is mounted on /workspace.
# At this time, mount the 9190 port used by RPC described later.
#
# .. note::
#
#   Please execute the following steps in the container.
#   We can execute :code:`docker exec -it tvm bash` to open a new terminal in the container.
#
# Next we build the TVM.
#
# .. code-block:: bash
#
#   mkdir build
#   cd build
#   cmake -DUSE_LLVM=llvm-config-8 \
#         -DUSE_RPC=ON \
#         -DUSE_SORT=ON \
#         -DUSE_VULKAN=ON \
#         -DUSE_GRAPH_EXECUTOR=ON \
#         ..
#   make -j10
#
# After building TVM successfully, Please set PYTHONPATH.
#
# .. code-block:: bash
#
#   echo 'export PYTHONPATH=/workspace/python:/workspace/vta/python:${PYTHONPATH}' >> ~/.bashrc
#   source ~/.bashrc

#################################################################
# Start RPC Tracker
# -----------------
# TVM uses RPC session to communicate with Android device.
#
# To start an RPC tracker, run this command in the container. The tracker is
# required during the whole tuning process, so we need to open a new terminal for
# this command:
#
# .. code-block:: bash
#
#   python3 -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
#
# The expected output is
#
# .. code-block:: bash
#
#   INFO:RPCTracker:bind to 0.0.0.0:9190

#################################################################
# Register Android device to RPC Tracker
# --------------------------------------
# Now we can register our Android device to the tracker.
#
# Follow this `readme page <https://github.com/apache/tvm/tree/main/apps/android_rpc>`_ to
# install TVM RPC APK on the android device.
#
# Here is an example of config.mk. I enabled OpenCL and Vulkan.
#
#
# .. code-block:: bash
#
#   APP_ABI = arm64-v8a
#
#   APP_PLATFORM = android-24
#
#   # whether enable OpenCL during compile
#   USE_OPENCL = 1
#
#   # whether to enable Vulkan during compile
#   USE_VULKAN = 1
#
#   ifeq ($(USE_VULKAN), 1)
#     # Statically linking vulkan requires API Level 24 or higher
#     APP_PLATFORM = android-24
#   endif
#
#   # the additional include headers you want to add, e.g., SDK_PATH/adrenosdk/Development/Inc
#   ADD_C_INCLUDES += /work/adrenosdk-linux-5_0/Development/Inc
#   ADD_C_INCLUDES =
#
#   # the additional link libs you want to add, e.g., ANDROID_LIB_PATH/libOpenCL.so
#   ADD_LDLIBS =
#
# .. note::
#
#   At this time, don't forget to `create a standalone toolchain <https://github.com/apache/tvm/tree/main/apps/android_rpc#architecture-and-android-standalone-toolchain>`_ .
#
#   for example
#
#   .. code-block:: bash
#
#     $ANDROID_NDK_HOME/build/tools/make-standalone-toolchain.sh \
#        --platform=android-24 --use-llvm --arch=arm64 --install-dir=/opt/android-toolchain-arm64
#     export TVM_NDK_CC=/opt/android-toolchain-arm64/bin/aarch64-linux-android-g++
#
# Next, start the Android application and enter the IP address and port of RPC Tracker.
# Then you have already registered your device.
#
# After registering devices, we can confirm it by querying rpc_tracker
#
# .. code-block:: bash
#
#   python3 -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
#
# For example, if we have 1 Android device.
# the output can be
#
# .. code-block:: bash
#
#    Queue Status
#    ----------------------------------
#    key          total  free  pending
#    ----------------------------------
#    android      1      1     0
#    ----------------------------------
#
# To confirm that you can communicate with Android, we can run following test script.
# If you use OpenCL and Vulkan, please set :code:`test_opencl` and :code:`test_vulkan` in the script.
#
# .. code-block:: bash
#
#   export TVM_TRACKER_HOST=0.0.0.0
#   export TVM_TRACKER_PORT=9190
#
# .. code-block:: bash
#
#   cd /workspace/apps/android_rpc
#   python3 tests/android_rpc_test.py
#

######################################################################
# Load pretrained keras model
# ---------------------------
# We load a pretrained MobileNetV2(alpha=0.5) classification model provided by keras.
keras.backend.clear_session()  # Destroys the current TF graph and creates a new one.
weights_url = "".join(
    [
        "https://github.com/JonathanCMitchell/",
        "mobilenet_v2_keras/releases/download/v1.1/",
        "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_0.5_224.h5",
    ]
)
weights_file = "mobilenet_v2_weights.h5"
weights_path = download_testdata(weights_url, weights_file, module="keras")
keras_mobilenet_v2 = MobileNetV2(
    alpha=0.5, include_top=True, weights=None, input_shape=(224, 224, 3), classes=1000
)
keras_mobilenet_v2.load_weights(weights_path)

######################################################################
# In order to test our model, here we download an image of cat and
# transform its format.
img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_name = "cat.png"
img_path = download_testdata(img_url, img_name, module="data")
image = Image.open(img_path).resize((224, 224))
dtype = "float32"


def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


x = transform_image(image)

######################################################################
# synset is used to transform the label from number of ImageNet class to
# the word human can understand.
synset_url = "".join(
    [
        "https://gist.githubusercontent.com/zhreshold/",
        "4d0b62f3d01426887599d4f7ede23ee5/raw/",
        "596b27d23537e5a1b5751d2b0481ef172f58b539/",
        "imagenet1000_clsid_to_human.txt",
    ]
)
synset_name = "imagenet1000_clsid_to_human.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synset = eval(f.read())


######################################################################
# Compile the model with relay
# ----------------------------
# If we run the example on our x86 server for demonstration, we can simply
# set it as :code:`llvm`. If running it on the Android device, we need to
# specify its instruction set. Set :code:`local_demo` to False if you want
# to run this tutorial with a real device.

local_demo = True

# by default on CPU target will execute.
# select 'cpu', 'opencl' and 'vulkan'
test_target = "cpu"

# Change target configuration.
# Run `adb shell cat /proc/cpuinfo` to find the arch.
arch = "arm64"
target = tvm.target.Target("llvm -mtriple=%s-linux-android" % arch)

if local_demo:
    target = tvm.target.Target("llvm")
elif test_target == "opencl":
    target = tvm.target.Target("opencl", host=target)
elif test_target == "vulkan":
    target = tvm.target.Target("vulkan", host=target)

input_name = "input_1"
shape_dict = {input_name: x.shape}
mod, params = relay.frontend.from_keras(keras_mobilenet_v2, shape_dict)

with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

# After `relay.build`, you will get three return values: graph,
# library and the new parameter, since we do some optimization that will
# change the parameters but keep the result of model as the same.

# Save the library at local temporary directory.
tmp = utils.tempdir()
lib_fname = tmp.relpath("net.so")
fcompile = ndk.create_shared if not local_demo else None
lib.export_library(lib_fname, fcompile)

######################################################################
# Deploy the Model Remotely by RPC
# --------------------------------
# With RPC, you can deploy the model remotely from your host machine
# to the remote android device.

tracker_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")
tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
key = "android"

if local_demo:
    remote = rpc.LocalSession()
else:
    tracker = rpc.connect_tracker(tracker_host, tracker_port)
    # When running a heavy model, we should increase the `session_timeout`
    remote = tracker.request(key, priority=0, session_timeout=60)

if local_demo:
    dev = remote.cpu(0)
elif test_target == "opencl":
    dev = remote.cl(0)
elif test_target == "vulkan":
    dev = remote.vulkan(0)
else:
    dev = remote.cpu(0)

# upload the library to remote device and load it
remote.upload(lib_fname)
rlib = remote.load_module("net.so")

# create the remote runtime module
module = runtime.GraphModule(rlib["default"](dev))

######################################################################
# Execute on TVM
# --------------

# set input data
module.set_input(input_name, tvm.nd.array(x.astype(dtype)))
# run
module.run()
# get output
out = module.get_output(0)

# get top1 result
top1 = np.argmax(out.numpy())
print("TVM prediction top-1: {}".format(synset[top1]))

print("Evaluate inference time cost...")
print(module.benchmark(dev, number=1, repeat=10))

######################################################################
# Sample Output
# -------------
# The following is the result of 'cpu', 'opencl' and 'vulkan' using Adreno 530 on Snapdragon 820
#
# Although we can run on a GPU, it is slower than CPU.
# To speed up, we need to write and optimize the schedule according to the GPU architecture.
#
# .. code-block:: bash
#
#    # cpu
#    TVM prediction top-1: tiger cat
#    Evaluate inference time cost...
#    Mean inference time (std dev): 37.92 ms (19.67 ms)
#
#    # opencl
#    TVM prediction top-1: tiger cat
#    Evaluate inference time cost...
#    Mean inference time (std dev): 419.83 ms (7.49 ms)
#
#    # vulkan
#    TVM prediction top-1: tiger cat
#    Evaluate inference time cost...
#    Mean inference time (std dev): 465.80 ms (4.52 ms)
