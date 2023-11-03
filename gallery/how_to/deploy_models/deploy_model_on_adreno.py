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
.. _tutorial-deploy-model-on-adreno:

Deploy the Pretrained Model on Adreno™
======================================
**Author**: Daniil Barinov, Siva Rama Krishna

This article is a step-by-step tutorial to deploy pretrained Pytorch ResNet-18 model on Adreno (on different precisions).

For us to begin with, PyTorch must be installed.
TorchVision is also required since we will be using it as our model zoo.

A quick solution is to install it via pip:

.. code-block:: bash

  %%shell
  pip install torch
  pip install torchvision

Besides that, you should have TVM builded for Android.
See the following instructions on how to build it.

`Deploy to Adreno GPU <https://tvm.apache.org/docs/how_to/deploy/adreno.html>`_

After the build section there should be two files in *build* directory «libtvm_runtime.so» and «tvm_rpc».
Let's push them to the device and run TVM RPC Server.
"""

######################################################################
# TVM RPC Server
# --------------
# To get the hash of the device use:
#
# .. code-block:: bash
#
#   adb devices
#
# Set the android device to use, if you have several devices connected to your computer.
#
# .. code-block:: bash
#
#   export ANDROID_SERIAL=<device-hash>
#
# Then to upload these two files to the device you should use:
#
# .. code-block:: bash
#
#   adb push {libtvm_runtime.so,tvm_rpc} /data/local/tmp
#
# At this moment you will have «libtvm_runtime.so» and «tvm_rpc» on path /data/local/tmp on your device.
# Sometimes cmake can’t find «libc++_shared.so». Use:
#
# .. code-block:: bash
#
#   find ${ANDROID_NDK_HOME} -name libc++_shared.so
#
# to find it and also push it with adb on the desired device:
#
# .. code-block:: bash
#
#   adb push libc++_shared.so /data/local/tmp
#
# We are now ready to run the TVM RPC Server.
# Launch rpc_tracker with following line in 1st console:
#
# .. code-block:: bash
#
#   python3 -m tvm.exec.rpc_tracker --port 9190
#
# Then we need to run tvm_rpc server from under the desired device in 2nd console:
#
# .. code-block:: bash
#
#   adb reverse tcp:9190 tcp:9190
#   adb forward tcp:5000 tcp:5000
#   adb forward tcp:5002 tcp:5001
#   adb forward tcp:5003 tcp:5002
#   adb forward tcp:5004 tcp:5003
#   adb shell LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/tvm_rpc server --host=0.0.0.0 --port=5000 --tracker=127.0.0.1:9190 --key=android --port-end=5100
#
# Before proceeding to compile and infer model, specify TVM_TRACKER_HOST and TVM_TRACKER_PORT
#
# .. code-block:: bash
#
#   export TVM_TRACKER_HOST=0.0.0.0
#   export TVM_TRACKER_PORT=9190
#
# check that the tracker is running and the device is available
#
# .. code-block:: bash
#
#     python -m tvm.exec.query_rpc_tracker --port 9190
#
# For example, if we have 1 Android device,
# the output can be:
#
# .. code-block:: bash
#
#    Queue Status
#    ----------------------------------
#    key          total  free  pending
#    ----------------------------------
#    android      1      1     0
#    ----------------------------------

#################################################################
# Configuration
# -------------

import os
import torch
import torchvision
import tvm
from tvm import te
from tvm import relay, rpc
from tvm.contrib import utils, ndk
from tvm.contrib import graph_executor
from tvm.relay.op.contrib import clml
from tvm import autotvm

# Below are set of configuration that controls the behaviour of this script like
# local run or device run, target definitions,  dtype setting and auto tuning enablement.
# Change these settings as needed if required.

# Adreno devices are efficient with float16 compared to float32
# Given the expected output doesn't effect by lowering precision
# it's advisable to use lower precision.
# We have a helper API to make the precision conversion simple and
# it supports dtype with "float16" and "float16_acc32" modes.
# Let's choose "float16" for calculation and "float32" for accumulation.

calculation_dtype = "float16"
acc_dtype = "float32"

# Specify Adreno target before compiling to generate texture
# leveraging kernels and get all the benefits of textures
# Note: This generated example running on our x86 server for demonstration.
# If running it on the Android device, we need to
# specify its instruction set. Set :code:`local_demo` to False if you want
# to run this tutorial with a real device over rpc.
local_demo = True

# by default on CPU target will execute.
# select 'cpu', 'opencl' and 'opencl -device=adreno'
test_target = "cpu"

# Change target configuration.
# Run `adb shell cat /proc/cpuinfo` to find the arch.
arch = "arm64"
target = tvm.target.Target("llvm -mtriple=%s-linux-android" % arch)

# Auto tuning is compute intensive and time taking task,
# hence disabling for default run. Please enable it if required.
is_tuning = False
tune_log = "adreno-resnet18.log"

# To enable OpenCLML accelerated operator library.
enable_clml = False

#################################################################
# Get a PyTorch Model
# -------------------
# Get resnet18 from torchvision models
model_name = "resnet18"
model = getattr(torchvision.models, model_name)(pretrained=True)
model = model.eval()

# We grab the TorchScripted model via tracing
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

#################################################################
# Load a test image
# -----------------
# As an example we would use classical cat image from ImageNet

from PIL import Image
from tvm.contrib.download import download_testdata
from matplotlib import pyplot as plt
import numpy as np

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
plt.imshow(img)
plt.show()

# Preprocess the image and convert to tensor
from torchvision import transforms

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)

#################################################################
# Convert PyTorch model to Relay module
# -------------------------------------
# TVM has frontend api for various frameworks under relay.frontend and now
# for pytorch model import we have relay.frontend.from_pytorch api.
# Input name can be arbitrary
input_name = "input0"
shape_list = [(input_name, img.shape)]

mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

#################################################################
# Precisions
# ----------

# Adreno devices are efficient with float16 compared to float32
# Given the expected output doesn't effect by lowering precision
# it's advisable to use lower precision.

# TVM support Mixed Precision through ToMixedPrecision transformation pass.
# We may need to register precision rules like precision type, accumultation
# datatype ...etc. for the required operators to override the default settings.
# The below helper api simplifies the precision conversions across the module.

# Calculation dtype is set to "float16" and accumulation dtype is set to "float32"
# in configuration section above.

from tvm.driver.tvmc.transform import apply_graph_transforms

mod = apply_graph_transforms(
    mod,
    {
        "mixed_precision": True,
        "mixed_precision_ops": ["nn.conv2d", "nn.dense"],
        "mixed_precision_calculation_type": calculation_dtype,
        "mixed_precision_acc_type": acc_dtype,
    },
)

#################################################################
# As you can see in the IR, the architecture now contains cast operations, which are
# needed to convert to FP16 precision.
# You can also use "float16" or "float32" precisions as other dtype options.

#################################################################
# Prepare TVM Target
# ------------------

# This generated example running on our x86 server for demonstration.

# To deply and tun on real target over RPC please set :code:`local_demo` to False in above configuration sestion.
# Also, :code:`test_target` is set to :code:`llvm` as this example to make compatible for x86 demonstration.
# Please change it to :code:`opencl` or :code:`opencl -device=adreno` for RPC target in configuration above.

if local_demo:
    target = tvm.target.Target("llvm")
elif test_target.find("opencl"):
    target = tvm.target.Target(test_target, host=target)

##################################################################
# AutoTuning
# ----------
# The below few instructions can auto tune the relay module with xgboost being the tuner algorithm.

# Auto Tuning process involces stages of extracting the tasks, defining tuning congiguration and
# tuning each task for best performing kernel configuration.

# Get RPC related settings.
rpc_tracker_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")
rpc_tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))
key = "android"

# Auto tuning is compute intensive and time taking task.
# It is set to False in above configuration as this script runs in x86 for demonstration.
# Please to set :code:`is_tuning` to True to enable auto tuning.

if is_tuning:
    # Auto Tuning Stage 1: Extract tunable tasks
    tasks = autotvm.task.extract_from_program(
        mod, target=test_target, target_host=target, params=params
    )

    # Auto Tuning Stage 2: Define tuning configuration
    tmp_log_file = tune_log + ".tmp"
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(
            build_func=ndk.create_shared, timeout=15
        ),  # Build the test kernel locally
        runner=autotvm.RPCRunner(  # The runner would be on a remote device.
            key,  # RPC Key
            host=rpc_tracker_host,  # Tracker host
            port=int(rpc_tracker_port),  # Tracker port
            number=3,  # Number of runs before averaging
            timeout=600,  # RPC Timeout
        ),
    )
    n_trial = 1024  # Number of iteration of training before choosing the best kernel config
    early_stopping = False  # Can be enabled to stop tuning while the loss is not minimizing.

    # Auto Tuning Stage 3: Iterate through the tasks and tune.
    from tvm.autotvm.tuner import XGBTuner

    for i, tsk in enumerate(reversed(tasks[:3])):
        print("Task:", tsk)
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # choose tuner
        tuner = "xgb"

        # create tuner
        if tuner == "xgb":
            tuner_obj = XGBTuner(tsk, loss_type="reg")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="curve")
        elif tuner == "xgb_rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_rank_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_rank_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_rank_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
        elif tuner == "xgb_rank_binary":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary")
        elif tuner == "xgb_rank_binary_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="knob")
        elif tuner == "xgb_rank_binary_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="itervar")
        elif tuner == "xgb_rank_binary_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )
    # Auto Tuning Stage 4: Pick the best performing configurations from the overall log.
    autotvm.record.pick_best(tmp_log_file, tune_log)

#################################################################
# Enable OpenCLML Offloading
# --------------------------
# OpenCLML offloading will try to accelerate supported operators
# by using OpenCLML proprietory operator library.

# By default :code:`enable_clml` is set to False in above configuration section.

if not local_demo and enable_clml:
    mod = clml.partition_for_clml(mod, params)

#################################################################
# Compilation
# -----------
# Use tuning cache if exists.
if os.path.exists(tune_log):
    with autotvm.apply_history_best(tune_log):
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=target, params=params)
else:
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

#################################################################
# Deploy the Model Remotely by RPC
# --------------------------------
# Using RPC you can deploy the model from host
# machine to the remote Adreno device
if local_demo:
    remote = rpc.LocalSession()
else:
    tracker = rpc.connect_tracker(rpc_tracker_host, rpc_tracker_port)
    # When running a heavy model, we should increase the `session_timeout`
    remote = tracker.request(key, priority=0, session_timeout=60)

if local_demo:
    dev = remote.cpu(0)
elif test_target.find("opencl"):
    dev = remote.cl(0)
else:
    dev = remote.cpu(0)

temp = utils.tempdir()
dso_binary = "dev_lib_cl.so"
dso_binary_path = temp.relpath(dso_binary)
fcompile = ndk.create_shared if not local_demo else None
lib.export_library(dso_binary_path, fcompile=fcompile)
remote_path = "/data/local/tmp/" + dso_binary
remote.upload(dso_binary_path)
rlib = remote.load_module(dso_binary)
m = graph_executor.GraphModule(rlib["default"](dev))

#################################################################
# Run inference
# -------------
# We now can set inputs, infer our model and get predictions as output
m.set_input(input_name, tvm.nd.array(img.astype("float32")))
m.run()
tvm_output = m.get_output(0)

#################################################################
# Get predictions and performance statistic
# -----------------------------------------
# This piece of code displays the top-1 and top-5 predictions, as
# well as provides information about the model's performance
from os.path import join, isfile
from matplotlib import pyplot as plt
from tvm.contrib import download

# Download ImageNet categories
categ_url = "https://github.com/uwsampl/web-data/raw/main/vta/models/"
categ_fn = "synset.txt"
download.download(join(categ_url, categ_fn), categ_fn)
synset = eval(open(categ_fn).read())

top_categories = np.argsort(tvm_output.asnumpy()[0])
top5 = np.flip(top_categories, axis=0)[:5]

# Report top-1 classification result
print("Top-1 id: {}, class name: {}".format(top5[1 - 1], synset[top5[1 - 1]]))

# Report top-5 classification results
print("\nTop5 predictions: \n")
print("\t#1:", synset[top5[1 - 1]])
print("\t#2:", synset[top5[2 - 1]])
print("\t#3:", synset[top5[3 - 1]])
print("\t#4:", synset[top5[4 - 1]])
print("\t#5:", synset[top5[5 - 1]])
print("\t", top5)
ImageNetClassifier = False
for k in top_categories[-5:]:
    if "cat" in synset[k]:
        ImageNetClassifier = True
assert ImageNetClassifier, "Failed ImageNet classifier validation check"

print("Evaluate inference time cost...")
print(m.benchmark(dev, number=1, repeat=10))
