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
.. _tutorial-micro-Pytorch:

microTVM PyTorch Tutorial
===========================
**Authors**:
`Mehrdad Hessar <https://github.com/mehrdadh>`_

This tutorial is showcasing microTVM host-driven AoT compilation with
a PyTorch model. This tutorial can be executed on a x86 CPU using C runtime (CRT).

**Note:** This tutorial only runs on x86 CPU using CRT and does not run on Zephyr
since the model would not fit on our current supported Zephyr boards.
"""

# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore

import pathlib

import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image

import tvm
from tvm import relay
from tvm.contrib.download import download_testdata
from tvm.relay.backend import Executor

##################################
# Load a pre-trained PyTorch model
# --------------------------------
#
# To begin with, load pre-trained MobileNetV2 from torchvision. Then,
# download a cat image and preprocess it to use as the model input.
#

model = torchvision.models.quantization.mobilenet_v2(weights="DEFAULT", quantize=True)
model = model.eval()

input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# Preprocess the image and convert to tensor
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

input_name = "input0"
shape_list = [(input_name, input_shape)]
relay_mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

#####################################
# Define Target, Runtime and Executor
# -----------------------------------
#
# In this tutorial we use AOT host-driven executor. To compile the model
# for an emulated embedded environment on an x86 machine we use C runtime (CRT)
# and we use `host` micro target. Using this setup, TVM compiles the model
# for C runtime which can run on a x86 CPU machine with the same flow that
# would run on a physical microcontroller.
#


# Simulate a microcontroller on the host machine. Uses the main() from `src/runtime/crt/host/main.cc`
# To use physical hardware, replace "host" with another physical micro target, e.g. `nrf52840`
# or `mps2_an521`. See more more target examples in micro_train.py and micro_tflite.py tutorials.
target = tvm.target.target.micro("host")

# Use the C runtime (crt) and enable static linking by setting system-lib to True
runtime = tvm.relay.backend.Runtime("crt", {"system-lib": True})

# Use the AOT executor rather than graph or vm executors. Don't use unpacked API or C calling style.
executor = Executor("aot")

####################
# Compile the model
# ------------------
#
# Now, we compile the model for the target:
#

with tvm.transform.PassContext(
    opt_level=3,
    config={"tir.disable_vectorize": True},
):
    module = tvm.relay.build(
        relay_mod, target=target, runtime=runtime, executor=executor, params=params
    )

###########################
# Create a microTVM project
# -------------------------
#
# Now that we have the compiled model as an IRModule, we need to create a firmware project
# to use the compiled model with microTVM. To do this, we use Project API.
#

template_project_path = pathlib.Path(tvm.micro.get_microtvm_template_projects("crt"))
project_options = {"verbose": False, "memory_size_bytes": 6 * 1024 * 1024}

temp_dir = tvm.contrib.utils.tempdir() / "project"
project = tvm.micro.generate_project(
    str(template_project_path),
    module,
    temp_dir,
    project_options,
)

####################################
# Build, flash and execute the model
# ----------------------------------
# Next, we build the microTVM project and flash it. Flash step is specific to
# physical microcontroller and it is skipped if it is simulating a microcontroller
# via the host `main.cc`` or if a Zephyr emulated board is selected as the target.
#

project.build()
project.flash()

input_data = {input_name: tvm.nd.array(img.astype("float32"))}
with tvm.micro.Session(project.transport()) as session:
    aot_executor = tvm.runtime.executor.aot_executor.AotModule(session.create_aot_executor())
    aot_executor.set_input(**input_data)
    aot_executor.run()
    result = aot_executor.get_output(0).numpy()

#####################
# Look up synset name
# -------------------
# Look up prediction top 1 index in 1000 class synset.
#

synset_url = (
    "https://raw.githubusercontent.com/Cadene/"
    "pretrained-models.pytorch/master/data/"
    "imagenet_synsets.txt"
)
synset_name = "imagenet_synsets.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synsets = f.readlines()

synsets = [x.strip() for x in synsets]
splits = [line.split(" ") for line in synsets]
key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

class_url = (
    "https://raw.githubusercontent.com/Cadene/"
    "pretrained-models.pytorch/master/data/"
    "imagenet_classes.txt"
)
class_path = download_testdata(class_url, "imagenet_classes.txt", module="data")
with open(class_path) as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

# Get top-1 result for TVM
top1_tvm = np.argmax(result)
tvm_class_key = class_id_to_key[top1_tvm]

# Convert input to PyTorch variable and get PyTorch result for comparison
with torch.no_grad():
    torch_img = torch.from_numpy(img)
    output = model(torch_img)

    # Get top-1 result for PyTorch
    top1_torch = np.argmax(output.numpy())
    torch_class_key = class_id_to_key[top1_torch]

print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
print("Torch top-1 id: {}, class name: {}".format(top1_torch, key_to_classname[torch_class_key]))
