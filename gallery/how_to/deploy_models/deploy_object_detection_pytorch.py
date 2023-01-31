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
Compile PyTorch Object Detection Models
=======================================
This article is an introductory tutorial to deploy PyTorch object
detection models with Relay VM.

For us to begin with, PyTorch should be installed.
TorchVision is also required since we will be using it as our model zoo.

A quick solution is to install via pip

.. code-block:: bash

    pip install torch
    pip install torchvision

or please refer to official site
https://pytorch.org/get-started/locally/

PyTorch versions should be backwards compatible but should be used
with the proper TorchVision version.

Currently, TVM supports PyTorch 1.7 and 1.4. Other versions may
be unstable.
"""

import tvm
from tvm import relay
from tvm import relay
from tvm.runtime.vm import VirtualMachine
from tvm.contrib.download import download_testdata

import numpy as np
import cv2

# PyTorch imports
import torch
import torchvision

######################################################################
# Load pre-trained maskrcnn from torchvision and do tracing
# ---------------------------------------------------------
in_size = 300

input_shape = (1, 3, in_size, in_size)


def do_trace(model, inp):
    model_trace = torch.jit.trace(model, inp)
    model_trace.eval()
    return model_trace


def dict_to_tuple(out_dict):
    if "masks" in out_dict.keys():
        return out_dict["boxes"], out_dict["scores"], out_dict["labels"], out_dict["masks"]
    return out_dict["boxes"], out_dict["scores"], out_dict["labels"]


class TraceWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inp):
        out = self.model(inp)
        return dict_to_tuple(out[0])


model_func = torchvision.models.detection.maskrcnn_resnet50_fpn
model = TraceWrapper(model_func(pretrained=True))

model.eval()
inp = torch.Tensor(np.random.uniform(0.0, 250.0, size=(1, 3, in_size, in_size)))

with torch.no_grad():
    out = model(inp)
    script_module = do_trace(model, inp)

######################################################################
# Download a test image and pre-process
# -------------------------------------
img_url = (
    "https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/detection/street_small.jpg"
)
img_path = download_testdata(img_url, "test_street_small.jpg", module="data")

img = cv2.imread(img_path).astype("float32")
img = cv2.resize(img, (in_size, in_size))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.transpose(img / 255.0, [2, 0, 1])
img = np.expand_dims(img, axis=0)

######################################################################
# Import the graph to Relay
# -------------------------
input_name = "input0"
shape_list = [(input_name, input_shape)]
mod, params = relay.frontend.from_pytorch(script_module, shape_list)

######################################################################
# Compile with Relay VM
# ---------------------
# Note: Currently only CPU target is supported. For x86 target, it is
# highly recommended to build TVM with Intel MKL and Intel OpenMP to get
# best performance, due to the existence of large dense operator in
# torchvision rcnn models.

# Add "-libs=mkl" to get best performance on x86 target.
# For x86 machine supports AVX512, the complete target is
# "llvm -mcpu=skylake-avx512 -libs=mkl"
target = "llvm"

with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"]):
    vm_exec = relay.vm.compile(mod, target=target, params=params)

######################################################################
# Inference with Relay VM
# -----------------------
dev = tvm.cpu()
vm = VirtualMachine(vm_exec, dev)
vm.set_input("main", **{input_name: img})
tvm_res = vm.run()

######################################################################
# Get boxes with score larger than 0.9
# ------------------------------------
score_threshold = 0.9
boxes = tvm_res[0].numpy().tolist()
valid_boxes = []
for i, score in enumerate(tvm_res[1].numpy().tolist()):
    if score > score_threshold:
        valid_boxes.append(boxes[i])
    else:
        break

print("Get {} valid boxes".format(len(valid_boxes)))
