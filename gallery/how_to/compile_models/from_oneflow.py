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
Compile OneFlow Models
======================
**Author**: `Xiaoyu Zhang <https://github.com/BBuf/>`_

This article is an introductory tutorial to deploy OneFlow models with Relay.

For us to begin with, OneFlow package should be installed.

A quick solution is to install via pip

.. code-block:: bash

    %%shell
    pip install flowvision==0.1.0
    pip install -f https://release.oneflow.info oneflow==0.7.0+cpu

or please refer to official site:
https://github.com/Oneflow-Inc/oneflow

Currently, TVM supports OneFlow 0.7.0. Other versions may be unstable.
"""

# sphinx_gallery_start_ignore
# sphinx_gallery_requires_cuda = True
# sphinx_gallery_end_ignore
import os, math
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

# oneflow imports
import flowvision
import oneflow as flow
import oneflow.nn as nn

import tvm
from tvm import relay
from tvm.contrib.download import download_testdata

######################################################################
# Load a pretrained OneFlow model and save model
# ----------------------------------------------
model_name = "resnet18"
model = getattr(flowvision.models, model_name)(pretrained=True)
model = model.eval()

model_dir = "resnet18_model"
if not os.path.exists(model_dir):
    flow.save(model.state_dict(), model_dir)

######################################################################
# Load a test image
# -----------------
# Classic cat example!
from PIL import Image

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

# Preprocess the image and convert to tensor
from flowvision import transforms

my_preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img.numpy(), 0)

######################################################################
# Import the graph to Relay
# -------------------------
# Convert OneFlow graph to Relay graph. The input name can be arbitrary.
class Graph(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.m = module

    def build(self, x):
        out = self.m(x)
        return out


graph = Graph(model)
_ = graph._compile(flow.randn(1, 3, 224, 224))

mod, params = relay.frontend.from_oneflow(graph, model_dir)

######################################################################
# Relay Build
# -----------
# Compile the graph to llvm target with given input specification.
target = tvm.target.Target("llvm", host="llvm")
dev = tvm.cpu(0)
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target, params=params)

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now we can try deploying the compiled model on target.
target = "cuda"
with tvm.transform.PassContext(opt_level=10):
    intrp = relay.build_module.create_executor("graph", mod, tvm.cuda(0), target)

print(type(img))
print(img.shape)
tvm_output = intrp.evaluate()(tvm.nd.array(img.astype("float32")), **params)

#####################################################################
# Look up synset name
# -------------------
# Look up prediction top 1 index in 1000 class synset.
synset_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_synsets.txt",
    ]
)
synset_name = "imagenet_synsets.txt"
synset_path = download_testdata(synset_url, synset_name, module="data")
with open(synset_path) as f:
    synsets = f.readlines()

synsets = [x.strip() for x in synsets]
splits = [line.split(" ") for line in synsets]
key_to_classname = {spl[0]: " ".join(spl[1:]) for spl in splits}

class_url = "".join(
    [
        "https://raw.githubusercontent.com/Cadene/",
        "pretrained-models.pytorch/master/data/",
        "imagenet_classes.txt",
    ]
)
class_name = "imagenet_classes.txt"
class_path = download_testdata(class_url, class_name, module="data")
with open(class_path) as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

# Get top-1 result for TVM
top1_tvm = np.argmax(tvm_output.numpy()[0])
tvm_class_key = class_id_to_key[top1_tvm]

# Convert input to OneFlow variable and get OneFlow result for comparison
with flow.no_grad():
    torch_img = flow.from_numpy(img)
    output = model(torch_img)

    # Get top-1 result for OneFlow
    top_oneflow = np.argmax(output.numpy())
    oneflow_class_key = class_id_to_key[top_oneflow]

print("Relay top-1 id: {}, class name: {}".format(top1_tvm, key_to_classname[tvm_class_key]))
print(
    "OneFlow top-1 id: {}, class name: {}".format(top_oneflow, key_to_classname[oneflow_class_key])
)
