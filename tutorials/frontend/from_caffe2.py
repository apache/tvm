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
Compile Caffe2 Models
=====================
**Author**: `Hiroyuki Makino <https://makihiro.github.io/>`_

This article is an introductory tutorial to deploy Caffe2 models with Relay.

For us to begin with, Caffe2 should be installed.

A quick solution is to install via conda

.. code-block:: bash

    # for cpu
    conda install pytorch-nightly-cpu -c pytorch
    # for gpu with CUDA 8
    conda install pytorch-nightly cuda80 -c pytorch

or please refer to official site
https://caffe2.ai/docs/getting-started.html
"""

######################################################################
# Load pretrained Caffe2 model
# ----------------------------
# We load a pretrained resnet50 classification model provided by Caffe2.
from caffe2.python.models.download import ModelDownloader

mf = ModelDownloader()


class Model:
    def __init__(self, model_name):
        self.init_net, self.predict_net, self.value_info = mf.get_c2_model(model_name)


resnet50 = Model("resnet50")

######################################################################
# Load a test image
# ------------------
# A single cat dominates the examples!
from tvm.contrib.download import download_testdata
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

img_url = "https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))
plt.imshow(img)
plt.show()
# input preprocess
def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :].astype("float32")
    return image


data = transform_image(img)

######################################################################
# Compile the model on Relay
# --------------------------

# Caffe2 input tensor name, shape and type
input_name = resnet50.predict_net.op[0].input[0]
shape_dict = {input_name: data.shape}
dtype_dict = {input_name: data.dtype}

# parse Caffe2 model and convert into Relay computation graph
from tvm import relay, transform

mod, params = relay.frontend.from_caffe2(
    resnet50.init_net, resnet50.predict_net, shape_dict, dtype_dict
)

# compile the model
# target x86 CPU
target = "llvm"
with transform.PassContext(opt_level=3):
    lib = relay.build(mod, target, params=params)

######################################################################
# Execute on TVM
# ---------------
# The process is no different from other examples.
import tvm
from tvm import te
from tvm.contrib import graph_runtime

# context x86 CPU, use tvm.gpu(0) if you run on GPU
ctx = tvm.cpu(0)
# create a runtime executor module
m = graph_runtime.GraphModule(lib["default"](ctx))
# set inputs
m.set_input(input_name, tvm.nd.array(data.astype("float32")))
# execute
m.run()
# get outputs
tvm_out = m.get_output(0)
top1_tvm = np.argmax(tvm_out.asnumpy()[0])

#####################################################################
# Look up synset name
# -------------------
# Look up prediction top 1 index in 1000 class synset.
from caffe2.python import workspace

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
print("Relay top-1 id: {}, class name: {}".format(top1_tvm, synset[top1_tvm]))
# confirm correctness with caffe2 output
p = workspace.Predictor(resnet50.init_net, resnet50.predict_net)
caffe2_out = p.run({input_name: data})
top1_caffe2 = np.argmax(caffe2_out)
print("Caffe2 top-1 id: {}, class name: {}".format(top1_caffe2, synset[top1_caffe2]))
