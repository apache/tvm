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
Compile PaddlePaddle Models
===========================
**Author**: `Ziyuan Ma <https://github.com/ZiyuanMa/>`_

This article is an introductory tutorial to deploy PaddlePaddle models with Relay.
To begin, we'll install PaddlePaddle>=2.1.3:

.. code-block:: bash

    %%shell
    pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple

For more details, refer to the official install instructions at:
https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html
"""

import tarfile
import paddle
import numpy as np
import tvm
from tvm import relay
from tvm.contrib.download import download_testdata

######################################################################
# Load pretrained ResNet50 model
# ---------------------------------------------
# We load a pretrained ResNet50 provided by PaddlePaddle.
url = "https://bj.bcebos.com/x2paddle/models/paddle_resnet50.tar"
model_path = download_testdata(url, "paddle_resnet50.tar", module="model")

with tarfile.open(model_path) as tar:
    names = tar.getnames()
    for name in names:
        tar.extract(name, "./")

model = paddle.jit.load("./paddle_resnet50/model")

######################################################################
# Load a test image
# ---------------------------------------------
# A single cat dominates the examples!

from PIL import Image
import paddle.vision.transforms as T


transforms = T.Compose(
    [
        T.Resize((256, 256)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
img_path = download_testdata(img_url, "cat.png", module="data")
img = Image.open(img_path).resize((224, 224))

img = transforms(img)
img = np.expand_dims(img, axis=0)

######################################################################
# Compile the model with relay
# ---------------------------------------------

target = "llvm"
shape_dict = {"inputs": img.shape}
mod, params = relay.frontend.from_paddle(model, shape_dict)

with tvm.transform.PassContext(opt_level=3):
    executor = relay.build_module.create_executor(
        "graph", mod, tvm.cpu(0), target, params
    ).evaluate()

######################################################################
# Execute on TVM
# ---------------------------------------------
dtype = "float32"
tvm_output = executor(tvm.nd.array(img.astype(dtype))).numpy()

######################################################################
# Look up synset name
# ---------------------------------------------
# Look up prediction top 1 index in 1000 class synset.

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
    synset = f.readlines()

top1 = np.argmax(tvm_output[0])
print(f"TVM prediction top-1 id: {top1}, class name: {synset[top1]}")
