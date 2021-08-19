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
**Author**: `Jiakui Hu <https://github.com/jkhu29/>`_

This article is an introductory tutorial to deploy OneFlow models with Relay.

For us to begin with, OneFlow should be installed.

A quick solution is to install via pip

.. code-block:: bash

    python3 -m pip install oneflow -f https://staging.oneflow.info/branch/master/[PLATFORM]

All available [PLATFORM] could be seen at official site:
https://github.com/Oneflow-Inc/oneflow

Currently, TVM supports OneFlow 0.5.0(nightly). Other versions may be unstable.
"""

import tvm
from tvm import relay
from tvm.contrib.download import download_testdata

import os, math
import numpy as np
from PIL import Image

# oneflow imports
import oneflow as flow
import oneflow.nn as nn
from oneflow import Tensor
from typing import Type, Any, Callable, Union, List, Optional

# prepare for psnr and ssim
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

######################################################################
# OneFlow model: SRGAN
# -------------------------------
# see more at https://github.com/Oneflow-Inc/oneflow_convert_tools/blob/tvm_oneflow/oneflow_tvm/
class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4), nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.PReLU()
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        block8.append(nn.Tanh())
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (block8 + 1.) / 2


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

######################################################################
# Load a pretrained OneFlow model
# -------------------------------
# We will download and load a pretrained provided in this example: SRGAN.
model_url = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/SRGAN_netG_epoch_4_99.zip"
model_file = "SRGAN_netG_epoch_4_99.zip"
model_path = download_testdata(model_url, model_file, module="oneflow")

os.system("unzip -q {}".format(model_path))
model_path = "SRGAN_netG_epoch_4_99"

sr_module = Generator(scale_factor=4)
pretrain_models = flow.load(model_path)
sr_module.load_state_dict(pretrain_models)
sr_module.eval().to("cuda")

######################################################################
# Load a test image
# ------------------
def load_image(image_path="", size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = np.ascontiguousarray(img).astype("float32") / 255
    img_flow = flow.Tensor(img).unsqueeze(0).permute(0, 3, 1, 2).to("cuda")
    return img_flow.numpy(), img_flow

img_url = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/monarchx4.png"
hr_url = "https://oneflow-static.oss-cn-beijing.aliyuncs.com/train_data_zjlab/monarch.png"
img_file = "monarchx4.png"
hr_file = "monarch.png"
img_path = download_testdata(img_url, img_file, module="data")
hr_path = download_testdata(hr_url, hr_file, module="data")
img, img_flow = load_image(img_path)

######################################################################
# Compile the model on Relay
# ---------------------------
# Convert OneFlow graph to Relay graph.
class Graph(flow.nn.Graph):
    def __init__(self, module):
        super().__init__()
        self.m = module

    def build(self, x):
        out = self.m(x)
        return out


graph = Graph(sr_module)
_ = graph._compile(img_flow)
mod, params = relay.frontend.from_oneflow(graph, model_path)

######################################################################
# Relay Build and Inference
# ---------------------------
# Convert OneFlow graph to Relay graph.
target = "cuda"
with tvm.transform.PassContext(opt_level=10):
    intrp = relay.build_module.create_executor("graph", mod, tvm.cuda(0), target)
dtype="float32"
tvm_output = intrp.evaluate()(tvm.nd.array(img.astype(dtype)), **params).numpy()

######################################################################
# Display results
# ---------------------------------------------
# show the SR result.
from matplotlib import pyplot as plt


tvm_output = flow.Tensor(tvm_output).squeeze(0).permute(1, 2, 0) * 255
tvm_img = tvm_output.numpy().astype(np.uint8)
plt.imshow(tvm_img)
plt.show()

######################################################################
# Compare the results
# ---------------------------
# Compare the evaluation indicators of oneflow and converted relay results.
with flow.no_grad():
    out = sr_module(img_flow)

for mode in ["oneflow", "tvm"]:
    if mode == "oneflow":
        out_a = out[0].data.to("cpu") * 255
        out_b = out_a.squeeze(0).permute(1, 2, 0)
        _img = out_b.numpy().astype(np.uint8)
    elif mode == "tvm":
        _img = tvm_img
    if hr_path != "":
        image_hr = np.array(Image.open(hr_path))
        psnr = peak_signal_noise_ratio(image_hr, _img)
        ssim = structural_similarity(image_hr, _img, multichannel=True)
        print("{}: psnr:{},ssim:{} \n".format(mode, psnr, ssim))
