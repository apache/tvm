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
Compile Chainer Models
=====================
**Author**: `Anshuman Tripathy <https://github.com/ANSHUMAN87>`_

This article is an introductory tutorial to deploy chainer models with Relay.

For us to begin with, chainer should be installed.
chainercv also required as it has collection of multiple pretrained models.

A quick solution is to install via pip

.. code-block:: bash

    pip install -U chainer --user
    pip install -U chainercv --user

or please refer to official site
https://docs.chainer.org/en/stable/install.html
"""
import chainer
from chainer import function
import chainercv.links as CV
import numpy as np

import tvm
import tvm.relay as relay
from tvm.contrib.download import download_testdata

######################################################################
# Some utilities
# ----------------
# Image preprocessing utilities
from PIL import Image
from chainercv.transforms import center_crop
from chainercv.transforms import resize
from chainercv.transforms import scale
from chainercv.transforms import ten_crop

def read_image(file_name, dtype='float32', color=True):
    f = Image.open(file_name)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('L')
        img = np.array(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))

def preprocess_image(img, model, crop_size, scale_size=None,
                     crop='center', mean=None):
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)

    if mean is None:
        mean = model.mean

    if scale_size is not None:
        if isinstance(scale_size, int):
            img = scale(img, size=scale_size)
        else:
            img = resize(img, size=scale_size)
    else:
        img = img.copy()

    if crop == '10':
        imgs = ten_crop(img, crop_size)
    elif crop == 'center':
        imgs = center_crop(img, crop_size)[np.newaxis]

    imgs -= mean[np.newaxis]

    return imgs


######################################################################
# Load pretrained chainer model
# ----------------------------
# We load a pretrained VGG16 imagenet classification model provided by chainercv.
vgg16_model = CV.VGG16(pretrained_model='imagenet')

######################################################################
# Load a test image
# ------------------
# All time favourite cat example
from matplotlib import pyplot as plt

img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_path = download_testdata(img_url, 'cat.png', module='data')
img = read_image(img_path)

plt.imshow(Image.open(img_path))
plt.show()

# input preprocess
data = preprocess_image(img, vgg16_model, crop_size=224, scale_size=256, crop='center')
print('input_1', data.shape)

######################################################################
# Compile the model with Relay
# ----------------------------
shape_dict = {'input_1': data.shape}
dtype_dict = {'input_1': data.dtype}
mod, params = relay.frontend.from_chainer(vgg16_model, shape_dict, dtype_dict)
# compile the model
target = 'cuda'
ctx = tvm.gpu(0)
with relay.build_config(opt_level=3):
    executor = relay.build_module.create_executor('graph', mod, ctx, target)

######################################################################
# Execute on TVM
# ---------------
dtype = 'float32'
tvm_out = executor.evaluate()(tvm.nd.array(data.astype(dtype)), **params)
top1_tvm = np.argmax(tvm_out.asnumpy()[0])

#####################################################################
# Look up synset name
# -------------------
# Look up prediction top 1 index in 1000 class synset.
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'imagenet1000_clsid_to_human.txt'
synset_path = download_testdata(synset_url, synset_name, module='data')
with open(synset_path) as f:
    synset = eval(f.read())
print('Relay top-1 id: {}, class name: {}'.format(top1_tvm, synset[top1_tvm]))

# confirm correctness with chainer output
with function.force_backprop_mode(), chainer.using_config('train', False):
    output = vgg16_model(data)

top1_chainer = np.argmax(output.array[0])
print('Chainer top-1 id: {}, class name: {}'.format(top1_chainer, synset[top1_chainer]))
