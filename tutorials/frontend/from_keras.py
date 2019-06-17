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
Compile Keras Models
=====================
**Author**: `Yuwei Hu <https://Huyuwei.github.io/>`_

This article is an introductory tutorial to deploy keras models with Relay.

For us to begin with, keras should be installed.
Tensorflow is also required since it's used as the default backend of keras.

A quick solution is to install via pip

.. code-block:: bash

    pip install -U keras --user
    pip install -U tensorflow --user

or please refer to official site
https://keras.io/#installation
"""
import tvm
import tvm.relay as relay
from tvm.contrib.download import download_testdata
import keras
import numpy as np

######################################################################
# Load pretrained keras model
# ----------------------------
# We load a pretrained resnet-50 classification model provided by keras.
weights_url = ''.join(['https://github.com/fchollet/deep-learning-models/releases/',
                       'download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'])
weights_file = 'resnet50_weights.h5'
weights_path = download_testdata(weights_url, weights_file, module='keras')
keras_resnet50 = keras.applications.resnet50.ResNet50(include_top=True, weights=None,
                                                      input_shape=(224, 224, 3), classes=1000)
keras_resnet50.load_weights(weights_path)

######################################################################
# Load a test image
# ------------------
# A single cat dominates the examples!
from PIL import Image
from matplotlib import pyplot as plt
from keras.applications.resnet50 import preprocess_input
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_path = download_testdata(img_url, 'cat.png', module='data')
img = Image.open(img_path).resize((224, 224))
plt.imshow(img)
plt.show()
# input preprocess
data = np.array(img)[np.newaxis, :].astype('float32')
data = preprocess_input(data).transpose([0, 3, 1, 2])
print('input_1', data.shape)

######################################################################
# Compile the model with Relay
# ----------------------------
# convert the keras model(NHWC layout) to Relay format(NCHW layout).
shape_dict = {'input_1': data.shape}
mod, params = relay.frontend.from_keras(keras_resnet50, shape_dict)
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
# confirm correctness with keras output
keras_out = keras_resnet50.predict(data.transpose([0, 2, 3, 1]))
top1_keras = np.argmax(keras_out)
print('Keras top-1 id: {}, class name: {}'.format(top1_keras, synset[top1_keras]))
