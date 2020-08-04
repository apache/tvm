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
Compile Caffe Models
**Author**: `Chen Feng <https://github.com/fernchen>`_

This article is an introductory tutorial to deploy Caffe models with relay.

To get started, Caffe package needs to be installed.

In order to install Caffe, please refer to https://caffe.berkeleyvision.org/installation.html.

If your os is ubuntu version >= 17,04, pre-compiled caffe can be install by:

.. code-block:: bash
    
    # install caffe
    sudo apt install caffe-cpu


Below you can file an example on how to compile Caffe model using TVM.
"""
######################################################################
# Download pretrained Caffe model
# ----------------------------------------------
from tvm.contrib.download import download_testdata

proto_file_url = ("https://github.com/shicai/MobileNet-Caffe/raw/"
                        "master/mobilenet_v2_deploy.prototxt")
blob_file_url = ("https://github.com/shicai/MobileNet-Caffe/blob/"
                        "master/mobilenet_v2.caffemodel?raw=true")

proto_file = download_testdata(proto_file_url, "mobilenetv2.prototxt", module="model")
blob_file = download_testdata(blob_file_url, "mobilenetv2.caffemodel", module="model")

######################################################################
# Load a test image
# -----------------
# A single cat dominates the examples!
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

image_url = "https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true"
image_path = download_testdata(image_url, 'cat.png', module='data')
resized_image = Image.open(image_path).resize((224, 224))
plt.imshow(resized_image)
plt.show()
image_data = np.asarray(resized_image).astype(np.float32)

# Preprocess image
image_data -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
image_data /= 58.8

# RGB2BGR for Caffe model
image_data = image_data[..., ::-1]

# Add a dimension to the image and transpose it so that we have NCHW format layout
image_data = np.expand_dims(image_data, axis=0)
image_data = np.transpose(image_data, (0, 3, 1, 2))

######################################################################
# Compile the model with relay
# ----------------------------
from google.protobuf import text_format
import caffe
from caffe.proto import caffe_pb2 as pb

# Get Caffe model from buffer
init_net = pb.NetParameter()
predict_net = pb.NetParameter()
with open(proto_file, 'r') as f:
    text_format.Merge(f.read(), predict_net)
with open(blob_file, 'rb') as f:
    init_net.ParseFromString(f.read())
# Caffe model input layer name, shape and dtype
shape_dict = {'data': image_data.shape}
dtype_dict = {'data': 'float32'}

import tvm
from tvm import relay
mod, params = relay.frontend.from_caffe(init_net, predict_net, shape_dict, dtype_dict)

# Build the module against to x86 CPU
target = 'llvm'
with tvm.transform.PassContext(opt_level=3):
    lib = relay.build(mod, target=target,params=params)

######################################################################
# Execute on TVM
# --------------
from tvm.contrib import graph_runtime

# Create a runtime executor module
ctx = tvm.cpu(0)
m = graph_runtime.GraphModule(lib['default'](ctx))

# Feed input data
m.set_input('data', tvm.nd.array(image_data))

# Feed related params
m.set_input(**params)

# execute
m.run()

# Get output
tvm_output = m.get_output(0).asnumpy()

######################################################################
# Display results
# ---------------

# Load label file
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'imagenet1000_clsid_to_human.txt'
synset_path = download_testdata(synset_url, synset_name, module='data')
with open(synset_path) as f:
    synset = eval(f.read())

# Get Top1 prediction
predictions = np.squeeze(tvm_output)
prediction = np.argmax(predictions)

# Convert id to class name and show the result
print("The image prediction result is: id " + str(prediction) + " name: " + synset[prediction])
