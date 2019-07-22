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
Compile TFLite Models
===================
**Author**: `Zhao Wu <https://github.com/FrozenGene>`_

This article is an introductory tutorial to deploy TFLite models with Relay.

To get started, Flatbuffers and TFLite package needs to be installed as prerequisites.

A quick solution is to install Flatbuffers via pip

.. code-block:: bash

    pip install flatbuffers --user


To install TFlite packages, you could use our prebuilt wheel:

.. code-block:: bash

    # For python3:
    wget https://github.com/FrozenGene/tflite/releases/download/v1.13.1/tflite-1.13.1-py3-none-any.whl
    pip3 install -U tflite-1.13.1-py3-none-any.whl --user

    # For python2:
    wget https://github.com/FrozenGene/tflite/releases/download/v1.13.1/tflite-1.13.1-py2-none-any.whl
    pip install -U tflite-1.13.1-py2-none-any.whl --user


or you could generate TFLite package yourself. The steps are the following:

.. code-block:: bash

    # Get the flatc compiler.
    # Please refer to https://github.com/google/flatbuffers for details
    # and make sure it is properly installed.
    flatc --version

    # Get the TFLite schema.
    wget https://raw.githubusercontent.com/tensorflow/tensorflow/r1.13/tensorflow/lite/schema/schema.fbs

    # Generate TFLite package.
    flatc --python schema.fbs

    # Add current folder (which contains generated tflite module) to PYTHONPATH.
    export PYTHONPATH=${PYTHONPATH:+$PYTHONPATH:}$(pwd)


Now please check if TFLite package is installed successfully, ``python -c "import tflite"``

Below you can find an example on how to compile TFLite model using TVM.
"""
######################################################################
# Utils for downloading and extracting zip files
# ---------------------------------------------
import os

def extract(path):
    import tarfile
    if path.endswith("tgz") or path.endswith("gz"):
        dir_path = os.path.dirname(path)
        tar = tarfile.open(path)
        tar.extractall(path=dir_path)
        tar.close()
    else:
        raise RuntimeError('Could not decompress the file: ' + path)


######################################################################
# Load pretrained TFLite model
# ---------------------------------------------
# we load mobilenet V1 TFLite model provided by Google
from tvm.contrib.download import download_testdata

model_url = "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz"

# we download model tar file and extract, finally get mobilenet_v1_1.0_224.tflite
model_path = download_testdata(model_url, "mobilenet_v1_1.0_224.tgz", module=['tf', 'official'])
model_dir = os.path.dirname(model_path)
extract(model_path)

# now we have mobilenet_v1_1.0_224.tflite on disk and open it
tflite_model_file = os.path.join(model_dir, "mobilenet_v1_1.0_224.tflite")
tflite_model_buf = open(tflite_model_file, "rb").read()

# get TFLite model from buffer
import tflite.Model
tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

######################################################################
# Load a test image
# ---------------------------------------------
# A single cat dominates the examples!
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

image_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
image_path = download_testdata(image_url, 'cat.png', module='data')
resized_image = Image.open(image_path).resize((224, 224))
plt.imshow(resized_image)
plt.show()
image_data = np.asarray(resized_image).astype("float32")

# after expand_dims, we have format NHWC
image_data = np.expand_dims(image_data, axis=0)

# preprocess image as described here:
# https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243
image_data[:, :, :, 0] = 2.0 / 255.0 * image_data[:, :, :, 0] - 1
image_data[:, :, :, 1] = 2.0 / 255.0 * image_data[:, :, :, 1] - 1
image_data[:, :, :, 2] = 2.0 / 255.0 * image_data[:, :, :, 2] - 1
print('input', image_data.shape)

######################################################################
# Compile the model with relay
# ---------------------------------------------

# TFLite input tensor name, shape and type
input_tensor = "input"
input_shape = (1, 224, 224, 3)
input_dtype = "float32"

# parse TFLite model and convert into Relay computation graph
from tvm import relay
mod, params = relay.frontend.from_tflite(tflite_model,
                                         shape_dict={input_tensor: input_shape},
                                         dtype_dict={input_tensor: input_dtype})

# target x86 CPU
target = "llvm"
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(mod, target, params=params)

######################################################################
# Execute on TVM
# ---------------------------------------------
import tvm
from tvm.contrib import graph_runtime as runtime

# create a runtime executor module
module = runtime.create(graph, lib, tvm.cpu())

# feed input data
module.set_input(input_tensor, tvm.nd.array(image_data))

# feed related params
module.set_input(**params)

# run
module.run()

# get output
tvm_output = module.get_output(0).asnumpy()

######################################################################
# Display results
# ---------------------------------------------

# load label file
label_file_url = ''.join(['https://raw.githubusercontent.com/',
                          'tensorflow/tensorflow/master/tensorflow/lite/java/demo/',
                          'app/src/main/assets/',
                          'labels_mobilenet_quant_v1_224.txt'])
label_file = "labels_mobilenet_quant_v1_224.txt"
label_path = download_testdata(label_file_url, label_file, module='data')

# list of 1001 classes
with open(label_path) as f:
    labels = f.readlines()

# convert result to 1D data
predictions = np.squeeze(tvm_output)

# get top 1 prediction
prediction = np.argmax(predictions)

# convert id to class name and show the result
print("The image prediction result is: id " + str(prediction) + " name: " + labels[prediction])
