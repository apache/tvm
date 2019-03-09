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
    wget https://raw.githubusercontent.com/dmlc/web-data/master/tensorflow/tflite/whl/tflite-0.0.1-py3-none-any.whl
    pip install tflite-0.0.1-py3-none-any.whl --user

    # For python2:
    wget https://raw.githubusercontent.com/dmlc/web-data/master/tensorflow/tflite/whl/tflite-0.0.1-py2-none-any.whl
    pip install tflite-0.0.1-py2-none-any.whl --user


or you could generate TFLite package by yourself. The steps are as following:

.. code-block:: bash

    # Get the flatc compiler.
    # Please refer to https://github.com/google/flatbuffers for details
    # and make sure it is properly installed.
    flatc --version

    # Get the TFLite schema.
    wget https://raw.githubusercontent.com/tensorflow/tensorflow/r1.12/tensorflow/contrib/lite/schema/schema.fbs

    # Generate TFLite package.
    flatc --python schema.fbs

    # Add it to PYTHONPATH.
    export PYTHONPATH=/path/to/tflite


Now please check if TFLite package is installed successfully, ``python -c "import tflite"``

Below you can find an example for how to compile TFLite model using TVM.
"""
######################################################################
# Utils for downloading and extracting zip files
# ---------------------------------------------

def download(url, path, overwrite=False):
    import os
    if os.path.isfile(path) and not overwrite:
        print('File {} existed, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
    except:
        import urllib
        urllib.urlretrieve(url, path)

def extract(path):
    import tarfile
    if path.endswith("tgz") or path.endswith("gz"):
        tar = tarfile.open(path)
        tar.extractall()
        tar.close()
    else:
        raise RuntimeError('Could not decompress the file: ' + path)


######################################################################
# Load pretrained TFLite model
# ---------------------------------------------
# we load mobilenet V1 TFLite model provided by Google
model_url = "http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224.tgz"

# we download model tar file and extract, finally get mobilenet_v1_1.0_224.tflite
download(model_url, "mobilenet_v1_1.0_224.tgz", False)
extract("mobilenet_v1_1.0_224.tgz")

# now we have mobilenet_v1_1.0_224.tflite on disk and open it
tflite_model_file = "mobilenet_v1_1.0_224.tflite"
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
download(image_url, 'cat.png')
resized_image = Image.open('cat.png').resize((224, 224))
plt.imshow(resized_image)
plt.show()
image_data = np.asarray(resized_image).astype("float32")

# convert HWC to CHW
image_data = image_data.transpose((2, 0, 1))

# after expand_dims, we have format NCHW
image_data = np.expand_dims(image_data, axis=0)

# preprocess image as described here:
# https://github.com/tensorflow/models/blob/edb6ed22a801665946c63d650ab9a0b23d98e1b1/research/slim/preprocessing/inception_preprocessing.py#L243
image_data[:, 0, :, :] = 2.0 / 255.0 * image_data[:, 0, :, :] - 1
image_data[:, 1, :, :] = 2.0 / 255.0 * image_data[:, 1, :, :] - 1
image_data[:, 2, :, :] = 2.0 / 255.0 * image_data[:, 2, :, :] - 1
print('input', image_data.shape)

####################################################################
#
# .. note:: Input layout:
#
#   Currently, TVM TFLite frontend accepts ``NCHW`` as input layout.

######################################################################
# Compile the model with relay
# ---------------------------------------------

# TFLite input tensor name, shape and type
input_tensor = "input"
input_shape = (1, 3, 224, 224)
input_dtype = "float32"

# parse TFLite model and convert into Relay computation graph
from tvm import relay
func, params = relay.frontend.from_tflite(tflite_model,
                                          shape_dict={input_tensor: input_shape},
                                          dtype_dict={input_tensor: input_dtype})

# targt x86 cpu
target = "llvm"
with relay.build_module.build_config(opt_level=3):
    graph, lib, params = relay.build(func, target, params=params)

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
download(label_file_url, label_file)

# map id to 1001 classes
labels = dict()
with open(label_file) as f:
    for id, line in enumerate(f):
        labels[id] = line

# convert result to 1D data
predictions = np.squeeze(tvm_output)

# get top 1 prediction
prediction = np.argmax(predictions)

# convert id to class name and show the result
print("The image prediction result is: id " + str(prediction) + " name: " + labels[prediction])
