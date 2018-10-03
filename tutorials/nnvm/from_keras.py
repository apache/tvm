"""
Compile Keras Models
=====================
**Author**: `Yuwei Hu <https://Huyuwei.github.io/>`_

This article is an introductory tutorial to deploy keras models with NNVM.

For us to begin with, keras should be installed.
Tensorflow is also required since it's used as the default backend of keras.

A quick solution is to install via pip

.. code-block:: bash

    pip install -U keras --user
    pip install -U tensorflow --user

or please refer to official site
https://keras.io/#installation
"""
import nnvm
import tvm
import keras
import numpy as np

def download(url, path, overwrite=False):
    import os
    if os.path.isfile(path) and not overwrite:
        print('File {} exists, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
    except:
        import urllib
        urllib.urlretrieve(url, path)

######################################################################
# Load pretrained keras model
# ----------------------------
# We load a pretrained resnet-50 classification model provided by keras.
weights_url = ''.join(['https://github.com/fchollet/deep-learning-models/releases/',
                       'download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'])
weights_file = 'resnet50_weights.h5'
download(weights_url, weights_file)
keras_resnet50 = keras.applications.resnet50.ResNet50(include_top=True, weights=None,
                                                      input_shape=(224, 224, 3), classes=1000)
keras_resnet50.load_weights('resnet50_weights.h5')

######################################################################
# Load a test image
# ------------------
# A single cat dominates the examples!
from PIL import Image
from matplotlib import pyplot as plt
from keras.applications.resnet50 import preprocess_input
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
download(img_url, 'cat.png')
img = Image.open('cat.png').resize((224, 224))
plt.imshow(img)
plt.show()
# input preprocess
data = np.array(img)[np.newaxis, :].astype('float32')
data = preprocess_input(data).transpose([0, 3, 1, 2])
print('input_1', data.shape)

######################################################################
# Compile the model on NNVM
# --------------------------
# We should be familiar with the process now.

# convert the keras model(NHWC layout) to NNVM format(NCHW layout).
sym, params = nnvm.frontend.from_keras(keras_resnet50)
# compile the model
target = 'cuda'
shape_dict = {'input_1': data.shape}
with nnvm.compiler.build_config(opt_level=3):
    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

######################################################################
# Execute on TVM
# ---------------
# The process is no different from other examples.
from tvm.contrib import graph_runtime
ctx = tvm.gpu(0)
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input('input_1', tvm.nd.array(data.astype('float32')))
m.set_input(**params)
# execute
m.run()
# get outputs
tvm_out = m.get_output(0)
top1_tvm = np.argmax(tvm_out.asnumpy()[0])

#####################################################################
# Look up synset name
# -------------------
# Look up prediction top 1 index in 1000 class synset.
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'synset.txt'
download(synset_url, synset_name)
with open(synset_name) as f:
    synset = eval(f.read())
print('NNVM top-1 id: {}, class name: {}'.format(top1_tvm, synset[top1_tvm]))
# confirm correctness with keras output
keras_out = keras_resnet50.predict(data.transpose([0, 2, 3, 1]))
top1_keras = np.argmax(keras_out)
print('Keras top-1 id: {}, class name: {}'.format(top1_keras, synset[top1_keras]))
