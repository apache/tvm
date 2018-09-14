"""
Compile CoreML Models
=====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_

This article is an introductory tutorial to deploy CoreML models with NNVM.

For us to begin with, coremltools module is required to be installed.

A quick solution is to install via pip

.. code-block:: bash

    pip install -U coremltools --user

or please refer to official site
https://github.com/apple/coremltools
"""
import nnvm
import tvm
import coremltools as cm
import numpy as np
from PIL import Image

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

######################################################################
# Load pretrained CoreML model
# ----------------------------
# We will download and load a pretrained mobilenet classification network
# provided by apple in this example
model_url = 'https://docs-assets.developer.apple.com/coreml/models/MobileNet.mlmodel'
model_file = 'mobilenet.mlmodel'
download(model_url, model_file)
# now you mobilenet.mlmodel on disk
mlmodel = cm.models.MLModel(model_file)
# we can load the graph as NNVM compatible model
sym, params = nnvm.frontend.from_coreml(mlmodel)

######################################################################
# Load a test image
# ------------------
# A single cat dominates the examples!
from PIL import Image
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
download(img_url, 'cat.png')
img = Image.open('cat.png').resize((224, 224))
#x = np.transpose(img, (2, 0, 1))[np.newaxis, :]
image = np.asarray(img)
image = image.transpose((2, 0, 1))
x = image[np.newaxis, :]
######################################################################
# Compile the model on NNVM
# ---------------------------
# We should be familiar with the process right now.
import nnvm.compiler
target = 'cuda'
shape_dict = {'image': x.shape}
with nnvm.compiler.build_config(opt_level=2, add_pass=['AlterOpLayout']):
    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

######################################################################
# Execute on TVM
# -------------------
# The process is no different from other example
from tvm.contrib import graph_runtime
ctx = tvm.gpu(0)
dtype = 'float32'
m = graph_runtime.create(graph, lib, ctx)
# set inputs
m.set_input('image', tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
tvm_output = m.get_output(0)
top1 = np.argmax(tvm_output.asnumpy()[0])

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
print('Top-1 id', top1, 'class name', synset[top1])
