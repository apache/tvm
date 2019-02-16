"""
.. _tutorial-from-mxnet:

Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_, `Eddie Yan <https://github.com/eqy>`_

This article is an introductory tutorial to deploy mxnet models with Relay.

For us to begin with, mxnet module is required to be installed.

A quick solution is

.. code-block:: bash

    pip install mxnet --user

or please refer to offical installation guide.
https://mxnet.incubator.apache.org/versions/master/install/index.html
"""
# some standard imports
import mxnet as mx
from tvm import relay
import tvm
import numpy as np

######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.utils import download
from PIL import Image
from matplotlib import pyplot as plt
block = get_model('resnet18_v1', pretrained=True)

img_name = 'cat.png'
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'synset.txt'
download('https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true', img_name)
download(synset_url, synset_name)
with open(synset_name) as f:
    synset = eval(f.read())
image = Image.open(img_name).resize((224, 224))
plt.imshow(image)
plt.show()

def transform_image(image):
    image = np.array(image) - np.array([123., 117., 104.])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

x = transform_image(image)
print('x', x.shape)

######################################################################
# Compile the Graph
# -----------------
# Now we would like to port the Gluon model to a portable computational graph.
# It's as easy as several lines.
# We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
input_shape = (1, 3, 224, 224)
dtype = 'float32'
net, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
# we want a probability so add a softmax operator
net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)

######################################################################
# now compile the graph
target = 'cuda'
shape_dict = {'data': x.shape}
with relay.build_config(opt_level=3):
    intrp = relay.build_module.create_executor('graph', net, tvm.gpu(0), target) 

######################################################################
# Execute the portable graph on TVM
# ---------------------------------
# Now, we would like to reproduce the same forward computation using TVM.
tvm_output = intrp.evaluate(net)(tvm.nd.array(x.astype(dtype)), **params)
top1 = np.argmax(tvm_output.asnumpy()[0])
print('TVM prediction top-1:', top1, synset[top1])
