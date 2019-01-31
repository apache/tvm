"""
Compile Caffe2 Models
=====================
**Author**: `Hiroyuki Makino <https://makihiro.github.io/>`_

This article is an introductory tutorial to deploy Caffe2 models with Relay.

For us to begin with, Caffe2 should be installed.

A quick solution is to install via pip

.. code-block:: bash



or please refer to official site
https://caffe2.ai/docs/getting-started.html
"""
import tvm
from tvm import relay
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
# Load pretrained caffe2 model
# ----------------------------
# We load a pretrained resnet50 classification model provided by caffe2.
from caffe2.python.models.download import ModelDownloader
mf = ModelDownloader()

class Model:
    def __init__(self, model_name):
        self.init_net, self.predict_net, self.value_info = mf.get_c2_model(model_name)

resnet50 = Model('resnet50')


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
# Compile the model on Relay
# --------------------------
# We should be familiar with the process now.

# convert the caffe2 model(NHWC layout) to relay functions.
target = 'cuda'
shape_dict = {'input_1': data.shape}
dtype_dict = {'input_1': data.dtype}
func, params = relay.frontend.from_caffe2(resnet50.init_net, resnet50.predict_net, shape_dict, dtype_dict)
# compile the model
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(func, target, params=params)

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
from caffe2.python import workspace
synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'synset.txt'
download(synset_url, synset_name)
with open(synset_name) as f:
    synset = eval(f.read())
print('Relay top-1 id: {}, class name: {}'.format(top1_tvm, synset[top1_tvm]))
# confirm correctness with caffe2 output
p = workspace.Predictor(resnet50.init_net, resnet50.predict_net)
caffe2_out = p.run({'data': data.transpose([0, 2, 3, 1])})
top1_caffe2 = np.argmax(caffe2_out)
print('Caffe2 top-1 id: {}, class name: {}'.format(top1_caffe2, synset[top1_caffe2]))
