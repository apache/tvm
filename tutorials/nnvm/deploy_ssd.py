"""
Deploy Single Shot Multibox Detector(SSD) model
======================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_

This article is an introductory tutorial to deploy SSD models with TVM.
We will use mxnet pretrained SSD model with Resnet50 as body network and
convert it to NNVM graph.
"""
import os
import urllib
import zipfile
import tvm
import cv2
import numpy as np
import mxnet as mx

from nnvm import compiler
from nnvm.frontend import from_mxnet
from tvm.contrib import graph_runtime
from mxnet.model import load_checkpoint

######################################################################
# To get started, clone mxnet repo from github
# and extract ssd symbol directory:
#
#   .. code-block:: bash
#
#     git clone https://github.com/apache/incubator-mxnet mxnet
#     mkdir symbol && cp -a mxnet/example/ssd/symbol/* symbol


######################################################################
# Set the parameters here.
#

model_name = "ssd_resnet50_512"
model_file = "%s.zip" % model_name
test_image = "person.jpg"
target = "llvm -mcpu=core-avx2"
dshape = (1, 3, 512, 512)
dtype = "float32"
ctx = tvm.cpu()

def download(url, path, overwrite=False):
    """Downloads the file from the internet.
    Set the input options correctly to overwrite or do the size comparison

    Parameters
    ----------
    url : str
        Download file url
    path : str
        File saved path.
    overwrite : boolean
        Dict of operator attributes
    """
    if os.path.isfile(path) and not overwrite:
        print('File {} exists, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    try:
        urllib.request.urlretrieve(url, path)
        print('')
    except:
        urllib.urlretrieve(url, path)

######################################################################
# Download MXNet SSD pre-trained model and demo image.
# ----------------------------
# Pre-trained model available at
# https://github.com/apache/incubator-\mxnet/tree/master/example/ssd

model_url = "https://github.com/zhreshold/mxnet-ssd/releases/download/v0.6/" \
            "resnet50_ssd_512_voc0712_trainval.zip"
image_url = "https://cloud.githubusercontent.com/assets/3307514/20012563/" \
            "cbb41382-a27d-11e6-92a9-18dab4fd1ad3.jpg"
dir = "ssd_model"
if not os.path.exists(dir):
    os.makedirs(dir)
model_file_path = "%s/%s" % (dir, model_file)
test_image_path = "%s/%s" % (dir, test_image)
download(model_url, model_file_path)
download(image_url, test_image_path)
zip_ref = zipfile.ZipFile(model_file_path, 'r')
zip_ref.extractall(dir)
zip_ref.close()

######################################################################
# Convert and compile model with NNVM for CPU.

from symbol.symbol_factory import get_symbol
sym = get_symbol("resnet50", dshape[2], num_classes=20)
_, arg_params, aux_params = load_checkpoint("%s/%s" % (dir, model_name), 0)
net, params = from_mxnet(sym, arg_params, aux_params)
with compiler.build_config(opt_level=3):
    graph, lib, params = compiler.build(net, target, {"data": dshape}, params=params)

######################################################################
# Create TVM runtime and do inference

img_data = cv2.imread(test_image_path)
img_data = cv2.resize(img_data, (dshape[2], dshape[3]))
img_data = np.transpose(np.array(img_data), (2, 0, 1))
img_data = np.expand_dims(img_data, axis=0)
np_data = np.random.uniform(0, 255, size=dshape).astype(dtype)
m = graph_runtime.create(graph, lib, ctx)
m.set_input('data', tvm.nd.array(img_data.astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
_, oshape = compiler.graph_util.infer_shape(graph, {"data": dshape})
tvm_output = m.get_output(0, tvm.nd.empty(oshape, dtype))
print(tvm_output.shape)
