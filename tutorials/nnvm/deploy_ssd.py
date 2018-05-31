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
import mxnet as mx
import cv2
import numpy as np

from nnvm import compiler
from nnvm.frontend import from_mxnet
from tvm.contrib import graph_runtime
from mxnet.model import load_checkpoint


######################################################################
# Set the parameters here.

model_name = "ssd_resnet50_512"
model_file = "%s.zip" % model_name
test_image = "dog.jpg"
target = "llvm"
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
image_url = "https://cloud.githubusercontent.com/assets/3307514/20012567/" \
            "cbb60336-a27d-11e6-93ff-cbc3f09f5c9e.jpg"
            
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

sym = mx.sym.load("ssd/ssd_resnet50_inference.json")
_, arg_params, aux_params = load_checkpoint("%s/%s" % (dir, model_name), 0)
net, params = from_mxnet(sym, arg_params, aux_params)
with compiler.build_config(opt_level=3):
    graph, lib, params = compiler.build(net, target, {"data": dshape}, params=params)

######################################################################
# Create TVM runtime and do inference

# Preprocess image
image = cv2.imread(test_image_path)
img_data = cv2.resize(image, (dshape[2], dshape[3]))
img_data = img_data[:, :, (2, 1, 0)].astype(np.float32)
img_data -= np.array([123, 117, 104])
img_data = np.transpose(np.array(img_data), (2, 0, 1))
img_data = np.expand_dims(img_data, axis=0)
# Build TVM runtime
m = graph_runtime.create(graph, lib, ctx)
m.set_input('data', tvm.nd.array(img_data.astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
_, oshape = compiler.graph_util.infer_shape(graph, shape={"data": dshape})
tvm_output = m.get_output(0, tvm.nd.empty(tuple(oshape[0]), dtype))


######################################################################
# Display result

class_names = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair",
               "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
def display(img, out, thresh=0.5):
    import random
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams['figure.figsize'] = (10,10)
    pens = dict()
    plt.clf()
    plt.imshow(img)
    for det in out:
        cid = int(det[0])
        if cid < 0:
            continue
        score = det[1]
        if score < thresh:
            continue
        if cid not in pens:
            pens[cid] = (random.random(), random.random(), random.random())
        scales = [img.shape[1], img.shape[0]] * 2
        xmin, ymin, xmax, ymax = [int(p * s) for p, s in zip(det[2:6].tolist(), scales)]
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                             edgecolor=pens[cid], linewidth=3)
        plt.gca().add_patch(rect)
        text = class_names[cid]
        plt.gca().text(xmin, ymin-2, '{:s} {:.3f}'.format(text, score),
                       bbox=dict(facecolor=pens[cid], alpha=0.5),
                       fontsize=12, color='white')
    plt.show()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
display(image, tvm_output.asnumpy()[0], thresh=0.45)

