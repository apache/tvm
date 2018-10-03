"""
Deploy Single Shot Multibox Detector(SSD) model
===============================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_

This article is an introductory tutorial to deploy SSD models with TVM.
We will use mxnet pretrained SSD model with Resnet50 as body network and
convert it to NNVM graph.
"""
import os
import zipfile
import tvm
import mxnet as mx
import cv2
import numpy as np

from nnvm import compiler
from nnvm.frontend import from_mxnet
from tvm.contrib.download import download
from tvm.contrib import graph_runtime
from mxnet.model import load_checkpoint


######################################################################
# Preliminary and Set parameters
# ------------------------------
# We should build TVM with sort support, in TVM root directory
#
# .. code-block:: bash
#
#   echo "set(USE_SORT ON)" > config.mk
#   make -j8
#
# .. note::
#
#   Currently we support compiling SSD on CPU only.
#   GPU support is in progress.
#

model_name = "ssd_resnet50_512"
model_file = "%s.zip" % model_name
test_image = "dog.jpg"
dshape = (1, 3, 512, 512)
dtype = "float32"
target = "llvm"
ctx = tvm.cpu()

######################################################################
# Download MXNet SSD pre-trained model and demo image
# ---------------------------------------------------
# Pre-trained model available at
# https://github.com/apache/incubator-\mxnet/tree/master/example/ssd

model_url = "https://github.com/zhreshold/mxnet-ssd/releases/download/v0.6/" \
            "resnet50_ssd_512_voc0712_trainval.zip"
image_url = "https://cloud.githubusercontent.com/assets/3307514/20012567/" \
            "cbb60336-a27d-11e6-93ff-cbc3f09f5c9e.jpg"
inference_symbol_folder = "c1904e900848df4548ce5dfb18c719c7-a28c4856c827fe766aa3da0e35bad41d44f0fb26"
inference_symbol_url = "https://gist.github.com/kevinthesun/c1904e900848df4548ce5dfb18c719c7/" \
                       "archive/a28c4856c827fe766aa3da0e35bad41d44f0fb26.zip"
            
dir = "ssd_model"
if not os.path.exists(dir):
    os.makedirs(dir)
model_file_path = "%s/%s" % (dir, model_file)
test_image_path = "%s/%s" % (dir, test_image)
inference_symbol_path = "%s/inference_model.zip" % dir
download(model_url, model_file_path)
download(image_url, test_image_path)
download(inference_symbol_url, inference_symbol_path)

zip_ref = zipfile.ZipFile(model_file_path, 'r')
zip_ref.extractall(dir)
zip_ref.close()
zip_ref = zipfile.ZipFile(inference_symbol_path)
zip_ref.extractall(dir)
zip_ref.close()

######################################################################
# Convert and compile model with NNVM for CPU.

sym = mx.sym.load("%s/%s/ssd_resnet50_inference.json" % (dir, inference_symbol_folder))
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
tvm_output = m.get_output(0)


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

