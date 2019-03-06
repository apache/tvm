"""
Compile YOLO-V2 and YOLO-V3 in DarkNet Models
=================================
**Author**: `Siju Samuel <https://siju-samuel.github.io/>`_

This article is an introductory tutorial to deploy darknet models with NNVM.
All the required models and libraries will be downloaded from the internet by the script.
This script runs the YOLO-V2 and YOLO-V3 Model with the bounding boxes
Darknet parsing have dependancy with CFFI and CV2 library
Please install CFFI and CV2 before executing this script

.. code-block:: bash

  pip install cffi
  pip install opencv-python
"""

# sys, numpy and matplotlib
import sys
import numpy as np
import matplotlib.pyplot as plt
from ctypes import *

# tvm, relay
import tvm
from tvm import relay
import nnvm.testing.yolo_detection
import nnvm.testing.darknet

from tvm.contrib.download import download
from nnvm.testing.darknet import __darknetffi__

# Model name
MODEL_NAME = 'yolov3'

######################################################################
# Download required files
# -----------------------
# Download cfg and weights file if first time.
CFG_NAME = MODEL_NAME + '.cfg'
WEIGHTS_NAME = MODEL_NAME + '.weights'
REPO_URL = 'https://github.com/siju-samuel/darknet/blob/master/'
CFG_URL = REPO_URL + 'cfg/' + CFG_NAME + '?raw=true'
WEIGHTS_URL = 'https://pjreddie.com/media/files/' + WEIGHTS_NAME

download(CFG_URL, CFG_NAME)
download(WEIGHTS_URL, WEIGHTS_NAME)

# Download and Load darknet library
DARKNET_LIB = 'libdarknet2.0.so'
DARKNET_URL = REPO_URL + 'lib/' + DARKNET_LIB + '?raw=true'

download(DARKNET_URL, DARKNET_LIB)

DARKNET_LIB = __darknetffi__.dlopen('./' + DARKNET_LIB)
cfg = "./" + str(CFG_NAME)
weights = "./" + str(WEIGHTS_NAME)
net = DARKNET_LIB.load_network(cfg.encode('utf-8'), weights.encode('utf-8'), 0)
dtype = 'float32'
batch_size = 1

data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape_dict = {'data': data.shape}
print("Converting darknet to relay functions...")
sym, params = relay.frontend.from_darknet(net, dtype=dtype, shape=data.shape)

######################################################################
# Import the graph to Relay
# -------------------------
# compile the model
target = 'llvm'
target_host = 'llvm'
ctx = tvm.cpu(0)
data = np.empty([batch_size, net.c, net.h, net.w], dtype)
shape = {'data': data.shape}
print("Compiling the model...")
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(sym, target=target, target_host=target_host, params=params)

[neth, netw] = shape['data'][2:] # Current image shape is 608x608
######################################################################
# Load a test image
# --------------------------------------------------------------------
test_image = 'dog.jpg'
print("Loading the test image...")
img_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + \
          test_image + '?raw=true'
download(img_url, test_image)

data = nnvm.testing.darknet.load_image(test_image, netw, neth)
######################################################################
# Execute on TVM Runtime
# ----------------------
# The process is no different from other examples.
from tvm.contrib import graph_runtime

m = graph_runtime.create(graph, lib, ctx)

# set inputs
m.set_input('data', tvm.nd.array(data.astype(dtype)))
m.set_input(**params)
# execute
print("Running the test image...")

m.run()
# get outputs
tvm_out = []
if MODEL_NAME == 'yolov2':
    layer_out = {}
    layer_out['type'] = 'Region'
    # Get the region layer attributes (n, out_c, out_h, out_w, classes, coords, background)
    layer_attr = m.get_output(2).asnumpy()
    layer_out['biases'] = m.get_output(1).asnumpy()
    out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                 layer_attr[2], layer_attr[3])
    layer_out['output'] = m.get_output(0).asnumpy().reshape(out_shape)
    layer_out['classes'] = layer_attr[4]
    layer_out['coords'] = layer_attr[5]
    layer_out['background'] = layer_attr[6]
    tvm_out.append(layer_out)

elif MODEL_NAME == 'yolov3':
    for i in range(3):
        layer_out = {}
        layer_out['type'] = 'Yolo'
        # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
        layer_attr = m.get_output(i*4+3).asnumpy()
        layer_out['biases'] = m.get_output(i*4+2).asnumpy()
        layer_out['mask'] = m.get_output(i*4+1).asnumpy()
        out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                     layer_attr[2], layer_attr[3])
        layer_out['output'] = m.get_output(i*4).asnumpy().reshape(out_shape)
        layer_out['classes'] = layer_attr[4]
        tvm_out.append(layer_out)

# do the detection and bring up the bounding boxes
thresh = 0.5
nms_thresh = 0.45
img = nnvm.testing.darknet.load_image_color(test_image)
_, im_h, im_w = img.shape
dets = nnvm.testing.yolo_detection.fill_network_boxes((netw, neth), (im_w, im_h), thresh,
                                                      1, tvm_out)
last_layer = net.layers[net.n - 1]
nnvm.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, nms_thresh)

coco_name = 'coco.names'
coco_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + coco_name + '?raw=true'
font_name = 'arial.ttf'
font_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + font_name + '?raw=true'
download(coco_url, coco_name)
download(font_url, font_name)

with open(coco_name) as f:
    content = f.readlines()

names = [x.strip() for x in content]

nnvm.testing.yolo_detection.draw_detections(img, dets, thresh, names, last_layer.classes)
plt.imshow(img.transpose(1, 2, 0))
plt.show()
