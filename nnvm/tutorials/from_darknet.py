"""
Tutorial for running Yolo-V2 in Darknet Models
=====================
**Author**: `Siju Samuel <https://siju-samuel.github.io/>`_

This article is an introductory tutorial to deploy darknet models with NNVM.

All the required models and libraries will be downloaded from the internet

by the script.

This script runs the YOLO-V2 Model with the bounding boxes

Darknet parsing have dependancy with CFFI and CV2 library

Please install CFFI and CV2 before executing this script

pip install cffi

pip install opencv-python
"""
from ctypes import *
import math
import random
import nnvm
import nnvm.frontend.darknet
from nnvm.testing.darknet import __darknetffi__
import matplotlib.pyplot as plt
import numpy as np
import tvm
import os, sys, time, urllib, requests
if sys.version_info >= (3,):
    import urllib.request as urllib2
    import urllib.parse as urlparse
else:
    import urllib2
    import urlparse

######################################################################
# Set the parameters here.
# Supported models alexnet, resnet50, resnet152, extraction, yolo
######################################################################
model_name = 'yolo'
test_image = 'dog.jpg'
target = 'llvm'
ctx = tvm.cpu(0)
######################################################################

def dlProgress(count, block_size, total_size):
    """Show the download progress."""
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
          (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

def download(url, path, overwrite=False, sizecompare=False):
    """Downloads the file from the internet.
    Set the input options correctly to overwrite or do the size comparison

    Parameters
    ----------
    url : str
        Operator name, such as Convolution, Connected, etc
    path : str
        List of input symbols.
    overwrite : dict
        Dict of operator attributes
    sizecompare : dict
        Dict of operator attributes

    Returns
    -------
    out_name : converted out name of operation
    sym : nnvm.Symbol
        Converted nnvm Symbol
    """
    if os.path.isfile(path) and not overwrite:
        if (sizecompare):
            fileSize = os.path.getsize(path)
            resHead = requests.head(url)
            resGet = requests.get(url,stream=True)
            if 'Content-Length' not in resHead.headers :
                resGet = urllib2.urlopen(url)
            urlFileSize = int(resGet.headers['Content-Length'])
            if urlFileSize != fileSize:
                print ("exist file got corrupted, downloading", path , " file freshly")
                download(url, path, True, False)
                return
        print('File {} exists, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    try:
        urllib.request.urlretrieve(url, path, reporthook=dlProgress)
        print('')
    except:
        urllib.urlretrieve(url, path, reporthook=dlProgress)

######################################################################
# Prepare cfg and weights file
# Pretrained model available https://pjreddie.com/darknet/imagenet/
# --------------------------------------------------------------------
# Download cfg and weights file first time.

cfg_name = model_name + '.cfg'
weights_name = model_name + '.weights'
cfg_url = 'https://github.com/siju-samuel/darknet/blob/master/cfg/' + \
            cfg_name + '?raw=true'
weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'

download(cfg_url, cfg_name)
download(weights_url, weights_name)

######################################################################
# Download and Load darknet library
# ---------------------------------

darknet_lib = 'libdarknet.so'
darknetlib_url = 'https://github.com/siju-samuel/darknet/blob/master/lib/' + \
                        darknet_lib + '?raw=true'
download(darknetlib_url, darknet_lib)

#if the file doesnt exist, then exit normally.
if os.path.isfile('./' + darknet_lib) is False:
    exit(0)

darknet_lib = __darknetffi__.dlopen('./' + darknet_lib)
cfg = "./" + str(cfg_name)
weights = "./" + str(weights_name)
net = darknet_lib.load_network(cfg.encode('utf-8'), weights.encode('utf-8'), 0)
dtype = 'float32'
batch_size = 1
print("Converting darknet to nnvm symbols...")
sym, params = nnvm.frontend.darknet.from_darknet(net, dtype)

######################################################################
# Compile the model on NNVM
# --------------------------------------------------------------------
# compile the model
data = np.empty([batch_size, net.c ,net.h, net.w], dtype);
shape = {'data': data.shape}
print("Compiling the model...")
with nnvm.compiler.build_config(opt_level=2):
    graph, lib, params = nnvm.compiler.build(sym, target, shape, dtype, params)

#####################################################################
# Save the json
# --------------------------------------------------------------------
def save_lib():
    #Save the graph, params and .so to the current directory
    print("Saving the compiled output...")
    path_name = 'nnvm_darknet_' + model_name
    path_lib = path_name + '_deploy_lib.so'
    lib.export_library(path_lib)
    with open(path_name
+ "deploy_graph.json", "w") as fo:
        fo.write(graph.json())
    with open(path_name
+ "deploy_param.params", "wb") as fo:
        fo.write(nnvm.compiler.save_param_dict(params))
#save_lib()

######################################################################
# Load a test image
# --------------------------------------------------------------------
print("Loading the test image...")
img_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + \
            test_image   +'?raw=true'
download(img_url, test_image)

data = nnvm.testing.darknet.load_image(test_image, net.w, net.h)

######################################################################
# Execute on TVM
# --------------------------------------------------------------------
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
out_shape = (net.outputs,)
tvm_out = m.get_output(0, tvm.nd.empty(out_shape, dtype)).asnumpy()

#do the detection and bring up the bounding boxes
thresh = 0.24
hier_thresh = 0.5
img = nnvm.testing.darknet.load_image_color(test_image)
_, im_h, im_w = img.shape
probs= []
boxes = []
region_layer = net.layers[net.n - 1]
boxes, probs = nnvm.testing.yolo2_detection.get_region_boxes(region_layer, im_w, im_h, net.w, net.h,
                       thresh, probs, boxes, 1, tvm_out)

boxes, probs = nnvm.testing.yolo2_detection.do_nms_sort(boxes, probs,
                       region_layer.w*region_layer.h*region_layer.n, region_layer.classes, 0.3)

coco_name = 'coco.names'
coco_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + coco_name   +'?raw=true'
font_name = 'arial.ttf'
font_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + font_name   +'?raw=true'
download(coco_url, coco_name)
download(font_url, font_name)

with open(coco_name) as f:
    content = f.readlines()

names = [x.strip() for x in content]

nnvm.testing.yolo2_detection.draw_detections(img, region_layer.w*region_layer.h*region_layer.n,
                 thresh, boxes, probs, names, region_layer.classes)
plt.imshow(img.transpose(1,2,0))
plt.show()
