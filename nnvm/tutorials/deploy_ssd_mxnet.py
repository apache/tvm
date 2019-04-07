# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Deploy Single Shot Multibox Detector(SSD) model
===============================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_, \
`Leyuan Wang <https://github.com/Laurawly>`_

This article is an introductory tutorial to deploy SSD models with TVM.
We will use mxnet pretrained SSD model with Resnet50 as body network and
convert it to NNVM graph;
"""
import os
import zipfile
import tvm
import mxnet as mx
import cv2
import numpy as np

from nnvm import compiler
from nnvm.frontend import from_mxnet
from tvm import relay
from tvm.contrib.download import download_testdata
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

model_name = "ssd_resnet50_512"
model_file = "%s.zip" % model_name
test_image = "dog.jpg"
dshape = (1, 3, 512, 512)
dtype = "float32"

# Target settings
# Use these commented settings to build for cuda.
#target = 'cuda'
#ctx = tvm.gpu(0)
# Use these commented settings to build for opencl.
#target = 'opencl'
#ctx = tvm.opencl(0)
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
inference_symbol_folder = \
    "c1904e900848df4548ce5dfb18c719c7-a28c4856c827fe766aa3da0e35bad41d44f0fb26"
inference_symbol_url = "https://gist.github.com/kevinthesun/c1904e900848df4548ce5dfb18c719c7/" \
                       "archive/a28c4856c827fe766aa3da0e35bad41d44f0fb26.zip"

model_file_path = download_testdata(model_url, model_file, module=["mxnet", "ssd_model"])
inference_symbol_path = download_testdata(inference_symbol_url, "inference_model.zip",
                                          module=["mxnet", "ssd_model"])
test_image_path = download_testdata(image_url, test_image, module="data")
model_dir = os.path.dirname(model_file_path)

zip_ref = zipfile.ZipFile(model_file_path, 'r')
zip_ref.extractall(model_dir)
zip_ref.close()
zip_ref = zipfile.ZipFile(inference_symbol_path)
zip_ref.extractall(model_dir)
zip_ref.close()

######################################################################
# Convert and compile model with NNVM or Relay for CPU.

sym = mx.sym.load("%s/%s/ssd_resnet50_inference.json" % (model_dir, inference_symbol_folder))
_, arg_params, aux_params = load_checkpoint("%s/%s" % (model_dir, model_name), 0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    "-f", "--frontend",
    help="Frontend for compilation, nnvm or relay",
    type=str,
    default="nnvm")
args = parser.parse_args()
if args.frontend == "relay":
    net, params = relay.frontend.from_mxnet(sym, {"data": dshape}, arg_params=arg_params, \
                                            aux_params=aux_params)
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(net, target, params=params)
elif args.frontend == "nnvm":
    net, params = from_mxnet(sym, arg_params, aux_params)
    with compiler.build_config(opt_level=3):
        graph, lib, params = compiler.build(
            net, target, {"data": dshape}, params=params)
else:
    parser.print_help()
    parser.exit()

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
    mpl.rcParams['figure.figsize'] = (10, 10)
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
