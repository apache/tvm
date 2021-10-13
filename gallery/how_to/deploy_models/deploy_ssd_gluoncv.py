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
**Author**: `Yao Wang <https://github.com/kevinthesun>`_
`Leyuan Wang <https://github.com/Laurawly>`_

This article is an introductory tutorial to deploy SSD models with TVM.
We will use GluonCV pre-trained SSD model and convert it to Relay IR
"""
import tvm
from tvm import te

from matplotlib import pyplot as plt
from tvm import relay
from tvm.contrib import graph_executor
from tvm.contrib.download import download_testdata
from gluoncv import model_zoo, data, utils


######################################################################
# Preliminary and Set parameters
# ------------------------------
# .. note::
#
#   We support compiling SSD on both CPUs and GPUs now.
#
#   To get best inference performance on CPU, change
#   target argument according to your device and
#   follow the :ref:`tune_relay_x86` to tune x86 CPU and
#   :ref:`tune_relay_arm` for arm CPU.
#
#   To get best inference performance on Intel graphics,
#   change target argument to :code:`opencl -device=intel_graphics`.
#   But when using Intel graphics on Mac, target needs to
#   be set to `opencl` only for the reason that Intel subgroup
#   extension is not supported on Mac.
#
#   To get best inference performance on CUDA-based GPUs,
#   change the target argument to :code:`cuda`; and for
#   OPENCL-based GPUs, change target argument to
#   :code:`opencl` followed by device argument according
#   to your device.

supported_model = [
    "ssd_512_resnet50_v1_voc",
    "ssd_512_resnet50_v1_coco",
    "ssd_512_resnet101_v2_voc",
    "ssd_512_mobilenet1.0_voc",
    "ssd_512_mobilenet1.0_coco",
    "ssd_300_vgg16_atrous_voc" "ssd_512_vgg16_atrous_coco",
]

model_name = supported_model[0]
dshape = (1, 3, 512, 512)

######################################################################
# Download and pre-process demo image

im_fname = download_testdata(
    "https://github.com/dmlc/web-data/blob/main/" + "gluoncv/detection/street_small.jpg?raw=true",
    "street_small.jpg",
    module="data",
)
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)

######################################################################
# Convert and compile model for CPU.

block = model_zoo.get_model(model_name, pretrained=True)


def build(target):
    mod, params = relay.frontend.from_mxnet(block, {"data": dshape})
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target, params=params)
    return lib


######################################################################
# Create TVM runtime and do inference
# .. note::
#
#   Use target = "cuda -libs" to enable thrust based sort, if you
#   enabled thrust during cmake by -DUSE_THRUST=ON.


def run(lib, dev):
    # Build TVM runtime
    m = graph_executor.GraphModule(lib["default"](dev))
    tvm_input = tvm.nd.array(x.asnumpy(), device=dev)
    m.set_input("data", tvm_input)
    # execute
    m.run()
    # get outputs
    class_IDs, scores, bounding_boxs = m.get_output(0), m.get_output(1), m.get_output(2)
    return class_IDs, scores, bounding_boxs


for target in ["llvm", "cuda"]:
    dev = tvm.device(target, 0)
    if dev.exist:
        lib = build(target)
        class_IDs, scores, bounding_boxs = run(lib, dev)

######################################################################
# Display result

ax = utils.viz.plot_bbox(
    img,
    bounding_boxs.numpy()[0],
    scores.numpy()[0],
    class_IDs.numpy()[0],
    class_names=block.classes,
)
plt.show()
