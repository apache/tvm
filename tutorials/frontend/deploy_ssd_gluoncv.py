"""
Deploy Single Shot Multibox Detector(SSD) model
===============================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_

This article is an introductory tutorial to deploy SSD models with TVM.
We will use GluonCV pre-trained SSD model and convert it to Relay IR
"""
import tvm

from matplotlib import pyplot as plt
from tvm.relay.testing.config import ctx_list
from tvm import relay
from tvm.contrib import graph_runtime
from gluoncv import model_zoo, data, utils


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
#   To get best inference performance on CPU, change
#   target argument according to your device and
#   follow the :ref:`tune_relay_x86` to tune x86 CPU and
#   :ref:`tune_relay_arm` for arm cpu.
#
#   SSD with VGG as body network is not supported yet since
#   x86 conv2d schedule doesn't support dilation.

supported_model = [
    'ssd_512_resnet18_v1_voc',
    'ssd_512_resnet18_v1_coco',
    'ssd_512_resnet50_v1_voc',
    'ssd_512_resnet50_v1_coco',
    'ssd_512_resnet101_v2_voc',
    'ssd_512_mobilenet1_0_voc',
    'ssd_512_mobilenet1_0_coco',
]

model_name = "ssd_512_resnet50_v1_voc"
dshape = (1, 3, 512, 512)
dtype = "float32"
target_list = ctx_list()

######################################################################
# Download and pre-process demo image

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/street_small.jpg?raw=true',
                          path='street_small.jpg')
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)

######################################################################
# Convert and compile model for CPU.

block = model_zoo.get_model(model_name, pretrained=True)

def compile(target):
    net, params = relay.frontend.from_mxnet(block, {"data": dshape})
    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(net, target, params=params)
    return graph, lib, params

######################################################################
# Create TVM runtime and do inference

def run(graph, lib, params, ctx):
    # Build TVM runtime
    m = graph_runtime.create(graph, lib, ctx)
    tvm_input = tvm.nd.array(x.asnumpy(), ctx=ctx)
    m.set_input('data', tvm_input)
    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    class_IDs, scores, bounding_boxs = m.get_output(0), m.get_output(1), m.get_output(2)
    return class_IDs, scores, bounding_boxs

for target, ctx in target_list:
    if target == "cuda":
        print("GPU not supported yet, skip.")
        continue
    graph, lib, params = compile(target)
    class_IDs, scores, bounding_boxs = run(graph, lib, params, ctx)

######################################################################
# Display result

ax = utils.viz.plot_bbox(img, bounding_boxs.asnumpy()[0], scores.asnumpy()[0],
                         class_IDs.asnumpy()[0], class_names=block.classes)
plt.show()
