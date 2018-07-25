"""
Compile Darknet model using NNVM api for Android Phone target
=================================
**Author**: `Dayananda V <https://github.com/dayanandasiet/>`

This article is an introductory tutorial to save compiled darknet models with NNVM
 CPU/GPU flavor for android phone target host.
Compiled model tvm functions(lib, param and graph) can use to load on android application
 using [android_deplo](https://github.com/dmlc/tvm/tree/master/apps/android_deploy).
"""
import os

from tvm.contrib import ndk

from tvm.contrib.download import download
from nnvm.testing.darknet import __darknetffi__
import nnvm

######################################################################
# Set the parameters here.
# Supported optimization level 0,1,2
# Supported flavor cpu, opencl, vulkan
# Supported models alexnet, resnet50, resnet152, extraction, yolo

arch = "arm64"
opt_level = 0
exec_flavor = "vulkan"
model_name = 'extraction'

######################################################################
# Prepare cfg and weights file
# ----------------------------
# Pretrained model available https://pjreddie.com/darknet/imagenet/
# Download cfg and weights file first time.

cfg_name = model_name + '.cfg'
weights_name = model_name + '.weights'
cfg_url = 'https://github.com/pjreddie/darknet/raw/master/cfg/' + \
            cfg_name

weights_url = 'http://pjreddie.com/media/files/' + weights_name + '?raw=true'

download(cfg_url, cfg_name)
download(weights_url, weights_name)

######################################################################
# Download and Load darknet library
# ---------------------------------
dtype = 'float32'
darknet_lib = 'libdarknet.so'
darknetlib_url = 'https://github.com/siju-samuel/darknet/raw/master/lib/' + \
                        darknet_lib
download(darknetlib_url, darknet_lib)

#if the file doesnt exist, then exit normally.
if os.path.isfile('./' + darknet_lib) is False:
    exit(0)

darknet_lib = __darknetffi__.dlopen('./' + darknet_lib)
cfg = "./" + str(cfg_name)
weights = "./" + str(weights_name)
net = darknet_lib.load_network(cfg.encode('utf-8'), weights.encode('utf-8'), 0)
batch_size = 1
print("Converting darknet to nnvm symbols...")
sym, params = nnvm.frontend.darknet.from_darknet(net, dtype)

######################################################################
# Target Creation [CPU/GPU(OPENCL/VULKAN)]
# -------------------------

if exec_flavor == "cpu":
    # Mobile CPU
    target = "llvm -target=%s-linux-android" % arch
    target_host = None
elif exec_flavor == "opencl":
    # Mobile GPU
    target = 'opencl'
    target_host = "llvm -target=%s-linux-android" % arch
elif exec_flavor == "vulkan":
    # Mobile GPU
    target = 'vulkan'
    target_host = "llvm -target=%s-linux-android" % arch

######################################################################
# Compile the model on NNVM
# -------------------------
# compile the model
shape = {'data': (batch_size, net.c, net.h, net.w)}
print("Compiling the model...")
with nnvm.compiler.build_config(opt_level=opt_level, add_pass=None):
    graph, lib, params = nnvm.compiler.build(sym, target, shape, dtype,
                                             params=params, target_host=target_host)


#####################################################################
# Save the nnvm graph and module
# -------------
# save compiled nnvm model pair of functions(lib, graph, param) to current directory
print("Saving the compiled nnvm model functions...")
lib.export_library("deploy_lib_" + exec_flavor + ".so", ndk.create_shared)
with open("deploy_graph_" + exec_flavor + ".json", "w") as filewriter:
    filewriter.write(graph.json())
with open("deploy_param.params", "wb") as filewriter:
    filewriter.write(nnvm.compiler.save_param_dict(params))

