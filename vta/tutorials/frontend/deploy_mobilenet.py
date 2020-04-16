from __future__ import absolute_import, print_function

import argparse, json, os, requests, sys, time
from io import BytesIO
from os.path import join, isfile
from PIL import Image

from mxnet.gluon.model_zoo import vision
import numpy as np
from matplotlib import pyplot as plt

import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_runtime, util, download
from tvm.contrib.debugger import debug_runtime
from tvm.relay import transform
import tvm.relay.testing

import vta
from vta.testing import simulator
from vta.top import graph_pack
from tvm.contrib.util import eprint

# Make sure that TVM was compiled with RPC=1
assert tvm.runtime.enabled("rpc")

######################################################################
# Define the platform and model targets
# -------------------------------------
# Execute on CPU vs. VTA, and define the model.

# Load VTA parameters from the vta/config/vta_config.json file
env = vta.get_env()

# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu
# multiple targets to run both on cpu and vta
targets = {
    "cpu": env.target_vta_cpu,
    "ext_dev": env.target
}

model = "mobilenetG"

######################################################################
# Obtain an execution remote
# --------------------------
# When target is 'pynq', reconfigure FPGA and runtime.
# Otherwise, if target is 'sim', execute locally.

if env.TARGET not in ["sim", "tsim", "intelfocl"]:

    # Get remote from tracker node if environment variable is set.
    # To set up the tracker, you'll need to follow the "Auto-tuning
    # a convolutional network for VTA" tutorial.
    tracker_host = os.environ.get("TVM_TRACKER_HOST", None)
    tracker_port = os.environ.get("TVM_TRACKER_PORT", None)
    # Otherwise if you have a device you want to program directly from
    # the host, make sure you've set the variables below to the IP of
    # your board.
    device_host = os.environ.get("VTA_PYNQ_RPC_HOST", "192.168.2.99")
    device_port = os.environ.get("VTA_PYNQ_RPC_PORT", "9091")
    if not tracker_host or not tracker_port:
        remote = rpc.connect(device_host, int(device_port))
    else:
        remote = autotvm.measure.request_remote(env.TARGET, tracker_host, int(tracker_port), timeout=10000)

    # Reconfigure the JIT runtime and FPGA.
    # You can program the FPGA with your own custom bitstream
    # by passing the path to the bitstream file instead of None.
    reconfig_start = time.time()
    vta.reconfig_runtime(remote)
    bitstream = os.environ.get("TVM_BIT", None)
    if bitstream:
        print("Program fpga with {}".format(bitstream))
        vta.program_fpga(remote, bitstream)

    reconfig_time = time.time() - reconfig_start
    print("Reconfigured FPGA and RPC runtime in {0:.2f}s!".format(reconfig_time))

# In simulation mode, host the RPC server locally.
else:
    remote = rpc.LocalSession()

# Get execution context from remote
# ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)
ctxes = [remote.ext_dev(0), remote.cpu(0)]

# Load pre-configured AutoTVM schedules
with autotvm.tophub.context(target):

    # Populate the shape and data type dictionary for ImageNet classifier input
    dtype_dict = {"data": 'float32'}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    # get the mobilenet model
    mod, params = relay.testing.mobilenet.get_workload(batch_size=1, dtype="float32",
                                                       depthwise_group_factor=env.BLOCK_IN)

    # Measure build start time
    build_start = time.time()

    # Update shape and type dictionary
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    if target.device_name == "vta":
        # Perform quantization in Relay
        # Note: We set opt_level to 3 in order to fold batch norm
        with relay.build_config(opt_level=3):
            with relay.quantize.qconfig(global_scale=8.0,
                                        skip_conv_layers=[0]):
                mod = relay.quantize.quantize(mod, params=params)
            # Perform graph packing and constant folding for VTA target
            assert env.BLOCK_IN == env.BLOCK_OUT
            relay_prog = graph_pack(
                mod["main"],
                env.BATCH,
                env.BLOCK_OUT,
                env.WGT_WIDTH,
                start_name="nn.conv2d",
                stop_name="nn.global_avg_pool2d")
    else:
        relay_prog = mod["main"]

    # Compile Relay program with AlterOpLayout disabled
    with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
        if target.device_name != "vta":
            graph, lib, params = relay.build(
                relay_prog, target=target,
                params=params, target_host=env.target_host)
        else:
            with vta.build_config(debug_flag=32):
                graph, lib, params = relay.build(
                    relay_prog, target=targets,
                    params=params, target_host=env.target_host)

    # Measure Relay build time
    build_time = time.time() - build_start
    print(model + " inference graph built in {0:.2f}s!".format(build_time))

    # Graph runtime
    m = graph_runtime.create(graph, lib, ctxes)

######################################################################
# Perform image classification inference
# --------------------------------------
# We run classification on an image sample from ImageNet
# We just need to download the categories files, `synset.txt`
# and an input test image.

# Download ImageNet categories
categ_url = "https://github.com/uwsaml/web-data/raw/master/vta/models/"
categ_fn = "synset.txt"
download.download(join(categ_url, categ_fn), categ_fn)
synset = eval(open(categ_fn).read())

# Download test image
image_url = 'https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg'
image_fn = 'cat.png'
download.download(image_url, image_fn)

# Prepare test image for inference
image = Image.open(image_fn).resize((224, 224))
plt.imshow(image)
plt.show()
image = np.array(image) - np.array([123., 117., 104.])
image /= np.array([58.395, 57.12, 57.375])
image = image.transpose((2, 0, 1))
image = image[np.newaxis, :]
image = np.repeat(image, env.BATCH, axis=0)

# Set the network parameters and inputs
m.set_input(**params)
m.set_input('data', image)

# Perform inference and gather execution statistics
# More on: https://docs.tvm.ai/api/python/module.html#tvm.runtime.Module.time_evaluator
num = 3 # number of times we run module for a single measurement
rep = 3 # number of measurements (we derive std dev from this)
timer = m.module.time_evaluator("run", ctxes[0], number=num, repeat=rep)

if env.TARGET in ["sim", "tsim"]:
    simulator.clear_stats()
    timer()

    sim_stats = simulator.stats()
    print("\nExecution statistics:")
    for k, v in sim_stats.items():
        # Since we execute the workload many times, we need to normalize stats
        # Note that there is always one warm up run
        # Therefore we divide the overall stats by (num * rep + 1)
        print("\t{:<16}: {:>16}".format(k, v // (num * rep + 1)))
else:
    tcost = timer()
    std = np.std(tcost.results) * 1000
    mean = tcost.mean * 1000
    print("\nPerformed inference in %.2fms (std = %.2f) for %d samples" % (mean, std, env.BATCH))
    print("Average per sample inference time: %.2fms" % (mean/env.BATCH))

# Get classification results
tvm_output = m.get_output(0, tvm.nd.empty((env.BATCH, 1000), "float32", remote.cpu(0)))
output = tvm_output.asnumpy()
for b in range(env.BATCH):
    top_categories = np.argsort(tvm_output.asnumpy()[b])
    # print("top_categories = ", top_categories)
    # Report top-5 classification results
    print("\n{} prediction for sample {}".format(model, b))
    print("\t#1:", synset[top_categories[-1]], output[b][top_categories[-1]])
    print("\t#2:", synset[top_categories[-2]], output[b][top_categories[-2]])
    print("\t#3:", synset[top_categories[-3]], output[b][top_categories[-3]])
    print("\t#4:", synset[top_categories[-4]], output[b][top_categories[-4]])
    print("\t#5:", synset[top_categories[-5]], output[b][top_categories[-5]])
    # This just checks that one of the 5 top categories
    # is one variety of cat; this is by no means an accurate
    # assessment of how quantization affects classification
    # accuracy but is meant to catch changes to the
    # quantization pass that would accuracy in the CI.
    cat_detected = False
    for k in top_categories[-5:]:
        if "cat" in synset[k]:
            cat_detected = True
    assert(cat_detected)
