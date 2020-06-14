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

model = "DCGAN"

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
    shape_dict = {"data": (env.BATCH, 100)}

    # get the mobilenet model
    mod, params = relay.testing.dcgan.get_workload(batch_size=1, dtype="float32", oshape=(3, 64, 64))

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
                                        skip_conv_layers=[3]):
                mod = relay.quantize.quantize(mod, params=params)
            # Perform graph packing and constant folding for VTA target
            assert env.BLOCK_IN == env.BLOCK_OUT
            relay_prog = graph_pack(
                mod["main"],
                env.BATCH,
                env.BLOCK_OUT,
                env.WGT_WIDTH,
                start_name="cast",
                stop_name="cast", stop_name_idx=52, device_annot=True)
    else:
        relay_prog = mod["main"]

    # Compile Relay program with AlterOpLayout disabled
    with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
        if target.device_name != "vta":
            graph, lib, params = relay.build(
                relay_prog, target=target,
                params=params, target_host=env.target_host)
        else:
            with vta.build_config(debug_flag=38):
                graph, lib, params = relay.build(
                    relay_prog, target=targets,
                    params=params, target_host=env.target_host)

    # Measure Relay build time
    build_time = time.time() - build_start
    print(model + " inference graph built in {0:.2f}s!".format(build_time))

    # Graph runtime
    m = graph_runtime.create(graph, lib, ctxes)

image = np.zeros((1, 100), dtype=np.float32)
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
    # timer()
    m['run']()

    sim_stats = simulator.stats()
    print("\nExecution statistics:")
    for k, v in sim_stats.items():
        # Since we execute the workload many times, we need to normalize stats
        # Note that there is always one warm up run
        # Therefore we divide the overall stats by (num * rep + 1)
        print("\t{:<16}: {:>16}".format(k, v // (num * rep + 1)))
else:
    m['run']()
    print("Run done")
    # tcost = timer()
    # std = np.std(tcost.results) * 1000
    # mean = tcost.mean * 1000
    # print("\nPerformed inference in %.2fms (std = %.2f) for %d samples" % (mean, std, env.BATCH))
    # print("Average per sample inference time: %.2fms" % (mean/env.BATCH))

# Get classification results
tvm_output = m.get_output(0, tvm.nd.empty((env.BATCH, 3, 64, 64), "float32", remote.cpu(0)))
output = tvm_output.asnumpy()
for b in range(env.BATCH):
    print(tvm_output.asnumpy()[b])
