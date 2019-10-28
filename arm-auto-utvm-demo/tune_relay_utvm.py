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
Auto-tuning a convolutional network on VTA
==========================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, `Thierry Moreau <https://homes.cs.washington.edu/~moreau/>`_

Auto-tuning for a specific accelerator design is critical for getting the best
performance for any given operator. This is a tutorial showcases how to tune a
whole convolutional network on VTA.

The operator implementation for VTA in TVM is written in template form.
The template has many tunable knobs (tile factor, virtual threads, etc).
We will tune all convolution operators in the neural network. After tuning,
we produce a log file which stores the best schedule parameters for all tuned
operators. When the TVM compiler compiles these operators, it will query this
log file to get the best knob parameters.

"""

######################################################################
# Install dependencies
# --------------------
# To use the autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado mxnet requests pillow
#
# To make TVM run faster during tuning, it is recommended to use cython
# as FFI of TVM. In the root directory of TVM, execute
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

import os
from mxnet.gluon.model_zoo import vision
import numpy as np
from PIL import Image

import topi
import tvm
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_runtime, util, download
from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import vta
from vta.testing import simulator
from vta.top import graph_pack

#################################################################
# Compile network
# ---------------
# Perform vta-specific compilation with Relay from a Gluon model

#def compile_network(env, target, model, start_pack, stop_pack):
#
#    # Populate the shape and data type dictionary
#    dtype_dict = {"data": 'float32'}
#    shape_dict = {"data": (env.BATCH, 3, 224, 224)}
#
#    # Get off the shelf gluon model, and convert to relay
#    gluon_model = vision.get_model(model, pretrained=True)
#    mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)
#
#    # Update shape and type dictionary
#    shape_dict.update({k: v.shape for k, v in params.items()})
#    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})
#
#    # Perform quantization in Relay
#    with relay.quantize.qconfig(global_scale=8.0,
#                                skip_conv_layers=[0]):
#        relay_prog = relay.quantize.quantize(mod["main"], params=params)
#
#    # Perform graph packing and constant folding for VTA target
#    if target.device_name == "vta":
#        assert env.BLOCK_IN == env.BLOCK_OUT
#        relay_prog = graph_pack(
#            relay_prog,
#            env.BATCH,
#            env.BLOCK_OUT,
#            env.WGT_WIDTH,
#            start_name=start_pack,
#            stop_name=stop_pack)
#
#    return relay_prog, params


#################################################################
# Start RPC Tracker
# -----------------
# TVM uses an RPC session to communicate with Pynq boards.
# During tuning, the tuner will send the generated code to the board and
# measure the speed of code on the board.
#
# To scale up tuning, TVM uses an RPC Tracker to manage multiple devices.
# The RPC Tracker is a centralized master node. We can register all devices to
# the tracker. For example, if we have 10 Pynq boards, we can register all of them
# to the tracker, and run 10 measurements in parallel, accelerating the tuning process.
#
# To start an RPC tracker, run this command on the host machine. The tracker is
# required during the whole tuning process, so we need to open a new terminal for
# this command:
#
# .. code-block:: bash
#
#   python -m tvm.exec.rpc_tracker --host=0.0.0.0 --port=9190
#
# The expected output is:
#
# .. code-block:: bash
#
#   INFO:RPCTracker:bind to 0.0.0.0:9190

#################################################################
# Register devices to RPC Tracker
# -----------------------------------
# Now we can register our devices to the tracker. The first step is to
# build the TVM runtime for the Pynq devices.
#
# Follow `this section <https://docs.tvm.ai/vta/install.html#pynq-side-rpc-server-build-deployment>`_
# to build the TVM runtime on the device. Then register the device to the tracker with:
#
# .. code-block:: bash
#
#   python -m tvm.exec.rpc_server --tracker=[HOST_IP]:9190 --key=pynq
#
# (replace :code:`[HOST_IP]` with the IP address of your host machine)
#
# After registering devices, we can confirm it by querying the rpc_tracker:
#
# .. code-block:: bash
#
#   python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
#
# For example, if we have 6 Pynq boards and 11 Raspberry Pi 3B,
# the output can be
#
# .. code-block:: bash
#
#    Queue Status
#    ----------------------------------
#    key          total  free  pending
#    ----------------------------------
#    pynq         6      6     0
#    rpi3b        11     11    0
#    ----------------------------------
#
# You can register multiple devices to the tracker to accelerate tuning.

###########################################
# Set Tuning Options
# ------------------
# Before tuning, we should apply some configurations.
# Here we use an Pynq-Z1 board as an example.

# Tracker host and port can be set by your environment
#tracker_host = os.environ.get("TVM_TRACKER_HOST", '0.0.0.0')
#tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))

# Load VTA parameters from the vta/config/vta_config.json file
#env = vta.get_env()

# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
#device = "vta"
#target = env.target if device == "vta" else env.target_vta_cpu

# Name of Gluon model to compile
# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.
#network = "resnet18_v1"
#start_pack="nn.max_pool2d"
#stop_pack="nn.global_avg_pool2d"

@autotvm.template
def matmul(N, L, M, dtype):
    A = tvm.placeholder((N, L), name='A', dtype=dtype)
    B = tvm.placeholder((L, M), name='B', dtype=dtype)

    k = tvm.reduce_axis((0, L), name='k')
    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
    s = tvm.create_schedule(C.op)

    # schedule
    y, x = s[C].op.axis
    k = s[C].op.reduce_axis[0]

    ##### define space begin #####
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", y, num_outputs=2)
    cfg.define_split("tile_x", x, num_outputs=2)
    ##### define space end #####

    # schedule according to config
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)

    s[C].reorder(yo, xo, k, yi, xi)

    return s, [A, B, C]


def tune():
    N, L, M = 32, 32, 32
    #task = autotvm.task.create(matmul, args=(N, L, M, 'float32'), target='c')
    micro_target = tvm.target.create('c -device=micro_dev')
    task = autotvm.task.create(matmul, args=(N, L, M, 'float32'), target=micro_target)
    print(task.config_space)

    DEVICE_TYPE = 'openocd'
    TOOLCHAIN_PREFIX = 'arm-none-eabi-'
    DEVICE = 'arm-cortex-m'
    SERVER_ADDR = '0.0.0.0'
    SERVER_PORT = 9190
    import tvm.micro as micro

    log_filename = '%s.%s.log' % (DEVICE, 'test')
    n_trial = 10
    early_stopping = None
    measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(
                build_func=tvm.micro.cross_compiler(TOOLCHAIN_PREFIX, micro.LibType.OPERATOR)),
            runner=autotvm.RPCRunner('micro', SERVER_ADDR, SERVER_PORT, n_parallel=1, number=5)
            )

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    tuner = RandomTuner(task)

    # do tuning
    tuner.tune(n_trial=min(n_trial, len(task.config_space)),
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(n_trial, prefix='[Matmul Task]'),
                autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


def evaluate():
    # compile kernels with history best records
    with autotvm.tophub.context(target, extra_files=[log_file]):
        # Compile network
        print("Compile...")
        with relay.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
            if target.device_name != "vta":
                graph, lib, params = relay.build(
                    relay_prog, target=target,
                    params=params, target_host=env.target_host)
            else:
                with vta.build_config():
                    graph, lib, params = relay.build(
                        relay_prog, target=target,
                        params=params, target_host=env.target_host)

        # Export library
        print("Upload...")
        temp = util.tempdir()
        lib.save(temp.relpath("graphlib.o"))
        remote.upload(temp.relpath("graphlib.o"))
        lib = remote.load_module("graphlib.o")

        # Generate the graph runtime
        ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)
        m = graph_runtime.create(graph, lib, ctx)

        # upload parameters to device
        image = tvm.nd.array(
            (np.random.uniform(size=(1, 3, 224, 224))).astype('float32'))
        m.set_input(**params)
        m.set_input('data', image)

        # evaluate
        print("Evaluate inference time cost...")
        timer = m.module.time_evaluator("run", ctx, number=1, repeat=10)
        tcost = timer()
        prof_res = np.array(tcost.results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

#sess['exit']()

## We should have extracted 10 convolution tasks
#assert len(tasks) == 10
#print("Extracted {} conv2d tasks:".format(len(tasks)))
#for tsk in tasks:
#    inp = tsk.args[0][1]
#    wgt = tsk.args[1][1]
#    batch = inp[0]*inp[4]
#    in_filter = inp[1]*inp[5]
#    out_filter = wgt[0]*wgt[4]
#    height, width = inp[2], inp[3]
#    hkernel, wkernel = wgt[2], wgt[3]
#    hstride, wstride = tsk.args[2][0], tsk.args[2][1]
#    hpad, wpad = tsk.args[3][0], tsk.args[3][1]
#    print("({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
#            batch, height, width, in_filter, out_filter,
#            hkernel, wkernel, hpad, wpad, hstride, wstride
#    ))

# We do not run the tuning in our webpage server since it takes too long.
# Comment the following line to run it by yourself.
#return

