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
#   pip3 install --user psutil xgboost tornado mxnet requests "Pillow<7" cloudpickle
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

from tvm import topi
import tvm
from tvm import te
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_executor, utils, download
from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner

import vta
from vta.testing import simulator
from vta.top import graph_pack

#################################################################
# Compile network
# ---------------
# Perform vta-specific compilation with Relay from a Gluon model


def compile_network(env, target, model, start_pack, stop_pack):

    # Populate the shape and data type dictionary
    dtype_dict = {"data": "float32"}
    shape_dict = {"data": (env.BATCH, 3, 224, 224)}

    # Get off the shelf gluon model, and convert to relay
    gluon_model = vision.get_model(model, pretrained=True)
    mod, params = relay.frontend.from_mxnet(gluon_model, shape_dict)

    # Update shape and type dictionary
    shape_dict.update({k: v.shape for k, v in params.items()})
    dtype_dict.update({k: str(v.dtype) for k, v in params.items()})

    # Perform quantization in Relay
    # Note: We set opt_level to 3 in order to fold batch norm
    with tvm.transform.PassContext(opt_level=3):
        with relay.quantize.qconfig(global_scale=8.0, skip_conv_layers=[0]):
            mod = relay.quantize.quantize(mod, params=params)

    # Perform graph packing and constant folding for VTA target
    if target.device_name == "vta":
        assert env.BLOCK_IN == env.BLOCK_OUT
        relay_prog = graph_pack(
            mod["main"],
            env.BATCH,
            env.BLOCK_OUT,
            env.WGT_WIDTH,
            start_name=start_pack,
            stop_name=stop_pack,
        )

    return relay_prog, params


#################################################################
# Start RPC Tracker
# -----------------
# TVM uses an RPC session to communicate with Pynq boards.
# During tuning, the tuner will send the generated code to the board and
# measure the speed of code on the board.
#
# To scale up tuning, TVM uses an RPC Tracker to manage multiple devices.
# The RPC Tracker is a centralized controller node. We can register all devices to
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
# Follow :ref:`vta-index`
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
tracker_host = os.environ.get("TVM_TRACKER_HOST", "127.0.0.1")
tracker_port = int(os.environ.get("TVM_TRACKER_PORT", 9190))

# Load VTA parameters from the 3rdparty/vta-hw/config/vta_config.json file
env = vta.get_env()

# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
# Set ``device=arm_cpu`` to run inference on the CPU
# or ``device=vta`` to run inference on the FPGA.
device = "vta"
target = env.target if device == "vta" else env.target_vta_cpu

# Name of Gluon model to compile
# The ``start_pack`` and ``stop_pack`` labels indicate where
# to start and end the graph packing relay pass: in other words
# where to start and finish offloading to VTA.
network = "resnet18_v1"
start_pack = "nn.max_pool2d"
stop_pack = "nn.global_avg_pool2d"

# Tuning option
log_file = "%s.%s.log" % (device, network)
tuning_option = {
    "log_filename": log_file,
    "tuner": "random",
    "n_trial": 1000,
    "early_stopping": None,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.RPCRunner(
            env.TARGET,
            host=tracker_host,
            port=tracker_port,
            number=5,
            timeout=60,
            module_loader=vta.module_loader(),
            # check_correctness=True, # TODO: re-enable when check_correctness works again.
        ),
    ),
}

####################################################################
#
# .. note:: How to set tuning options
#
#   In general, the default values provided here work well.
#   If you have enough time budget, you can set :code:`n_trial`, :code:`early_stopping`
#   to larger values, makes the tuning run for longer.
#   If your device is under-powered or your conv2d operators are large, consider
#   setting a longer timeout.
#

###################################################################
# Begin Tuning
# ------------
# Now we can extract tuning tasks from the network and begin tuning.
# Here, we provide a simple utility function to tune a list of tasks.
# This function is just an initial implementation which tunes them in sequential order.
# We will introduce a more sophisticated tuning scheduler in the future.
#
# Given that the tuning will be done on Pynq FPGA boards, make sure that
# the ```TARGET`` entry in the ``vta_config.json`` file is set to ``pynq``.


# You can skip the implementation of this function for this tutorial.
def tune_tasks(
    tasks,
    measure_option,
    tuner="xgb",
    n_trial=1000,
    early_stopping=None,
    log_filename="tuning.log",
    use_transfer_learning=True,
):

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))

        # create tuner
        if tuner == "xgb":
            tuner_obj = XGBTuner(tsk, loss_type="reg")
        elif tuner == "xgb_knob":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="knob")
        elif tuner == "xgb_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="itervar")
        elif tuner == "xgb_curve":
            tuner_obj = XGBTuner(tsk, loss_type="reg", feature_type="curve")
        elif tuner == "xgb_rank":
            tuner_obj = XGBTuner(tsk, loss_type="rank")
        elif tuner == "xgb_rank_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="knob")
        elif tuner == "xgb_rank_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="itervar")
        elif tuner == "xgb_rank_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank", feature_type="curve")
        elif tuner == "xgb_rank_binary":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary")
        elif tuner == "xgb_rank_binary_knob":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="knob")
        elif tuner == "xgb_rank_binary_itervar":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="itervar")
        elif tuner == "xgb_rank_binary_curve":
            tuner_obj = XGBTuner(tsk, loss_type="rank-binary", feature_type="curve")
        elif tuner == "ga":
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == "random":
            tuner_obj = RandomTuner(tsk)
        elif tuner == "gridsearch":
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tsk_trial = min(n_trial, len(tsk.config_space))
        tuner_obj.tune(
            n_trial=tsk_trial,
            early_stopping=early_stopping,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(tsk_trial, prefix=prefix),
                autotvm.callback.log_to_file(tmp_log_file),
            ],
        )

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


########################################################################
# Register VTA-specific tuning tasks


def register_vta_tuning_tasks():
    from tvm.autotvm.task import TaskExtractEnv

    @tvm.te.tag_scope(tag=topi.tag.ELEMWISE)
    def my_clip(x, a_min, a_max):
        """Unlike topi's current clip, put min and max into two stages."""
        const_min = tvm.tir.const(a_min, x.dtype)
        const_max = tvm.tir.const(a_max, x.dtype)
        x = te.compute(x.shape, lambda *i: tvm.te.min(x(*i), const_max), name="clipA")
        x = te.compute(x.shape, lambda *i: tvm.te.max(x(*i), const_min), name="clipB")
        return x

    # init autotvm env to register VTA operator
    TaskExtractEnv()

    @autotvm.template("conv2d_packed.vta")
    def _topi_nn_conv2d(*args, **kwargs):
        assert not kwargs, "Do not support kwargs in template function call"
        A, W = args[:2]

        with tvm.target.vta():
            res = vta.top.conv2d_packed(*args, **kwargs)
            res = topi.right_shift(res, 8)
            res = my_clip(res, 0, 127)
            res = topi.cast(res, "int8")

        if tvm.target.Target.current().device_name == "vta":
            s = vta.top.schedule_conv2d_packed([res])
        else:
            s = te.create_schedule([res.op])
        return s, [A, W, res]


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.


def tune_and_evaluate(tuning_opt):

    # Register VTA tuning tasks
    register_vta_tuning_tasks()

    # Perform task extraction on Relay program
    print("Extract tasks...")
    relay_prog, params = compile_network(env, target, network, start_pack, stop_pack)
    mod = tvm.IRModule.from_expr(relay_prog)
    tasks = autotvm.task.extract_from_program(
        mod,
        params=params,
        ops=(relay.op.get("nn.conv2d"),),
        target=target,
        target_host=env.target_host,
    )

    # filter out non-packed conv2d task
    tasks = list(filter(lambda t: len(t.args[0][1]) > 4 and "conv" in t.name, tasks))

    # We should have extracted 10 convolution tasks
    assert len(tasks) == 10
    print("Extracted {} conv2d tasks:".format(len(tasks)))
    for tsk in tasks:
        inp = tsk.args[0][1]
        wgt = tsk.args[1][1]
        batch = inp[0] * inp[4]
        in_filter = inp[1] * inp[5]
        out_filter = wgt[0] * wgt[4]
        height, width = inp[2], inp[3]
        hkernel, wkernel = wgt[2], wgt[3]
        hstride, wstride = tsk.args[2][0], tsk.args[2][1]
        hpad, wpad = tsk.args[3][0], tsk.args[3][1]
        print(
            "({}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {})".format(
                batch,
                height,
                width,
                in_filter,
                out_filter,
                hkernel,
                wkernel,
                hpad,
                wpad,
                hstride,
                wstride,
            )
        )

    # We do not run the tuning in our webpage server since it takes too long.
    # Comment the following line to run it by yourself.
    return

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # evaluate with tuning history
    if env.TARGET != "sim":
        # Get remote from fleet node
        remote = autotvm.measure.request_remote(
            env.TARGET, tracker_host, tracker_port, timeout=10000
        )
        # Reconfigure the JIT runtime and FPGA.
        vta.reconfig_runtime(remote)
        vta.program_fpga(remote, bitstream=None)
    else:
        # In simulation mode, host the RPC server locally.
        remote = rpc.LocalSession()

    # compile kernels with history best records
    with autotvm.tophub.context(target, extra_files=[log_file]):
        # Compile network
        print("Compile...")
        if target.device_name != "vta":
            with tvm.transform.PassContext(opt_level=3, disabled_pass={"AlterOpLayout"}):
                lib = relay.build(
                    relay_prog, target=target, params=params, target_host=env.target_host
                )
        else:
            with vta.build_config(opt_level=3, disabled_pass={"AlterOpLayout"}):
                lib = relay.build(
                    relay_prog, target=target, params=params, target_host=env.target_host
                )

        # Export library
        print("Upload...")
        temp = utils.tempdir()
        lib.export_library(temp.relpath("graphlib.tar"))
        remote.upload(temp.relpath("graphlib.tar"))
        lib = remote.load_module("graphlib.tar")

        # Generate the graph executor
        ctx = remote.ext_dev(0) if device == "vta" else remote.cpu(0)
        m = graph_executor.GraphModule(lib["default"](ctx))

        # upload parameters to device
        image = tvm.nd.array((np.random.uniform(size=(1, 3, 224, 224))).astype("float32"))
        m.set_input("data", image)

        # evaluate
        print("Evaluate inference time cost...")
        timer = m.module.time_evaluator("run", ctx, number=1, repeat=10)
        tcost = timer()
        prof_res = np.array(tcost.results) * 1000  # convert to millisecond
        print(
            "Mean inference time (std dev): %.2f ms (%.2f ms)"
            % (np.mean(prof_res), np.std(prof_res))
        )


# Run the tuning and evaluate the results
tune_and_evaluate(tuning_option)

######################################################################
# Sample Output
# -------------
# The tuning needs to compile many programs and extract feature from them.
# So a high performance CPU is recommended.
# One sample output is listed below.
# It takes about 2 hours on a 16T CPU, and 6 Pynq boards.
#
# .. code-block:: bash
#
#    Extract tasks...
#    [Warning] Invalid shape during AutoTVM task creation
#    Extracted 10 conv2d tasks:
#        Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 16, 14, 14, 1, 16), 'int8'), ('TENSOR', (32, 16, 1, 1, 16, 16), 'int8'), (2, 2), (0, 0), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 16, 14, 14, 1, 16, 'int8'), (32, 16, 1, 1, 16, 16, 'int8'), (2, 2), (0, 0), (1, 1), 'NCHW1n16c', 'int32'))
#        Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 8, 28, 28, 1, 16), 'int8'), ('TENSOR', (16, 8, 1, 1, 16, 16), 'int8'), (2, 2), (0, 0), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 8, 28, 28, 1, 16, 'int8'), (16, 8, 1, 1, 16, 16, 'int8'), (2, 2), (0, 0), (1, 1), 'NCHW1n16c', 'int32'))
#        Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 4, 56, 56, 1, 16), 'int8'), ('TENSOR', (8, 4, 1, 1, 16, 16), 'int8'), (2, 2), (0, 0), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 4, 56, 56, 1, 16, 'int8'), (8, 4, 1, 1, 16, 16, 'int8'), (2, 2), (0, 0), (1, 1), 'NCHW1n16c', 'int32'))
#        Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 4, 56, 56, 1, 16), 'int8'), ('TENSOR', (4, 4, 3, 3, 16, 16), 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 4, 56, 56, 1, 16, 'int8'), (4, 4, 3, 3, 16, 16, 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'))
#        Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 8, 28, 28, 1, 16), 'int8'), ('TENSOR', (8, 8, 3, 3, 16, 16), 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 8, 28, 28, 1, 16, 'int8'), (8, 8, 3, 3, 16, 16, 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'))
#        Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 4, 56, 56, 1, 16), 'int8'), ('TENSOR', (8, 4, 3, 3, 16, 16), 'int8'), (2, 2), (1, 1), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 4, 56, 56, 1, 16, 'int8'), (8, 4, 3, 3, 16, 16, 'int8'), (2, 2), (1, 1), (1, 1), 'NCHW1n16c', 'int32'))
#        Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 16, 14, 14, 1, 16), 'int8'), ('TENSOR', (16, 16, 3, 3, 16, 16), 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 16, 14, 14, 1, 16, 'int8'), (16, 16, 3, 3, 16, 16, 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'))
#        Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 8, 28, 28, 1, 16), 'int8'), ('TENSOR', (16, 8, 3, 3, 16, 16), 'int8'), (2, 2), (1, 1), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 8, 28, 28, 1, 16, 'int8'), (16, 8, 3, 3, 16, 16, 'int8'), (2, 2), (1, 1), (1, 1), 'NCHW1n16c', 'int32'))
#        Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 32, 7, 7, 1, 16), 'int8'), ('TENSOR', (32, 32, 3, 3, 16, 16), 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 32, 7, 7, 1, 16, 'int8'), (32, 32, 3, 3, 16, 16, 'int8'), (1, 1), (1, 1), (1, 1), 'NCHW1n16c', 'int32'))
#        Task(func_name=topi_nn_conv2d, args=(('TENSOR', (1, 16, 14, 14, 1, 16), 'int8'), ('TENSOR', (32, 16, 3, 3, 16, 16), 'int8'), (2, 2), (1, 1), (1, 1), 'NCHW1n16c', 'int32'), kwargs={}, workload=('conv2d', (1, 16, 14, 14, 1, 16, 'int8'), (32, 16, 3, 3, 16, 16, 'int8'), (2, 2), (1, 1), (1, 1), 'NCHW1n16c', 'int32'))
#    Tuning...
#    [Task  1/10]  Current/Best:    0.72/  23.24 GFLOPS | Progress: (480/1000) | 640.31 s Done.
#    [Task  2/10]  Current/Best:    0.00/  27.69 GFLOPS | Progress: (576/1000) | 810.09 s Done.
#    [Task  3/10]  Current/Best:    0.00/  22.97 GFLOPS | Progress: (1000/1000) | 1125.37 s Done.
#    [Task  4/10]  Current/Best:    0.00/  31.26 GFLOPS | Progress: (1000/1000) | 1025.52 s Done.
#    [Task  5/10]  Current/Best:    0.00/  15.15 GFLOPS | Progress: (1000/1000) | 1236.58 s Done.
#    [Task  6/10]  Current/Best:    0.00/  22.74 GFLOPS | Progress: (1000/1000) | 906.60 s Done.
#    [Task  7/10]  Current/Best:    0.00/  15.27 GFLOPS | Progress: (1000/1000) | 1056.25 s Done.
#    [Task  8/10]  Current/Best:    0.00/   2.18 GFLOPS | Progress: (1000/1000) | 2275.29 s Done.
#    [Task  9/10]  Current/Best:    2.23/   3.99 GFLOPS | Progress: (1000/1000) | 2527.25 s Done.
#    [Task 10/10]  Current/Best:    1.56/   6.32 GFLOPS | Progress: (480/1000) | 1304.84 s Done.
#    Compile...
#    Upload...
#    Evaluate inference time cost...
#    Mean inference time (std dev): 621.79 ms (0.14 ms)

######################################################################
#
# .. note:: **Experiencing Difficulties?**
#
#   The auto tuning module is error-prone. If you always see " 0.00/ 0.00 GFLOPS",
#   then there must be something wrong.
#
#   First, make sure you set the correct configuration of your device.
#   Then, you can print debug information by adding these lines in the beginning
#   of the script. It will print every measurement result, where you can find useful
#   error messages.
#
#   .. code-block:: python
#
#      import logging
#      logging.getLogger('autotvm').setLevel(logging.DEBUG)
#
#   Finally, always feel free to ask our community for help on https://discuss.tvm.apache.org
