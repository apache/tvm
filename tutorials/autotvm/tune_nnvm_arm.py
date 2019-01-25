"""
Auto-tuning a convolutional network for ARM CPU
====================================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, `Zhao Wu <https://github.com/FrozenGene>`_

Auto-tuning for a specific ARM device is critical for getting the best
performance. This is a tutorial about how to tune a whole convolutional
network.

The operator implementation for ARM CPU in TVM is written in template form.
The template has many tunable knobs (tile factor, vectorization, unrolling, etc).
We will tune all convolution and depthwise convolution operators
in the neural network. After tuning, we produce a log file which stores
the best knob values for all required operators. When the tvm compiler compiles
these operators, it will query this log file to get the best knob values.

We also released pre-tuned parameters for some arm devices. You can go to
`ARM CPU Benchmark <https://github.com/dmlc/tvm/wiki/Benchmark#arm-cpu>`_
to see the results.
"""

######################################################################
# Install dependencies
# --------------------
# To use the autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado
#
# To make tvm run faster during tuning, it is recommended to use cython
# as FFI of tvm. In the root directory of tvm, execute
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

import os

import numpy as np

import nnvm.testing
import nnvm.compiler
import tvm
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime

#################################################################
# Define network
# --------------
# First we need to define the network in nnvm symbol API.
# We can load some pre-defined network from :code:`nnvm.testing`.
# We can also load models from MXNet, ONNX and TensorFlow (see NNVM
# tutorials :ref:`tutorial-nnvm` for more details).

def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        net, params = nnvm.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        net, params = nnvm.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size)
    elif name == 'mobilenet':
        net, params = nnvm.testing.mobilenet.get_workload(batch_size=batch_size)
    elif name == 'squeezenet_v1.1':
        net, params = nnvm.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1')
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        net, params = nnvm.testing.inception_v3.get_workload(batch_size=batch_size)
    elif name == 'custom':
        # an example for custom network
        from nnvm.testing import utils
        net = nnvm.sym.Variable('data')
        net = nnvm.sym.conv2d(net, channels=4, kernel_size=(3,3), padding=(1,1))
        net = nnvm.sym.flatten(net)
        net = nnvm.sym.dense(net, units=1000)
        net, params = utils.create_workload(net, batch_size, (3, 224, 224))
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        net, params = nnvm.frontend.from_mxnet(block)
        net = nnvm.sym.softmax(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return net, params, input_shape, output_shape


#################################################################
# Start RPC Tracker
# -----------------
# TVM uses RPC session to communicate with ARM boards.
# During tuning, the tuner will send the generated code to the board and
# measure the speed of code on the board.
#
# To scale up the tuning, TVM uses RPC Tracker to manage distributed devices.
# The RPC Tracker is a centralized master node. We can register all devices to
# the tracker. For example, if we have 10 phones, we can register all of them
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
# The expected output is
#
# .. code-block:: bash
#
#   INFO:RPCTracker:bind to 0.0.0.0:9190

#################################################################
# Register devices to RPC Tracker
# -----------------------------------
# Now we can register our devices to the tracker. The first step is to
# build tvm runtime for the ARM devices.
#
# * For Linux:
#   Follow this section :ref:`build-tvm-runtime-on-device` to build
#   tvm runtime on the device. Then register the device to tracker by
#
#   .. code-block:: bash
#
#     python -m tvm.exec.rpc_server --tracker=[HOST_IP]:9190 --key=rk3399
#
#   (replace :code:`[HOST_IP]` with the IP address of your host machine)
#
# * For Android:
#   Follow this `readme page <https://github.com/dmlc/tvm/tree/master/apps/android_rpc>`_ to
#   install tvm rpc apk on the android device. Make sure you can pass the android rpc test.
#   Then you have already registred your device. During tuning, you have to go to developer option
#   and enable "Keep screen awake during changing" and charge your phone to make it stable.
#
# After registering devices, we can confirm it by querying rpc_tracker
#
# .. code-block:: bash
#
#   python -m tvm.exec.query_rpc_tracker --host=0.0.0.0 --port=9190
#
# For example, if we have 2 Huawei mate10 pro, 11 Raspberry Pi 3B and 2 rk3399,
# the output can be
#
# .. code-block:: bash
#
#    Queue Status
#    ----------------------------------
#    key          total  free  pending
#    ----------------------------------
#    mate10pro    2      2     0
#    rk3399       2      2     0
#    rpi3b        11     11    0
#    ----------------------------------
#
# You can register multiple devices to the tracker to accelerate the measurement in tuning.

###########################################
# Set Tuning Options
# ------------------
# Before tuning, we should apply some configurations. Here I use an RK3399 board
# as example. In your setting, you should modify the target and device_key accordingly.
# set :code:`use_android` to True if you use android phone.

#### DEVICE CONFIG ####

# Replace "aarch64-linux-gnu" with the correct target of your board.
# This target is used for cross compilation. You can query it by :code:`gcc -v` on your device.
target = tvm.target.create('llvm -device=arm_cpu -target=aarch64-linux-gnu')

# Also replace this with the device key in your tracker
device_key = 'rk3399'

# Set this to True if you use android phone
use_android = False

#### TUNING OPTION ####
network = 'resnet-18'
log_file = "%s.%s.log" % (device_key, network)
dtype = 'float32'

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 2000,
    'early_stopping': 800,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(
            build_func='ndk' if use_android else 'default'),
        runner=autotvm.RPCRunner(
            device_key, host='localhost', port=9190,
            number=5,
            timeout=4,
        ),
    ),
}

####################################################################
#
# .. note:: How to set tuning options
#
#   In general, the default values provided here work well.
#   If you have enough time budget, you can set :code:`n_trial`, :code:`early_stopping` larger,
#   which makes the tuning run longer.
#   If your device runs very slow or your conv2d operators have many GFLOPs, considering to
#   set timeout larger.
#
#   If your model has depthwise convolution, you could consider setting
#   :code:`try_spatial_pack_depthwise` be :code:`True`, which perform better than default
#   optimization in general. For example, on ARM CPU A53 2.0GHz, we find it could boost 1.6x
#   performance of depthwise convolution on Mobilenet V1 model.

###################################################################
# Begin Tuning
# ------------
# Now we can extract tuning tasks from the network and begin tuning.
# Here, we provide a simple utility function to tune a list of tasks.
# This function is just an initial implementation which tunes them in sequential order.
# We will introduce a more sophisticated tuning scheduler in the future.

# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=True,
               try_spatial_pack_depthwise=False):
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass

    # if we want to use spatial pack for depthwise convolution
    if try_spatial_pack_depthwise:
        tuner = 'xgb_knob'
        for i in range(len(tasks)):
            if tasks[i].name == 'topi_nn_depthwise_conv2d_nchw':
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host,
                                          'contrib_spatial_pack')
                tasks[i] = tsk

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    os.remove(tmp_log_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt):
    # extract workloads from nnvm graph
    print("Extract tasks...")
    net, params, input_shape, out_shape = get_network(network, batch_size=1)
    tasks = autotvm.task.extract_from_graph(net, target=target,
                                            shape={'data': input_shape}, dtype=dtype,
                                            symbols=(nnvm.sym.conv2d,))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(
                net, target=target, shape={'data': input_shape}, params=params, dtype=dtype)

        # export library
        tmp = tempdir()
        if use_android:
            from tvm.contrib import ndk
            filename = "net.so"
            lib.export_library(tmp.relpath(filename), ndk.create_shared)
        else:
            filename = "net.tar"
            lib.export_library(tmp.relpath(filename))

        # upload module to device
        print("Upload...")
        remote = autotvm.measure.request_remote(device_key, 'localhost', 9190,
                                                timeout=10000)
        remote.upload(tmp.relpath(filename))
        rlib = remote.load_module(filename)

        # upload parameters to device
        ctx = remote.context(str(target), 0)
        module = runtime.create(graph, rlib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

# tune_and_evaluate(tuning_option)

######################################################################
# Sample Output
# -------------
# The tuning needs to compile many programs and extract feature from them.
# So a high performance CPU is recommended.
# One sample output is listed below.
# It takes about 2 hours on a 32T AMD Ryzen Threadripper.
#
# .. code-block:: bash
#
#    Extract tasks...
#    Tuning...
#    [Task  1/12]  Current/Best:   22.37/  52.19 GFLOPS | Progress: (544/1000) | 406.59 s Done.
#    [Task  2/12]  Current/Best:    6.51/  18.77 GFLOPS | Progress: (608/1000) | 325.05 s Done.
#    [Task  3/12]  Current/Best:    4.67/  24.87 GFLOPS | Progress: (480/1000) | 372.31 s Done.
#    [Task  4/12]  Current/Best:   11.35/  46.83 GFLOPS | Progress: (736/1000) | 602.39 s Done.
#    [Task  5/12]  Current/Best:    1.01/  19.80 GFLOPS | Progress: (448/1000) | 262.16 s Done.
#    [Task  6/12]  Current/Best:    2.47/  23.76 GFLOPS | Progress: (672/1000) | 563.85 s Done.
#    [Task  7/12]  Current/Best:   14.57/  33.97 GFLOPS | Progress: (544/1000) | 465.15 s Done.
#    [Task  8/12]  Current/Best:    1.13/  17.65 GFLOPS | Progress: (576/1000) | 365.08 s Done.
#    [Task  9/12]  Current/Best:   14.45/  22.66 GFLOPS | Progress: (928/1000) | 724.25 s Done.
#    [Task 10/12]  Current/Best:    3.22/  15.36 GFLOPS | Progress: (864/1000) | 564.27 s Done.
#    [Task 11/12]  Current/Best:   11.03/  32.23 GFLOPS | Progress: (736/1000) | 635.15 s Done.
#    [Task 12/12]  Current/Best:    8.00/  21.65 GFLOPS | Progress: (1000/1000) | 1111.81 s Done.
#    Compile...
#    Upload...
#    Evaluate inference time cost...
#    Mean inference time (std dev): 162.59 ms (0.06 ms)

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
#   Finally, always feel free to ask our community for help on https://discuss.tvm.ai
