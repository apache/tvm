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
Auto-tuning a convolutional network for x86 CPU
===============================================
**Author**: `Yao Wang <https://github.com/kevinthesun>`_, `Eddie Yan <https://github.com/eqy>`_

This is a tutorial about how to tune convolution neural network
for x86 CPU.
"""
import os
import numpy as np

import tvm
from tvm import autotvm
from tvm import relay
from tvm.relay import testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.autotvm.graph_tuner import DPTuner, PBQPTuner
import tvm.contrib.graph_runtime as runtime

#################################################################
# Define network
# --------------
# First we need to define the network in relay frontend API.
# We can either load some pre-defined network from :code:`relay.testing`
# or building :any:`relay.testing.resnet` with relay.
# We can also load models from MXNet, ONNX and TensorFlow.
#
# In this tutorial, we choose resnet-18 as tuning example.


def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={input_name: input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = relay.Module.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


# Replace "llvm" with the correct target of your CPU.
# For example, for AWS EC2 c5 instance with Intel Xeon
# Platinum 8000 series, the target should be "llvm -mcpu=skylake-avx512".
# For AWS EC2 c4 instance with Intel Xeon E5-2666 v3, it should be
# "llvm -mcpu=core-avx2".
target = "llvm"

batch_size = 1
dtype = "float32"
model_name = "resnet-18"
log_file = "%s.log" % model_name
graph_opt_sch_file = "%s_graph_opt.log" % model_name

# Set the input name of the graph
# For ONNX models, it is typically "0".
input_name = "data"

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)


#################################################################
# Configure tensor tuning settings and create tasks
# -------------------------------------------------
# To get better kernel execution performance on x86 CPU,
# we need to change data layout of convolution kernel from
# "NCHW" to "NCHWc". To deal with this situation, we define
# conv2d_NCHWc operator in topi. We will tune this operator
# instead of plain conv2d.
#
# We will use local mode for tuning configuration. RPC tracker
# mode can be setup similarly to the approach in
# :ref:`tune_relay_arm` tutorial.

tuning_option = {
    'log_filename': log_file,
    'tuner': 'random',
    'early_stopping': None,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=10, repeat=1,
                                   min_repeat_ms=1000),
    ),
}


# You can skip the implementation of this function for this tutorial.
def tune_kernels(tasks,
                 measure_option,
                 tuner='gridsearch',
                 early_stopping=None,
                 log_filename='tuning.log'):

    for i, tsk in enumerate(tasks):
        prefix = "[Task %2d/%2d] " % (i+1, len(tasks))

        # converting conv2d tasks to conv2d_NCHWc tasks
        op_name = tsk.workload[0]
        if op_name == 'conv2d':
            func_create = 'topi_x86_conv2d_NCHWc'
        elif op_name == 'depthwise_conv2d_nchw':
            func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
        else:
            raise ValueError("Tuning {} is not supported on x86".format(op_name))

        task = autotvm.task.create(func_create, args=tsk.args,
                                   target=target, template_key='direct')
        task.workload = tsk.workload

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(task, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(task, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(task)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(task)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # do tuning
        n_trial=len(task.config_space)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(log_filename)])


# Use graph tuner to achieve graph level optimal schedules
# Set use_DP=False if it takes too long to finish.
def tune_graph(graph, dshape, records, opt_sch_file, use_DP=True):
    target_op = [relay.nn.conv2d]
    Tuner = DPTuner if use_DP else PBQPTuner
    executor = Tuner(graph, {input_name: dshape}, records, target_op, target)
    executor.benchmark_layout_transform(min_exec_num=2000)
    executor.run()
    executor.write_opt_sch2record_file(opt_sch_file)


########################################################################
# Finally, we launch tuning jobs and evaluate the end-to-end performance.

def tune_and_evaluate(tuning_opt):
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, data_shape, out_shape = get_network(model_name, batch_size)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params, ops=(relay.op.nn.conv2d,))

    # run tuning tasks
    print("Tuning...")
    tune_kernels(tasks, **tuning_opt)
    tune_graph(mod["main"], data_shape, log_file, graph_opt_sch_file)

    # compile kernels with graph-level best records
    with autotvm.apply_graph_best(graph_opt_sch_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # upload parameters to device
        ctx = tvm.cpu()
        data_tvm = tvm.nd.array((np.random.uniform(size=data_shape)).astype(dtype))
        module = runtime.create(graph, lib, ctx)
        module.set_input(input_name, data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=100, repeat=3)
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
#
# .. code-block:: bash
#
#    Extract tasks...
#    Tuning...
#    [Task  1/12]  Current/Best:  598.05/2497.63 GFLOPS | Progress: (252/252) | 1357.95 s Done.
#    [Task  2/12]  Current/Best:  522.63/2279.24 GFLOPS | Progress: (784/784) | 3989.60 s Done.
#    [Task  3/12]  Current/Best:  447.33/1927.69 GFLOPS | Progress: (784/784) | 3869.14 s Done.
#    [Task  4/12]  Current/Best:  481.11/1912.34 GFLOPS | Progress: (672/672) | 3274.25 s Done.
#    [Task  5/12]  Current/Best:  414.09/1598.45 GFLOPS | Progress: (672/672) | 2720.78 s Done.
#    [Task  6/12]  Current/Best:  508.96/2273.20 GFLOPS | Progress: (768/768) | 3718.75 s Done.
#    [Task  7/12]  Current/Best:  469.14/1955.79 GFLOPS | Progress: (576/576) | 2665.67 s Done.
#    [Task  8/12]  Current/Best:  230.91/1658.97 GFLOPS | Progress: (576/576) | 2435.01 s Done.
#    [Task  9/12]  Current/Best:  487.75/2295.19 GFLOPS | Progress: (648/648) | 3009.95 s Done.
#    [Task 10/12]  Current/Best:  182.33/1734.45 GFLOPS | Progress: (360/360) | 1755.06 s Done.
#    [Task 11/12]  Current/Best:  372.18/1745.15 GFLOPS | Progress: (360/360) | 1684.50 s Done.
#    [Task 12/12]  Current/Best:  215.34/2271.11 GFLOPS | Progress: (400/400) | 2128.74 s Done.
#    Compile...
#    Evaluate inference time cost...
#    Mean inference time (std dev): 3.16 ms (0.03 ms)
