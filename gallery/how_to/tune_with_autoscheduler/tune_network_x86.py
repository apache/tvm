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
Auto-scheduling a Neural Network for x86 CPU
============================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, \
            `Chengfan Jia <https://github.com/jcf94/>`_

Auto-tuning for specific devices and workloads is critical for getting the
best performance. This is a tutorial on how to tune a whole neural
network for x86 CPU with the auto-scheduler.

To auto-tune a neural network, we partition the network into small subgraphs and
tune them independently. Each subgraph is treated as one search task.
A task scheduler slices the time and dynamically allocates time resources to
these tasks. The task scheduler predicts the impact of each task on the end-to-end
execution time and prioritizes the one that can reduce the execution time the most.

For each subgraph, we use the compute declaration in :code:`tvm/python/topi` to
get the computational DAG in the tensor expression form.
We then use the auto-scheduler to construct a search space of this DAG and search
for good schedules (low-level optimizations).

Different from the template-based :ref:`autotvm <tutorials-autotvm-sec>` which relies on
manual templates to define the search space, the auto-scheduler does not require any
schedule templates. In other words, the auto-scheduler only uses the compute declarations
in :code:`tvm/python/topi` and does not use existing schedule templates.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""

# sphinx_gallery_start_ignore
from tvm import testing

testing.utils.install_request_hook(depth=3)
# sphinx_gallery_end_ignore

import numpy as np

import tvm
from tvm import relay, auto_scheduler
from tvm.relay import data_dep_optimization as ddo
import tvm.relay.testing
from tvm.contrib import graph_executor

#################################################################
# Define a Network
# ----------------
# First, we need to define the network with relay frontend API.
# We can load some pre-defined network from :code:`tvm.relay.testing`.
# We can also load models from MXNet, ONNX, PyTorch, and TensorFlow
# (see :ref:`front end tutorials<tutorial-frontend>`).
#
# For convolutional neural networks, although auto-scheduler can work correctly
# with any layout, we found the best performance is typically achieved with NHWC layout.
# We also implemented more optimizations for NHWC layout with the auto-scheduler.
# So it is recommended to convert your models to NHWC layout to use the auto-scheduler.
# You can use :ref:`ConvertLayout <convert-layout-usage>` pass to do the layout conversion in TVM.


def get_network(name, batch_size, layout="NHWC", dtype="float32", use_sparse=False):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet50_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    elif name == "mlp":
        mod, params = relay.testing.mlp.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=image_shape, num_classes=1000
        )
    else:
        raise ValueError("Network not found.")

    if use_sparse:
        from tvm.topi.sparse.utils import convert_model_dense_to_sparse

        mod, params = convert_model_dense_to_sparse(mod, params, bs_r=4, random_params=True)

    return mod, params, input_shape, output_shape


# Define the neural network and compilation target.
# If the target machine supports avx512 instructions, replace the
# "llvm -mcpu=core-avx2" with "llvm -mcpu=skylake-avx512"
network = "resnet-50"
use_sparse = False
batch_size = 1
layout = "NHWC"
target = tvm.target.Target("llvm -mcpu=core-avx2")
dtype = "float32"
log_file = "%s-%s-B%d-%s.json" % (network, layout, batch_size, target.kind.name)

#################################################################
# Extract Search Tasks
# --------------------
# Next, we extract the search tasks and their weights from a network.
# The weight of a task is the number of appearances of the task's subgraph
# in the whole network.
# By using the weight, we can approximate the end-to-end latency of the network
# as :code:`sum(latency[t] * weight[t])`, where :code:`latency[t]` is the
# latency of a task and :code:`weight[t]` is the weight of the task.
# The task scheduler will just optimize this objective.

# Extract tasks from the network
print("Get model...")
mod, params, input_shape, output_shape = get_network(
    network,
    batch_size,
    layout,
    dtype=dtype,
    use_sparse=use_sparse,
)
print("Extract tasks...")
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    print(task.compute_dag)

#################################################################
# Begin Tuning
# ------------
# Now, we set some options for tuning and launch the search tasks
#
# * :code:`num_measure_trials` is the number of measurement trials we can use during the tuning.
#   You can set it to a small number (e.g., 200) for a fast demonstrative run.
#   In practice, we recommend setting it around :code:`800 * len(tasks)`,
#   which is typically enough for the search to converge.
#   For example, there are 29 tasks in resnet-50, so we can set it as 20000.
#   You can adjust this parameter according to your time budget.
# * In addition, we use :code:`RecordToFile` to dump measurement records into a log file,
#   The measurement records can be used to query the history best, resume the search,
#   and do more analyses later.
# * see :any:`auto_scheduler.TuningOptions`,
#   :any:`auto_scheduler.LocalRunner` for more parameters.
#


def run_tuning():
    print("Begin tuning...")
    tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=200,  # change this to 20000 to achieve the best performance
        runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    if use_sparse:
        from tvm.topi.sparse.utils import sparse_sketch_rules

        search_policy = [
            auto_scheduler.SketchPolicy(
                task,
                program_cost_model=auto_scheduler.XGBModel(),
                init_search_callbacks=sparse_sketch_rules(),
            )
            for task in tasks
        ]

        tuner.tune(tune_option, search_policy=search_policy)
    else:
        tuner.tune(tune_option)


# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

# run_tuning()


######################################################################
# .. note:: Explain the printed information during tuning
#
#   During the tuning, a lot of information will be printed on the console.
#   They are used for debugging purposes. The most important info is the output
#   of the task scheduler. The following table is a sample output.
#
#   .. code-block:: c
#
#     ----------------------------------------------------------------------
#     ------------------------------  [ Task Scheduler ]
#     ----------------------------------------------------------------------
#     |  ID  | Latency (ms) | Speed (GFLOPS) | Trials |
#     -------------------------------------------------
#     |    0 |        0.010 |           0.40 |     64 |
#     |    1 |        0.087 |          47.19 |     64 |
#     |    2 |        0.008 |          -0.00 |     64 |
#     |    3 |        0.177 |         582.07 |     64 |
#     |    4 |        0.268 |         862.37 |    256 |
#     |    5 |        0.166 |         621.13 |    128 |
#     |    6 |        0.170 |         605.10 |    128 |
#     |    7 |        0.128 |         403.20 |     64 |
#     |    8 |        0.189 |         545.71 |     64 |
#     |    9 |        0.231 |        1001.01 |    448 |
#     |   10 |        0.155 |         664.80 |    256 |
#     |   11 |        0.155 |         662.86 |    256 |
#     |   12 |        0.119 |         434.08 |     64 |
#     |   13 |        0.199 |         522.13 |     64 |
#     |   14 |        0.235 |         986.56 |    320 |
#     |   15 |        0.149 |         689.13 |    128 |
#     |   16 |        0.155 |         664.80 |    192 |
#     |   17 |        0.151 |         340.64 |     64 |
#     |   18 |        0.176 |         597.55 |    128 |
#     |   19 |        0.220 |        1054.37 |    192 |
#     |   20 |        0.150 |         686.01 |    128 |
#     |   21 |        0.159 |         650.88 |    128 |
#     |   22 |        0.073 |         358.19 |     64 |
#     |   23 |        0.031 |          70.63 |     64 |
#     |   24 |        0.251 |         947.73 |    128 |
#     |   25 |        0.157 |         652.47 |    128 |
#     |   26 |        0.215 |         954.84 |    128 |
#     |   27 |        0.237 |         868.92 |    128 |
#     |   28 |        0.266 |         774.06 |    128 |
#     -------------------------------------------------
#     Estimated total latency: 10.016 ms      Trials: 3992    Used time : 1131 s      Next ID: 15
#
#   This table lists the latency and (estimated) speed of all tasks.
#   It also lists the allocation of measurement trials for all tasks.
#   The last line prints the total weighted latency of these tasks,
#   which can be a rough estimation of the end-to-end execution time
#   of the network.
#   The last line also prints the total number of measurement trials,
#   total time spent on auto-tuning and the id of the next task to tune.
#
#   There will also be some "tvm::Error"s errors, because the
#   auto-scheduler will try some invalid schedules.
#   You can safely ignore them if the tuning can continue, because these
#   errors are isolated from the main process.
#

######################################################################
# .. note:: Terminate the tuning earlier
#
#   You can terminate the tuning earlier by forcibly killing this process.
#   As long as you get at least one valid schedule for each task in the log file,
#   you should be able to do the compilation (the secion below).
#


#################################################################
# Compile and Evaluate
# --------------------
# After auto-tuning, we can compile the network with the best schedules we found.
# All measurement records are dumped into the log file during auto-tuning,
# so we can read the log file and load the best schedules.

# Compile with the history best
print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        lib = relay.build(mod, target=target, params=params)

# Create graph executor
dev = tvm.device(str(target), 0)
module = graph_executor.GraphModule(lib["default"](dev))
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
module.set_input("data", data_tvm)

# Evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=3, min_repeat_ms=500))


#################################################################
# Other Tips
# ----------
# 1. During the tuning, the auto-scheduler needs to compile many programs and
#    extract feature from them. This part is CPU-intensive,
#    so a high-performance CPU with many cores is recommended for faster search.
# 2. You can use :code:`python3 -m tvm.auto_scheduler.measure_record --mode distill -i log.json`
#    to distill the large log file and only save the best useful records.
# 3. You can resume a search from the previous log file. You just need to
#    add a new argument :code:`load_log_file` when creating the task scheduler
#    in function :code:`run_tuning`. Say,
#    :code:`tuner = auto_scheduler.TaskScheduler(tasks, task_weights, load_log_file=log_file)`
# 4. If you have multiple target CPUs, you can use all of them for measurements to
#    parallelize the measurements. Check this :ref:`section <tutorials-autotvm-scale-up-rpc-tracker>`
#    to learn how to use the RPC Tracker and RPC Server.
#    To use the RPC Tracker in auto-scheduler, replace the runner in :code:`TuningOptions`
#    with :any:`auto_scheduler.RPCRunner`.
