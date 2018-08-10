"""
Graph level tuning for CNN models on Intel CPU
=========================================================================
**Author**: `Yao Wang <https://https://github.com/kevinthesun>`_

This article is an introductory tutorial to use AutoTVM graph tuner to get
optimal convolution schedules for CNN models on Intel CPU.
"""
import os
import json
import numpy as np
import nnvm
import tvm

from nnvm import symbol as sym
from nnvm.testing.utils import create_workload
from tvm.contrib import graph_runtime
from tvm.autotvm.graph_tuner import tensor_executor
from tvm.autotvm.graph_tuner import graph_executor
from tvm.autotvm.graph_tuner.utils import read_sch_from_json, infer_layout_shape_avx, \
    get_conv2d_workload, write_sch_to_json, load_conv_sch_avx


####################################################################################
# Step 1: Create a vanilla CNN graph and set parameters
# -----------------------------------------------------
# In this tutorial, we use a small CNN graph to reduce the tuning time.
# For Imagenet CNN models, it can take several hours to complete tensor searching.
# Learning-based method introduced by AutoTVM can be used to speed up this process.

# create vanilla CNN model
batch_size = 1
image_shape = (3, 14, 14)
dshape = (batch_size,) + image_shape
out_num = 10
oshape = (1, out_num)

data = sym.Variable("data")
conv0 = sym.conv2d(data, channels=8, kernel_size=(3,3), padding=(1,1))
relu0 = sym.relu(conv0)
bn0 = sym.batch_norm(relu0)
conv1 = sym.conv2d(bn0, channels=16, kernel_size=(1,1))
relu1 = sym.relu(conv1)
bn1 = sym.batch_norm(relu1)
flatten = sym.flatten(bn1)
out = sym.dense(flatten, units=out_num)

net, params = create_workload(out, batch_size, image_shape)
graph = nnvm.graph.create(net)

# In this tutorial, we use x86 AVX convolution schedule template. As a result, we need
# to use Intel CPU with AVX instruction set to get optimal performance. We set target
# to "llvm" for demo purpose.
target = "llvm"
input_shapes = {"data": dshape}
target_op = "conv2d"
data_layout = "NCHWc"
log_file = "%s/demo_graph_tuner.log" % os.getcwd()

# We use single cores for searching. The number of cores used for searching
# doesn't affect the final schedules. For example, we can search with single core
# but apply the optimal schedules on models running with multiple cores.
#
# If multiple cores are available, it is recommended to use them to accelerate
# searching process.
num_threads = 1
os.environ["TVM_NUM_THREADS"] = str(num_threads)

# RPC mode is supported in graph tuner. Currently graph tuner supports connecting
# to rpc server launched on target device from host machine. This is useful
# when searching schedules for devices with limited resources.
#
# To use rpc mode, we need to set environment variable "TVM_NUM_THREADS"
# on target cpu devices if we want to use multiple cores for searching.
#
# .. note::
#
#   RPC tracker is not supported yet. It will be added later.
#
# By default we use local mode.
rpc_mode = 0
if rpc_mode == 0:
    rpc_hosts = None
    rpc_port = None

# To use rpc host mode, setup rpc server with TVM runtime rpc module on target devices
# and make sure you can connect to them. We can set multiple target devices and graph
# tuner will automatically distribute workloads.
if rpc_mode == 1:
    rpc_hosts = ["localhost", "localhost"]
    rpc_ports = [9090, 9091]

####################################################################################
# Step 2: Tensor level tuning for convolution workloads in the graph
# ------------------------------------------------------------------
# Convolution is the performance-bottleneck operator in CNN model.
# In this step, we run experiments for all convolution workloads in vanilla model to
# pick fast schedules. In this step, we don't worry graph-level optimization yet.
#
# .. note::
#
#   It is highly recommended to stored the schedules got in this step in a centralized
#   place, such as in a json file.(Graph tuner provide utility functions to support this)
#   Since schedules for workloads are model agnostic, we can largely reduce the number of
#   searching. For example, if we store the schedules of resnet50 in a json file and we
#   now want to optimize resnet152, we just need to read from the json file instead of
#   rerun tensor searching.
#
# Tensor tuner executes exhaustive search for search space defined by
# tensor_executor.Conv2dAVXExecutor. This step can be replaced completely by learning
# based methods for shorter tuning time.

# Get all convolution workloads in graph
sch_file = "%s/vanilla_graph_schedule.json" % os.getcwd()
new_workloads = get_conv2d_workload(graph, input_shapes)
for wkl in new_workloads:
    print(wkl)

# Run benchmark for all new workloads and generate some schedule candidates for each workload
# If we have already saved schedules into a file, we can load them by simply call
#   .. code-block:: python
#
#     sch_dict = read_sch_from_json(sch_file)
# In this case, tensor tuner will check
# unless force_update is set to True.
sch_dict = {}
tensor_tuner = tensor_executor.Conv2dAVXExecutor(sch_dict, target, verbose=False, log_file=log_file)
tensor_tuner.workload_execute(new_workloads)
sch_dict = tensor_tuner.schedule_dict

# Optional but highly recommended: writing result schedule dictionary back to json file.
write_sch_to_json(sch_dict, sch_file)

####################################################################################
# Step 3: Graph level tuning to get end to end optimal performance
# ----------------------------------------------------------------
# Simply picking the fastest schedule for each workload, will not necessarily gives
# optimal end to end performance. This is due to the potential layout transformation
# overhead. We need to balance the trade off between fast schedules and extra layout
# transformations. In this step, we perform global tuning to achieve graph level optimal
# schedules.
graph_tuner = graph_executor.DPExecutor(graph, input_shapes, sch_dict, target_op, data_layout,
                                        get_conv2d_workload, infer_layout_shape_avx, verbose=False,
                                        log_file=log_file)
graph_tuner.benchmark_layout_transform(target, min_exec_num=100)
graph_tuner.run()
# The result optimal schedules are a list with the same order for
# all convolution operators in network.
opt_sch_list = graph_tuner.get_optimal_schedules()

# It is recommended to store result back to a json file
opt_sch_file = "%s/vanilla_model_opt_sch.json" % os.getcwd()
opt_sch_dict = {"schedules": [str(sch) for sch in opt_sch_list]}
with open(opt_sch_file, "w") as of:
    json.dump(opt_sch_dict, of, indent=2)

####################################################################################
# Step 4: Load optimal schedules, compile model and run inference.
data_array = np.random.uniform(0, 255, size=dshape).astype("float32")
ctx = tvm.cpu()
load_conv_sch_avx(opt_sch_list)
with nnvm.compiler.build_config(opt_level=3):
    graph, lib, params = nnvm.compiler.build(
        net, target, shape=input_shapes, params=params)
module = graph_runtime.create(graph, lib, ctx)
module.set_input(**params)
module.set_input("data", tvm.nd.array(data_array, ctx=ctx))
time_eval = module.module.time_evaluator("run", ctx, number=100)
print("Average inference time for vanilla CNN model: %f second." % time_eval().mean)

os.remove(log_file)
os.remove(sch_file)
os.remove(opt_sch_file)
