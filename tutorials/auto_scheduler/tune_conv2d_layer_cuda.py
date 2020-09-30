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
.. _auto-scheduler-conv-gpu:

Auto-scheduling a convolution layer for GPU
===========================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, \
            `Chengfan Jia <https://github.com/jcf94/>`_


Different from the existing :ref:`autotvm <tutorials-autotvm-sec>` which relies on 
manual templates to define the search space, the auto-scheduler does not require any templates.
The auto-scheduler is template-free, so users only need to write the computation declaration without
any schedule commands or templates.
The auto-scheduler can automatically generate a large
search space and find a good schedule in the space.

We use a convolution layer as an example in this tutorial.
"""

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.testing import conv2d_nchw_python

######################################################################
# Define the computation
# ^^^^^^^^^^^^^^^^^^^^^^
# To begin with, let us define the computation of a convolution layer.
# The function should return the list of input/output tensors.
# From these tensors, the auto-scheduler can get the whole computational graph.


@auto_scheduler.register_workload
def conv2d_layer(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name="data")
    kernel = te.placeholder((CO, CI, KH, KW), name="kernel")
    bias = te.placeholder((1, CO, 1, 1), name="bias")
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype="float32")
    out = topi.nn.relu(conv + bias)
    return [data, kernel, bias, out]


######################################################################
# Create the search task
# ^^^^^^^^^^^^^^^^^^^^^^
# We then create a search task for the last convolution layer in the resnet.

target = tvm.target.Target("cuda")

# the last layer in resnet
N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
task = auto_scheduler.create_task(conv2d_layer, (N, H, W, CO, CI, KH, KW, strides, padding), target)

# Inspect the computational graph
print(task.compute_dag)

######################################################################
# Next, we set parameters for the auto-scheduler. These parameters
# mainly specify how we do the measurement during the search and auto-tuning.
#
# * :code:`measure_ctx` launches a different process for measurement. This
#   provides an isolation. It can protect the master process from GPU crashes
#   happended during measurement and avoid other runtime conflicts.
# * :code:`min_repeat_ms` defines the minimum duration of one "repeat" in every measurement.
#   This can warmup the GPU, which is necessary to get accurate measurement results.
#   Typically, we recommend a value > 300 ms.
# * :code:`num_measure_trials` is the number of measurement trials we can use during the search.
#   We only make 10 trials in this tutorial for a fast demonstration. In practice, 1000 is a
#   good value for the search to converge. You can do more trials according to your time budget.
# * In addition, we use :code:`RecordToFile` to dump measurement records into a file `conv2d.json`.
#   The measurement records can be used to query the history best, resume the search,
#   and do more analyses later.
# * see :any:`auto_scheduler.TuningOptions`,
#   :any:`auto_scheduler.LocalRPCMeasureContext` for more parameters.

measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=300)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile("conv2d.json")],
)

######################################################################
# Run the search
# ^^^^^^^^^^^^^^
# Now we get all inputs ready. Pretty simple, isn't it?
# We can kick off the search and let the auto-scheduler do its magic.
# After some measurement trials, it will return the best schedule it found.

sch, args = auto_scheduler.auto_schedule(task, tuning_options=tune_option)

######################################################################
# We can lower the schedule to see the IR after auto-scheduling.
# The auto-scheduler correctly performs optimizations including multi-level tiling,
# cooperative fetching, unrolling and operator fusion.

print(tvm.lower(sch, args, simple_mode=True))

######################################################################
# Check correctness and evaluate performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We build the binary and check its correctness and performance.

func = tvm.build(sch, args, target)

# check correctness
data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
weight_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
bias_np = np.random.uniform(size=(1, CO, 1, 1)).astype(np.float32)
conv_np = conv2d_nchw_python(data_np, weight_np, strides, padding)
out_np = np.maximum(conv_np + bias_np, 0.0)

ctx = tvm.gpu()
data_tvm = tvm.nd.array(data_np, ctx=ctx)
weight_tvm = tvm.nd.array(weight_np, ctx=ctx)
bias_tvm = tvm.nd.array(bias_np, ctx=ctx)
out_tvm = tvm.nd.empty(out_np.shape, ctx=ctx)
func(data_tvm, weight_tvm, bias_tvm, out_tvm)

# Check results
np.testing.assert_allclose(out_np, out_tvm.asnumpy(), rtol=1e-3)

# Evaluate execution time
evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(data_tvm, weight_tvm, bias_tvm, out_tvm).results) * 1000)
)

######################################################################
# Using the record file
# ^^^^^^^^^^^^^^^^^^^^^
# During the search, all measuremnt records are dumpped into the record
# file "conv2d.json". The measurement records can be used to re-apply search results,
# resume the search, and perform other analyses.

######################################################################
# Here is an example where we load the best schedule from a file,
# print the equivalent python schedule API, and build the binary again.

# Load the measuremnt record for the best schedule
inp, res = auto_scheduler.load_best("conv2d.json", task.workload_key)

# Print equivalent python schedule API. This can be used for debugging and
# learning the behavior of the auto-scheduler.
print("Equivalent python schedule:")
print(task.compute_dag.print_python_code_from_state(inp.state))

# Rebuild the binary. This shows how you can apply the best schedule from a
# log file without reruning the search again.
sch, args = task.compute_dag.apply_steps_from_state(inp.state)
func = tvm.build(sch, args, target)

######################################################################
# A more complicated example is to resume the search.
# In this case, we need to create the search policy and cost model by ourselves
# and resume the status of search policy and cost model with the log file.
# In the example below we resume the status and do more 5 trials.


log_file = "conv2d.json"
cost_model = auto_scheduler.XGBModel()
cost_model.update_from_file(log_file)
search_policy = auto_scheduler.SketchPolicy(
    task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
)
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=5,
    runner=measure_ctx.runner,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
)
sch, args = auto_scheduler.auto_schedule(task, search_policy, tuning_options=tune_option)

# kill the measurement process
del measure_ctx
