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
Auto-scheduling High Performance Convolution on NVIDIA GPUs
===========================================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, \
            `Chengfan Jia <https://github.com/jcf94>`_, \
            `Minmin Sun <https://github.com/minminsun>`_, \
            `Zhao Wu <https://github.com/FrozenGene>`_

This is an tutorial for searching high performance schedule for NVIDIA GPU using
Ansor auto-scheduler. By running Ansor on this template, we can outperform the
vendor provided library CuDNN in many cases.
"""

######################################################################
# Install dependencies
# --------------------
# To use autotvm package in tvm, we need to install some extra dependencies.
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost tornado
#
# To make TVM run faster in tuning, it is recommended to use cython
# as FFI of tvm. In the root directory of tvm, execute
#
# .. code-block:: bash
#
#   pip3 install --user cython
#   sudo make cython3
#
# Now return to python code. Import packages.

import random
import sys

import numpy as np
import tvm
import topi
from topi.testing import conv2d_nchw_python
from tvm import te

# the module is called `ansor`
from tvm import ansor

######################################################################
# Step 1:  Define the search task
# -------------------------------
# There are plenty of useful schedule primitives in tvm. You can also find
# some tutorials that describe them in more details, such as
# (1). :ref:`opt-conv-gpu`
# (2). `Optimizing DepthwiseConv on NVIDIA GPU <https://tvm.apache.org/2017/08/22/Optimize-Deep-Learning-GPU-Operators-with-TVM-A-Depthwise-Convolution-Example>`_
#
# It's usually a hard job if one wants to get a high performance schedule for a
# specific workload. Even writing an AutoTVM tunable template needs user to have
# expertises on how each schedule primitive works as well as how they finally
# reflect on the hardward architecture.
#
# However, with Ansor this will be quite simple. Firstly, define the target workload.
# Both :code:`tvm.te` API or topi op API are fine to be used.
#
# We can use the retuned :code:`Tensors` to create a ComputeDAG just like what we do
# in :ref:`ansor-simple-subgraph`, while the way to use workload registry is more
# recommended.

# Use an extra function decorator to regist this workload
@ansor.register_auto_scheduler_workload_func
def conv2d_nchw(N, H, W, CO, CI, KH, KW, stride, padding):
    data = te.placeholder((N, CI, H, W), name='data')
    kernel = te.placeholder((CO, CI, KH, KW), name='kernel')
    conv = topi.nn.conv2d_nchw(data, kernel, stride, padding, dilation=1, out_dtype='float32')

    return [data, kernel, conv]

######################################################################
# Step 2:  Search through the schedule space
# ------------------------------------------
# We pick the last layer on resnet as test case.
# Since our space is very large, :code:`XGBModel` is most suitable
# for our case. Here we only do 20 trials for demonstration.
# In practice, making 1000 trials usually can find some good kernels
# for this workload.

tgt = tvm.target.cuda()

# The last layer in resnet
N, H, W, CO, CI, KH, KW, strides, padding = 1, 7, 7, 512, 512, 3, 3, (1, 1), (1, 1)
# Generate workload key with the ansor API
wkl_key = ansor.make_workload_key_func(conv2d_nchw, (N, H, W, CO, CI, KH, KW, strides, padding))
# Generate ComputeDAG using the workload key
dag = ansor.workload_key_to_dag(wkl_key)
task = ansor.SearchTask(dag, wkl_key, target=tgt)

log_file = "conv2d_nchw.json"
seed = 0
random.seed(seed)
cost_model = ansor.XGBModel()
search_policy = ansor.MetaTileRewritePolicy(cost_model, seed=seed)

#########################################################################
# The :code:`ansor.RPCRunnerWarpper` is used to create a RPC runner environment,
# 
# Use local gpu, measure 10 times for every schedule to reduce variance. The timeout
# for each running is set to 4 seconds.
#
# During the searching process, we may generate several invalid schedules and they
# will be filtered out. It's fine to see "Encountered errors during feature extraction."
# in the tuning logs.

with ansor.RPCRunnerWarpper("cuda", repeat=3, min_repeat_ms=100, timeout=4) as rpc_runner:
    tune_option = ansor.TuneOption(n_trials=20,
                                   runner=rpc_runner.runner,
                                   callbacks=[ansor.LogToFile(log_file)])
    state = ansor.auto_schedule(task, search_policy,
                                tune_option=tune_option)
    print(state)

#########################################################################
# Finally we can directly use the returned result to get the generated schedule,
# while in the following tutorial we'll show how to inspect the best config from
# log file, check correctness, and measure running time.

# Get history best from log file
inp, res = ansor.best_measure_pair_in_file(log_file)
# Get the task ComputeDAG from log result
dag = ansor.workload_key_to_dag(inp.task.workload_key)
# Apply log result to TVM schedule
s, arg_bufs = dag.apply_steps_from_state(inp.state)
func = tvm.build(s, arg_bufs, target=tgt)

# check correctness
a_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
w_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)
c_np = conv2d_nchw_python(a_np, w_np, strides, padding)

ctx = tvm.gpu()
a_tvm = tvm.nd.array(a_np, ctx=ctx)
w_tvm = tvm.nd.array(w_np, ctx=ctx)
c_tvm = tvm.nd.empty(c_np.shape, ctx=ctx)
func(a_tvm, w_tvm, c_tvm)

tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

# Evaluate running time. Here we choose a large repeat number (400) to reduce the noise
# and the overhead of kernel launch. You can also use nvprof to validate the result.
evaluator = func.time_evaluator(func.entry_name, ctx, number=400)
print('Time cost of this operator: %f' % evaluator(a_tvm, w_tvm, c_tvm).mean)

