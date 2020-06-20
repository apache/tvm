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
.. _ansor-simple-subgraph:

Writing compute expression and Using Ansor auto-scheduler
=========================================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, \
            `Chengfan Jia <https://github.com/jcf94>`_, \
            `Minmin Sun <https://github.com/minminsun>`_, \
            `Zhao Wu <https://github.com/FrozenGene>`_

This is an introduction tutorial to the auto-scheduler module in TVM.

There are two steps in auto-scheduling.
The first step is defining the target task.
The second step is running a search algorithm to auto explore the schedule.
In this tutorial, you can learn how to perform these two steps in TVM.
The whole workflow is illustrated by a matrix multiplication with bias add example.
"""

######################################################################
# Install dependencies
# --------------------
# To use Ansor package in TVM, we need to install some extra dependencies.
# This step (installing xgboost) can be skipped as it doesn't need XGBoost
# (change "3" to "2" if you use python2):
#
# .. code-block:: bash
#
#   pip3 install --user psutil xgboost
#
# To make TVM run faster in tuning, it is recommended to use cython
# as FFI of TVM. In the root directory of TVM, execute
# (change "3" to "2" if you use python2):
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
from tvm import te

# the module is called `ansor`
from tvm import ansor

######################################################################
# Step 1:  Define the target compute subgraph
# -------------------------------------------
# In this section, we will write a deterministic TVM compute expression code
# to a compute subgraph.
#
# .. note:: Comparing to :ref:`tutorials-autotvm-sec`
#
#  In Ansor, we do not need users to provide a schedule template, the only input
#  is the compute expression writing by :code:`tvm.te` API or topi op API.
#
# Here is how we implement a matrix multiplication subgraph in TVM.

# Matmul with bias add
def matmul_add(N, L, M, dtype):
    A = te.placeholder((N, L), name='A', dtype=dtype)
    B = te.placeholder((L, M), name='B', dtype=dtype)
    C = te.placeholder((N, M), name='C', dtype=dtype)

    k = te.reduce_axis((0, L), name='k')
    mul = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
                     name='Mul')
    D = te.compute((N, M), lambda i, j: C[i, j] + mul[i, j], name='D')

    return [A, B, C, D]

######################################################################
# Step 2:  Search through the schedule space
# ------------------------------------------
# In step 1, we build the compute subgraph.
# The next step is to pick a cost model as well as a search policy and explore the
# possible schedule.
#
# Auto-scheduler in TVM
# ^^^^^^^^^^^^^^^^^^^^^
# The job for the Ansor auto-scheduler can be described by following pseudo code
#
#   .. code-block:: c
#
#    ct = 0
#    while ct < max_number_of_trials:
#        auto generate a batch of schedules
#        measure this batch of schedules on real hardware and get results
#        ct += batch_size
#
# When proposing the next batch of schedules, Ansor can take different cost models to
# guide the schedule generating process.
#
# * :code:`RandomModel`: Generate and take new schedule randomly
# * :code:`XGBModel`: Use XGBoost model to estimate the performance of potential schedules, try to pick schedules with better performance in each step
#
# XGBModel can explore more efficiently and find better schedules.

################################################################
# Begin tuning
# ^^^^^^^^^^^^
# Here we continue our matrix multiplication example.
#
# The :code:`ansor.ComputeDAG` takes the Tensor list as input, and generates
# a dag structure. During which process, :code:`ansor.ComputeDAG` will
# do some analyzes with the target subgraph and the results will be used in
# search policy later.
#
# Then we create the :code:`tvm.target` and a tuning task.

N, L, M = 128, 128, 128
A, B, C, D = matmul_add(N, L, M, 'float32')
dag = ansor.ComputeDAG([A, B, C, D])

print(dag)
print(dag.access_analyzer)

tgt = tvm.target.create("llvm")
task = ansor.SearchTask(dag, "test", tgt)

################################################################
# Next, we choose random model and create a default search policy:
# :code:`ansor.MetaTileRewritePolicy`.
#
# We only make 5 trials in this tutorial for demonstration. In practice,
# you can do more trials according to your time budget.
# :code:`ansor.LogToFile` callback will log the tuning results into a
# log file, which can be used to get the best config later.
# :code:`ansor.PreloadMeasuredStates` callback will load measured states
# from history log before schedule search, we can add this callback to make
# sure a same schedule will never be measured for multiple times.

log_file = "matmul_add.json"

seed = 0
random.seed(seed)
cost_model = ansor.RandomModel()
search_policy = ansor.MetaTileRewritePolicy(cost_model, seed=seed)

tune_option = ansor.TuneOption(n_trials=5,
                               measure_callbacks=[ansor.LogToFile(log_file)],
                               pre_search_callbacks=[ansor.PreloadMeasuredStates(log_file)])

################################################################
# Then just call :code:`ansor.auto_schedule` and Ansor will try to find a high
# performance schedule for the target subgraph automatically.
#
# The returned result will be a :code:`te.schedule` and a list of :code:`te.Tensor`,
# which can be used as the input of :code:`tvm.lower` or :code:`tvm.build`.

s, arg_bufs = ansor.auto_schedule(task, search_policy=search_policy,
                                  tune_option=tune_option)

print("==== Get Lowered Stmt ====")
print(tvm.lower(s, arg_bufs, simple_mode=True))

#########################################################################
# Check the correctness to make sure we generate a right schedule.

func = tvm.build(s, arg_bufs)

# check correctness
a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)
c_np = np.random.uniform(size=(N, M)).astype(np.float32)
d_np = a_np.dot(b_np) + c_np

d_tvm = tvm.nd.empty(d_np.shape)
func(tvm.nd.array(a_np), tvm.nd.array(b_np), tvm.nd.array(c_np), d_tvm)

tvm.testing.assert_allclose(d_np, d_tvm.asnumpy(), rtol=1e-2)
