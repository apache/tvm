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
Auto-scheduling matrix multiplication for CPU
=============================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, \
            `Chengfan Jia <https://github.com/jcf94/>`_

Different from the existing :ref:`autotvm <tutorials-autotvm-sec>` which relies on 
manual templates to define the search space, the auto-scheduler does not require any templates.
The auto-scheduler is template-free, so users only need to write the computation declaration without
any schedule commands or templates.
The auto-scheduler can automatically generate a large
search space and find a good schedule in the space.

We use matrix multiplication as an example in this tutorial.
"""

import numpy as np
import tvm
from tvm import te, auto_scheduler

######################################################################
# Define the computation
# ^^^^^^^^^^^^^^^^^^^^^^
# To begin with, let us define the computation of a matmul with bias add.
# The function should return the list of input/output tensors.
# From these tensors, the auto-scheduler can get the whole computational graph.


@auto_scheduler.register_workload
def matmul_add(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    C = te.placeholder((N, M), name="C", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="matmul")
    out = te.compute((N, M), lambda i, j: matmul[i, j] + C[i, j], name="out")

    return [A, B, C, out]


######################################################################
# Create the search task
# ^^^^^^^^^^^^^^^^^^^^^^
# We then create a search task with N=L=M=128 and dtype="float32"
# If your machine supports avx instructions, you can
#
#   - replace "llvm" below with "llvm -mcpu=core-avx2" to enable AVX2
#   - replace "llvm" below with "llvm -mcpu=skylake-avx512" to enable AVX-512

target = tvm.target.Target("llvm")
task = tvm.auto_scheduler.create_task(matmul_add, (128, 128, 128, "float32"), target)

# Inspect the computational graph
print(task.compute_dag)

######################################################################
# Next, we set parameters for the auto-scheduler.
#
# * :code:`num_measure_trials` is the number of measurement trials we can use during the search.
#   We only make 10 trials in this tutorial for a fast demonstration. In practice, 1000 is a
#   good value for the search to converge. You can do more trials according to your time budget.
# * In addition, we use :code:`RecordToFile` to dump measurement records into a file `matmul.json`.
#   The measurement records can be used to query the history best, resume the search,
#   and do more analyses later.
# * see :any:`auto_scheduler.TuningOptions` for more parameters

tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10, measure_callbacks=[auto_scheduler.RecordToFile("matmul.json")]
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
# parallelization, vectorization, unrolling and operator fusion.

print(tvm.lower(sch, args, simple_mode=True))

######################################################################
# Check correctness and evaluate performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We build the binary and check its correctness and performance.

func = tvm.build(sch, args)
a_np = np.random.uniform(size=(128, 128)).astype(np.float32)
b_np = np.random.uniform(size=(128, 128)).astype(np.float32)
c_np = np.random.uniform(size=(128, 128)).astype(np.float32)
out_np = a_np.dot(b_np) + c_np

ctx = tvm.cpu()
a_tvm = tvm.nd.array(a_np, ctx=ctx)
b_tvm = tvm.nd.array(b_np, ctx=ctx)
c_tvm = tvm.nd.array(c_np, ctx=ctx)
out_tvm = tvm.nd.empty(out_np.shape, ctx=ctx)
func(a_tvm, b_tvm, c_tvm, out_tvm)

# Check results
np.testing.assert_allclose(out_np, out_tvm.asnumpy(), rtol=1e-3)

# Evaluate execution time.
evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results) * 1000)
)


######################################################################
# Using the record file
# ^^^^^^^^^^^^^^^^^^^^^
# During the search, all measuremnt records are dumpped into the record
# file "matmul.json". The measurement records can be used to re-apply search results,
# resume the search, and perform other analyses.

######################################################################
# Here is an example where we load the best schedule from a file,
# print the equivalent python schedule API, and build the binary again.

# Load the measuremnt record for the best schedule
inp, res = auto_scheduler.load_best("matmul.json", task.workload_key)

# Print equivalent python schedule API. This can be used for debugging and
# learning the behavior of the auto-scheduler.
print("Equivalent python schedule:")
print(task.compute_dag.print_python_code_from_state(inp.state))

# Rebuild the binary. This shows how you can apply the best schedule from a
# log file without reruning the search again.
sch, args = task.compute_dag.apply_steps_from_state(inp.state)
func = tvm.build(sch, args)

######################################################################
# A more complicated example is to resume the search.
# In this case, we need to create the search policy and cost model by ourselves
# and resume the status of search policy and cost model with the log file.
# In the example below we resume the status and do more 5 trials.


def resume_search(task, log_file):
    cost_model = auto_scheduler.XGBModel()
    cost_model.update_from_file(log_file)
    search_policy = auto_scheduler.SketchPolicy(
        task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=5, measure_callbacks=[auto_scheduler.RecordToFile(log_file)]
    )
    sch, args = auto_scheduler.auto_schedule(task, search_policy, tuning_options=tune_option)


# resume_search(task, "matmul.json")

######################################################################
# .. note::
#   We cannot run the line above because of the conflict between
#   python's multiprocessing and tvm's thread pool.
#   After running a tvm generated binary the python's multiprocessing library
#   will hang forever. You have to make sure that you don't run any tvm
#   generated binaries before calling auot-scheduler's search.
#   To run the function above, you should comment out all code in
#   "Check correctness and evaluate performance" section.
#
#   You should be careful about this problem in your applications.
#   There are other workarounds for this problem.
#   For example, you can start a new thread/process (with the builtin python library
#   threading or multiprocessing) and run the tvm binaries in the new thread/process.
#   This provides an isolation and avoids the conflict in the main thread/process.
#   You can also use :any:`auto_scheduler.LocalRPCMeasureContext` for auto-scheduler,
#   as shown in the GPU tutorial (:ref:`auto-scheduler-conv-gpu`).
