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
Optimizing Operators with Auto-scheduling
=========================================
**Author**: `Lianmin Zheng <https://github.com/merrymercy>`_, \
            `Chengfan Jia <https://github.com/jcf94/>`_

In this tutorial, we will show how TVM's Auto Scheduling feature can find
optimal schedules without the need for writing a custom template.

Different from the template-based :doc:`AutoTVM <autotvm_matmul_x86>` which relies on
manual templates to define the search space, the auto-scheduler does not
require any templates.  Users only need to write the computation declaration
without any schedule commands or templates.  The auto-scheduler can
automatically generate a large search space and find a good schedule in the
space.

We use matrix multiplication as an example in this tutorial.

.. note::
  Note that this tutorial will not run on Windows or recent versions of macOS. To
  get it to run, you will need to wrap the body of this tutorial in a :code:`if
  __name__ == "__main__":` block.
"""


import numpy as np
import tvm
from tvm import te, auto_scheduler

################################################################################
# Defining the Matrix Multiplication
# ----------------------------------
# To start, we define a matrix multiplication with a bias addition.  Note that
# this uses standard operations available in TVMs Tensor Expression language.
# The major difference is the use of the :any:`register_workload` decorator at the top
# of the function definition.  The function should return a list of
# input/output tensors.  From these tensors, the auto-scheduler can get the
# whole computational graph.


@auto_scheduler.register_workload  # Note the auto_scheduler decorator
def matmul_add(N, L, M, dtype):
    A = te.placeholder((N, L), name="A", dtype=dtype)
    B = te.placeholder((L, M), name="B", dtype=dtype)
    C = te.placeholder((N, M), name="C", dtype=dtype)

    k = te.reduce_axis((0, L), name="k")
    matmul = te.compute(
        (N, M),
        lambda i, j: te.sum(A[i, k] * B[k, j], axis=k),
        name="matmul",
        attrs={"layout_free_placeholders": [B]},  # enable automatic layout transform for tensor B
    )
    out = te.compute((N, M), lambda i, j: matmul[i, j] + C[i, j], name="out")

    return [A, B, C, out]


################################################################################
# Create the search task
# ----------------------
# With the function defined, we can now create the task for the auto_scheduler
# to search against. We specify the particular parameters for this matrix
# multiplication, in this case a multiplication of two square matrices of size
# 1024x1024. We then create a search task with N=L=M=1024 and dtype="float32"
#
# .. admonition:: Improve performance with custom targets
#
#   In order for TVM to take full advantage of specific hardware platforms,
#   you will want to manually specify your CPU capabilities. For example:
#
#     - replace ``llvm`` below with ``llvm -mcpu=core-avx2`` to enable AVX2
#     - replace ``llvm`` below with ``llvm -mcpu=skylake-avx512`` to enable AVX-512

target = tvm.target.Target("llvm")
N = L = M = 1024
task = tvm.auto_scheduler.SearchTask(func=matmul_add, args=(N, L, M, "float32"), target=target)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

################################################################################
# Set Parameters for Auto-Scheduler
# ---------------------------------
# Next, we set parameters for the auto-scheduler.
#
# * :code:`num_measure_trials` is the number of measurement trials we can use
#   during the search.  We only make 10 trials in this tutorial for a fast
#   demonstration. In practice, 1000 is a good value for the search to converge.
#   You can do more trials according to your time budget.
# * In addition, we use :any:`RecordToFile <auto_scheduler.RecordToFile>` to log measurement records into a
#   file ``matmul.json``.  The measurement records can be used to query the history
#   best, resume the search, and do more analyses later.
# * see :any:`TuningOptions <auto_scheduler.TuningOptions>` for more parameters

log_file = "matmul.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

################################################################################
# Run the search
# --------------
# Now we get all inputs ready. Pretty simple, isn't it?  We can kick off the
# search and let the auto-scheduler do its magic.  After some measurement
# trials, we can load the best schedule from the log file and apply it.

# Run auto-tuning (search)
task.tune(tune_option)
# Apply the best schedule
sch, args = task.apply_best(log_file)

################################################################################
# Inspecting the Optimized Schedule
# ---------------------------------
# We can lower the schedule to see the IR after auto-scheduling.  The
# auto-scheduler correctly performs optimizations including multi-level tiling,
# layout transformation, parallelization, vectorization, unrolling, and
# operator fusion.

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))

################################################################################
# Check correctness and evaluate performance
# ------------------------------------------
# We build the binary and check its correctness and performance.

func = tvm.build(sch, args, target)
a_np = np.random.uniform(size=(N, L)).astype(np.float32)
b_np = np.random.uniform(size=(L, M)).astype(np.float32)
c_np = np.random.uniform(size=(N, M)).astype(np.float32)
out_np = a_np.dot(b_np) + c_np

dev = tvm.cpu()
a_tvm = tvm.nd.array(a_np, device=dev)
b_tvm = tvm.nd.array(b_np, device=dev)
c_tvm = tvm.nd.array(c_np, device=dev)
out_tvm = tvm.nd.empty(out_np.shape, device=dev)
func(a_tvm, b_tvm, c_tvm, out_tvm)

# Check results
np.testing.assert_allclose(out_np, out_tvm.numpy(), rtol=1e-3)

# Evaluate execution time.
evaluator = func.time_evaluator(func.entry_name, dev, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(a_tvm, b_tvm, c_tvm, out_tvm).results) * 1000)
)


################################################################################
# Using the record file
# ---------------------
# During the search, all measurement records are logged into the record file
# ``matmul.json```. The measurement records can be used to re-apply search
# results, resume the search, and perform other analyses.
#
# Here is an example where we load the best schedule from a file, and print the
# equivalent python schedule API. This can be used for debugging and learning
# the behavior of the auto-scheduler.

print("Equivalent python schedule:")
print(task.print_best(log_file))

################################################################################
# A more complicated example is to resume the search.  In this case, we need to
# create the search policy and cost model by ourselves and resume the status of
# search policy and cost model with the log file.  In the example below we
# resume the status and do more 5 trials.


def resume_search(task, log_file):
    print("Resume search:")
    cost_model = auto_scheduler.XGBModel()
    cost_model.update_from_file(log_file)
    search_policy = auto_scheduler.SketchPolicy(
        task, cost_model, init_search_callbacks=[auto_scheduler.PreloadMeasuredStates(log_file)]
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=5, measure_callbacks=[auto_scheduler.RecordToFile(log_file)]
    )
    task.tune(tune_option, search_policy=search_policy)


resume_search(task, log_file)

################################################################################
# Final Notes and Summary
# -----------------------
# In this tutorial, we have shown how to use the TVM Auto-Scheduler to
# automatically optimize a matrix multiplication, without the need to specify a
# search template.  It ends a series of examples that starts from the Tensor
# Expression (TE) language that demonstrates how TVM can optimize computational
# operations.
