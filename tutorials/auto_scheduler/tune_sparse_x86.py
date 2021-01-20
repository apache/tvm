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
Auto-scheduling Matrix Multiplication for CPU
=============================================
**Author**: `Chengfan Jia <https://github.com/jcf94/>`_

This is a tutorial on how to use the auto-scheduler for CPUs.

Different from the template-based :ref:`autotvm <tutorials-autotvm-sec>` which relies on
manual templates to define the search space, the auto-scheduler does not require any templates.
Users only need to write the computation declaration without any schedule commands or templates.
The auto-scheduler can automatically generate a large search space and
find a good schedule in the space.

We use matrix multiplication as an example in this tutorial.

Note that this tutorial will not run on Windows or recent versions of macOS. To
get it to run, you will need to wrap the body of this tutorial in a :code:`if
__name__ == "__main__":` block.
"""

import os
import itertools

import numpy as np
import tvm
from tvm import te, auto_scheduler, topi
from tvm.topi.utils import get_const_tuple

import scipy.sparse as sp

######################################################################
# Define the computation
# ^^^^^^^^^^^^^^^^^^^^^^
# To begin with, let us define the computation of a matmul with bias add.
# The function should return the list of input/output tensors.
# From these tensors, the auto-scheduler can get the whole computational graph.


def random_bsr_matrix(M, N, BS_R, BS_C, density, dtype):
    import itertools

    Y = np.zeros((M, N), dtype=dtype)
    assert M % BS_R == 0
    assert N % BS_C == 0
    nnz = int(density * M * N)
    num_blocks = int(nnz / (BS_R * BS_C)) + 1
    candidate_blocks = np.asarray(list(itertools.product(range(0, M, BS_R), range(0, N, BS_C))))
    assert candidate_blocks.shape[0] == M // BS_R * N // BS_C
    chosen_blocks = candidate_blocks[
        np.random.choice(candidate_blocks.shape[0], size=num_blocks, replace=False)
    ]
    for i in range(len(chosen_blocks)):
        r, c = chosen_blocks[i]
        Y[r : r + BS_R, c : c + BS_C] = np.random.randn(BS_R, BS_C)
    s = sp.bsr_matrix(Y, blocksize=(BS_R, BS_C))
    assert s.data.shape == (num_blocks, BS_R, BS_C)
    assert s.indices.shape == (num_blocks,)
    assert s.indptr.shape == (M // BS_R + 1,)
    return s


######################################################################
# Create the search task
# ^^^^^^^^^^^^^^^^^^^^^^
# We then create a search task with N=L=M=1024 and dtype="float32"
# If your machine supports avx instructions, you can
#
#   - replace "llvm" below with "llvm -mcpu=core-avx2" to enable AVX2
#   - replace "llvm" below with "llvm -mcpu=skylake-avx512" to enable AVX-512

target = tvm.target.Target("llvm -mcpu=core-avx2")

M = K = N = 512
BS_R = 16
BS_C = 1
density = 0.6

X_np = np.random.randn(M, K).astype("float32")
X_np = np.maximum(np.zeros((M, K), dtype="float32"), X_np)  # Relu
W_sp_np = random_bsr_matrix(N, K, BS_R, BS_C, density=density, dtype="float32")
W_np = W_sp_np.todense()
Y_np = X_np @ W_np.T

prefix = "sparse_dense_bsr_%d_%d_%d_%d_%d_%.2f_" % (M, N, K, BS_R, BS_C, density)
auto_scheduler.measure.register_special_buffer(prefix + "W_data", W_sp_np.data)
auto_scheduler.measure.register_special_buffer(prefix + "W_indices", W_sp_np.indices)
auto_scheduler.measure.register_special_buffer(prefix + "W_indptr", W_sp_np.indptr)

@auto_scheduler.register_workload
def sparse_dense(dense_shape, w_data_shape, w_indices_shape, w_indptr_shape, dtype):
    X = te.placeholder(shape=dense_shape, dtype=dtype)
    W_data = te.placeholder(shape=w_data_shape, dtype=dtype)
    W_indices = te.placeholder(shape=w_indices_shape, dtype="int32")
    W_indptr = te.placeholder(shape=w_indptr_shape, dtype="int32")

    out = topi.nn.sparse_dense(topi.nn.relu(X), W_data, W_indices, W_indptr)

    return [X, W_data, W_indices, W_indptr, out]

task = tvm.auto_scheduler.SearchTask(
    func=sparse_dense,
    args=(
        X_np.shape,
        W_sp_np.data.shape,
        W_sp_np.indices.shape,
        W_sp_np.indptr.shape,
        "float32"
    ),
    target=target
)

# Inspect the computational graph
print("Computational DAG:")
print(task.compute_dag)

######################################################################

def meet_condition_func(search_policy, state, stage_id):
    state = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if state.stages[stage_id].op.tag in [
        "sparse_dense_sp_rhs_bsrmm", "sparse_dense_sp_rhs_bsrmm_block"
    ]:
        return auto_scheduler.PreloadCustomSketchRule.APPLY_AND_SKIP_REST
    else:
        return auto_scheduler.PreloadCustomSketchRule.PASS

def apply_func(search_policy, state, stage_id):
    ret = []
    s0 = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
    if s0.stages[stage_id].op.tag == "sparse_dense_sp_rhs_bsrmm_block":
        return [s0.state_object, stage_id - 1]

    sparse_dense = s0.stages[stage_id].op
    sparse_dense_block = s0.stages[stage_id - 1].op
    assert sparse_dense.tag == "sparse_dense_sp_rhs_bsrmm"
    assert sparse_dense_block.tag == "sparse_dense_sp_rhs_bsrmm_block"

    s1 = s0.copy()
    i, nb_j, j, row_offset, c = s0[sparse_dense_block].iters
    m, n = s0[sparse_dense].iters
    i0, i1, i2 = s0.split(sparse_dense_block, i, [None, None])
    m0, m1 = s0.follow_split(sparse_dense, m, len(s0.transform_steps) - 1, 1)
    j0, j1 = s0.split(sparse_dense_block, nb_j, [None])
    n0, n1 = s0.follow_split(sparse_dense, n, len(s0.transform_steps) - 1, 1)
    s0.reorder(sparse_dense_block, [i0, j0, i1, j1, row_offset, i2, j, c])
    s0.reorder(sparse_dense, [m0, n0, m1, n1])
    s0.compute_at(sparse_dense_block, sparse_dense, n0)

    ret.append([s0.state_object, stage_id - 2])

    return ret

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

log_file = "sparse_dense.json"
tune_option = auto_scheduler.TuningOptions(
    num_measure_trials=10,
    measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    verbose=2,
)

search_policy = auto_scheduler.SketchPolicy(
    task,
    program_cost_model=auto_scheduler.XGBModel(),
    init_search_callbacks=[
        auto_scheduler.PreloadCustomSketchRule(meet_condition_func, apply_func, "SparseDense")
    ]
)

######################################################################
# Run the search
# ^^^^^^^^^^^^^^
# Now we get all inputs ready. Pretty simple, isn't it?
# We can kick off the search and let the auto-scheduler do its magic.
# After some measurement trials, we can load the best schedule from the log
# file and apply it.

# Run auto-tuning (search)
task.tune(tune_option, search_policy)
# Apply the best schedule
sch, args = task.apply_best(log_file)

# args = sparse_dense(
#         X_np.shape,
#         W_sp_np.data.shape,
#         W_sp_np.indices.shape,
#         W_sp_np.indptr.shape,
#         "float32")

# sch = tvm.te.create_schedule([arg.op for arg in args])

######################################################################
# We can lower the schedule to see the IR after auto-scheduling.
# The auto-scheduler correctly performs optimizations including multi-level tiling,
# layout transformation, parallelization, vectorization, unrolling, and operator fusion.

print("Lowered TIR:")
print(tvm.lower(sch, args, simple_mode=True))

######################################################################
# Check correctness and evaluate performance
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We build the binary and check its correctness and performance.

func = tvm.build(sch, args, target)

ctx = tvm.cpu()

X_tvm = tvm.nd.array(X_np, ctx=ctx)
W_data_tvm = tvm.nd.array(W_sp_np.data, ctx=ctx)
W_indices_tvm = tvm.nd.array(W_sp_np.indices, ctx=ctx)
W_indptr_tvm = tvm.nd.array(W_sp_np.indptr, ctx=ctx)
Y_tvm = tvm.nd.empty(Y_np.shape, ctx=ctx)

func(X_tvm, W_data_tvm, W_indices_tvm, W_indptr_tvm, Y_tvm)

# Check results
tvm.testing.assert_allclose(Y_np, Y_tvm.asnumpy(), atol=1e-4, rtol=1e-4)

# Evaluate execution time.
evaluator = func.time_evaluator(func.entry_name, ctx, min_repeat_ms=500)
print(
    "Execution time of this operator: %.3f ms"
    % (np.median(evaluator(X_tvm, W_data_tvm, W_indices_tvm, W_indptr_tvm, Y_tvm).results) * 1000)
)


######################################################################
# Using the record file
# ^^^^^^^^^^^^^^^^^^^^^
# During the search, all measurement records are dumped into the record
# file "matmul.json". The measurement records can be used to re-apply search results,
# resume the search, and perform other analyses.

######################################################################
# Here is an example where we load the best schedule from a file,
# and print the equivalent python schedule API. This can be used for
# debugging and learning the behavior of the auto-scheduler.

print("Equivalent python schedule:")
print(task.print_best(log_file))

######################################################################
# A more complicated example is to resume the search.
# In this case, we need to create the search policy and cost model by ourselves
# and resume the status of search policy and cost model with the log file.
# In the example below we resume the status and do more 5 trials.


def resume_search(task, log_file):
    print("Resume search:")
    cost_model = auto_scheduler.XGBModel()
    cost_model.update_from_file(log_file)
    search_policy = auto_scheduler.SketchPolicy(
        task, cost_model, init_search_callbacks=[
            auto_scheduler.PreloadMeasuredStates(log_file),
            auto_scheduler.PreloadCustomSketchRule(meet_condition_func, apply_func, "SparseDense")
        ]
    )
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=5, measure_callbacks=[auto_scheduler.RecordToFile(log_file)]
    )
    task.tune(tune_option, search_policy=search_policy)


resume_search(task, log_file)
