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
# ruff: noqa: E402, F401, I001

"""
.. _meta_schedule_advanced:

MetaSchedule: Advanced Auto-Tuning Guide
=========================================
MetaSchedule is TVM's search-based auto-tuning framework. It explores different TIR schedules
(loop tiling, vectorization, thread binding, etc.) and measures them on real hardware to find
the fastest implementation for each operator.

For the basic tune-and-apply workflow, see :ref:`optimize_model`. This tutorial focuses on
advanced usage: inspecting tunable tasks, selective operator tuning, database management,
cross-model reuse, and the lower-level tuning API.

.. contents:: Table of Contents
    :local:
    :depth: 1
"""

######################################################################
# Prepare a Model
# ---------------
# We start with a simple MLP model exported as a Relax IRModule, then legalize it
# so that high-level Relax operators are lowered to TIR functions that MetaSchedule can tune.

import os
import numpy as np

import tvm
from tvm import relax
from tvm.relax.frontend import nn

IS_IN_CI = os.getenv("CI", "") == "true"


class MLPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


input_shape = (1, 784)
mod, params = MLPModel().export_tvm({"forward": {"x": nn.spec.Tensor(input_shape, "float32")}})

# Legalize: lower Relax operators to TIR PrimFuncs
target = tvm.target.Target({"kind": "llvm", "num-cores": 4})
with target:
    mod = relax.get_pipeline("zero")(mod)

mod.show()

######################################################################
# Inspecting Tunable Tasks
# ------------------------
# Before tuning, it is useful to see what MetaSchedule will actually tune. The
# ``extract_tasks`` function analyzes an IRModule and returns one ``ExtractedTask`` per
# unique TIR workload. Each task has a ``task_name`` and a ``weight`` (how many times
# this workload is called in the graph — the task scheduler uses weights to allocate
# more tuning budget to frequently-called operators).

from tvm.s_tir.meta_schedule.relax_integration import extract_tasks

tasks = extract_tasks(mod, target)
for i, task in enumerate(tasks):
    print(f"Task {i}: {task.task_name}  (weight={task.weight})")

######################################################################
# This tells you exactly how many operators need tuning and their relative importance.
# Use this to decide whether to tune all operators or focus on a subset.

######################################################################
# Selective Operator Tuning
# -------------------------
# Tuning every operator can be time-consuming. ``MetaScheduleTuneIRMod`` accepts an
# ``op_names`` parameter to tune only operators whose task name contains any of the given
# strings. Operators without tuning records are left unscheduled — you can later apply
# DLight or other rule-based schedules to cover them.
#
# .. note::
#
#   ``MetaScheduleTuneIRMod`` works at the IRModule level and supports ``op_names`` filtering,
#   while ``MetaScheduleTuneTIR`` tunes all TIR functions without filtering. Choose based on
#   your needs.
#
# .. note::
#
#   To save CI time and avoid flakiness, we skip the tuning process in CI environment.
#

if not IS_IN_CI:
    WORK_DIR = "./tuning_logs"
    with target:
        tuned_mod = tvm.ir.transform.Sequential(
            [
                relax.transform.MetaScheduleTuneIRMod(
                    params={},
                    work_dir=WORK_DIR,
                    max_trials_global=300,
                    op_names=["matmul"],  # Only tune matmul-related operators
                ),
                relax.transform.MetaScheduleApplyDatabase(work_dir=WORK_DIR),
            ]
        )(mod)

    tuned_mod.show()

######################################################################
# Database Persistence and Resumption
# ------------------------------------
# When you use a fixed ``work_dir`` (instead of ``tempfile.TemporaryDirectory``), tuning
# results are persisted in two JSON files:
#
# - ``database_workload.json``: One line per unique workload (structural hash + serialized
#   IRModule).
# - ``database_tuning_record.json``: One line per tuning record (workload index + schedule
#   trace + measured run times).
#
# Both files use a newline-delimited JSON format. Records are appended incrementally as
# tuning progresses, so **interrupting and resuming is safe**. When you re-run tuning with
# the same ``work_dir``, existing records are loaded and used as warm-start seeds for the
# evolutionary search — the tuner does not skip already-seen workloads entirely, but starts
# from a better initial population, so re-runs are faster than starting from scratch.
#
# You can quickly check tuning progress from the command line:
#
# .. code-block:: bash
#
#   # Count how many tuning records have been collected
#   wc -l tuning_logs/database_tuning_record.json
#
# Once tuning is done, subsequent compilations only need ``MetaScheduleApplyDatabase``
# which reads the database and applies the best schedules — this takes seconds, not hours:
#
# .. code-block:: python
#
#   # Fast: apply previously tuned results (no search)
#   with target:
#       mod = relax.transform.MetaScheduleApplyDatabase(work_dir="./tuning_logs")(mod)
#

######################################################################
# Querying the Tuning Database
# ----------------------------
# The ``JSONDatabase`` class provides a Python API to inspect tuning results
# programmatically. This is useful for analyzing tuning quality, comparing different
# tuning runs, or debugging performance issues.

from tvm.s_tir.meta_schedule.database import JSONDatabase

if not IS_IN_CI:
    db = JSONDatabase(work_dir=WORK_DIR)
    print(f"Total tuning records: {len(db)}")

    # List all records with their best measured runtime
    records = db.get_all_tuning_records()
    for rec in records:
        if rec.run_secs:
            best = min(float(s) for s in rec.run_secs)
            print(f"  Best time: {best * 1e3:.3f} ms")

######################################################################
# You can also query the best schedule for a specific TIR function by passing its
# IRModule. For example, to query a single PrimFunc extracted from the full module:
#
# .. code-block:: python
#
#   # tir_mod: an IRModule containing a single PrimFunc named "main"
#   record = db.query_tuning_record(tir_mod, target, workload_name="main")
#   if record:
#       print(f"Best time: {min(float(s) for s in record.run_secs) * 1e3:.3f} ms")
#       # Reconstruct the optimized schedule
#       sch = db.query_schedule(tir_mod, target, workload_name="main")
#       sch.mod.show()
#

######################################################################
# Cross-Model Database Reuse
# --------------------------
# MetaSchedule identifies workloads by their structural hash. If two models contain
# operators with the same shape, dtype, and computation, they share the same hash and
# can reuse tuning records. This means a matmul ``(M=1, N=256, K=784)`` tuned for one
# model will automatically be reused by any other model with the same matmul shape.
#
# **module_equality options**:
#
# - ``"structural"`` (default): Exact structural match. Safe but strict.
# - ``"anchor-block"``: Match based on the dominant compute block, ignoring
#   surrounding context. More permissive — enables sharing across fused operators
#   that have the same core computation but different fusion boundaries.
#
# **OrderedUnionDatabase** enables a layered lookup strategy: check a local database
# first, then fall back to a shared team database:

from tvm.s_tir.meta_schedule.database import OrderedUnionDatabase
from tvm.s_tir.meta_schedule.relax_integration import tune_relax

######################################################################
#
# .. code-block:: python
#
#   local_db = JSONDatabase(work_dir="./my_tuning_logs")
#   shared_db = JSONDatabase(work_dir="/shared/tuning_db")
#   combined_db = OrderedUnionDatabase(local_db, shared_db)
#
# With this setup, ``combined_db.query_tuning_record(...)`` checks ``local_db`` first.
# Only if no match is found does it fall back to ``shared_db``. This lets a team maintain
# a shared tuning database while individuals only tune new operators locally.
#
# To make ``MetaScheduleApplyDatabase`` use the combined database during compilation,
# enter it as a context manager. The pass checks ``Database.current()`` first, and only
# falls back to ``work_dir`` when no database is in scope:
#
# .. code-block:: python
#
#   with target, combined_db:
#       mod = relax.transform.MetaScheduleApplyDatabase()(mod)
#

######################################################################
# Lower-Level API: ``tune_relax``
# --------------------------------
# The transform-based API (``MetaScheduleTuneTIR`` / ``MetaScheduleTuneIRMod``) covers
# most use cases. For advanced scenarios -- custom cost models, remote runners, or
# fine-grained control -- use the lower-level ``tune_relax`` function directly:

######################################################################
#
# .. code-block:: python
#
#   db = tune_relax(
#       mod=mod,
#       params={},
#       target=target,
#       work_dir="./tuning_logs",
#       max_trials_global=2000,
#       max_trials_per_task=500,
#       op_names=["matmul"],          # Selective tuning
#       cost_model="xgb",             # "xgb" (default), "mlp", or "random"
#       num_trials_per_iter=64,        # Batch size per search iteration
#       runner="local",                # "local" or RPCRunner for remote devices
#       module_equality="structural",  # "structural" or "anchor-block"
#   )
#
# Key parameters:
#
# - **cost_model**: ``"xgb"`` (XGBoost, default) uses gradient-boosted trees to predict
#   schedule performance, reducing the number of actual measurements needed. ``"mlp"``
#   uses a neural network-based model. ``"random"`` disables prediction (baseline).
# - **num_trials_per_iter**: How many candidates are measured in each search iteration.
#   Larger values improve hardware utilization but use more memory.
# - **runner**: Use ``"local"`` for the current machine. For cross-compilation scenarios
#   (e.g., tuning for a remote device), use ``RPCRunner``.
# - **module_equality**: Controls how workloads are matched. ``"anchor-block"`` improves
#   database hit rate across models at the cost of slightly less precise matching.

######################################################################
# Build and Run
# -------------
# Finally, we build and run the model to verify the result. If tuning was skipped
# (e.g., in CI), we compile the untuned module directly — LLVM can still generate
# valid (though unoptimized) code for CPU targets without explicit scheduling.

final_mod = tuned_mod if not IS_IN_CI else mod

ex = tvm.compile(final_mod, target)
vm = relax.VirtualMachine(ex, tvm.cpu())
data = tvm.runtime.tensor(np.random.rand(*input_shape).astype("float32"))
tvm_params = [tvm.runtime.tensor(np.random.rand(*p.shape).astype(p.dtype)) for _, p in params]
result = vm["forward"](data, *tvm_params).numpy()
print("Output shape:", result.shape)
print("Output:", result)

######################################################################
# Summary
# -------
# This tutorial covered advanced MetaSchedule usage beyond the basic tune-and-apply flow:
#
# - **Inspect tasks** with ``extract_tasks`` to understand what will be tuned and plan your
#   tuning budget: ``max_trials_global`` is shared across all tasks, so set it proportional
#   to the number of tasks (e.g., 200-500 trials per task for good results).
# - **Selective tuning** with ``op_names`` to focus on performance-critical operators and
#   skip the rest.
# - **Persist results** with a fixed ``work_dir``. Tuning is resumable — existing records
#   warm-start the search on re-run.
# - **Query the database** to analyze tuning quality and debug performance.
# - **Reuse across models** via ``OrderedUnionDatabase`` and ``module_equality="anchor-block"``
#   to amortize tuning cost across a team or model family.
# - **Lower-level API** (``tune_relax``) for custom cost models, remote runners, and
#   fine-grained control.
#
# For the basic end-to-end workflow, see :ref:`optimize_model`. For rule-based scheduling
# without search, see DLight documentation.
