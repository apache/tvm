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

"""Test search policy"""

import random
import multiprocessing
import numpy as np
import tempfile

import tvm
import tvm.testing
from tvm import auto_scheduler
from tvm.auto_scheduler.utils import get_const_tuple

from tvm.testing.auto_scheduler import (
    matmul_auto_scheduler_test,
    zero_rank_compute_auto_scheduler_test,
    zero_rank_reduce_auto_scheduler_test,
)
import multiprocessing


class CustomMeasureCallback(auto_scheduler.measure.PythonBasedMeasureCallback):
    """A simple Python-based callback for testing."""

    def callback(self, policy, inputs, results):
        assert isinstance(policy, auto_scheduler.search_policy.SearchPolicy)
        for inp, res in zip(inputs, results):
            assert isinstance(inp, auto_scheduler.MeasureInput)
            assert isinstance(res, auto_scheduler.MeasureResult)


def search_common(
    task=None,
    target="llvm",
    search_policy="sketch",
    runner="local",
    num_measure_trials=100,
    cost_model=auto_scheduler.RandomModel(),
    init_search_callbacks=None,
):
    if task is None:
        task = auto_scheduler.SearchTask(
            func=matmul_auto_scheduler_test, args=(64, 64, 64), target=target
        )
    target = task.target

    print("Test search policy '%s' for '%s'" % (search_policy, target))

    with tempfile.NamedTemporaryFile() as fp:
        log_file = fp.name

        init_search_callbacks = init_search_callbacks or []
        init_search_callbacks.append(auto_scheduler.PreloadMeasuredStates(log_file))

        if search_policy == "empty":
            search_policy = auto_scheduler.EmptyPolicy(task)
        elif search_policy == "sketch":
            search_policy = auto_scheduler.SketchPolicy(
                task, program_cost_model=cost_model, init_search_callbacks=init_search_callbacks
            )
        else:
            raise ValueError("Invalid policy: " + search_policy)

        # Tune
        tuning_options = auto_scheduler.TuningOptions(
            num_measure_trials=num_measure_trials,
            num_measures_per_round=2,
            early_stopping=1,
            runner=runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file), CustomMeasureCallback()],
        )
        task.tune(tuning_options=tuning_options, search_policy=search_policy)

        # Compile with the best schedule
        sch, args = task.apply_best(log_file)
        mod = tvm.build(sch, args, target)

        # Compile with naive schedule for correctness check
        sch, args = task.compute_dag.apply_steps_from_state(task.compute_dag.init_state)
        mod_ref = tvm.build(sch, args, "llvm")

        ctx = tvm.device(str(target), 0)
        np_arrays = [np.random.uniform(size=get_const_tuple(x.shape)).astype(x.dtype) for x in args]

        tvm_arrays = [tvm.nd.array(x, ctx) for x in np_arrays]
        mod(*tvm_arrays)
        actual = [x.numpy() for x in tvm_arrays]

        tvm_arrays = [tvm.nd.array(x) for x in np_arrays]
        mod_ref(*tvm_arrays)
        expected = [x.numpy() for x in tvm_arrays]

        for x, y in zip(actual, expected):
            tvm.testing.assert_allclose(x, y, rtol=1e-5)


@tvm.testing.requires_llvm
def test_workload_registry_empty_policy():
    search_common(search_policy="empty", num_measure_trials=2)

    N = 64
    target = "llvm"
    search_common(
        task=auto_scheduler.SearchTask(
            func="matmul_auto_scheduler_test", args=(N, N, N), target=target
        ),
        num_measure_trials=2,
        search_policy="empty",
    )
    search_common(
        task=auto_scheduler.SearchTask(
            func="matmul_auto_scheduler_test_rename_1", args=(N, N, N), target=target
        ),
        num_measure_trials=2,
        search_policy="empty",
    )


@tvm.testing.requires_llvm
def test_sketch_search_policy_basic():
    search_common()


def sketch_search_policy_basic_spawn():
    assert multiprocessing.get_start_method(False) == "spawn"
    test_sketch_search_policy_basic()


@tvm.testing.requires_llvm
def test_sketch_search_policy_basic_spawn():
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=sketch_search_policy_basic_spawn)
    p.start()
    p.join()


@tvm.testing.requires_llvm
def test_sketch_search_policy_xgbmodel():
    search_common(cost_model=auto_scheduler.XGBModel())


@tvm.testing.requires_cuda
def test_sketch_search_policy_cuda_rpc_runner():
    measure_ctx = auto_scheduler.LocalRPCMeasureContext()
    search_common(target="cuda", runner=measure_ctx.runner)


@tvm.testing.requires_cuda
def test_sketch_search_policy_cuda_xgbmodel_rpc_runner():
    measure_ctx = auto_scheduler.LocalRPCMeasureContext()
    search_common(target="cuda", runner=measure_ctx.runner, cost_model=auto_scheduler.XGBModel())


@tvm.testing.requires_llvm
@tvm.testing.requires_cuda
def test_sketch_search_policy_zero_rank():
    measure_ctx = auto_scheduler.LocalRPCMeasureContext()
    for target in ["llvm", "cuda"]:
        task = auto_scheduler.SearchTask(
            func=zero_rank_compute_auto_scheduler_test, args=(10,), target=target
        )
        search_common(task, runner=measure_ctx.runner)

        task = auto_scheduler.SearchTask(
            func=zero_rank_reduce_auto_scheduler_test, args=(10,), target=target
        )
        search_common(task, runner=measure_ctx.runner)


@tvm.testing.requires_llvm
def test_sketch_search_policy_custom_sketch():
    def meet_condition_func(search_policy, state, stage_id):
        return auto_scheduler.PreloadCustomSketchRule.APPLY_AND_SKIP_REST

    def apply_func(search_policy, state, stage_id):
        ret = []
        state = auto_scheduler.loop_state.State(state, search_policy.search_task.compute_dag)
        C = state.stage_ops[2]

        ret.append([state.state_object, -1])

        s1 = state.copy()
        i, _, _ = s1[C].iters
        s1.split(C, i, [8])
        ret.append([s1.state_object, -1])
        return ret

    search_common(
        cost_model=auto_scheduler.XGBModel(),
        init_search_callbacks=[
            auto_scheduler.PreloadCustomSketchRule(meet_condition_func, apply_func)
        ],
    )


if __name__ == "__main__":
    test_workload_registry_empty_policy()
    test_sketch_search_policy_basic()
    test_sketch_search_policy_basic_spawn()
    test_sketch_search_policy_xgbmodel()
    test_sketch_search_policy_cuda_rpc_runner()
    test_sketch_search_policy_cuda_xgbmodel_rpc_runner()
    test_sketch_search_policy_zero_rank()
    test_sketch_search_policy_custom_sketch()
