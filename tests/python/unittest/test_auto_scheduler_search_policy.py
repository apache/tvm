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
from tvm.testing import PropagatingThread
from tvm import auto_scheduler

from test_auto_scheduler_common import matmul_auto_scheduler_test
import multiprocessing


class CustomMeasureCallback(auto_scheduler.measure.PythonBasedMeasureCallback):
    """A simple Python-based callback for testing."""

    def callback(self, policy, inputs, results):
        assert isinstance(policy, auto_scheduler.search_policy.SearchPolicy)
        for inp, res in zip(inputs, results):
            assert isinstance(inp, auto_scheduler.MeasureInput)
            assert isinstance(res, auto_scheduler.MeasureResult)


def search_common(
    workload=matmul_auto_scheduler_test,
    target="llvm",
    search_policy="sketch",
    seed=0,
    runner="local",
    num_measure_trials=100,
    cost_model=auto_scheduler.RandomModel(),
    init_search_callbacks=None,
):
    print("Test search policy '%s' for '%s'" % (search_policy, target))

    random.seed(seed)
    N = 128
    target = tvm.target.Target(target)
    task = auto_scheduler.SearchTask(func=workload, args=(N, N, N), target=target)

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

        tuning_options = auto_scheduler.TuningOptions(
            num_measure_trials=num_measure_trials,
            num_measures_per_round=2,
            early_stopping=1,
            runner=runner,
            verbose=2,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file), CustomMeasureCallback()],
        )
        task.tune(tuning_options=tuning_options, search_policy=search_policy)
        sch, args = task.apply_best(log_file)

        print("==== Python Code ====")
        print(task.print_best(log_file))

        try:
            print("==== Lowered Stmt ====")
            print(tvm.lower(sch, args, simple_mode=True))
            mod = tvm.build(sch, args, target)

            ctx = tvm.context(str(target), 0)
            dtype = task.compute_dag.tensors[0].dtype
            a = tvm.nd.array(np.random.uniform(size=(N, N)).astype(dtype), ctx)
            b = tvm.nd.array(np.random.uniform(size=(N, N)).astype(dtype), ctx)
            c = tvm.nd.array(np.zeros((N, N), dtype=dtype), ctx)
            mod(a, b, c)
            tvm.testing.assert_allclose(c.asnumpy(), np.dot(a.asnumpy(), b.asnumpy()), rtol=1e-5)
            print("==== Verification passed ====")
        except Exception:
            raise Exception("Error encountered with seed: %d" % (seed))
    print()


@tvm.testing.requires_llvm
def test_workload_registry_search_basic():
    # wrap the search in a new thread to avoid the conflict
    # between python's multiprocessing and tvm's thread pool
    t = PropagatingThread(
        target=search_common, kwargs={"search_policy": "empty", "num_measure_trials": 2}
    )
    t.start()
    t.join()

    t = PropagatingThread(
        target=search_common,
        kwargs={
            "workload": "matmul_auto_scheduler_test",
            "num_measure_trials": 2,
            "search_policy": "empty",
        },
    )
    t.start()
    t.join()

    t = PropagatingThread(
        target=search_common,
        kwargs={
            "workload": "matmul_auto_scheduler_test_rename_1",
            "num_measure_trials": 2,
            "search_policy": "empty",
        },
    )
    t.start()
    t.join()


@tvm.testing.requires_llvm
def test_sketch_search_policy_basic():
    # wrap the search in a new thread to avoid the conflict
    # between python's multiprocessing and tvm's thread pool
    t = PropagatingThread(target=search_common)
    t.start()
    t.join()


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
    # wrap the search in a new thread to avoid the conflict
    # between python's multiprocessing and tvm's thread pool
    t = PropagatingThread(
        target=search_common,
        kwargs={
            "cost_model": auto_scheduler.XGBModel(),
        },
    )
    t.start()
    t.join()


@tvm.testing.requires_cuda
def test_sketch_search_policy_cuda_rpc_runner():
    measure_ctx = auto_scheduler.LocalRPCMeasureContext()
    # wrap the search in a new thread to avoid the conflict
    # between python's multiprocessing and tvm's thread pool
    t = PropagatingThread(
        target=search_common,
        kwargs={
            "target": "cuda",
            "runner": measure_ctx.runner,
        },
    )
    t.start()
    t.join()


@tvm.testing.requires_cuda
def test_sketch_search_policy_cuda_xgbmodel_rpc_runner():
    measure_ctx = auto_scheduler.LocalRPCMeasureContext()
    # wrap the search in a new thread to avoid the conflict
    # between python's multiprocessing and tvm's thread pool
    t = PropagatingThread(
        target=search_common,
        kwargs={
            "target": "cuda",
            "runner": measure_ctx.runner,
            "cost_model": auto_scheduler.XGBModel(),
        },
    )
    t.start()
    t.join()


if __name__ == "__main__":
    test_workload_registry_search_basic()
    test_sketch_search_policy_basic()
    test_sketch_search_policy_basic_spawn()
    test_sketch_search_policy_xgbmodel()
    test_sketch_search_policy_cuda_rpc_runner()
    test_sketch_search_policy_cuda_xgbmodel_rpc_runner()
