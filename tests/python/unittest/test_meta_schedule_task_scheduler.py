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
""" Test Meta Schedule Task Scheduler """

from typing import List

import sys
import random

import pytest

import tvm
from tvm.script import tir as T
from tvm.ir import IRModule
from tvm.tir import Schedule
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.space_generator import ScheduleFn
from tvm.meta_schedule.search_strategy import ReplayTrace
from tvm.meta_schedule.builder import PyBuilder, BuilderInput, BuilderResult
from tvm.meta_schedule.runner import PyRunner, RunnerInput, RunnerFuture, RunnerResult
from tvm.meta_schedule.database import PyDatabase, TuningRecord, Workload
from tvm.meta_schedule.task_scheduler import RoundRobin, PyTaskScheduler


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class MatmulReluModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, d: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        D = T.match_buffer(d, (1024, 1024), "float32")
        C = T.alloc_buffer((1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(1024, 1024):
            with T.block("relu"):
                vi, vj = T.axis.remap("SS", [i, j])
                D[vi, vj] = T.max(C[vi, vj], 0.0)


@tvm.script.ir_module
class BatchMatmulModule:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [16, 128, 128])
        B = T.match_buffer(b, [16, 128, 128])
        C = T.match_buffer(c, [16, 128, 128])
        for n, i, j, k in T.grid(16, 128, 128, 128):
            with T.block("matmul"):
                vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
                with T.init():
                    C[vn, vi, vj] = 0.0
                C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks


def _schedule_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    # TODO(@zxybazh): Change to `sample_perfect_tile` after upstreaming
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=[2, 4, 64, 2])
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=[4, 64, 2, 2])
    k_0, k_1 = sch.split(loop=k, factors=[32, 32])
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def _schedule_batch_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k, t = sch.get_loops(block=block)
    # TODO(@zxybazh): Change to `sample_perfect_tile` after upstreaming
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=[2, 2, 2, 2])
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=[2, 4, 64, 2])
    k_0, k_1 = sch.split(loop=k, factors=[32, 32])
    t_0, t_1 = sch.split(loop=t, factors=[2, 512])
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3, t_0, t_1)


class DummyRunnerFuture(RunnerFuture):
    def done(self) -> bool:
        return True

    def result(self) -> RunnerResult:
        return RunnerResult([random.uniform(5, 30) for _ in range(random.randint(1, 10))], None)


class DummyBuilder(PyBuilder):
    def build(self, build_inputs: List[BuilderInput]) -> List[BuilderResult]:
        return [BuilderResult("test_path", None) for _ in build_inputs]


class DummyRunner(PyRunner):
    def run(self, runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
        return [DummyRunnerFuture() for _ in runner_inputs]


class DummyDatabase(PyDatabase):
    def __init__(self):
        super().__init__()
        self.records = []
        self.workload_reg = []

    def commit_tuning_record(self, record: TuningRecord) -> None:
        self.records.append(record)

    def commit_workload(self, mod: IRModule) -> Workload:
        for workload in self.workload_reg:
            if tvm.ir.structural_equal(workload.mod, mod):
                return workload
        workload = Workload(mod)
        self.workload_reg.append(workload)
        return workload

    def get_top_k(self, workload: Workload, top_k: int) -> List[TuningRecord]:
        return list(
            filter(
                lambda x: x.workload == workload,
                sorted(self.records, key=lambda x: sum(x.run_secs) / len(x.run_secs)),
            )
        )[: int(top_k)]

    def __len__(self) -> int:
        return len(self.records)

    def print_results(self) -> None:
        print("\n".join([str(r) for r in self.records]))


def test_meta_schedule_task_scheduler_single():
    num_trials_per_iter = 3
    num_trials_total = 10
    sch_fn = ScheduleFn(sch_fn=_schedule_matmul)
    replay = ReplayTrace(num_trials_per_iter, num_trials_total)
    task = TuneContext(
        MatmulModule,
        target=tvm.target.Target("llvm"),
        space_generator=sch_fn,
        search_strategy=replay,
        task_name="Test",
        rand_state=42,
    )
    database = DummyDatabase()
    round_robin = RoundRobin([task], DummyBuilder(), DummyRunner(), database)
    round_robin.tune()
    assert len(database) == num_trials_total


def test_meta_schedule_task_scheduler_multiple():
    num_trials_per_iter = 6
    num_trials_total = 101
    tasks = [
        TuneContext(
            MatmulModule,
            target=tvm.target.Target("llvm"),
            space_generator=ScheduleFn(sch_fn=_schedule_matmul),
            search_strategy=ReplayTrace(num_trials_per_iter, num_trials_total),
            task_name="Matmul",
            rand_state=42,
        ),
        TuneContext(
            MatmulReluModule,
            target=tvm.target.Target("llvm"),
            space_generator=ScheduleFn(sch_fn=_schedule_matmul),
            search_strategy=ReplayTrace(num_trials_per_iter, num_trials_total),
            task_name="MatmulRelu",
            rand_state=0xDEADBEEF,
        ),
        TuneContext(
            BatchMatmulModule,
            target=tvm.target.Target("llvm"),
            space_generator=ScheduleFn(sch_fn=_schedule_batch_matmul),
            search_strategy=ReplayTrace(num_trials_per_iter, num_trials_total),
            task_name="BatchMatmul",
            rand_state=0x114514,
        ),
    ]
    database = DummyDatabase()
    round_robin = RoundRobin(tasks, DummyBuilder(), DummyRunner(), database)
    round_robin.tune()
    assert len(database) == num_trials_total * len(tasks)
    print(database.workload_reg)
    for task in tasks:
        assert len(database.get_top_k(database.commit_workload(task.mod), 1e9)) == num_trials_total


def test_meta_schedule_task_scheduler_NIE():
    class MyTaskScheduler(PyTaskScheduler):
        pass

    with pytest.raises(NotImplementedError):
        MyTaskScheduler([], DummyBuilder(), DummyRunner(), DummyDatabase())


def test_meta_schedule_task_scheduler_override_next_task_id_only():
    class MyTaskScheduler(PyTaskScheduler):
        done = set()

        def next_task_id(self) -> int:
            while len(self.done) != len(tasks):
                x = random.randint(0, len(tasks) - 1)
                task = tasks[x]
                if not task.is_stopped:
                    """Calling base func via following route:
                    Python side:
                        PyTaskScheduler does not have `_is_task_running`
                        Call TaskScheduler's `is_task_running`, which calls ffi
                    C++ side:
                        The ffi calls TaskScheduler's `is_task_running`
                        But it is overrided in PyTaskScheduler
                        PyTaskScheduler checks if the function is overrided in python
                        If not, it returns the TaskScheduler's vtable, calling
                            TaskScheduler::IsTaskRunning
                    """
                    if self._is_task_running(x):
                        # Same Here
                        self._join_running_task(x)
                    return x
                else:
                    self.done.add(x)
            return -1

    num_trials_per_iter = 6
    num_trials_total = 101
    tasks = [
        TuneContext(
            MatmulModule,
            target=tvm.target.Target("llvm"),
            space_generator=ScheduleFn(sch_fn=_schedule_matmul),
            search_strategy=ReplayTrace(num_trials_per_iter, num_trials_total),
            task_name="Matmul",
            rand_state=42,
        ),
        TuneContext(
            MatmulReluModule,
            target=tvm.target.Target("llvm"),
            space_generator=ScheduleFn(sch_fn=_schedule_matmul),
            search_strategy=ReplayTrace(num_trials_per_iter, num_trials_total),
            task_name="MatmulRelu",
            rand_state=0xDEADBEEF,
        ),
        TuneContext(
            BatchMatmulModule,
            target=tvm.target.Target("llvm"),
            space_generator=ScheduleFn(sch_fn=_schedule_batch_matmul),
            search_strategy=ReplayTrace(num_trials_per_iter, num_trials_total),
            task_name="BatchMatmul",
            rand_state=0x114514,
        ),
    ]
    database = DummyDatabase()
    scheduler = MyTaskScheduler(tasks, DummyBuilder(), DummyRunner(), database)
    scheduler.tune()
    assert len(database) == num_trials_total * len(tasks)
    for task in tasks:
        assert len(database.get_top_k(database.commit_workload(task.mod), 1e9)) == num_trials_total


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
