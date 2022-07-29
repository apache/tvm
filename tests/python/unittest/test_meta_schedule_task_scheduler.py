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

import random
import weakref
from typing import Set

import pytest
import tvm
import tvm.testing
from tvm.support import libinfo
from tvm import meta_schedule as ms
from tvm._ffi.base import TVMError
from tvm.meta_schedule.testing.dummy_object import DummyBuilder, DummyRunner
from tvm.script import tir as T
from tvm.tir import Schedule

# from tvm.meta_schedule import TuneContext, measure_callback
# from tvm.meta_schedule.search_strategy import ReplayTrace
# from tvm.meta_schedule.space_generator import ScheduleFn
# from tvm.meta_schedule.task_scheduler import GradientBased, PyTaskScheduler, RoundRobin
# from tvm.meta_schedule.utils import derived_object

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


@tvm.script.ir_module
class MatmulModule:
    @T.prim_func
    def main(  # type: ignore
        a: T.handle,
        b: T.handle,
        c: T.handle,
    ) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0  # type: ignore
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@tvm.script.ir_module
class MatmulReluModule:
    @T.prim_func
    def main(  # type: ignore
        a: T.handle,
        b: T.handle,
        d: T.handle,
    ) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        D = T.match_buffer(d, (1024, 1024), "float32")
        C = T.alloc_buffer((1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0  # type: ignore
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(1024, 1024):
            with T.block("relu"):
                vi, vj = T.axis.remap("SS", [i, j])
                D[vi, vj] = T.max(C[vi, vj], 0.0)  # type: ignore


@tvm.script.ir_module
class BatchMatmulModule:
    @T.prim_func
    def main(  # type: ignore
        a: T.handle,
        b: T.handle,
        c: T.handle,
    ) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        A = T.match_buffer(a, [16, 128, 128])
        B = T.match_buffer(b, [16, 128, 128])
        C = T.match_buffer(c, [16, 128, 128])
        for n, i, j, k in T.grid(16, 128, 128, 128):
            with T.block("matmul"):
                vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
                with T.init():
                    C[vn, vi, vj] = 0.0  # type: ignore
                C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks


def _schedule_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=[2, 4, 64, 2])
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=[4, 64, 2, 2])
    k_0, k_1 = sch.split(loop=k, factors=[32, 32])
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


def _schedule_batch_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k, t = sch.get_loops(block=block)
    i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=[2, 2, 2, 2])
    j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=[2, 4, 64, 2])
    k_0, k_1 = sch.split(loop=k, factors=[32, 32])
    t_0, t_1 = sch.split(loop=t, factors=[2, 512])
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3, t_0, t_1)


@ms.derived_object
class MyTaskScheduler(ms.task_scheduler.PyTaskScheduler):
    done: Set = set()

    def next_task_id(self) -> int:
        while len(self.done) != len(self.tasks):
            x = random.randint(0, len(self.tasks) - 1)
            task = self.tasks[x]
            if not task.is_terminated:
                """Calling base func via following route:
                Python side:
                    PyTaskScheduler does not have `_touch_task`
                    Call TaskScheduler's `touch_task`, which calls ffi
                C++ side:
                    The ffi calls TaskScheduler's `touch_task`
                    But it is overridden in PyTaskScheduler
                    PyTaskScheduler checks if the function is overridden in python
                    If not, it returns the TaskScheduler's vtable, calling
                        TaskScheduler::TouchTask
                """
                if task.runner_futures is not None:
                    self.join_running_task(x)
                return x
            self.done.add(x)
        return -1


def test_meta_schedule_task_scheduler_single():
    num_trials_per_iter = 3
    max_trials_per_task = 10
    database = ms.database.MemoryDatabase()
    round_robin = ms.task_scheduler.RoundRobin(
        [
            ms.TuneContext(
                MatmulModule,
                target=tvm.target.Target("llvm"),
                space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul),
                search_strategy=ms.search_strategy.ReplayTrace(
                    num_trials_per_iter,
                    max_trials_per_task,
                ),
                task_name="Test",
                rand_state=42,
            )
        ],
        [1.0],
        builder=DummyBuilder(),
        runner=DummyRunner(),
        database=database,
        measure_callbacks=[ms.measure_callback.AddToDatabase()],
        max_trials=max_trials_per_task,
    )
    round_robin.tune()
    assert len(database) == max_trials_per_task


def test_meta_schedule_task_scheduler_multiple():
    num_trials_per_iter = 6
    max_trials_per_task = 101
    tasks = [
        ms.TuneContext(
            MatmulModule,
            target=tvm.target.Target("llvm"),
            space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul),
            search_strategy=ms.search_strategy.ReplayTrace(
                num_trials_per_iter,
                max_trials_per_task,
            ),
            task_name="Matmul",
            rand_state=42,
        ),
        ms.TuneContext(
            MatmulReluModule,
            target=tvm.target.Target("llvm"),
            space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul),
            search_strategy=ms.search_strategy.ReplayTrace(
                num_trials_per_iter,
                max_trials_per_task,
            ),
            task_name="MatmulRelu",
            rand_state=0xDEADBEEF,
        ),
        ms.TuneContext(
            BatchMatmulModule,
            target=tvm.target.Target("llvm"),
            space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_batch_matmul),
            search_strategy=ms.search_strategy.ReplayTrace(
                num_trials_per_iter,
                max_trials_per_task,
            ),
            task_name="BatchMatmul",
            rand_state=0x114514,
        ),
    ]
    database = ms.database.MemoryDatabase()
    round_robin = ms.task_scheduler.RoundRobin(
        tasks,
        [1.0, 1.0, 1.0],
        builder=DummyBuilder(),
        runner=DummyRunner(),
        database=database,
        measure_callbacks=[ms.measure_callback.AddToDatabase()],
        max_trials=max_trials_per_task * len(tasks),
    )
    round_robin.tune()
    assert len(database) == max_trials_per_task * len(tasks)
    for task in tasks:
        assert (
            len(
                database.get_top_k(
                    database.commit_workload(task.mod),
                    100000,
                )
            )
            == max_trials_per_task
        )


def test_meta_schedule_task_scheduler_NIE():  # pylint: disable=invalid-name
    @ms.derived_object
    class NIETaskScheduler(ms.task_scheduler.PyTaskScheduler):
        pass

    with pytest.raises(TVMError, match="PyTaskScheduler's NextTaskId method not implemented!"):
        scheduler = NIETaskScheduler(
            tasks=[],
            builder=DummyBuilder(),
            runner=DummyRunner(),
            database=ms.database.MemoryDatabase(),
            max_trials=1,
        )
        scheduler.next_task_id()


def test_meta_schedule_task_scheduler_avoid_cyclic():  # pylint: disable=invalid-name
    database = ms.database.MemoryDatabase()
    scheduler = MyTaskScheduler(
        [],
        builder=DummyBuilder(),
        runner=DummyRunner(),
        database=database,
        measure_callbacks=[
            ms.measure_callback.AddToDatabase(),
        ],
        max_trials=10,
    )
    test = weakref.ref(scheduler)  # test if it can be destructed successfully
    del scheduler
    assert test() is None


def test_meta_schedule_task_scheduler_override_next_task_id_only():  # pylint: disable=invalid-name
    num_trials_per_iter = 6
    max_trials_per_task = 101
    tasks = [
        ms.TuneContext(
            MatmulModule,
            target=tvm.target.Target("llvm"),
            space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul),
            search_strategy=ms.search_strategy.ReplayTrace(
                num_trials_per_iter,
                max_trials_per_task,
            ),
            task_name="Matmul",
            rand_state=42,
        ),
        ms.TuneContext(
            MatmulReluModule,
            target=tvm.target.Target("llvm"),
            space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul),
            search_strategy=ms.search_strategy.ReplayTrace(
                num_trials_per_iter,
                max_trials_per_task,
            ),
            task_name="MatmulRelu",
            rand_state=0xDEADBEEF,
        ),
        ms.TuneContext(
            BatchMatmulModule,
            target=tvm.target.Target("llvm"),
            space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_batch_matmul),
            search_strategy=ms.search_strategy.ReplayTrace(
                num_trials_per_iter,
                max_trials_per_task,
            ),
            task_name="BatchMatmul",
            rand_state=0x114514,
        ),
    ]
    database = ms.database.MemoryDatabase()
    scheduler = MyTaskScheduler(
        tasks,
        builder=DummyBuilder(),
        runner=DummyRunner(),
        database=database,
        measure_callbacks=[ms.measure_callback.AddToDatabase()],
        max_trials=max_trials_per_task * len(tasks),
    )
    scheduler.tune()
    assert len(database) == max_trials_per_task * len(tasks)
    for task in tasks:
        assert (
            len(
                database.get_top_k(
                    database.commit_workload(task.mod),
                    100000,
                )
            )
            == max_trials_per_task
        )


def test_meta_schedule_task_scheduler_multiple_gradient_based():
    num_trials_per_iter = 6
    max_trials_per_task = 101
    tasks = [
        ms.TuneContext(
            MatmulModule,
            target=tvm.target.Target("llvm"),
            space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul),
            search_strategy=ms.search_strategy.ReplayTrace(
                num_trials_per_iter,
                max_trials_per_task,
            ),
            task_name="Matmul",
            rand_state=42,
        ),
        ms.TuneContext(
            MatmulReluModule,
            target=tvm.target.Target("llvm"),
            space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul),
            search_strategy=ms.search_strategy.ReplayTrace(
                num_trials_per_iter,
                max_trials_per_task,
            ),
            task_name="MatmulRelu",
            rand_state=0xDEADBEEF,
        ),
        ms.TuneContext(
            BatchMatmulModule,
            target=tvm.target.Target("llvm"),
            space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_batch_matmul),
            search_strategy=ms.search_strategy.ReplayTrace(
                num_trials_per_iter,
                max_trials_per_task,
            ),
            task_name="BatchMatmul",
            rand_state=0x114514,
        ),
    ]
    database = ms.database.MemoryDatabase()
    gradient_based = ms.task_scheduler.GradientBased(
        tasks,
        task_weights=[1.0, 1.0, 1.0],
        builder=DummyBuilder(),
        runner=DummyRunner(),
        database=database,
        measure_callbacks=[ms.measure_callback.AddToDatabase()],
        seed=0x20220214,
        max_trials=max_trials_per_task * len(tasks),
    )
    gradient_based.tune()
    assert len(database) == max_trials_per_task * len(tasks)
    for task in tasks:
        assert (
            len(database.get_top_k(database.commit_workload(task.mod), 10000))
            == max_trials_per_task
        )


if __name__ == "__main__":
    tvm.testing.main()
