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
""" Test Meta Schedule SearchStrategy """
# pylint: disable=missing-function-docstring
from typing import List

import pytest
import tvm
import tvm.testing
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.dummy_object import DummyMutator
from tvm.script import tir as T
from tvm.tir.schedule import Schedule, Trace

MATMUL_M = 32

# pylint: disable=missing-class-docstring,invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument, unbalanced-tuple-unpacking
# fmt: off

@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None: # type: ignore
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (32, 32), "float32")
        B = T.match_buffer(b, (32, 32), "float32")
        C = T.match_buffer(c, (32, 32), "float32")
        for i, j, k in T.grid(32, 32, 32):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0 # type: ignore
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# fmt: on
# pylint: enable=missing-class-docstring,invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def _is_trace_equal(sch_1: Schedule, sch_2: Schedule, remove_decisions=True) -> bool:
    if remove_decisions:
        trace_1 = Trace(sch_1.trace.insts, {})
        trace_2 = Trace(sch_2.trace.insts, {})
    else:
        trace_1 = sch_1.trace
        trace_2 = sch_2.trace
    return str(trace_1) == str(trace_2)


def _schedule_matmul(sch: Schedule):
    block = sch.get_block("matmul")
    i, j, k = sch.get_loops(block=block)
    i_0, i_1, i_2, i_3 = sch.split(i, sch.sample_perfect_tile(i, n=4))
    j_0, j_1, j_2, j_3 = sch.split(j, sch.sample_perfect_tile(j, n=4))
    k_0, k_1 = sch.split(k, sch.sample_perfect_tile(k, n=2))
    sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)


@pytest.mark.parametrize(
    "TestClass",
    [
        ms.search_strategy.ReplayFunc,
        ms.search_strategy.ReplayTrace,
    ],
)
def test_meta_schedule_replay_func(
    TestClass: ms.search_strategy.SearchStrategy,
):  # pylint: disable = invalid-name
    num_trials_per_iter = 7
    max_trials_per_task = 20

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul, postprocs=[]),
        search_strategy=TestClass(),
    )
    strategy = context.search_strategy
    spaces = context.space_generator.generate_design_space(context.mod)
    strategy.pre_tuning(
        max_trials=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        design_spaces=spaces,
    )
    (correct_sch,) = ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(
        Matmul
    )
    num_trials_each_iter: List[int] = []
    candidates = strategy.generate_measure_candidates()
    while candidates is not None:
        num_trials_each_iter.append(len(candidates))
        runner_results: List[ms.runner.RunnerResult] = []
        for candidate in candidates:
            _is_trace_equal(
                candidate.sch,
                correct_sch,
                remove_decisions=(isinstance(strategy, ms.search_strategy.ReplayTrace)),
            )
            runner_results.append(
                ms.runner.RunnerResult(
                    run_secs=[0.11, 0.41, 0.54],
                    error_msg=None,
                )
            )
        strategy.notify_runner_results(candidates, runner_results)
        candidates = strategy.generate_measure_candidates()
    strategy.post_tuning()
    assert num_trials_each_iter == [7, 7, 6]


def test_meta_schedule_evolutionary_search():  # pylint: disable = invalid-name
    def _schedule_matmul_small(sch: Schedule):
        block = sch.get_block("matmul")
        _, j, k = sch.get_loops(block=block)
        _, _ = sch.split(j, sch.sample_perfect_tile(j, n=2))
        _, _ = sch.split(k, sch.sample_perfect_tile(k, n=2))

    num_trials_per_iter = 10
    max_trials_per_task = 2000
    (correct_sch,) = ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(
        Matmul
    )

    context = ms.TuneContext(
        mod=Matmul,
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_schedule_matmul_small,
            sch_rules=[],
            postprocs=[],
            mutator_probs={
                DummyMutator(): 1.0,
            },
        ),
        search_strategy=ms.search_strategy.EvolutionarySearch(
            population_size=5,
            init_measured_ratio=0.1,
            init_min_unmeasured=50,
            genetic_num_iters=3,
            genetic_mutate_prob=0.5,
            genetic_max_fail_count=10,
            eps_greedy=0.9,
        ),
        target=tvm.target.Target("llvm"),
        num_threads=1,  # because we are using a mutator from the python side
    )
    strategy = context.search_strategy
    strategy.pre_tuning(
        max_trials=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        design_spaces=context.space_generator.generate_design_space(context.mod),
        database=ms.database.MemoryDatabase(),
        cost_model=ms.cost_model.RandomModel(),
    )
    num_trials_each_iter: List[int] = []
    candidates = strategy.generate_measure_candidates()
    while candidates is not None:
        num_trials_each_iter.append(len(candidates))
        runner_results: List[ms.runner.RunnerResult] = []
        for candidate in candidates:
            _is_trace_equal(
                candidate.sch,
                correct_sch,
                remove_decisions=(isinstance(strategy, ms.search_strategy.ReplayTrace)),
            )
            runner_results.append(
                ms.runner.RunnerResult(
                    run_secs=[0.11, 0.41, 0.54],
                    error_msg=None,
                )
            )
        strategy.notify_runner_results(candidates, runner_results)
        candidates = strategy.generate_measure_candidates()
    strategy.post_tuning()
    assert sum(num_trials_each_iter) == 25
    assert num_trials_each_iter.count(0) < 5


def test_meta_schedule_evolutionary_search_early_stop():  # pylint: disable = invalid-name
    def _schedule_matmul_empty(sch: Schedule):
        return sch

    (correct_sch,) = ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(
        Matmul
    )

    num_trials_per_iter = 10
    max_trials_per_task = 100

    context = ms.TuneContext(
        mod=Matmul,
        search_strategy=ms.search_strategy.EvolutionarySearch(
            population_size=5,
            init_measured_ratio=0.1,
            init_min_unmeasured=50,
            genetic_num_iters=3,
            genetic_mutate_prob=0.5,
            genetic_max_fail_count=10,
            eps_greedy=0.9,
        ),
        space_generator=ms.space_generator.ScheduleFn(
            sch_fn=_schedule_matmul_empty,
            sch_rules=[],
            postprocs=[],
            mutator_probs={
                DummyMutator(): 1.0,
            },
        ),
        target=tvm.target.Target("llvm"),
        num_threads=1,
    )
    strategy = context.search_strategy
    strategy.pre_tuning(
        max_trials=max_trials_per_task,
        num_trials_per_iter=num_trials_per_iter,
        design_spaces=context.space_generator.generate_design_space(context.mod),
        database=ms.database.MemoryDatabase(),
        cost_model=ms.cost_model.RandomModel(),
    )
    num_trials_each_iter: List[int] = []
    candidates = strategy.generate_measure_candidates()
    while candidates is not None:
        num_trials_each_iter.append(len(candidates))
        runner_results: List[ms.runner.RunnerResult] = []
        for candidate in candidates:
            _is_trace_equal(
                candidate.sch,
                correct_sch,
                remove_decisions=(isinstance(strategy, ms.search_strategy.ReplayTrace)),
            )
            runner_results.append(
                ms.runner.RunnerResult(
                    run_secs=[0.11, 0.41, 0.54],
                    error_msg=None,
                ),
            )
        strategy.notify_runner_results(candidates, runner_results)
        candidates = strategy.generate_measure_candidates()
    strategy.post_tuning()
    assert num_trials_each_iter == [1, 0, 0, 0, 0]


if __name__ == "__main__":
    test_meta_schedule_replay_func(ms.search_strategy.ReplayFunc)
    test_meta_schedule_replay_func(ms.search_strategy.ReplayTrace)
    test_meta_schedule_evolutionary_search()
    test_meta_schedule_evolutionary_search_early_stop()
