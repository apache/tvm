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
import sys
from typing import List, Optional, Tuple, Union

import numpy as np
import pytest
import tvm
from tvm.ir import IRModule
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.builder import LocalBuilder
from tvm.meta_schedule.cost_model import PyCostModel
from tvm.meta_schedule.database import PyDatabase, TuningRecord, Workload
from tvm.meta_schedule.mutator.mutator import PyMutator
from tvm.meta_schedule.runner import LocalRunner, RunnerResult
from tvm.meta_schedule.search_strategy import (
    EvolutionarySearch,
    MeasureCandidate,
    ReplayFunc,
    ReplayTrace,
    SearchStrategy,
)
from tvm.meta_schedule.space_generator import ScheduleFn
from tvm.meta_schedule.task_scheduler import RoundRobin
from tvm.script import tir as T
from tvm.tir.schedule import Schedule, Trace


MATMUL_M = 32

# pylint: disable=missing-class-docstring,invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument, unbalanced-tuple-unpacking
# fmt: off

@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (32, 32), "float32")
        B = T.match_buffer(b, (32, 32), "float32")
        C = T.match_buffer(c, (32, 32), "float32")
        for i, j, k in T.grid(32, 32, 32):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
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


@pytest.mark.parametrize("TestClass", [ReplayFunc, ReplayTrace])
def test_meta_schedule_replay_func(TestClass: SearchStrategy):  # pylint: disable = invalid-name
    num_trials_per_iter = 7
    num_trials_total = 20

    strategy = TestClass(num_trials_per_iter=num_trials_per_iter, num_trials_total=num_trials_total)
    context = TuneContext(mod=Matmul, space_generator=ScheduleFn(sch_fn=_schedule_matmul))
    context.space_generator.initialize_with_tune_context(context)
    spaces = context.space_generator.generate_design_space(context.mod)

    strategy.initialize_with_tune_context(context)
    strategy.pre_tuning(spaces)
    (correct_sch,) = ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(Matmul)
    num_trials_each_iter: List[int] = []
    candidates = strategy.generate_measure_candidates()
    while candidates is not None:
        num_trials_each_iter.append(len(candidates))
        runner_results: List[RunnerResult] = []
        for candidate in candidates:
            _is_trace_equal(
                candidate.sch,
                correct_sch,
                remove_decisions=(isinstance(strategy, ReplayTrace)),
            )
            runner_results.append(RunnerResult(run_secs=[0.11, 0.41, 0.54], error_msg=None))
        strategy.notify_runner_results(context, candidates, runner_results)
        candidates = strategy.generate_measure_candidates()
    strategy.post_tuning()
    assert num_trials_each_iter == [7, 7, 6]


def test_meta_schedule_evolutionary_search():  # pylint: disable = invalid-name
    class DummyMutator(PyMutator):
        """Dummy Mutator for testing"""

        def initialize_with_tune_context(self, context: "TuneContext") -> None:
            pass

        def apply(self, trace: Trace) -> Optional[Trace]:
            return Trace(trace.insts, {})

    class DummyDatabase(PyDatabase):
        """Dummy Database for testing"""

        def __init__(self):
            super().__init__()
            self.records = []
            self.workload_reg = []

        def has_workload(self, mod: IRModule) -> bool:
            for workload in self.workload_reg:
                if tvm.ir.structural_equal(workload.mod, mod):
                    return True
            return False

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

    class RandomModel(PyCostModel):
        """Random cost model for testing"""

        random_state: Union[Tuple[str, np.ndarray, int, int, float], dict]
        path: Optional[str]

        def __init__(
            self,
            *,
            seed: Optional[int] = None,
            path: Optional[str] = None,
            max_range: Optional[int] = 100,
        ):
            super().__init__()
            if path is not None:
                self.load(path)
            else:
                np.random.seed(seed)
                self.random_state = np.random.get_state()
            self.max_range = max_range

        def load(self, path: str) -> None:
            self.random_state = tuple(np.load(path, allow_pickle=True))

        def save(self, path: str) -> None:
            np.save(path, np.array(self.random_state, dtype=object), allow_pickle=True)

        def update(
            self,
            context: TuneContext,
            candidates: List[MeasureCandidate],
            results: List[RunnerResult],
        ) -> None:
            pass

        def predict(self, context: TuneContext, candidates: List[MeasureCandidate]) -> np.ndarray:
            np.random.set_state(self.random_state)
            result = np.random.rand(len(candidates)) * self.max_range
            self.random_state = np.random.get_state()
            return result

    num_trials_per_iter = 10
    num_trials_total = 100

    strategy = EvolutionarySearch(
        num_trials_per_iter=num_trials_per_iter,
        num_trials_total=num_trials_total,
        population_size=5,
        init_measured_ratio=0.1,
        init_max_fail_count=10,
        genetic_num_iters=3,
        genetic_mutate_prob=0.5,
        genetic_max_fail_count=10,
        eps_greedy=0.9,
    )
    context = TuneContext(
        mod=Matmul,
        space_generator=ScheduleFn(sch_fn=_schedule_matmul),
        mutator_probs={
            DummyMutator(): 1.0,
        },
        target=tvm.target.Target("llvm"),
        num_threads=1,  # because we are using a mutator from the python side
    )
    _scheduler = RoundRobin(
        tasks=[context],
        builder=LocalBuilder(),
        runner=LocalRunner(),
        database=DummyDatabase(),
        cost_model=RandomModel(),
        measure_callbacks=[],
    )
    context.space_generator.initialize_with_tune_context(context)
    spaces = context.space_generator.generate_design_space(context.mod)

    strategy.initialize_with_tune_context(context)
    strategy.pre_tuning(spaces)
    (correct_sch,) = ScheduleFn(sch_fn=_schedule_matmul).generate_design_space(Matmul)
    num_trials_each_iter: List[int] = []
    candidates = strategy.generate_measure_candidates()
    while candidates is not None:
        num_trials_each_iter.append(len(candidates))
        runner_results: List[RunnerResult] = []
        for candidate in candidates:
            _is_trace_equal(
                candidate.sch,
                correct_sch,
                remove_decisions=(isinstance(strategy, ReplayTrace)),
            )
            runner_results.append(RunnerResult(run_secs=[0.11, 0.41, 0.54], error_msg=None))
        strategy.notify_runner_results(context, candidates, runner_results)
        candidates = strategy.generate_measure_candidates()
    strategy.post_tuning()
    print(num_trials_each_iter)
    correct_count = 10  # For each iteration except the last one
    assert num_trials_each_iter == [correct_count] * (num_trials_total // correct_count) + (
        [num_trials_total % correct_count] if num_trials_total % correct_count != 0 else []
    )
    del _scheduler


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
