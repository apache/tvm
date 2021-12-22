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
import pytest
from typing import List

import tvm
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.runner import RunnerResult
from tvm.meta_schedule.search_strategy import (
    ReplayTrace,
    SearchStrategy,
)
from tvm.meta_schedule.space_generator import ScheduleFn
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


@pytest.mark.parametrize("TestClass", [ReplayTrace])
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__] + sys.argv[1:]))
