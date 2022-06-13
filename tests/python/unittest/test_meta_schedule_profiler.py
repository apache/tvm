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
""" Test Meta Schedule Profiler """
from typing import List
import time

import tvm
from tvm.meta_schedule.search_strategy import ReplayTrace
from tvm import meta_schedule as ms
from tvm.tir.schedule import Schedule
from tvm.script import tir as T


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


def test_meta_schedule_profiler_replay_trace():
    def _schedule_matmul(sch: Schedule):
        block = sch.get_block("matmul")
        i, j, k = sch.get_loops(block=block)
        i_0, i_1, i_2, i_3 = sch.split(i, sch.sample_perfect_tile(i, n=4))
        j_0, j_1, j_2, j_3 = sch.split(j, sch.sample_perfect_tile(j, n=4))
        k_0, k_1 = sch.split(k, sch.sample_perfect_tile(k, n=2))
        sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)

    with ms.Profiler() as profiler:
        num_trials_per_iter = 7
        max_trials_per_task = 20

        context = ms.TuneContext(
            mod=Matmul,
            space_generator=ms.space_generator.ScheduleFn(sch_fn=_schedule_matmul),
            search_strategy=ReplayTrace(
                num_trials_per_iter=num_trials_per_iter, max_trials_per_task=max_trials_per_task
            ),
        )
        context.initialize()
        strategy = context.search_strategy
        spaces = context.generate_design_space()
        strategy.pre_tuning(spaces)
        num_trials_each_iter: List[int] = []
        candidates = strategy.generate_measure_candidates()
        while candidates is not None:
            num_trials_each_iter.append(len(candidates))
            runner_results: List[ms.runner.RunnerResult] = []
            for _ in candidates:
                runner_results.append(
                    ms.runner.RunnerResult(
                        run_secs=[1.0],
                        error_msg=None,
                    )
                )
            strategy.notify_runner_results(candidates, runner_results)
            candidates = strategy.generate_measure_candidates()
        strategy.post_tuning()

    for term in [
        "ReplayTraceNode::NotifyRunnerResults",
        "ReplayTraceNode::GenerateMeasureCandidates",
        "ReplayTraceNode::PostTuning",
        "ReplayTraceNode::PreTuning",
    ]:
        assert term in profiler.get()
        assert profiler.get()[term] > 0


def test_meta_schedule_profiler_context_manager():
    with ms.Profiler() as profiler:
        with profiler.timeit("Level0"):
            time.sleep(2)
            with profiler.timeit("Level1"):
                time.sleep(3)
    # Note that the results are in minutes
    assert 5 <= profiler.get()["Level0"] * 60 <= 6
    assert 3 <= profiler.get()["Level1"] * 60 <= 4


if __name__ == "__main__":
    tvm.testing.main()
