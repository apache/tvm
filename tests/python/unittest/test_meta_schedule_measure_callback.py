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
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import re
import tempfile
from typing import List

import pytest
import tvm
from tvm import meta_schedule as ms
from tvm.script import tir as T
from tvm.tir.schedule import Schedule

# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,
# fmt: off

@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def test_meta_schedule_measure_callback():
    @ms.derived_object
    class FancyMeasureCallback(ms.measure_callback.PyMeasureCallback):
        def apply(
            self,
            task_scheduler: ms.task_scheduler.TaskScheduler,
            task_id: int,
            measure_candidates: List[ms.MeasureCandidate],
            builder_results: List[ms.builder.BuilderResult],
            runner_results: List[ms.runner.RunnerResult],
        ) -> None:
            assert len(measure_candidates) == 1
            tvm.ir.assert_structural_equal(measure_candidates[0].sch.mod, Matmul)
            assert (
                len(builder_results) == 1
                and builder_results[0].error_msg is None
                and builder_results[0].artifact_path == "test_build"
            )
            assert (
                len(runner_results) == 1
                and runner_results[0].error_msg is None
                and len(runner_results[0].run_secs) == 2
            )

    measure_callback = FancyMeasureCallback()
    measure_callback.apply(
        ms.task_scheduler.RoundRobin(),
        0,
        [ms.MeasureCandidate(Schedule(Matmul), None)],
        [ms.builder.BuilderResult("test_build", None)],
        [ms.runner.RunnerResult([1.0, 2.1], None)],
    )


def test_meta_schedule_measure_callback_fail():
    @ms.derived_object
    class FailingMeasureCallback(ms.measure_callback.PyMeasureCallback):
        def apply(
            self,
            task_scheduler: ms.task_scheduler.TaskScheduler,
            task_id: int,
            measure_candidates: List[ms.MeasureCandidate],
            builder_results: List[ms.builder.BuilderResult],
            runner_results: List[ms.runner.RunnerResult],
        ) -> None:
            raise ValueError("test")

    measure_callback = FailingMeasureCallback()
    with pytest.raises(ValueError, match="test"):
        measure_callback.apply(
            ms.task_scheduler.RoundRobin(),
            0,
            [ms.MeasureCandidate(Schedule(Matmul), None)],
            [ms.builder.BuilderResult("test_build", None)],
            [ms.runner.RunnerResult([1.0, 2.1], None)],
        )


def test_meta_schedule_measure_callback_as_string():
    @ms.derived_object
    class NotSoFancyMeasureCallback(ms.measure_callback.PyMeasureCallback):
        def apply(
            self,
            task_scheduler: ms.task_scheduler.TaskScheduler,
            task_id: int,
            measure_candidates: List[ms.MeasureCandidate],
            builder_results: List[ms.builder.BuilderResult],
            runner_results: List[ms.runner.RunnerResult],
        ) -> None:
            pass

    measure_callback = NotSoFancyMeasureCallback()
    pattern = re.compile(r"meta_schedule.NotSoFancyMeasureCallback\(0x[a-f|0-9]*\)")
    assert pattern.match(str(measure_callback))


def test_meta_schedule_measure_callback_update_cost_model_with_zero():
    @ms.derived_object
    class AllZeroRunnerFuture(ms.runner.PyRunnerFuture):
        def done(self) -> bool:
            return True

        def result(self) -> ms.runner.RunnerResult:
            return ms.runner.RunnerResult([0.0, 0.0], None)

    @ms.derived_object
    class AllZeroRunner(ms.runner.PyRunner):
        def run(self, runner_inputs: List[ms.runner.RunnerInput]) -> List[ms.runner.RunnerResult]:
            return [AllZeroRunnerFuture() for _ in runner_inputs]

    with tempfile.TemporaryDirectory() as work_dir:
        ms.tune_tir(
            mod=Matmul,
            target="llvm -num-cores=1",
            work_dir=work_dir,
            max_trials_global=10,
            runner=AllZeroRunner(),
            measure_callbacks=[ms.measure_callback.UpdateCostModel()],
        )


def test_meta_schedule_measure_callback_update_cost_model_with_runtime_error():
    @ms.derived_object
    class EmptyRunnerFuture(ms.runner.PyRunnerFuture):
        def done(self) -> bool:
            return True

        def result(self) -> ms.runner.RunnerResult:
            return ms.runner.RunnerResult(None, "error")

    @ms.derived_object
    class EmptyRunner(ms.runner.PyRunner):
        def run(self, runner_inputs: List[ms.runner.RunnerInput]) -> List[ms.runner.RunnerResult]:
            return [EmptyRunnerFuture() for _ in runner_inputs]

    with tempfile.TemporaryDirectory() as work_dir:
        ms.tune_tir(
            mod=Matmul,
            target="llvm -num-cores=1",
            work_dir=work_dir,
            max_trials_global=10,
            runner=EmptyRunner(),
            measure_callbacks=[ms.measure_callback.UpdateCostModel()],
        )


if __name__ == "__main__":
    test_meta_schedule_measure_callback()
    test_meta_schedule_measure_callback_fail()
    test_meta_schedule_measure_callback_as_string()
    test_meta_schedule_measure_callback_update_cost_model_with_zero()
    test_meta_schedule_measure_callback_update_cost_model_with_runtime_error()
