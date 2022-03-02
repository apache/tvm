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
from typing import List

import pytest
import tvm
from tvm.ir.base import assert_structural_equal
from tvm.meta_schedule.builder import BuilderResult
from tvm.meta_schedule.measure_callback import PyMeasureCallback
from tvm.meta_schedule.runner import RunnerResult
from tvm.meta_schedule.search_strategy import MeasureCandidate
from tvm.meta_schedule.task_scheduler.task_scheduler import TaskScheduler
from tvm.meta_schedule.utils import _get_hex_address
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
    class FancyMeasureCallback(PyMeasureCallback):
        def apply(
            self,
            task_scheduler: TaskScheduler,
            task_id: int,
            measure_candidates: List[MeasureCandidate],
            builds: List[BuilderResult],
            results: List[RunnerResult],
        ) -> None:
            assert len(measure_candidates) == 1
            assert_structural_equal(measure_candidates[0].sch.mod, Matmul)
            assert (
                len(builds) == 1
                and builds[0].error_msg is None
                and builds[0].artifact_path == "test_build"
            )
            assert (
                len(results) == 1 and results[0].error_msg is None and len(results[0].run_secs) == 2
            )

    measure_callback = FancyMeasureCallback()
    measure_callback.apply(
        TaskScheduler(),
        0,
        [MeasureCandidate(Schedule(Matmul), None)],
        [BuilderResult("test_build", None)],
        [RunnerResult([1.0, 2.1], None)],
    )


def test_meta_schedule_measure_callback_fail():
    class FailingMeasureCallback(PyMeasureCallback):
        def apply(
            self,
            task_scheduler: TaskScheduler,
            task_id: int,
            measure_candidates: List[MeasureCandidate],
            builds: List[BuilderResult],
            results: List[RunnerResult],
        ) -> None:
            raise ValueError("test")

    measure_callback = FailingMeasureCallback()
    with pytest.raises(ValueError, match="test"):
        measure_callback.apply(
            TaskScheduler(),
            0,
            [MeasureCandidate(Schedule(Matmul), None)],
            [BuilderResult("test_build", None)],
            [RunnerResult([1.0, 2.1], None)],
        )


def test_meta_schedule_measure_callback_as_string():
    class NotSoFancyMeasureCallback(PyMeasureCallback):
        def apply(
            self,
            task_scheduler: "TaskScheduler",
            task_id: int,
            measure_candidates: List[MeasureCandidate],
            builds: List[BuilderResult],
            results: List[RunnerResult],
        ) -> None:
            pass

        def __str__(self) -> str:
            return f"NotSoFancyMeasureCallback({_get_hex_address(self.handle)})"

    measure_callback = NotSoFancyMeasureCallback()
    pattern = re.compile(r"NotSoFancyMeasureCallback\(0x[a-f|0-9]*\)")
    assert pattern.match(str(measure_callback))


if __name__ == "__main__":
    test_meta_schedule_measure_callback()
    test_meta_schedule_measure_callback_fail()
    test_meta_schedule_measure_callback_as_string()
