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
"""Testing utilitiy functions in meta schedule"""
from typing import List, Optional
import random

import tvm

from tvm.meta_schedule import TuneContext  # pylint: disable=unused-import
from tvm.meta_schedule.utils import derived_object
from tvm.meta_schedule.mutator.mutator import PyMutator
from tvm.meta_schedule.database import PyDatabase, Workload, TuningRecord
from tvm.meta_schedule.builder import PyBuilder, BuilderInput, BuilderResult
from tvm.meta_schedule.runner import (
    RunnerInput,
    RunnerResult,
    RunnerFuture,
    PyRunnerFuture,
    PyRunner,
)
from tvm.ir import IRModule
from tvm.tir.schedule import Trace


@derived_object
class DummyDatabase(PyDatabase):
    """
    An in-memory database based on python list for testing.
    """

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


@derived_object
class DummyRunnerFuture(PyRunnerFuture):
    def done(self) -> bool:
        return True

    def result(self) -> RunnerResult:
        run_secs = [random.uniform(5, 30) for _ in range(random.randint(1, 10))]
        return RunnerResult(run_secs, None)


@derived_object
class DummyBuilder(PyBuilder):
    def build(self, build_inputs: List[BuilderInput]) -> List[BuilderResult]:
        return [BuilderResult("test_path", None) for _ in build_inputs]


@derived_object
class DummyRunner(PyRunner):
    def run(self, runner_inputs: List[RunnerInput]) -> List[RunnerFuture]:
        return [DummyRunnerFuture() for _ in runner_inputs]  # type: ignore


@derived_object
class DummyMutator(PyMutator):
    """Dummy Mutator for testing"""

    def initialize_with_tune_context(self, context: "TuneContext") -> None:
        pass

    def apply(self, trace: Trace, _) -> Optional[Trace]:
        return Trace(trace.insts, {})
