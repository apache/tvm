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
"""A database that stores TuningRecords in memory"""
from typing import List

from ...ir import IRModule, structural_equal
from ..utils import derived_object
from .database import PyDatabase, TuningRecord, Workload


@derived_object
class MemoryDatabase(PyDatabase):
    """An in-memory database based on python list for testing."""

    def __init__(self):
        super().__init__()
        self.records = []
        self.workload_reg = []

    def has_workload(self, mod: IRModule) -> bool:
        for workload in self.workload_reg:
            if structural_equal(workload.mod, mod):
                return True
        return False

    def commit_tuning_record(self, record: TuningRecord) -> None:
        self.records.append(record)

    def commit_workload(self, mod: IRModule) -> Workload:
        for workload in self.workload_reg:
            if structural_equal(workload.mod, mod):
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

    def get_all_tuning_records(self) -> List[TuningRecord]:
        return self.records

    def __len__(self) -> int:
        return len(self.records)

    def print_results(self) -> None:
        print("\n".join([str(r) for r in self.records]))
