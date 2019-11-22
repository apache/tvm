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
"""Test Tuning Job"""
import copy
import numpy as np
import tempfile

from tvm.autotvm.tuner.tuning_job import TuningJob

from test_autotvm_common import get_sample_records


class MockTuner:
    pass


def test_log_configs():
    with tempfile.TemporaryFile("w") as log_file:
        job = TuningJob(
            log_file,
            "llvm",
        )
        timings = [[1, 4, 2], [5, 1, 3], [8, 3, 2], [9, 7, 8], [3, 7, 6],
                   [2, 8, 2], [1, 2, 1], [7, 7, 7], [9, 3, 3], [5, 3, 9]]
        avg_timings = np.mean(timings, axis=1)
        min_time = min(avg_timings)
        # Create first workload results
        records = get_sample_records(10)
        workloads = [
            ("matmul", 64, 64, 64, "float32"),
            ("matmul", 128, 128, 128, "float32"),
            ("matmul", 256, 256, 256, "float32"),
        ]
        inputs = []
        results = []
        for workload in workloads:
            for record, timing in zip(records, timings):
                inp, result = copy.deepcopy(record)
                result = result._replace(costs=tuple(timing))
                inp.task.workload = workload
                inputs.append(inp)
                results.append(result)

        job.log_configs(MockTuner(), inputs, results)
        # Check the workloads have been saved
        assert set(workloads) == set(job.results_by_workload.keys())
        for workload in job.results_by_workload:
            inp, result, tuner_name, trials = job.results_by_workload[workload]
            # Check the best record was saved
            assert tuner_name == "MockTuner"
            assert trials == 10
            assert inp.task.workload == workload
            assert np.mean(result.costs) == min_time


def test_get_records():
    with tempfile.NamedTemporaryFile("w") as log_file:
        job = TuningJob(
            log_file.name,
            "llvm",
        )
        records = get_sample_records(10)
        inputs = []
        results = []
        for record in records:
            inp, result = record
            inputs.append(inp)
            results.append(result)

        job.log_configs(MockTuner(), inputs, results)
        record_generator = job.get_records()
        record_count = len([1 for _ in record_generator])
        assert record_count == 10


if __name__ == "__main__":
    test_log_configs()
    test_get_records()
