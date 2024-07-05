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
"""Test task scheduler"""

import tempfile

import tvm
import tvm.testing
from tvm import auto_scheduler

from tvm.testing.auto_scheduler import matmul_auto_scheduler_test


@tvm.testing.requires_llvm
def test_dynamic_gradient_descent():
    tasks = []
    for n in [2, 4]:
        tasks.append(
            auto_scheduler.SearchTask(
                func=matmul_auto_scheduler_test, args=(n, n, n), target="llvm"
            )
        )

    with tempfile.NamedTemporaryFile() as fp:
        log_file = fp.name

        # Tune all tasks
        for task in tasks:
            slide_window_size = 10  # Size of the sliding window used in dynamic gradient search
            max_tuning_time = 120  # Maximum tuning time in seconds, 120 is the suggested value
            max_trials = 15  # Maximum number of measurement trials to perform in dynamic gradient search, use 1000 to get better performance
            n_start = 5  # Number of start points from the initial sampled population
            init_size = (
                5  # Number of samples to generate the initial model, 64 is the suggested value
            )
            predict_score_threshold_ratio = 0.6  # Threshold for the predict score
            measure_threshold_ratio = 0.6  # Threshold for the measured throughput

            # Tuning options, tested with local runner and builder
            tune_option = auto_scheduler.TuningOptions(
                runner=auto_scheduler.LocalRunner(timeout=10),
                builder=auto_scheduler.LocalBuilder(timeout=10),
            )

            # Initialize tuner
            tuner = auto_scheduler.dynamic_gradient_search.DynamicGradientSearchTuner(
                task,
                log_file,
                tune_option,
                n_start,
                init_size,
                slide_window_size,
                max_trials,
                max_tuning_time,
                predict_score_threshold_ratio,
                measure_threshold_ratio,
            )

            # Run tuner
            tuner.dynamic_gradient_search()

        # Check the allocation results
        counters = {}
        for task in tasks:
            counters[task.workload_key] = 0

        for inp, _ in auto_scheduler.load_records(log_file):
            counters[inp.task.workload_key] += 1

        assert counters[tasks[0].workload_key] == max_trials


if __name__ == "__main__":
    test_dynamic_gradient_descent()
