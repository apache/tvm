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
""" Test task scheduler """

import tempfile

import numpy as np

import tvm
import tvm.testing
from tvm import auto_scheduler

from tvm.testing.auto_scheduler import matmul_auto_scheduler_test


@tvm.testing.requires_llvm
def test_task_scheduler_gradient_droplet():
    tasks = []
    for n in [2, 4]:
        tasks.append(
            auto_scheduler.SearchTask(
                func=matmul_auto_scheduler_test, args=(n, n, n), target="llvm"
            )
        )

    def objective_func(costs):
        return 1e5 * costs[0]

    with tempfile.NamedTemporaryFile() as fp:
        log_file = fp.name

        n_trials = 5

        # Tune all tasks
        measure_ctx = auto_scheduler.LocalRPCMeasureContext()
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=n_trials,
            runner=measure_ctx.runner,
            num_measures_per_round=1,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
        task_scheduler = auto_scheduler.TaskScheduler(
            tasks, objective_func=objective_func, callbacks=[]
        )

        # Forcely rewrite the initial values.
        # This can make this test more stable on the slow CI machines
        task_scheduler.best_costs = np.array([1e2, 1e-8])

        task_scheduler.tune(tune_option, search_policy="sketch.random")

        # Use the droplet algorithm to optimize the kernel
        auto_scheduler.task_scheduler.droplet_exploitation(log_file, tasks[0].target)

        # Check the allocation results
        counters = {}
        for task in tasks:
            counters[task.workload_key] = 0

        for inp, _ in auto_scheduler.load_records(log_file):
            counters[inp.task.workload_key] += 1

        # droplet adds an optimized solution at the end
        assert counters[tasks[0].workload_key] == n_trials
        assert counters[tasks[1].workload_key] == 2
        del measure_ctx


if __name__ == "__main__":
    test_task_scheduler_gradient_droplet()
