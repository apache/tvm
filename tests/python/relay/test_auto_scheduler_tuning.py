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
"""Test end-to-end network tuning with auto-scheduler"""
import tempfile

import tvm.testing
from tvm import auto_scheduler, relay

from test_auto_scheduler_task_extraction import get_network


@tvm.testing.requires_cuda
def test_tuning_cuda():
    auto_scheduler.enable_relay_integration()

    # Extract tasks
    mod, params = get_network("mlp")
    target = tvm.target.Target("cuda")
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    objective = lambda costs: sum(c * w for c, w in zip(costs, task_weights))

    with tempfile.NamedTemporaryFile() as fp:
        log_file = fp.name

        # Tuning
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(timeout=100)
        tuner = auto_scheduler.TaskScheduler(tasks, objective)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=2,
            num_measures_per_round=1,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
        tuner.tune(tune_option, search_policy="sketch.random")
        del measure_ctx

        # Compile with the history best
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(opt_level=3):
                lib = relay.build(mod, target=target, params=params)

    # Todo(merrymercy): compile without any history to test the fallback mechanism

    auto_scheduler.enable_relay_integration(False)


if __name__ == "__main__":
    test_tuning_cuda()
