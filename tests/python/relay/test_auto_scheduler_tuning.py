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

import numpy as np

from tvm import auto_scheduler, relay
from tvm.contrib import graph_runtime
import tvm.testing

from test_auto_scheduler_task_extraction import get_network


def tune_network(network, target):
    # Extract tasks
    mod, params = get_network(network)
    target = tvm.target.Target(target)
    tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

    with tempfile.NamedTemporaryFile() as fp:
        log_file = fp.name

        # Tuning
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(timeout=60)
        tuner = auto_scheduler.TaskScheduler(tasks, task_weights)
        tune_option = auto_scheduler.TuningOptions(
            num_measure_trials=100,
            num_measures_per_round=2,
            early_stopping=1,
            runner=measure_ctx.runner,
            builder=auto_scheduler.LocalBuilder(timeout=60),
            measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
        )
        tuner.tune(tune_option, search_policy="sketch.random")
        del measure_ctx

        # Compile with the history best
        with auto_scheduler.ApplyHistoryBest(log_file):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib = relay.build(mod, target=target, params=params)

        # Sample a schedule when missing
        with auto_scheduler.ApplyHistoryBestOrSample(None, num_measure=2):
            with tvm.transform.PassContext(
                opt_level=3, config={"relay.backend.use_auto_scheduler": True}
            ):
                lib2 = relay.build(mod, target=target, params=params)

        # Compile without auto-scheduler and any other optimization for correctness check
        with tvm.transform.PassContext(opt_level=0):
            ref_lib = relay.build(mod, target=target, params=params)

        # Check the correctness
        def get_output(data, lib):
            dev = tvm.gpu()
            module = graph_runtime.GraphModule(lib["default"](dev))
            module.set_input("data", data)
            module.run()
            return module.get_output(0).asnumpy()

        np.random.seed(0)
        if network == "mlp":
            data = np.random.uniform(size=(1, 32))
        elif network == "winograd-test":
            data = np.random.uniform(size=(1, 23, 40, 32))
        else:
            raise ValueError("Unknown network: " + network)

        actual_output1 = get_output(data, lib)
        actual_output2 = get_output(data, lib2)
        expected_output = get_output(data, ref_lib)

        tvm.testing.assert_allclose(actual_output1, expected_output, rtol=1e-4, atol=1e-4)
        tvm.testing.assert_allclose(actual_output2, expected_output, rtol=1e-4, atol=1e-4)


@tvm.testing.requires_cuda
def test_tuning_cuda():
    tune_network("mlp", "cuda")
    tune_network("winograd-test", "cuda")


if __name__ == "__main__":
    test_tuning_cuda()
