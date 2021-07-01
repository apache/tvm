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
"""Test builder and runner"""
import logging
import multiprocessing
import time

import numpy as np

import tvm
from tvm import te
from test_autotvm_common import DummyRunner, bad_matmul, get_sample_task
from tvm import autotvm
from tvm.autotvm.measure.measure import MeasureErrorNo, MeasureResult
from tvm.autotvm import measure
from inspect import Signature


def test_task_tuner_without_measurement():
    """test task and tuner without measurement"""
    task, _ = get_sample_task()

    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=DummyRunner())

    logging.info("%s", task.config_space)

    for tuner_class in [
        autotvm.tuner.RandomTuner,
        autotvm.tuner.GridSearchTuner,
        autotvm.tuner.GATuner,
        autotvm.tuner.XGBTuner,
    ]:
        tuner = tuner_class(task)
        tuner.tune(n_trial=10, measure_option=measure_option)
        assert tuner.best_flops > 1


def task_tuner_spawn():
    assert multiprocessing.get_start_method(False) == "spawn"
    test_task_tuner_without_measurement()


def test_task_tuner_without_measurement_spawn():
    # Subprocesses inherit the spawn method of their parents
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=task_tuner_spawn)
    p.start()
    p.join()


def test_task_runner_with_ref_input():
    """test runner ref_input without measurement"""
    refinp = [np.random.rand(128, 128) for i in range(3)]
    measure_option = autotvm.measure_option(builder="local", runner="local", ref_input=refinp)

    class DummyExecutor(measure.executor.Executor):
        def submit(self, func, *args, **kwargs):
            sig = Signature.from_callable(measure.measure_methods.run_through_rpc)
            assert sig.bind(*args, **kwargs).arguments["ref_input"] == refinp
            return measure.local_executor.LocalFutureNoFork(None)

    measure_option["runner"].executor = DummyExecutor()
    measure_option["runner"].run([None], [None])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_task_tuner_without_measurement()
    test_task_tuner_without_measurement_spawn()
    test_task_runner_with_ref_input()
