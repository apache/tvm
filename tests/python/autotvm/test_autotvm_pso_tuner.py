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
"""Test Particle Swarm Optimization Tuner"""
import logging

from tvm import autotvm
from tvm.testing.autotvm import DummyRunner, get_sample_task

# Set up logging level
logging.basicConfig(filename="test_autotvm_pso_tuner.log", level=logging.DEBUG)
logger = logging.getLogger("autotvm_pso_tuner")
logger.setLevel(logging.DEBUG)


def test_pso_tuner():  # pylint: disable=missing-function-docstring
    task, _ = get_sample_task()
    logger.info("task: %s", task)
    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=DummyRunner())
    tuner = autotvm.tuner.PSOTuner(task)

    tuner.tune(n_trial=8, measure_option=measure_option)


def test_pso_tuner_multi_filter():  # pylint: disable=missing-function-docstring
    task, _ = get_sample_task()
    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=DummyRunner())
    task.config_space.multi_filter(
        filter=lambda entity: 8 <= (entity["tile_x"].size[1] * entity["tile_y"].size[1]) < 1024
    )  # pylint: disable=line-too-long
    tuner = autotvm.tuner.PSOTuner(task)
    tuner.tune(n_trial=8, measure_option=measure_option)


if __name__ == "__main__":
    # test_pso_tuner_()
    test_pso_tuner()
    test_pso_tuner_multi_filter()
