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
"""Test index based tuners"""

from test_autotvm_common import DummyRunner, get_sample_task
from tvm import autotvm
from tvm.autotvm.tuner import GridSearchTuner, RandomTuner


def test_gridsearch_tuner():
    """Test GridSearchTuner"""

    task, _ = get_sample_task()
    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=DummyRunner())

    # When no range index, range_length should be the length of config space
    tuner = autotvm.tuner.GridSearchTuner(task)
    assert tuner.range_length == len(task.config_space)
    assert tuner.index_offset == 0

    # With range index, range_length should be the length of the specified range
    tuner = autotvm.tuner.GridSearchTuner(task, range_idx=(8, 15))
    assert tuner.range_length == 8
    assert tuner.index_offset == 8

    # Tuner should only focus on the specified range
    tuner.tune(n_trial=8, measure_option=measure_option)
    assert tuner.counter == 8
    assert not tuner.has_next()


def test_random_tuner():
    """Test RandomTuner"""

    task, _ = get_sample_task()
    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=DummyRunner())

    tuner = autotvm.tuner.RandomTuner(task, range_idx=(8, 15))
    assert tuner.range_length == 8
    assert tuner.index_offset == 8

    # Tuner should only focus on the specified range and should visit all indices
    tuner.tune(n_trial=8, measure_option=measure_option)
    assert tuner.counter == 8
    assert not tuner.has_next()
    visited = set()
    for idx in tuner.visited:
        assert idx not in visited
        assert 8 <= idx <= 15


if __name__ == "__main__":
    test_gridsearch_tuner()
    test_random_tuner()
