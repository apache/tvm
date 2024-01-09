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
"""Test genetic algorithm tuner"""

from tvm.testing.autotvm import DummyRunner, get_sample_task
from tvm import autotvm


def test_ga_tuner():
    """Test GATuner"""
    # Test population size smaller than space size tuning configuration
    task, _ = get_sample_task()
    tuner = autotvm.tuner.GATuner(task, pop_size=32)
    valid_indexes = list(
        filter(lambda idx: tuner.space.is_index_valid(idx), range(tuner.space.range_length))
    )
    assert tuner.visited.issubset(valid_indexes)
    assert tuner.pop_size == len(tuner.visited) == len(tuner.genes)
    assert len(tuner.space) == 64

    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=DummyRunner())
    tuner.tune(n_trial=len(tuner.space), measure_option=measure_option)
    assert tuner.visited.issubset(valid_indexes)

    # Test population size bigger than space size tuning configuration
    task, _ = get_sample_task()
    tuner = autotvm.tuner.GATuner(task, pop_size=100)
    valid_indexes = list(
        filter(lambda idx: tuner.space.is_index_valid(idx), range(tuner.space.range_length))
    )
    assert tuner.visited.issubset(valid_indexes)
    assert tuner.pop_size == len(tuner.visited) == len(tuner.genes)
    assert len(tuner.space) == 64

    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=DummyRunner())
    tuner.tune(n_trial=len(tuner.space), measure_option=measure_option)
    assert tuner.visited.issubset(valid_indexes)

    # Test population size smaller than multi-filtered space size tuning configuration
    task, _ = get_sample_task()
    task.config_space.multi_filter(
        filter=lambda entity: 8 <= (entity["tile_x"].size[1] * entity["tile_y"].size[1]) < 1024
    )
    tuner = autotvm.tuner.GATuner(task, pop_size=32)
    valid_indexes = list(
        filter(lambda idx: tuner.space.is_index_valid(idx), range(tuner.space.range_length))
    )
    assert tuner.visited.issubset(valid_indexes)
    assert tuner.pop_size == len(tuner.visited) == len(tuner.genes)
    assert len(tuner.space) == 43

    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=DummyRunner())
    tuner.tune(n_trial=len(tuner.space), measure_option=measure_option)
    assert tuner.visited.issubset(valid_indexes)

    # Test population size bigger than multi-filtered space size tuning configuration
    task, _ = get_sample_task()
    task.config_space.multi_filter(
        filter=lambda entity: 8 <= (entity["tile_x"].size[1] * entity["tile_y"].size[1]) < 1024
    )
    tuner = autotvm.tuner.GATuner(task, pop_size=100)
    valid_indexes = list(
        filter(lambda idx: tuner.space.is_index_valid(idx), range(tuner.space.range_length))
    )
    assert tuner.visited.issubset(valid_indexes)
    assert tuner.pop_size == len(tuner.visited) == len(tuner.genes)
    assert len(tuner.space) == 43

    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=DummyRunner())
    tuner.tune(n_trial=len(tuner.space), measure_option=measure_option)
    assert tuner.visited.issubset(valid_indexes)


if __name__ == "__main__":
    test_ga_tuner()
