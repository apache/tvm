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

import multiprocessing
from tvm.testing.autotvm import DummyRunner, get_sample_task
from tvm import autotvm


def test_grid_search_tuner():
    """Test GridSearchTuner"""

    task, _ = get_sample_task()
    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=DummyRunner())

    # When no range index, range_length should be the length of config space
    tuner = autotvm.tuner.GridSearchTuner(task)
    assert tuner.begin_idx == 0
    assert tuner.end_idx == 64
    assert tuner.index == 0
    assert tuner.range_length == 64
    assert tuner.visited_max == 64

    # With range index, range_length should be the length of the specified range
    tuner = autotvm.tuner.GridSearchTuner(task, range_idx=(8, 15))
    assert tuner.begin_idx == 8
    assert tuner.end_idx == 16
    assert tuner.index == 8
    assert tuner.range_length == 8
    assert tuner.visited_max == 8

    # Tuner should only focus on the specified range
    tuner.tune(n_trial=8, measure_option=measure_option)
    assert len(tuner.visited) == 8
    assert not tuner.has_next()

    # With multi-filter
    task, _ = get_sample_task()
    task.config_space.multi_filter(
        filter=lambda entity: 32 <= (entity["tile_x"].size[1] * entity["tile_y"].size[1]) < 1024
    )

    tuner = autotvm.tuner.GridSearchTuner(task)
    assert tuner.begin_idx == 0
    assert tuner.end_idx == 64
    assert tuner.index == 5
    assert tuner.range_length == 64
    assert tuner.visited_max == 34

    # With range index, range_length should be the length of the specified range
    tuner = autotvm.tuner.GridSearchTuner(task, range_idx=(8, 15))
    assert tuner.begin_idx == 8
    assert tuner.end_idx == 16
    assert tuner.index == 12
    assert tuner.range_length == 8
    assert tuner.visited_max == 4

    # Tuner should only focus on the specified range
    tuner.tune(n_trial=8, measure_option=measure_option)
    assert len(tuner.visited) == 4
    assert not tuner.has_next()


def grid_search_spawn():
    assert multiprocessing.get_spawn_method(False) == "spawn"
    test_grid_search_tuner()


def test_grid_search_tuner_spawn():
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=test_grid_search_tuner)
    p.start()
    p.join()


def test_random_tuner():
    """Test RandomTuner"""

    task, _ = get_sample_task()
    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=DummyRunner())

    tuner = autotvm.tuner.RandomTuner(task, range_idx=(8, 15))
    assert tuner.begin_idx == 8
    assert tuner.end_idx == 16
    assert tuner.range_length == 8
    assert tuner.visited_max == 8

    # Tuner should only focus on the specified range and should visit all indices
    tuner.tune(n_trial=8, measure_option=measure_option)
    assert len(tuner.visited) == 8
    assert not tuner.has_next()
    for idx in tuner.visited:
        assert 8 <= idx <= 15

    # With multi-filter
    task, _ = get_sample_task()
    task.config_space.multi_filter(
        filter=lambda entity: 32 <= (entity["tile_x"].size[1] * entity["tile_y"].size[1]) < 1024
    )
    tuner = autotvm.tuner.RandomTuner(task, range_idx=(8, 15))
    assert tuner.begin_idx == 8
    assert tuner.end_idx == 16
    assert tuner.range_length == 8
    assert tuner.visited_max == 4

    # Tuner should only focus on the specified range and should visit all indices
    tuner.tune(n_trial=8, measure_option=measure_option)
    assert len(tuner.visited) == 4
    assert not tuner.has_next()
    for idx in tuner.visited:
        assert 8 <= idx <= 15


if __name__ == "__main__":
    test_grid_search_tuner()
    test_grid_search_tuner_spawn()
    test_random_tuner()
