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
"""Test droplet algorithm tuner"""

from tvm.testing.autotvm import DummyRunner, get_sample_task, get_sample_records
from tvm import autotvm


def test_tuner():
    """Test Droplet Tuner"""
    
    task, _ = get_sample_task()
    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=DummyRunner())

    # When no range index, range_length should be the length of config space
    tuner = autotvm.tuner.DropletTuner(task)

    assert tuner.p_value == 0.05
    assert tuner.start_position == None

    assert len(tuner.start_position) == len(tuner.dims)
    assert len(tuner.best_choice) == 3
    assert tuner.execution == 1
    assert tuner.batch == 16
    assert tuner.total_execution == max(tuner.dims)
    assert tuner.step == 1

    assert not tuner.has_next()


def test_multi_filter():
    
    # Test with multi-filter
    task, _ = get_sample_task()
    task.config_space.multi_filter(
        filter=lambda entity: 32 <= (entity["tile_x"].size[1] * entity["tile_y"].size[1]) < 1024
    )
    tuner = autotvm.tuner.DropletTuner(task)
    valid_indexes = list(
        filter(lambda idx: tuner.space.is_index_valid(idx), range(tuner.space.range_length))
    )
    assert tuner.visited.issubset(valid_indexes)
    assert len(tuner.next) <= len(tuner.visited)

    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=DummyRunner())
    tuner.tune(n_trial=len(tuner.space), measure_option=measure_option)
    assert tuner.visited.issubset(valid_indexes)


def test_update():

    task, target = get_sample_task()
    tuner = autotvm.tuner.DropletTuner(task)
    n_records = 5
    records = get_sample_records(n=n_records)
    tuner.update([inp for inp, _ in records], [res for _, res in records])
    assert len(tuner.visited) == n_records


if __name__ == "__main__":
    test_tuner()
    test_multi_filter()
    test_update()
