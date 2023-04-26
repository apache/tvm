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
    assert len(tuner.best_choice) == 3
    assert tuner.execution == 1
    assert tuner.batch == 16
    assert tuner.total_execution == max(tuner.dims)
    assert tuner.step == 1


def test_multi_filter():
    # Test with multi-filter
    task, _ = get_sample_task()
    task.config_space.multi_filter(
        filter=lambda entity: 0 <= (entity["tile_x"].size[1] * entity["tile_y"].size[1]) < 1024
    )
    tuner = autotvm.tuner.DropletTuner(task)
    valid_indexes = list(
        filter(lambda idx: tuner.space.is_index_valid(idx), range(tuner.space.range_length))
    )
    assert tuner.visited.issubset(valid_indexes)

    measure_option = autotvm.measure_option(builder=autotvm.LocalBuilder(), runner=DummyRunner())
    tuner.tune(n_trial=8, measure_option=measure_option)
    assert tuner.visited.issubset(valid_indexes)


if __name__ == "__main__":
    test_tuner()
    test_multi_filter()
