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
"""Test selective tuning"""

import tvm
from collections import OrderedDict
from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner
from test_autotvm_common import matmul

def test_selective():
    shapes = [(10, 10, 10), (15, 10, 10), (20, 10, 10), (20, 10, 15)]

    tasks = [
        autotvm.task.create(matmul,
                            args=(*shape, 'float32'),
                            target='llvm -device=bad_device')
        for shape in shapes
    ]

    # Test task selection
    autotvm.task.mark_depend(tasks)
    assert all([t.depend == tasks[2] for t in tasks])

    c = autotvm.task.space.ConfigEntity(
        -1, None, "",
        OrderedDict({
            'tile_x': autotvm.task.space.SplitEntity([-1, 10]),
            'tile_y': autotvm.task.space.SplitEntity([-1, 5])
        }), [])
    tasks[2].tuned_configs = [c]

    tuner = XGBTuner(tasks[0], loss_type='rank')
    tuner.tune(n_trial=10,
               early_stopping=None,
               measure_option=autotvm.measure_option(
                   builder=autotvm.LocalBuilder(timeout=10),
                   runner=autotvm.LocalRunner(number=5, repeat=1)))
    assert len(tasks[0].tuned_configs) == 1

if __name__ == '__main__':
    test_selective()