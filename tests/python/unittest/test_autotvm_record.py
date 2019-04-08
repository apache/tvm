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
"""test the correctness of dump and load of data log"""
import time

import tvm
from tvm.contrib import util

from tvm import autotvm
from tvm.autotvm.measure import MeasureInput, MeasureResult, MeasureErrorNo
from tvm.autotvm.record import encode, decode, ApplyHistoryBest, measure_str_key

from test_autotvm_common import get_sample_task

def test_load_dump():
    task, target = get_sample_task()

    inp = MeasureInput(target, task, task.config_space.get(0))
    result = MeasureResult((2.0, 2.23, 0.23, 0.123, 0.234, 0.123), MeasureErrorNo.NO_ERROR,
                           2.3, time.time())

    for protocol in ['json', 'pickle']:
        row = encode(inp, result, protocol=protocol)
        inp_2, result_2 = decode(row, protocol=protocol)

        assert measure_str_key(inp) == measure_str_key(inp_2), \
            "%s vs %s" % (measure_str_key(inp), measure_str_key(inp_2))
        assert result.costs == result_2.costs
        assert result.error_no == result_2.error_no
        assert result.timestamp == result_2.timestamp


def test_file_io():
    temp = util.tempdir()
    file_path = temp.relpath("temp.log")

    tsk, target = get_sample_task()
    inputs = [MeasureInput(target, tsk, tsk.config_space.get(i)) for i in range(0, 10)]
    results = [MeasureResult((i, ), 0, 0, 0) for i in range(0, 10)]

    with open(file_path, "w") as fo:
        cb = autotvm.callback.log_to_file(fo)
        cb(None, inputs, results)

    ref = zip(inputs, results)
    for x, y in zip(ref, autotvm.record.load_from_file(file_path)):
        assert x[1] == y[1]


def test_apply_history_best():
    tsk, target = get_sample_task()

    records = [
        (MeasureInput(target, tsk, tsk.config_space.get(0)), MeasureResult((0.1,), 0, 2.3, 0)),
        (MeasureInput(target, tsk, tsk.config_space.get(1)), MeasureResult((0.3,), 0, 2.3, 0)),
        (MeasureInput(target, tsk, tsk.config_space.get(2)), MeasureResult((0.01,), 0, 2.3, 0)),
        (MeasureInput(target, tsk, tsk.config_space.get(4)), MeasureResult((0.4,), 0, 2.3, 0))
    ]
    hist_best = ApplyHistoryBest(records)
    x = hist_best.query(target, tsk.workload)
    assert str(x) == str(tsk.config_space.get(2))


if __name__ == "__main__":
    test_load_dump()
    test_apply_history_best()
    test_file_io()
