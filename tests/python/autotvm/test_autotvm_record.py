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
from io import StringIO
from os import PathLike
import time

from tvm.contrib import utils

from tvm import autotvm
from tvm.autotvm.measure import MeasureInput, MeasureResult, MeasureErrorNo
from tvm.autotvm.record import encode, decode, ApplyHistoryBest, measure_str_key

from tvm.testing.autotvm import get_sample_task


def test_load_dump():
    task, target = get_sample_task()

    inp = MeasureInput(target, task, task.config_space.get(0))
    result = MeasureResult(
        (2.0, 2.23, 0.23, 0.123, 0.234, 0.123), MeasureErrorNo.NO_ERROR, 2.3, time.time()
    )

    for protocol in ["json", "pickle"]:
        row = encode(inp, result, protocol=protocol)
        inp_2, result_2 = decode(row, protocol=protocol)

        assert measure_str_key(inp) == measure_str_key(inp_2), "%s vs %s" % (
            measure_str_key(inp),
            measure_str_key(inp_2),
        )
        assert result.costs == result_2.costs
        assert result.error_no == result_2.error_no
        assert result.timestamp == result_2.timestamp


def test_file_io():
    temp = utils.tempdir()
    file_path = temp.relpath("temp.log")

    tsk, target = get_sample_task()
    inputs = [MeasureInput(target, tsk, tsk.config_space.get(i)) for i in range(0, 10)]
    results = [MeasureResult((i,), 0, 0, 0) for i in range(0, 10)]

    invalid_inp = MeasureInput(target, tsk, tsk.config_space.get(10))
    invalid_res = MeasureResult((10,), 0, 0, 0)

    # Erase the entity map to test if it will be ignored when loading back.
    invalid_inp.config._entity_map = {}

    with open(file_path, "w") as fo:
        cb = autotvm.callback.log_to_file(fo)
        cb(None, inputs, results)
        cb(None, [invalid_inp], [invalid_res])

    ref = zip(inputs, results)
    for x, y in zip(ref, autotvm.record.load_from_file(file_path)):
        assert x[1] == y[1]

    # Confirm functionality of multiple file loads
    hist_best = ApplyHistoryBest([file_path, file_path])
    x = hist_best.query(target, tsk.workload)
    assert str(x) == str(inputs[0][2])


def test_apply_history_best(tmpdir):
    tsk, target = get_sample_task()
    best = str(tsk.config_space.get(2))

    inputs_batch_1 = [MeasureInput(target, tsk, tsk.config_space.get(i)) for i in range(3)]
    results_batch_1 = [MeasureResult((i,), 0, 0, 0) for i in range(1, 3)]
    results_batch_1.append(MeasureResult((0.5,), 0, 2.3, 0))

    # Write data out to file
    filepath_batch_1 = tmpdir / "batch_1.log"
    with open(filepath_batch_1, "w") as file:
        autotvm.callback.log_to_file(file)(None, inputs_batch_1, results_batch_1)

    # Load best results from Path
    assert isinstance(filepath_batch_1, PathLike)
    hist_best = ApplyHistoryBest(filepath_batch_1)
    assert str(hist_best.query(target, tsk.workload)) == best

    # Load best results from str(Path)
    hist_best = ApplyHistoryBest(str(filepath_batch_1))
    assert str(hist_best.query(target, tsk.workload)) == best

    # Write data into StringIO buffer
    stringio_batch_1 = StringIO()
    assert isinstance(filepath_batch_1, PathLike)
    callback = autotvm.callback.log_to_file(stringio_batch_1)
    callback(None, inputs_batch_1, results_batch_1)
    stringio_batch_1.seek(0)

    # Load best results from strIO
    hist_best = ApplyHistoryBest(stringio_batch_1)
    assert str(hist_best.query(target, tsk.workload)) == best

    # Load best result from list of tuples (MeasureInput, MeasureResult)
    hist_best = ApplyHistoryBest(list(zip(inputs_batch_1, results_batch_1)))
    assert str(hist_best.query(target, tsk.workload)) == best

    # Same thing, but iterable instead of list (i.e. no subscripting)
    hist_best = ApplyHistoryBest(zip(inputs_batch_1, results_batch_1))
    assert str(hist_best.query(target, tsk.workload)) == best


def test_apply_history_best_multiple_batches(tmpdir):
    tsk, target = get_sample_task()
    best = str(tsk.config_space.get(2))

    inputs_batch_1 = [MeasureInput(target, tsk, tsk.config_space.get(i)) for i in range(2)]
    results_batch_1 = [MeasureResult((i,), 0, 0, 0) for i in range(1, 3)]
    filepath_batch_1 = tmpdir / "batch_1.log"
    with open(filepath_batch_1, "w") as file:
        autotvm.callback.log_to_file(file)(None, inputs_batch_1, results_batch_1)

    inputs_batch_2 = [MeasureInput(target, tsk, tsk.config_space.get(i)) for i in range(2, 4)]
    results_batch_2 = [MeasureResult((0.5,), 0, 0, 0), MeasureResult((3,), 0, 0, 0)]
    filepath_batch_2 = tmpdir / "batch_2.log"
    with open(filepath_batch_2, "w") as file:
        autotvm.callback.log_to_file(file)(None, inputs_batch_2, results_batch_2)

    # Check two Path filepaths works
    hist_best = ApplyHistoryBest([filepath_batch_1, filepath_batch_2])
    assert str(hist_best.query(target, tsk.workload)) == best

    # Check that an arbitrary Iterable of Paths works
    # Calling zip() on a single list gives a non-subscriptable Iterable
    hist_best = ApplyHistoryBest(zip([filepath_batch_1, filepath_batch_2]))
    assert str(hist_best.query(target, tsk.workload)) == best

    # Check that Iterable of Iterable of tuples is correctly merged
    hist_best = ApplyHistoryBest(
        zip(
            [
                zip(inputs_batch_1, results_batch_1),
                zip(inputs_batch_2, results_batch_2),
            ]
        )
    )
    assert str(hist_best.query(target, tsk.workload)) == best


if __name__ == "__main__":
    test_load_dump()
    test_apply_history_best()
    test_file_io()
