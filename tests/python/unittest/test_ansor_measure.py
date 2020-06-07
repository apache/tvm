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

"""Test measurement and log serialization"""

import tvm
from tvm import ansor
import tempfile

from test_ansor_common import get_tiled_matmul


def test_serialization():
    dag, s = get_tiled_matmul()
    target = tvm.target.create("llvm")
    task = ansor.SearchTask(dag, "test", target)

    inp = ansor.measure.MeasureInput(task, s)
    res = ansor.measure.MeasureResult([0.1], 0, "", 0.2, 1)

    with tempfile.NamedTemporaryFile() as fp:
        ansor.serialization.write_measure_records_to_file(fp.name, [inp], [res])

        log_reader = ansor.serialization.LogReader(fp.name)
        inputs, results = log_reader.read_lines()
        assert len(inputs) == 1

        s1 = dag.infer_bound_from_state(s)
        s2 = dag.infer_bound_from_state(inputs[0].state)

        assert s1 == s2
        assert not (s1 == dag.get_init_state().state_object)


def test_measure_local_builder_runner():
    dag, s0 = get_tiled_matmul()

    tgt = tvm.target.create("llvm")
    task = ansor.SearchTask(dag, "test", tgt)

    minp = ansor.MeasureInput(task, s0)
    local_builder = ansor.LocalBuilder()
    local_runner = ansor.LocalRunner()

    bress = local_builder.build([minp])
    assert bress[0].error_no == 0
    mress = local_runner.run([minp], bress)
    assert mress[0].error_no == 0


if __name__ == "__main__":
    test_serialization()
    test_measure_local_builder_runner()
