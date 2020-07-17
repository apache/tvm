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

""" Test measurement and log serialization. """

import tvm
import topi
from tvm import te, auto_scheduler
import tempfile

from test_auto_scheduler_common import get_tiled_matmul


def test_record():
    if not tvm.runtime.enabled("llvm"):
        return

    A = te.placeholder((512, 512), name='A')
    B = te.placeholder((512, 512), name='B')
    k = te.reduce_axis((0, 512), name='k')
    C = te.compute((512, 512), lambda i, j: te.sum(A[i][k] * B[k][j], axis=[k]), name='C')
    D = topi.nn.relu(C)
    k = te.reduce_axis((0, 512), name='k')
    E = te.compute((512, 512), lambda i, j: te.sum(A[i][k] * D[k][j], axis=[k]), name='C')
    F = topi.nn.relu(E)

    dag = auto_scheduler.ComputeDAG([A, B, F])
    s = dag.get_init_state()

    # Split
    its0 = s.split(C, s[C].iters[0], [4, 8, 8])
    its1 = s.split(C, s[C].iters[4], [8, 4, 4])
    # Reorder
    s.reorder(C, [its0[0], its1[0], its0[1], its1[1], its0[2], its1[2], its0[3], s[C].iters[8],
                  its1[3]])
    # Fuse
    s.fuse(C, [s[C].iters[0], s[C].iters[1], s[C].iters[2]])
    # Compute at
    s.split(F, s[F].iters[0], [2])
    s.compute_at(E, F, s[F].iters[0])
    # Compute inline
    s.compute_inline(D)
    # Compute root
    s.compute_root(D)
    # Parallel
    s.parallel(C, s[C].iters[0])
    # Thread bind
    s.bind(C, s[C].iters[1], "blockIdx.x")
    s.bind(C, s[C].iters[2], "threadIdx.y")
    s.bind(C, s[C].iters[3], "vthread")
    # Unroll
    s.unroll(C, s[C].iters[4])
    # Vectorize
    s.vectorize(C, s[C].iters[6])

    target = tvm.target.create("llvm")
    task = auto_scheduler.SearchTask(dag, "test", target)

    inp = auto_scheduler.measure.MeasureInput(task, s)
    res = auto_scheduler.measure.MeasureResult([0.1], 0, "", 0.2, 1)

    with tempfile.NamedTemporaryFile() as fp:
        auto_scheduler.save_records(fp.name, [inp], [res])

        log_reader = auto_scheduler.RecordReader(fp.name)
        inputs, results = log_reader.read_lines()
        assert len(inputs) == 1

        s1 = dag.infer_bound_from_state(s)
        s2 = dag.infer_bound_from_state(inputs[0].state)

        assert s1 == s2
        assert not (s1 == dag.get_init_state())


def test_measure_local_builder_runner():
    dag, s0 = get_tiled_matmul()

    if not tvm.runtime.enabled("llvm"):
        return
    tgt = tvm.target.create("llvm")
    task = auto_scheduler.SearchTask(dag, "test", tgt)

    minp = auto_scheduler.MeasureInput(task, s0)
    local_builder = auto_scheduler.LocalBuilder()
    local_runner = auto_scheduler.LocalRunner()

    bress = local_builder.build([minp])
    assert bress[0].error_no == 0
    mress = local_runner.run([minp], bress)
    assert mress[0].error_no == 0


if __name__ == "__main__":
    test_record()
    test_measure_local_builder_runner()
