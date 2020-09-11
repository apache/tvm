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
from tvm import topi
from tvm import te, auto_scheduler
import tempfile
import tvm.testing

from test_auto_scheduler_common import matmul_auto_scheduler_test, get_tiled_matmul


def record_common(dag, s):
    target = tvm.target.Target("llvm")
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


def test_record_split_reorder_fuse_annotation():
    if not tvm.testing.device_enabled("llvm"):
        return

    A = te.placeholder((512, 512), name="A")
    B = te.placeholder((512, 512), name="B")
    k = te.reduce_axis((0, 512), name="k")
    C = te.compute((512, 512), lambda i, j: te.sum(A[i][k] * B[k][j], axis=[k]), name="C")

    dag = auto_scheduler.ComputeDAG([A, B, C])
    s = dag.get_init_state()

    # Split
    its0 = s.split(C, s[C].iters[0], [4, 8, 8])
    its1 = s.split(C, s[C].iters[4], [8, 4, 4])
    # Reorder
    s.reorder(
        C, [its0[0], its1[0], its0[1], its1[1], its0[2], its1[2], its0[3], s[C].iters[8], its1[3]]
    )
    # Fuse
    s.fuse(C, [s[C].iters[0], s[C].iters[1], s[C].iters[2]])
    # Parallel
    s.parallel(C, s[C].iters[0])
    # Thread bind(The blockIdx & threadIdx are used in GPU, just for record testing here)
    s.bind(C, s[C].iters[1], "blockIdx.x")
    s.bind(C, s[C].iters[2], "threadIdx.z")
    s.bind(C, s[C].iters[3], "vthread")
    # Unroll
    s.unroll(C, s[C].iters[4])
    # Vectorize
    s.vectorize(C, s[C].iters[6])

    record_common(dag, s)


def test_record_compute_at_root_inline_cache_read_write():
    if not tvm.testing.device_enabled("llvm"):
        return

    A = te.placeholder((512, 512), name="A")
    AA = topi.nn.relu(A)
    B = te.placeholder((512, 512), name="B")
    k = te.reduce_axis((0, 512), name="k")
    C = te.compute((512, 512), lambda i, j: te.sum(AA[i][k] * B[k][j], axis=[k]), name="C")

    dag = auto_scheduler.ComputeDAG([A, B, C])
    s = dag.get_init_state()

    # Cache Write
    C_shared = s.cache_write(C, "shared")
    # Compute At
    s.compute_at(C_shared, C, s[C].iters[0])
    # Cache Read
    B_global = s.cache_read(B, "global", [C_shared])
    s.compute_at(B_global, C_shared, s[C_shared].iters[2])
    # Compute Inline
    s.compute_inline(AA)
    # Compute Root
    s.compute_root(C_shared)

    record_common(dag, s)


def test_record_follow_split_follow_fused_split():
    if not tvm.testing.device_enabled("llvm"):
        return

    A = te.placeholder((512, 512), name="A")
    B = te.placeholder((512, 512), name="B")
    k = te.reduce_axis((0, 512), name="k")
    C = te.compute((512, 512), lambda i, j: te.sum(A[i][k] * B[k][j], axis=[k]), name="C")
    D = topi.nn.relu(C)
    E = topi.nn.relu(D)

    dag = auto_scheduler.ComputeDAG([A, B, E])
    s = dag.get_init_state()

    # Follow Split
    s.split(C, s[C].iters[0], [4, 2, 8, 4], True)
    split_step0 = len(s.transform_steps) - 1
    s.follow_split(C, s[C].iters[5], split_step0, 4)
    # Follow Fused Split
    its0 = s.split(E, s[E].iters[0], [4, 2, 8, 4], True)
    split_step1 = len(s.transform_steps) - 1
    its1 = s.split(E, s[E].iters[5], [2, 4, 2, 4], True)
    split_step2 = len(s.transform_steps) - 1
    its = []
    for i0, i1 in zip(its0, its1):
        its.append(i0)
        its.append(i1)
    for i in range(0, 5):
        s.fuse(E, [s[E].iters[i], s[E].iters[i + 1]])
    s.follow_fused_split(D, s[D].iters[0], [split_step1, split_step2], 2, True)

    record_common(dag, s)


def test_record_pragma_storage_align_rfactor():
    if not tvm.testing.device_enabled("llvm"):
        return

    A = te.placeholder((512, 512), name="A")
    B = te.placeholder((512, 512), name="B")
    k = te.reduce_axis((0, 512), name="k")
    C = te.compute((512, 512), lambda i, j: te.sum(A[i][k] * B[k][j], axis=[k]), name="C")

    dag = auto_scheduler.ComputeDAG([A, B, C])
    s = dag.get_init_state()

    # Rfactor
    ko, _ = s.split(C, s[C].iters[2], [16])
    s.rfactor(C, ko, 2)
    # Pragma
    s.pragma(C, s[C].iters[0], "auto_unroll_max_step$64")
    # StorageAlign
    s.storage_align(C, s[C].iters[-1], 8, 4)

    record_common(dag, s)


def test_measure_local_builder_runner(enable_cpu_cache_flush=False):
    if not tvm.testing.device_enabled("llvm"):
        return

    dag, s0 = get_tiled_matmul()
    tgt = tvm.target.Target("llvm")
    task = auto_scheduler.SearchTask(dag, "test", tgt)

    minp = auto_scheduler.MeasureInput(task, s0)
    local_builder = auto_scheduler.LocalBuilder()
    local_runner = auto_scheduler.LocalRunner(
        timeout=60, enable_cpu_cache_flush=enable_cpu_cache_flush
    )

    bress = local_builder.build([minp])
    assert bress[0].error_no == 0
    mress = local_runner.run([minp], bress)
    assert mress[0].error_no == 0


def test_measure_local_builder_rpc_runner(enable_cpu_cache_flush=False):
    if not tvm.testing.device_enabled("llvm"):
        return

    dag, s0 = get_tiled_matmul()
    tgt = tvm.target.Target("llvm")
    task = auto_scheduler.SearchTask(dag, "test", tgt)

    minp = auto_scheduler.MeasureInput(task, s0)
    local_builder = auto_scheduler.LocalBuilder()
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(
        timeout=60, enable_cpu_cache_flush=enable_cpu_cache_flush
    )
    rpc_runner = measure_ctx.runner

    bress = local_builder.build([minp])
    assert bress[0].error_no == 0
    mress = rpc_runner.run([minp], bress)
    assert mress[0].error_no == 0


if __name__ == "__main__":
    test_record_split_reorder_fuse_annotation()
    test_record_compute_at_root_inline_cache_read_write()
    test_record_follow_split_follow_fused_split()
    test_record_pragma_storage_align_rfactor()
    test_measure_local_builder_runner(enable_cpu_cache_flush=True)
    test_measure_local_builder_runner(enable_cpu_cache_flush=False)
    test_measure_local_builder_rpc_runner(enable_cpu_cache_flush=True)
    test_measure_local_builder_rpc_runner(enable_cpu_cache_flush=False)
