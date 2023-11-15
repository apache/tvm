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
import json

import multiprocessing
import numpy as np
import tvm
from tvm import topi
from tvm import te, auto_scheduler
import tempfile
import tvm.testing
import pickle
from tvm.testing.auto_scheduler import matmul_auto_scheduler_test
from tvm.auto_scheduler import workload_registry


def record_common(dag, s):
    target = tvm.target.Target("llvm")
    task = auto_scheduler.SearchTask(compute_dag=dag, workload_key="test", target=target)

    inp = auto_scheduler.measure.MeasureInput(task, s)
    res = auto_scheduler.measure.MeasureResult([0.1], 0, "", 0.2, 1)

    # Test in-memory record processing.
    record_str = auto_scheduler.measure_record.dump_record_to_string(inp, res)
    r_inp, r_res = auto_scheduler.measure_record.load_record_from_string(record_str)
    # Only check the workload_key for simplification.
    assert inp.task.workload_key == r_inp.task.workload_key
    assert str(res) == str(r_res)

    # Test file-based record processing.
    with tempfile.NamedTemporaryFile() as fp:
        auto_scheduler.save_records(fp.name, [inp], [res])

        log_reader = auto_scheduler.RecordReader(fp.name)
        inputs, _ = log_reader.read_lines()
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


def test_recover_measure_input():
    task = auto_scheduler.SearchTask(
        func=matmul_auto_scheduler_test, args=(512, 512, 512), target="llvm"
    )

    inp = auto_scheduler.measure.MeasureInput(task, task.compute_dag.init_state)
    res = auto_scheduler.measure.MeasureResult([0.1], 0, "", 0.2, 1)

    with tempfile.NamedTemporaryFile() as fp:
        auto_scheduler.save_records(fp.name, [inp], [res])

        log_reader = auto_scheduler.RecordReader(fp.name)
        inputs, _ = log_reader.read_lines()
        assert len(inputs) == 1

        raw_inp = inputs[0]

        correct_inp = auto_scheduler.measure.recover_measure_input(raw_inp)
        assert str(correct_inp.task.compute_dag) == str(inp.task.compute_dag)

        correct_inp = auto_scheduler.measure.recover_measure_input(raw_inp, rebuild_state=True)
        assert str(correct_inp.state) == str(inp.state)


def test_workload_dis_factor():
    calc = auto_scheduler.utils.calc_workload_dis_factor
    decode = auto_scheduler.utils.decode_workload_key

    # Identical
    target_wkl_key = json.dumps(
        ["func1", [8, 3, 224, 224], [32, 3, 3, 3], [0, 0], [1, 1], "float32"]
    )
    assert calc(decode(target_wkl_key), decode(target_wkl_key)) == 1

    # Compatible with a factor
    wkl_key = json.dumps(["func1", [1, 3, 112, 112], [32, 3, 3, 3], [0, 0], [1, 1], "float32"])
    assert calc(decode(target_wkl_key), decode(wkl_key)) == 8 * 2 * 2

    # Incompatible argument with zeros
    wkl_key = json.dumps(["func1", [8, 3, 224, 224], [32, 3, 3, 3], [1, 1], [1, 1], "float32"])
    assert calc(decode(target_wkl_key), decode(wkl_key)) == float("inf")
    wkl_key = json.dumps(["func1", [8, 3, 224, 224], [32, 3, 3, 3], [0, 0], [0, 0], "float32"])
    assert calc(decode(target_wkl_key), decode(wkl_key)) == float("inf")

    # Incompatible non-integter argument
    wkl_key = json.dumps(["func1", [8, 3, 224, 224], [32, 3, 3, 3], [0, 0], [1, 1], "int8"])
    assert calc(decode(target_wkl_key), decode(wkl_key)) == float("inf")

    # Incompatible function
    wkl_key = json.dumps(["func2", [8, 3, 224, 224], [32, 3, 3, 3], [0, 0], [1, 1], "float32"])
    assert calc(decode(target_wkl_key), decode(wkl_key)) == float("inf")

    # Incompatible due to non-dividable factor
    wkl_key = json.dumps(["func1", [8, 3, 223, 223], [32, 3, 3, 3], [0, 0], [1, 1], "float32"])
    assert calc(decode(target_wkl_key), decode(wkl_key)) == float("inf")


def test_measure_local_builder_runner():
    if not tvm.testing.device_enabled("llvm"):
        return

    task = auto_scheduler.SearchTask(
        func=matmul_auto_scheduler_test, args=(512, 512, 512), target="llvm"
    )

    for enable_cpu_cache_flush in [True, False]:
        minp = auto_scheduler.MeasureInput(task, task.compute_dag.init_state)
        local_builder = auto_scheduler.LocalBuilder()
        local_runner = auto_scheduler.LocalRunner(
            timeout=60, enable_cpu_cache_flush=enable_cpu_cache_flush
        )

        bress = local_builder.build([minp])
        assert bress[0].error_no == 0
        mress = local_runner.run([minp], bress)
        assert mress[0].error_no == 0


def test_dag_measure_local_builder_runner():
    if not tvm.testing.device_enabled("llvm"):
        return

    A = te.placeholder((512, 512), name="A")
    B = te.placeholder((512, 512), name="B")
    k = te.reduce_axis((0, 512), name="k")
    C = te.compute((512, 512), lambda i, j: te.sum(A[i][k] * B[k][j], axis=[k]), name="C")
    D = topi.nn.relu(C)
    E = topi.nn.relu(D)

    tensors = [A, B, E]
    dag = auto_scheduler.ComputeDAG(tensors)
    key = workload_registry.register_workload_tensors(dag.workload_key(), tensors)
    transfer_data = workload_registry.serialize_workload_registry_entry(key)
    f_data = pickle.dumps(transfer_data)
    f_new = pickle.loads(f_data)
    del workload_registry.WORKLOAD_FUNC_REGISTRY[key]
    workload_registry.deserialize_workload_registry_entry(f_new)

    target = tvm.target.Target("llvm")
    task = auto_scheduler.SearchTask(compute_dag=dag, workload_key=key, target=target)

    for enable_cpu_cache_flush in [True, False]:
        minp = auto_scheduler.MeasureInput(task, task.compute_dag.init_state)
        local_builder = auto_scheduler.LocalBuilder()
        local_runner = auto_scheduler.LocalRunner(
            timeout=60, enable_cpu_cache_flush=enable_cpu_cache_flush
        )

        bress = local_builder.build([minp])
        assert bress[0].error_no == 0
        mress = local_runner.run([minp], bress)
        assert mress[0].error_no == 0


def test_workload_serialization():
    key = tvm.auto_scheduler.utils.get_func_name(matmul_auto_scheduler_test)
    transfer_data = workload_registry.serialize_workload_registry_entry(key)
    f_data = pickle.dumps(transfer_data)
    f_new = pickle.loads(f_data)
    del workload_registry.WORKLOAD_FUNC_REGISTRY[key]
    workload_registry.deserialize_workload_registry_entry(f_new)


def test_measure_local_builder_rpc_runner():
    if not tvm.testing.device_enabled("llvm"):
        return

    task = auto_scheduler.SearchTask(
        func=matmul_auto_scheduler_test, args=(512, 512, 512), target="llvm"
    )

    for enable_cpu_cache_flush in [True, False]:
        minp = auto_scheduler.MeasureInput(task, task.compute_dag.init_state)
        local_builder = auto_scheduler.LocalBuilder()
        measure_ctx = auto_scheduler.LocalRPCMeasureContext(
            timeout=60, enable_cpu_cache_flush=enable_cpu_cache_flush
        )
        rpc_runner = measure_ctx.runner

        bress = local_builder.build([minp])
        assert bress[0].error_no == 0
        mress = rpc_runner.run([minp], bress)
        assert mress[0].error_no == 0

        del measure_ctx


def measure_local_builder_rpc_runner_spawn():
    assert multiprocessing.get_start_method(False) == "spawn"
    test_measure_local_builder_rpc_runner()


@tvm.testing.requires_llvm
def test_measure_local_builder_rpc_runner_spawn():
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=measure_local_builder_rpc_runner_spawn)
    p.start()
    p.join()


@tvm.testing.requires_llvm
def test_measure_target_host():
    task = auto_scheduler.SearchTask(
        func=matmul_auto_scheduler_test,
        args=(512, 512, 512),
        target=tvm.target.Target("llvm", "llvm -mtriple=aarch64-linux-gnu"),
    )

    inp = auto_scheduler.measure.MeasureInput(task, task.compute_dag.init_state)
    res = auto_scheduler.measure.MeasureResult([0.1], 0, "", 0.2, 1)

    with tempfile.NamedTemporaryFile() as fp:
        auto_scheduler.save_records(fp.name, [inp], [res])

        log_reader = auto_scheduler.RecordReader(fp.name)
        inputs, _ = log_reader.read_lines()
        assert len(inputs) == 1

        raw_inp = inputs[0]

        recovered_inp = auto_scheduler.measure.recover_measure_input(raw_inp)
        assert str(recovered_inp.task.target.host) == str(inp.task.target.host)


@tvm.testing.requires_llvm
def test_measure_special_inputs_map_by_name_local_runner():
    @auto_scheduler.register_workload
    def foo():
        X = te.placeholder(shape=[10], dtype="int32")
        Index = te.placeholder(shape=[1], dtype="int32", name="Index")
        Y = te.compute((1,), lambda i: X[Index[i]])
        return [X, Index, Y]

    # This workload cannot use random input for the `Index` input
    task = auto_scheduler.SearchTask(
        func=foo,
        target="llvm",
        task_inputs={
            "Index": tvm.nd.array(np.array([5], dtype="int32")),
        },
    )

    minp = auto_scheduler.MeasureInput(task, task.compute_dag.init_state)
    local_builder = auto_scheduler.LocalBuilder()
    local_runner = auto_scheduler.LocalRunner(timeout=10)

    bress = local_builder.build([minp])
    assert bress[0].error_no == 0
    mress = local_runner.run([minp], bress)
    assert mress[0].error_no == 0


@tvm.testing.requires_llvm
def test_measure_special_inputs_map_by_name_rpc_runner():
    @auto_scheduler.register_workload
    def foo():
        X = te.placeholder(shape=[10], dtype="int32")
        Index = te.placeholder(shape=[1], dtype="int32", name="Index")
        Y = te.compute((1,), lambda i: X[Index[i]])
        return [X, Index, Y]

    # This workload cannot use random input for the `Index` input
    task = auto_scheduler.SearchTask(
        func=foo,
        target="llvm",
        task_inputs={
            "Index": tvm.nd.array(np.array([5], dtype="int32")),
        },
    )

    for enable_cpu_cache_flush in [True, False]:
        minp = auto_scheduler.MeasureInput(task, task.compute_dag.init_state)
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
    tvm.testing.main()
