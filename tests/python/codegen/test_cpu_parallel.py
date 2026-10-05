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
"""CPU worker ownership and synchronization across parallel loop phases."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm.s_tir
import tvm.testing
from tvm.script import tirx as T
from tvm.testing import env


@pytest.fixture
def run_team():
    # The runtime thread pool is host-thread local. Do not change another test's
    # team size, and propagate assertions/exceptions from the fresh host thread.
    with ThreadPoolExecutor(max_workers=1) as executor:

        def run(workers, function):
            def invoke():
                tvm.get_global_func("runtime.config_threadpool")(-1, workers)
                assert tvm.get_global_func("runtime.NumThreads")() == workers
                return function()

            return executor.submit(invoke).result()

        yield run


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize("workers", [1, 2, 4])
@pytest.mark.parametrize("n", [3, 17])
@pytest.mark.parametrize("policy", [None, False, True])
def test_iteration_ownership(run_team, workers, n, policy):
    annotations = {} if policy is None else {"parallel_stride_pattern": policy}
    owners = [None] * n

    @tvm.register_global_func("test.cpu_parallel.owner", override=True)
    def record(i):
        assert owners[i] is None
        owners[i] = threading.get_ident()

    @T.prim_func
    def kernel():
        for i in T.parallel(n, annotations=annotations):
            T.call_packed("test.cpu_parallel.owner", i)

    built = tvm.compile(kernel, target="llvm")
    run_team(workers, built)
    assert None not in owners
    chunk = (n + workers - 1) // workers
    expected = [i % workers if policy else i // chunk for i in range(n)]
    # Compare equivalence classes of worker identities, independent of OS IDs.
    for i in range(n):
        for j in range(n):
            assert (owners[i] == owners[j]) == (expected[i] == expected[j])


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize("workers", [2, 4])
@pytest.mark.parametrize("n", [3, 17])
def test_mixed_policies_and_barrier_communication(run_team, workers, n):
    owners = [[None] * n for _ in range(4)]
    events = []

    @tvm.register_global_func("test.cpu_parallel.phase", override=True)
    def record(phase, i):
        if i == 0:
            # Make other workers reach the following phase before this worker
            # unless the explicit barrier really synchronizes the team.
            time.sleep(0.01)
        owners[phase][i] = threading.get_ident()
        events.append(phase)

    @T.prim_func
    def kernel(A: T.Tensor((n,), "int32"), B: T.Tensor((n,), "int32"), bias: T.int32):
        with T.attr(0, "pragma_parallel_launch_point", 0):
            for i in T.parallel(n, annotations={"parallel_stride_pattern": True}):
                T.call_packed("test.cpu_parallel.phase", 0, i)
                A[i] = i + bias
            T.parallel_barrier()
            for i in T.parallel(n):
                T.call_packed("test.cpu_parallel.phase", 1, i)
                B[i] = A[n - i - 1] * 2
            T.parallel_barrier()
            for i in T.parallel(n, annotations={"parallel_stride_pattern": False}):
                T.call_packed("test.cpu_parallel.phase", 2, i)
                A[i] = B[n - i - 1] + 1
            T.parallel_barrier()
            for i in T.parallel(n, annotations={"parallel_stride_pattern": True}):
                T.call_packed("test.cpu_parallel.phase", 3, i)
                B[i] = A[n - i - 1] + 2
            T.parallel_barrier()
            T.parallel_barrier()

    built = tvm.compile(kernel, target="llvm")
    a = tvm.runtime.tensor(np.full(n, -1, dtype="int32"))
    b = tvm.runtime.tensor(np.full(n, -1, dtype="int32"))
    for bias in (7, 13):
        events.clear()
        run_team(workers, lambda: built(a, b, bias))
        assert events == [phase for phase in range(4) for _ in range(n)]
        np.testing.assert_array_equal(b.numpy(), 2 * (np.arange(n)[::-1] + bias) + 3)
        task_ids = owners[0][:workers]
        assert len(set(task_ids)) == min(workers, n)
        chunk = (n + workers - 1) // workers
        for phase in range(4):
            expected = [task_ids[i % workers if phase in (0, 3) else i // chunk] for i in range(n)]
            assert owners[phase] == expected


def test_barrier_script_roundtrip_and_effects():
    @T.prim_func
    def kernel(A: T.Tensor((7,), "int32")):
        with T.attr(0, "pragma_parallel_launch_point", 0):
            for i in T.parallel(7, annotations={"parallel_stride_pattern": True}):
                A[i] = i
            T.parallel_barrier()
            T.parallel_barrier()

    restored = tvm.script.from_source(kernel.script(), extra_vars={"T": T})
    tvm.ir.assert_structural_equal(kernel, restored, map_free_vars=True)
    op = tvm.ir.Op.get("tirx.parallel_barrier")
    assert op.get_attr("TCallEffectKind") == tvm.tirx.CallEffectKind.Opaque.value
    lowered = tvm.tirx.transform.RemoveNoOp()(tvm.IRModule.from_expr(kernel))
    calls = []
    tvm_ffi.structural_walk(
        lowered,
        lambda node: calls.append(node)
        if isinstance(node, tvm.ir.Call) and node.op == op
        else None,
    )
    assert len(calls) == 2


@pytest.mark.parametrize("n", [1, 7])
@pytest.mark.parametrize("policy", [False, True])
def test_opaque_lowering_preserves_policy(n, policy):
    @T.prim_func
    def kernel(A: T.Tensor((n,), "int32")):
        for i in T.parallel(n, annotations={"parallel_stride_pattern": policy}):
            A[i] = i

    mod = tvm.s_tir.transform.LowerOpaqueBlock()(tvm.IRModule.from_expr(kernel))
    loop = mod["kernel"].body
    assert isinstance(loop, tvm.tirx.For)
    assert loop.annotations["parallel_stride_pattern"] == policy
    assert isinstance(loop.body, tvm.tirx.BufferStore)


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_barrier_requires_team():
    @T.prim_func
    def kernel():
        T.parallel_barrier()

    with pytest.raises(RuntimeError, match="parallel_barrier requires a parallel launch"):
        tvm.compile(kernel, target="llvm")


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
def test_barrier_rejects_partitioned_loop():
    @T.prim_func
    def kernel(A: T.Tensor((7,), "int32")):
        for i in T.parallel(7):
            A[i] = i
            T.parallel_barrier()

    with pytest.raises(RuntimeError, match="parallel_barrier must be outside parallel loops"):
        tvm.compile(kernel, target="llvm")


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize("value", [1, 2, "true"])
def test_policy_requires_boolean(value):
    @T.prim_func
    def kernel(A: T.Tensor((7,), "int32")):
        for i in T.parallel(7, annotations={"parallel_stride_pattern": value}):
            A[i] = i

    with pytest.raises(ValueError, match="parallel_stride_pattern must be a constant boolean"):
        tvm.compile(kernel, target="llvm")


@pytest.mark.skipif(not env.has_llvm(), reason="need llvm")
@pytest.mark.parametrize(
    "pragma", ["pragma_parallel_stride_pattern", "pragma_parallel_barrier_when_finish"]
)
def test_retired_pragmas_require_explicit_migration(pragma):
    @T.prim_func
    def kernel(A: T.Tensor((7,), "int32")):
        with T.attr(0, "pragma_parallel_launch_point", 0):
            with T.attr(0, pragma, 1):
                for i in T.parallel(7):
                    A[i] = i
            for j in T.parallel(7):
                A[j] = A[j] + 1

    with pytest.raises(ValueError, match=pragma + " is retired") as error:
        tvm.compile(kernel, target="llvm")
    if pragma == "pragma_parallel_stride_pattern":
        assert "including later loops in the same launch" in str(error.value)
    else:
        assert "after the former attribute body" in str(error.value)


if __name__ == "__main__":
    tvm.testing.main()
