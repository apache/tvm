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
import pytest

from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.analysis import verify_tirx_well_formed as verify


def test_root_scope():
    # fmt: off
    @T.prim_func(check_well_formed=False)
    def test1() -> None:
        T.device_entry()
        pass

    @T.prim_func(check_well_formed=False)
    def test2() -> None:
        pass

    @T.prim_func(check_well_formed=False)
    def test3() -> None:
        pass

    @T.prim_func(check_well_formed=False)
    def test4() -> None:
        T.device_entry()
        pass

        # fmt: on

    verify(test1)
    verify(test2)
    verify(test3)
    verify(test4)


def test_nested_scope():
    # fmt: off
    @T.prim_func(check_well_formed=False)
    def test1() -> None:
        T.device_entry()
        pass
        pass

    @T.prim_func(check_well_formed=False)
    def test2() -> None:
        T.device_entry()
        pass

    @T.prim_func(check_well_formed=False)
    def test3() -> None:
        T.device_entry()
        pass
    @T.prim_func(check_well_formed=False)
    def test4() -> None:
        T.device_entry()
        pass
        pass

        # fmt: on

    verify(test1)
    verify(test2)
    verify(test3)
    verify(test4)


def test_scope_id_consistency():
    # fmt: off
    @T.prim_func(check_well_formed=False)
    def test1():
        T.device_entry()
        T.cta_id([32])
        T.warp_id([4])
        T.lane_id([32])
        pass

    @T.prim_func(check_well_formed=False)
    def test2():
        T.device_entry()
        T.cta_id([32])
        T.warp_id([4])
        T.lane_id([32])
        T.thread_id([128])
        pass

    @T.prim_func(check_well_formed=False)
    def test3():
        T.device_entry()
        T.cta_id([32])
        T.warp_id([2])
        T.lane_id([32])
        T.thread_id([128])
        pass

    @T.prim_func(check_well_formed=False)
    def test4():
        T.device_entry()
        bx, by, bz = T.cta_id([8, 10, 12])
        cbx, cby, cbz = T.cta_id_in_cluster([2, 2, 1])
        clx, cly, clz = T.cluster_id([4, 5, 12])
        T.evaluate(bx + by + bz)
        T.evaluate(cbx + cby + cbz)
        T.evaluate(clx + cly + clz)

    @T.prim_func(check_well_formed=False)
    def test5():
        T.device_entry()
        bx, by, bz = T.cta_id([8, 10, 12])
        cbx, cby, cbz = T.cta_id_in_cluster([2, 2, 1])
        clx, cly, clz = T.cluster_id([3, 5, 12])
        T.evaluate(bx + by + bz)
        T.evaluate(cbx + cby + cbz)
        T.evaluate(clx + cly + clz)

    @T.prim_func(check_well_formed=False)
    def test6():
        T.device_entry()
        clx, cly, clz = T.cluster_id([4, 5, 12])
        bx, by, bz = T.cta_id([8, 10, 12])
        cbx, cby, cbz = T.cta_id_in_cluster([2, 2, 1])
        T.evaluate(bx + by + bz)
        T.evaluate(cbx + cby + cbz)
        T.evaluate(clx + cly + clz)

    @T.prim_func(check_well_formed=False)
    def test7():
        T.device_entry()
        clx, cly, clz = T.cluster_id([3, 5, 12])
        bx, by, bz = T.cta_id([8, 10, 12])
        cbx, cby, cbz = T.cta_id_in_cluster([2, 2, 1])
        T.evaluate(bx + by + bz)
        T.evaluate(cbx + cby + cbz)
        T.evaluate(clx + cly + clz)

        # fmt: on

    verify(test1)
    verify(test2)
    with pytest.raises(Exception, match="Inconsistent extents for scope"):
        verify(test3)
    verify(test4)
    with pytest.raises(Exception, match="Inconsistent extents|non-divisible extents"):
        verify(test5)
    verify(test6)
    with pytest.raises(Exception, match="Inconsistent extents|non-divisible extents"):
        verify(test7)


def test_layout():
    ### TileLayout
    # fmt: off
    @T.prim_func(check_well_formed=False)
    def test1():
        T.device_entry()
        T.cta_id([32])
        T.warp_id([4])
        T.lane_id([32])
        A = T.alloc_buffer((2,), layout=T.TileLayout(T.S[2, 1]))

        A[0] = 0
        # fmt: on
    verify(test1)

    ### SwizzleLayout
    # fmt: off
    @T.prim_func(check_well_formed=False)
    def test2():
        T.device_entry()
        T.cta_id([32])
        T.warp_id([4])
        T.lane_id([32])
        A = T.alloc_buffer((512,), scope="shared", layout=T.SwizzleLayout(3, 3, 3))

        A[0] = 0
        # fmt: on
    verify(test2)


def test_host():
    # fmt: off
    @T.prim_func(check_well_formed=False)
    def test1(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (16, 16), dtype="float32", align=16)

        A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float32", 2, A.data, 16, 16, 64, 16, 16, 1, 1, 0, 0, 0, 0)  # noqa: E501

        T.device_entry()
        for blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                bar = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
                phase = T.alloc_buffer((1,), "int32", scope="local")
                A_smem = T.alloc_buffer((16, 16), "float32", scope="shared", align=128)

                phase[0] = 0
                if threadIdx == 0:
                    T.ptx.mbarrier.init(bar.data, 1)
                    T.ptx.fence.proxy_async("shared::cta")
                    T.ptx.cp_async.bulk.tensor.g2c(2, A_smem.data, bar.data, T.address_of(A_map), 0, 1, "", 0, 0)  # noqa: E501
                    T.ptx.mbarrier.arrive.expect_tx(bar.data, 16*16*4)
                T.ptx.mbarrier.try_wait(bar.data, phase[0])
                phase[0] = phase[0] ^ 1
                T.print_buffer(A_smem.data, "float32", False, False, 2, 16*16)
        # fmt: on
    verify(test1)


def test_device_func():
    # Per-call exec-scope migration: scope is now attached per op via the
    # ``T.op[scope](...)`` subscription surface instead of a ``with T.cta():``
    # region. ``test1`` exercises a per-call-scoped op; ``test2`` the plain
    # (unscoped) op. The old multi-root-scope negative case asserted the removed
    # "only one root scope" verifier rule and no longer has an equivalent, so it
    # is dropped.
    # fmt: off
    @T.prim_func(check_well_formed=False)
    def test1(A: T.Buffer((128,), "float32")):
        T.device_entry()
        T.cta_id([1])
        T.thread_id([128])
        Tx.cta.fill(A, 0.)

    @T.prim_func(check_well_formed=False)
    def test2(A: T.Buffer((128,), "float32")):
        T.device_entry()
        T.cta_id([128])
        T.thread_id([128])
        Tx.fill(A, 0.)
    # fmt: on
    verify(test1, device_func=True)
    verify(test2, device_func=True)


def test_preferred_cluster_validation():
    # fmt: off
    # Valid: cluster→cta with preferred_extents matching size
    @T.prim_func(check_well_formed=False)
    def test1() -> None:
        T.device_entry()
        cbx, cby = T.cta_id_in_cluster([2, 1], preferred=[2, 2])
        tx = T.thread_id([128])
        T.evaluate(cbx + cby + tx)

        # Invalid: preferred size doesn't match extents size (caught at verify time)
    @T.prim_func(check_well_formed=False)
    def test2() -> None:
        T.device_entry()
        cbx, cby = T.cta_id_in_cluster([2, 1], preferred=[2])
        tx = T.thread_id([128])
        T.evaluate(cbx + cby + tx)
        # fmt: on

    verify(test1)
    with pytest.raises(Exception, match="preferred_extents must have the same size"):
        verify(test2)

    # Invalid: preferred on a non-cluster→cta scope (caught at IR build time)
    with pytest.raises(Exception):
        # fmt: off
        @T.prim_func(check_well_formed=False)
        def test3() -> None:
            T.device_entry()
            bx = T.cta_id([128], preferred=[256])
            tx = T.thread_id([128])
            T.evaluate(bx + tx)
            # fmt: on


def test_scope_id_deferred_relaxed_at_construction():
    """Deferred scope_id (no extents) must pass the well-formed check even when
    no sibling provides enough info to resolve it -- strict resolution is
    deferred to LowerTIRx."""

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def partial_only_cta():
        T.device_entry()
        bx = T.cta_id()           # deferred kernel→cta, no closure source
        tx = T.thread_id([128])   # explicit
        T.evaluate(bx + tx)

    @T.prim_func(check_well_formed=False)
    def all_deferred():
        T.device_entry()
        bx = T.cta_id()
        wg = T.warpgroup_id()
        warp = T.warp_id_in_wg()
        lane = T.lane_id()
        T.evaluate(bx + wg + warp + lane)

    @T.prim_func(check_well_formed=False)
    def mixed():
        T.device_entry()
                # kCtaWarp=4, kWarpThread=32 → kCtaThread=128 derivable.
        T.warp_id([4])
        T.lane_id([32])
        T.thread_id()             # deferred kCtaThread, resolvable via closure
        pass
        # fmt: on

        # All three accepted by well-formed: deferred extents are tolerated.
    verify(partial_only_cta)
    verify(all_deferred)
    verify(mixed)


def test_scope_id_deferred_consistency_still_enforced():
    """Even with deferred defs, known-known consistency between sibling defs
    must still be enforced by the closure check."""

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def inconsistent():
        # 4 warps * 32 lanes = 128 threads, but explicit thread_id says 64 -> error.
        T.device_entry()
        T.cta_id([32])
        T.warp_id([4])
        T.lane_id([32])
        T.thread_id()       # deferred (shouldn't shadow the conflict)
        T.thread_id([64])   # conflicts with derived kCtaThread=128
        pass
        # fmt: on

    with pytest.raises(Exception, match="Inconsistent extents for scope"):
        verify(inconsistent)


def test_scope_id_deferred_multi_var_rejected():
    """Deferred form (no extents) requires exactly one Var. Multi-var defers
    have no well-defined recovery from fused closure values."""

    # The C++ ScopeIdDef ctor enforces this; constructing such a def from the
    # parser path is not currently expressible (parser only emits single-Var
    # deferred), but we exercise the FFI-level guard directly.
    from tvm.tirx.exec_scope import ScopeIdDef
    from tvm.tirx.expr import Var

    # Single-Var deferred form is fine.
    ScopeIdDef([Var("", "int32")], None, "kernel", "cta")

    # Two-Var deferred should be rejected.
    with pytest.raises(Exception, match="Deferred ScopeIdDef.*must define exactly one Var"):
        ScopeIdDef([Var("", "int32"), Var("", "int32")], None, "kernel", "cta")


if __name__ == "__main__":
    test_root_scope()
    test_nested_scope()
    test_scope_id_consistency()
    test_layout()
    test_host()
    test_device_func()
    test_scope_id_deferred_relaxed_at_construction()
    test_scope_id_deferred_consistency_still_enforced()
    test_scope_id_deferred_multi_var_rejected()
