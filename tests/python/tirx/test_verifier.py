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

from tvm.script import tirx as Tx
from tvm.tirx.analysis import verify_tirx_well_formed as verify


def test_root_scope():
    # fmt: off
    @Tx.prim_func(check_well_formed=False)
    def test1() -> None:
        with Tx.thread():
            pass

    @Tx.prim_func(check_well_formed=False)
    def test2() -> None:
        with Tx.warp():
            with Tx.thread():
                pass

    @Tx.prim_func(check_well_formed=False)
    def test3() -> None:
        with Tx.cta():
            with Tx.warp():
                with Tx.thread():
                    pass

    @Tx.prim_func(check_well_formed=False)
    def test4() -> None:
        with Tx.kernel():
            with Tx.cta():
                with Tx.warp():
                    with Tx.thread():
                        pass

    # fmt: on

    verify(test1)
    verify(test2)
    verify(test3)
    verify(test4)


def test_nested_scope():
    # fmt: off
    @Tx.prim_func(check_well_formed=False)
    def test1() -> None:
        with Tx.kernel():
            with Tx.cta():
                with Tx.warp():
                    with Tx.thread():
                        pass
                with Tx.thread():
                    pass

    @Tx.prim_func(check_well_formed=False)
    def test2() -> None:
        with Tx.kernel():
            with Tx.thread():
                with Tx.cta():
                    with Tx.thread():
                        pass

    @Tx.prim_func(check_well_formed=False)
    def test3() -> None:
        with Tx.kernel():
            with Tx.warp():
                with Tx.thread():
                    with Tx.cta():
                        with Tx.thread():
                            pass
    @Tx.prim_func(check_well_formed=False)
    def test4() -> None:
        with Tx.kernel():
            with Tx.thread():
                with Tx.warpgroup():
                    with Tx.warp():
                        with Tx.thread():
                            pass
                with Tx.warpgroup():
                    with Tx.warp():
                        with Tx.thread():
                            pass

    # fmt: on

    verify(test1)
    verify(test2)
    verify(test3)
    verify(test4)


def test_scope_id_consistency():
    # fmt: off
    @Tx.prim_func(check_well_formed=False)
    def test1():
        with Tx.kernel():
            Tx.cta_id([32])
            Tx.warp_id([4])
            Tx.lane_id([32])

            with Tx.thread():
                pass

    @Tx.prim_func(check_well_formed=False)
    def test2():
        with Tx.kernel():
            Tx.cta_id([32])
            Tx.warp_id([4])
            Tx.lane_id([32])
            Tx.thread_id([128])

            with Tx.thread():
                pass

    @Tx.prim_func(check_well_formed=False)
    def test3():
        with Tx.kernel():
            Tx.cta_id([32])
            Tx.warp_id([2])
            Tx.lane_id([32])
            Tx.thread_id([128])

            with Tx.thread():
                pass

    @Tx.prim_func(check_well_formed=False)
    def test4():
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([8, 10, 12])
            cbx, cby, cbz = Tx.cta_id_in_cluster([2, 2, 1])
            clx, cly, clz = Tx.cluster_id([4, 5, 12])
            with Tx.cta():
                with Tx.warp():
                    with Tx.thread():
                        Tx.evaluate(bx + by + bz)
                        Tx.evaluate(cbx + cby + cbz)
                        Tx.evaluate(clx + cly + clz)

    @Tx.prim_func(check_well_formed=False)
    def test5():
        with Tx.kernel():
            bx, by, bz = Tx.cta_id([8, 10, 12])
            cbx, cby, cbz = Tx.cta_id_in_cluster([2, 2, 1])
            clx, cly, clz = Tx.cluster_id([3, 5, 12])
            with Tx.cta():
                with Tx.warp():
                    with Tx.thread():
                        Tx.evaluate(bx + by + bz)
                        Tx.evaluate(cbx + cby + cbz)
                        Tx.evaluate(clx + cly + clz)

    @Tx.prim_func(check_well_formed=False)
    def test6():
        with Tx.kernel():
            clx, cly, clz = Tx.cluster_id([4, 5, 12])
            bx, by, bz = Tx.cta_id([8, 10, 12])
            with Tx.cluster():
                cbx, cby, cbz = Tx.cta_id_in_cluster([2, 2, 1])
                with Tx.warp():
                    with Tx.thread():
                        Tx.evaluate(bx + by + bz)
                        Tx.evaluate(cbx + cby + cbz)
                        Tx.evaluate(clx + cly + clz)

    @Tx.prim_func(check_well_formed=False)
    def test7():
        with Tx.kernel():
            clx, cly, clz = Tx.cluster_id([3, 5, 12])
            bx, by, bz = Tx.cta_id([8, 10, 12])
            with Tx.cluster():
                cbx, cby, cbz = Tx.cta_id_in_cluster([2, 2, 1])
                with Tx.warp():
                    with Tx.thread():
                        Tx.evaluate(bx + by + bz)
                        Tx.evaluate(cbx + cby + cbz)
                        Tx.evaluate(clx + cly + clz)

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
    @Tx.prim_func(check_well_formed=False)
    def test1():
        with Tx.kernel():
            Tx.cta_id([32])
            Tx.warp_id([4])
            Tx.lane_id([32])

            with Tx.thread():
                A = Tx.alloc_buffer((2,), layout=Tx.TileLayout(Tx.S[2, 1]))

                A[0] = 0
    # fmt: on
    verify(test1)

    ### SwizzleLayout
    # fmt: off
    @Tx.prim_func(check_well_formed=False)
    def test2():
        with Tx.kernel():
            Tx.cta_id([32])
            Tx.warp_id([4])
            Tx.lane_id([32])

            with Tx.thread():
                A = Tx.alloc_buffer((512,), scope="shared", layout=Tx.SwizzleLayout(3, 3, 3))

                A[0] = 0
    # fmt: on
    verify(test2)


def test_host():
    # fmt: off
    @Tx.prim_func(check_well_formed=False)
    def test1(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (16, 16), dtype="float32", align=16)

        A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float32", 2, A.data, 16, 16, 64, 16, 16, 1, 1, 0, 0, 0, 0)  # noqa: E501

        with Tx.kernel():
            for blockIdx in Tx.thread_binding(1, thread="blockIdx.x"):
                for threadIdx in Tx.thread_binding(128, thread="threadIdx.x"):
                    with Tx.thread():
                        bar = Tx.alloc_buffer((1,), "uint64", scope="shared", align=8)
                        phase = Tx.alloc_buffer((1,), "int32", scope="local")
                        A_smem = Tx.alloc_buffer((16, 16), "float32", scope="shared", align=128)

                        phase[0] = 0
                        if threadIdx == 0:
                            Tx.ptx.mbarrier.init(bar.data, 1)
                            Tx.ptx.fence.proxy_async("shared::cta")
                            Tx.ptx.cp_async.bulk.tensor.g2c(2, A_smem.data, bar.data, Tx.address_of(A_map), 0, 1, "", 0, 0)  # noqa: E501
                            Tx.ptx.mbarrier.arrive.expect_tx(bar.data, 16*16*4)
                        Tx.ptx.mbarrier.try_wait(bar.data, phase[0])
                        phase[0] = phase[0] ^ 1
                        Tx.print_buffer(A_smem.data, "float32", False, False, 2, 16*16)
    # fmt: on
    verify(test1)


def test_device_func():
    # fmt: off
    @Tx.prim_func(check_well_formed=False)
    def test1(A: Tx.Buffer((128,), "float32")):
        with Tx.cta():
            Tx.thread_id([128])
            Tx.fill(A, 0.)

    @Tx.prim_func(check_well_formed=False)
    def test2(A: Tx.Buffer((128,), "float32")):
        with Tx.kernel():
            Tx.cta_id([128])
            Tx.thread_id([128])
            Tx.fill(A, 0.)

    @Tx.prim_func(check_well_formed=False)
    def test3(A: Tx.Buffer((128,), "float32")):
        with Tx.cta():
            Tx.thread_id([128])
            Tx.fill(A, 0.)
        with Tx.cta():
            Tx.thread_id([128])
            Tx.fill(A, 0.)
    # fmt: on
    verify(test1, device_func=True)
    with pytest.raises(Exception, match="higher than kernel scope"):
        verify(test2, device_func=True)
    with pytest.raises(Exception, match="Only one root scope is allowed in device function"):
        verify(test3, device_func=True)


def test_preferred_cluster_validation():
    # fmt: off
    # Valid: cluster→cta with preferred_extents matching size
    @Tx.prim_func(check_well_formed=False)
    def test1() -> None:
        with Tx.kernel():
            cbx, cby = Tx.cta_id_in_cluster([2, 1], preferred=[2, 2])
            tx = Tx.thread_id([128])
            with Tx.thread():
                Tx.evaluate(cbx + cby + tx)

    # Invalid: preferred size doesn't match extents size (caught at verify time)
    @Tx.prim_func(check_well_formed=False)
    def test2() -> None:
        with Tx.kernel():
            cbx, cby = Tx.cta_id_in_cluster([2, 1], preferred=[2])
            tx = Tx.thread_id([128])
            with Tx.thread():
                Tx.evaluate(cbx + cby + tx)
    # fmt: on

    verify(test1)
    with pytest.raises(Exception, match="preferred_extents must have the same size"):
        verify(test2)

    # Invalid: preferred on a non-cluster→cta scope (caught at IR build time)
    with pytest.raises(Exception):
        # fmt: off
        @Tx.prim_func(check_well_formed=False)
        def test3() -> None:
            with Tx.kernel():
                bx = Tx.cta_id([128], preferred=[256])
                tx = Tx.thread_id([128])
                with Tx.thread():
                    Tx.evaluate(bx + tx)
        # fmt: on


def test_scope_id_deferred_relaxed_at_construction():
    """Deferred scope_id (no extents) must pass the well-formed check even when
    no sibling provides enough info to resolve it -- strict resolution is
    deferred to LowerTIRx."""

    # fmt: off
    @Tx.prim_func(check_well_formed=False)
    def partial_only_cta():
        with Tx.kernel():
            bx = Tx.cta_id()           # deferred kernel→cta, no closure source
            tx = Tx.thread_id([128])   # explicit
            with Tx.thread():
                Tx.evaluate(bx + tx)

    @Tx.prim_func(check_well_formed=False)
    def all_deferred():
        with Tx.kernel():
            bx = Tx.cta_id()
            wg = Tx.warpgroup_id()
            warp = Tx.warp_id_in_wg()
            lane = Tx.lane_id()
            with Tx.thread():
                Tx.evaluate(bx + wg + warp + lane)

    @Tx.prim_func(check_well_formed=False)
    def mixed():
        with Tx.kernel():
            # kCtaWarp=4, kWarpThread=32 → kCtaThread=128 derivable.
            Tx.warp_id([4])
            Tx.lane_id([32])
            Tx.thread_id()             # deferred kCtaThread, resolvable via closure
            with Tx.thread():
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
    @Tx.prim_func(check_well_formed=False)
    def inconsistent():
        # 4 warps * 32 lanes = 128 threads, but explicit thread_id says 64 -> error.
        with Tx.kernel():
            Tx.cta_id([32])
            Tx.warp_id([4])
            Tx.lane_id([32])
            Tx.thread_id()       # deferred (shouldn't shadow the conflict)
            Tx.thread_id([64])   # conflicts with derived kCtaThread=128
            with Tx.thread():
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
