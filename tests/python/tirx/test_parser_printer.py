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
import math

import pytest

import tvm
import tvm.script
import tvm.testing
from tvm.ir import PointerType, PrimType, assert_structural_equal
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TCol, TLane, laneid, warpid


def from_source(code):
    return tvm.script.from_source(code)


def _make_minimal_tirx_prim_func():
    source = (
        "# from tvm.script import tirx as T\n\n"
        "@T.prim_func()\n"
        "def f(a: T.handle):\n"
        '    A = T.match_buffer(a, (1,), "float32")\n'
        "    A[0] = T.float32(1)"
    )
    return from_source(source)


def from_source_tir(code):
    return tvm.script.from_source(code, s_tir=True)


def test_roundtrip_scopeid1():
    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (64,), "float32", scope="global")

        T.device_entry()
        bx, by, bz = T.cta_id([1, 1, 1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        A_local = T.alloc_buffer([1], dtype="float16", scope="local")
        for i in T.serial(2):
            A_local[0] = A[lane_id * 2 + i]
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_scopeid2():
    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle) -> None:
        _ = T.match_buffer(A_ptr, (64,), "float32", scope="global")

        T.device_entry()
        bx, by, bz = T.cta_id([8, 10, 12])
        cbx, cby, cbz = T.cta_id_in_cluster([2, 2, 1])
        cta_id_in_pair = T.cta_id_in_pair()
        clx, cly, clz = T.cluster_id([4, 5, 12])
        T.evaluate(bx + by + bz)
        T.evaluate(cbx + cby + cbz)
        T.evaluate(cta_id_in_pair)
        T.evaluate(clx + cly + clz)
        # fmt: on

    code = test.script()
    assert "cta_id_in_pair = T.cta_id_in_pair()" in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_scopeid_deferred():
    """Deferred ScopeIdDef (extent=None) survives print→parse round-trip
    as a no-arg ``T.cta_id()``/``T.thread_id()`` etc. call."""

    # fmt: off
    @T.prim_func(private=True)
    def test(A_ptr: T.handle) -> None:
        _ = T.match_buffer(A_ptr, (64,), "float32", scope="global")
        T.device_entry()
        bx = T.cta_id()                       # deferred kernel→cta
        cbx = T.cta_id_in_cluster([2])
        clx = T.cluster_id([4])
        tx = T.thread_id()                    # deferred cta→thread
        T.warp_id([4])
        T.lane_id([32])
        T.evaluate(bx + cbx + clx + tx)
        # fmt: on

    code = test.script()
    assert "bx = T.cta_id()" in code
    assert "tx = T.thread_id()" in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_exec_scope_filter_guard_roundtrip():
    @T.prim_func(private=True)
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (1,), "float32", scope="global")

        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([128])
        if (0 <= tx) & (tx < 1):
            A[0] = T.float32(1)

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_layout():
    def get_layout1():
        return T.TileLayout(T.S[(8, 8, 8, 4, 2) : (6, 4 @ laneid, 2, 1 @ laneid, 1)])

    def get_layout2():
        return T.TileLayout(T.S[(8, 8, 8, 4, 2) : (64, 4 @ laneid, 8, 2, 1)])

    def get_layout3():
        return T.TileLayout(T.S[(8, 16, 8, 16) : (1024, 16, 128, 1)])

    def get_layout4():
        return T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(512,)]))

    def get_layout5():
        return T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(64, 64, 4) : (64, 1, 64 * 64)]))

    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle) -> None:
        _ = T.match_buffer(A_ptr, (64,), "float32", scope="global")

        T.device_entry()
        bx, by, bz = T.cta_id([1, 1, 1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        C = T.alloc_buffer([128, 128], dtype="float16", scope="shared", layout=get_layout3())
        D = T.alloc_buffer([128, 32], dtype="float16", scope="shared", layout=get_layout4())
        A_warp = T.alloc_buffer([64, 64], dtype="float16", scope="shared", layout=get_layout1())
        B_warp = T.alloc_buffer([64, 64], dtype="float16", scope="shared", layout=get_layout2())

        E = T.alloc_buffer([64, 256], dtype="float16", scope="shared", layout=get_layout5())
        T.evaluate(A_warp[0, 0] + B_warp[0, 0] + C[0, 0] + D[0, 0] + E[0, 0])
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_layout_replica_and_offset():
    """Round-trip layouts that exercise the replica and offset (single- and
    multi-axis) printer paths. The multi-axis case relies on
    `_LayoutSpec.__add__` correctly merging successive offset terms instead
    of overwriting (see `_merge_offset` in `tvm.tirx.layout`)."""

    def get_shard_replica():
        return T.TileLayout(T.S[8 : 4 @ laneid] + T.R[4 : 1 @ laneid])

    def get_shard_offset_single():
        return T.TileLayout(T.S[8 : 4 @ laneid] + 1 @ laneid)

    def get_shard_offset_multi():
        return T.TileLayout(T.S[8 : 4 @ laneid] + 1 @ laneid + 2 @ warpid + 64)

    def get_full():
        return T.TileLayout(T.S[(1,) : (1,)] + T.R[(8, 4) : (4 @ laneid, 1 @ laneid)] + 2 @ warpid)

    # fmt: off
    @T.prim_func
    def test() -> None:
        T.device_entry()
        A = T.alloc_buffer([8], dtype="float16", scope="shared", layout=get_shard_replica())
        B = T.alloc_buffer([8], dtype="float16", scope="shared", layout=get_shard_offset_single())
        C = T.alloc_buffer([8], dtype="float16", scope="shared", layout=get_shard_offset_multi())
        D = T.alloc_buffer([32], dtype="float16", scope="shared", layout=get_full())
        T.evaluate(A[0] + B[0] + C[0] + D[0])
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_print_kwargs_schedule_op_full_code():
    # fmt: off
    @T.prim_func
    def test():
        A = T.alloc_buffer((16,), "float32")
        Tx.memset(A[0:16], T.float32(1.25), dispatch="v10", bar=7, foo=42)
    # fmt: on

    expected = (
        "# from tvm.script import tirx as T\n"
        "# from tvm.tirx.layout import Axis\n\n"
        "@T.prim_func\n"
        "def test():\n"
        "    A = T.alloc_buffer((16,))\n"
        '    T.tile.memset(A[0:16], T.float32(1.25), dispatch="v10", bar=7, foo=42)'
    )
    code = test.script()
    assert code == expected
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_default_script_prefix_tirx_irmodule_non_main():
    """IRModule with non-main TIRx PrimFunc should default to T prefix."""
    mod = tvm.IRModule({"foo": _make_minimal_tirx_prim_func()})
    code = mod.script()
    assert "# from tvm.script import tirx as T" in code
    assert "# from tvm.script import tir as T" not in code
    assert "@T.prim_func" in code
    assert "def foo(" in code
    parsed = from_source(code)
    assert parsed.script() == code
    assert_structural_equal(mod, parsed)


L_LANE = T.TileLayout(T.S[32 : 1 @ laneid])


def test_roundtrip_buffer_view_get1():
    # fmt: off
    @T.prim_func
    def test() -> None:
        T.device_entry()
        A = T.alloc_buffer([2], dtype="float16", scope="local")
        A_layout = T.TileLayout(T.S[(1, 2) : (2, 1)])
        A_warp_layout = A_layout.tile(L_LANE, (8, 4), (1, 2))
        A_warp = A.view(8, 8, layout=A_warp_layout)
        A_local = A_warp.local(2)
        A_local[0] = T.float16(0)

        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_buffer_view_get2():
    # fmt: off
    @T.prim_func
    def test(out_ptr: T.handle) -> None:
        out = T.match_buffer(out_ptr, (2), "float32", scope="global")

        T.device_entry()
        bx, by, bz = T.cta_id([32, 32, 1])
        tx, ty, tz = T.thread_id([16, 8, 1])
        warp_id = T.warp_id([4])
        lane_id = T.lane_id([32])
        A = T.alloc_buffer([2,], dtype="float16", scope="local")
        A_layout = T.TileLayout(T.S[(1, 2) : (2, 1)])
        B_layout = A_layout.tile(L_LANE, (8, 4), (1, 2))
        B = A.view(8, 8, layout=B_layout)
        D = B.local(2)
        out[0] = A[0] + B[0, 0] + D[0]
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_buffer_view_get3():
    # fmt: off
    @T.prim_func
    def test() -> None:
        T.device_entry()
        A = T.alloc_buffer([8, 8], dtype="float32", scope="local")
        A_f16 = A.view("float16")
        A_f64 = A.view("float64")
        A_f16[0, 0] = T.float16(0)
        A_f64[0, 0] = T.float64(0)

        # fmt: on
    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op1():
    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (64,), "float32", scope="global")

        T.device_entry()
        bx, by, bz = T.cta_id([1, 1, 1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        A_smem = T.alloc_buffer([64], dtype="float32", scope="shared")

        Tx.cta.copy(A_smem, A)
        for i in range(10):
            Tx.cta.fill(A_smem, T.float32(0))
            Tx.cta.gemm(A_smem, A_smem, A_smem, A_smem)
        Tx.cta.copy(A, A_smem)
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op2():
    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, 128), "float16", scope="global")
        B = T.match_buffer(B_ptr, (128, 64), "float16", scope="global")
        C = T.match_buffer(C_ptr, (128, 64), "float32", scope="global")

        T.device_entry()
        bx, by, bz = T.cta_id([1, 1, 1])
        warp_id = T.warp_id([4])
        lane_id = T.lane_id([32])
        A_smem = T.alloc_buffer([128, 32], dtype="float16", scope="shared")
        B_smem = T.alloc_buffer([32, 64], dtype="float16", scope="shared")

        C_local = T.alloc_buffer([128, 64], dtype="float32", scope="local")
        for k in range(4):
            Tx.cta.copy(A_smem, A[:, k * 32 : k * 32 + 32])
            Tx.cta.copy(B_smem, B[k * 32 : k * 32 + 32, 0:64])
            Tx.cta.gemm(C_local, A_smem, B_smem, C_local)
        Tx.cta.copy(C, C_local)
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op3():
    # fmt: off
    NUM_STAGES = 3
    K = 4096

    @T.prim_func
    def test(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128, K), "float16", scope="global")
        B = T.match_buffer(B_ptr, (K, 64), "float16", scope="global")
        C = T.match_buffer(C_ptr, (128, 64), "float32", scope="global")

        T.device_entry()
        bx, by, bz = T.cta_id([1, 1, 1])
        warp_id = T.warp_id([4])
        lane_id = T.lane_id([32])
        A_smem = T.alloc_buffer([NUM_STAGES, 128, 32], dtype="float16", scope="shared")
        B_smem = T.alloc_buffer([NUM_STAGES, 32, 64], dtype="float16", scope="shared")

        C_local = T.alloc_buffer([128, 64], dtype="float32", scope="local")
        for i in range(NUM_STAGES - 1):
            Tx.cta.copy(A_smem[i, :, :], A[:, i * 32 : i * 32 + 32])
            Tx.cta.copy(B_smem[i, :, :], B[i * 32 : i * 32 + 32, :])

        for k in range(K // 32):
            copy_k = T.meta_var(k + NUM_STAGES - 1)
            gemm_stage = T.meta_var(k % NUM_STAGES)
            copy_stage = T.meta_var(copy_k % NUM_STAGES)
            Tx.cta.copy(A_smem[copy_stage, :, :], A[:, copy_k * 32 : copy_k * 32 + 32])
            Tx.cta.copy(B_smem[copy_stage, :, :], B[copy_k * 32 : copy_k * 32 + 32, :])
            Tx.cta.gemm(C_local, A_smem[gemm_stage, :, :], B_smem[gemm_stage, :, :], C_local)

        Tx.cta.copy(C, C_local)
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_tensormap():
    # fmt: off
    @T.prim_func
    def func1(A_ptr: T.handle):
        T.func_attr({"global_symbol": "func"})
        _ = T.match_buffer(A_ptr, [128], "float32")

        A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.tensormap_init", T.address_of(A_map), A_ptr)
    # fmt: on
    code = func1.script()
    assert from_source(code).script() == code
    assert_structural_equal(func1, from_source(code))


def test_roundtrip_tensormap_kernel_param():
    # fmt: off
    @T.prim_func
    def func1(A_map: T.TensorMap()):
        T.func_attr({"global_symbol": "func"})
        T.evaluate(T.address_of(A_map))
    # fmt: on
    code = func1.script()
    assert "T.TensorMap()" in code
    assert from_source(code).script() == code
    assert_structural_equal(func1, from_source(code))


def test_roundtrip_break_for():
    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (10,), "int32")

        T.device_entry()
        for i in T.serial(10):
            if i > 5:
                break
            A[i] = i
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_break_while():
    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (10,), "int32")

        T.device_entry()
        i = T.alloc_buffer((1,), "int32", scope="local")
        i[0] = 0
        while i[0] < 10:
            A[i[0]] = i[0] * 2
            if A[i[0]] > 10:
                break
            i[0] = i[0] + 1
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_break_nested():
    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (9,), "int32")

        T.device_entry()
        idx = T.alloc_buffer((1,), "int32", scope="local")
        idx[0] = 0
        for i in T.serial(3):
            for j in T.serial(3):
                A[idx[0]] = i * 10 + j
                idx[0] += 1
                if j == 1:
                    break
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_continue_for():
    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (10,), "int32")

        T.device_entry()
        for i in T.serial(10):
            if (i % 2) == 0:
                continue
            A[i] = i
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_continue_while():
    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (10,), "int32")

        T.device_entry()
        i = T.alloc_buffer((1,), "int32", scope="local")
        i[0] = 0
        while i[0] < 10:
            if (i[0] % 2) == 1:
                i[0] += 1
                continue
            A[i[0]] = i[0]
            i[0] += 1
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_continue_nested():
    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (9,), "int32")

        T.device_entry()
        idx = T.alloc_buffer((1,), dtype="int32", scope="local")
        idx[0] = 0
        for i in T.serial(3):
            for j in T.serial(3):
                if j == 1:
                    continue
                A[idx[0]] = i * 10 + j
                idx[0] += 1
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_break_and_continue():
    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (10,), "int32")

        T.device_entry()
        for i in T.serial(10):
            if i == 2:
                continue
            if i == 7:
                break
            A[i] = i
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_unreachable_after_break():
    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (5,), "int32")

        T.device_entry()
        for i in T.serial(5):
            A[i] = i
            break
                    # This line is never reached
            A[i] = -1
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_allocated_addr():
    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
        A = T.alloc_buffer([10], "float32", scope="trn.sbuf", allocated_addr=1024)
        for i in T.serial(2):
            Tx.memset(A[i*5:i*5+5], T.float32(0.0))

        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_implicit_buffer_region():
    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (10, 10, 10), "float32", layout=T.TileLayout(T.S[10, 10, 10]))
        T.device_entry()
        Tx.memset(A[0], T.float32(0.0))

        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_alloc_under_any_scope():
    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
        for i in T.serial(10):
            A = T.alloc_buffer([100], "float32", scope="trn.sbuf", allocated_addr=1024)
            Tx.memset(A[i*10:i*10+10], T.float32(0.0))

        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_compose_op():
    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
        A = T.alloc_buffer([10], "float32", scope="trn.sbuf")
        B = T.alloc_buffer([10], "float32", scope="trn.sbuf")
        C = T.alloc_buffer([10], "float32", scope="trn.sbuf")
        with Tx.compose_op():
            Tx.add(B, A, T.float32(1))
            Tx.add(C, B, T.float32(1))
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op_call_workspace():
    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, [10], "float32", scope="global")
        B = T.match_buffer(B_ptr, [10], "float32", scope="global")
        T.device_entry()
        smem = T.alloc_buffer([10], "float32", scope="shared")
        Tx.add(B, A, T.float32(1), workspace={"smem": smem})
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_compose_op_call_workspace():
    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
        A = T.alloc_buffer([10], "float32", scope="trn.sbuf")
        B = T.alloc_buffer([10], "float32", scope="trn.sbuf")
        C = T.alloc_buffer([10], "float32", scope="trn.sbuf")
        psum = T.alloc_buffer([10], "float32", scope="trn.psum")
        intermediate = T.alloc_buffer([10], "float32", scope="trn.sbuf")
        with Tx.compose_op(workspace={"intermediate": intermediate}):
            Tx.add(B, A, T.float32(1))
            Tx.add(C, B, T.float32(1), workspace={"psum": psum})
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op_call_config():
    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, [10], "float32", scope="global")
        B = T.match_buffer(B_ptr, [10], "float32", scope="global")
        T.device_entry()
        Tx.add(B, A, T.float32(1), schedule="A")
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_compose_op_call_config():
    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
        A = T.alloc_buffer([10], "float32", scope="trn.sbuf")
        B = T.alloc_buffer([10], "float32", scope="trn.sbuf")
        C = T.alloc_buffer([10], "float32", scope="trn.sbuf")
        psum = T.alloc_buffer([10], "float32", scope="trn.psum")
        with Tx.compose_op( schedule="A"):
            Tx.add(B, A, T.float32(1))
            Tx.add(C, B, T.float32(1), workspace={"psum": psum})
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_predicate():
    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
        A = T.alloc_buffer([10, 10], "float32")
        B = T.alloc_buffer([10, 10], "float32")
        Tx.select(B, A, 1.0, lambda i, j: i < j)
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_grid():
    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
        for lvs in T.grid(10, (2, 12)):
            T.evaluate(lvs[0] + lvs[1])
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_alloc_apis():
    # fmt: off
    @T.meta_class
    class Test:
        def __init__(self, Ta, inner_pool):
            self.Ta = Ta
            self.inner_pool = inner_pool
            self.Tb = T.shared_scalar("float16")
            self.idx = T.local_scalar("int32")
            self.inner_pool2 = T.decl_scalar("float16", self.inner_pool.data, "shared.dyn", 5)

        @T.inline
        def init(self):
            self.Ta = self.Ta + T.float16(1)
            self.Tb = self.Tb + T.float16(2)
            self.idx.source[0] = T.int32(0)
            self.idx = self.idx + T.int32(1)
            self.inner_pool2 = self.inner_pool2 + T.float16(1)
            T.evaluate(T.address_of(self.Ta))
            T.evaluate(T.address_of(self.Tb))
            T.evaluate(T.address_of(self.idx))
            T.evaluate(T.address_of(self.inner_pool))
            T.evaluate(T.address_of(self.inner_pool2))

    @T.prim_func
    def test():
        T.device_entry()
                # normal buffer
        A = T.alloc_shared([10], "float16")
        B = T.alloc_local([10], "float16")
                # scalar buffer (alloc)
        C = T.shared_scalar("float16")
        D: T.float16
        pool = T.alloc_buffer([10], "uint8", scope="shared.dyn")
                # scalar buffer (decl)
        E = T.decl_scalar("float16", pool.data, "shared.dyn", 0)
                # normal 1-dim buffer with shape (1,)
        F = T.alloc_local((1,), "float16")
        Ta: T.float16
        inner_pool = T.decl_buffer(shape=[10], data=pool.data, dtype="uint8", scope="shared.dyn")
        test = Test(Ta, inner_pool)  # noqa: F821
        test.init()
        A[0] = C
        A[0] = C + D  # noqa: F821
        A[1] = B[0] * C
        D.source[0] = D + T.float16(1)  # noqa: F821
        D = D + T.float16(1)  # noqa: F821
        C = D
        T.evaluate(E)
        E = E + T.float16(1)
                # normal 1-dim buffer with shape (1,) can be assigned directly,
                # but not loaded directly
        F = F[0] + T.float16(1)
        C += D
        D += E + C + D
        T.evaluate(T.address_of(C))
        T.evaluate(C.source.access_ptr("rw", offset=0))
        T.evaluate(C.source.data)
        T.evaluate(D)
        T.evaluate(T.address_of(D))
        # fmt: on

    code = test.script()
    print(code)
    assert ".buffer" not in code
    assert from_source(code).script() == code


def test_alloc_apis_reject_name_argument():
    with pytest.raises(TypeError):
        T.alloc_buffer((1,), "int32", name="buf")

    with pytest.raises(TypeError):
        T.local_scalar("int32", name="idx")


def test_meta_class_constructor_rejects_unowned_resource():
    @T.meta_class
    class Bad:
        def __init__(self):
            tmp = T.alloc_buffer((1,), "int32", scope="local")

    with pytest.raises(tvm.error.DiagnosticError):

        @T.prim_func
        def test():
            T.device_entry()
            bad = Bad()


def test_meta_class_multiple_instances_auto_name_owned_resources():
    @T.meta_class
    class Holder:
        def __init__(self, external):
            self.external = external
            self.buf = T.alloc_buffer((2,), "int32", scope="local")
            self.scalar = T.local_scalar("int32")

    @T.prim_func
    def test():
        T.device_entry()
        external = T.alloc_buffer((2,), "int32", scope="local")
        first = Holder(external)
        second = Holder(external)
        T.evaluate(
            first.buf[0]
            + second.buf[1]
            + first.scalar
            + second.scalar
            + first.external[0]
            + second.external[1]
        )

    code = test.script()
    bufs = _collect_buffers(test)
    assert "external" in bufs
    assert "first_external" not in bufs
    assert "second_external" not in bufs
    assert {"first_buf", "second_buf", "first_scalar", "second_scalar"}.issubset(bufs)
    assert 'first_buf = T.alloc_local((2,), "int32")' in code
    assert 'second_buf = T.alloc_local((2,), "int32")' in code
    assert "first_scalar: T.int32" in code
    assert "second_scalar: T.int32" in code
    assert from_source(code).script() == code


def test_macro():
    # fmt: off
    @T.inline
    def mul(x, c):
        T.evaluate(x * c)

    @T.prim_func(private=True)
    def test():
        T.device_entry()
        for x in range(10):

            @T.inline
            def add(c):
                T.evaluate(x + c)

            @T.inline
            def two_add_and_mul(c):
                add(c)
                add(c + c)
                mul(x, c)

            two_add_and_mul(1)
            two_add_and_mul(2)


    @T.prim_func(private=True)
    def expected():
        T.device_entry()
        for x in range(10):
            T.evaluate(x + 1)
            T.evaluate(x + 2)
            T.evaluate(x)
            T.evaluate(x + 2)
            T.evaluate(x + 4)
            T.evaluate(x * 2)
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))
    assert_structural_equal(test, expected)


def test_macro_recursive():
    # fmt: off
    @T.prim_func(private=True)
    def test():
        T.device_entry()
        for x in T.serial(10):

            @T.inline
            def add(x, c):
                if c > 0:
                    add(x, c - 1)
                T.evaluate(x)

            add(x, 5)

    @T.prim_func(private=True)
    def expected():
        T.device_entry()
        for x in range(10):
            T.evaluate(x)
            T.evaluate(x)
            T.evaluate(x)
            T.evaluate(x)
            T.evaluate(x)
            T.evaluate(x)
        # fmt: on
    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))
    assert_structural_equal(expected, from_source(code))


def test_list_comprehension():
    # fmt: off
    @T.prim_func(private=True)
    def test():
        T.device_entry()
        acc = T.alloc_local([10], "bool")
        regs = T.meta_var([acc[_] for _ in range(10)])
        T.evaluate(regs[0])
        T.evaluate(tvm.tirx.all(*regs))
        T.evaluate(tvm.tirx.all(*[acc[_] for _ in range(10)]))
        T.evaluate(tvm.tirx.all(*([acc[_] for _ in range(2, 4)] + [acc[_] for _ in range(6, 8)])))
        # fmt: on
    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_range():
    # fmt: off
    @T.prim_func(private=True)
    def test():
        l = T.meta_var([i for i in range(10)])  # noqa: E741
        T.evaluate(l[3])

    @T.prim_func(private=True)
    def expected():
        T.evaluate(3)
    # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))
    tvm.ir.assert_structural_equal(test, expected)


def test_shared_meta_var_alias():
    assert I.meta_var is T.meta_var

    @T.prim_func(private=True)
    def via_ir_namespace():
        value = I.meta_var(T.int32(1))
        T.evaluate(value)

    @T.prim_func(private=True)
    def via_tirx_alias():
        value = T.meta_var(T.int32(1))
        T.evaluate(value)

    assert_structural_equal(via_ir_namespace, via_tirx_alias)
    code = via_ir_namespace.script()
    assert "meta_var" not in code
    assert_structural_equal(via_ir_namespace, from_source(code))


def test_buffer():
    # fmt: off
    @T.prim_func(private=True)
    def test(
        A: T.Buffer((10, 11), "float32", layout=None),
        B: T.Buffer((10, 11), "float32", scope="global"),
        C: T.Buffer((10, 11), "float32", layout="default"),
        D: T.Buffer((10, 11), "float32", layout=T.TileLayout(T.S[(10, 11) : (1, 10)])),
        E_ptr: T.handle,
        F_ptr: T.handle,
        G_ptr: T.handle,
        H_ptr: T.handle,
    ):
        _E = T.match_buffer(E_ptr, [10, 11], "float16", layout=None)
        _F = T.match_buffer(F_ptr, [10, 11], "float16", scope="global")
        _G = T.match_buffer(G_ptr, [10, 11], "float16", layout="default")
        _H = T.match_buffer(H_ptr, [10, 11], "float16", layout=T.TileLayout(T.S[(10, 11) : (1, 10)]))  # noqa: E501

        _A0 = T.decl_buffer((10, 11), "float32", data=A.data, layout=None)
        _B0 = T.decl_buffer((10, 11), "float32", data=B.data, scope="global")
        _C0 = T.decl_buffer((10, 11), "float32", data=C.data, layout="default")
        _D0 = T.decl_buffer((10, 11), "float32", data=D.data, layout=T.TileLayout(T.S[(10, 11) : (1, 10)]))  # noqa: E501
        _A1 = T.alloc_buffer((10, 11), "float32", layout=None)
        _B1 = T.alloc_buffer((10, 11), "float32", scope="global")
        _C1 = T.alloc_buffer((10, 11), "float32", layout="default")
        _D1 = T.alloc_buffer((10, 11), "float32", layout=T.TileLayout(T.S[(10, 11) : (1, 10)]))

        pass
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_kwargs_op_call():
    # fmt: off
    @T.prim_func(private=True)
    def test(A: T.Buffer((10, 10), "float32"), B: T.Buffer((10, 10), "float32")):
        T.device_entry()
        kwargs = T.meta_var({"dispatch": "tma_auto", "cta_group": 2})
        Tx.copy_async(A[:, :], B[:, :], **kwargs)
        # fmt: on
    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_workspace_default_none():
    """Regression: TIRX op IR builder functions (binary_reduce, unary_reduce,
    binary_chain, reduce_negate) should handle workspace=None (the default)
    without error. Previously these functions were missing the
    ``if workspace is None: workspace = {}`` guard."""
    from tvm.tirx import BufferRegion

    A_buf = tvm.tirx.decl_buffer((128, 128), "float16", name="A")
    B_buf = tvm.tirx.decl_buffer((128, 128), "float16", name="B")
    C_buf = tvm.tirx.decl_buffer((128,), "float16", name="C")
    A = BufferRegion(A_buf, [tvm.ir.Range(0, 128), tvm.ir.Range(0, 128)])
    B = BufferRegion(B_buf, [tvm.ir.Range(0, 128), tvm.ir.Range(0, 128)])
    C = BufferRegion(C_buf, [tvm.ir.Range(0, 128)])

    # These should not crash when workspace is not provided (defaults to None)
    from tvm.tirx.operator.tile_primitive import ops as tirx_op

    op_br = tirx_op.BinaryReduce(
        B, C, A, B, tirx_op.get_tirx_op("add"), tirx_op.get_tirx_op("max"), (-1,)
    )
    assert len(op_br.workspace) == 0

    op_ur = tirx_op.UnaryReduce(
        B, C, A, tirx_op.get_tirx_op("sqrt"), tirx_op.get_tirx_op("sum"), None, None, (-1,)
    )
    assert len(op_ur.workspace) == 0

    op_bc = tirx_op.BinaryChain(
        B, A, A, A, tirx_op.get_tirx_op("add"), tirx_op.get_tirx_op("mul"), False
    )
    assert len(op_bc.workspace) == 0

    op_rn = tirx_op.ReduceNegate(C, A, (-1,), False, tirx_op.get_tirx_op("sum"))
    assert len(op_rn.workspace) == 0


def test_scalar_assign_in_macro():
    """Regression: the parser's scalar-assignment sugar (scalar = Expr) must
    work in macro context via self.attr.

    The parser narrowed ``except Exception: pass`` around the scalar-detection
    path. This test verifies that Expr assignment to a scalar attribute in
    a macro still goes through buffer_store correctly.

    The full integration regression for the TypeError fallthrough path
    (meta_var assigned to a scalar variable) is covered by
    test_hgemm::test_hgemm (tile_scheduler.m_idx pattern)."""

    # fmt: off
    class State:
        def __init__(self, counter):
            self.counter = counter

        @T.inline
        def add_one(self):
            # Expr assigned to scalar via self.attr → buffer_store succeeds
            self.counter = self.counter + T.int32(1)

    @T.prim_func
    def test():
        T.device_entry()
        counter: T.int32
        state = T.meta_var(State(counter))  # noqa: F821
        state.add_one()
        T.evaluate(state.counter)
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_scalar_assign_error_not_swallowed():
    """Regression: genuine errors (non-TypeError) from buffer_store during
    scalar-assignment sugar must propagate, not be silently swallowed.

    Before the fix, both eval_expr and buffer_store were wrapped in a single
    broad ``except Exception: pass``, so any error from buffer_store would be
    swallowed and the assignment would silently fall through to eval_assign."""
    from unittest.mock import patch

    original = tvm.tirx.script.builder.buffer_store

    def bomb(*args, **kwargs):
        # Intercept only the scalar-assignment path (indices == [0])
        if args[2] == [0]:
            raise ValueError("boom")
        return original(*args, **kwargs)

    src = """
# from tvm.script import tirx as T

@T.prim_func
def func():
    T.device_entry()
    v: T.int32
    v = v + T.int32(1)
"""
    # The ValueError propagates through the parser framework which wraps it
    # into a DiagnosticError.  Before the fix the broad ``except Exception``
    # would silently swallow it and fall through to eval_assign.
    with patch("tvm.tirx.script.builder.buffer_store", side_effect=bomb):
        with pytest.raises(tvm.error.DiagnosticError):
            from_source(src)


def test_scalar_annotation_syntax():
    """Test the scalar annotation syntax: x: T.int32 = init, x: T.int32, and T.let."""

    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
                # Scalar with init value
        x: T.int32 = 0
        y: T.float16 = T.float16(1.0)
                # Scalar without init
        z: T.int32
                # Use scalars
        x = x + T.int32(1)
        z = x + T.int32(2)
        y = y + T.float16(3.0)
        T.evaluate(x + z)
        T.evaluate(y)
        # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_scalar_allocbuffer_annotation_and_init_merge():
    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
        phase_mma = T.alloc_local((1,), "int32")
        phase_mma[0] = T.int32(0)
        phase_aux = T.alloc_local((1,), "int32")
        T.evaluate(phase_mma[0] + phase_aux[0])
        # fmt: on

    code = test.script()
    assert "phase_mma: T.int32 = 0" in code
    assert "phase_aux: T.int32" in code
    assert "phase_mma = T.alloc_local" not in code
    assert "phase_aux = T.alloc_local" not in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_scalar_allocbuffer_layout_none_keeps_alloc_local():
    # fmt: off
    @T.prim_func
    def test():
        T.device_entry()
        phase_mma = T.alloc_local((1,), "int32", layout=None)
        phase_mma[0] = T.int32(0)
        T.evaluate(phase_mma[0])
        # fmt: on

    code = test.script()
    assert 'phase_mma = T.alloc_local((1,), "int32", layout=None)' in code
    assert "phase_mma: T.int32" not in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_scalar_allocbuffer_annotation_sugar():
    # fmt: off
    @T.prim_func
    def test():
        x = T.alloc_buffer((1,), "int32", scope="local")
        x[0] = T.int32(0)
        T.evaluate(x[0])
    # fmt: on

    code = test.script()
    assert "x: T.int32 = 0" in code
    assert "x = T.alloc_buffer" not in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_let_annotation_syntax():
    """Test explicit LetStmt syntax: T.let[T.int32] and T.let."""

    # fmt: off
    @T.prim_func
    def test():
        blockIdx_x = T.launch_thread("blockIdx.x", 4)
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        # Explicit LetStmt with type
        bx: T.let[T.int32] = blockIdx_x
        tx: T.let[T.int32] = threadIdx_x
        # Explicit LetStmt with auto-type
        combined: T.let = bx + tx
        T.device_entry()
        T.evaluate(bx + tx + combined)
        # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_tuple_let_binding_and_traversal():
    @T.prim_func
    def from_list(x: T.int32, y: T.float32) -> T.int32:
        pair: T.let = [x, (y,)]
        return pair[0]

    @T.prim_func
    def from_tuple(x: T.int32, y: T.float32) -> T.int32:
        pair: T.let = (x, (y,))
        return pair[0]

    def tuple_value(func):
        visited = []
        tvm.tirx.stmt_functor.post_order_visit(func.body, visited.append)
        bind = next(node for node in visited if isinstance(node, tvm.tirx.Bind))
        return bind.value

    list_value = tuple_value(from_list)
    tuple_value = tuple_value(from_tuple)
    assert isinstance(list_value, tvm.ir.Tuple)
    assert isinstance(list_value.fields[1], tvm.ir.Tuple)
    assert_structural_equal(list_value, tuple_value, map_free_vars=True)

    code = from_list.script()
    assert "pair: T.let[T.Tuple(T.int32, T.Tuple(T.float32))] = x, (y,)" in code
    assert from_source(code).script() == code
    assert_structural_equal(from_list, from_source(code))


def test_annotation_syntax_comprehensive():
    """Comprehensive test for scalar annotation, T.let, banned annotations, and bare assignment."""

    # 1. T.let with T.Var(PointerType) — round-trip
    # fmt: off
    @T.prim_func
    def test_let_var():
        T.device_entry()
        smem = T.alloc_shared([128], "float16")
        ptr: T.let[T.Var(name="ptr", ty=PointerType(PrimType("void")))] = T.reinterpret(
            "handle", smem.access_ptr("rw")
        )
        T.evaluate(ptr)
        # fmt: on
    code = test_let_var.script()
    assert from_source(code).script() == code

    # 2. Banned: handle as scalar annotation
    src_handle = """
from tvm.script import tirx as T
@T.prim_func
def func():
    x: T.handle = T.int64(0)
"""
    with pytest.raises(tvm.error.DiagnosticError):
        from_source(src_handle)

    # 3. Banned: non-PrimType annotation without T.let
    src_ptr = """
from tvm.script import tirx as T
from tvm.ir import PointerType, PrimType
@T.prim_func
def func():
    x: T.Var(name="x", ty=PointerType(PrimType("float16"))) = T.int64(0)
"""
    with pytest.raises(tvm.error.DiagnosticError):
        from_source(src_ptr)

    # 4. Bare assignment to new variable creates scalar — round-trip
    # fmt: off
    @T.prim_func
    def test_bare_assign():
        T.device_entry()
        tid = T.launch_thread("threadIdx.x", 128)
        x = tid + T.int32(1)
        x = x + T.int32(2)
        T.evaluate(x)
        # fmt: on
    code = test_bare_assign.script()
    assert from_source(code).script() == code


def test_roundtrip_buffer_permute():
    # fmt: off
    @T.prim_func
    def test() -> None:
        T.device_entry()
        A = T.alloc_buffer([8, 4], dtype="float16", scope="local",
                            layout=T.TileLayout(T.S[(8, 4) : (4, 1)]))
        B = A.permute(1, 0)
        B[0, 0] = T.float16(0)
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_buffer_local_auto():
    # fmt: off
    @T.prim_func
    def test() -> None:
        T.device_entry()
        A = T.alloc_buffer([2], dtype="float16", scope="local")
        A_layout = T.TileLayout(T.S[(1, 2) : (2, 1)])
        B = A.view(8, 8, layout=A_layout.tile(L_LANE, (8, 4), (1, 2)))
        B_local = B.local()
        B_local[0] = T.float16(0)
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


###############################################################################
# IR verification tests - verify DeclBuffer properties, not just round-trip
###############################################################################


def _collect_buffers(func):
    """Collect all buffers from DeclBuffer and AllocBuffer nodes, returning {name: Buffer}."""
    bufs = {}

    def _visit(node):
        if isinstance(node, tvm.tirx.DeclBuffer | tvm.tirx.AllocBuffer):
            bufs[node.buffer.name] = node.buffer

    tvm.tirx.stmt_functor.post_order_visit(func.body, _visit)
    return bufs


def _collect_buffer_sources(func):
    """Collect the explicit data source of each DeclBuffer."""
    sources = {}

    def _visit(node):
        if isinstance(node, tvm.tirx.DeclBuffer):
            sources[node.buffer.name] = node.data

    tvm.tirx.stmt_functor.post_order_visit(func.body, _visit)
    return sources


def test_buffer_local_ir():
    """Verify .local() infers the physical span and uses an identity layout."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([2], dtype="float16", scope="local")
        A_layout = T.TileLayout(T.S[(1, 2) : (2, 1)])
        B = A.view(8, 8, layout=A_layout.tile(L_LANE, (8, 4), (1, 2)))
        B_local = B.local()
        B_local[0] = T.float16(0)
        # fmt: on

    bufs = _collect_buffers(func)
    b_local = bufs["B_local"]
    b_buf = bufs["B"]

    # Shared data pointer
    assert_structural_equal(_collect_buffer_sources(func)["B_local"], b_buf.data)
    # Shape: single dim matching the raw physical storage span
    assert len(b_local.ty.shape) == 1
    storage = b_buf.ty.layout.storage()
    assert int(b_local.ty.shape[0]) == int(storage.span())
    # The inferred view uses physical storage order, not storage-iterator order.
    assert b_local.ty.layout.is_trivial()

    # Round-trip
    code = func.script()
    assert "B_local = B.local()" in code
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_buffer_local_physical_order():
    """Both inferred and explicit shapes map a non-trivial fragment physically."""
    from tvm.tirx.layout import tcgen05_atom_layout

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([32], dtype="float32", scope="local")
        B = A.view(64, 64, layout=tcgen05_atom_layout("16x256b", (64, 64), "float32"))
        B_flat = B.local()
        B_2d = B.local(4, 8)
        B_flat[2] = T.float32(1)
        B_2d[0, 2] = T.float32(2)
        # fmt: on

    bufs = _collect_buffers(func)
    b_buf = bufs["B"]
    b_flat = bufs["B_flat"]
    b_2d = bufs["B_2d"]

    # The parent storage view enumerates storage iters in a different order
    # from their physical strides, so inheriting it would permute registers.
    assert not b_buf.ty.layout.storage().is_trivial()

    for local in [b_flat, b_2d]:
        assert_structural_equal(_collect_buffer_sources(func)[local.name], b_buf.data)
        assert local.ty.layout.is_trivial()
    assert [int(dim) for dim in b_flat.ty.shape] == [32]
    assert [int(dim) for dim in b_2d.ty.shape] == [4, 8]

    # Index 2 in either row-major shape is the same physical register.
    flat_offset = b_flat.ty.layout.apply(2, shape=list(b_flat.ty.shape))["m"]
    reshaped_offset = b_2d.ty.layout.apply(0, 2, shape=list(b_2d.ty.shape))["m"]
    assert int(flat_offset) == int(reshaped_offset) == 2

    code = func.script()
    assert "B_flat = B.local()" in code
    assert "B_2d = B.local(4, 8)" in code
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_buffer_local_layout_overrides_roundtrip():
    """Storage and arbitrary mediated layouts remain explicit overrides."""
    from tvm.tirx.layout import tcgen05_atom_layout

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([32], dtype="float32", scope="local")
        B = A.view(64, 64, layout=tcgen05_atom_layout("16x256b", (64, 64), "float32"))
        B_storage = B.local(layout=B.layout.storage())
        # An explicit layout is an escape hatch and may describe a smaller
        # mediated view than the parent's full per-thread storage.
        B_custom = B.local(2, 4, layout=T.TileLayout(T.S[(2, 4) : (1, 2)]))
        B_storage[0] = T.float32(1)
        B_custom[0, 0] = T.float32(2)
        # fmt: on

    bufs = _collect_buffers(func)
    b_buf = bufs["B"]
    b_storage = bufs["B_storage"]
    b_custom = bufs["B_custom"]
    assert_structural_equal(b_storage.ty.layout, b_buf.ty.layout.storage())
    assert not b_storage.ty.layout.is_trivial()
    assert not b_custom.ty.layout.is_trivial()

    code = func.script()
    storage_line = next(line for line in code.splitlines() if "B_storage =" in line)
    custom_line = next(line for line in code.splitlines() if "B_custom =" in line)
    assert ".local(layout=" in storage_line
    assert ".local(2, 4, layout=" in custom_line
    assert_structural_equal(func, from_source(code))
    assert from_source(code).script() == code


def test_buffer_local_explicit_layout_without_parent_layout():
    """An explicit shape and layout do not inspect the parent's absent layout."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer((4,), dtype="float32", scope="local", layout=None)
        B = A.local(4, layout=T.TileLayout(T.S[4]))
        B[0] = T.float32(1)
        # fmt: on

    bufs = _collect_buffers(func)
    assert bufs["A"].ty.layout is None
    assert bufs["B"].ty.layout.is_trivial()
    code = func.script()
    parsed = from_source(code)
    assert_structural_equal(func, parsed)
    assert parsed.script() == code


def test_buffer_local_compose_layout_printer_roundtrip():
    """Generic view sugar keeps a physical local view's identity layout."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            (8, 8),
            dtype="float32",
            scope="local",
            layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(8, 8)])),
        )
        B = A.local()
        B[0] = T.float32(1)
        # fmt: on

    bufs = _collect_buffers(func)
    assert [int(dim) for dim in bufs["B"].ty.shape] == [64]
    assert bufs["B"].ty.layout.is_trivial()
    code = func.script()
    local_line = next(line for line in code.splitlines() if "B =" in line)
    assert ".view(64, layout=" in local_line
    parsed = from_source(code)
    assert_structural_equal(func, parsed)
    assert parsed.script() == code


def test_buffer_local_inference_without_parent_layout_has_clear_diagnostic():
    """Shape inference requires a parent storage layout."""

    with pytest.raises(tvm.error.DiagnosticError, match="parent buffer has layout=None"):
        # fmt: off
        @T.prim_func
        def func() -> None:
            T.device_entry()
            A = T.alloc_buffer((4,), dtype="float32", scope="local", layout=None)
            B = A.local(layout=T.TileLayout(T.S[4]))
            B[0] = T.float32(1)
            # fmt: on


def test_buffer_local_physical_span_includes_gaps_and_offset():
    """The raw local view includes every slot up to the storage span."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([6], dtype="float32", scope="local")
        B = A.view(32, 2, layout=T.TileLayout(T.S[(32, 2) : (1 @ laneid, 2)] + 3))
        B_flat = B.local()
        B_2d = B.local(2, 3)
        B_storage = B.local(2, layout=B.layout.storage())
        B_flat[5] = T.float32(1)
        B_2d[1, 2] = T.float32(2)
        B_storage[1] = T.float32(3)
        # fmt: on

    bufs = _collect_buffers(func)
    b_buf = bufs["B"]
    b_flat = bufs["B_flat"]
    b_2d = bufs["B_2d"]
    b_storage = bufs["B_storage"]
    assert int(b_buf.ty.layout.storage().span()) == 6
    assert int(b_buf.ty.layout.storage().size()) == 2
    assert [int(dim) for dim in b_flat.ty.shape] == [6]
    assert [int(dim) for dim in b_2d.ty.shape] == [2, 3]
    assert [int(dim) for dim in b_storage.ty.shape] == [2]
    for local in [b_flat, b_2d]:
        assert local.ty.layout.is_trivial()
    for i in range(6):
        assert int(b_flat.ty.layout.apply(i, shape=list(b_flat.ty.shape))["m"]) == i
    assert int(b_2d.ty.layout.apply(1, 2, shape=list(b_2d.ty.shape))["m"]) == 5
    assert_structural_equal(b_storage.ty.layout, b_buf.ty.layout.storage())
    assert int(b_storage.ty.layout.apply(0, shape=list(b_storage.ty.shape))["m"]) == 3
    assert int(b_storage.ty.layout.apply(1, shape=list(b_storage.ty.shape))["m"]) == 5

    code = func.script()
    storage_line = next(line for line in code.splitlines() if "B_storage =" in line)
    assert ".local(layout=" in storage_line
    assert_structural_equal(func, from_source(code))
    assert from_source(code).script() == code


def test_buffer_local_printer_is_stable_with_multiple_aliases():
    """Thread-layout parents win deterministically over sibling aliases."""
    from tvm.tirx.layout import tcgen05_atom_layout

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([32], dtype="float32", scope="local")
        B = A.view(64, 64, layout=tcgen05_atom_layout("16x256b", (64, 64), "float32"))
        B_flat = B.local()
        B_2d = B.local(4, 8)
        B_storage = B.local(layout=B.layout.storage())
        B_flat[0] = B_2d[0, 0] + B_storage[0]
        # fmt: on

    expected = func.script()
    assert "B_flat = B.local()" in expected
    assert "B_2d = B.local(4, 8)" in expected
    storage_line = next(line for line in expected.splitlines() if "B_storage =" in line)
    assert ".local(layout=" in storage_line
    for _ in range(20):
        parsed = from_source(expected)
        assert parsed.script() == expected
        assert_structural_equal(func, parsed)


def test_buffer_local_printer_preserves_inherited_metadata():
    """Local sugar falls back when it would discard Buffer metadata."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            [32, 2],
            dtype="float32",
            elem_offset=8,
            scope="local",
            layout=T.TileLayout(T.S[(32, 2) : (1 @ laneid, 2)]),
        )
        B_align = T.decl_buffer(
            (2,),
            dtype="float32",
            data=A.data,
            elem_offset=8,
            scope="local",
            align=128,
        )
        B_factor = T.decl_buffer(
            (2,),
            dtype="float32",
            data=A.data,
            elem_offset=8,
            scope="local",
            offset_factor=8,
        )
        B_align[0] = B_factor[0]
        # fmt: on

    code = func.script()
    align_line = next(line for line in code.splitlines() if "B_align =" in line)
    factor_line = next(line for line in code.splitlines() if "B_factor =" in line)
    assert "T.decl_buffer" in align_line and "align=128" in align_line
    assert "T.decl_buffer" in factor_line and "offset_factor=8" in factor_line
    assert ".local(" not in align_line
    assert ".local(" not in factor_line
    parsed = from_source(code)
    assert_structural_equal(func, parsed)
    assert parsed.script() == code


def test_buffer_local_rejects_shape_that_does_not_match_physical_span():
    """An explicit local shape product must preserve the physical span."""

    with pytest.raises(tvm.error.DiagnosticError, match="physical storage span 6 per thread"):
        # fmt: off
        @T.prim_func
        def func() -> None:
            T.device_entry()
            A = T.alloc_buffer([6], dtype="float32", scope="local")
            B = A.view(32, 2, layout=T.TileLayout(T.S[(32, 2) : (1 @ laneid, 2)] + 3))
            B_local = B.local(2)
            B_local[0] = T.float32(0)
            # fmt: on


def test_pointer_expression_assignment_uses_bind():
    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        buf = T.alloc_buffer((4,), "uint32", scope="shared")
        ptr = buf.ptr_to([1])
        T.evaluate(T.reinterpret("uint64", ptr))
    # fmt: on

    binds = []
    tvm.tirx.stmt_functor.post_order_visit(
        func.body, lambda node: binds.append(node) if isinstance(node, tvm.tirx.Bind) else None
    )
    assert len(binds) == 1
    assert isinstance(binds[0].var.ty, PointerType)
    assert_structural_equal(binds[0].var.ty, binds[0].value.ty)

    code = func.script()
    assert_structural_equal(func, from_source(code))


def test_pointer_expression_assignment_rejects_reassignment():
    with pytest.raises(tvm.error.DiagnosticError, match="cannot be reassigned"):
        # fmt: off
        @T.prim_func
        def func() -> None:
            T.device_entry()
            buf = T.alloc_buffer((4,), "uint32", scope="shared")
            ptr = buf.ptr_to([0])
            ptr = buf.ptr_to([1])
            T.evaluate(T.reinterpret("uint64", ptr))
        # fmt: on


def test_pointer_expression_assignment_can_shadow_extra_var():
    source = """
@T.prim_func
def func() -> None:
    T.device_entry()
    buf = T.alloc_buffer((4,), "uint32", scope="shared")
    ptr = buf.ptr_to([1])
    view = T.decl_buffer((3,), "uint32", data=ptr, scope="shared")
    view[0] = T.uint32(0)
"""
    func = tvm.script.from_source(source, extra_vars={"T": T, "ptr": object()})

    binds = []
    tvm.tirx.stmt_functor.post_order_visit(
        func.body, lambda node: binds.append(node) if isinstance(node, tvm.tirx.Bind) else None
    )
    assert len(binds) == 1
    assert_structural_equal(func, from_source(func.script()))


def test_buffer_permute_ir():
    """Verify .permute(1, 0): shape swapped, layout permuted, shared data."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([8, 4], dtype="float16", scope="local",
                            layout=T.TileLayout(T.S[(8, 4) : (4, 1)]))
        B = A.permute(1, 0)
        B[0, 0] = T.float16(0)
        # fmt: on

    bufs = _collect_buffers(func)
    a_buf = bufs["A"]
    b_buf = bufs["B"]

    # Shared data pointer
    assert_structural_equal(_collect_buffer_sources(func)["B"], a_buf.data)
    # Shape: [4, 8] from [8, 4]
    assert int(b_buf.ty.shape[0]) == 4
    assert int(b_buf.ty.shape[1]) == 8
    # Layout: permuted
    assert_structural_equal(b_buf.ty.layout, a_buf.ty.layout.permute_dims([1, 0]))

    code = func.script()
    assert from_source(code).script() == code


def test_buffer_rearrange_allows_arbitrary_axis_names():
    @T.prim_func
    def ordinary_axis() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            (8, 4),
            "float16",
            scope="local",
            layout=T.TileLayout(T.S[(8, 4) : (4, 1)]),
        )
        B = A.rearrange("(outer inner) tail -> outer tail inner", outer=2)
        B[0, 0, 0] = T.float16(0)

    @T.prim_func
    def buf_axis() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            (8, 4),
            "float16",
            scope="local",
            layout=T.TileLayout(T.S[(8, 4) : (4, 1)]),
        )
        B = A.rearrange("(buf inner) tail -> buf tail inner", buf=2)
        B[0, 0, 0] = T.float16(0)

    @T.prim_func
    def self_axis() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            (8, 4),
            "float16",
            scope="local",
            layout=T.TileLayout(T.S[(8, 4) : (4, 1)]),
        )
        B = A.rearrange("(self inner) tail -> self tail inner", self=2)
        B[0, 0, 0] = T.float16(0)

    @T.prim_func
    def pattern_axis() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            (8, 4),
            "float16",
            scope="local",
            layout=T.TileLayout(T.S[(8, 4) : (4, 1)]),
        )
        B = A.rearrange("(pattern inner) tail -> pattern tail inner", pattern=2)
        B[0, 0, 0] = T.float16(0)

    @T.prim_func
    def keyword_pattern() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            (8, 4),
            "float16",
            scope="local",
            layout=T.TileLayout(T.S[(8, 4) : (4, 1)]),
        )
        B = A.rearrange(pattern="(outer inner) tail -> outer tail inner", outer=2)
        B[0, 0, 0] = T.float16(0)

    expected = _collect_buffers(ordinary_axis)["B"]
    for func in (buf_axis, self_axis, pattern_axis, keyword_pattern):
        actual = _collect_buffers(func)["B"]
        assert_structural_equal(actual.shape, expected.shape)
        assert_structural_equal(actual.layout, expected.layout)


def test_buffer_permute_compose_layout_ir():
    """Verify .permute on a swizzle-composed layout: the swizzle is preserved
    and the inner tile layout's dim groups are permuted (the reshape-permute-
    reshape idiom used to refactor gather views without restating strides)."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer(
            [4, 4, 4, 64], dtype="bfloat16", scope="shared.dyn",
            layout=T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(4, 4, 4, 64) : (1024, 256, 64, 1)])),
        )
        B = A.permute(1, 0, 2, 3)
        B[0, 0, 0, 0] = T.bfloat16(0)
        # fmt: on

    bufs = _collect_buffers(func)
    a_buf = bufs["A"]
    b_buf = bufs["B"]

    assert_structural_equal(_collect_buffer_sources(func)["B"], a_buf.data)
    assert [int(s) for s in b_buf.shape] == [4, 4, 4, 64]
    expected = tvm.tirx.layout.ComposeLayout(
        a_buf.layout.per_element,
        a_buf.layout.swizzle_len,
        a_buf.layout.atom_len,
        a_buf.layout.tile_layout.permute_dims([1, 0, 2, 3]),
        a_buf.layout.swizzle_inner,
    )
    assert_structural_equal(b_buf.layout, expected)

    code = func.script()
    assert from_source(code).script() == code


def test_buffer_sub_multi_iter_dim_ir():
    """sub with an int index on a dim carried by several layout iters
    decomposes the index mixed-radix across the iters' strides."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([8, 16], dtype="float16", scope="local",
                           layout=T.TileLayout(T.S[(2, 4, 16) : (1024, 64, 1)]))
        B = A.sub[5]
        B[0] = T.float16(0)
        # fmt: on

    bufs = _collect_buffers(func)
    a_buf, b_buf = bufs["A"], bufs["B"]
    # 5 -> (5 // 4, 5 % 4) = (1, 1) -> 1 * 1024 + 1 * 64
    assert int(tvm.arith.Analyzer().simplify(b_buf.elem_offset - a_buf.elem_offset)) == 1088
    assert [int(s) for s in b_buf.shape] == [16]
    assert_structural_equal(b_buf.layout, tvm.tirx.layout.TileLayout(T.S[(16,) : (1,)]))

    code = func.script()
    assert from_source(code).script() == code


def test_buffer_sub_multi_iter_misaligned_rejected():
    buf = tvm.tirx.decl_buffer(
        (8, 16), "float16", layout=tvm.tirx.layout.TileLayout(T.S[(2, 4, 16) : (1024, 64, 1)])
    )
    # sub[2:6] narrows the multi-iter dim 0 at a misaligned offset.
    with pytest.raises(ValueError, match="multiples of the inner iter block"):
        buf.sub[2:6]


def test_buffer_sub_ir():
    """buf.sub follows numpy basic indexing as a view constructor: int drops
    the dim, a:b narrows, a::s strides. Offsets fold into elem_offset through
    the dim's layout iter strides; the derived layout carries the survivors."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([4, 8, 16], dtype="float16", scope="local",
                           layout=T.TileLayout(T.S[(4, 8, 16) : (256, 16, 1)]))
        B = A.sub[1, 2:6]
        B[0, 0] = T.float16(0)
        C = A.sub[:, 1::2]
        C[0, 0, 0] = T.float16(0)
        # fmt: on

    bufs = _collect_buffers(func)
    a_buf, b_buf, c_buf = bufs["A"], bufs["B"], bufs["C"]
    # sub[1, 2:6]: drop dim 0 at 1 (1 * 256) then narrow dim 1 to [2, 6) (2 * 16)
    assert [int(s) for s in b_buf.shape] == [4, 16]
    assert int(tvm.arith.Analyzer().simplify(b_buf.elem_offset - a_buf.elem_offset)) == 288
    assert_structural_equal(b_buf.layout, tvm.tirx.layout.TileLayout(T.S[(4, 16) : (16, 1)]))
    # sub[:, 1::2]: keep dim 0, split dim 1 into (4, 2) and fix the remainder at 1
    assert [int(s) for s in c_buf.shape] == [4, 4, 16]
    assert int(tvm.arith.Analyzer().simplify(c_buf.elem_offset - a_buf.elem_offset)) == 16
    assert_structural_equal(
        c_buf.layout, tvm.tirx.layout.TileLayout(T.S[(4, 4, 16) : (256, 32, 1)])
    )

    code = func.script()
    assert from_source(code).script() == code


def test_buffer_view_surgery_static_bounds_rejected():
    """Statically-known out-of-range sub arguments must be rejected loudly
    (review finding: OOB offsets were silent)."""
    buf = tvm.tirx.decl_buffer(
        (10,), "float16", layout=tvm.tirx.layout.TileLayout(T.S[(10,) : (1,)])
    )
    grid = tvm.tirx.decl_buffer(
        (4, 8), "float16", layout=tvm.tirx.layout.TileLayout(T.S[(4, 8) : (8, 1)])
    )
    # int index: static bounds
    with pytest.raises(ValueError, match="out of range"):
        buf.sub[10]
    with pytest.raises(ValueError, match="out of range"):
        buf.sub[-1]
    # slice narrow: static range bounds
    with pytest.raises(ValueError, match="exceeds"):
        buf.sub[8:12]
    with pytest.raises(ValueError, match="must be non-negative"):
        buf.sub[-2:2]
    with pytest.raises(ValueError, match="must be positive"):
        buf.sub[5:3]
    # grid.sub: out-of-range int, exceeding narrow, stepped-start out of range
    with pytest.raises(ValueError, match="out of range"):
        grid.sub[10, :]
    with pytest.raises(ValueError, match="exceeds"):
        grid.sub[:, 4:12]
    with pytest.raises(ValueError, match=r"in \[0, 2\)"):
        grid.sub[:, -1::2]


def test_buffer_sub_swizzle_commutation():
    """A folded view offset moves into elem_offset only when it commutes
    with the swizzle, i.e. is a multiple of the swizzle period
    2^(per_element + atom_len + swizzle_len). Sub-period offsets stay inside
    the derived tile layout's offset so the swizzle keeps applying to them
    (review finding: folding them outside produced wrong addresses). Both
    placements must be address-equivalent to the parent layout."""

    def addr(buf, base, *coords):
        analyzer = tvm.arith.Analyzer()
        if len(coords) == 1:
            rel = buf.layout.apply(coords[0])["m"]
        else:
            rel = buf.layout.apply(*coords, shape=[int(s) for s in buf.shape])["m"]
        return int(analyzer.simplify((buf.elem_offset - base) + rel))

    analyzer = tvm.arith.Analyzer()
    compose = T.ComposeLayout(
        3, 3, 3, T.TileLayout(T.S[(4, 1024) : (1024, 1)])
    )  # period = 2^(3+3+3) = 512 elements

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([4, 1024], dtype="bfloat16", scope="shared.dyn", layout=compose)
        B = A.sub[1]  # offset 1024 = 2 * period: folds into elem_offset
        B[0] = T.bfloat16(0)
        C = A.sub[:, 512:1024]  # offset 512 = period: folds into elem_offset
        C[0, 0] = T.bfloat16(0)
        # fmt: on

    bufs = _collect_buffers(func)
    a_buf, b_buf, c_buf = bufs["A"], bufs["B"], bufs["C"]
    base = a_buf.elem_offset
    assert int(analyzer.simplify(b_buf.elem_offset - base)) == 1024
    for j in (0, 1, 63, 511, 1023):
        assert addr(a_buf, base, 1024 + j) == addr(b_buf, base, j)
    for j in (0, 1, 255, 511):
        assert addr(a_buf, base, 512 + j) == addr(c_buf, base, 0, j)

    code = func.script()
    assert from_source(code).script() == code

    # Sub-period offsets do not commute: they stay inside the tile layout's
    # offset (elem_offset unchanged) and every address matches the parent.
    compose2 = T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(2, 16, 8) : (128, 8, 1)]))

    # fmt: off
    @T.prim_func
    def func2() -> None:
        T.device_entry()
        A = T.alloc_buffer([2, 16, 8], dtype="float16", scope="shared.dyn", layout=compose2)
        B = A.sub[:, 1]  # offset 8
        B[0, 0] = T.float16(0)
        C = A.sub[:, :, 1]  # offset 1
        C[0, 0] = T.float16(0)
        D = A.sub[:, 1:3]  # offset 8
        D[0, 0, 0] = T.float16(0)
        E = A.sub[:, 1:3]  # narrow via sub
        E[0, 0, 0] = T.float16(0)
        for w in T.serial(16):
            F = A.sub[:, w]  # dynamic sub-period offset
            F[0, 0] = T.float16(0)
        # fmt: on

    bufs = _collect_buffers(func2)
    a2, base2 = bufs["A"], bufs["A"].elem_offset
    shape2 = [2, 16, 8]
    for name, to_parent in {
        "B": lambda c: (c[0], 1, c[1]),
        "C": lambda c: (c[0], c[1], 1),
        "D": lambda c: (c[0], 1 + c[1], c[2]),
        "E": lambda c: (c[0], 1 + c[1], c[2]),
    }.items():
        child = bufs[name]
        assert int(analyzer.simplify(child.elem_offset - base2)) == 0
        child_shape = [int(s) for s in child.shape]
        for flat in range(math.prod(child_shape)):
            coords, rem = [], flat
            for extent in reversed(child_shape):
                coords.append(rem % extent)
                rem //= extent
            coords = tuple(reversed(coords))
            assert addr(a2, base2, *to_parent(coords)) == addr(child, base2, *coords), (
                name,
                coords,
            )

    code = func2.script()
    assert from_source(code).script() == code

    # fixed-point windows (all touched addresses below 2^(per_element +
    # atom_len)) are correct through the same layout-offset placement
    compose3 = T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(64,) : (1,)]))

    # fmt: off
    @T.prim_func
    def func3() -> None:
        T.device_entry()
        A = T.alloc_buffer([64], dtype="bfloat16", scope="shared.dyn", layout=compose3)
        B = A.sub[8:16]
        B[0] = T.bfloat16(0)
        # fmt: on

    bufs = _collect_buffers(func3)
    a3, b3 = bufs["A"], bufs["B"]
    for j in range(8):
        assert addr(a3, a3.elem_offset, 8 + j) == addr(b3, a3.elem_offset, j) == 8 + j


def test_buffer_tile_ir():
    """buf.tile((dim, factors))[picks] splits dims into factors and picks
    chunks in one call: int/Expr picks a factor, ':' keeps it, kept
    factors merge back. Equivalent to the view (reshape) + sub chain."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([3, 64, 512], dtype="float16", scope="shared",
                           layout=T.TileLayout(T.S[(3, 64, 512) : (64 * 512, 512, 1)]))
        for w in T.serial(4):
            B = A.tile((1, (-1, 4, 4)))[:, w, :]
            B[0, 0, 0] = T.float16(0)
            C = A.view(3, 4, 4, 4, 512).sub[:, :, w].view(3, 16, 512)
            C[0, 0, 0] = T.float16(0)
        D = A.tile((1, (-1, 4)))[:, 2]
        D[0, 0, 0] = T.float16(0)
        E = A.sub[:, 2::4]
        E[0, 0, 0] = T.float16(0)
        F = A.tile((1, (4, -1)))[2, :]
        F[0, 0, 0] = T.float16(0)
        G = A.sub[:, 32:48]
        G[0, 0, 0] = T.float16(0)

    @T.prim_func
    def func_multi() -> None:
        T.device_entry()
        A = T.alloc_buffer([64, 128], dtype="float16", scope="shared",
                           layout=T.TileLayout(T.S[(64, 128) : (128, 1)]))
        for wx in T.serial(4):
            for wy in T.serial(2):
                H = A.tile((0, (-1, 4, 2)), (1, (-1, 2, 8)))[:, wx, :, :, wy, :]
                H[0, 0] = T.float16(0)
                J = (A.view(8, 4, 2, 128).sub[:, wx].view(16, 128)
                      .view(16, 8, 2, 8).sub[:, :, wy].view(16, 64))
                J[0, 0] = T.float16(0)

    @T.prim_func
    def func_multipick() -> None:
        T.device_entry()
        A = T.alloc_buffer([128, 16], dtype="float16", scope="shared",
                           layout=T.TileLayout(T.S[(128, 16) : (16, 1)]))
        for a in T.serial(2):
            for b in T.serial(4):
                K = A.tile((0, (2, 4, -1)))[a, b, :]
                K[0, 0] = T.float16(0)
                L = A.view(2, 4, 16, 16).sub[a, b]
                L[0, 0] = T.float16(0)
    # fmt: on

    b = _collect_buffers(func)
    assert [int(s) for s in b["B"].shape] == [3, 16, 512]
    assert_structural_equal(b["B"].layout, b["C"].layout)
    assert_structural_equal(b["D"].layout, b["E"].layout)
    assert_structural_equal(b["F"].layout, b["G"].layout)
    m = _collect_buffers(func_multi)
    assert [int(s) for s in m["H"].shape] == [16, 64]
    assert_structural_equal(m["H"].layout, m["J"].layout)
    mp = _collect_buffers(func_multipick)
    assert [int(s) for s in mp["K"].shape] == [16, 16]
    assert_structural_equal(mp["K"].layout, mp["L"].layout)

    code = func_multipick.script()
    assert from_source(code).script() == code


def test_buffer_tile_rejected():
    buf = tvm.tirx.decl_buffer(
        (3, 64, 512),
        "float16",
        layout=tvm.tirx.layout.TileLayout(T.S[(3, 64, 512) : (64 * 512, 512, 1)]),
    )
    with pytest.raises(ValueError, match="takes a dim and a factors"):
        buf.tile(1, 4, 4)  # positional single-dim form takes exactly (dim, factors)
    with pytest.raises(ValueError, match="picks no factor"):
        buf.tile(1, (-1, 4))[:, :]  # a chunk must pick at least one factor
    with pytest.raises(ValueError, match="non-empty tuple"):
        buf.tile((1, 4))  # factors must be a tuple
    with pytest.raises(ValueError, match="non-empty tuple"):
        buf.tile((1, ()))
    with pytest.raises(ValueError, match="tiled more than once"):
        buf.tile((1, (4, -1)), (1, (2, -1)))
    with pytest.raises(ValueError, match="index"):
        buf.tile((1, (-1, 4)))[2]  # 2 factors, 1 index
    with pytest.raises(ValueError, match="must be ':'"):
        buf.tile((1, (-1, 4)))[:, 1:3]  # sub-slice on a factor


def test_buffer_chunk_ir():
    """buf.chunk(spec)[picks] narrows each chunked dim to its picked chunk's
    contiguous [c*k : (c+1)*k) range (k = E // n), rank-preserving: a per-dim
    tuple where None passes the pick straight through and n divides that dim
    into n equal chunks. chunk(spec)[picks] is the exact same BufferRegion as
    the hand-written a*k:(a+1)*k slice — no reshape, no extra dim."""

    from tvm.tirx.stmt import BufferRegion

    compose = T.ComposeLayout(3, 3, 3, T.TileLayout(T.S[(4, 512) : (512, 1)]))
    A = tvm.tirx.decl_buffer(
        (4, 8, 16), "float16", layout=tvm.tirx.layout.TileLayout(T.S[(4, 8, 16) : (128, 16, 1)])
    )
    C = tvm.tirx.decl_buffer((4, 512), "bfloat16", layout=compose)

    # chunk((None, None, 2))[:, :, 1] narrows dim 2 (extent 16) to chunk 1 of 2
    # → [8:16] (k = 16 // 2 = 8); rank preserved, dims 0/1 pass through as ':'.
    reg = A.chunk((None, None, 2))[:, :, 1]
    assert isinstance(reg, BufferRegion)
    assert len(reg.region) == 3  # rank-preserving: no extra extent-1 chunk dim
    assert (int(reg.region[2].min), int(reg.region[2].extent)) == (8, 8)
    assert_structural_equal(reg, A[:, :, 8:16])

    # a None dim passes an int pick straight through (int → extent-1 region),
    # while the chunked dim still narrows to its picked chunk.
    reg2 = A.chunk((None, None, 2))[3, :, 0]
    assert_structural_equal(reg2, A[3, :, 0:8])

    # chunk((None, 4))[:, 2] on the swizzle-carrying compose layout: dim 1
    # (extent 512) → chunk 2 of 4 → [256:384] (k = 128), byte-identical slice.
    reg_c = C.chunk((None, 4))[:, 2]
    assert (int(reg_c.region[1].min), int(reg_c.region[1].extent)) == (256, 128)
    assert_structural_equal(reg_c, C[:, 256:384])

    # a symbolic (Expr) chunk index translates to c*k : (c+1)*k as well.
    c = T.Var(name="c", ty="int32")
    assert_structural_equal(A.chunk((None, None, 2))[:, :, c], A[:, :, c * 8 : c * 8 + 8])

    # validation
    with pytest.raises(ValueError, match="per-dim tuple"):
        A.chunk(2)  # spec must be a per-dim tuple, not a bare int
    with pytest.raises(ValueError, match="spec length"):
        A.chunk((None, 2))  # length 2 != rank 3
    with pytest.raises(ValueError, match="None or a positive int"):
        A.chunk((None, None, 0))  # 0 is not a positive chunk count
    with pytest.raises(ValueError, match="chunk index, not a slice"):
        A.chunk((None, None, 2))[:, :, 0:1]  # a chunked dim takes a chunk index
    with pytest.raises(ValueError, match="rank-3 spec"):
        A.chunk((None, None, 2))[0, 0, 0, 0]  # too many indices


def test_buffer_view_dtype_ir():
    """Verify .view('float32') on float16: dtype correct, last dim halved, shared data."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        A = T.alloc_buffer([8, 8], dtype="float16", scope="local")
        B = A.view("float32")
        B[0, 0] = T.float32(0)
        # fmt: on

    bufs = _collect_buffers(func)
    a_buf = bufs["A"]
    b_buf = bufs["B"]

    # Shared data pointer
    assert_structural_equal(_collect_buffer_sources(func)["B"], a_buf.data)
    # dtype
    assert str(b_buf.ty.dtype) == "float32"
    # Shape: [8, 4] (last dim halved since float32 is 2x float16)
    assert int(b_buf.ty.shape[0]) == 8
    assert int(b_buf.ty.shape[1]) == 4

    code = func.script()
    assert from_source(code).script() == code


def test_buffer_slice_region():
    """Verify A[slice] returns BufferRegion (not DeclBuffer)."""
    from tvm.tirx.stmt import BufferRegion

    buf = tvm.tirx.decl_buffer((128, 64), "float16")
    br = buf[32:64, 0:32]
    assert isinstance(br, BufferRegion)
    assert br.buffer.same_as(buf)
    assert int(br.region[0].extent) == 32
    assert int(br.region[1].extent) == 32

    load = buf[1, 2]
    assert isinstance(load, tvm.ir.TensorLoad)

    partial = buf[1]
    assert isinstance(partial, BufferRegion)
    with pytest.raises(TypeError):
        _ = partial[2]


def test_global_call_realizes_buffer_elements():
    @I.ir_module(s_tir=True)
    class Module:
        @T.prim_func(private=True, s_tir=True)
        def add(a: T.float32, b: T.float32) -> T.float32:
            return a + b

        @T.prim_func(s_tir=True)
        def main(
            A: T.Buffer((16,), "float32"),
            B: T.Buffer((16,), "float32"),
            C: T.Buffer((16,), "float32"),
        ):
            for i in range(16):
                C[i] = Module.add(A[i], B[i])

    assert isinstance(Module["main"], tvm.tirx.PrimFunc)


def test_roundtrip_serial_unroll_false():
    """T.serial(N, unroll=False) should round-trip."""

    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128,), "float32", scope="global")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        for _ in T.serial(10, unroll=False):
            Tx.cta.fill(A[0:32], T.float32(0))
        # fmt: on

    code = test.script()
    assert "unroll=False" in code, f"printer should emit unroll=False, got:\n{code}"
    assert "annotations" not in code, "printer should NOT emit annotations dict"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_serial_unroll_true():
    """T.serial(N, unroll=True) should round-trip as a pragma-unroll request."""

    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128,), "float32", scope="global")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        for _ in T.serial(10, unroll=True):
            Tx.cta.fill(A[0:32], T.float32(0))
        # fmt: on

    code = test.script()
    assert "unroll=True" in code, f"printer should emit unroll=True, got:\n{code}"
    assert "annotations" not in code, "printer should NOT emit annotations dict"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_serial_unroll_count():
    """T.serial(N, unroll=2) should preserve the requested unroll count."""

    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128,), "float32", scope="global")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        for _ in T.serial(10, unroll=2):
            Tx.cta.fill(A[0:32], T.float32(0))
        # fmt: on

    code = test.script()
    assert "unroll=2" in code, f"printer should emit unroll=2, got:\n{code}"
    assert "annotations" not in code, "printer should NOT emit annotations dict"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_serial_unroll_false_with_other_annotations():
    """When other annotations exist alongside disable_unroll, fall back to full dict."""

    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128,), "float32", scope="global")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        for _ in T.serial(10, annotations={"disable_unroll": True, "custom": 42}):
            Tx.cta.fill(A[0:32], T.float32(0))
        # fmt: on

    code = test.script()
    assert "annotations=" in code, "printer should emit full annotations when multiple keys exist"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_unary_inplace():
    """Single-arg unary ops (in-place) should round-trip."""

    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128,), "float32", scope="global")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        Tx.warp.exp2(A[0:32])
        Tx.warp.sqrt(A[32:64])
        Tx.warp.reciprocal(A[64:96])
        # fmt: on

    code = test.script()
    # Each op should appear with a single arg (no duplicate src, no trailing Nones)
    assert 'T.warp.exp2(A[0:32])' in code, f"expected single-arg exp2, got:\n{code}"
    assert 'T.warp.sqrt(A[32:64])' in code, f"expected single-arg sqrt, got:\n{code}"
    assert 'T.warp.reciprocal(A[64:96])' in code, (
        f"expected single-arg reciprocal, got:\n{code}"
    )
    assert "None" not in code, f"trailing None args should be trimmed:\n{code}"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_unary_different_dst_src():
    """Unary ops with different dst and src should keep both args."""

    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128,), "float32", scope="global")
        B = T.match_buffer(B_ptr, (128,), "float32", scope="global")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        Tx.warp.exp2(A[0:32], B[0:32])
        # fmt: on

    code = test.script()
    assert 'T.warp.exp2(A[0:32], B[0:32])' in code, (
        f"different dst/src should keep both:\n{code}"
    )
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_persistent_decorator():
    """@T.prim_func(persistent=True) should round-trip."""

    # fmt: off
    @T.prim_func(persistent=True)
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128,), "float32", scope="global")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        Tx.cta.fill(A[0:32], T.float32(0))
        # fmt: on

    code = test.script()
    assert "persistent=True" in code, f"persistent not in decorator:\n{code}"
    assert "tirx.persistent_kernel" not in code, "should NOT appear as func_attr"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_persistent_not_present():
    """Without persistent=True, the keyword should not appear."""

    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128,), "float32", scope="global")
        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])
        Tx.cta.fill(A[0:32], T.float32(0))
        # fmt: on

    code = test.script()
    assert "persistent" not in code, f"persistent should NOT appear:\n{code}"


def test_warp_role():
    """WarpRole should emit guarded warp scopes plus setmaxnreg."""
    from tvm.tirx.lang.warp_role import WarpRole

    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128,), "float32", scope="global")
        T.device_entry()
        cta_id = T.cta_id([1])
        wg_id = T.warpgroup_id([4])
        warp_id = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
        with WarpRole(warp_id, 1, regs=48):
            Tx.cta.fill(A[0:32], T.float32(0))
        with WarpRole(warp_id, 0, regs=232, increase=True):
            Tx.cta.fill(A[32:64], T.float32(1))
        # fmt: on

    code = test.script()
    assert "warp_id == 1" in code, f"should have warp_id==1 guard:\n{code}"
    assert "warp_id == 0" in code, f"should have warp_id==0 guard:\n{code}"
    assert "setmaxnreg" in code, f"should have setmaxnreg:\n{code}"
    assert "if warp_id == 1:" in code, f"should have warp_id==1 if-guard:\n{code}"
    assert "if warp_id == 0:" in code, f"should have warp_id==0 if-guard:\n{code}"
    # The printed code is valid TIR — it should parse back
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_warpgroup_role():
    """WarpgroupRole should emit guarded warpgroup scope plus setmaxnreg."""
    from tvm.tirx.lang.warp_role import WarpgroupRole

    # fmt: off
    @T.prim_func
    def test(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (128,), "float32", scope="global")
        T.device_entry()
        cta_id = T.cta_id([1])
        wg_id = T.warpgroup_id([4])
        warp_id_in_wg = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
        with WarpgroupRole(wg_id, 2, regs=200, increase=True):
            Tx.cta.fill(A[0:32], T.float32(0))
        # fmt: on

    code = test.script()
    assert "wg_id == 2" in code, f"should have wg_id==2 guard:\n{code}"
    assert "setmaxnreg" in code, f"should have setmaxnreg:\n{code}"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_vector_annotation_syntax_1d():
    """Test x: T.f32[N] produces the same IR as T.alloc_local([N], 'float32')."""

    # fmt: off
    @T.prim_func
    def func():
        T.device_entry()
        v: T.float32[8]
        T.evaluate(v[0])  # noqa: F821

    @T.prim_func
    def func():  # noqa: F811
        T.device_entry()
        v = T.alloc_local([8], "float32")
        T.evaluate(v[0])
        # fmt: on

        # func was redefined; compare first (annotation) with second (alloc_local).
        # Re-create the annotation version for comparison:

        # fmt: off
    @T.prim_func
    def annotation_func():
        T.device_entry()
        v: T.float32[8]
        T.evaluate(v[0])  # noqa: F821
        # fmt: on

        # Verify both produce valid IR that round-trips through printer/parser
    code = func.script()
    assert from_source(code).script() == code
    code2 = annotation_func.script()
    assert from_source(code2).script() == code2
    # The printed form should be identical (both become alloc_local in print)
    assert code.replace("annotation_func", "func") == code


def test_vector_annotation_syntax_multidim():
    """Test x: T.f32[M, N] produces the same IR as T.alloc_local([M, N], 'float32')."""

    # fmt: off
    @T.prim_func
    def func():
        T.device_entry()
        m: T.float32[4, 8]
        T.evaluate(m[0, 0])  # noqa: F821
        # fmt: on

    code = func.script()
    assert "alloc_local((4, 8)" in code or "float32[4, 8]" in code
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_buffer_sub_tmem_offset_uses_physical_columns():
    """A tmem layout measures TCol in elements, but allocated_addr measures
    physical 32-bit columns.  Folding a sub-view offset must scale by dtype
    width exactly once (the FlashMLA Q-tail view is the bf16 regression)."""

    # fmt: off
    @T.prim_func
    def func() -> None:
        T.device_entry()
        Q = T.decl_buffer(
            (2, 64, 288), "bfloat16", scope="tmem", allocated_addr=256,
            layout=T.TileLayout(T.S[(2, 64, 288) : (64 @ TLane, 1 @ TLane, 1 @ TCol)]),
        )
        Q_tail = Q.sub[:, :, 256:288]
        F8 = T.decl_buffer(
            (64, 128), "float8_e4m3fn", scope="tmem", allocated_addr=32,
            layout=T.TileLayout(T.S[(64, 128) : (1 @ TLane, 1 @ TCol)]),
        )
        F8_tail = F8.sub[:, 64:96]
        F32 = T.decl_buffer(
            (64, 128), "float32", scope="tmem", allocated_addr=64,
            layout=T.TileLayout(T.S[(64, 128) : (1 @ TLane, 1 @ TCol)]),
        )
        F32_tail = F32.sub[:, 32:64]
        T.evaluate(Q_tail[0, 0, 0])
        T.evaluate(F8_tail[0, 0])
        T.evaluate(F32_tail[0, 0])
        # fmt: on

    bufs = _collect_buffers(func)
    assert int(bufs["Q_tail"].allocated_addr[0]) == 384  # 256 + 256 * 16 / 32
    assert int(bufs["F8_tail"].allocated_addr[0]) == 48  # 32 + 64 * 8 / 32
    assert int(bufs["F32_tail"].allocated_addr[0]) == 96  # 64 + 32 * 32 / 32
    for name in ("Q_tail", "F8_tail", "F32_tail"):
        assert int(bufs[name].layout.offset.get(TCol, 0)) == 0

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_buffer_sub_tmem_rejects_partial_column_offset():
    buf_layout = tvm.tirx.layout.TileLayout(T.S[(64, 16) : (1 @ TLane, 1 @ TCol)])

    def build():
        # fmt: off
        @T.prim_func
        def func() -> None:
            T.device_entry()
            A = T.decl_buffer(
                (64, 16), "bfloat16", scope="tmem", allocated_addr=0, layout=buf_layout,
            )
            _ = A.sub[:, 1:3]
            # fmt: on

        return func

    with pytest.raises(tvm.error.DiagnosticError, match="aligned to a physical 32-bit column"):
        build()


def test_vector_annotation_shorthand_aliases():
    """Test shorthand aliases: T.f32, T.i32, T.f16, etc."""

    # fmt: off
    @T.prim_func
    def func():
        T.device_entry()
        a: T.f32[4]
        b: T.i32[2]
        c: T.f16[8]
        T.evaluate(a[0] + T.float32(b[0]) + T.float32(c[0]))  # noqa: F821
        # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_scalar_annotation_shorthand():
    """Test x: T.f32 (scalar) shorthand produces same IR as x: T.float32."""

    # fmt: off
    @T.prim_func
    def func():
        T.device_entry()
        x: T.f32 = 0
        y: T.i32
        x = x + T.float32(1.0)
        y = T.int32(2)
        T.evaluate(x + T.float32(y))
        # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_vector_annotation_with_python_variable_size():
    """Test x: T.f16[vec_size] where vec_size is a Python variable."""
    vec_size = 16

    # fmt: off
    @T.prim_func
    def func():
        T.device_entry()
        v: T.f16[vec_size]
        T.evaluate(T.float32(v[0]))  # noqa: F821
        # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_roundtrip_tmem_decl_buffer():
    """DeclBuffer with tmem scope: data kwarg must be suppressed, allocated_addr
    must print as Expr (not Array), and scalar buffer index must not get
    a .source suffix."""

    # fmt: off
    @T.prim_func
    def func():
        with T.launch_thread("blockIdx.x", 1):
            T.launch_thread("threadIdx.x", 128)
            addr = T.alloc_shared((1,), "uint32", layout=None)
            addr_alias = T.Buffer((1,), "uint32", data=addr.data, scope="shared")
            buf = T.decl_buffer((64,), scope="tmem", layout=None, allocated_addr=addr_alias[0])
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))
    decls = []
    tvm.tirx.stmt_functor.post_order_visit(
        func.body,
        lambda node: decls.append(node) if isinstance(node, tvm.tirx.DeclBuffer) else None,
    )
    assert len(decls) == 1
    tmem_decl = next(decl for decl in decls if decl.buffer.scope() == "tmem")
    assert tmem_decl.data.op.name == "tirx.reinterpret"


def test_roundtrip_cuda_func_call_source_code():
    """cuda_func_call with multiline source_code must print as keyword arg with
    inline string literal, not as a metadata reference."""

    # fmt: off
    @T.prim_func
    def func():
        T.device_entry()
        desc = T.alloc_local((1,), "uint64")
        T.cuda.func_call("my_func", T.address_of(desc[0]), source_code="\n__device__ void my_func(uint64_t* p) {\n    *p = 42;\n}\n")  # noqa: E501
        # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_roundtrip_cp_async_bulk_tensor_g2s_cluster():
    """The TMA load composite [tensorMap, coords] operand must round-trip."""

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def func(A_ptr: T.handle):
        _ = T.match_buffer(A_ptr, (16, 16), "float32")
        A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        with T.launch_thread("blockIdx.x", 1):
            T.launch_thread("threadIdx.x", 128)
            A_smem = T.alloc_buffer((16, 16), "float32", scope="shared")
            T.ptx["cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"](
                A_smem.data, T.address_of(A_map), 0, 0, T.uint32(0)
            )
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_roundtrip_cp_async_bulk_tensor_s2g():
    """The TMA store composite [tensorMap, coords] operand must round-trip."""

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def func(A_ptr: T.handle):
        _ = T.match_buffer(A_ptr, (16, 16), "float32")
        A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        with T.launch_thread("blockIdx.x", 1):
            T.launch_thread("threadIdx.x", 128)
            A_smem = T.alloc_buffer((16, 16), "float32", scope="shared")
            T.ptx["cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"](
                T.address_of(A_map), 0, 0, A_smem.data
            )
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_roundtrip_cp_async_bulk_tensor_prefetch():
    """The tensor prefetch composite [tensorMap, coords] operand must round-trip."""

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def func(A_ptr: T.handle):
        _ = T.match_buffer(A_ptr, (16, 16), "float32")
        A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        with T.launch_thread("blockIdx.x", 1):
            T.launch_thread("threadIdx.x", 128)
            T.ptx["cp.async.bulk.prefetch.tensor.2d.L2.global.tile"](
                T.address_of(A_map), 0, 0
            )
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_roundtrip_cp_async_bulk_tensor_s2g_reduce():
    """The tensor reduction composite [tensorMap, coords] operand must round-trip."""

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def func(A_ptr: T.handle):
        _ = T.match_buffer(A_ptr, (16, 16), "float32")
        A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        with T.launch_thread("blockIdx.x", 1):
            T.launch_thread("threadIdx.x", 128)
            A_smem = T.alloc_buffer((16, 16), "float32", scope="shared")
            T.ptx["cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"](
                T.address_of(A_map), 0, 0, A_smem.data
            )
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def _assert_roundtrip(func):
    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_loop_var_dtype_uint32():
    # fmt: off
    @T.prim_func
    def func(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (128,), "float32")
        for i in T.serial(128, dtype="uint32"):
            A[i] = T.float32(1)
    # fmt: on

    loop = func.body
    assert loop.loop_var.ty == PrimType("uint32")
    assert loop.min.ty == PrimType("uint32")
    assert loop.extent.ty == PrimType("uint32")
    _assert_roundtrip(func)


def test_loop_var_dtype_uint32_with_step():
    # fmt: off
    @T.prim_func
    def func(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (128,), "float32")
        for i in T.serial(4, 128, step=2, dtype="uint32"):
            A[i] = T.float32(1)
    # fmt: on

    loop = func.body
    assert loop.loop_var.ty == PrimType("uint32")
    assert loop.min.ty == PrimType("uint32")
    assert loop.extent.ty == PrimType("uint32")
    assert loop.step.ty == PrimType("uint32")
    _assert_roundtrip(func)


@pytest.mark.parametrize("for_kind", ["serial", "parallel", "vectorized", "unroll"])
def test_loop_var_dtype_uint32_all_for_kinds(for_kind):
    # fmt: off
    @T.prim_func
    def func(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (4,), "float32")
        for i in getattr(T, for_kind)(4, dtype="uint32"):
            A[i] = T.float32(1)
    # fmt: on

    assert func.body.loop_var.ty == PrimType("uint32")
    _assert_roundtrip(func)


def test_grid_loop_var_dtype_uint32():
    # fmt: off
    @T.prim_func
    def func(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (8, 16), "float32")
        for i, j in T.grid(8, 16, dtype="uint32"):
            A[i, j] = T.float32(1)
    # fmt: on

    outer = func.body
    assert outer.loop_var.ty == PrimType("uint32")
    assert outer.body.loop_var.ty == PrimType("uint32")
    _assert_roundtrip(func)


def test_loop_var_dtype_defaults_to_int32():
    # fmt: off
    @T.prim_func
    def func(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (128,), "float32")
        for i in range(128):
            A[i] = T.float32(1)
    # fmt: on

    assert func.body.loop_var.ty == PrimType("int32")
    _assert_roundtrip(func)


def test_loop_var_dtype_inferred_from_unsigned_extent():
    """A uint32 extent makes the loop var uint32 without an explicit dtype."""

    # fmt: off
    @T.prim_func
    def func(A_ptr: T.handle, n: T.uint32):
        A = T.match_buffer(A_ptr, (128,), "float32")
        for i in range(n):
            A[i] = T.float32(1)
    # fmt: on

    assert func.body.loop_var.ty == PrimType("uint32")
    _assert_roundtrip(func)


def test_loop_var_dtype_casts_mismatched_bound():
    """A non-literal bound of another dtype is cast to the requested loop dtype."""

    # fmt: off
    @T.prim_func
    def func(A_ptr: T.handle, n: T.int32):
        A = T.match_buffer(A_ptr, (128,), "float32")
        for i in T.serial(n, dtype="uint32"):
            A[i] = T.float32(1)
    # fmt: on

    loop = func.body
    assert loop.loop_var.ty == PrimType("uint32")
    assert loop.extent.ty == PrimType("uint32")
    _assert_roundtrip(func)


@pytest.mark.parametrize("dtype", ["int64", "uint64", "int16", "float32"])
def test_loop_var_dtype_rejects_unsupported(dtype):
    with pytest.raises(Exception, match='must be "int32" or "uint32"'):
        T.serial(4, dtype=dtype)


def test_thread_binding_has_no_dtype_parameter():
    with pytest.raises(TypeError):
        T.thread_binding(0, 128, "threadIdx.x", dtype="uint32")


def test_hand_built_for_promotes_int_literal_bounds_to_uint32():
    """The For constructor retypes literal bounds to the loop var's dtype."""
    loop_var = tvm.tirx.Var("i", "uint32")
    loop = tvm.tirx.For(loop_var, 0, 128, tvm.tirx.ForKind.SERIAL, tvm.tirx.Evaluate(0))
    assert loop.min.ty == PrimType("uint32")
    assert loop.extent.ty == PrimType("uint32")


def test_hand_built_for_rejects_negative_literal_for_uint32():
    loop_var = tvm.tirx.Var("i", "uint32")
    with pytest.raises(Exception, match="not representable"):
        tvm.tirx.For(loop_var, -1, 128, tvm.tirx.ForKind.SERIAL, tvm.tirx.Evaluate(0))


def test_scope_id_dtype_uint32():
    # fmt: off
    @T.prim_func
    def func(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (128,), "float32")
        T.device_entry()
        bx = T.cta_id([1])
        tx = T.thread_id([128], dtype="uint32")
        A[tx] = T.float32(bx)
    # fmt: on

    scope_defs = []
    tvm.tirx.stmt_functor.post_order_visit(
        func.body,
        lambda s: (
            scope_defs.append(getattr(s, "def")) if isinstance(s, tvm.tirx.ScopeIdDefStmt) else None
        ),
    )
    dtypes = {str(d.def_ids[0].ty) for d in scope_defs}
    assert dtypes == {"int32", "uint32"}
    # The extents stay int32 regardless of the def var dtype.
    for d in scope_defs:
        assert d.extents[0].ty == PrimType("int32")

    code = func.script()
    assert 'T.thread_id([128], dtype="uint32")' in code
    assert "T.cta_id([1])" in code
    _assert_roundtrip(func)


def test_scope_id_dtype_uint32_lane_and_warp():
    # fmt: off
    @T.prim_func
    def func(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (32,), "float32")
        T.device_entry()
        _ = T.cta_id([1])
        warp = T.warp_id([4], dtype="uint32")
        lane = T.lane_id([32], dtype="uint32")
        A[lane] = T.float32(warp)
    # fmt: on

    code = func.script()
    assert 'T.warp_id([4], dtype="uint32")' in code
    assert 'T.lane_id([32], dtype="uint32")' in code
    _assert_roundtrip(func)


def test_scope_id_dtype_uint32_with_preferred():
    # fmt: off
    @T.prim_func
    def func(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (4,), "float32")
        T.device_entry()
        _ = T.cluster_id([2])
        cx, cy = T.cta_id_in_cluster([2, 2], preferred=[2, 2], dtype="uint32")
        tx = T.thread_id([32])
        if tx == 0:
            A[cx + cy] = T.float32(1)
    # fmt: on

    code = func.script()
    assert 'dtype="uint32"' in code
    _assert_roundtrip(func)


def test_scope_id_dtype_uint32_deferred_extent():
    """The deferred (extent=None) form carries the dtype too."""

    # fmt: off
    @T.prim_func
    def func(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (32,), "float32")
        T.device_entry()
        _ = T.cta_id([1])
        lane = T.lane_id(dtype="uint32")
        warp = T.warp_id([4])
        A[lane] = T.float32(warp)
    # fmt: on

    scope_defs = []
    tvm.tirx.stmt_functor.post_order_visit(
        func.body,
        lambda s: (
            scope_defs.append(getattr(s, "def")) if isinstance(s, tvm.tirx.ScopeIdDefStmt) else None
        ),
    )
    deferred = [d for d in scope_defs if d.extents is None]
    assert len(deferred) == 1
    assert deferred[0].def_ids[0].ty == PrimType("uint32")
    _assert_roundtrip(func)


@pytest.mark.parametrize("dtype", ["int64", "float32"])
def test_scope_id_dtype_rejects_unsupported(dtype):
    # fmt: off
    with pytest.raises(Exception, match='must be "int32" or "uint32"'):

        @T.prim_func
        def func(A_ptr: T.handle):
            A = T.match_buffer(A_ptr, (128,), "float32")
            T.device_entry()
            _ = T.cta_id([1])
            tx = T.thread_id([128], dtype=dtype)
            A[tx] = T.float32(1)
    # fmt: on


if __name__ == "__main__":
    tvm.testing.main()
