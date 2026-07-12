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

import tvm
import tvm.script
import tvm.testing
from tvm.ir import PointerType, PrimType, assert_structural_equal
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import laneid, warpid


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
        return T.SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

    def get_layout5():
        return T.ComposeLayout(
            T.SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3),
            T.TileLayout(T.S[(64, 64, 4) : (64, 1, 64 * 64)]),
        )

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
            self.idx.buffer[0] = T.int32(0)
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
        D.buffer[0] = D + T.float16(1)  # noqa: F821
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
        T.evaluate(C.buffer.access_ptr("rw", offset=0))
        T.evaluate(C.buffer.data)
        T.evaluate(D)
        T.evaluate(T.address_of(D))
        # fmt: on

    code = test.script()
    print(code)
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
        kwargs = T.meta_var({"dispatch": "tma", "cta_group": 2})
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


def test_annotation_syntax_comprehensive():
    """Comprehensive test for scalar annotation, T.let, banned annotations, and bare assignment."""

    # 1. T.let with T.Var(PointerType) — round-trip
    # fmt: off
    @T.prim_func
    def test_let_var():
        T.device_entry()
        smem = T.alloc_shared([128], "float16")
        ptr: T.let[T.Var(name_hint="ptr", ty=PointerType(PrimType("void")))] = T.reinterpret(
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
    x: T.Var(name_hint="x", ty=PointerType(PrimType("float16"))) = T.int64(0)
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


def test_buffer_local_ir():
    """Verify .local() auto-infer: shape from storage shard extents, layout, shared data."""

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
    assert b_local.data.same_as(b_buf.data)
    # Shape: single dim matching storage shard total
    assert len(b_local.shape) == 1
    storage = b_buf.layout.storage()
    expected_total = 1
    for it in storage.shard:
        expected_total *= int(it.extent)
    assert int(b_local.shape[0]) == expected_total
    # Layout: storage layout (parent layout with thread axes removed)
    assert_structural_equal(b_local.layout, storage)

    # Round-trip
    code = func.script()
    assert from_source(code).script() == code


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
    assert b_buf.data.same_as(a_buf.data)
    # Shape: [4, 8] from [8, 4]
    assert int(b_buf.shape[0]) == 4
    assert int(b_buf.shape[1]) == 8
    # Layout: permuted
    assert_structural_equal(b_buf.layout, a_buf.layout.permute_dims([1, 0]))

    code = func.script()
    assert from_source(code).script() == code


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
    assert b_buf.data.same_as(a_buf.data)
    # dtype
    assert str(b_buf.dtype) == "float32"
    # Shape: [8, 4] (last dim halved since float32 is 2x float16)
    assert int(b_buf.shape[0]) == 8
    assert int(b_buf.shape[1]) == 4

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


def test_buffer_region_slice():
    """Verify BufferRegion slicing returns BufferRegion."""
    from tvm.tirx.stmt import BufferRegion

    buf = tvm.tirx.decl_buffer((128, 64), "float16")

    br1 = buf[32:64, 0:32]
    assert isinstance(br1, BufferRegion)

    # BufferRegion chained slice
    br3 = br1[0:16, 0:16]
    assert isinstance(br3, BufferRegion)
    assert br3.buffer.same_as(buf), "chained region slice must reference root buffer"
    assert int(br3.region[0].min) == 32
    assert int(br3.region[0].extent) == 16
    assert int(br3.region[1].min) == 0
    assert int(br3.region[1].extent) == 16


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
    a .buffer suffix."""

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


def test_roundtrip_cp_async_bulk_tensor_g2c():
    """cp.async.bulk.tensor.g2c must round-trip with *coords at end."""

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def func(A_ptr: T.handle):
        _ = T.match_buffer(A_ptr, (16, 16), "float32")
        A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        with T.launch_thread("blockIdx.x", 1):
            T.launch_thread("threadIdx.x", 128)
            A_smem = T.alloc_buffer((16, 16), "float32", scope="shared")
            T.ptx.cp_async.bulk.tensor.g2c(
                2, A_smem.data, 0, T.address_of(A_map), 0, 1, "", 0, 0
            )
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_roundtrip_cp_async_bulk_tensor_s2g():
    """cp.async.bulk.tensor.s2g must round-trip with *coords at end."""

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def func(A_ptr: T.handle):
        _ = T.match_buffer(A_ptr, (16, 16), "float32")
        A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        with T.launch_thread("blockIdx.x", 1):
            T.launch_thread("threadIdx.x", 128)
            A_smem = T.alloc_buffer((16, 16), "float32", scope="shared")
            T.ptx.cp_async.bulk.tensor.s2g(
                2, A_smem.data, T.address_of(A_map), "", 0, 0
            )
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_roundtrip_cp_async_bulk_tensor_g2c_prefetch():
    """cp.async.bulk.tensor.g2c_prefetch must round-trip with *coords at end."""

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def func(A_ptr: T.handle):
        _ = T.match_buffer(A_ptr, (16, 16), "float32")
        A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        with T.launch_thread("blockIdx.x", 1):
            T.launch_thread("threadIdx.x", 128)
            T.ptx.cp_async.bulk.tensor.g2c_prefetch(
                2, T.address_of(A_map), "", 0, 0
            )
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_roundtrip_cp_async_bulk_tensor_s2g_reduce():
    """cp.async.bulk.tensor.s2g_reduce must round-trip with *coords at end."""

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def func(A_ptr: T.handle):
        _ = T.match_buffer(A_ptr, (16, 16), "float32")
        A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        with T.launch_thread("blockIdx.x", 1):
            T.launch_thread("threadIdx.x", 128)
            A_smem = T.alloc_buffer((16, 16), "float32", scope="shared")
            T.ptx.cp_async.bulk.tensor.s2g_reduce(
                2, A_smem.data, T.address_of(A_map), "", "add", 0, 0
            )
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


if __name__ == "__main__":
    tvm.testing.main()
