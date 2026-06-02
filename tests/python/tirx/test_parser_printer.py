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
from tvm.script import tirx as Tx
from tvm.tirx.layout import laneid, warpid


def from_source(code):
    return tvm.script.from_source(code)


def _make_minimal_tirx_prim_func():
    source = (
        "# from tvm.script import tirx as Tx\n\n"
        "@Tx.prim_func()\n"
        "def f(a: Tx.handle):\n"
        '    A = Tx.match_buffer(a, (1,), "float32")\n'
        "    with Tx.thread():\n"
        "        with Tx.cta():\n"
        "            with Tx.thread():\n"
        "                A[0] = Tx.float32(1)"
    )
    return from_source(source)


def from_source_tir(code):
    return tvm.script.from_source(code, s_tir=True)


def test_roundtrip_scopeid1():
    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (64,), "float32", scope="global")

        Tx.device_entry()
        bx, by, bz = Tx.cta_id([1, 1, 1])
        warp_id = Tx.warp_id([1])
        lane_id = Tx.lane_id([32])
        with Tx.cta():
            with Tx.warp():
                with Tx.thread():
                    A_local = Tx.alloc_buffer([1], dtype="float16", scope="local")
                    for i in Tx.serial(2):
                        A_local[0] = A[lane_id * 2 + i]
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_scopeid2():
    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle) -> None:
        _ = Tx.match_buffer(A_ptr, (64,), "float32", scope="global")

        Tx.device_entry()
        bx, by, bz = Tx.cta_id([8, 10, 12])
        cbx, cby, cbz = Tx.cta_id_in_cluster([2, 2, 1])
        cta_id_in_pair = Tx.cta_id_in_pair()
        clx, cly, clz = Tx.cluster_id([4, 5, 12])
        with Tx.cta():
            with Tx.warp():
                with Tx.thread():
                    Tx.evaluate(bx + by + bz)
                    Tx.evaluate(cbx + cby + cbz)
                    Tx.evaluate(cta_id_in_pair)
                    Tx.evaluate(clx + cly + clz)
        # fmt: on

    code = test.script()
    assert "cta_id_in_pair = Tx.cta_id_in_pair()" in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_scopeid_deferred():
    """Deferred ScopeIdDef (extent=None) survives print→parse round-trip
    as a no-arg ``Tx.cta_id()``/``Tx.thread_id()`` etc. call."""

    # fmt: off
    @Tx.prim_func(private=True)
    def test(A_ptr: Tx.handle) -> None:
        _ = Tx.match_buffer(A_ptr, (64,), "float32", scope="global")
        Tx.device_entry()
        bx = Tx.cta_id()                       # deferred kernel→cta
        cbx = Tx.cta_id_in_cluster([2])
        clx = Tx.cluster_id([4])
        tx = Tx.thread_id()                    # deferred cta→thread
        Tx.warp_id([4])
        Tx.lane_id([32])
        with Tx.thread():
            Tx.evaluate(bx + cbx + clx + tx)
        # fmt: on

    code = test.script()
    assert "bx = Tx.cta_id()" in code
    assert "tx = Tx.thread_id()" in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_exec_scope_filter_guard_roundtrip_with_scope_arg_sugar():
    @Tx.prim_func(private=True)
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (1,), "float32", scope="global")

        Tx.device_entry()
        Tx.cta_id([1])
        tx = Tx.thread_id([128])
        with Tx.cta():
            with Tx.thread((0 <= tx) & (tx < 1)):
                A[0] = Tx.float32(1)

    code = test.script()
    assert "with Tx.thread(Tx.bitwise_and(0 <= tx, tx < 1)):" in code
    assert "if Tx.filter(tx, 0, 1):" not in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_layout():
    def get_layout1():
        return Tx.TileLayout(Tx.S[(8, 8, 8, 4, 2) : (6, 4 @ laneid, 2, 1 @ laneid, 1)])

    def get_layout2():
        return Tx.TileLayout(Tx.S[(8, 8, 8, 4, 2) : (64, 4 @ laneid, 8, 2, 1)])

    def get_layout3():
        return Tx.TileLayout(Tx.S[(8, 16, 8, 16) : (1024, 16, 128, 1)])

    def get_layout4():
        return Tx.SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3)

    def get_layout5():
        return Tx.ComposeLayout(
            Tx.SwizzleLayout(per_element=3, swizzle_len=3, atom_len=3),
            Tx.TileLayout(Tx.S[(64, 64, 4) : (64, 1, 64 * 64)]),
        )

    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle) -> None:
        _ = Tx.match_buffer(A_ptr, (64,), "float32", scope="global")

        Tx.device_entry()
        bx, by, bz = Tx.cta_id([1, 1, 1])
        warp_id = Tx.warp_id([1])
        lane_id = Tx.lane_id([32])
        C = Tx.alloc_buffer([128, 128], dtype="float16", scope="shared", layout=get_layout3())
        D = Tx.alloc_buffer([128, 32], dtype="float16", scope="shared", layout=get_layout4())

        with Tx.cta():
            A_warp = Tx.alloc_buffer([64, 64], dtype="float16", scope="shared", layout=get_layout1())  # noqa: E501
            B_warp = Tx.alloc_buffer([64, 64], dtype="float16", scope="shared", layout=get_layout2())  # noqa: E501

            E = Tx.alloc_buffer([64, 256], dtype="float16", scope="shared", layout=get_layout5())

            with Tx.thread():
                Tx.evaluate(A_warp[0, 0] + B_warp[0, 0] + C[0, 0] + D[0, 0] + E[0, 0])
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
        return Tx.TileLayout(Tx.S[8 : 4 @ laneid] + Tx.R[4 : 1 @ laneid])

    def get_shard_offset_single():
        return Tx.TileLayout(Tx.S[8 : 4 @ laneid] + 1 @ laneid)

    def get_shard_offset_multi():
        return Tx.TileLayout(Tx.S[8 : 4 @ laneid] + 1 @ laneid + 2 @ warpid + 64)

    def get_full():
        return Tx.TileLayout(
            Tx.S[(1,) : (1,)] + Tx.R[(8, 4) : (4 @ laneid, 1 @ laneid)] + 2 @ warpid
        )

    # fmt: off
    @Tx.prim_func
    def test() -> None:
        Tx.device_entry()
        with Tx.cta():
            A = Tx.alloc_buffer([8], dtype="float16", scope="shared", layout=get_shard_replica())
            B = Tx.alloc_buffer([8], dtype="float16", scope="shared", layout=get_shard_offset_single())  # noqa: E501
            C = Tx.alloc_buffer([8], dtype="float16", scope="shared", layout=get_shard_offset_multi())  # noqa: E501
            D = Tx.alloc_buffer([32], dtype="float16", scope="shared", layout=get_full())

            with Tx.thread():
                Tx.evaluate(A[0] + B[0] + C[0] + D[0])
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_print_kwargs_schedule_op_full_code():
    # fmt: off
    @Tx.prim_func
    def test():
        A = Tx.alloc_buffer((16,), "float32")
        Tx.memset(A[0:16], Tx.float32(1.25), dispatch="v10", bar=7, foo=42)
    # fmt: on

    expected = (
        "# from tvm.script import tirx as Tx\n"
        "# from tvm.tirx.layout import Axis\n\n"
        "@Tx.prim_func\n"
        "def test():\n"
        "    A = Tx.alloc_buffer((16,))\n"
        '    Tx.memset(A[0:16], Tx.float32(1.25), dispatch="v10", bar=7, foo=42)'
    )
    code = test.script()
    assert code == expected
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_default_script_prefix_tirx_irmodule_non_main():
    """IRModule with non-main TIRx PrimFunc should default to Tx prefix."""
    mod = tvm.IRModule({"foo": _make_minimal_tirx_prim_func()})
    code = mod.script()
    assert "# from tvm.script import tirx as Tx" in code
    assert "# from tvm.script import tir as T" not in code
    assert "@Tx.prim_func" in code
    assert "def foo(" in code
    assert "with Tx.thread():" in code
    parsed = from_source(code)
    assert parsed.script() == code
    assert_structural_equal(mod, parsed)


L_LANE = Tx.TileLayout(Tx.S[32 : 1 @ laneid])


def test_roundtrip_buffer_view_get1():
    # fmt: off
    @Tx.prim_func
    def test() -> None:
        Tx.device_entry()
        with Tx.cta():
            A = Tx.alloc_buffer([2], dtype="float16", scope="local")
            A_layout = Tx.TileLayout(Tx.S[(1, 2) : (2, 1)])
            A_warp_layout = A_layout.tile(L_LANE, (8, 4), (1, 2))
            A_warp = A.view(8, 8, layout=A_warp_layout)

            with Tx.thread():
                A_local = A_warp.local(2)
                A_local[0] = Tx.float16(0)

        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_buffer_view_get2():
    # fmt: off
    @Tx.prim_func
    def test(out_ptr: Tx.handle) -> None:
        out = Tx.match_buffer(out_ptr, (2), "float32", scope="global")

        Tx.device_entry()
        bx, by, bz = Tx.cta_id([32, 32, 1])
        tx, ty, tz = Tx.thread_id([16, 8, 1])
        warp_id = Tx.warp_id([4])
        lane_id = Tx.lane_id([32])
        with Tx.cta():
            A = Tx.alloc_buffer([2,], dtype="float16", scope="local")
            A_layout = Tx.TileLayout(Tx.S[(1, 2) : (2, 1)])
            B_layout = A_layout.tile(L_LANE, (8, 4), (1, 2))
            B = A.view(8, 8, layout=B_layout)
            D = B.local(2)

            with Tx.thread():
                out[0] = A[0] + B[0, 0] + D[0]
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_buffer_view_get3():
    # fmt: off
    @Tx.prim_func
    def test() -> None:
        Tx.device_entry()
        with Tx.cta():
            A = Tx.alloc_buffer([8, 8], dtype="float32", scope="local")
            A_f16 = A.view("float16")
            A_f64 = A.view("float64")

            with Tx.thread():
                A_f16[0, 0] = Tx.float16(0)
                A_f64[0, 0] = Tx.float64(0)

        # fmt: on
    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op1():
    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (64,), "float32", scope="global")

        Tx.device_entry()
        bx, by, bz = Tx.cta_id([1, 1, 1])
        warp_id = Tx.warp_id([1])
        lane_id = Tx.lane_id([32])
        with Tx.cta():
            A_smem = Tx.alloc_buffer([64], dtype="float32", scope="shared")

            Tx.copy(A_smem, A)
            for i in range(10):
                Tx.fill(A_smem, Tx.float32(0))
                Tx.gemm(A_smem, A_smem, A_smem, A_smem)
            Tx.copy(A, A_smem)
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op2():
    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle, B_ptr: Tx.handle, C_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128, 128), "float16", scope="global")
        B = Tx.match_buffer(B_ptr, (128, 64), "float16", scope="global")
        C = Tx.match_buffer(C_ptr, (128, 64), "float32", scope="global")

        Tx.device_entry()
        bx, by, bz = Tx.cta_id([1, 1, 1])
        warp_id = Tx.warp_id([4])
        lane_id = Tx.lane_id([32])
        with Tx.cta():
            A_smem = Tx.alloc_buffer([128, 32], dtype="float16", scope="shared")
            B_smem = Tx.alloc_buffer([32, 64], dtype="float16", scope="shared")

            C_local = Tx.alloc_buffer([128, 64], dtype="float32", scope="local")
            for k in range(4):
                Tx.copy(A_smem, A[:, k * 32 : k * 32 + 32])
                Tx.copy(B_smem, B[k * 32 : k * 32 + 32, 0:64])
                Tx.gemm(C_local, A_smem, B_smem, C_local)
            Tx.copy(C, C_local)
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op3():
    # fmt: off
    NUM_STAGES = 3
    K = 4096

    @Tx.prim_func
    def test(A_ptr: Tx.handle, B_ptr: Tx.handle, C_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128, K), "float16", scope="global")
        B = Tx.match_buffer(B_ptr, (K, 64), "float16", scope="global")
        C = Tx.match_buffer(C_ptr, (128, 64), "float32", scope="global")

        Tx.device_entry()
        bx, by, bz = Tx.cta_id([1, 1, 1])
        warp_id = Tx.warp_id([4])
        lane_id = Tx.lane_id([32])
        with Tx.cta():
            A_smem = Tx.alloc_buffer([NUM_STAGES, 128, 32], dtype="float16", scope="shared")
            B_smem = Tx.alloc_buffer([NUM_STAGES, 32, 64], dtype="float16", scope="shared")

            C_local = Tx.alloc_buffer([128, 64], dtype="float32", scope="local")
            for i in range(NUM_STAGES - 1):
                Tx.copy(A_smem[i, :, :], A[:, i * 32 : i * 32 + 32])
                Tx.copy(B_smem[i, :, :], B[i * 32 : i * 32 + 32, :])

            for k in range(K // 32):
                copy_k = Tx.meta_var(k + NUM_STAGES - 1)
                gemm_stage = Tx.meta_var(k % NUM_STAGES)
                copy_stage = Tx.meta_var(copy_k % NUM_STAGES)
                Tx.copy(A_smem[copy_stage, :, :], A[:, copy_k * 32 : copy_k * 32 + 32])
                Tx.copy(B_smem[copy_stage, :, :], B[copy_k * 32 : copy_k * 32 + 32, :])
                Tx.gemm(C_local, A_smem[gemm_stage, :, :], B_smem[gemm_stage, :, :], C_local)

            Tx.copy(C, C_local)
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_tensormap():
    # fmt: off
    @Tx.prim_func
    def func1(A_ptr: Tx.handle):
        Tx.func_attr({"global_symbol": "func"})
        _ = Tx.match_buffer(A_ptr, [128], "float32")

        A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        Tx.call_packed("runtime.tensormap_init", Tx.address_of(A_map), A_ptr)
    # fmt: on
    code = func1.script()
    assert from_source(code).script() == code
    assert_structural_equal(func1, from_source(code))


def test_roundtrip_tensormap_kernel_param():
    # fmt: off
    @Tx.prim_func
    def func1(A_map: Tx.TensorMap()):
        Tx.func_attr({"global_symbol": "func"})
        Tx.evaluate(Tx.address_of(A_map))
    # fmt: on
    code = func1.script()
    assert "Tx.TensorMap()" in code
    assert from_source(code).script() == code
    assert_structural_equal(func1, from_source(code))


def test_roundtrip_break_for():
    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (10,), "int32")

        Tx.device_entry()
        with Tx.cta():
            for i in Tx.serial(10):
                if i > 5:
                    break
                A[i] = i
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_break_while():
    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (10,), "int32")

        Tx.device_entry()
        with Tx.cta():
            i = Tx.alloc_buffer((1,), "int32", scope="local")
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
    @Tx.prim_func
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (9,), "int32")

        Tx.device_entry()
        with Tx.cta():
            idx = Tx.alloc_buffer((1,), "int32", scope="local")
            idx[0] = 0
            for i in Tx.serial(3):
                for j in Tx.serial(3):
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
    @Tx.prim_func
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (10,), "int32")

        Tx.device_entry()
        with Tx.cta():
            for i in Tx.serial(10):
                if (i % 2) == 0:
                    continue
                A[i] = i
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_continue_while():
    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (10,), "int32")

        Tx.device_entry()
        with Tx.cta():
            i = Tx.alloc_buffer((1,), "int32", scope="local")
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
    @Tx.prim_func
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (9,), "int32")

        Tx.device_entry()
        with Tx.cta():
            idx = Tx.alloc_buffer((1,), dtype="int32", scope="local")
            idx[0] = 0
            for i in Tx.serial(3):
                for j in Tx.serial(3):
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
    @Tx.prim_func
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (10,), "int32")

        Tx.device_entry()
        with Tx.cta():
            for i in Tx.serial(10):
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
    @Tx.prim_func
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (5,), "int32")

        Tx.device_entry()
        with Tx.cta():
            for i in Tx.serial(5):
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
    @Tx.prim_func
    def test():
        Tx.device_entry()
        A = Tx.alloc_buffer([10], "float32", scope="trn.sbuf", allocated_addr=1024)
        for i in Tx.serial(2):
            Tx.memset(A[i*5:i*5+5], Tx.float32(0.0))

        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_implicit_buffer_region():
    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, (10, 10, 10), "float32", layout=Tx.TileLayout(Tx.S[10, 10, 10]))
        Tx.device_entry()
        Tx.memset(A[0], Tx.float32(0.0))

        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_alloc_under_any_scope():
    # fmt: off
    @Tx.prim_func
    def test():
        Tx.device_entry()
        for i in Tx.serial(10):
            A = Tx.alloc_buffer([100], "float32", scope="trn.sbuf", allocated_addr=1024)
            Tx.memset(A[i*10:i*10+10], Tx.float32(0.0))

        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_compose_op():
    # fmt: off
    @Tx.prim_func
    def test():
        Tx.device_entry()
        A = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
        B = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
        C = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
        with Tx.compose_op():
            Tx.add(B, A, Tx.float32(1))
            Tx.add(C, B, Tx.float32(1))
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op_call_workspace():
    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, [10], "float32", scope="global")
        B = Tx.match_buffer(B_ptr, [10], "float32", scope="global")
        Tx.device_entry()
        smem = Tx.alloc_buffer([10], "float32", scope="shared")
        Tx.add(B, A, Tx.float32(1), workspace={"smem": smem})
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_compose_op_call_workspace():
    # fmt: off
    @Tx.prim_func
    def test():
        Tx.device_entry()
        A = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
        B = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
        C = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
        psum = Tx.alloc_buffer([10], "float32", scope="trn.psum")
        intermediate = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
        with Tx.compose_op(workspace={"intermediate": intermediate}):
            Tx.add(B, A, Tx.float32(1))
            Tx.add(C, B, Tx.float32(1), workspace={"psum": psum})
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op_call_config():
    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle, B_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, [10], "float32", scope="global")
        B = Tx.match_buffer(B_ptr, [10], "float32", scope="global")
        Tx.device_entry()
        Tx.add(B, A, Tx.float32(1), schedule="A")
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_compose_op_call_config():
    # fmt: off
    @Tx.prim_func
    def test():
        Tx.device_entry()
        A = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
        B = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
        C = Tx.alloc_buffer([10], "float32", scope="trn.sbuf")
        psum = Tx.alloc_buffer([10], "float32", scope="trn.psum")
        with Tx.compose_op( schedule="A"):
            Tx.add(B, A, Tx.float32(1))
            Tx.add(C, B, Tx.float32(1), workspace={"psum": psum})
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_predicate():
    # fmt: off
    @Tx.prim_func
    def test():
        Tx.device_entry()
        A = Tx.alloc_buffer([10, 10], "float32")
        B = Tx.alloc_buffer([10, 10], "float32")
        Tx.select(B, A, 1.0, lambda i, j: i < j)
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_grid():
    # fmt: off
    @Tx.prim_func
    def test():
        Tx.device_entry()
        with Tx.thread():
            for lvs in Tx.grid(10, (2, 12)):
                Tx.evaluate(lvs[0] + lvs[1])
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_alloc_apis():
    # fmt: off
    @Tx.meta_class
    class Test:
        def __init__(self, Ta, inner_pool):
            self.Ta = Ta
            self.inner_pool = inner_pool
            self.Tb = Tx.shared_scalar("float16")
            self.idx = Tx.local_scalar("int32")
            self.inner_pool2 = Tx.decl_scalar("float16", self.inner_pool.data, "shared.dyn", 5)

        @Tx.inline
        def init(self):
            self.Ta = self.Ta + Tx.float16(1)
            self.Tb = self.Tb + Tx.float16(2)
            self.idx.buffer[0] = Tx.int32(0)
            self.idx = self.idx + Tx.int32(1)
            self.inner_pool2 = self.inner_pool2 + Tx.float16(1)
            Tx.evaluate(Tx.address_of(self.Ta))
            Tx.evaluate(Tx.address_of(self.Tb))
            Tx.evaluate(Tx.address_of(self.idx))
            Tx.evaluate(Tx.address_of(self.inner_pool))
            Tx.evaluate(Tx.address_of(self.inner_pool2))

    @Tx.prim_func
    def test():
        Tx.device_entry()
                # normal buffer
        A = Tx.alloc_shared([10], "float16")
        B = Tx.alloc_local([10], "float16")
                # scalar buffer (alloc)
        C = Tx.shared_scalar("float16")
        D: Tx.float16
        pool = Tx.alloc_buffer([10], "uint8", scope="shared.dyn")
                # scalar buffer (decl)
        E = Tx.decl_scalar("float16", pool.data, "shared.dyn", 0)
                # normal 1-dim buffer with shape (1,)
        F = Tx.alloc_local((1,), "float16")
        with Tx.thread():
            Ta: Tx.float16
            inner_pool = Tx.decl_buffer(shape=[10], data=pool.data, dtype="uint8", scope="shared.dyn")  # noqa: E501
            test = Test(Ta, inner_pool)  # noqa: F821
            test.init()
            A[0] = C
            A[0] = C + D  # noqa: F821
            A[1] = B[0] * C
            D.buffer[0] = D + Tx.float16(1)  # noqa: F821
            D = D + Tx.float16(1)  # noqa: F821
            C = D
            Tx.evaluate(E)
            E = E + Tx.float16(1)
                    # normal 1-dim buffer with shape (1,) can be assigned directly,
                    # but not loaded directly
            F = F[0] + Tx.float16(1)
            C += D
            D += E + C + D
            Tx.evaluate(Tx.address_of(C))
            Tx.evaluate(C.buffer.access_ptr("rw", offset=0))
            Tx.evaluate(C.buffer.data)
            Tx.evaluate(D)
            Tx.evaluate(Tx.address_of(D))
        # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code


def test_alloc_apis_reject_name_argument():
    with pytest.raises(TypeError):
        Tx.alloc_buffer((1,), "int32", name="buf")

    with pytest.raises(TypeError):
        Tx.local_scalar("int32", name="idx")


def test_meta_class_constructor_rejects_unowned_resource():
    @Tx.meta_class
    class Bad:
        def __init__(self):
            tmp = Tx.alloc_buffer((1,), "int32", scope="local")

    with pytest.raises(tvm.error.DiagnosticError):

        @Tx.prim_func
        def test():
            Tx.device_entry()
            bad = Bad()


def test_meta_class_multiple_instances_auto_name_owned_resources():
    @Tx.meta_class
    class Holder:
        def __init__(self, external):
            self.external = external
            self.buf = Tx.alloc_buffer((2,), "int32", scope="local")
            self.scalar = Tx.local_scalar("int32")

    @Tx.prim_func
    def test():
        Tx.device_entry()
        with Tx.thread():
            external = Tx.alloc_buffer((2,), "int32", scope="local")
            first = Holder(external)
            second = Holder(external)
            Tx.evaluate(
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
    assert 'first_buf = Tx.alloc_local((2,), "int32")' in code
    assert 'second_buf = Tx.alloc_local((2,), "int32")' in code
    assert "first_scalar: Tx.int32" in code
    assert "second_scalar: Tx.int32" in code
    assert from_source(code).script() == code


def test_macro():
    # fmt: off
    @Tx.inline
    def mul(x, c):
        Tx.evaluate(x * c)

    @Tx.prim_func(private=True)
    def test():
        Tx.device_entry()
        for x in range(10):

            @Tx.inline
            def add(c):
                Tx.evaluate(x + c)

            @Tx.inline
            def two_add_and_mul(c):
                add(c)
                add(c + c)
                mul(x, c)

            two_add_and_mul(1)
            two_add_and_mul(2)


    @Tx.prim_func(private=True)
    def expected():
        Tx.device_entry()
        for x in range(10):
            Tx.evaluate(x + 1)
            Tx.evaluate(x + 2)
            Tx.evaluate(x)
            Tx.evaluate(x + 2)
            Tx.evaluate(x + 4)
            Tx.evaluate(x * 2)
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))
    assert_structural_equal(test, expected)


def test_macro_recursive():
    # fmt: off
    @Tx.prim_func(private=True)
    def test():
        Tx.device_entry()
        for x in Tx.serial(10):

            @Tx.inline
            def add(x, c):
                if c > 0:
                    add(x, c - 1)
                Tx.evaluate(x)

            add(x, 5)

    @Tx.prim_func(private=True)
    def expected():
        Tx.device_entry()
        for x in range(10):
            Tx.evaluate(x)
            Tx.evaluate(x)
            Tx.evaluate(x)
            Tx.evaluate(x)
            Tx.evaluate(x)
            Tx.evaluate(x)
        # fmt: on
    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))
    assert_structural_equal(expected, from_source(code))


def test_list_comprehension():
    # fmt: off
    @Tx.prim_func(private=True)
    def test():
        Tx.device_entry()
        with Tx.thread():
            acc = Tx.alloc_local([10], "bool")
            regs = Tx.meta_var([acc[_] for _ in range(10)])
            Tx.evaluate(regs[0])
            Tx.evaluate(tvm.tirx.all(*regs))
            Tx.evaluate(tvm.tirx.all(*[acc[_] for _ in range(10)]))
            Tx.evaluate(tvm.tirx.all(*([acc[_] for _ in range(2, 4)] + [acc[_] for _ in range(6, 8)])))  # noqa: E501
        # fmt: on
    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_range():
    # fmt: off
    @Tx.prim_func(private=True)
    def test():
        l = Tx.meta_var([i for i in range(10)])  # noqa: E741
        Tx.evaluate(l[3])

    @Tx.prim_func(private=True)
    def expected():
        Tx.evaluate(3)
    # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))
    tvm.ir.assert_structural_equal(test, expected)


def test_buffer():
    # fmt: off
    @Tx.prim_func(private=True)
    def test(
        A: Tx.Buffer((10, 11), "float32", layout=None),
        B: Tx.Buffer((10, 11), "float32", scope="global"),
        C: Tx.Buffer((10, 11), "float32", layout="default"),
        D: Tx.Buffer((10, 11), "float32", layout=Tx.TileLayout(Tx.S[(10, 11) : (1, 10)])),
        E_ptr: Tx.handle,
        F_ptr: Tx.handle,
        G_ptr: Tx.handle,
        H_ptr: Tx.handle,
    ):
        _E = Tx.match_buffer(E_ptr, [10, 11], "float16", layout=None)
        _F = Tx.match_buffer(F_ptr, [10, 11], "float16", scope="global")
        _G = Tx.match_buffer(G_ptr, [10, 11], "float16", layout="default")
        _H = Tx.match_buffer(H_ptr, [10, 11], "float16", layout=Tx.TileLayout(Tx.S[(10, 11) : (1, 10)]))  # noqa: E501

        _A0 = Tx.decl_buffer((10, 11), "float32", data=A.data, layout=None)
        _B0 = Tx.decl_buffer((10, 11), "float32", data=B.data, scope="global")
        _C0 = Tx.decl_buffer((10, 11), "float32", data=C.data, layout="default")
        _D0 = Tx.decl_buffer((10, 11), "float32", data=D.data, layout=Tx.TileLayout(Tx.S[(10, 11) : (1, 10)]))  # noqa: E501

        with Tx.thread():
            _A1 = Tx.alloc_buffer((10, 11), "float32", layout=None)
            _B1 = Tx.alloc_buffer((10, 11), "float32", scope="global")
            _C1 = Tx.alloc_buffer((10, 11), "float32", layout="default")
            _D1 = Tx.alloc_buffer((10, 11), "float32", layout=Tx.TileLayout(Tx.S[(10, 11) : (1, 10)]))  # noqa: E501

            pass
    # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_kwargs_op_call():
    # fmt: off
    @Tx.prim_func(private=True)
    def test(A: Tx.Buffer((10, 10), "float32"), B: Tx.Buffer((10, 10), "float32")):
        Tx.device_entry()
        kwargs = Tx.meta_var({"dispatch": "tma", "cta_group": 2})
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
    """Regression: the parser's scalar-assignment sugar (scalar = PrimExpr) must
    work in macro context via self.attr.

    The parser narrowed ``except Exception: pass`` around the scalar-detection
    path. This test verifies that PrimExpr assignment to a scalar attribute in
    a macro still goes through buffer_store correctly.

    The full integration regression for the TypeError fallthrough path
    (meta_var assigned to a scalar variable) is covered by
    test_hgemm::test_hgemm (tile_scheduler.m_idx pattern)."""

    # fmt: off
    class State:
        def __init__(self, counter):
            self.counter = counter

        @Tx.inline
        def add_one(self):
            # PrimExpr assigned to scalar via self.attr → buffer_store succeeds
            self.counter = self.counter + Tx.int32(1)

    @Tx.prim_func
    def test():
        Tx.device_entry()
        with Tx.thread():
            counter: Tx.int32
            state = Tx.meta_var(State(counter))  # noqa: F821
            state.add_one()
            Tx.evaluate(state.counter)
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
# from tvm.script import tirx as Tx

@Tx.prim_func
def func():
    Tx.device_entry()
    with Tx.thread():
        v: Tx.int32
        v = v + Tx.int32(1)
"""
    # The ValueError propagates through the parser framework which wraps it
    # into a DiagnosticError.  Before the fix the broad ``except Exception``
    # would silently swallow it and fall through to eval_assign.
    with patch("tvm.tirx.script.builder.buffer_store", side_effect=bomb):
        with pytest.raises(tvm.error.DiagnosticError):
            from_source(src)


def test_scalar_annotation_syntax():
    """Test the scalar annotation syntax: x: Tx.int32 = init, x: Tx.int32, and T.let."""

    # fmt: off
    @Tx.prim_func
    def test():
        Tx.device_entry()
        with Tx.thread():
                    # Scalar with init value
            x: Tx.int32 = 0
            y: Tx.float16 = Tx.float16(1.0)
                    # Scalar without init
            z: Tx.int32
                    # Use scalars
            x = x + Tx.int32(1)
            z = x + Tx.int32(2)
            y = y + Tx.float16(3.0)
            Tx.evaluate(x + z)
            Tx.evaluate(y)
        # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_scalar_allocbuffer_annotation_and_init_merge():
    # fmt: off
    @Tx.prim_func
    def test():
        Tx.device_entry()
        with Tx.thread():
            phase_mma = Tx.alloc_local((1,), "int32")
            phase_mma[0] = Tx.int32(0)
            phase_aux = Tx.alloc_local((1,), "int32")
            Tx.evaluate(phase_mma[0] + phase_aux[0])
        # fmt: on

    code = test.script()
    assert "phase_mma: Tx.int32 = 0" in code
    assert "phase_aux: Tx.int32" in code
    assert "phase_mma = Tx.alloc_local" not in code
    assert "phase_aux = Tx.alloc_local" not in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_scalar_allocbuffer_layout_none_keeps_alloc_local():
    # fmt: off
    @Tx.prim_func
    def test():
        Tx.device_entry()
        with Tx.thread():
            phase_mma = Tx.alloc_local((1,), "int32", layout=None)
            phase_mma[0] = Tx.int32(0)
            Tx.evaluate(phase_mma[0])
        # fmt: on

    code = test.script()
    assert 'phase_mma = Tx.alloc_local((1,), "int32", layout=None)' in code
    assert "phase_mma: Tx.int32" not in code
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
    assert "x: Tx.int32 = 0" in code
    assert "x = Tx.alloc_buffer" not in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_let_annotation_syntax():
    """Test explicit LetStmt syntax: T.let[T.int32] and T.let."""

    # fmt: off
    @Tx.prim_func
    def test():
        blockIdx_x = Tx.launch_thread("blockIdx.x", 4)
        threadIdx_x = Tx.launch_thread("threadIdx.x", 128)
        # Explicit LetStmt with type
        bx: Tx.let[Tx.int32] = blockIdx_x
        tx: Tx.let[Tx.int32] = threadIdx_x
        # Explicit LetStmt with auto-type
        combined: Tx.let = bx + tx
        Tx.device_entry()
        with Tx.thread():
            Tx.evaluate(bx + tx + combined)
        # fmt: on

    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_annotation_syntax_comprehensive():
    """Comprehensive test for scalar annotation, T.let, banned annotations, and bare assignment."""

    # 1. T.let with Tx.Var(PointerType) — round-trip
    # fmt: off
    @Tx.prim_func
    def test_let_var():
        Tx.device_entry()
        smem = Tx.alloc_shared([128], "float16")
        with Tx.thread():
            ptr: Tx.let[Tx.Var(name="ptr", dtype=PointerType(PrimType("uint64")))] = Tx.reinterpret(
                "handle", smem.access_ptr("rw")
            )
            Tx.evaluate(ptr)
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
    x: T.Var(name="x", dtype=PointerType(PrimType("float16"))) = T.int64(0)
"""
    with pytest.raises(tvm.error.DiagnosticError):
        from_source(src_ptr)

    # 4. Bare assignment to new variable creates scalar — round-trip
    # fmt: off
    @Tx.prim_func
    def test_bare_assign():
        Tx.device_entry()
        with Tx.thread():
            tid = Tx.launch_thread("threadIdx.x", 128)
            x = tid + Tx.int32(1)
            x = x + Tx.int32(2)
            Tx.evaluate(x)
        # fmt: on
    code = test_bare_assign.script()
    assert from_source(code).script() == code


def test_roundtrip_buffer_permute():
    # fmt: off
    @Tx.prim_func
    def test() -> None:
        Tx.device_entry()
        with Tx.cta():
            A = Tx.alloc_buffer([8, 4], dtype="float16", scope="local",
                                layout=Tx.TileLayout(Tx.S[(8, 4) : (4, 1)]))
            B = A.permute(1, 0)

            with Tx.thread():
                B[0, 0] = Tx.float16(0)
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_buffer_local_auto():
    # fmt: off
    @Tx.prim_func
    def test() -> None:
        Tx.device_entry()
        with Tx.cta():
            A = Tx.alloc_buffer([2], dtype="float16", scope="local")
            A_layout = Tx.TileLayout(Tx.S[(1, 2) : (2, 1)])
            B = A.view(8, 8, layout=A_layout.tile(L_LANE, (8, 4), (1, 2)))

            with Tx.thread():
                B_local = B.local()
                B_local[0] = Tx.float16(0)
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
    @Tx.prim_func
    def func() -> None:
        Tx.device_entry()
        with Tx.cta():
            A = Tx.alloc_buffer([2], dtype="float16", scope="local")
            A_layout = Tx.TileLayout(Tx.S[(1, 2) : (2, 1)])
            B = A.view(8, 8, layout=A_layout.tile(L_LANE, (8, 4), (1, 2)))

            with Tx.thread():
                B_local = B.local()
                B_local[0] = Tx.float16(0)
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
    @Tx.prim_func
    def func() -> None:
        Tx.device_entry()
        with Tx.cta():
            A = Tx.alloc_buffer([8, 4], dtype="float16", scope="local",
                                layout=Tx.TileLayout(Tx.S[(8, 4) : (4, 1)]))
            B = A.permute(1, 0)
            with Tx.thread():
                B[0, 0] = Tx.float16(0)
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
    @Tx.prim_func
    def func() -> None:
        Tx.device_entry()
        with Tx.cta():
            A = Tx.alloc_buffer([8, 8], dtype="float16", scope="local")
            B = A.view("float32")
            with Tx.thread():
                B[0, 0] = Tx.float32(0)
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
    """Tx.serial(N, unroll=False) should round-trip."""

    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128,), "float32", scope="global")
        Tx.device_entry()
        cta_id = Tx.cta_id([1])
        warp_id = Tx.warp_id([1])
        lane_id = Tx.lane_id([32])
        with Tx.cta():
            for _ in Tx.serial(10, unroll=False):
                Tx.fill(A[0:32], Tx.float32(0))
        # fmt: on

    code = test.script()
    assert "unroll=False" in code, f"printer should emit unroll=False, got:\n{code}"
    assert "annotations" not in code, "printer should NOT emit annotations dict"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_serial_unroll_true():
    """Tx.serial(N, unroll=True) should round-trip as a pragma-unroll request."""

    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128,), "float32", scope="global")
        Tx.device_entry()
        cta_id = Tx.cta_id([1])
        warp_id = Tx.warp_id([1])
        lane_id = Tx.lane_id([32])
        with Tx.cta():
            for _ in Tx.serial(10, unroll=True):
                Tx.fill(A[0:32], Tx.float32(0))
        # fmt: on

    code = test.script()
    assert "unroll=True" in code, f"printer should emit unroll=True, got:\n{code}"
    assert "annotations" not in code, "printer should NOT emit annotations dict"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_serial_unroll_false_with_other_annotations():
    """When other annotations exist alongside disable_unroll, fall back to full dict."""

    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128,), "float32", scope="global")
        Tx.device_entry()
        cta_id = Tx.cta_id([1])
        warp_id = Tx.warp_id([1])
        lane_id = Tx.lane_id([32])
        with Tx.cta():
            for _ in Tx.serial(10, annotations={"disable_unroll": True, "custom": 42}):
                Tx.fill(A[0:32], Tx.float32(0))
        # fmt: on

    code = test.script()
    assert "annotations=" in code, "printer should emit full annotations when multiple keys exist"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_unary_inplace():
    """Single-arg unary ops (in-place) should round-trip."""

    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128,), "float32", scope="global")
        Tx.device_entry()
        cta_id = Tx.cta_id([1])
        warp_id = Tx.warp_id([1])
        lane_id = Tx.lane_id([32])
        with Tx.cta():
            with Tx.warp():
                Tx.exp2(A[0:32])
                Tx.sqrt(A[32:64])
                Tx.reciprocal(A[64:96])
        # fmt: on

    code = test.script()
    # Each op should appear with a single arg (no duplicate src, no trailing Nones)
    assert "Tx.exp2(A[0:32])" in code, f"expected single-arg exp2, got:\n{code}"
    assert "Tx.sqrt(A[32:64])" in code, f"expected single-arg sqrt, got:\n{code}"
    assert "Tx.reciprocal(A[64:96])" in code, f"expected single-arg reciprocal, got:\n{code}"
    assert "None" not in code, f"trailing None args should be trimmed:\n{code}"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_unary_different_dst_src():
    """Unary ops with different dst and src should keep both args."""

    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128,), "float32", scope="global")
        B = Tx.match_buffer(B_ptr, (128,), "float32", scope="global")
        Tx.device_entry()
        cta_id = Tx.cta_id([1])
        warp_id = Tx.warp_id([1])
        lane_id = Tx.lane_id([32])
        with Tx.cta():
            with Tx.warp():
                Tx.exp2(A[0:32], B[0:32])
        # fmt: on

    code = test.script()
    assert "Tx.exp2(A[0:32], B[0:32])" in code, f"different dst/src should keep both:\n{code}"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_persistent_decorator():
    """@Tx.prim_func(persistent=True) should round-trip."""

    # fmt: off
    @Tx.prim_func(persistent=True)
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128,), "float32", scope="global")
        Tx.device_entry()
        cta_id = Tx.cta_id([1])
        warp_id = Tx.warp_id([1])
        lane_id = Tx.lane_id([32])
        with Tx.cta():
            Tx.fill(A[0:32], Tx.float32(0))
        # fmt: on

    code = test.script()
    assert "persistent=True" in code, f"persistent not in decorator:\n{code}"
    assert "tirx.persistent_kernel" not in code, "should NOT appear as func_attr"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_persistent_not_present():
    """Without persistent=True, the keyword should not appear."""

    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128,), "float32", scope="global")
        Tx.device_entry()
        cta_id = Tx.cta_id([1])
        warp_id = Tx.warp_id([1])
        lane_id = Tx.lane_id([32])
        with Tx.cta():
            Tx.fill(A[0:32], Tx.float32(0))
        # fmt: on

    code = test.script()
    assert "persistent" not in code, f"persistent should NOT appear:\n{code}"


def test_warp_role():
    """WarpRole should emit guarded warp scopes plus setmaxnreg."""
    from tvm.tirx.lang.warp_role import WarpRole

    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128,), "float32", scope="global")
        Tx.device_entry()
        cta_id = Tx.cta_id([1])
        wg_id = Tx.warpgroup_id([4])
        warp_id = Tx.warp_id_in_wg([4])
        lane_id = Tx.lane_id([32])
        with Tx.cta():
            with WarpRole(warp_id, 1, regs=48):
                Tx.fill(A[0:32], Tx.float32(0))
            with WarpRole(warp_id, 0, regs=232, increase=True):
                Tx.fill(A[32:64], Tx.float32(1))
        # fmt: on

    code = test.script()
    assert "warp_id == 1" in code, f"should have warp_id==1 guard:\n{code}"
    assert "warp_id == 0" in code, f"should have warp_id==0 guard:\n{code}"
    assert "setmaxnreg" in code, f"should have setmaxnreg:\n{code}"
    assert "with Tx.warp(warp_id == 1):" in code, f"should have guarded Tx.warp scope:\n{code}"
    assert "with Tx.warp(warp_id == 0):" in code, f"should have guarded Tx.warp scope:\n{code}"
    # The printed code is valid TIR — it should parse back
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_warpgroup_role():
    """WarpgroupRole should emit guarded warpgroup scope plus setmaxnreg."""
    from tvm.tirx.lang.warp_role import WarpgroupRole

    # fmt: off
    @Tx.prim_func
    def test(A_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, (128,), "float32", scope="global")
        Tx.device_entry()
        cta_id = Tx.cta_id([1])
        wg_id = Tx.warpgroup_id([4])
        warp_id_in_wg = Tx.warp_id_in_wg([4])
        lane_id = Tx.lane_id([32])
        with Tx.cta():
            with WarpgroupRole(wg_id, 2, regs=200, increase=True):
                Tx.fill(A[0:32], Tx.float32(0))
        # fmt: on

    code = test.script()
    assert "wg_id == 2" in code, f"should have wg_id==2 guard:\n{code}"
    assert "setmaxnreg" in code, f"should have setmaxnreg:\n{code}"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_vector_annotation_syntax_1d():
    """Test x: Tx.f32[N] produces the same IR as Tx.alloc_local([N], 'float32')."""

    # fmt: off
    @Tx.prim_func
    def func():
        Tx.device_entry()
        with Tx.thread():
            v: Tx.float32[8]
            Tx.evaluate(v[0])  # noqa: F821

    @Tx.prim_func
    def func():  # noqa: F811
        Tx.device_entry()
        with Tx.thread():
            v = Tx.alloc_local([8], "float32")
            Tx.evaluate(v[0])
        # fmt: on

        # func was redefined; compare first (annotation) with second (alloc_local).
        # Re-create the annotation version for comparison:

        # fmt: off
    @Tx.prim_func
    def annotation_func():
        Tx.device_entry()
        with Tx.thread():
            v: Tx.float32[8]
            Tx.evaluate(v[0])  # noqa: F821
        # fmt: on

        # Verify both produce valid IR that round-trips through printer/parser
    code = func.script()
    assert from_source(code).script() == code
    code2 = annotation_func.script()
    assert from_source(code2).script() == code2
    # The printed form should be identical (both become alloc_local in print)
    assert code.replace("annotation_func", "func") == code


def test_vector_annotation_syntax_multidim():
    """Test x: Tx.f32[M, N] produces the same IR as Tx.alloc_local([M, N], 'float32')."""

    # fmt: off
    @Tx.prim_func
    def func():
        Tx.device_entry()
        with Tx.thread():
            m: Tx.float32[4, 8]
            Tx.evaluate(m[0, 0])  # noqa: F821
        # fmt: on

    code = func.script()
    assert "alloc_local((4, 8)" in code or "float32[4, 8]" in code
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_vector_annotation_shorthand_aliases():
    """Test shorthand aliases: Tx.f32, Tx.i32, Tx.f16, etc."""

    # fmt: off
    @Tx.prim_func
    def func():
        Tx.device_entry()
        with Tx.thread():
            a: Tx.f32[4]
            b: Tx.i32[2]
            c: Tx.f16[8]
            Tx.evaluate(a[0] + Tx.float32(b[0]) + Tx.float32(c[0]))  # noqa: F821
        # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_scalar_annotation_shorthand():
    """Test x: Tx.f32 (scalar) shorthand produces same IR as x: Tx.float32."""

    # fmt: off
    @Tx.prim_func
    def func():
        Tx.device_entry()
        with Tx.thread():
            x: Tx.f32 = 0
            y: Tx.i32
            x = x + Tx.float32(1.0)
            y = Tx.int32(2)
            Tx.evaluate(x + Tx.float32(y))
        # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_vector_annotation_with_python_variable_size():
    """Test x: Tx.f16[vec_size] where vec_size is a Python variable."""
    vec_size = 16

    # fmt: off
    @Tx.prim_func
    def func():
        Tx.device_entry()
        with Tx.thread():
            v: Tx.f16[vec_size]
            Tx.evaluate(Tx.float32(v[0]))  # noqa: F821
        # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_roundtrip_tmem_decl_buffer():
    """DeclBuffer with tmem scope: data kwarg must be suppressed, allocated_addr
    must print as PrimExpr (not Array), and scalar buffer index must not get
    a .buffer suffix."""

    # fmt: off
    @Tx.prim_func
    def func():
        with Tx.launch_thread("blockIdx.x", 1):
            Tx.launch_thread("threadIdx.x", 128)
            addr = Tx.alloc_shared((1,), "uint32", layout=None)
            addr_alias = Tx.Buffer((1,), "uint32", data=addr.data, scope="shared")
            buf = Tx.decl_buffer((64,), scope="tmem", layout=None, allocated_addr=addr_alias[0])
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_roundtrip_cuda_func_call_source_code():
    """cuda_func_call with multiline source_code must print as keyword arg with
    inline string literal, not as a metadata reference."""

    # fmt: off
    @Tx.prim_func
    def func():
        Tx.device_entry()
        with Tx.cta():
            desc = Tx.alloc_local((1,), "uint64")
            Tx.cuda.func_call("my_func", Tx.address_of(desc[0]), source_code="\n__device__ void my_func(uint64_t* p) {\n    *p = 42;\n}\n")  # noqa: E501
        # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_roundtrip_cp_async_bulk_tensor_g2c():
    """cp.async.bulk.tensor.g2c must round-trip with *coords at end."""

    # fmt: off
    @Tx.prim_func(check_well_formed=False)
    def func(A_ptr: Tx.handle):
        _ = Tx.match_buffer(A_ptr, (16, 16), "float32")
        A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        with Tx.launch_thread("blockIdx.x", 1):
            Tx.launch_thread("threadIdx.x", 128)
            A_smem = Tx.alloc_buffer((16, 16), "float32", scope="shared")
            Tx.ptx.cp_async.bulk.tensor.g2c(
                2, A_smem.data, 0, Tx.address_of(A_map), 0, 1, "", 0, 0
            )
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_roundtrip_cp_async_bulk_tensor_s2g():
    """cp.async.bulk.tensor.s2g must round-trip with *coords at end."""

    # fmt: off
    @Tx.prim_func(check_well_formed=False)
    def func(A_ptr: Tx.handle):
        _ = Tx.match_buffer(A_ptr, (16, 16), "float32")
        A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        with Tx.launch_thread("blockIdx.x", 1):
            Tx.launch_thread("threadIdx.x", 128)
            A_smem = Tx.alloc_buffer((16, 16), "float32", scope="shared")
            Tx.ptx.cp_async.bulk.tensor.s2g(
                2, A_smem.data, Tx.address_of(A_map), "", 0, 0
            )
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_roundtrip_cp_async_bulk_tensor_g2c_prefetch():
    """cp.async.bulk.tensor.g2c_prefetch must round-trip with *coords at end."""

    # fmt: off
    @Tx.prim_func(check_well_formed=False)
    def func(A_ptr: Tx.handle):
        _ = Tx.match_buffer(A_ptr, (16, 16), "float32")
        A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        with Tx.launch_thread("blockIdx.x", 1):
            Tx.launch_thread("threadIdx.x", 128)
            Tx.ptx.cp_async.bulk.tensor.g2c_prefetch(
                2, Tx.address_of(A_map), "", 0, 0
            )
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_roundtrip_cp_async_bulk_tensor_s2g_reduce():
    """cp.async.bulk.tensor.s2g_reduce must round-trip with *coords at end."""

    # fmt: off
    @Tx.prim_func(check_well_formed=False)
    def func(A_ptr: Tx.handle):
        _ = Tx.match_buffer(A_ptr, (16, 16), "float32")
        A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        with Tx.launch_thread("blockIdx.x", 1):
            Tx.launch_thread("threadIdx.x", 128)
            A_smem = Tx.alloc_buffer((16, 16), "float32", scope="shared")
            Tx.ptx.cp_async.bulk.tensor.s2g_reduce(
                2, A_smem.data, Tx.address_of(A_map), "", "add", 0, 0
            )
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


if __name__ == "__main__":
    tvm.testing.main()
