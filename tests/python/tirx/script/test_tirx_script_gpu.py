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

"""TIRx script gpu."""

import pytest
import tvm_ffi

import tvm
import tvm.script
import tvm.testing
from tvm.ir import PrimType, assert_structural_equal
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx


def test_roundtrip_scopeid1():
    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((64,), 'float32', scope='global')) -> None:

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


def from_source(code):
    return tvm.script.from_source(code, extra_vars={"I": tvm.script.ir, "T": tvm.script.tirx})


def test_roundtrip_scopeid2():
    # fmt: off
    @T.prim_func
    def test(_: T.Buffer((64,), 'float32', scope='global')) -> None:

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
    assert " = T.cta_id_in_pair()" in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_scopeid_deferred():
    """Deferred ScopeIdDef (extent=None) survives print→parse round-trip
    as a no-arg ``T.cta_id()``/``T.thread_id()`` etc. call."""

    # fmt: off
    @T.prim_func(private=True)
    def test(_: T.Buffer((64,), 'float32', scope='global')) -> None:

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
    assert " = T.cta_id()" in code
    assert " = T.thread_id()" in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_exec_scope_filter_guard_roundtrip():
    @T.prim_func(private=True)
    def test(A: T.Buffer((1,), "float32", scope="global")) -> None:
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([128])
        if (0 <= tx) & (tx < 1):
            A[0] = T.float32(1)

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op1():
    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((64,), 'float32', scope='global')) -> None:

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
    def test(
        A: T.Buffer((128, 128), "float16", scope="global"),
        B: T.Buffer((128, 64), "float16", scope="global"),
        C: T.Buffer((128, 64), "float32", scope="global"),
    ) -> None:

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
    def test(
        A: T.Buffer((128, K), "float16", scope="global"),
        B: T.Buffer((K, 64), "float16", scope="global"),
        C: T.Buffer((128, 64), "float32", scope="global"),
    ) -> None:

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
    def func1(A: T.Buffer([128], "float32")):
        T.func_attr({"global_symbol": "func"})

        A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.tensormap_init", T.address_of(A_map), T.reinterpret("handle", A.data)
        )
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


def test_roundtrip_op_call_workspace():
    # fmt: off
    @T.prim_func
    def test(
        A: T.Buffer([10], "float32", scope="global"), B: T.Buffer([10], "float32", scope="global")
    ):

        T.device_entry()
        smem = T.alloc_buffer([10], "float32", scope="shared")
        Tx.add(B, A, T.float32(1), workspace={"smem": smem})
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op_call_config():
    # fmt: off
    @T.prim_func
    def test(
        A: T.Buffer([10], "float32", scope="global"), B: T.Buffer([10], "float32", scope="global")
    ):

        T.device_entry()
        Tx.add(B, A, T.float32(1), schedule="A")
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


def test_roundtrip_persistent_decorator():
    """@T.prim_func(persistent=True) should round-trip."""

    # fmt: off
    @T.prim_func(persistent=True)
    def test(A: T.Buffer((128,), 'float32', scope='global')) -> None:

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
    def test(A: T.Buffer((128,), 'float32', scope='global')) -> None:

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
    def test(A: T.Buffer((128,), 'float32', scope='global')) -> None:

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
    warp_name = next(
        line.partition(" = ")[0].strip()
        for line in code.splitlines()
        if " = T.warp_id_in_wg([4])" in line
    )
    assert f"{warp_name} == 1" in code, f"should have warp_id==1 guard:\n{code}"
    assert f"{warp_name} == 0" in code, f"should have warp_id==0 guard:\n{code}"
    assert "setmaxnreg" in code, f"should have setmaxnreg:\n{code}"
    assert f"if {warp_name} == 1:" in code, f"should have warp_id==1 if-guard:\n{code}"
    assert f"if {warp_name} == 0:" in code, f"should have warp_id==0 if-guard:\n{code}"
    # The printed code is valid TIR — it should parse back
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_warpgroup_role():
    """WarpgroupRole should emit guarded warpgroup scope plus setmaxnreg."""
    from tvm.tirx.lang.warp_role import WarpgroupRole

    # fmt: off
    @T.prim_func
    def test(A: T.Buffer((128,), 'float32', scope='global')) -> None:

        T.device_entry()
        cta_id = T.cta_id([1])
        wg_id = T.warpgroup_id([4])
        warp_id_in_wg = T.warp_id_in_wg([4])
        lane_id = T.lane_id([32])
        with WarpgroupRole(wg_id, 2, regs=200, increase=True):
            Tx.cta.fill(A[0:32], T.float32(0))
        # fmt: on

    code = test.script()
    group_name = next(
        line.partition(" = ")[0].strip()
        for line in code.splitlines()
        if " = T.warpgroup_id([4])" in line
    )
    assert f"{group_name} == 2" in code, f"should have wg_id==2 guard:\n{code}"
    assert "setmaxnreg" in code, f"should have setmaxnreg:\n{code}"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


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
    def func(_: T.Buffer((16, 16), 'float32')):

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
    def func(_: T.Buffer((16, 16), 'float32')):

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
    def func(_: T.Buffer((16, 16), 'float32')):

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
    def func(_: T.Buffer((16, 16), 'float32')):

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


def test_scope_id_dtype_uint32():
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer((128,), 'float32')):

        T.device_entry()
        bx = T.cta_id([1])
        tx = T.thread_id([128], dtype="uint32")
        A[tx] = T.float32(bx)
    # fmt: on

    scope_defs = []
    tvm_ffi.structural_walk(
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


def _assert_roundtrip(func):
    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_scope_id_dtype_uint32_lane_and_warp():
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer((32,), 'float32')):

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
    def func(A: T.Buffer((4,), 'float32')):

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
    def func(A: T.Buffer((32,), 'float32')):

        T.device_entry()
        _ = T.cta_id([1])
        lane = T.lane_id(dtype="uint32")
        warp = T.warp_id([4])
        A[lane] = T.float32(warp)
    # fmt: on

    scope_defs = []
    tvm_ffi.structural_walk(
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
        def func(A: T.Buffer((128,), 'float32')):

            T.device_entry()
            _ = T.cta_id([1])
            tx = T.thread_id([128], dtype=dtype)
            A[tx] = T.float32(1)
