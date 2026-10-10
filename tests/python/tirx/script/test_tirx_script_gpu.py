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


def test_roundtrip_scopeid1():
    # fmt: off
    @T.function
    def test(A: T.Tensor((64,), 'float32', scope='global')) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1, 1, 1), block=1 * 32))
        bx, by, bz = (T.cuda.block_idx('x'), T.cuda.block_idx('y'), T.cuda.block_idx('z'))
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        A_local = T.alloc_tensor([1], dtype="float16", scope="local")
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
    @T.function
    def test(_: T.Tensor((64,), 'float32', scope='global')) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(8, 10, 12), block=32, cluster=(2, 2, 1)))
        bx, by, bz = (T.cuda.block_idx('x'), T.cuda.block_idx('y'), T.cuda.block_idx('z'))
        cbx, cby, cbz = (
            T.cuda.cluster_cta_id("x"),
            T.cuda.cluster_cta_id("y"),
            T.cuda.cluster_cta_id("z"),
        )
        cta_id_in_pair = T.cuda.cta_pair_id()
        clx, cly, clz = (T.cuda.cluster_id('x'), T.cuda.cluster_id('y'), T.cuda.cluster_id('z'))
        T.evaluate(bx + by + bz)
        T.evaluate(cbx + cby + cbz)
        T.evaluate(cta_id_in_pair)
        T.evaluate(clx + cly + clz)
        # fmt: on

    code = test.script()
    assert " = T.cuda.cta_pair_id()" in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_scopeid_deferred():
    """CUDA index calls survive print/parse without embedding launch extents."""

    # fmt: off
    @T.function(private=True)
    def test(_: T.Tensor((64,), 'float32', scope='global')) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=4 * 2, block=4 * 32, cluster=(2,)))
        bx = T.cuda.block_idx('x')                       # deferred kernel→cta
        cbx = T.cuda.cluster_cta_id('x')
        clx = T.cuda.cluster_id('x')
        tx = T.cuda.thread_idx('x')                    # deferred cta→thread
        _lane = T.cuda.lane_id()
        T.evaluate(bx + cbx + clx + tx)
        # fmt: on

    code = test.script()
    assert " = T.cuda.block_idx(\"x\")" in code
    assert " = T.cuda.thread_idx(\"x\")" in code
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_exec_scope_filter_guard_roundtrip():
    @T.function(private=True)
    def test(A: T.Tensor((1,), "float32", scope="global")) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(128,)))
        tx = T.cuda.thread_idx("x")
        if (0 <= tx) & (tx < 1):
            A[0] = T.float32(1)

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op1():
    # fmt: off
    @T.function
    def test(A: T.Tensor((64,), 'float32', scope='global')) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1, 1, 1), block=1 * 32))
        bx, by, bz = (T.cuda.block_idx('x'), T.cuda.block_idx('y'), T.cuda.block_idx('z'))
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        A_smem = T.alloc_tensor([64], dtype="float32", scope="shared")

        transfer_src_125 = T.meta_var(A[tuple([slice(None) for _ in A.shape])])
        transfer_dst_125 = T.meta_var(A_smem)
        transfer_reg_125 = T.alloc_tensor(
            [r.extent for r in transfer_src_125.region],
            transfer_src_125.source.dtype,
            scope="local",
        )
        T.cuda.tile.ld(transfer_reg_125, transfer_src_125, scope="cta")
        T.cuda.tile.st(transfer_dst_125, transfer_reg_125, scope="cta")
        for i in range(10):
            T.cuda.tile.mov(A_smem, T.float32(0), scope='cta')
            T.cuda.tile.mma_sync(A_smem, A_smem, A_smem, A_smem, scope='cta')
        transfer_src_129 = T.meta_var(A_smem[tuple([slice(None) for _ in A_smem.shape])])
        transfer_dst_129 = T.meta_var(A)
        transfer_reg_129 = T.alloc_tensor(
            [r.extent for r in transfer_src_129.region],
            transfer_src_129.source.dtype,
            scope="local",
        )
        T.cuda.tile.ld(transfer_reg_129, transfer_src_129, scope="cta")
        T.cuda.tile.st(transfer_dst_129, transfer_reg_129, scope="cta")
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op2():
    # fmt: off
    @T.function
    def test(
        A: T.Tensor((128, 128), "float16", scope="global"),
        B: T.Tensor((128, 64), "float16", scope="global"),
        C: T.Tensor((128, 64), "float32", scope="global"),
    ) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1, 1, 1), block=4 * 32))
        bx, by, bz = (T.cuda.block_idx('x'), T.cuda.block_idx('y'), T.cuda.block_idx('z'))
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        A_smem = T.alloc_tensor([128, 32], dtype="float16", scope="shared")
        B_smem = T.alloc_tensor([32, 64], dtype="float16", scope="shared")

        C_local = T.alloc_tensor([128, 64], dtype="float32", scope="local")
        for k in range(4):
            transfer_src_155 = T.meta_var(A[:, k * 32:k * 32 + 32])
            transfer_dst_155 = T.meta_var(A_smem)
            transfer_reg_155 = T.alloc_tensor(
                [r.extent for r in transfer_src_155.region],
                transfer_src_155.source.dtype,
                scope="local",
            )
            T.cuda.tile.ld(transfer_reg_155, transfer_src_155, scope="cta")
            T.cuda.tile.st(transfer_dst_155, transfer_reg_155, scope="cta")
            transfer_src_156 = T.meta_var(B[k * 32:k * 32 + 32, 0:64])
            transfer_dst_156 = T.meta_var(B_smem)
            transfer_reg_156 = T.alloc_tensor(
                [r.extent for r in transfer_src_156.region],
                transfer_src_156.source.dtype,
                scope="local",
            )
            T.cuda.tile.ld(transfer_reg_156, transfer_src_156, scope="cta")
            T.cuda.tile.st(transfer_dst_156, transfer_reg_156, scope="cta")
            T.cuda.tile.mma_sync(C_local, A_smem, B_smem, C_local, scope='cta')
        T.cuda.tile.st(C, C_local, scope='cta')
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op3():
    # fmt: off
    NUM_STAGES = 3
    K = 4096

    @T.function
    def test(
        A: T.Tensor((128, K), "float16", scope="global"),
        B: T.Tensor((K, 64), "float16", scope="global"),
        C: T.Tensor((128, 64), "float32", scope="global"),
    ) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1, 1, 1), block=4 * 32))
        bx, by, bz = (T.cuda.block_idx('x'), T.cuda.block_idx('y'), T.cuda.block_idx('z'))
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        A_smem = T.alloc_tensor([NUM_STAGES, 128, 32], dtype="float16", scope="shared")
        B_smem = T.alloc_tensor([NUM_STAGES, 32, 64], dtype="float16", scope="shared")

        C_local = T.alloc_tensor([128, 64], dtype="float32", scope="local")
        for i in range(NUM_STAGES - 1):
            transfer_src_187 = T.meta_var(A[:, i * 32:i * 32 + 32])
            transfer_dst_187 = T.meta_var(A_smem[i, :, :])
            transfer_reg_187 = T.alloc_tensor(
                [r.extent for r in transfer_src_187.region],
                transfer_src_187.source.dtype,
                scope="local",
            )
            T.cuda.tile.ld(transfer_reg_187, transfer_src_187, scope="cta")
            T.cuda.tile.st(transfer_dst_187, transfer_reg_187, scope="cta")
            transfer_src_188 = T.meta_var(B[i * 32:i * 32 + 32, :])
            transfer_dst_188 = T.meta_var(B_smem[i, :, :])
            transfer_reg_188 = T.alloc_tensor(
                [r.extent for r in transfer_src_188.region],
                transfer_src_188.source.dtype,
                scope="local",
            )
            T.cuda.tile.ld(transfer_reg_188, transfer_src_188, scope="cta")
            T.cuda.tile.st(transfer_dst_188, transfer_reg_188, scope="cta")

        for k in range(K // 32):
            copy_k = T.meta_var(k + NUM_STAGES - 1)
            gemm_stage = T.meta_var(k % NUM_STAGES)
            copy_stage = T.meta_var(copy_k % NUM_STAGES)
            transfer_src_194 = T.meta_var(A[:, copy_k * 32:copy_k * 32 + 32])
            transfer_dst_194 = T.meta_var(A_smem[copy_stage, :, :])
            transfer_reg_194 = T.alloc_tensor(
                [r.extent for r in transfer_src_194.region],
                transfer_src_194.source.dtype,
                scope="local",
            )
            T.cuda.tile.ld(transfer_reg_194, transfer_src_194, scope="cta")
            T.cuda.tile.st(transfer_dst_194, transfer_reg_194, scope="cta")
            transfer_src_195 = T.meta_var(B[copy_k * 32:copy_k * 32 + 32, :])
            transfer_dst_195 = T.meta_var(B_smem[copy_stage, :, :])
            transfer_reg_195 = T.alloc_tensor(
                [r.extent for r in transfer_src_195.region],
                transfer_src_195.source.dtype,
                scope="local",
            )
            T.cuda.tile.ld(transfer_reg_195, transfer_src_195, scope="cta")
            T.cuda.tile.st(transfer_dst_195, transfer_reg_195, scope="cta")
            T.cuda.tile.mma_sync(
                C_local, A_smem[gemm_stage, :, :], B_smem[gemm_stage, :, :], C_local, scope="cta"
            )

        T.cuda.tile.st(C, C_local, scope='cta')
        # fmt: on

    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_tensormap():
    # fmt: off
    @T.function
    def func1(A: T.Tensor([128], "float32")):
        T.func_attr({"global_symbol": "func"})

        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        T.call_packed(
            "runtime.tensormap_init", T.address_of(A_map), T.reinterpret( A.data, ty="handle")
        )
    # fmt: on
    code = func1.script()
    assert from_source(code).script() == code
    assert_structural_equal(func1, from_source(code))


def test_roundtrip_tensormap_kernel_param():
    # fmt: off
    @T.function
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
    @T.function
    def test(
        A: T.Tensor([10], "float32", scope="global"), B: T.Tensor([10], "float32", scope="global")
    ):

        T.device_entry()
        smem = T.alloc_tensor([10], "float32", scope="shared")
        T.trn.tile.activation(B, A, bias=1.0, const_bias=smem)
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_op_call_config():
    # fmt: off
    @T.function
    def test(
        A: T.Tensor([10], "float32", scope="global"), B: T.Tensor([10], "float32", scope="global")
    ):

        T.device_entry()
        T.cuda.tile.add(B, A, T.float32(1), rounding_mode="rz")
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_predicate():
    # fmt: off
    @T.function
    def test():
        T.device_entry()
        A = T.alloc_tensor([10, 10], "float32")
        B = T.alloc_tensor([10, 10], "float32")
        T.trn.tile.affine_select(B, A, 1.0, lambda i, j: i < j)
        # fmt: on
    code = test.script()
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_kwargs_op_call():
    # fmt: off
    @T.function(private=True)
    def test(A: T.Tensor((10, 10), "float32"), B: T.Tensor((10, 10), "float32")):
        T.device_entry()
        kwargs = T.meta_var({"descriptor_mode": "auto", "cta_group": 2})
        T.cuda.tile.cp_async_bulk_tensor_load(A[:, :], B[:, :], A.data, **kwargs)
        # fmt: on
    code = test.script()
    print(code)
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_persistent_decorator():
    """@T.function(persistent=True) should round-trip."""

    # fmt: off
    @T.function(persistent=True)
    def test(A: T.Tensor((128,), 'float32', scope='global')) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 32))
        cta_id = T.cuda.block_idx('x')
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        T.cuda.tile.mov(A[0:32], T.float32(0), scope='cta')
        # fmt: on

    code = test.script()
    assert "persistent=True" in code, f"persistent not in decorator:\n{code}"
    assert "tirx.persistent_kernel" not in code, "should NOT appear as func_attr"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_persistent_not_present():
    """Without persistent=True, the keyword should not appear."""

    # fmt: off
    @T.function
    def test(A: T.Tensor((128,), 'float32', scope='global')) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 32))
        cta_id = T.cuda.block_idx('x')
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()
        T.cuda.tile.mov(A[0:32], T.float32(0), scope='cta')
        # fmt: on

    code = test.script()
    assert "persistent" not in code, f"persistent should NOT appear:\n{code}"


def test_warp_role():
    """WarpRole should emit guarded warp scopes plus setmaxnreg."""
    from tvm.tirx.lang.warp_role import WarpRole

    # fmt: off
    @T.function
    def test(A: T.Tensor((128,), 'float32', scope='global')) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=4 * 128))
        cta_id = T.cuda.block_idx('x')
        wg_id = T.cuda.warpgroup_id()
        warp_id = T.cuda.warp_in_warpgroup()
        lane_id = T.cuda.lane_id()
        with WarpRole(warp_id, 1, regs=48):
            T.cuda.tile.mov(A[0:32], T.float32(0), scope='cta')
        with WarpRole(warp_id, 0, regs=232, increase=True):
            T.cuda.tile.mov(A[32:64], T.float32(1), scope='cta')
        # fmt: on

    code = test.script()
    warp_name = next(
        line.partition(":")[0].strip()
        for line in code.splitlines()
        if " = T.cuda.warp_in_warpgroup()" in line
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
    @T.function
    def test(A: T.Tensor((128,), 'float32', scope='global')) -> None:

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=4 * 128))
        cta_id = T.cuda.block_idx('x')
        wg_id = T.cuda.warpgroup_id()
        warp_id_in_wg = T.cuda.warp_in_warpgroup()
        lane_id = T.cuda.lane_id()
        with WarpgroupRole(wg_id, 2, regs=200, increase=True):
            T.cuda.tile.mov(A[0:32], T.float32(0), scope='cta')
        # fmt: on

    code = test.script()
    group_name = next(
        line.partition(":")[0].strip()
        for line in code.splitlines()
        if " = T.cuda.warpgroup_id()" in line
    )
    assert f"{group_name} == 2" in code, f"should have wg_id==2 guard:\n{code}"
    assert "setmaxnreg" in code, f"should have setmaxnreg:\n{code}"
    assert from_source(code).script() == code
    assert_structural_equal(test, from_source(code))


def test_roundtrip_cuda_func_call_source_code():
    """cuda_func_call with a multiline trailing source operand must print with
    inline string literal, not as a metadata reference."""

    # fmt: off
    @T.function
    def func():
        T.device_entry()
        desc = T.alloc_local((1,), "uint64")
        T.cuda.func_call("my_func", T.address_of(desc[0]), "\n__device__ void my_func(uint64_t* p) {\n    *p = 42;\n}\n")  # noqa: E501
        # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_roundtrip_cp_async_bulk_tensor_g2s_cluster():
    """The TMA load composite [tensorMap, coords] operand must round-trip."""

    # fmt: off
    @T.function(check_well_formed=False)
    def func(_: T.Tensor((16, 16), 'float32')):

        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        with T.launch_thread("blockIdx.x", 1):
            T.launch_thread("threadIdx.x", 128)
            A_smem = T.alloc_tensor((16, 16), "float32", scope="shared")
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
    @T.function(check_well_formed=False)
    def func(_: T.Tensor((16, 16), 'float32')):

        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        with T.launch_thread("blockIdx.x", 1):
            T.launch_thread("threadIdx.x", 128)
            A_smem = T.alloc_tensor((16, 16), "float32", scope="shared")
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
    @T.function(check_well_formed=False)
    def func(_: T.Tensor((16, 16), 'float32')):

        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
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
    @T.function(check_well_formed=False)
    def func(_: T.Tensor((16, 16), 'float32')):

        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        with T.launch_thread("blockIdx.x", 1):
            T.launch_thread("threadIdx.x", 128)
            A_smem = T.alloc_tensor((16, 16), "float32", scope="shared")
            T.ptx["cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group"](
                T.address_of(A_map), 0, 0, A_smem.data
            )
    # fmt: on

    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_scope_id_dtype_uint32():
    # fmt: off
    @T.function
    def func(A: T.Tensor((128,), 'float32')):

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(128,)))
        bx = T.cuda.block_idx('x')
        tx = T.cast(T.cuda.thread_idx('x'), "uint32")
        A[tx] = T.float32(bx)
    # fmt: on

    scope_defs = []
    tvm_ffi.structural_walk(
        func.body,
        lambda s: scope_defs.append(s) if isinstance(s, tvm.ir.Bind) else None,
    )
    dtypes = {str(d.var.ty) for d in scope_defs}
    assert dtypes == {"int32", "uint32"}
    code = func.script()
    assert 'T.Cast("uint32", T.cuda.thread_idx("x"))' in code
    assert 'T.cuda.block_idx("x")' in code
    _assert_roundtrip(func)


def _assert_roundtrip(func):
    code = func.script()
    assert from_source(code).script() == code
    assert_structural_equal(func, from_source(code))


def test_scope_id_dtype_uint32_lane_and_warp():
    # fmt: off
    @T.function
    def func(A: T.Tensor((32,), 'float32')):

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=4 * 32))
        _ = T.cuda.block_idx('x')
        warp = T.cast(T.cuda.warp_id(), "uint32")
        lane = T.cast(T.cuda.lane_id(), "uint32")
        A[lane] = T.float32(warp)
    # fmt: on

    code = func.script()
    assert 'T.Cast("uint32", T.cuda.warp_id())' in code
    assert 'T.Cast("uint32", T.cuda.lane_id())' in code
    _assert_roundtrip(func)


def test_scope_id_dtype_uint32_with_preferred():
    # fmt: off
    @T.function
    def func(A: T.Tensor((4,), 'float32')):

        T.device_entry(
            launch=T.cuda.LaunchConfig(
                grid=(4, 2), block=32, cluster=(2, 2), preferred_cluster=(2, 2)
            )
        )
        _ = T.cuda.cluster_id('x')
        cx, cy = (
            T.cast(T.cuda.cluster_cta_id("x"), "uint32"),
            T.cast(T.cuda.cluster_cta_id("y"), "uint32"),
        )
        tx = T.cuda.thread_idx('x')
        if tx == 0:
            A[cx + cy] = T.float32(1)
    # fmt: on

    code = func.script()
    assert 'T.Cast("uint32"' in code
    _assert_roundtrip(func)


def test_scope_id_dtype_uint32_deferred_extent():
    """Explicit casts retain the binding dtype."""

    # fmt: off
    @T.function
    def func(A: T.Tensor((32,), 'float32')):

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=4 * 32))
        _ = T.cuda.block_idx('x')
        lane = T.cast(T.cuda.lane_id(), "uint32")
        warp = T.cuda.warp_id()
        A[lane] = T.float32(warp)
    # fmt: on

    scope_defs = []
    tvm_ffi.structural_walk(
        func.body,
        lambda s: scope_defs.append(s) if isinstance(s, tvm.ir.Bind) else None,
    )
    lane = next(bind for bind in scope_defs if bind.var.name == "lane")
    assert lane.var.ty == PrimType("uint32")
    _assert_roundtrip(func)


@pytest.mark.parametrize("axis", ["w", "xy", "", 0])
def test_cuda_index_rejects_invalid_axis(axis):
    with pytest.raises((TypeError, ValueError)):
        T.cuda.block_idx(axis)
