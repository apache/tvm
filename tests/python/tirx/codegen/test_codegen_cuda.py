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
# pylint: disable=missing-function-docstring
import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.testing import env


def _get_source(func: tvm.tirx.PrimFunc) -> str:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return src, mod


def _helper_source(src: str, helper_name: str) -> str:
    start = src.index(helper_name)
    next_helper = src.find("__device__", start + len(helper_name))
    if next_helper == -1:
        return src[start:]
    return src[start:next_helper]


def test_tirx_launch_bounds_omits_min_blocks_without_persistent_schedule():
    @T.prim_func
    def main(A: T.Buffer((4,), "int32")):
        T.device_entry()
        bx = T.cta_id([4])
        tx = T.thread_id([128])
        if tx == 0:
            A[bx] = A[bx] + 1

    src, _ = _get_source(main)
    assert 'extern "C" __global__ void __launch_bounds__(128) main_kernel' in src
    assert "__launch_bounds__(128, 1)" not in src


def test_tirx_launch_bounds_min_blocks_attr_sets_one_block_per_sm():
    @T.prim_func
    def main(A: T.Buffer((4,), "int32")):
        T.device_entry()
        T.attr({"tirx.launch_bounds_min_blocks_per_sm": 1})
        bx = T.cta_id([4])
        tx = T.thread_id([128])
        if tx == 0:
            A[bx] = A[bx] + 1

    src, _ = _get_source(main)
    assert 'extern "C" __global__ void __launch_bounds__(128, 1) main_kernel' in src
    assert "tirx.launch_bounds_min_blocks_per_sm" not in src


def test_serial_pragma_unroll_codegen():
    @T.prim_func
    def main(A: T.Buffer((4,), "int32")):
        T.device_entry()
        tx = T.thread_id([32])
        if tx == 0:
            for i in T.serial(4, unroll=True):
                if i == 2:
                    break
                A[i] = A[i] + 1

    src, _ = _get_source(main)
    assert "#pragma unroll\n" in src
    assert "for (" in src
    assert "break;" in src


def test_cluster_cta_id_codegen_uses_coordinate_sregs():
    @T.prim_func
    def main(A: T.Buffer((1,), "int32")):
        T.device_entry()
        cbx, cby = T.cta_id_in_cluster([2, 2])
        tx = T.thread_id([32])
        if tx == 0:
            A[0] = cbx + cby

    src, _ = _get_source(main)
    assert "%cluster_ctaid.x" in src
    assert "%cluster_ctaid.y" in src
    assert "%cluster_ctarank" not in src
    assert "cooperative_groups::cluster_group::block_index" not in src


def test_cuda_handle_uint64_reinterpret_codegen():
    @T.prim_func
    def main(A: T.Buffer((1,), "uint64")):
        T.device_entry()
        tx = T.thread_id([32])
        if tx == 0:
            ptr = T.reinterpret("handle", A[0])
            A[0] = T.reinterpret("uint64", ptr)

    src, _ = _get_source(main)
    assert "reinterpret_cast<void*>" in src
    assert "reinterpret_cast<uint64_t>" in src
    assert "*(void* *)" not in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_atomic_add():
    @T.prim_func
    def main(A: T.Buffer((1,), "int32"), B: T.Buffer((1,), "float32")):
        T.device_entry()
        cta_id = T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            T.cuda.atomic_add(A.data, T.int32(1))
            T.cuda.atomic_add(B.data, T.float32(1.0))

    src, mod = _get_source(main)
    assert "tvm_builtin_cuda_atomic_add" in src
    A_np = np.zeros(1, dtype="int32")
    B_np = np.zeros(1, dtype="float32")

    def run_and_check():
        dev = tvm.cuda()
        A_tvm = tvm.runtime.tensor(A_np, device=dev)
        B_tvm = tvm.runtime.tensor(B_np, device=dev)
        mod["main"](A_tvm, B_tvm)
        np.testing.assert_allclose(A_tvm.numpy(), 1)
        np.testing.assert_allclose(B_tvm.numpy(), 1.0)

    tvm.testing.run_with_gpu_lock(run_and_check)


def test_ptx_ld_acquire_and_volatile_codegen():
    @T.prim_func
    def main(A: T.Buffer((1,), "uint64"), B: T.Buffer((1,), "int32"), C: T.Buffer((1,), "uint32")):
        T.device_entry()
        tx = T.thread_id([32])
        if tx == 0:
            A[0] = T.ptx.ld_acquire(A.data, "uint64", "u64", scope="gpu", space="global")
            B[0] = T.ptx.ld_acquire(B.data, "int32", "s32", scope="sys", space="global")
            C[0] = T.ptx.ld_acquire(C.data, "uint32", "b32", scope="gpu", space="global")
            T.ptx.ld_global_acquire(B[0], B.data)
            A[0] = T.ptx.ld_volatile(A.data, "uint64", "u64", space="global")

    src, _ = _get_source(main)
    assert "ld.acquire.gpu.global.u64" in src
    assert "ld.acquire.sys.global.s32" in src
    assert "ld.acquire.gpu.global.b32" in src
    assert "ptx_ld_global_acquire_int32" in src
    assert "ptx_ld_global_acquire_b32" not in src
    assert "ld.volatile.global.u64" in src


def test_megamoe_extracted_intrinsics_codegen():
    @T.prim_func
    def main(
        U32: T.Buffer((4,), "uint32"),
        I32: T.Buffer((1,), "int32"),
        U64: T.Buffer((1,), "uint64"),
        F32: T.Buffer((4,), "float32"),
    ):
        T.device_entry()
        tx = T.thread_id([32])
        if tx == 0:
            T.ptx.red_scalar(
                U64.data,
                U64[0],
                sem="release",
                scope="gpu",
                space="global",
                op="or",
                ptx_type="b64",
            )
            T.ptx.red_scalar(
                I32.data,
                I32[0],
                sem="release",
                scope="sys",
                space="global",
                op="add",
                ptx_type="s32",
            )
            U32[0] = T.ptx.atom_scalar(
                U32.data,
                U32[0],
                sem="release",
                scope="gpu",
                space="global",
                op="add",
                ptx_type="u32",
            )
            U64[0] = T.ptx.atom_scalar(
                U64.data, U64[0], scope="sys", space="global", op="add", ptx_type="u64"
            )
            T.ptx.red_scalar(
                U32.data, U32[0], scope="gpu", space="global", op="add", ptx_type="u32"
            )
            T.ptx.st(U32.data, U32[0], space="shared", ptx_type="u32")
            T.ptx.st(
                U32.data,
                U32[0],
                U32[1],
                U32[2],
                U32[3],
                space="shared",
                vec="v4",
                ptx_type="b32",
            )
            T.ptx.st_bulk(U32.data, T.uint32(16), weak=True, space="shared::cta")
            U32[0] = T.ptx.fns_b32(U32[0], U32[1], I32[0])
            T.ptx.stmatrix(
                True,  # trans
                1,  # num
                ".b8",  # dtype
                U32.data,  # smem_ptr
                U32.data,  # src0
                shape="m16n8",
                space="shared",
            )

            F32[1] = T.cuda.uint_as_float(U32[0])
            F32[2] = T.ptx.ld(F32.data, "float32", "f32", space="global")
            U32[3] = T.cuda.float_as_uint(F32[1])
            F32[0] = T.ptx.add_rn_f32_bf16(F32[0], T.cast(U32[0], "uint16"))
            U64[0] = T.reinterpret("uint64", U32.data)
            U32[0] = T.cuda.ballot_sync(T.uint32(0xFFFFFFFF), I32[0])
            I32[0] = T.cuda.ffs_u32(U32[0])
            U32[0] = T.cuda.reduce_add_sync_u32(T.uint32(0xFFFFFFFF), U32[0])
            U32[0] = T.cuda.reduce_min_sync_u32(T.uint32(0xFFFFFFFF), U32[0])
            U64[0] = T.cuda.clock64()
            U32[0] = T.cuda.float22bfloat162_rn(F32[0], F32[1])

    src, _ = _get_source(main)
    for snippet in [
        "red.release.gpu.global.or.b64",
        "red.release.sys.global.add.s32",
        "atom.release.gpu.global.add.u32",
        "atom.sys.global.add.u64",
        "red.gpu.global.add.u32",
        "st.shared.u32",
        "st.shared.v4.b32",
        "st.bulk.weak.shared::cta",
        "fns.b32",
        "stmatrix.sync.aligned.m16n8.x1.trans.shared.b8",
        "ld.global.f32",
        "add.rn.f32.bf16",
        "__uint_as_float",
        "__float_as_uint",
        "__ballot_sync",
        "__ffs",
        "__reduce_add_sync",
        "__reduce_min_sync",
        "clock64()",
        "__float22bfloat162_rn",
    ]:
        assert snippet in src


def test_ptx_cp_async_bulk_non_tma_form_codegen():
    @T.prim_func
    def main(
        A: T.Buffer((128,), "float32"),
        B: T.Buffer((128,), "float32"),
        C: T.Buffer((1,), "uint64"),
    ):
        T.device_entry()
        tx = T.thread_id([32])
        if tx == 0:
            smem = T.alloc_shared([128], "float32")
            T.ptx.cp_async_bulk_g2s_cta(
                smem.ptr_to([0]), A.data, T.uint32(64), smem.ptr_to([0]), cache_policy=C[0]
            )
            T.ptx.cp_async_bulk_g2s_cluster(
                smem.ptr_to([0]), A.data, T.uint32(64), smem.ptr_to([0]), cache_policy=C[0]
            )
            T.ptx.cp_async_bulk_s2g(B.data, smem.ptr_to([0]), T.uint32(64), cache_policy=C[0])

    src, _ = _get_source(main)
    assert "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint" in src
    assert "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint" in src
    assert "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint" in src
    assert "unsigned long long cache_policy" in src


def test_tensor_map_param_codegen():
    @T.prim_func
    def main(A_map: T.TensorMap()):
        T.device_entry()
        tx = T.thread_id([32])
        if tx == 0:
            T.evaluate(T.address_of(A_map))

    src, _ = _get_source(main)
    assert "const __grid_constant__ CUtensorMap A_map" in src
    assert "((unsigned long long)(&(A_map)))" in src


def test_tma_cache_policy_operand_codegen():
    @T.prim_func
    def main(Cache: T.Buffer((1,), "uint64")):
        A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        B_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)

        T.device_entry()
        tx = T.thread_id([32])
        if tx == 0:
            smem = T.alloc_buffer((128,), "float32", scope="shared", align=128)
            bar = T.shared_scalar("uint64")
            T.ptx.cp_async.bulk.tensor.g2c(
                2,
                smem.data,
                T.address_of(bar),
                T.address_of(A_map),
                1,
                2,
                "",
                0,
                0,
                cache_policy=Cache[0],
            )
            T.ptx.cp_async.bulk.tensor.g2c(
                2,
                smem.data,
                T.address_of(bar),
                T.address_of(A_map),
                3,
                2,
                "",
                0,
                0,
                cache_policy=Cache[0],
            )
            T.ptx.cp_async.bulk.tensor.s2g(
                2, smem.data, T.address_of(A_map), "", 0, 0, cache_policy=Cache[0]
            )
            masked_bar = T.cuda.sm100_tma_2sm_mbarrier_addr(T.address_of(bar))
            T.ptx.cp_async.bulk.tensor.g2c_bar_addr(
                2,
                smem.data,
                masked_bar,
                T.address_of(A_map),
                1,
                2,
                "",
                0,
                0,
                cache_policy=Cache[0],
            )
            if tx == 0:
                T.ptx.cp_async.bulk.tensor.g2c_bar_addr(
                    2,
                    smem.data,
                    masked_bar,
                    T.address_of(A_map),
                    1,
                    2,
                    "",
                    0,
                    0,
                    cache_policy=Cache[0],
                )
            else:
                T.ptx.cp_async.bulk.tensor.g2c_bar_addr(
                    2,
                    smem.data,
                    masked_bar,
                    T.address_of(B_map),
                    1,
                    2,
                    "",
                    0,
                    0,
                    cache_policy=Cache[0],
                )

    src, _ = _get_source(main)
    assert "ptx_cp_async_bulk_tensor_g2cluster_tile_2d_cache_hint" in src
    assert "ptx_cp_async_bulk_tensor_g2cluster_tile_2d_multicast_cache_hint" in src
    assert "g2cluster_unicast" not in src
    assert "ptx_cp_async_bulk_tensor_g2cta" not in src
    assert (
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
    ) in src
    assert (
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.multicast::cluster"
        ".cta_group::2.L2::cache_hint"
    ) in src
    assert "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint" in src
    assert "tvm_builtin_cp_async_bulk_tensor_2d_g2c_cta_group2" not in src
    assert "tvm_builtin_cuda_cvta_generic_to_shared((&(bar_ptr[0]))) & (uint)4278190079" in src
    assert "ptx_cp_async_bulk_tensor_g2cluster_tile_2d_cache_hint_bar_addr" in src
    assert "unsigned long long cache_policy" in src


def test_cuda_thread_fence():
    @T.prim_func
    def main(A: T.Buffer((16, 16), "int32")):
        T.device_entry()
        cta_id = T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            T.cuda.thread_fence()

    src, mod = _get_source(main)
    assert "tvm_builtin_cuda_thread_fence" in src


def test_cuda_nano_sleep():
    @T.prim_func
    def main(A: T.Buffer((16, 16), "int32")):
        T.device_entry()
        cta_id = T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            T.cuda.nano_sleep(1)

    src, mod = _get_source(main)
    assert "tvm_builtin_cuda_nano_sleep" in src


def test_cuda_atomic_cas():
    @T.prim_func
    def main(A: T.Buffer((16, 16), "int32")):
        T.device_entry()
        cta_id = T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            T.cuda.atomic_cas(A.data, T.int32(1), T.int32(2))

    src, mod = _get_source(main)
    assert "tvm_builtin_cuda_atomic_cas" in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_func_call():
    def test_add_one():
        add_one = """
__device__ int32_t add_one(int32_t a) {
    return a + 1;
}
"""

        @T.prim_func
        def main(a: T.Buffer((16, 16), "int32"), b: T.Buffer((16, 16), "int32")):
            T.device_entry()
            cta_id = T.cta_id([1])
            tx = T.thread_id([32])
            if tx == 0:
                for i, j in T.grid(16, 16):
                    b[i, j] = T.cuda.func_call(
                        "add_one", a[i, j], source_code=add_one, return_type="int32"
                    )

        src, mod = _get_source(main)
        A = np.random.randint(0, 10, (16, 16)).astype("int32")
        B = np.zeros((16, 16), dtype="int32")

        def run_and_check():
            dev = tvm.cuda()
            A_tvm = tvm.runtime.tensor(A, device=dev)
            B_tvm = tvm.runtime.tensor(B, device=dev)
            mod["main"](A_tvm, B_tvm)
            np.testing.assert_allclose(B_tvm.numpy(), A + 1)

        tvm.testing.run_with_gpu_lock(run_and_check)
        print(src)

    test_add_one()

    def test_print():
        print_func = """
__device__ void print(int32_t a) {
    printf("%d\\n", a);
}
"""

        @T.prim_func
        def main(a: T.Buffer((16, 16), "int32")):
            T.device_entry()
            cta_id = T.cta_id([1])
            tx = T.thread_id([32])
            if tx == 0:
                for i, j in T.grid(16, 16):
                    T.cuda.func_call("print", a[i, j], source_code=print_func)

        src, mod = _get_source(main)
        A = np.random.randint(0, 10, (16, 16)).astype("int32")

        def run_and_check():
            dev = tvm.cuda()
            A_tvm = tvm.runtime.tensor(A, device=dev)
            mod["main"](A_tvm)
            dev.sync()

        tvm.testing.run_with_gpu_lock(run_and_check)
        print(src)

    test_print()


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_warp_shuffle_xor_sync():
    # fmt: off
    @T.prim_func
    def func(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (32,), dtype="float32", align=16)

        T.device_entry()
        cta_id = T.cta_id([1])
        warp_id = T.warp_id([1])
        lane_id = T.lane_id([32])

        A_local = T.alloc_buffer([1], "float32", scope="local")
        i = T.alloc_buffer([1], "int32", scope="local")

        A_local[0] = T.float32(31 - lane_id)
        i[0] = 16
        while i[0] >= 1:
            A_local[0] += T.tvm_warp_shuffle_xor(0xFFFFFFFF, A_local[0], i[0], 32, 32)
            i[0] = i[0] // 2

        A[lane_id] = A_local[0]
        # fmt: on

    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    A_np = np.zeros(32, dtype="float32")
    assert "__shfl_xor_sync" in mod.mod.imports[0].inspect_source()
    A_ref = np.ones(32, dtype="float32") * 496

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, device=dev)
        mod(A)
        np.testing.assert_allclose(A.numpy(), A_ref)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("cp_size", [4, 8, 16])
@pytest.mark.parametrize("cache_hint", ["", "evict_last"])
@pytest.mark.parametrize("prefetch_size", [-1, 64, 128, 256])
@pytest.mark.parametrize("predicate", [-1, T.int32(0), T.int32(1)])
@pytest.mark.parametrize("fill_mode", ["", "zero"])
def test_ptx_cp_async(cp_size, cache_hint, prefetch_size, predicate, fill_mode):
    if fill_mode != "" and predicate == -1:
        return

    N = cp_size // 2

    # fmt: off
    @T.prim_func
    def main(A: T.Buffer((N), "float16")):
        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([32])
        A_shared = T.alloc_shared([N], "float16")
        for i in T.vectorized(N):
            A_shared[i] = 5.0
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.cp_async(A_shared.ptr_to([0]), A.ptr_to([0]), cp_size, cache_hint=cache_hint, prefetch_size=prefetch_size, predicate=predicate, fill_mode=fill_mode)  # noqa: E501
        T.ptx.cp_async.commit_group()
        T.ptx.cp_async.wait_group(0)
        for i in T.serial(N):
            A[i] = A_shared[i] + 1.0
        # fmt: on

    src, mod = _get_source(main)
    A_np = np.ones(N, dtype="float16")
    A_ref = np.ones(N, dtype="float16") * 2
    if int(predicate) == 0:
        if fill_mode == "zero":
            A_ref = np.ones(N, dtype="float16")
        else:
            A_ref = np.ones(N, dtype="float16") * 6

    def run_and_check():
        dev = tvm.cuda()
        A = tvm.runtime.tensor(A_np, device=dev)
        mod(A)
        np.testing.assert_allclose(A.numpy(), A_ref)

    tvm.testing.run_with_gpu_lock(run_and_check)
    print(src)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("num", [1, 2, 4])
def test_ptx_ldmatrix(trans, num):
    dtype = ".b16"

    # fmt: off
    @T.prim_func
    def main(A: T.Buffer((16, 16), "float16"), B: T.Buffer((16, 16), "float16")):
        T.device_entry()
        cta_id = T.cta_id([1])
        tx = T.thread_id([32])
        A_shared = T.alloc_shared([16, 16], "float16")
        if tx == 0:
            for i, j in T.grid(16, 16):
                A_shared[i, j] = A[i, j]
        T.cuda.cta_sync()
        A_local = T.alloc_local([8], "float16")
        A_local[0] = -1.0
                # ldmatrix .x{num}.b16 writes `num` 32-bit registers; A_local
                # is a contiguous fp16[8] buffer, so consecutive register
                # destinations land 2 fp16 elements apart.
        if num == 1:
            T.ptx.ldmatrix(
                trans, num, dtype,
                A_shared.ptr_to([tx % 16, tx // 16 * 8]),
                T.address_of(A_local[0]),
            )
        elif num == 2:
            T.ptx.ldmatrix(
                trans, num, dtype,
                A_shared.ptr_to([tx % 16, tx // 16 * 8]),
                T.address_of(A_local[0]),
                T.address_of(A_local[2]),
            )
        else:
            T.ptx.ldmatrix(
                trans, num, dtype,
                A_shared.ptr_to([tx % 16, tx // 16 * 8]),
                T.address_of(A_local[0]),
                T.address_of(A_local[2]),
                T.address_of(A_local[4]),
                T.address_of(A_local[6]),
            )
        for i in range(8):
            row: T.let = (i // 2) % 2 * 8
            col: T.let = (i // 4) * 8
            B[row + tx // 4, col + tx % 4 * 2 + i % 2] = A_local[i]
        # fmt: on

    src, mod = _get_source(main)
    A_np = np.arange(16 * 16, dtype="float16").reshape((16, 16))
    B_np = np.zeros((16, 16), dtype="float16")
    B_ref = np.zeros((16, 16), dtype="float16")
    if num == 1:
        B_ref[0:8, 0:8] = A_np[0:8, 0:8] if not trans else A_np[0:8, 0:8].T
    elif num == 2:
        B_ref[0:8, 0:8] = A_np[0:8, 0:8] if not trans else A_np[0:8, 0:8].T
        B_ref[8:16, 0:8] = A_np[8:16, 0:8] if not trans else A_np[8:16, 0:8].T
    elif num == 4:
        B_ref[0:8, 0:8] = A_np[0:8, 0:8] if not trans else A_np[0:8, 0:8].T
        B_ref[0:8, 8:16] = A_np[0:8, 8:16] if not trans else A_np[0:8, 8:16].T
        B_ref[8:16, 0:8] = A_np[8:16, 0:8] if not trans else A_np[8:16, 0:8].T
        B_ref[8:16, 8:16] = A_np[8:16, 8:16] if not trans else A_np[8:16, 8:16].T

    def run_and_check():
        dev = tvm.cuda()
        A = tvm.runtime.tensor(A_np, device=dev)
        B = tvm.runtime.tensor(B_np, device=dev)
        mod(A, B)
        np.testing.assert_allclose(B.numpy(), B_ref)

    tvm.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm.testing.main()
