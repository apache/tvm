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
import math

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tirx import Buffer


def _get_source(func: tvm.tirx.PrimFunc) -> tuple[str, tvm.IRModule]:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return src, mod


def _run_tensormap_encode(shape, dtype, encode_args):
    # fmt: off
    @Tx.prim_func
    def main(A_ptr: Tx.handle):
        A = Tx.match_buffer(A_ptr, shape, dtype=dtype, align=32)

        A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
        Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_map, dtype, len(shape), A.data, *encode_args)  # noqa: E501

        with Tx.kernel():
            for blockIdx in Tx.thread_binding(1, thread="blockIdx.x"):
                for threadIdx in Tx.thread_binding(1, thread="threadIdx.x"):
                    with Tx.thread():
                        Tx.evaluate(blockIdx + threadIdx)
    # fmt: on

    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": main})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    A = tvm.runtime.tensor(np.zeros(shape, dtype=dtype), device=tvm.cuda(0))
    mod(A)


@pytest.mark.parametrize("inc", [False, True])
@tvm.testing.requires_cuda_compute_version(9)
def test_ptx_setmaxnreg(inc):
    # fmt: off
    @Tx.prim_func
    def func(A: Tx.Buffer(1)):
        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tid = Tx.thread_id([128])
            with Tx.thread():
                Tx.ptx.setmaxnreg(inc, 32)
    # fmt: on

    src, mod = _get_source(func)
    assert "setmaxnreg" in src
    if inc:
        assert "inc" in src
    else:
        assert "dec" in src


@pytest.mark.parametrize("trans", [False, True])
@tvm.testing.requires_cuda_compute_version(9)
def test_stmatrix_sync_aligned(trans):
    # fmt: off
    @Tx.prim_func
    def func(A: Tx.Buffer((16, 16), "float16")):
        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tx = Tx.thread_id([32])
            with Tx.cta():
                A_smem = Tx.alloc_buffer((16, 16), "float16", scope="shared", align=16)
                with Tx.thread():
                    reg = Tx.alloc_buffer((8,), "float16", scope="local")
                    for i in range(8):
                        reg[i] = tx * 8 + i
                    Tx.ptx.stmatrix(A_smem.ptr_to([tx % 16, tx // 16 * 8]), reg.ptr_to([0]), num=4, trans=trans)  # noqa: E501
                    if tx == 0:
                        for i, j in Tx.grid(16, 16):
                            A[i, j] = A_smem[i, j]
    # fmt: on

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    with target:
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        if not trans:
            assert "stmatrix.sync.aligned.m8n8.x4.shared.b16" in src
        else:
            assert "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16" in src
        A_np = np.zeros((16, 16), dtype="float16")
        A = tvm.runtime.tensor(A_np, device=DEV)
        mod(A)
        A_ref = np.zeros((16, 16), dtype="float16")
        for tx in range(32):
            row = tx // 4
            col = tx % 4 * 2
            if not trans:
                A_ref[row, col] = tx * 8
                A_ref[row, col + 1] = tx * 8 + 1
                A_ref[row + 8, col] = tx * 8 + 2
                A_ref[row + 8, col + 1] = tx * 8 + 3
                A_ref[row, col + 8] = tx * 8 + 4
                A_ref[row, col + 9] = tx * 8 + 5
                A_ref[row + 8, col + 8] = tx * 8 + 6
                A_ref[row + 8, col + 9] = tx * 8 + 7
            else:
                A_ref[col, row] = tx * 8
                A_ref[col + 1, row] = tx * 8 + 1
                A_ref[col + 8, row] = tx * 8 + 2
                A_ref[col + 9, row] = tx * 8 + 3
                A_ref[col, row + 8] = tx * 8 + 4
                A_ref[col + 1, row + 8] = tx * 8 + 5
                A_ref[col + 8, row + 8] = tx * 8 + 6
                A_ref[col + 9, row + 8] = tx * 8 + 7
        np.testing.assert_allclose(A.numpy(), A_ref)


@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("num", [1, 2, 4])
def test_ptx_stmatrix(trans, num):
    # fmt: off
    @Tx.prim_func
    def main(A: Tx.Buffer((16, 16), "float16")):
        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tx = Tx.thread_id([32])
            A_shared = Tx.alloc_shared([16, 16], "float16")
            if Tx.filter(tx, tx == 0):
                with Tx.thread():
                    for i, j in Tx.grid(16, 16):
                        A_shared[i, j] = Tx.float16(0.0)
            Tx.cuda.cta_sync()
            with Tx.thread():
                A_local = Tx.alloc_local([8], "float16")
                for i in range(8):
                    A_local[i] = (i // 2) * 64 + tx * 2 + i % 2
                Tx.ptx.stmatrix(A_shared.ptr_to([tx % 16, tx // 16 * 8]), A_local.ptr_to([0]), num=num, trans=trans)  # noqa: E501
            Tx.cuda.cta_sync()
            if Tx.filter(tx, tx == 0):
                with Tx.thread():
                    for i, j in Tx.grid(16, 16):
                        A[i, j] = A_shared[i, j]
    # fmt: on

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": main})
    with target:
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
    A_np = np.zeros((16, 16), dtype="float16")
    A_ref = np.zeros((16, 16), dtype="float16")
    A_full = np.zeros((16, 16), dtype="float16")
    A_full[0:8, 0:8] = np.arange(8 * 8, dtype="float16").reshape((8, 8))
    A_full[8:16, 0:8] = np.arange(8 * 8, 16 * 8, dtype="float16").reshape((8, 8))
    A_full[0:8, 8:16] = np.arange(16 * 8, 24 * 8, dtype="float16").reshape((8, 8))
    A_full[8:16, 8:16] = np.arange(24 * 8, 32 * 8, dtype="float16").reshape((8, 8))
    A = tvm.runtime.tensor(A_np, device=DEV)

    mod(A)
    print(src)

    if num == 1:
        A_ref[0:8, 0:8] = A_full[0:8, 0:8] if not trans else A_full[0:8, 0:8].T
    elif num == 2:
        A_ref[0:8, 0:8] = A_full[0:8, 0:8] if not trans else A_full[0:8, 0:8].T
        A_ref[8:16, 0:8] = A_full[8:16, 0:8] if not trans else A_full[8:16, 0:8].T
    elif num == 4:
        A_ref[0:8, 0:8] = A_full[0:8, 0:8] if not trans else A_full[0:8, 0:8].T
        A_ref[0:8, 8:16] = A_full[0:8, 8:16] if not trans else A_full[0:8, 8:16].T
        A_ref[8:16, 0:8] = A_full[8:16, 0:8] if not trans else A_full[8:16, 0:8].T
        A_ref[8:16, 8:16] = A_full[8:16, 8:16] if not trans else A_full[8:16, 8:16].T

    np.testing.assert_allclose(A.numpy(), A_ref)


@tvm.testing.requires_cuda_compute_version(9)
def test_bar_arrive():
    # fmt: off
    @Tx.prim_func
    def func(A: Tx.Buffer(1)):
        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tid = Tx.thread_id([128])
            with Tx.thread():
                Tx.ptx.bar.arrive(0, 128)
    # fmt: on

    src, mod = _get_source(func)
    assert "tvm_builtin_ptx_bar_arrive(0, 128)" in src
    assert 'bar.arrive %0, %1;" : : "r"(name_bar_id), "r"(thread_count) : "memory"' in src


@tvm.testing.requires_cuda_compute_version(9)
def test_bar_sync():
    # fmt: off
    @Tx.prim_func
    def func(A: Tx.Buffer(1)):
        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tid = Tx.thread_id([128])
            with Tx.thread():
                Tx.ptx.bar.sync(0, 128)
    # fmt: on

    src, mod = _get_source(func)
    assert "tvm_builtin_ptx_bar_sync(0, 128)" in src
    assert 'bar.sync %0, %1;" : : "r"(name_bar_id), "r"(thread_count) : "memory"' in src


@tvm.testing.requires_cuda_compute_version(9)
def test_fence_mbarrier_init_release_clsuter():
    # fmt: off
    @Tx.prim_func
    def func(A: Tx.Buffer(1)):
        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tid = Tx.thread_id([128])
            with Tx.thread():
                Tx.ptx.fence.mbarrier_init()
    # fmt: on

    src, mod = _get_source(func)
    assert "fence.mbarrier_init.release.cluster" in src


@tvm.testing.requires_cuda_compute_version(9)
def test_ptx_elect_sync():
    # fmt: off
    @Tx.prim_func
    def func(A: Tx.Buffer(1)):
        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tx = Tx.thread_id([128])
            with Tx.thread():
                if (Tx.ptx.elect_sync()):
                    A[tx] = tx
    # fmt: on

    src, mod = _get_source(func)
    print(src)
    assert "elect.sync %%rx|%%px, %2;" in src


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize("sem,scope", [("sc", "cta"), ("acq_rel", "gpu"), ("sc", "sys")])
def test_ptx_fence(sem, scope):
    # fmt: off
    @Tx.prim_func
    def func(A: Tx.Buffer(1)):
        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tid = Tx.thread_id([128])
            with Tx.thread():
                Tx.ptx.fence(sem, scope)
    # fmt: on

    src, mod = _get_source(func)
    assert f"fence.{sem}.{scope};" in src


@tvm.testing.requires_cuda_compute_version(9)
def test_fence_proxy_async():
    # fmt: off
    @Tx.prim_func
    def func(A: Tx.Buffer(1)):
        with Tx.kernel():
            cta_id = Tx.cta_id([1])
            tid = Tx.thread_id([128])
            with Tx.thread():
                Tx.ptx.fence.proxy_async("global")
                Tx.ptx.fence.proxy_async("shared::cta")

    # fmt: on

    src, mod = _get_source(func)
    assert "fence.proxy.async.global" in src
    assert "fence.proxy.async.shared::cta" in src


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize("dtype", ["float16", "float32", "float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize(
    "inputs",
    [
        ((128,), [128, 128, 1, 0, 0, 0, 0]),
        ((16, 16), [16, 16, 16, 16, 16, 1, 1, 0, 0, 0, 0]),
        ((16, 64), [64, 16, 64, 64, 16, 1, 1, 0, 0, 0, 0]),
    ],
)
def test_cp_async_bulk_tensor_global_to_shared_unicast(dtype, inputs):
    import ml_dtypes

    def get_ir(shape, tma_args):
        t_dtype = tvm.DataType(dtype)
        total_bytes = math.prod(shape) * t_dtype.bits // 8
        coord = [0 for _ in shape]
        tma_args_copy = tma_args.copy()
        for i in range(len(shape) - 1):
            tma_args_copy[len(shape) + i] *= t_dtype.bits // 8

        # fmt: off
        @Tx.prim_func
        def main(A_ptr: Tx.handle, B_ptr: Tx.handle):
            A = Tx.match_buffer(A_ptr, shape, dtype=dtype, align=16)
            B = Tx.match_buffer(B_ptr, shape, dtype=dtype, align=16)

            A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
            Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_map, dtype, len(shape), A.data, *tma_args_copy)  # noqa: E501
            B_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
            Tx.call_packed("runtime.cuTensorMapEncodeTiled", B_map, dtype, len(shape), B.data, *tma_args_copy)  # noqa: E501

            with Tx.kernel():
                for blockIdx in Tx.thread_binding(1, thread="blockIdx.x"):
                    for threadIdx in Tx.thread_binding(128, thread="threadIdx.x"):
                        with Tx.thread():
                            bar = Tx.shared_scalar("uint64")
                            phase: Tx.int32
                            A_smem = Tx.alloc_buffer(shape, dtype, scope="shared", align=128)

                            phase = 0
                            if threadIdx == 0:
                                Tx.ptx.mbarrier.init(Tx.address_of(bar), 1)
                                Tx.ptx.fence.proxy_async("shared::cta")
                                Tx.ptx.cp_async.bulk.tensor.g2c(len(shape), A_smem.data, Tx.address_of(bar), Tx.address_of(A_map), 0, 1, "", *coord)  # noqa: E501
                                Tx.ptx.mbarrier.arrive.expect_tx(Tx.address_of(bar), total_bytes)
                            Tx.ptx.mbarrier.try_wait(Tx.address_of(bar), phase)
                            phase = phase ^ 1

                            Tx.cuda.cta_sync()
                            Tx.ptx.fence.proxy_async("shared::cta")

                            if threadIdx == 0:
                                Tx.ptx.cp_async.bulk.tensor.s2g(len(shape), A_smem.access_ptr("r", offset=0), Tx.address_of(B_map), "", *coord)  # noqa: E501
                                Tx.ptx.cp_async.bulk.commit_group()
                                Tx.ptx.cp_async.bulk.wait_group(0)
        # fmt: on

        return main

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    shape, tma_args = inputs
    mod = tvm.IRModule({"main": get_ir(shape, tma_args)})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    assert "const __grid_constant__ CUtensorMap" in src

    A_np = np.random.randn(math.prod(shape))

    def get_np_dtype(dtype):
        if dtype == "float8_e4m3fn":
            return ml_dtypes.float8_e4m3fn
        if dtype == "float8_e5m2":
            return ml_dtypes.float8_e5m2
        return np.dtype(dtype)

    A_np = np.array(A_np).reshape(shape).astype(get_np_dtype(dtype))
    B_np = np.zeros(shape).astype(get_np_dtype(dtype))
    A = tvm.runtime.tensor(A_np, device=DEV)
    B = tvm.runtime.tensor(B_np, device=DEV)
    mod(A, B)
    assert np.allclose(A.numpy().astype("float32"), B.numpy().astype("float32"))


@tvm.testing.requires_cuda_compute_version(9)
@pytest.mark.parametrize(
    ("shape", "dtype", "encode_args", "error_msg"),
    [
        (
            (16, 16),
            "float16",
            [0, 16, 32, 16, 16, 1, 1, 0, 0, 0, 0],
            r"globalDim\[0\] must be non-zero",
        ),
        (
            (16, 16),
            "float16",
            [(1 << 32) + 1, 16, 32, 16, 16, 1, 1, 0, 0, 0, 0],
            r"globalDim\[0\] must be less than or equal to 2\^32",
        ),
        (
            (16, 16),
            "float16",
            [16, 16, 1 << 40, 16, 16, 1, 1, 0, 0, 0, 0],
            r"globalStrides\[0\] must be less than 2\^40",
        ),
        (
            (16, 16),
            "float16",
            [16, 16, 32, 0, 16, 1, 1, 0, 0, 0, 0],
            r"boxDim\[0\] must be non-zero",
        ),
        (
            (16, 16),
            "float16",
            [16, 16, 32, 7, 16, 1, 1, 0, 0, 0, 0],
            r"boxDim\[0\] \* elementSizeInBytes\(tensorDataType\) must be a multiple of 16 bytes",
        ),
        (
            (16, 16),
            "float16",
            [16, 16, 32, 16, 16, 0, 1, 0, 0, 0, 0],
            r"elementStrides\[0\] must be non-zero",
        ),
        (
            (16, 16),
            "float16",
            [16, 16, 32, 16, 16, 9, 1, 0, 0, 0, 0],
            r"elementStrides\[0\] must be less than or equal to 8",
        ),
        (
            (16, 16),
            "float16",
            [16, 16, 32, 16, 16, 1, 1, 2, 0, 0, 0],
            r"tensorRank must be greater than or equal to 3 when interleave is not NONE",
        ),
        (
            (8, 8, 8),
            "float16",
            [8, 8, 8, 16, 128, 8, 8, 8, 1, 1, 1, 2, 0, 0, 0],
            r"globalStrides\[0\] must be a multiple of 32",
        ),
        (
            (16, 16),
            "int32",
            [16, 16, 64, 4, 16, 1, 1, 0, 0, 0, 1],
            (
                r"CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA requires a "
                r"floating-point tensorDataType"
            ),
        ),
    ],
)
def test_tensormap_encode_tiled_runtime_validation(shape, dtype, encode_args, error_msg):
    with pytest.raises(tvm.error.InternalError, match=error_msg):
        _run_tensormap_encode(shape, dtype, encode_args)


@pytest.mark.parametrize("swizzle", [1, 2, 3])
@pytest.mark.parametrize("dtype", ["uint8", "float16", "float32"])
@tvm.testing.requires_cuda_compute_version(9)
def test_cp_async_bulk_tensor_global_to_shared_swizzle(swizzle, dtype):
    def get_ir(swizzle, dtype):
        dtype = tvm.DataType(dtype)
        elem_bytes = dtype.bits // 8

        shape = [16, 64]
        tma_args = [16, 64, 16, 16, 64, 1, 1, 0, 0, 0, 0]  # 8x16B, atom for WGMMA
        shape[0] = shape[0] * (1 << swizzle) // elem_bytes
        tma_args[0] = tma_args[0] * (1 << swizzle) // elem_bytes
        tma_args[2] = tma_args[2] * (1 << swizzle)
        tma_args[3] = tma_args[3] * (1 << swizzle) // elem_bytes

        load_args = tma_args.copy()
        load_args[-3] = swizzle
        store_args = tma_args.copy()

        shape = tuple(shape)
        total_elems = math.prod(shape)
        total_bytes = total_elems * elem_bytes
        coord = [0 for _ in shape]

        # fmt: off
        @Tx.prim_func
        def main(A_ptr: Tx.handle, B_ptr: Tx.handle):
            A = Tx.match_buffer(A_ptr, total_elems, dtype=dtype, align=16)
            B = Tx.match_buffer(B_ptr, total_elems, dtype=dtype, align=16)

            A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
            Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_map, dtype, len(shape), A.data, *load_args)  # noqa: E501
            B_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
            Tx.call_packed("runtime.cuTensorMapEncodeTiled", B_map, dtype, len(shape), B.data, *store_args)  # noqa: E501

            with Tx.kernel():
                for blockIdx in Tx.thread_binding(1, thread="blockIdx.x"):
                    for threadIdx in Tx.thread_binding(128, thread="threadIdx.x"):
                        with Tx.thread():
                            A_smem = Tx.alloc_buffer((total_elems,), dtype, scope="shared", align=128)  # noqa: E501
                            bar = Tx.shared_scalar("uint64")
                            phase: Tx.int32

                            phase = 0
                            if threadIdx == 0:
                                Tx.ptx.mbarrier.init(Tx.address_of(bar), 1)
                                Tx.ptx.fence.proxy_async("shared::cta")
                                Tx.ptx.cp_async.bulk.tensor.g2c(len(shape), A_smem.data, Tx.address_of(bar), Tx.address_of(A_map), 0, 1, "", *coord)  # noqa: E501
                                Tx.ptx.mbarrier.arrive.expect_tx(Tx.address_of(bar), total_bytes)
                                Tx.ptx.mbarrier.try_wait(Tx.address_of(bar), phase)
                            phase = phase ^ 1

                            Tx.cuda.cta_sync()
                            Tx.ptx.fence.proxy_async("shared::cta")

                            if threadIdx == 0:
                                Tx.ptx.cp_async.bulk.tensor.s2g(len(shape), A_smem.access_ptr("r", offset=0), Tx.address_of(B_map), "", *coord)  # noqa: E501
                                Tx.ptx.cp_async.bulk.commit_group()
                                Tx.ptx.cp_async.bulk.wait_group(0)
        # fmt: on

        return main, shape

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    func, shape = get_ir(swizzle, dtype)
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    assert "const __grid_constant__ CUtensorMap" in src

    total_elems = math.prod(shape)
    A_np = [i for i in range(total_elems)]
    A_np = np.array(A_np).astype(dtype)
    B_np = np.zeros((total_elems,)).astype(dtype)
    A = tvm.runtime.tensor(A_np, device=DEV)
    B = tvm.runtime.tensor(B_np, device=DEV)
    mod(A, B)
    dtype = tvm.DataType(dtype)
    layout = Tx.SwizzleLayout(
        per_element=int(math.log2(128 // dtype.bits)), swizzle_len=swizzle, atom_len=3
    )
    B_np = B.numpy()
    B_swizzle = [B_np[int(layout.apply(i)["m"])] for i in range(total_elems)]
    B_swizzle = np.array(B_swizzle).astype(str(dtype))
    assert np.allclose(A.numpy(), B_swizzle)


@pytest.mark.parametrize(
    "inputs",
    [
        ((128,), [128, 128, 1, 0, 0, 0, 0]),
        ((16, 16), [16, 16, 64, 16, 16, 1, 1, 0, 0, 0, 0]),
        ((4, 4, 4), [4, 4, 4, 16, 64, 4, 4, 4, 1, 1, 1, 0, 0, 0, 0]),
        ((4, 4, 4, 4), [4, 4, 4, 4, 16, 64, 256, 4, 4, 4, 4, 1, 1, 1, 1, 0, 0, 0, 0]),
        (
            (4, 2, 2, 2, 2),
            [4, 2, 2, 2, 2, 16, 32, 64, 128, 4, 2, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        ),
    ],
)
@tvm.testing.requires_cuda_compute_version(9)
def test_cp_async_bulk_tensor_global_to_shared_multicast1(inputs):
    # 1 CTA does the copy, and then multicast to all CTAs in the cluster
    def get_ir(shape, tma_args):
        total_bytes = 4 * math.prod(shape)
        coord = [0 for _ in shape]

        # fmt: off
        @Tx.prim_func
        def main(A_ptr: Tx.handle, B_ptr: Tx.handle):
            A = Tx.match_buffer(A_ptr, shape, dtype="float32", align=16)
            B = Tx.match_buffer(B_ptr, shape, dtype="float32", align=16)

            A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
            Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float32", len(shape), A.data, *tma_args)  # noqa: E501
            B_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
            Tx.call_packed("runtime.cuTensorMapEncodeTiled", B_map, "float32", len(shape), B.data, *tma_args)  # noqa: E501

            with Tx.kernel():
                for clusterCtaIdx in Tx.thread_binding(4, thread="clusterCtaIdx.x"):
                    for bx in Tx.thread_binding(4, thread="blockIdx.x"):
                        for tx in Tx.thread_binding(128, thread="threadIdx.x"):
                            with Tx.thread():
                                bar = Tx.shared_scalar("uint64")
                                phase: Tx.int32
                                A_smem = Tx.alloc_buffer(shape[::-1], "float32", scope="shared", align=128)  # noqa: E501

                                phase = 0
                                if tx == 0:
                                    # leader thread in each CTA
                                    Tx.ptx.mbarrier.init(Tx.address_of(bar), 1)
                                    Tx.ptx.fence.proxy_async("shared::cta")
                                    Tx.ptx.mbarrier.arrive.expect_tx(Tx.address_of(bar), total_bytes)  # noqa: E501
                                    if clusterCtaIdx == 0:
                                        # only the first CTA in the cluster does the copy, and then multicast  # noqa: E501
                                        Tx.ptx.cp_async.bulk.tensor.g2c(len(shape), A_smem.data, Tx.address_of(bar), Tx.address_of(A_map), int("1111", 2), 1, "", *coord)  # noqa: E501
                                # wait for the copy to finish
                                Tx.ptx.mbarrier.try_wait(Tx.address_of(bar), phase)
                                phase = phase ^ 1
                                Tx.cuda.cta_sync()
                                Tx.ptx.fence.proxy_async("shared::cta")

                                if bx == 2:
                                    if tx == 0:
                                        Tx.ptx.cp_async.bulk.tensor.s2g(len(shape), A_smem.access_ptr("r", offset=0), Tx.address_of(B_map), "", *coord)  # noqa: E501
                                        Tx.ptx.cp_async.bulk.commit_group()
                                        Tx.ptx.cp_async.bulk.wait_group(0)
        # fmt: on

        return main

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    shape, tma_args = inputs
    mod = tvm.IRModule({"main": get_ir(shape, tma_args)})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    assert "const __grid_constant__ CUtensorMap" in src

    A_np = [i for i in range(math.prod(shape))]
    A_np = np.array(A_np, dtype="float32").reshape(shape)
    B_np = np.zeros(shape, dtype="float32")
    A = tvm.runtime.tensor(A_np, device=DEV)
    B = tvm.runtime.tensor(B_np, device=DEV)
    mod(A, B)


@pytest.mark.parametrize(
    "inputs",
    [
        ((128,), [128, 32, 1, 0, 0, 0, 0]),
        ((16, 16), [16, 16, 64, 16, 4, 1, 1, 0, 0, 0, 0]),
        ((16, 16, 4), [16, 16, 4, 64, 64 * 16, 16, 16, 1, 1, 1, 1, 0, 0, 0, 0]),
    ],
)
@tvm.testing.requires_cuda_compute_version(9)
def test_cp_async_bulk_tensor_global_to_shared_multicast2(inputs):
    # 4 CTAs in the cluster do the copy of separate chunks, and then multicast to all CTAs in the cluster  # noqa: E501
    def get_ir(shape, tma_args):
        assert shape[0] % 4 == 0
        total_bytes = 4 * math.prod(shape)
        coord0 = [0 for _ in shape]
        coord1 = [0 for _ in shape[:-1]] + [shape[-1] // 4]
        coord2 = [0 for _ in shape[:-1]] + [shape[-1] // 2]
        coord3 = [0 for _ in shape[:-1]] + [3 * shape[-1] // 4]

        tma_store_args = tma_args.copy()
        tma_store_args[3 * len(shape) - 2] = shape[-1]

        # fmt: off
        @Tx.prim_func
        def main(A_ptr: Tx.handle, B_ptr: Tx.handle):
            A = Tx.match_buffer(A_ptr, shape, dtype="float32", align=16)
            B = Tx.match_buffer(B_ptr, shape, dtype="float32", align=16)

            A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
            Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float32", len(shape), A.data, *tma_args)  # noqa: E501
            B_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
            Tx.call_packed("runtime.cuTensorMapEncodeTiled", B_map, "float32", len(shape), B.data, *tma_store_args)  # noqa: E501

            with Tx.kernel():
                for clusterCtaIdx in Tx.thread_binding(4, thread="clusterCtaIdx.x"):
                    for bx in Tx.thread_binding(4, thread="blockIdx.x"):
                        for tx in Tx.thread_binding(128, thread="threadIdx.x"):
                            with Tx.thread():
                                bar = Tx.shared_scalar("uint64")
                                phase: Tx.int32
                                A_smem = Tx.alloc_buffer(shape[::-1], "float32", scope="shared", align=128)  # noqa: E501

                                phase = 0
                                if tx == 0:
                                    # leader thread in each CTA
                                    Tx.ptx.mbarrier.init(Tx.address_of(bar), 1)
                                    Tx.ptx.fence.proxy_async("shared::cta")
                                    Tx.ptx.mbarrier.arrive.expect_tx(Tx.address_of(bar), total_bytes)  # noqa: E501
                                    if clusterCtaIdx == 0:
                                        Tx.ptx.cp_async.bulk.tensor.g2c(len(shape), A_smem.access_ptr(Buffer.WRITE, offset=A_smem.elem_offset_of(coord0[::-1])),  # noqa: E501
                                                                       Tx.address_of(bar), Tx.address_of(A_map), int("1111", 2), 1, "", *coord0)  # noqa: E501
                                    if clusterCtaIdx == 1:
                                        Tx.ptx.cp_async.bulk.tensor.g2c(len(shape), A_smem.access_ptr(Buffer.WRITE, offset=A_smem.elem_offset_of(coord1[::-1])),  # noqa: E501
                                                                       Tx.address_of(bar), Tx.address_of(A_map), int("1111", 2), 1, "", *coord1)  # noqa: E501
                                    if clusterCtaIdx == 2:
                                        Tx.ptx.cp_async.bulk.tensor.g2c(len(shape), A_smem.access_ptr(Buffer.WRITE, offset=A_smem.elem_offset_of(coord2[::-1])),  # noqa: E501
                                                                       Tx.address_of(bar), Tx.address_of(A_map), int("1111", 2), 1, "", *coord2)  # noqa: E501
                                    if clusterCtaIdx == 3:
                                        Tx.ptx.cp_async.bulk.tensor.g2c(len(shape), A_smem.access_ptr(Buffer.WRITE, offset=A_smem.elem_offset_of(coord3[::-1])),  # noqa: E501
                                                                       Tx.address_of(bar), Tx.address_of(A_map), int("1111", 2), 1, "", *coord3)  # noqa: E501
                                # wait for the copy to finish
                                Tx.ptx.mbarrier.try_wait(Tx.address_of(bar), phase)
                                phase = phase ^ 1
                                Tx.cuda.cta_sync()

                                if bx == 1:
                                    if tx == 0:
                                        Tx.ptx.cp_async.bulk.tensor.s2g(len(shape), A_smem.access_ptr("r", offset=0), Tx.address_of(B_map), "", *coord0)  # noqa: E501
                                        Tx.ptx.cp_async.bulk.commit_group()
                                        Tx.ptx.cp_async.bulk.wait_group(0)
        # fmt: on

        return main

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    shape, tma_args = inputs
    mod = tvm.IRModule({"main": get_ir(shape, tma_args)})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    assert "const __grid_constant__ CUtensorMap" in src

    A_np = [i for i in range(math.prod(shape))]
    A_np = np.array(A_np, dtype="float32").reshape(shape)
    B_np = np.zeros(shape, dtype="float32")
    A = tvm.runtime.tensor(A_np, device=DEV)
    B = tvm.runtime.tensor(B_np, device=DEV)
    mod(A, B)
    assert np.allclose(A.numpy(), B.numpy())


@pytest.mark.parametrize(
    "inputs",
    [
        ((128,), [128, 128, 1, 0, 0, 0, 0]),
        ((16, 16), [16, 16, 64, 16, 16, 1, 1, 0, 0, 0, 0]),
        ((16, 16, 4), [16, 16, 4, 64, 64 * 16, 16, 16, 4, 1, 1, 1, 0, 0, 0, 0]),
    ],
)
@tvm.testing.requires_cuda_compute_version(9)
def test_cp_async_bulk_tensor_shared_to_global(inputs):
    def get_ir(shape, tma_args):
        assert shape[0] % 4 == 0
        elems = math.prod(shape)
        coord = [0 for _ in shape]

        # fmt: off
        @Tx.prim_func
        def main(A_ptr: Tx.handle):
            A = Tx.match_buffer(A_ptr, shape, dtype="float32", align=16)

            A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
            Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float32", len(shape), A.data, *tma_args)  # noqa: E501

            with Tx.kernel():
                cta_id = Tx.cta_id([1])
                tx = Tx.thread_id([128])

                with Tx.thread():
                    A_smem = Tx.alloc_buffer(elems, "float32", scope="shared", align=128)

                    if tx == 0:
                        for i in Tx.serial(0, elems):
                            A_smem[i] = i
                    Tx.ptx.fence.proxy_async("shared::cta")
                    Tx.cuda.cta_sync()

                    if tx == 0:
                        Tx.ptx.cp_async.bulk.tensor.s2g(len(shape), A_smem.access_ptr("r", offset=0), Tx.address_of(A_map), "", *coord)  # noqa: E501
                        Tx.ptx.cp_async.bulk.commit_group()
                        Tx.ptx.cp_async.bulk.wait_group(0)
        # fmt: on

        return main

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    shape, tma_args = inputs
    mod = tvm.IRModule({"main": get_ir(shape, tma_args)})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    assert "const __grid_constant__ CUtensorMap" in src

    A_np = np.zeros(shape, dtype="float32")
    A = tvm.runtime.tensor(A_np, device=DEV)
    mod(A)

    A_ref = [i for i in range(math.prod(shape))]
    A_ref = np.array(A_ref, dtype="float32").reshape(shape)
    np.testing.assert_allclose(A.numpy(), A_ref)


@tvm.testing.requires_cuda_compute_version(9, exact=True)
def test_wgmma_ss_nt():
    def get_ir(
        shapeA,
        shapeB,
        shapeC,
        A_tma_args,
        B_tma_args,
        in_dtype,
        out_dtype,
        A_encode_args,
        B_encode_args,
    ):
        coordA = [0 for _ in shapeA]
        coordB = [0 for _ in shapeB]
        A_bytes = tvm.DataType(in_dtype).bits // 8 * math.prod(shapeA)
        B_bytes = tvm.DataType(in_dtype).bits // 8 * math.prod(shapeB)

        C_elems = math.prod(shapeC) // 128

        M, K = shapeA if not transA else shapeA[::-1]
        N, _ = shapeB if not transB else shapeB[::-1]

        def get_init_value(dtype):
            if dtype == "float32":
                return Tx.float32(0.0)
            assert False, f"Unsupported dtype {dtype}"

        def get_accum_list(C, C_elems):
            return [C[i] for i in range(C_elems)]

        # fmt: off
        @Tx.prim_func
        def main(A_ptr: Tx.handle, B_ptr: Tx.handle, C_ptr: Tx.handle):
            A = Tx.match_buffer(A_ptr, shapeA, dtype=in_dtype, align=16)
            B = Tx.match_buffer(B_ptr, shapeB, dtype=in_dtype, align=16)
            C = Tx.match_buffer(C_ptr, shapeC, dtype=out_dtype, align=16)

            A_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
            Tx.call_packed("runtime.cuTensorMapEncodeTiled", A_map, in_dtype, len(shapeA), A.data, *A_tma_args)  # noqa: E501
            B_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
            Tx.call_packed("runtime.cuTensorMapEncodeTiled", B_map, in_dtype, len(shapeB), B.data, *B_tma_args)  # noqa: E501

            with Tx.kernel():
                cta_id = Tx.cta_id([1])
                tx = Tx.thread_id([128]) # A warpgroup is 128 threads

                with Tx.thread():
                    A_smem = Tx.alloc_buffer(shapeA, in_dtype, scope="shared", align=1024)
                    B_smem = Tx.alloc_buffer(shapeB, in_dtype, scope="shared", align=1024)
                    bar = Tx.shared_scalar("uint64")
                    phase: Tx.int32

                    descA: Tx.uint64
                    descB: Tx.uint64
                    C_local = Tx.alloc_buffer((C_elems,), out_dtype, scope="local")

                    # init phase and bar
                    phase = 0
                    if tx == 0:
                        Tx.ptx.mbarrier.init(Tx.address_of(bar), 1)
                    Tx.ptx.fence.proxy_async("shared::cta")
                    Tx.cuda.cta_sync()
                    # load A and B to smem
                    if tx == 0:
                        Tx.ptx.cp_async.bulk.tensor.g2c(len(shapeA), A_smem.data, Tx.address_of(bar), Tx.address_of(A_map), 0, 1, "", *coordA)  # noqa: E501
                        Tx.ptx.cp_async.bulk.tensor.g2c(len(shapeB), B_smem.data, Tx.address_of(bar), Tx.address_of(B_map), 0, 1, "", *coordB)  # noqa: E501
                        Tx.ptx.mbarrier.arrive.expect_tx(Tx.address_of(bar), A_bytes + B_bytes)
                    Tx.ptx.mbarrier.try_wait(Tx.address_of(bar), phase)
                    phase = phase ^ 1
                    Tx.cuda.cta_sync()

                    # init C_local
                    for i in Tx.serial(0, C_elems):
                        C_local[i] = Tx.Cast(out_dtype, get_init_value(out_dtype))
                        Tx.ptx.wgmma.noop_barrier(C_local[i])

                    # do wgmma
                    Tx.ptx.wgmma.encode_matrix_descriptor(Tx.address_of(descA), A_smem.data, *A_encode_args)  # noqa: E501, F821
                    Tx.ptx.wgmma.encode_matrix_descriptor(Tx.address_of(descB), B_smem.data, *B_encode_args)  # noqa: E501, F821
                    Tx.ptx.wgmma.fence()
                    Tx.ptx.wgmma.mma_async.ss(descA, descB, *get_accum_list(C_local, C_elems),  # noqa: F821
                                             M=M, N=N, K=K, in_dtype=in_dtype, out_dtype=out_dtype, transA=transA, transB=transB, scaleA=1.0, scaleB=1.0, scaleD=False)  # noqa: E501
                    Tx.ptx.wgmma.commit_group()
                    Tx.ptx.wgmma.wait_group(0)

                    for i in Tx.serial(0, C_elems):
                        Tx.ptx.wgmma.noop_barrier(C_local[i])

                    # store C_local to C
                    for i in Tx.serial(0, C_elems // 4):
                        row = Tx.meta_var((tx % 32) // 4 + (tx // 32) * 16)
                        col = Tx.meta_var(i * 8 + tx % 4 * 2)
                        C[row, col] = C_local[i * 4]
                        C[row, col + 1] = C_local[i * 4 + 1]
                        C[row + 8, col] = C_local[i * 4 + 2]
                        C[row + 8, col + 1] = C_local[i * 4 + 3]
        # fmt: on

        return main

    in_dtype = "float16"
    out_dtype = "float32"
    transA = transB = True
    swizzleA = swizzleB = 3

    t_in_dtype = tvm.DataType(in_dtype)
    elem_bytes = t_in_dtype.bits // 8

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    M = 64
    N = 64
    K = 256 // t_in_dtype.bits
    shapeA = (M, K) if not transA else (K, M)
    shapeB = (N, K) if not transB else (K, N)
    shapeC = (M, N)

    # A tma args
    A_outer, A_inner = shapeA
    A_tma_args = [A_inner, A_outer, A_inner * elem_bytes, A_inner, A_outer, 1, 1, 0, swizzleA, 0, 0]
    # B tma args
    B_outer, B_inner = shapeB
    B_tma_args = [B_inner, B_outer, B_inner * elem_bytes, B_inner, B_outer, 1, 1, 0, swizzleB, 0, 0]
    # A encode args
    A_encode_args = [1, 64, swizzleA]
    B_encode_args = [1, 64, swizzleB]

    func = get_ir(
        shapeA,
        shapeB,
        shapeC,
        A_tma_args,
        B_tma_args,
        in_dtype,
        out_dtype,
        A_encode_args,
        B_encode_args,
    )
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    np.random.seed(0)
    A_np = np.random.randn(*shapeA).astype(in_dtype)
    B_np = np.random.randn(*shapeB).astype(in_dtype)
    C_np = np.zeros(shapeC).astype(out_dtype)

    A_tvm = tvm.runtime.tensor(A_np, device=DEV)
    B_tvm = tvm.runtime.tensor(B_np, device=DEV)
    C_tvm = tvm.runtime.tensor(C_np, device=DEV)
    mod(A_tvm, B_tvm, C_tvm)

    C_ref = np.dot(A_np.T, B_np).astype(out_dtype)
    tvm.testing.assert_allclose(C_tvm.numpy(), C_ref, rtol=1e-3, atol=1e-3)


@tvm.testing.requires_cuda_compute_version(9, exact=True)
def test_wgmma_rs_nt():
    def get_ir(
        shapeA, shapeB, shapeC, B_tma_args, in_dtype, in_dtype_bits, out_dtype, B_encode_args
    ):
        coordB = [0 for _ in shapeB]
        B_bytes = tvm.DataType(in_dtype).bits // 8 * math.prod(shapeB)

        A_elems = math.prod(shapeA) // 128
        C_elems = math.prod(shapeC) // 128

        M, K = shapeA if not transA else shapeA[::-1]
        N, _ = shapeB if not transB else shapeB[::-1]

        def get_init_value(dtype):
            if dtype == "float32":
                return Tx.float32(0.0)
            assert False, f"Unsupported dtype {dtype}"

        def get_A_list(A_local, A_elems):
            return [A_local[i] for i in range(A_elems)]

        def get_accum_list(C, C_elems):
            return [C[i] for i in range(C_elems)]

        # fmt: off
        @Tx.prim_func
        def main(A_ptr: Tx.handle, B_ptr: Tx.handle, C_ptr: Tx.handle):
            A = Tx.match_buffer(A_ptr, shapeA, dtype=in_dtype, align=16)
            B = Tx.match_buffer(B_ptr, shapeB, dtype=in_dtype, align=16)
            C = Tx.match_buffer(C_ptr, shapeC, dtype=out_dtype, align=16)

            B_map: Tx.let[Tx.handle("tensormap")] = Tx.tvm_stack_alloca("tensormap", 1)
            Tx.call_packed("runtime.cuTensorMapEncodeTiled", B_map, in_dtype, len(shapeB), B.data, *B_tma_args)  # noqa: E501

            with Tx.kernel():
                cta_id = Tx.cta_id([1])
                tx = Tx.thread_id([128]) # A warpgroup is 128 threads

                with Tx.thread():
                    B_smem = Tx.alloc_buffer(shapeB, in_dtype, scope="shared", align=1024)
                    # bar = Tx.alloc_buffer((1,), "uint64", scope="shared", align=8)
                    bar = Tx.shared_scalar("uint64")

                    # descB = Tx.alloc_buffer((1,), "uint64", scope="local")
                    descB: Tx.uint64
                    A_local = Tx.alloc_buffer((A_elems,), in_dtype, scope="local")
                    C_local = Tx.alloc_buffer((C_elems,), out_dtype, scope="local")

                    A_elems_b32 = Tx.meta_var(A_elems // (32 // in_dtype_bits))
                    A_local_b32 = Tx.decl_buffer((A_elems_b32,), "uint32", data=A_local.data)

                    # load A to regs
                    for i in Tx.serial(0, A_elems // 4):
                        row = Tx.meta_var((tx % 32) // 4 + (tx // 32) * 16)
                        col = Tx.meta_var(i * 8 + tx % 4 * 2)
                        A_local[i * 4] = A[row, col]
                        A_local[i * 4 + 1] = A[row, col + 1]
                        A_local[i * 4 + 2] = A[row + 8, col]
                        A_local[i * 4 + 3] = A[row + 8, col + 1]
                    # init bar, and make sure it's visible to all threads and async proxy
                    if tx == 0:
                        Tx.ptx.mbarrier.init(Tx.address_of(bar), 1)
                    Tx.ptx.fence.proxy_async("shared::cta")
                    Tx.cuda.cta_sync()
                    # load B to smem
                    if tx == 0:
                        Tx.ptx.cp_async.bulk.tensor.g2c(len(shapeB), B_smem.data, Tx.address_of(bar), Tx.address_of(B_map), 0, 1, "", *coordB)  # noqa: E501
                        Tx.ptx.mbarrier.arrive.expect_tx(Tx.address_of(bar), B_bytes)
                    Tx.ptx.mbarrier.try_wait(Tx.address_of(bar), 0)
                    Tx.cuda.cta_sync()

                    # init C_local
                    for i in Tx.serial(0, C_elems):
                        C_local[i] = Tx.Cast(out_dtype, get_init_value(out_dtype))

                    # fence A_local and C_local
                    for i in Tx.serial(0, A_elems_b32):
                        Tx.ptx.wgmma.noop_barrier(A_local_b32[i])
                    for i in Tx.serial(0, C_elems):
                        Tx.ptx.wgmma.noop_barrier(C_local[i])
                    # do wgmma
                    Tx.ptx.wgmma.encode_matrix_descriptor(Tx.address_of(descB), B_smem.data, *B_encode_args)  # noqa: E501, F821
                    Tx.ptx.wgmma.fence()
                    Tx.ptx.wgmma.mma_async.rs(descB, *(get_A_list(A_local_b32, A_elems_b32) + get_accum_list(C_local, C_elems)),  # noqa: E501, F821
                                             M=M, N=N, K=K, in_dtype=in_dtype, out_dtype=out_dtype, transA=transA, transB=transB, scaleA=1.0, scaleB=1.0, scaleD=False)  # noqa: E501
                    Tx.ptx.wgmma.commit_group()
                    Tx.ptx.wgmma.wait_group(0)

                    # fence A_local
                    for i in Tx.serial(0, A_elems_b32):
                        Tx.ptx.wgmma.noop_barrier(A_local_b32[i])
                    # fence C_local
                    for i in Tx.serial(0, C_elems):
                        Tx.ptx.wgmma.noop_barrier(C_local[i])

                    # store C_local to C
                    for i in Tx.serial(0, C_elems // 4):
                        row = Tx.meta_var((tx % 32) // 4 + (tx // 32) * 16)
                        col = Tx.meta_var(i * 8 + tx % 4 * 2)
                        C[row, col] = C_local[i * 4]
                        C[row, col + 1] = C_local[i * 4 + 1]
                        C[row + 8, col] = C_local[i * 4 + 2]
                        C[row + 8, col + 1] = C_local[i * 4 + 3]
        # fmt: on

        return main

    in_dtype = "float16"
    in_dtype_bits = 16
    out_dtype = "float32"
    transA = False
    transB = True
    swizzleB = 3

    t_in_dtype = tvm.DataType(in_dtype)
    elem_bytes = t_in_dtype.bits // 8

    DEV = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    M = 64
    N = 64
    K = 256 // t_in_dtype.bits
    shapeA = (M, K) if not transA else (K, M)
    shapeB = (N, K) if not transB else (K, N)
    shapeC = (M, N)

    # B tma args
    B_outer, B_inner = shapeB
    B_tma_args = [B_inner, B_outer, B_inner * elem_bytes, B_inner, B_outer, 1, 1, 0, swizzleB, 0, 0]
    # B encode args
    B_encode_args = [1, 64, swizzleB]

    func = get_ir(
        shapeA, shapeB, shapeC, B_tma_args, in_dtype, in_dtype_bits, out_dtype, B_encode_args
    )
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    np.random.seed(0)
    A_np = np.random.randn(*shapeA).astype(in_dtype)
    B_np = np.random.randn(*shapeB).astype(in_dtype)
    C_np = np.zeros(shapeC).astype(out_dtype)

    A_tvm = tvm.runtime.tensor(A_np, device=DEV)
    B_tvm = tvm.runtime.tensor(B_np, device=DEV)
    C_tvm = tvm.runtime.tensor(C_np, device=DEV)
    mod(A_tvm, B_tvm, C_tvm)

    np.printoptions(threshold=np.inf)
    np.printoptions(linewidth=np.inf)
    np.printoptions(precision=2)

    C_ref = np.dot(A_np, B_np).astype(out_dtype)
    tvm.testing.assert_allclose(C_tvm.numpy(), C_ref, rtol=1e-3, atol=1e-3)


@tvm.testing.requires_cuda_compute_version(9)
def test_ptx_map_shared_rank():
    @Tx.prim_func
    def func(A: Tx.Buffer(1)):
        with Tx.kernel():
            cbx = Tx.cta_id_in_cluster([2])
            cta_id = Tx.cta_id([2])
            tx = Tx.thread_id([128])
            with Tx.cta():
                A_smem = Tx.alloc_buffer([1], "uint32", scope="shared")
                if Tx.filter(tx, cbx == 0 and tx == 0):
                    with Tx.thread():
                        Tx.ptx.map_shared_rank(A_smem.data, cbx)

    src, mod = _get_source(func)
    print(src)
    assert "tvm_builtin_ptx_mapa_u64(A_smem" in src


if __name__ == "__main__":
    tvm.testing.main()
