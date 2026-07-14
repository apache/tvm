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
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx import Buffer


def _get_source(func: tvm.tirx.PrimFunc) -> tuple[str, tvm.IRModule]:
    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    return src, mod


def _run_tensormap_encode(shape, dtype, encode_args):
    # fmt: off
    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, shape, dtype=dtype, align=32)

        A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
        T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, dtype, len(shape), A.data, *encode_args)  # noqa: E501

        T.device_entry()
        for blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for threadIdx in T.thread_binding(1, thread="threadIdx.x"):
                T.evaluate(blockIdx + threadIdx)
        # fmt: on

    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": main})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    def run_and_check():
        A = tvm.runtime.tensor(np.zeros(shape, dtype=dtype), device=tvm.cuda(0))
        mod(A)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize("inc", [False, True])
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_ptx_setmaxnreg(inc):
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer(1)):
        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([128])
        T.ptx.setmaxnreg(inc, 32)
        # fmt: on

    src, mod = _get_source(func)
    assert "setmaxnreg" in src
    if inc:
        assert "inc" in src
    else:
        assert "dec" in src


@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_stmatrix_sync_aligned(trans):
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer((16, 16), "float16")):
        T.device_entry()
        cta_id = T.cta_id([1])
        tx = T.thread_id([32])
        A_smem = T.alloc_buffer((16, 16), "float16", scope="shared", align=16)
        reg = T.alloc_buffer((8,), "float16", scope="local")
        for i in range(8):
            reg[i] = tx * 8 + i
        T.ptx.stmatrix(
            trans, 4, ".b16",
            A_smem.ptr_to([tx % 16, tx // 16 * 8]),
            reg.ptr_to([0]), reg.ptr_to([2]), reg.ptr_to([4]), reg.ptr_to([6]),
        )
        if tx == 0:
            for i, j in T.grid(16, 16):
                A[i, j] = A_smem[i, j]
        # fmt: on

    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": func})
    with target:
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        if not trans:
            assert "stmatrix.sync.aligned.m8n8.x4.shared.b16" in src
        else:
            assert "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16" in src
    def run_and_check():
        dev = tvm.cuda(0)
        A_np = np.zeros((16, 16), dtype="float16")
        A = tvm.runtime.tensor(A_np, device=dev)
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

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("num", [1, 2, 4])
def test_ptx_stmatrix(trans, num):
    # fmt: off
    @T.prim_func
    def main(A: T.Buffer((16, 16), "float16")):
        T.device_entry()
        cta_id = T.cta_id([1])
        tx = T.thread_id([32])
        A_shared = T.alloc_shared([16, 16], "float16")
        if tx == 0:
            for i, j in T.grid(16, 16):
                A_shared[i, j] = T.float16(0.0)
        T.cuda.cta_sync()
        A_local = T.alloc_local([8], "float16")
        for i in range(8):
            A_local[i] = (i // 2) * 64 + tx * 2 + i % 2
        T.ptx.stmatrix(
            trans, num, ".b16",
            A_shared.ptr_to([tx % 16, tx // 16 * 8]),
            *[A_local.ptr_to([i * 2]) for i in range(num)],
        )
        T.cuda.cta_sync()
        if tx == 0:
            for i, j in T.grid(16, 16):
                A[i, j] = A_shared[i, j]
        # fmt: on

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

    def run_and_check():
        A = tvm.runtime.tensor(A_np, device=tvm.cuda(0))
        mod(A)
        np.testing.assert_allclose(A.numpy(), A_ref)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize("trans", [False, True])
@pytest.mark.parametrize("num", [1, 2, 4])
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_ptx_stmatrix_noncontiguous(trans, num):
    """Symmetric stmatrix API: ``num`` independent src handles.

    Spaces fragments by 4 fp16 (vs the natural 2 contiguous) so per-src
    pointers are non-contiguous — exercises what the old single-``local_ptr``
    API couldn't express.
    """
    STRIDE = 4  # 2 fp16 data + 2 fp16 gap per fragment
    LOCAL_SIZE = STRIDE * num

    # fmt: off
    @T.prim_func
    def main(A: T.Buffer((16, 16), "float16")):
        T.device_entry()
        cta_id = T.cta_id([1])
        tx = T.thread_id([32])
        A_shared = T.alloc_shared([16, 16], "float16")
        if tx == 0:
            for i, j in T.grid(16, 16):
                A_shared[i, j] = T.float16(0.0)
        T.cuda.cta_sync()
        A_local = T.alloc_local([LOCAL_SIZE], "float16")
        for i in range(num):
            A_local[i * STRIDE + 0] = T.float16(i * 64 + tx * 2 + 0)
            A_local[i * STRIDE + 1] = T.float16(i * 64 + tx * 2 + 1)
        T.ptx.stmatrix(
            trans, num, ".b16",
            A_shared.ptr_to([tx % 16, tx // 16 * 8]),
            *[A_local.ptr_to([i * STRIDE]) for i in range(num)],
        )
        T.cuda.cta_sync()
        if tx == 0:
            for i, j in T.grid(16, 16):
                A[i, j] = A_shared[i, j]
    # fmt: on

    target = tvm.target.Target("cuda")
    mod = tvm.IRModule({"main": main})
    with target:
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
        src = mod.mod.imports[0].inspect_source()
        trans_inst = ".trans" if trans else ""
        assert f"stmatrix.sync.aligned.m8n8.x{num}{trans_inst}.shared.b16" in src
        # num distinct src register loads in the helper body.
        for i in range(num):
            assert f"*(uint32_t*)src{i}" in src

    A_np = np.zeros((16, 16), dtype="float16")
    A_ref = np.zeros((16, 16), dtype="float16")
    A_full = np.zeros((16, 16), dtype="float16")
    A_full[0:8, 0:8] = np.arange(8 * 8, dtype="float16").reshape((8, 8))
    A_full[8:16, 0:8] = np.arange(8 * 8, 16 * 8, dtype="float16").reshape((8, 8))
    A_full[0:8, 8:16] = np.arange(16 * 8, 24 * 8, dtype="float16").reshape((8, 8))
    A_full[8:16, 8:16] = np.arange(24 * 8, 32 * 8, dtype="float16").reshape((8, 8))
    if num >= 1:
        A_ref[0:8, 0:8] = A_full[0:8, 0:8] if not trans else A_full[0:8, 0:8].T
    if num >= 2:
        A_ref[8:16, 0:8] = A_full[8:16, 0:8] if not trans else A_full[8:16, 0:8].T
    if num >= 4:
        A_ref[0:8, 8:16] = A_full[0:8, 8:16] if not trans else A_full[0:8, 8:16].T
        A_ref[8:16, 8:16] = A_full[8:16, 8:16] if not trans else A_full[8:16, 8:16].T

    def run_and_check():
        A = tvm.runtime.tensor(A_np, device=tvm.cuda(0))
        mod(A)
        np.testing.assert_allclose(A.numpy(), A_ref)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_bar_arrive():
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer(1)):
        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([128])
        T.ptx.bar.arrive(0, 128)
        # fmt: on

    src, mod = _get_source(func)
    assert "tvm_builtin_ptx_bar_arrive(0, 128)" in src
    assert 'bar.arrive %0, %1;" : : "r"(name_bar_id), "r"(thread_count) : "memory"' in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_bar_sync():
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer(1)):
        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([128])
        T.ptx.bar.sync(0, 128)
        # fmt: on

    src, mod = _get_source(func)
    assert "tvm_builtin_ptx_bar_sync(0, 128)" in src
    assert 'bar.sync %0, %1;" : : "r"(name_bar_id), "r"(thread_count) : "memory"' in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_fence_mbarrier_init_release_clsuter():
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer(1)):
        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([128])
        T.ptx.fence.mbarrier_init()
        # fmt: on

    src, mod = _get_source(func)
    assert "fence.mbarrier_init.release.cluster" in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_ptx_elect_sync():
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer(1)):
        T.device_entry()
        cta_id = T.cta_id([1])
        tx = T.thread_id([128])
        if (T.ptx.elect_sync()):
            A[tx] = tx
        # fmt: on

    src, mod = _get_source(func)
    print(src)
    assert "elect.sync %%rx|%%px, %2;" in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize("sem,scope", [("sc", "cta"), ("acq_rel", "gpu"), ("sc", "sys")])
def test_ptx_fence(sem, scope):
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer(1)):
        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([128])
        T.ptx.fence(sem, scope)
        # fmt: on

    src, mod = _get_source(func)
    assert f"fence.{sem}.{scope};" in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_fence_proxy_async():
    # fmt: off
    @T.prim_func
    def func(A: T.Buffer(1)):
        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([128])
        T.ptx.fence.proxy_async("global")
        T.ptx.fence.proxy_async("shared::cta")

        # fmt: on

    src, mod = _get_source(func)
    assert "fence.proxy.async.global" in src
    assert "fence.proxy.async.shared::cta" in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
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
        @T.prim_func
        def main(A_ptr: T.handle, B_ptr: T.handle):
            A = T.match_buffer(A_ptr, shape, dtype=dtype, align=16)
            B = T.match_buffer(B_ptr, shape, dtype=dtype, align=16)

            A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, dtype, len(shape), A.data, *tma_args_copy)  # noqa: E501
            B_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, dtype, len(shape), B.data, *tma_args_copy)  # noqa: E501

            T.device_entry()
            for blockIdx in T.thread_binding(1, thread="blockIdx.x"):
                for threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                    bar = T.shared_scalar("uint64")
                    phase: T.int32
                    A_smem = T.alloc_buffer(shape, dtype, scope="shared", align=128)

                    phase = 0
                    if threadIdx == 0:
                        T.ptx.mbarrier.init(T.address_of(bar), 1)
                        T.ptx.fence.proxy_async("shared::cta")
                        T.ptx.cp_async.bulk.tensor.g2c(len(shape), A_smem.data, T.address_of(bar), T.address_of(A_map), 0, 1, "", *coord)  # noqa: E501
                        T.ptx.mbarrier.arrive.expect_tx(T.address_of(bar), total_bytes)
                    T.ptx.mbarrier.try_wait(T.address_of(bar), phase)
                    phase = phase ^ 1

                    T.cuda.cta_sync()
                    T.ptx.fence.proxy_async("shared::cta")

                    if threadIdx == 0:
                        T.ptx.cp_async.bulk.tensor.s2g(len(shape), A_smem.access_ptr("r", offset=0), T.address_of(B_map), "", *coord)  # noqa: E501
                        T.ptx.cp_async.bulk.commit_group()
                        T.ptx.cp_async.bulk.wait_group(0)
            # fmt: on

        return main

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

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, device=dev)
        B = tvm.runtime.tensor(B_np, device=dev)
        mod(A, B)
        assert np.allclose(A.numpy().astype("float32"), B.numpy().astype("float32"))

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
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
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
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
        @T.prim_func
        def main(A_ptr: T.handle, B_ptr: T.handle):
            A = T.match_buffer(A_ptr, total_elems, dtype=dtype, align=16)
            B = T.match_buffer(B_ptr, total_elems, dtype=dtype, align=16)

            A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, dtype, len(shape), A.data, *load_args)  # noqa: E501
            B_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, dtype, len(shape), B.data, *store_args)  # noqa: E501

            T.device_entry()
            for blockIdx in T.thread_binding(1, thread="blockIdx.x"):
                for threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                    A_smem = T.alloc_buffer((total_elems,), dtype, scope="shared", align=128)
                    bar = T.shared_scalar("uint64")
                    phase: T.int32

                    phase = 0
                    if threadIdx == 0:
                        T.ptx.mbarrier.init(T.address_of(bar), 1)
                        T.ptx.fence.proxy_async("shared::cta")
                        T.ptx.cp_async.bulk.tensor.g2c(len(shape), A_smem.data, T.address_of(bar), T.address_of(A_map), 0, 1, "", *coord)  # noqa: E501
                        T.ptx.mbarrier.arrive.expect_tx(T.address_of(bar), total_bytes)
                        T.ptx.mbarrier.try_wait(T.address_of(bar), phase)
                    phase = phase ^ 1

                    T.cuda.cta_sync()
                    T.ptx.fence.proxy_async("shared::cta")

                    if threadIdx == 0:
                        T.ptx.cp_async.bulk.tensor.s2g(len(shape), A_smem.access_ptr("r", offset=0), T.address_of(B_map), "", *coord)  # noqa: E501
                        T.ptx.cp_async.bulk.commit_group()
                        T.ptx.cp_async.bulk.wait_group(0)
            # fmt: on

        return main, shape

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
    dtype = tvm.DataType(dtype)
    layout = T.SwizzleLayout(
        per_element=int(math.log2(128 // dtype.bits)), swizzle_len=swizzle, atom_len=3
    )

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, device=dev)
        B = tvm.runtime.tensor(B_np, device=dev)
        mod(A, B)
        B_result = B.numpy()
        B_swizzle = [B_result[int(layout.apply(i)["m"])] for i in range(total_elems)]
        B_swizzle = np.array(B_swizzle).astype(str(dtype))
        assert np.allclose(A.numpy(), B_swizzle)

    tvm.testing.run_with_gpu_lock(run_and_check)


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
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_cp_async_bulk_tensor_global_to_shared_multicast1(inputs):
    # 1 CTA does the copy, and then multicast to all CTAs in the cluster
    def get_ir(shape, tma_args):
        total_bytes = 4 * math.prod(shape)
        coord = [0 for _ in shape]

        # fmt: off
        @T.prim_func
        def main(A_ptr: T.handle, B_ptr: T.handle):
            A = T.match_buffer(A_ptr, shape, dtype="float32", align=16)
            B = T.match_buffer(B_ptr, shape, dtype="float32", align=16)

            A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float32", len(shape), A.data, *tma_args)  # noqa: E501
            B_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, "float32", len(shape), B.data, *tma_args)  # noqa: E501

            T.device_entry()
            for clusterCtaIdx in T.thread_binding(4, thread="clusterCtaIdx.x"):
                for bx in T.thread_binding(4, thread="blockIdx.x"):
                    for tx in T.thread_binding(128, thread="threadIdx.x"):
                        bar = T.shared_scalar("uint64")
                        phase: T.int32
                        A_smem = T.alloc_buffer(shape[::-1], "float32", scope="shared", align=128)

                        phase = 0
                        if tx == 0:
                                    # leader thread in each CTA
                            T.ptx.mbarrier.init(T.address_of(bar), 1)
                            T.ptx.fence.proxy_async("shared::cta")
                            T.ptx.mbarrier.arrive.expect_tx(T.address_of(bar), total_bytes)
                            if clusterCtaIdx == 0:
                                        # only the first CTA in the cluster does the copy, and then multicast  # noqa: E501
                                T.ptx.cp_async.bulk.tensor.g2c(len(shape), A_smem.data, T.address_of(bar), T.address_of(A_map), int("1111", 2), 1, "", *coord)  # noqa: E501
                                # wait for the copy to finish
                        T.ptx.mbarrier.try_wait(T.address_of(bar), phase)
                        phase = phase ^ 1
                        T.cuda.cta_sync()
                        T.ptx.fence.proxy_async("shared::cta")

                        if bx == 2:
                            if tx == 0:
                                T.ptx.cp_async.bulk.tensor.s2g(len(shape), A_smem.access_ptr("r", offset=0), T.address_of(B_map), "", *coord)  # noqa: E501
                                T.ptx.cp_async.bulk.commit_group()
                                T.ptx.cp_async.bulk.wait_group(0)
            # fmt: on

        return main

    target = tvm.target.Target("cuda")
    shape, tma_args = inputs
    mod = tvm.IRModule({"main": get_ir(shape, tma_args)})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    assert "const __grid_constant__ CUtensorMap" in src

    A_np = [i for i in range(math.prod(shape))]
    A_np = np.array(A_np, dtype="float32").reshape(shape)
    B_np = np.zeros(shape, dtype="float32")

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, device=dev)
        B = tvm.runtime.tensor(B_np, device=dev)
        mod(A, B)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize(
    "inputs",
    [
        ((128,), [128, 32, 1, 0, 0, 0, 0]),
        ((16, 16), [16, 16, 64, 16, 4, 1, 1, 0, 0, 0, 0]),
        ((16, 16, 4), [16, 16, 4, 64, 64 * 16, 16, 16, 1, 1, 1, 1, 0, 0, 0, 0]),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
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
        @T.prim_func
        def main(A_ptr: T.handle, B_ptr: T.handle):
            A = T.match_buffer(A_ptr, shape, dtype="float32", align=16)
            B = T.match_buffer(B_ptr, shape, dtype="float32", align=16)

            A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float32", len(shape), A.data, *tma_args)  # noqa: E501
            B_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, "float32", len(shape), B.data, *tma_store_args)  # noqa: E501

            T.device_entry()
            for clusterCtaIdx in T.thread_binding(4, thread="clusterCtaIdx.x"):
                for bx in T.thread_binding(4, thread="blockIdx.x"):
                    for tx in T.thread_binding(128, thread="threadIdx.x"):
                        bar = T.shared_scalar("uint64")
                        phase: T.int32
                        A_smem = T.alloc_buffer(shape[::-1], "float32", scope="shared", align=128)

                        phase = 0
                        if tx == 0:
                                    # leader thread in each CTA
                            T.ptx.mbarrier.init(T.address_of(bar), 1)
                            T.ptx.fence.proxy_async("shared::cta")
                            T.ptx.mbarrier.arrive.expect_tx(T.address_of(bar), total_bytes)
                            if clusterCtaIdx == 0:
                                T.ptx.cp_async.bulk.tensor.g2c(len(shape), A_smem.access_ptr(Buffer.WRITE, offset=A_smem.elem_offset_of(coord0[::-1])),  # noqa: E501
                                                               T.address_of(bar), T.address_of(A_map), int("1111", 2), 1, "", *coord0)  # noqa: E501
                            if clusterCtaIdx == 1:
                                T.ptx.cp_async.bulk.tensor.g2c(len(shape), A_smem.access_ptr(Buffer.WRITE, offset=A_smem.elem_offset_of(coord1[::-1])),  # noqa: E501
                                                               T.address_of(bar), T.address_of(A_map), int("1111", 2), 1, "", *coord1)  # noqa: E501
                            if clusterCtaIdx == 2:
                                T.ptx.cp_async.bulk.tensor.g2c(len(shape), A_smem.access_ptr(Buffer.WRITE, offset=A_smem.elem_offset_of(coord2[::-1])),  # noqa: E501
                                                               T.address_of(bar), T.address_of(A_map), int("1111", 2), 1, "", *coord2)  # noqa: E501
                            if clusterCtaIdx == 3:
                                T.ptx.cp_async.bulk.tensor.g2c(len(shape), A_smem.access_ptr(Buffer.WRITE, offset=A_smem.elem_offset_of(coord3[::-1])),  # noqa: E501
                                                               T.address_of(bar), T.address_of(A_map), int("1111", 2), 1, "", *coord3)  # noqa: E501
                                # wait for the copy to finish
                        T.ptx.mbarrier.try_wait(T.address_of(bar), phase)
                        phase = phase ^ 1
                        T.cuda.cta_sync()

                        if bx == 1:
                            if tx == 0:
                                T.ptx.cp_async.bulk.tensor.s2g(len(shape), A_smem.access_ptr("r", offset=0), T.address_of(B_map), "", *coord0)  # noqa: E501
                                T.ptx.cp_async.bulk.commit_group()
                                T.ptx.cp_async.bulk.wait_group(0)
            # fmt: on

        return main

    target = tvm.target.Target("cuda")
    shape, tma_args = inputs
    mod = tvm.IRModule({"main": get_ir(shape, tma_args)})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    assert "const __grid_constant__ CUtensorMap" in src

    A_np = [i for i in range(math.prod(shape))]
    A_np = np.array(A_np, dtype="float32").reshape(shape)
    B_np = np.zeros(shape, dtype="float32")

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, device=dev)
        B = tvm.runtime.tensor(B_np, device=dev)
        mod(A, B)
        assert np.allclose(A.numpy(), B.numpy())

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.parametrize(
    "inputs",
    [
        ((128,), [128, 128, 1, 0, 0, 0, 0]),
        ((16, 16), [16, 16, 64, 16, 16, 1, 1, 0, 0, 0, 0]),
        ((16, 16, 4), [16, 16, 4, 64, 64 * 16, 16, 16, 4, 1, 1, 1, 0, 0, 0, 0]),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_cp_async_bulk_tensor_shared_to_global(inputs):
    def get_ir(shape, tma_args):
        assert shape[0] % 4 == 0
        elems = math.prod(shape)
        coord = [0 for _ in shape]

        # fmt: off
        @T.prim_func
        def main(A_ptr: T.handle):
            A = T.match_buffer(A_ptr, shape, dtype="float32", align=16)

            A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, "float32", len(shape), A.data, *tma_args)  # noqa: E501

            T.device_entry()
            cta_id = T.cta_id([1])
            tx = T.thread_id([128])

            A_smem = T.alloc_buffer(elems, "float32", scope="shared", align=128)

            if tx == 0:
                for i in T.serial(0, elems):
                    A_smem[i] = i
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.cta_sync()

            if tx == 0:
                T.ptx.cp_async.bulk.tensor.s2g(len(shape), A_smem.access_ptr("r", offset=0), T.address_of(A_map), "", *coord)  # noqa: E501
                T.ptx.cp_async.bulk.commit_group()
                T.ptx.cp_async.bulk.wait_group(0)
            # fmt: on

        return main

    target = tvm.target.Target("cuda")
    shape, tma_args = inputs
    mod = tvm.IRModule({"main": get_ir(shape, tma_args)})
    mod = tvm.compile(mod, target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    assert "const __grid_constant__ CUtensorMap" in src

    A_np = np.zeros(shape, dtype="float32")
    A_ref = [i for i in range(math.prod(shape))]
    A_ref = np.array(A_ref, dtype="float32").reshape(shape)

    def run_and_check():
        A = tvm.runtime.tensor(A_np, device=tvm.cuda(0))
        mod(A)
        np.testing.assert_allclose(A.numpy(), A_ref)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9, exact=True), reason="need cuda compute == 9.0")
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
                return T.float32(0.0)
            assert False, f"Unsupported dtype {dtype}"

        def get_accum_list(C, C_elems):
            return [C[i] for i in range(C_elems)]

        # fmt: off
        @T.prim_func
        def main(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle):
            A = T.match_buffer(A_ptr, shapeA, dtype=in_dtype, align=16)
            B = T.match_buffer(B_ptr, shapeB, dtype=in_dtype, align=16)
            C = T.match_buffer(C_ptr, shapeC, dtype=out_dtype, align=16)

            A_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", A_map, in_dtype, len(shapeA), A.data, *A_tma_args)  # noqa: E501
            B_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, in_dtype, len(shapeB), B.data, *B_tma_args)  # noqa: E501

            T.device_entry()
            cta_id = T.cta_id([1])
            tx = T.thread_id([128]) # A warpgroup is 128 threads

            A_smem = T.alloc_buffer(shapeA, in_dtype, scope="shared", align=1024)
            B_smem = T.alloc_buffer(shapeB, in_dtype, scope="shared", align=1024)
            bar = T.shared_scalar("uint64")
            phase: T.int32

            descA: T.uint64
            descB: T.uint64
            C_local = T.alloc_buffer((C_elems,), out_dtype, scope="local")

                    # init phase and bar
            phase = 0
            if tx == 0:
                T.ptx.mbarrier.init(T.address_of(bar), 1)
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.cta_sync()
                    # load A and B to smem
            if tx == 0:
                T.ptx.cp_async.bulk.tensor.g2c(len(shapeA), A_smem.data, T.address_of(bar), T.address_of(A_map), 0, 1, "", *coordA)  # noqa: E501
                T.ptx.cp_async.bulk.tensor.g2c(len(shapeB), B_smem.data, T.address_of(bar), T.address_of(B_map), 0, 1, "", *coordB)  # noqa: E501
                T.ptx.mbarrier.arrive.expect_tx(T.address_of(bar), A_bytes + B_bytes)
            T.ptx.mbarrier.try_wait(T.address_of(bar), phase)
            phase = phase ^ 1
            T.cuda.cta_sync()

                    # init C_local
            for i in T.serial(0, C_elems):
                C_local[i] = T.Cast(out_dtype, get_init_value(out_dtype))
                T.ptx.wgmma.noop_barrier(C_local[i])

                    # do wgmma
            T.ptx.wgmma.encode_matrix_descriptor(T.address_of(descA), A_smem.data, *A_encode_args)  # noqa: F821
            T.ptx.wgmma.encode_matrix_descriptor(T.address_of(descB), B_smem.data, *B_encode_args)  # noqa: F821
            T.ptx.wgmma.fence()
            T.ptx.wgmma.mma_async.ss(descA, descB, *get_accum_list(C_local, C_elems),  # noqa: F821
                                     M=M, N=N, K=K, in_dtype=in_dtype, out_dtype=out_dtype, transA=transA, transB=transB, scaleA=1.0, scaleB=1.0, scaleD=False)  # noqa: E501
            T.ptx.wgmma.commit_group()
            T.ptx.wgmma.wait_group(0)

            for i in T.serial(0, C_elems):
                T.ptx.wgmma.noop_barrier(C_local[i])

                    # store C_local to C
            for i in T.serial(0, C_elems // 4):
                row = T.meta_var((tx % 32) // 4 + (tx // 32) * 16)
                col = T.meta_var(i * 8 + tx % 4 * 2)
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

    def run_and_check():
        dev = tvm.cuda(0)
        A_tvm = tvm.runtime.tensor(A_np, device=dev)
        B_tvm = tvm.runtime.tensor(B_np, device=dev)
        C_tvm = tvm.runtime.tensor(C_np, device=dev)
        mod(A_tvm, B_tvm, C_tvm)
        C_ref = np.dot(A_np.T, B_np).astype(out_dtype)
        tvm.testing.assert_allclose(C_tvm.numpy(), C_ref, rtol=1e-3, atol=1e-3)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9, exact=True), reason="need cuda compute == 9.0")
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
                return T.float32(0.0)
            assert False, f"Unsupported dtype {dtype}"

        def get_A_list(A_local, A_elems):
            return [A_local[i] for i in range(A_elems)]

        def get_accum_list(C, C_elems):
            return [C[i] for i in range(C_elems)]

        # fmt: off
        @T.prim_func
        def main(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle):
            A = T.match_buffer(A_ptr, shapeA, dtype=in_dtype, align=16)
            B = T.match_buffer(B_ptr, shapeB, dtype=in_dtype, align=16)
            C = T.match_buffer(C_ptr, shapeC, dtype=out_dtype, align=16)

            B_map: T.let[T.handle("tensormap")] = T.tvm_stack_alloca("tensormap", 1)
            T.call_packed("runtime.cuTensorMapEncodeTiled", B_map, in_dtype, len(shapeB), B.data, *B_tma_args)  # noqa: E501

            T.device_entry()
            cta_id = T.cta_id([1])
            tx = T.thread_id([128]) # A warpgroup is 128 threads

            B_smem = T.alloc_buffer(shapeB, in_dtype, scope="shared", align=1024)
                    # bar = T.alloc_buffer((1,), "uint64", scope="shared", align=8)
            bar = T.shared_scalar("uint64")

                    # descB = T.alloc_buffer((1,), "uint64", scope="local")
            descB: T.uint64
            A_local = T.alloc_buffer((A_elems,), in_dtype, scope="local")
            C_local = T.alloc_buffer((C_elems,), out_dtype, scope="local")

            A_elems_b32 = T.meta_var(A_elems // (32 // in_dtype_bits))
            A_local_b32 = T.decl_buffer((A_elems_b32,), "uint32", data=A_local.data)

                    # load A to regs
            for i in T.serial(0, A_elems // 4):
                row = T.meta_var((tx % 32) // 4 + (tx // 32) * 16)
                col = T.meta_var(i * 8 + tx % 4 * 2)
                A_local[i * 4] = A[row, col]
                A_local[i * 4 + 1] = A[row, col + 1]
                A_local[i * 4 + 2] = A[row + 8, col]
                A_local[i * 4 + 3] = A[row + 8, col + 1]
                    # init bar, and make sure it's visible to all threads and async proxy
            if tx == 0:
                T.ptx.mbarrier.init(T.address_of(bar), 1)
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.cta_sync()
                    # load B to smem
            if tx == 0:
                T.ptx.cp_async.bulk.tensor.g2c(len(shapeB), B_smem.data, T.address_of(bar), T.address_of(B_map), 0, 1, "", *coordB)  # noqa: E501
                T.ptx.mbarrier.arrive.expect_tx(T.address_of(bar), B_bytes)
            T.ptx.mbarrier.try_wait(T.address_of(bar), 0)
            T.cuda.cta_sync()

                    # init C_local
            for i in T.serial(0, C_elems):
                C_local[i] = T.Cast(out_dtype, get_init_value(out_dtype))

                    # fence A_local and C_local
            for i in T.serial(0, A_elems_b32):
                T.ptx.wgmma.noop_barrier(A_local_b32[i])
            for i in T.serial(0, C_elems):
                T.ptx.wgmma.noop_barrier(C_local[i])
                    # do wgmma
            T.ptx.wgmma.encode_matrix_descriptor(T.address_of(descB), B_smem.data, *B_encode_args)  # noqa: F821
            T.ptx.wgmma.fence()
            T.ptx.wgmma.mma_async.rs(descB, *(get_A_list(A_local_b32, A_elems_b32) + get_accum_list(C_local, C_elems)),  # noqa: E501, F821
                                     M=M, N=N, K=K, in_dtype=in_dtype, out_dtype=out_dtype, transA=transA, transB=transB, scaleA=1.0, scaleB=1.0, scaleD=False)  # noqa: E501
            T.ptx.wgmma.commit_group()
            T.ptx.wgmma.wait_group(0)

                    # fence A_local
            for i in T.serial(0, A_elems_b32):
                T.ptx.wgmma.noop_barrier(A_local_b32[i])
                    # fence C_local
            for i in T.serial(0, C_elems):
                T.ptx.wgmma.noop_barrier(C_local[i])

                    # store C_local to C
            for i in T.serial(0, C_elems // 4):
                row = T.meta_var((tx % 32) // 4 + (tx // 32) * 16)
                col = T.meta_var(i * 8 + tx % 4 * 2)
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

    np.printoptions(threshold=np.inf)
    np.printoptions(linewidth=np.inf)
    np.printoptions(precision=2)

    C_ref = np.dot(A_np, B_np).astype(out_dtype)

    def run_and_check():
        dev = tvm.cuda(0)
        A_tvm = tvm.runtime.tensor(A_np, device=dev)
        B_tvm = tvm.runtime.tensor(B_np, device=dev)
        C_tvm = tvm.runtime.tensor(C_np, device=dev)
        mod(A_tvm, B_tvm, C_tvm)
        tvm.testing.assert_allclose(C_tvm.numpy(), C_ref, rtol=1e-3, atol=1e-3)

    tvm.testing.run_with_gpu_lock(run_and_check)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
def test_ptx_map_shared_rank():
    @T.prim_func
    def func(A: T.Buffer(1)):
        T.device_entry()
        cbx = T.cta_id_in_cluster([2])
        cta_id = T.cta_id([2])
        tx = T.thread_id([128])
        A_smem = T.alloc_buffer([1], "uint32", scope="shared")
        if cbx == 0 and tx == 0:
            T.ptx.map_shared_rank(A_smem.data, cbx)

    src, mod = _get_source(func)
    print(src)
    assert "tvm_builtin_ptx_mapa_u64(A_smem" in src


if __name__ == "__main__":
    tvm.testing.main()
