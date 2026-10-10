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
import gc
import re

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.backend.cuda.lang.clc import query_cancel_first_ctaid_x
from tvm.script import tirx as T
from tvm.testing import env


def _get_source(
    func: tvm.tirx.Function, target=None, compile_config=None
) -> tuple[str, tvm.IRModule]:
    if target is None:
        target = {"kind": "cuda", "arch": "sm_100a"}
    target = tvm.target.Target(target)
    mod = tvm.IRModule({"main": func})
    with target:
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx", compile_config=compile_config)
    src = mod.mod.imports[0].inspect_source()
    return src, mod


@pytest.mark.parametrize("dtype", ["float16", "float16x2", "float32x2", "int32x2"])
def test_isnan_scalar_and_vector_codegen(dtype):
    result_dtype = "boolx2" if dtype.endswith("x2") else "bool"

    @T.function
    def func(A: T.Tensor((4,), dtype), B: T.Tensor((4,), result_dtype)):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(4,)))
        tx = T.cuda.thread_idx("x")
        B[tx] = T.isnan(A[tx])

    source, _ = _get_source(func)
    assert "!=" in source


def _helper_source(src: str, helper_name: str) -> str:
    pattern = rf"__forceinline__ __device__ [^{{;]+ {re.escape(helper_name)}\("
    match = re.search(pattern, src)
    if match is None:
        raise ValueError(f"helper {helper_name!r} not found")
    start = match.start()
    next_helper = src.find("__device__", start + len(helper_name))
    if next_helper == -1:
        return src[start:]
    return src[start:next_helper]


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_multi_gpu(), reason="need multiple GPUs")
def test_cuda_ndarray_destructor_preserves_current_device():
    torch = pytest.importorskip("torch")

    original_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        data = tvm.runtime.tensor(np.zeros(1, dtype="int32"), device=tvm.cuda(0))

        torch.cuda.set_device(1)
        del data
        gc.collect()

        assert torch.cuda.current_device() == 1
    finally:
        torch.cuda.set_device(original_device)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_multi_gpu(), reason="need multiple GPUs")
def test_cuda_stream_free_preserves_current_device():
    torch = pytest.importorskip("torch")

    original_device = torch.cuda.current_device()
    stream = None
    try:
        torch.cuda.set_device(0)
        stream = tvm.cuda(0).create_raw_stream()

        torch.cuda.set_device(1)
        tvm.cuda(0).free_raw_stream(stream)
        stream = None

        assert torch.cuda.current_device() == 1
    finally:
        if stream is not None:
            tvm.cuda(0).free_raw_stream(stream)
        torch.cuda.set_device(original_device)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_multi_gpu(), reason="need multiple GPUs")
def test_cuda_module_destructor_preserves_current_device():
    torch = pytest.importorskip("torch")

    @T.function
    def main(A: T.Tensor((1,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(1,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            A[0] = A[0] + 1

    _, mod = _get_source(main, target="cuda")
    original_device = torch.cuda.current_device()
    try:
        torch.cuda.set_device(0)
        data = tvm.runtime.tensor(np.zeros(1, dtype="int32"), device=tvm.cuda(0))
        mod["main"](data)
        tvm.cuda(0).sync()
        del data
        gc.collect()

        torch.cuda.set_device(1)
        del mod
        gc.collect()

        assert torch.cuda.current_device() == 1
    finally:
        torch.cuda.set_device(original_device)


def test_vector_pointer_preserves_packed_offset(monkeypatch):
    buffer = tvm.tirx.decl_tensor((8,), "int4x4", name="A")
    data = tvm.tirx.Var("A_data", buffer.data.ty)
    pointer = tvm.tirx.ptr_byte_offset(buffer.data, 4, ty=buffer.data.ty)
    body = tvm.ir.SeqStmt(
        [
            tvm.ir.Bind(
                buffer,
                tvm.ir.Call(
                    "tirx.decl_tensor",
                    [
                        data,
                        tvm.ir.Tuple(buffer.shape),
                        tvm.ir.DataTypeImm(tvm.DataType(buffer.dtype)),
                        tvm.ir.StringImm(buffer.scope()),
                    ],
                    ty=buffer.ty,
                ),
            ),
            tvm.ir.Evaluate(tvm.tirx.call_extern("void", "consume", pointer)),
        ]
    )
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    func = (
        tvm.tirx.Function([data], body)
        .with_attr("global_symbol", "main")
        .with_attr("target", target)
    )
    lowered = tvm.tirx.transform.LowerIntrin()(tvm.IRModule.from_expr(func))

    monkeypatch.setenv("TVM_COMPILE_FORCE_FALLBACK", "1")
    source = tvm.get_global_func("target.build.cuda")(lowered, target).inspect_source()
    call = next(line.strip() for line in source.splitlines() if line.strip().startswith("consume("))

    assert "make_int4" not in call
    assert " + 4" in call


def test_tirx_launch_bounds_omits_min_blocks_without_persistent_schedule():
    @T.function
    def main(A: T.Tensor((4,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(4,), block=(128,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            A[bx] = A[bx] + 1

    src, _ = _get_source(main)
    assert 'extern "C" __global__ void __launch_bounds__(128) main_kernel' in src
    assert "__launch_bounds__(128, 1)" not in src


def test_tirx_launch_bounds_min_blocks_sets_one_block_per_sm():
    @T.function
    def main(A: T.Tensor((4,), "int32")):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(4,), block=(128,)),
            kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=1),
        )
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            A[bx] = A[bx] + 1

    src, _ = _get_source(main)
    assert 'extern "C" __global__ void __launch_bounds__(128, 1) main_kernel' in src
    assert "tirx.cuda.launch_bounds_min_blocks_per_sm" not in src


def test_single_thread_block_preserves_explicit_launch_bounds():
    @T.function
    def main(A: T.Tensor((1,), "int32")):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=1, block=1),
            kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=2),
        )
        A[0] = 1

    src, _ = _get_source(main)
    assert "__launch_bounds__(1, 2)" in src


def test_tirx_launch_bounds_max_blocks_per_cluster_emits_third_operand():
    @T.function
    def main(A: T.Tensor((4,), "int32")):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(4,), block=(384,)),
            kernel_attrs=T.cuda.KernelAttributes(min_blocks_per_sm=1, max_blocks_per_cluster=1),
        )
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            A[bx] = A[bx] + 1

    src, _ = _get_source(main)
    assert 'extern "C" __global__ void __launch_bounds__(384, 1, 1) main_kernel' in src
    assert "tirx.cuda.launch_bounds_max_blocks_per_cluster" not in src


def test_tirx_max_registers_emits_cuda_maxnreg():
    @T.function
    def main(A: T.Tensor((4,), "int32")):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(4,), block=(128,)),
            kernel_attrs=T.cuda.KernelAttributes(max_registers_per_thread=92),
        )
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            A[bx] = A[bx] + 1

    src, _ = _get_source(main)
    assert 'extern "C" __global__ void __maxnreg__(92) main_kernel' in src
    assert "__launch_bounds__" not in src
    assert "tirx.cuda.max_registers_per_thread" not in src


def test_tirx_max_registers_rejects_launch_bounds():
    with pytest.raises(ValueError, match="conflicts"):
        T.cuda.KernelAttributes(max_registers_per_thread=92, min_blocks_per_sm=1)


def test_tirx_required_block_size_emits_cuda_block_size():
    @T.function
    def main(A: T.Tensor((8,), "int32")):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(4, 2), block=(128,), cluster=(1, 2)),
            kernel_attrs=T.cuda.KernelAttributes(required_block_size=True),
        )
        bx, by = (T.cuda.block_idx("x"), T.cuda.block_idx("y"))
        _, cy = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            A[bx * 2 + by] = cy

    src, _ = _get_source(main)
    assert 'extern "C" __global__ void __block_size__((128, 1, 1), (1, 2, 1)) main_kernel' in src
    assert "__launch_bounds__" not in src
    assert "tirx.cuda.required_block_size" not in src


def test_tirx_required_block_size_emits_launch_bounds_when_requested():
    @T.function
    def main(A: T.Tensor((4,), "int32")):
        T.device_entry(
            launch=T.cuda.LaunchConfig(grid=(4,), block=(128,)),
            kernel_attrs=T.cuda.KernelAttributes(required_block_size=True, min_blocks_per_sm=1),
        )
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            A[bx] = A[bx] + 1

    src, _ = _get_source(main)
    assert (
        'extern "C" __global__ void __block_size__((128, 1, 1), (1, 1, 1)) '
        "__launch_bounds__(128, 1) main_kernel" in src
    )
    assert "tirx.cuda.required_block_size" not in src
    assert "tirx.cuda.launch_bounds_min_blocks_per_sm" not in src


def test_tirx_cuda_kernel_return_zero_codegen_is_void_early_return():
    @T.function
    def main(A: T.Tensor((4,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(4,), block=(32,)))
        bx = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        if bx >= 3:
            return 0
        if tx == 0:
            A[bx] = A[bx] + 1

    src, _ = _get_source(main)
    assert re.search(r"if \(\(\(int\)blockIdx\.x\) == 3\)", src)
    assert "return;" in src
    assert "return 0;" not in src


def test_serial_pragma_unroll_codegen():
    @T.function
    def main(A: T.Tensor((4,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            for i in T.serial(4, unroll=True):
                if i == 2:
                    break
                A[i] = A[i] + 1

    src, _ = _get_source(main)
    assert "#pragma unroll\n" in src
    assert "for (" in src
    assert "break;" in src


def test_serial_pragma_unroll_count_codegen():
    @T.function
    def main(A: T.Tensor((4,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            for i in T.serial(4, unroll=2):
                A[i] = A[i] + 1

    src, _ = _get_source(main)
    assert re.search(r"#pragma unroll 2\s*for \(", src)


def test_serial_disable_unroll_pragma_immediately_precedes_dynamic_for():
    @T.function
    def main(A: T.Tensor((4,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            begin: T.let = T.if_then_else(A[0] > 0, A[1], A[2])
            end: T.let = T.if_then_else(A[0] > 1, A[2], A[3])
            for i in T.serial(begin, end, unroll=False):
                A[0] = A[0] + i

    src, _ = _get_source(main)
    assert re.search(r"#pragma unroll 1\s*for \(", src)


def test_cluster_cta_id_codegen_uses_coordinate_sregs():
    @T.function
    def main(A: T.Tensor((1,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(2, 2), block=(32,), cluster=(2, 2)))
        cbx, cby = (T.cuda.cluster_cta_id("x"), T.cuda.cluster_cta_id("y"))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            A[0] = cbx + cby

    src, _ = _get_source(main)
    assert "%cluster_ctaid.x" in src
    assert "%cluster_ctaid.y" in src
    assert "%cluster_ctarank" not in src
    assert "cooperative_groups::cluster_group::block_index" not in src


@pytest.mark.gpu
def test_cuda_handle_uint64_reinterpret_codegen():
    @T.function
    def main(A: T.Tensor((1,), "uint64")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            ptr: T.let = T.reinterpret(A[0], ty="handle")
            A[0] = T.reinterpret(ptr, ty="uint64")

    src, _ = _get_source(main)
    assert "(void*)A_ptr[0]" in src
    assert "reinterpret_cast<uint64_t>" in src
    assert "*(void* *)" not in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_cuda_atomic_add():
    @T.function
    def main(A: T.Tensor((1,), "int32"), B: T.Tensor((1,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(32,)))
        cta_id = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            T.cuda.atomic_add(A.data, T.int32(1))
            T.cuda.atomic_add(B.data, T.float32(1.0))

    src, mod = _get_source(main, target="cuda")
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
    @T.function
    def main(A: T.Tensor((1,), "uint64"), B: T.Tensor((1,), "int32"), C: T.Tensor((1,), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            T.ptx.ld.acquire.gpu.global_.u64(A[0], A.data)
            T.ptx.ld.acquire.sys.global_.s32(B[0], B.data)
            T.ptx.ld.acquire.gpu.global_.b32(C[0], C.data)
            T.ptx.ld.acquire.gpu.global_.b32(B[0], B.data)
            T.ptx.ld.volatile.global_.u64(A[0], A.data)

    src, _ = _get_source(main)
    assert "ld.acquire.gpu.global.u64" in src
    assert "ld.acquire.sys.global.s32" in src
    assert "ld.acquire.gpu.global.b32" in src
    assert "tvm_builtin_ptx_ld_acquire_gpu_global_b32_s32" in src
    assert "ld.volatile.global.u64" in src


def test_ptx_f32x2_value_codegen():
    @T.function
    def main(A: T.Tensor((2,), "uint64"), B: T.Tensor((2,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            lhs: T.let = T.cuda.make_float2(B[0], B[1])
            rhs: T.let = T.cuda.make_float2(B[1], B[0])
            prod = T.local_scalar("uint64")
            sum_pair = T.local_scalar("uint64")
            T.ptx.mul.f32x2(prod, lhs, rhs)
            T.ptx.fma.rn.f32x2(A[0], lhs, rhs, prod)
            T.ptx.add.rn.f32x2(sum_pair, lhs, rhs)
            T.ptx.sub.rn.f32x2(A[1], sum_pair, rhs)

    src, _ = _get_source(main)
    assert "tvm_builtin_ptx_mul_f32x2" in src
    assert "tvm_builtin_ptx_fma_rn_f32x2" in src
    assert "tvm_builtin_ptx_add_rn_f32x2" in src
    assert "tvm_builtin_ptx_sub_rn_f32x2" in src
    # `.rnd` is optional on add/mul and mandatory on fma; the tokens written at
    # the call site are exactly the tokens emitted.
    assert "mul.f32x2 %0, %1, %2;" in src
    assert "mul.rn.f32x2 %0, %1, %2;" not in src
    assert "fma.rn.f32x2 %0, %1, %2, %3;" in src
    assert "add.rn.f32x2 %0, %1, %2;" in src


def test_ptx_neg_f32_codegen():
    """`neg{.ftz}.f32` (ISA 9.7.3.10) -- the exact form, without fast-math .ftz."""

    @T.function
    def main(A: T.Tensor((2,), "float32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            T.ptx.neg.f32(A[1], A[0])

    src, _ = _get_source(main)
    assert "neg.f32 %0, %1;" in src
    assert "neg.ftz.f32" not in src


def test_ptx_sub_f16x2_codegen():
    """The packed half line `sub{.rnd}{.ftz}{.sat}.f16x2` (ISA 9.7.4.2)."""

    @T.function
    def main(A: T.Tensor((3,), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            T.ptx.sub.f16x2(A[2], A[0], A[1])

    src, _ = _get_source(main)
    assert "sub.f16x2 %0, %1, %2;" in src


@pytest.mark.skipif(
    not (env.has_cuda_compute(10, 0) and env.has_nvcc_version(13, 4)),
    reason="packed bf16 conversion requires sm_100; the dialect certifies on CUDA 13.4",
)
def test_sparse_decode_conversion_intrinsics_codegen():
    @T.function
    def main(
        U16: T.Tensor((1,), "uint16"),
        U32: T.Tensor((2,), "uint32"),
        U64: T.Tensor((1,), "uint64"),
        F32: T.Tensor((2,), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            pair: T.let = T.cuda.make_float2(F32[0], F32[1])
            T.ptx.cvt.rz.ue8m0x2.f32(U16[0], F32[1], F32[0])
            T.ptx.cvt.rn.bf16x2.ue8m0x2(U32[0], U16[0])
            T.ptx.cvt.rn.bf16x2.e4m3x2(U32[1], U16[0])
            T.ptx.add.f32x2(U64[0], pair, pair)

    src, _ = _get_source(main, compile_config=T.cuda.CompileConfig(compiler="nvcc"))
    assert "cvt.rz.ue8m0x2.f32 %0, %1, %2;" in src
    assert "cvt.rn.bf16x2.ue8m0x2 %0, %1;" in src
    assert "cvt.rn.bf16x2.e4m3x2 %0, %1;" in src
    assert "__nv_cvt_" not in src
    assert "add.f32x2 %0, %1, %2;" in src
    assert "add.rn.f32x2 %0, %1, %2;" not in src


@pytest.mark.gpu
def test_megamoe_extracted_intrinsics_codegen():
    @T.function
    def main(
        U32: T.Tensor((4,), "uint32"),
        I32: T.Tensor((1,), "int32"),
        U64: T.Tensor((1,), "uint64"),
        F32: T.Tensor((4,), "float32"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            T.ptx.red.release.gpu.global_.or_.b64(U64.data, U64[0])
            T.ptx.red.release.sys.global_.add.s32(I32.data, I32[0])
            T.ptx.atom.release.gpu.global_.add.u32(U32[0], U32.data, U32[0])
            T.ptx.atom.sys.global_.add.u64(U64[0], U64.data, U64[0])
            T.ptx.red.gpu.global_.add.u32(U32.data, U32[0])
            T.ptx.st.shared.u32(U32.data, U32[0])
            T.ptx.st.shared.v4.b32(U32.data, U32[0], U32[1], U32[2], U32[3])
            T.ptx.st_bulk.weak.shared__cta(U32.data, T.uint64(16))
            T.ptx.fns.b32(U32[0], U32[0], U32[1], I32[0])
            T.ptx.movmatrix.sync.aligned.m8n8.trans.b16(U32[1], U32[0])
            T.ptx.stmatrix.sync.aligned.m16n8.x1.trans.shared.b8(U32.data, U32[0])

            F32[1] = T.cuda.uint_as_float(U32[0])
            T.ptx.ld.global_.f32(F32[2], F32.data)
            F32[2] = F32[1]
            F32[3] = T.cuda.fdividef(F32[0], F32[1])
            U32[3] = T.cuda.float_as_uint(F32[1])
            T.ptx.add.rn.f32.bf16(F32[0], T.cast(U32[0], "uint16"), F32[0])
            U64[0] = T.reinterpret(U32.data, ty="uint64")
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
        "movmatrix.sync.aligned.m8n8.trans.b16",
        "stmatrix.sync.aligned.m16n8.x1.trans.shared.b8",
        "ld.global.f32",
        "tvm_builtin_ptx_ld_global_f32(F32_ptr[2], F32_ptr);",
        "__fdividef",
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
    @T.function
    def main(
        A: T.Tensor((128,), "float32"),
        B: T.Tensor((128,), "float32"),
        C: T.Tensor((1,), "uint64"),
    ):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            smem = T.alloc_shared([128], "float32")
            T.ptx["cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"](
                smem.ptr_to([0]), A.data, T.uint32(64), smem.ptr_to([0]), C[0]
            )
            T.ptx[
                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
            ](smem.ptr_to([0]), A.data, T.uint32(64), smem.ptr_to([0]), C[0])
            T.ptx["cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint"](
                B.data, smem.ptr_to([0]), T.uint32(64), C[0]
            )
            T.ptx.cp.async_.bulk.commit_group()
            T.ptx.cp.async_.bulk.wait_group.read(0)
            T.ptx.cp.async_.bulk.wait_group(1)

    src, _ = _get_source(main)
    assert "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint" in src
    assert "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint" in src
    assert "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint" in src
    assert "uint64_t __cache_policy" in src
    assert 'asm volatile("cp.async.bulk.wait_group.read 0;" :  :  : "memory");' in src
    assert 'asm volatile("cp.async.bulk.wait_group 1;" :  :  : "memory");' in src


def test_ptx_sync_and_clc_codegen():
    @T.function
    def main(A: T.Tensor((1,), "uint32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            bar = T.alloc_tensor((5,), "uint64", scope="shared", align=16)
            response = T.alloc_tensor((4,), "uint32", scope="shared", align=16)
            T.ptx.cp.async_.mbarrier.arrive.shared.b64(bar.ptr_to([0]))
            T.ptx.cp.async_.mbarrier.arrive.noinc.shared__cta.b64(bar.ptr_to([0]))
            T.cuda.mbarrier_wait(bar.ptr_to([0]), T.int32(0))
            T.ptx.mbarrier.complete_tx.shared.b64(bar.ptr_to([0]), T.uint32(T.uint32(16)))
            T.ptx.mbarrier.complete_tx.relaxed.cta.shared__cta.b64(bar.ptr_to([1]), T.uint32(24))
            T.ptx.mbarrier.complete_tx.relaxed.cta.shared__cta.b64(
                bar.ptr_to([2]), T.uint32(28), pred=T.uint32(1)
            )
            # The remote form is a mapa plus a complete_tx on the mapped window
            # address, which is what the fused legacy helper emitted inside one
            # asm block.
            remote_bar = T.alloc_local([1], "uint32")
            T.ptx.mapa.shared__cluster.u32(
                remote_bar[0], T.cuda.cvta_generic_to_shared(bar.ptr_to([3])), T.uint32(1)
            )
            T.ptx.mbarrier.complete_tx.relaxed.cluster.shared__cluster.b64(
                remote_bar[0], T.uint32(32)
            )
            T.ptx.mbarrier.complete_tx.relaxed.cluster.shared__cluster.b64(
                remote_bar[0], T.uint32(40), pred=T.uint32(1)
            )
            T.ptx[
                "clusterlaunchcontrol.try_cancel.async.shared::cta"
                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
            ](response.ptr_to([0]), bar.ptr_to([0]))
            nxt = T.local_scalar("uint32")
            query_cancel_first_ctaid_x(nxt, response.ptr_to([0]))
            A[0] = nxt
            query_cancel_first_ctaid_x(nxt, response.ptr_to([0]), use_ld_acquire=False)
            A[0] = nxt
            T.ptx.griddepcontrol.launch_dependents()

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.compile(tvm.IRModule({"main": main}), target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    assert "cp.async.mbarrier.arrive.shared.b64" in src
    assert "cp.async.mbarrier.arrive.noinc.shared::cta.b64" in src
    # The spin-wait moved to T.cuda.mbarrier_wait, which takes its timeout as a
    # parameter rather than baking 10000000 into the asm text.
    assert "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;" in src
    assert "tvm_builtin_cuda_mbarrier_wait" in src
    # (No assertion on the timeout local: T.cuda.mbarrier_wait keeps the
    # `ticks = 0x989680` hint the TIRx spin-wait convention specifies.)
    assert "mbarrier.complete_tx.relaxed.cluster.shared::cluster.b64" in src
    assert "mbarrier.complete_tx.relaxed.cta.shared::cta.b64" in src
    assert "@p mbarrier.complete_tx.relaxed.cta.shared::cta.b64" in src
    assert "@p mbarrier.complete_tx.relaxed.cluster.shared::cluster.b64" in src
    # A ptx helper is exactly one instruction, so the mapping is a call of its
    # own rather than something folded into the complete_tx helper.
    assert "mapa.shared::cluster.u32" in src
    for helper in (
        "tvm_builtin_ptx_mbarrier_complete_tx_complete_tx_relaxed_cta_shared__cta_b64",
        "tvm_builtin_ptx_mbarrier_complete_tx_complete_tx_relaxed_cluster_shared__cluster_b64",
    ):
        assert "mapa" not in _helper_source(src, helper)
    assert "mbarrier.complete_tx.shared::cluster.relaxed.cluster.b64" not in src
    assert "clusterlaunchcontrol.try_cancel.async.shared::cta" in src
    assert "ld.acquire.cta.shared.b128" in src
    assert "ld.shared.b128" in src
    assert "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128" in src
    assert "griddepcontrol.launch_dependents" in src


def test_ptx_mbarrier_arrive_new_forms_codegen():
    @T.function
    def main(Pred: T.Tensor((1,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            bar = T.alloc_tensor((6,), "uint64", scope="shared", align=16)
            state = T.local_scalar("uint64")
            T.ptx.mbarrier.arrive.relaxed.cta.shared__cta.b64(bar.ptr_to([0]))
            T.ptx.mbarrier.arrive.relaxed.cluster.shared__cluster.b64(bar.ptr_to([1]))
            T.ptx.mbarrier.arrive.release.cta.shared.b64(bar.ptr_to([2]), pred=Pred[0])
            T.ptx.mbarrier.arrive.expect_tx.relaxed.cluster.shared__cluster.b64(
                bar.ptr_to([3]), T.uint32(128)
            )
            T.ptx.mbarrier.arrive.noComplete.release.cta.shared.b64(
                state, bar.ptr_to([4]), T.uint32(2)
            )
            # noComplete writes a real state result, yet pred= stays legal:
            # the accumulator binds "+", so a false predicate leaves it intact.
            T.ptx.mbarrier.arrive.noComplete.release.cta.shared__cta.b64(
                state, bar.ptr_to([5]), T.uint32(3), pred=Pred[0]
            )

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.compile(tvm.IRModule({"main": main}), target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source()
    assert "mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" in src
    assert "mbarrier.arrive.relaxed.cluster.shared::cluster.b64 _, [%0];" in src
    assert "@p mbarrier.arrive.release.cta.shared.b64 _, [%0];" in src
    assert "mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [%0], %1;" in src
    assert "mbarrier.arrive.noComplete.release.cta.shared.b64 %0, [%1], %2;" in src
    assert "@p mbarrier.arrive.noComplete.release.cta.shared::cta.b64 %0, [%1], %2;" in src


def test_tensor_map_param_codegen():
    @T.function
    def main(A_map: T.TensorMap()):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            T.evaluate(T.address_of(A_map))

    src, _ = _get_source(main)
    assert "const __grid_constant__ CUtensorMap A_map" in src
    assert "((unsigned long long)(&(A_map)))" in src


_TMA_G2S_CG2_CACHE = (
    "cp.async.bulk.tensor.2d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
)
_TMA_G2S_MC_CG2_CACHE = (
    "cp.async.bulk.tensor.2d.shared::cluster.global"
    ".mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
)
_TMA_CTA_GATHER4_CG2_CACHE = (
    "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
    ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
)
_TMA_S2G_CACHE = "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"


def test_tma_cache_policy_operand_codegen():
    @T.function
    def main(Cache: T.Tensor((1,), "uint64")):
        A_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)
        B_map: T.let[T.handle("tensormap")] = T.stack_alloca("tensormap", 1)

        T.device_entry(launch=T.cuda.LaunchConfig(grid=1, block=(32,)))
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            smem = T.alloc_tensor((128,), "float32", scope="shared", align=128)
            bar = T.shared_scalar("uint64")
            T.ptx[_TMA_G2S_CG2_CACHE](
                smem.data, T.address_of(A_map), 0, 0, T.address_of(bar), Cache[0]
            )
            T.ptx[_TMA_G2S_MC_CG2_CACHE](
                smem.data, T.address_of(A_map), 0, 0, T.address_of(bar), 3, Cache[0]
            )
            T.ptx[_TMA_S2G_CACHE](T.address_of(A_map), 0, 0, smem.data, Cache[0])
            T.ptx[_TMA_CTA_GATHER4_CG2_CACHE](
                smem.data, T.address_of(A_map), 0, 1, 2, 3, 4, T.address_of(bar), Cache[0]
            )
            leader_mbar_addr = T.cuda.sm100_2sm_leader_smem_addr(T.address_of(bar))
            T.ptx[_TMA_G2S_CG2_CACHE](
                smem.data, T.address_of(A_map), 0, 0, leader_mbar_addr, Cache[0]
            )
            if tx == 0:
                T.ptx[_TMA_G2S_CG2_CACHE](
                    smem.data, T.address_of(A_map), 0, 0, leader_mbar_addr, Cache[0]
                )
            else:
                T.ptx[_TMA_G2S_CG2_CACHE](
                    smem.data, T.address_of(B_map), 0, 0, leader_mbar_addr, Cache[0]
                )

    src, _ = _get_source(main)
    assert _TMA_G2S_CG2_CACHE in src
    assert _TMA_G2S_MC_CG2_CACHE in src
    assert _TMA_CTA_GATHER4_CG2_CACHE in src
    assert _TMA_S2G_CACHE in src
    assert "bar_addr &= 0xFEFFFFFFu;" not in src
    assert "mbar_addr &= 0xFEFFFFFFu;" not in src
    assert "tvm_builtin_cuda_cvta_generic_to_shared((&(bar_ptr[0]))) & (uint)4278190079" in src
    assert "uint64_t __cache_policy" in src


def test_cuda_thread_fence():
    @T.function
    def main(A: T.Tensor((16, 16), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(32,)))
        cta_id = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            T.cuda.thread_fence()

    src, mod = _get_source(main)
    assert "tvm_builtin_cuda_thread_fence" in src


def test_cuda_nano_sleep():
    @T.function
    def main(A: T.Tensor((16, 16), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(32,)))
        cta_id = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
        if tx == 0:
            T.cuda.nano_sleep(1)

    src, mod = _get_source(main)
    assert "tvm_builtin_cuda_nano_sleep" in src


def test_cuda_atomic_cas():
    @T.function
    def main(A: T.Tensor((16, 16), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(32,)))
        cta_id = T.cuda.block_idx("x")
        tx = T.cuda.thread_idx("x")
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

        @T.function
        def main(a: T.Tensor((16, 16), "int32"), b: T.Tensor((16, 16), "int32")):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(32,)))
            cta_id = T.cuda.block_idx("x")
            tx = T.cuda.thread_idx("x")
            if tx == 0:
                for i, j in T.grid(16, 16):
                    b[i, j] = T.cuda.func_call("add_one", a[i, j], add_one, ty="int32")

        src, mod = _get_source(main, target="cuda")
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

        @T.function
        def main(a: T.Tensor((16, 16), "int32")):
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(32,)))
            cta_id = T.cuda.block_idx("x")
            tx = T.cuda.thread_idx("x")
            if tx == 0:
                for i, j in T.grid(16, 16):
                    T.cuda.func_call("print", a[i, j], print_func)

        src, mod = _get_source(main, target="cuda")
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
    @T.function
    def func(A: T.Tensor((32,), dtype='float32', align=16)):

        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=1 * 32))
        cta_id = T.cuda.block_idx('x')
        warp_id = T.cuda.warp_id()
        lane_id = T.cuda.lane_id()

        A_local = T.alloc_tensor([1], "float32", scope="local")
        i = T.alloc_tensor([1], "int32", scope="local")

        A_local[0] = T.float32(31 - lane_id)
        i[0] = 16
        while i[0] >= 1:
            A_local[0] += T.gpu_warp_shuffle_xor(0xFFFFFFFF, A_local[0], i[0], 32, 32)
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

    cop = "cg" if cp_size == 16 else "ca"
    cache_tok = ".L2::cache_hint" if cache_hint else ""
    pref_tok = "" if prefetch_size == -1 else f".L2::{prefetch_size}B"
    chain = f"cp.async.{cop}.shared.global{cache_tok}{pref_tok}"
    cache_args = (T.uint64(0x14F0000000000000),) if cache_hint else ()
    has_pred = not (isinstance(predicate, int) and predicate == -1)
    src_size = None
    if fill_mode == "zero":
        from tvm.tirx.op import if_then_else

        src_size = T.cast(if_then_else(predicate != 0, cp_size, 0), "uint32")

    # fmt: off
    @T.function
    def main(A: T.Tensor((N), "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(32,)))
        cta_id = T.cuda.block_idx('x')
        tid = T.cuda.thread_idx('x')
        A_shared = T.alloc_shared([N], "float16")
        for i in T.vectorized(N):
            A_shared[i] = 5.0
        T.ptx.fence.proxy.async_.shared__cta()
        if T.constexpr(fill_mode == "zero"):
            T.ptx[chain](A_shared.ptr_to([0]), A.ptr_to([0]), cp_size, src_size, *cache_args)
        elif T.constexpr(has_pred):
            T.ptx[chain](A_shared.ptr_to([0]), A.ptr_to([0]), cp_size, *cache_args, pred=predicate)
        else:
            T.ptx[chain](A_shared.ptr_to([0]), A.ptr_to([0]), cp_size, *cache_args)
        T.ptx.cp.async_.commit_group()
        T.ptx.cp.async_.wait_group(0)
        for i in T.serial(N):
            A[i] = A_shared[i] + 1.0
        # fmt: on

    src, mod = _get_source(main, target="cuda")
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
    # fmt: off
    @T.function
    def main(A: T.Tensor((16, 16), "float16"), B: T.Tensor((16, 16), "float16")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(32,)))
        cta_id = T.cuda.block_idx('x')
        tx = T.cuda.thread_idx('x')
        A_shared = T.alloc_shared([16, 16], "float16")
        if tx == 0:
            for i, j in T.grid(16, 16):
                A_shared[i, j] = A[i, j]
        T.cuda.cta_sync()
        A_local = T.alloc_local([8], "float16")
        A_local[0] = -1.0
        # ldmatrix .x{num}.b16 writes `num` b32 registers; A_local is a
        # contiguous fp16[8] buffer, so the registers land through a uint32
        # view, two fp16 elements per word.
        A_words = A_local.view("uint32")
        if T.constexpr(num == 1):
            T.ptx[
                "ldmatrix.sync.aligned.m8n8.x1"
                f"{'.trans' if T.constexpr(trans) else ''}.shared.b16"
            ](
                A_words[0],
                A_shared.ptr_to([tx % 16, tx // 16 * 8]),
            )
        elif T.constexpr(num == 2):
            T.ptx[
                "ldmatrix.sync.aligned.m8n8.x2"
                f"{'.trans' if T.constexpr(trans) else ''}.shared.b16"
            ](
                A_words[0],
                A_words[1],
                A_shared.ptr_to([tx % 16, tx // 16 * 8]),
            )
        else:
            T.ptx[
                "ldmatrix.sync.aligned.m8n8.x4"
                f"{'.trans' if T.constexpr(trans) else ''}.shared.b16"
            ](
                A_words[0],
                A_words[1],
                A_words[2],
                A_words[3],
                A_shared.ptr_to([tx % 16, tx // 16 * 8]),
            )
        for i in range(8):
            row: T.let = (i // 2) % 2 * 8
            col: T.let = (i // 4) * 8
            B[row + tx // 4, col + tx % 4 * 2 + i % 2] = A_local[i]
        # fmt: on

    src, mod = _get_source(main, target="cuda")
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


def test_uint32_loop_var_and_scope_id_emit_unsigned():
    @T.function
    def main(A: T.Tensor((128,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(128,)))
        _ = T.cuda.block_idx("x")
        tx = T.cast(T.cuda.thread_idx("x"), "uint32")
        for k in T.serial(4, dtype="uint32"):
            A[tx] = A[tx] + T.int32(k)

    src, _ = _get_source(main)
    # The loop var is declared unsigned and iterates over unsigned bounds.
    assert re.search(r"for \(uint k = \(uint\)0; k < \(uint\)4;", src), src
    # The scope id is bound as an unsigned value.
    assert re.search(r"uint tx = ", src), src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_uint32_loop_var_runs_correctly():
    @T.function
    def main(A: T.Tensor((128,), "int32"), B: T.Tensor((128,), "int32")):
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(128,)))
        _ = T.cuda.block_idx("x")
        tx = T.cast(T.cuda.thread_idx("x"), "uint32")
        acc = T.alloc_tensor((1,), "int32", scope="local")
        acc[0] = 0
        for k in T.serial(4, dtype="uint32"):
            acc[0] = acc[0] + A[tx] + T.int32(k)
        B[tx] = acc[0]

    _, mod = _get_source(main, target="cuda")

    A_np = np.arange(128, dtype="int32")
    B_ref = A_np * 4 + (0 + 1 + 2 + 3)

    def run_and_check():
        dev = tvm.cuda()
        A = tvm.runtime.tensor(A_np, device=dev)
        B = tvm.runtime.tensor(np.zeros(128, dtype="int32"), device=dev)
        mod(A, B)
        np.testing.assert_allclose(B.numpy(), B_ref)

    tvm.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm.testing.main()
