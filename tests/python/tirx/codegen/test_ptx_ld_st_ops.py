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
"""Unit tests for PTX ``T.ptx.ld`` / ``T.ptx.ld_global_nc`` / ``T.ptx.st`` ops."""

import numpy as np
import pytest

import tvm
from tvm.ir import Op
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx.cuda.operator.tile_primitive.copy._common import (
    copy_ptx_form,
    copy_ptx_ld_return_type,
)

TARGET = tvm.target.Target("cuda")

# num_bytes → kernel layout. ``fill_offset`` fills lane i with ``i + fill_offset``.
_SHARED_COPY_CASES = {
    16: {"nelems": 4, "smem_dtype": "uint32", "tmp_dtype": "uint32", "fill_offset": 1},
    8: {"nelems": 2, "smem_dtype": "uint32", "tmp_dtype": "uint32", "fill_offset": 10},
    4: {"nelems": 1, "smem_dtype": "uint32", "tmp_dtype": "uint32", "fill_value": 42},
    2: {"nelems": 1, "smem_dtype": "float16", "tmp_dtype": "uint16", "fill_fp16": 7.0},
    1: {"nelems": 1, "smem_dtype": "uint8", "tmp_dtype": "uint32", "fill_u8": 255},
}

_PTX_ST_SRC_VECTOR_CASES = [
    ("uint8", "u8", "v4", "uchar4 src_", "unsigned int r3 = static_cast<unsigned int>(src_.w)"),
    ("int8", "s8", "v4", "char4 src_", "int r3 = static_cast<int>(src_.w)"),
    ("uint16", "u16", "v4", "ushort4 src_", "unsigned short r3 = src_.w"),
    ("int16", "s16", "v4", "short4 src_", "short r3 = src_.w"),
    ("uint32", "b32", "v4", "uint4 src_", "unsigned int r3 = src_.w"),
    ("uint32", "u32", "v4", "uint4 src_", "unsigned int r3 = src_.w"),
    ("int32", "s32", "v4", "int4 src_", "int r3 = src_.w"),
    ("float32", "f32", "v4", "float4 src_", "float r3 = src_.w"),
    ("float64", "f64", "v2", "double2 src_", "double r1 = src_.y"),
]


def _build_and_run(func, *np_args):
    mod = tvm.compile(tvm.IRModule({"main": func}), target=TARGET, tir_pipeline="tirx")

    def run_and_check():
        dev = tvm.cuda(0)
        rt_args = [tvm.runtime.tensor(a, device=dev) for a in np_args]
        mod(*rt_args)
        return tuple(a.numpy() for a in rt_args)

    return (*tvm.testing.run_with_gpu_lock(run_and_check), mod)


def _expected_values(num_bytes: int) -> np.ndarray:
    spec = _SHARED_COPY_CASES[num_bytes]
    if "fill_offset" in spec:
        off, nelems = spec["fill_offset"], spec["nelems"]
        return np.array([off + i for i in range(nelems)], dtype=np.uint32)
    if "fill_fp16" in spec:
        return np.array([spec["fill_fp16"]], dtype=np.float16)
    if "fill_u8" in spec:
        return np.array([spec["fill_u8"]], dtype=np.uint8)
    return np.array([spec["fill_value"]], dtype=np.uint32)


def _shared_scratch_copy_kernel(num_bytes: int):
    """Build shared → local scratch → shared copy kernel for ``num_bytes`` width."""
    spec = _SHARED_COPY_CASES[num_bytes]
    smem_dtype = spec["smem_dtype"]
    tmp_dtype = spec["tmp_dtype"]
    nelems = spec["nelems"]
    fill_offset = spec.get("fill_offset")
    fill_value = spec.get("fill_value")
    fill_fp16 = spec.get("fill_fp16")
    fill_u8 = spec.get("fill_u8")
    vec, ptx_type = copy_ptx_form(num_bytes)
    return_type = copy_ptx_ld_return_type(ptx_type)

    @T.prim_func
    def func(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (nelems,), smem_dtype)
        T.device_entry()
        T.cta_id([1])
        T.warp_id([1])
        lane = T.lane_id([32])
        src_buf = T.alloc_buffer((nelems,), smem_dtype, scope="shared")
        dst_buf = T.alloc_buffer((nelems,), smem_dtype, scope="shared")
        tmp = T.alloc_local((nelems,), tmp_dtype)
        if fill_offset is not None:
            if lane < nelems:
                src_buf[lane] = T.uint32(lane + fill_offset)
        elif fill_fp16 is not None:
            if lane == 0:
                src_buf[0] = T.float16(fill_fp16)
        elif fill_u8 is not None:
            if lane == 0:
                src_buf[0] = T.uint8(fill_u8)
        elif lane == 0:
            src_buf[0] = T.uint32(fill_value)
        T.cuda.cta_sync()
        if lane == 0:
            T.ptx.ld(
                src_buf.ptr_to([0]),
                return_type,
                ptx_type,
                dst=tmp.ptr_to([0]),
                space="shared",
                vec=vec,
            )
            T.ptx.st(
                dst_buf.ptr_to([0]),
                src=tmp.ptr_to([0]),
                space="shared",
                vec=vec,
                ptx_type=ptx_type,
            )
        T.cuda.cta_sync()
        if lane < nelems:
            out[lane] = dst_buf[lane]

    return func


def _ptx_st_src_vector_kernel(dtype: str, ptx_type: str, vec: str):
    vec_len = int(vec[1:])

    @T.prim_func
    def func(out_ptr: T.handle) -> None:
        out = T.match_buffer(out_ptr, (1,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        smem = T.alloc_buffer((vec_len,), dtype, scope="shared")
        reg = T.alloc_local((vec_len,), dtype)
        if tx == 0:
            T.ptx.st(
                smem.ptr_to([0]), src=reg.ptr_to([0]), space="shared", vec=vec, ptx_type=ptx_type
            )
            out[0] = T.uint32(0)

    return func


def test_ptx_ld_st_ops_registered():
    """PTX ld/st must be registered TIR ops and exposed on the T.ptx namespace."""
    for name in ("tirx.ptx.ld", "tirx.ptx.ld_global_nc", "tirx.ptx.st"):
        Op.get(name)  # raises if unregistered

    for attr in (
        "ld",
        "ld_global_nc",
        "st",
        "ld_acquire",
        "st_release",
        "ld_volatile",
        "st_volatile",
    ):
        assert hasattr(T.ptx, attr), attr


def test_ptx_ld_st_codegen_emits_shared_asm():
    """Shared ↔ register typed copies must codegen to ``ld.shared`` / ``st.shared``."""

    # fmt: off
    @T.prim_func
    def copy_kernel(d_ptr: T.handle) -> None:
        D = T.match_buffer(d_ptr, (4,), "uint32")
        T.device_entry()
        T.warp_id([4])
        T.cta_id([1])
        T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])
        smem = T.alloc_buffer((4,), "uint32", scope="shared")
        reg = T.alloc_local((4,), "uint32")
        if tid_in_wg == 0:
            T.ptx.st(
                smem.ptr_to([0]), src=reg.ptr_to([0]), space="shared", vec="v4", ptx_type="u32"
            )
        T.cuda.cta_sync()
        if tid_in_wg == 0:
            T.ptx.ld(
                smem.ptr_to([0]), "uint32", "u32", dst=reg.ptr_to([0]), space="shared", vec="v4"
            )
        Tx.copy(D[0:4], reg[:])
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": copy_kernel}), target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "ld.shared" in src, "PTX ld did not emit ld.shared"
    assert "st.shared" in src, "PTX st did not emit st.shared"
    assert "tvm_builtin_ptx_ld" in src
    assert "tvm_builtin_ptx_st" in src


def test_ptx_ld_st_raw_shared_address_codegen():
    @T.prim_func
    def main(out: T.Buffer((2,), "uint64")):
        T.device_entry()
        tx = T.thread_id([32])
        smem = T.alloc_buffer((2,), "uint64", scope="shared")
        values = T.alloc_local((4,), "uint32")
        if tx == 0:
            raw_addr: T.uint32 = T.cuda.cvta_generic_to_shared(smem.data)
            out[0] = T.ptx.ld(raw_addr, "uint64", "u64", space="shared")
            out[1] = T.ptx.ld(smem.data, "uint64", "u64", space="shared")
            T.ptx.st(
                raw_addr,
                src=values.ptr_to([0]),
                weak=True,
                space="shared::cta",
                ptx_type="b128",
            )

    with TARGET:
        mod = tvm.compile(tvm.IRModule({"main": main}), target=TARGET, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "unsigned int address" in src
    assert "ld.shared.u64 %0, [%1];" in src
    assert src.count("__cvta_generic_to_shared") == 2
    assert 'asm volatile("st.weak.shared::cta.b128 [%0], %1;"' in src
    assert '"q"(value));' in src
    b128_helper = src[src.index("tvm_builtin_ptx_st_weak_shared_cta_b128_from_src") :]
    b128_helper = b128_helper[: b128_helper.index("\n}")]
    assert '"memory"' not in b128_helper


def test_ptx_ld_st_reject_invalid_integer_addresses_and_b128_forms():
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="uint32 address"):

        @T.prim_func
        def bad_global_raw(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            out[0] = T.ptx.ld(T.uint32(0), "uint32", "u32", space="global")

    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="integer address"):

        @T.prim_func
        def bad_shared_u64(out: T.Buffer((1,), "uint64")):
            T.device_entry()
            out[0] = T.ptx.ld(T.uint64(0), "uint64", "u64", space="shared")

    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="b128 requires"):

        @T.prim_func
        def bad_b128(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            T.ptx.st(
                T.uint32(0),
                out[0],
                weak=True,
                space="shared::cta",
                ptx_type="b128",
            )


def test_ptx_ld_global_nc_v8_codegen():
    """FlashMLA index loads need ``ld.global.nc`` with a 256B prefetch."""

    @T.prim_func
    def copy_kernel(src_ptr: T.handle, out_ptr: T.handle) -> None:
        src = T.match_buffer(src_ptr, (8,), "int32")
        out = T.match_buffer(out_ptr, (8,), "int32")
        T.device_entry()
        tx = T.thread_id([32])
        tmp = T.alloc_local((8,), "int32")
        if tx == 0:
            T.ptx.ld_global_nc(
                src.data,
                "int32",
                "s32",
                dst=tmp.ptr_to([0]),
                vec="v8",
                l1_evict="L1::no_allocate",
                l2_evict="L2::evict_first",
                prefetch_size="L2::256B",
            )
            for i in T.unroll(8):
                out[i] = tmp[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": copy_kernel}), target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "ld.global.nc.L1::no_allocate.L2::evict_first.L2::256B.v8.s32" in src
    assert "tvm_builtin_ptx_ld_global_nc_v8_s32_to_dst" in src
    assert "tvm_builtin_ptx_ld_plain_global_nc" not in src
    assert "dst[7] = r7" in src


def test_ptx_ld_global_nc_v4_u64_256b_codegen():
    """FlashMLA 32-byte index loads may use four 64-bit PTX outputs."""

    @T.prim_func
    def copy_kernel(src_ptr: T.handle, out_ptr: T.handle) -> None:
        src = T.match_buffer(src_ptr, (4,), "uint64")
        out = T.match_buffer(out_ptr, (4,), "uint64")
        T.device_entry()
        tx = T.thread_id([32])
        tmp = T.alloc_local((4,), "uint64")
        if tx == 0:
            T.ptx.ld_global_nc(
                src.data,
                "uint64",
                "u64",
                dst=tmp.ptr_to([0]),
                vec="v4",
                l1_evict="L1::no_allocate",
                l2_evict="L2::evict_normal",
                prefetch_size="L2::256B",
            )
            for i in T.unroll(4):
                out[i] = tmp[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": copy_kernel}), target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v4.u64" in src
    assert "tvm_builtin_ptx_ld_global_nc_v4_u64_to_dst" in src
    assert "tvm_builtin_ptx_ld_plain_global_nc" not in src
    assert "unsigned long long r3" in src


def test_ptx_ld_global_nc_scalar_cop_cache_hint_codegen():
    """The ``ld.global{.cop}.nc`` syntax form supports scalar cache-policy operands."""

    @T.prim_func
    def copy_kernel(src_ptr: T.handle, out_ptr: T.handle) -> None:
        src = T.match_buffer(src_ptr, (1,), "int32")
        out = T.match_buffer(out_ptr, (1,), "int32")
        T.device_entry()
        tx = T.thread_id([32])
        if tx == 0:
            out[0] = T.ptx.ld_global_nc(
                src.data,
                "int32",
                "s32",
                cop="ca",
                cache_hint="evict_last",
                prefetch_size="L2::128B",
            )

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": copy_kernel}), target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "ld.global.ca.nc.L2::cache_hint.L2::128B.s32" in src
    assert "tvm_builtin_ptx_ld_global_nc_ca_s32_int32_cache_hint_L2_128B" in src
    assert "unsigned long long cache_policy" in src


def test_ptx_ld_rejects_legacy_global_nc_cop():
    """``ld.global.nc`` must use the dedicated ``T.ptx.ld_global_nc`` wrapper."""

    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="ld_global_nc"):

        @T.prim_func
        def copy_kernel(src_ptr: T.handle) -> None:
            src = T.match_buffer(src_ptr, (1,), "int32")
            T.device_entry()
            tx = T.thread_id([32])
            if tx == 0:
                value: T.int32 = T.ptx.ld(src.data, "int32", "s32", cop="nc")
                T.evaluate(value)


def test_ptx_ld_global_nc_rejects_cop_with_eviction():
    """The ``.cop`` syntax form cannot mix with L1/L2 eviction modifiers."""

    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="l1_evict"):

        @T.prim_func
        def copy_kernel(src_ptr: T.handle) -> None:
            src = T.match_buffer(src_ptr, (1,), "int32")
            T.device_entry()
            tx = T.thread_id([32])
            if tx == 0:
                value: T.int32 = T.ptx.ld_global_nc(
                    src.data,
                    "int32",
                    "s32",
                    cop="ca",
                    l1_evict="L1::evict_first",
                )
                T.evaluate(value)


def test_ptx_ld_global_nc_rejects_non_nc_cop():
    """``ld.global.nc`` has a narrower ``.cop`` set than generic ``ld.global``."""

    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="cop"):

        @T.prim_func
        def copy_kernel(src_ptr: T.handle) -> None:
            src = T.match_buffer(src_ptr, (1,), "int32")
            T.device_entry()
            tx = T.thread_id([32])
            if tx == 0:
                value: T.int32 = T.ptx.ld_global_nc(src.data, "int32", "s32", cop="lu")
                T.evaluate(value)


def test_ptx_ld_vector_scatter_dst_codegen():
    """Vector loads may write independent destination pointers."""

    @T.prim_func
    def copy_kernel(src_ptr: T.handle, out_ptr: T.handle) -> None:
        src = T.match_buffer(src_ptr, (4,), "int32")
        out = T.match_buffer(out_ptr, (4,), "int32")
        T.device_entry()
        tx = T.thread_id([32])
        tmp0 = T.alloc_local((1,), "int32")
        tmp1 = T.alloc_local((1,), "int32")
        tmp2 = T.alloc_local((1,), "int32")
        tmp3 = T.alloc_local((1,), "int32")
        if tx == 0:
            T.ptx.ld_global_nc(
                src.data,
                "int32",
                "s32",
                dst=(
                    tmp0.ptr_to([0]),
                    tmp1.ptr_to([0]),
                    tmp2.ptr_to([0]),
                    tmp3.ptr_to([0]),
                ),
                vec="v4",
            )
            out[0] = tmp0[0]
            out[1] = tmp1[0]
            out[2] = tmp2[0]
            out[3] = tmp3[0]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": copy_kernel}), target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "ld.global.nc.v4.s32" in src
    assert "tvm_builtin_ptx_ld_global_nc_v4_s32_to_dst4" in src
    assert "tvm_builtin_ptx_ld_plain_global_nc" not in src
    assert "void* dst0, void* dst1, void* dst2, void* dst3, void* src_ptr" in src
    assert "*reinterpret_cast<int*>(dst3) = r3" in src


@pytest.mark.parametrize(
    ("dtype", "ptx_type", "vec", "vector_decl", "last_load"),
    _PTX_ST_SRC_VECTOR_CASES,
)
def test_ptx_st_from_src_vector_typed_load_codegen(dtype, ptx_type, vec, vector_decl, last_load):
    """``T.ptx.st(src=...)`` loads vector source pointers with the PTX element type."""

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": _ptx_st_src_vector_kernel(dtype, ptx_type, vec)}),
            target=target,
            tir_pipeline="tirx",
        )
    src = mod.mod.imports[0].inspect_source("cuda")
    assert vector_decl in src
    assert last_load in src
    assert f"st.shared.{vec}.{ptx_type}" in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize(
    "num_bytes",
    [16, 8, 4, 2, 1],
    ids=["128b", "64b", "32b", "16b", "8b"],
)
def test_ptx_ld_st_shared_copy_gpu(num_bytes):
    """GPU roundtrip for each supported PTX ld/st copy width (shared → scratch → shared)."""
    expected = _expected_values(num_bytes)
    kernel = _shared_scratch_copy_kernel(num_bytes)
    out_np = np.zeros_like(expected)
    result, mod = _build_and_run(kernel, out_np)
    if expected.dtype == np.uint8:
        np.testing.assert_array_equal(result, expected)
    elif expected.dtype == np.float16:
        np.testing.assert_allclose(result, expected)
    else:
        np.testing.assert_array_equal(result, expected)
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "tvm_builtin_ptx_ld" in src
    assert "tvm_builtin_ptx_st" in src
    vec, _ptx_type = copy_ptx_form(num_bytes)
    if vec == "v4":
        assert "ld.shared.v4" in src
        assert "st.shared.v4" in src
    elif vec == "v2":
        assert "ld.shared.v2" in src
        assert "st.shared.v2" in src
