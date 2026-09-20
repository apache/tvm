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
"""Unit tests for the ptx ``ld`` / ``st`` entries, scalar and vector."""

import numpy as np
import pytest

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx.cuda.tile_primitive.copy._common import copy_ptx_form

TARGET = tvm.target.Target("cuda")

# num_bytes → kernel layout. ``fill_offset`` fills lane i with ``i + fill_offset``.
_SHARED_COPY_CASES = {
    16: {"nelems": 4, "smem_dtype": "uint32", "tmp_dtype": "uint32", "fill_offset": 1},
    8: {"nelems": 2, "smem_dtype": "uint32", "tmp_dtype": "uint32", "fill_offset": 10},
    4: {"nelems": 1, "smem_dtype": "uint32", "tmp_dtype": "uint32", "fill_value": 42},
    2: {"nelems": 1, "smem_dtype": "float16", "tmp_dtype": "uint16", "fill_fp16": 7.0},
    1: {"nelems": 1, "smem_dtype": "uint8", "tmp_dtype": "uint32", "fill_u8": 255},
}


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
    tail, lanes, reg_dtype = copy_ptx_form(num_bytes)
    ld_chain, st_chain = f"ld.shared.{tail}", f"st.shared.{tail}"

    @T.prim_func
    def func(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (nelems,), smem_dtype)
        T.device_entry()
        T.cta_id([1])
        T.warp_id([1])
        lane = T.lane_id([32])
        src_buf = T.alloc_buffer((nelems,), smem_dtype, scope="shared")
        dst_buf = T.alloc_buffer((nelems,), smem_dtype, scope="shared")
        tmp = T.alloc_local((lanes,), reg_dtype)
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
            T.ptx[ld_chain](*[tmp[i] for i in range(lanes)], src_buf.ptr_to([0]))
            T.ptx[st_chain](dst_buf.ptr_to([0]), *[tmp[i] for i in range(lanes)])
        T.cuda.cta_sync()
        if lane < nelems:
            out[lane] = dst_buf[lane]

    return func


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
            T.ptx.st.shared.v4.u32(smem.ptr_to([0]), reg[0], reg[1], reg[2], reg[3])
        T.cuda.cta_sync()
        if tid_in_wg == 0:
            T.ptx.ld.shared.v4.u32(reg[0], reg[1], reg[2], reg[3], smem.ptr_to([0]))
        Tx.copy(D[0:4], reg[:])
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": copy_kernel}), target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "ld.shared" in src, "PTX ld did not emit ld.shared"
    assert "st.shared" in src, "PTX st did not emit st.shared"
    assert "ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];" in src
    assert "st.shared.v4.u32 [%0], {%1, %2, %3, %4};" in src


def test_ptx_ld_st_raw_shared_address_codegen():
    @T.prim_func
    def main(out: T.Buffer((2,), "uint64")):
        T.device_entry()
        tx = T.thread_id([32])
        smem = T.alloc_buffer((2,), "uint64", scope="shared")
        values = T.alloc_local((4,), "uint32")
        if tx == 0:
            raw_addr: T.uint32 = T.cuda.cvta_generic_to_shared(smem.data)
            T.ptx.ld.shared.u64(out[0], raw_addr)
            T.ptx.ld.shared.u64(out[1], smem.data)
            T.ptx.st.weak.shared__cta.b128(raw_addr, values.view("uint128")[0])

    with TARGET:
        mod = tvm.compile(tvm.IRModule({"main": main}), target=TARGET, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "ld.shared.u64 %0, [%1];" in src
    # One cvta, for the generic pointer. The raw window address is already a
    # uint32 and passes straight through -- ptx converts only pointers.
    assert src.count("__cvta_generic_to_shared") == 1
    assert '"st.weak.shared::cta.b128 [%0], %1;"' in src
    assert '"q"(__value)' in src


def test_ptx_ld_st_immediate_offset_codegen():
    """An immediate displacement must stay inside the PTX memory operand."""

    @T.prim_func
    def main(src: T.Buffer((4,), "uint64"), out: T.Buffer((4,), "uint64")):
        T.device_entry()
        tx = T.thread_id([32])
        values = T.alloc_local((2,), "uint64")
        if tx == 0:
            T.ptx.ld.global_.v2.b64(values[0], values[1], T.ptx.addr(src.data, 16))
            T.ptx.st.global_.v2.b64(T.ptx.addr(out.data, 16), values[0], values[1])

    with TARGET:
        mod = tvm.compile(tvm.IRModule({"main": main}), target=TARGET, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "ld.global.v2.b64 {%0, %1}, [%2+16];" in src
    assert "st.global.v2.b64 [%0+16], {%1, %2};" in src


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
            T.ptx["ld.global.nc.L1::no_allocate.L2::evict_first.L2::256B.v8.s32"](
                *[tmp[i] for i in range(8)], src.data
            )
            for i in T.unroll(8):
                out[i] = tmp[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": copy_kernel}), target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "ld.global.nc.L1::no_allocate.L2::evict_first.L2::256B.v8.s32" in src
    assert "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];" in src


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
            T.ptx["ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v4.u64"](
                tmp[0], tmp[1], tmp[2], tmp[3], src.data
            )
            for i in T.unroll(4):
                out[i] = tmp[i]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": copy_kernel}), target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "ld.global.nc.L1::no_allocate.L2::evict_normal.L2::256B.v4.u64" in src
    assert "{%0, %1, %2, %3}, [%4];" in src


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
            T.ptx["ld.global.nc.v4.s32"](tmp0[0], tmp1[0], tmp2[0], tmp3[0], src.data)
            out[0] = tmp0[0]
            out[1] = tmp1[0]
            out[2] = tmp2[0]
            out[3] = tmp3[0]

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": copy_kernel}), target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "ld.global.nc.v4.s32 {%0, %1, %2, %3}, [%4];" in src
    # Four independent destinations: each lane is its own reference parameter.
    assert "int32_t& __d0, int32_t& __d1, int32_t& __d2, int32_t& __d3" in src


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
    tail, _lanes, _dtype = copy_ptx_form(num_bytes)
    vec = tail.split(".")[0] if tail.startswith("v") else ""
    if vec == "v4":
        assert "ld.shared.v4" in src
        assert "st.shared.v4" in src
    elif vec == "v2":
        assert "ld.shared.v2" in src
        assert "st.shared.v2" in src
