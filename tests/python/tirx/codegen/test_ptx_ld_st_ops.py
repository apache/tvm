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
"""Unit tests for generic PTX ``T.ptx.ld`` / ``T.ptx.st`` vector copy ops."""

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

DEV = tvm.cuda(0)
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
    rt_args = [tvm.runtime.tensor(a, device=DEV) for a in np_args]
    mod(*rt_args)
    return (*tuple(a.numpy() for a in rt_args), mod)


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


def test_ptx_ld_st_ops_registered():
    """PTX ld/st must be registered TIR ops and exposed on the T.ptx namespace."""
    for name in ("tirx.ptx.ld", "tirx.ptx.st"):
        Op.get(name)  # raises if unregistered

    for attr in (
        "ld",
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
