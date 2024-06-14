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
# pylint: disable=invalid-name,unused-argument

"""Matmul schedules for the `arm_cpu` device key."""

import tvm
from tvm import te
from tvm import autotvm
from tvm.script import tir as T
from tvm.topi import nn
from tvm.topi.utils import get_const_tuple
from tvm.topi.arm_cpu.pstate_attributes import SMEAttributes
from tvm.topi.arm_cpu.arm_utils import pad_dim_to_multiple
from tvm.dlight.base.analysis import normalize_prim_func


@autotvm.register_topi_compute("matmul.arm_cpu.sme")
def compute_matmul_sme(cfg, data_a, data_b, _, out_dtype, transpose_a=False, transpose_b=True):
    """
    SME Matmul compute definition.
    """
    assert bool(transpose_a) is False, "Transposed lhs not currently supported."
    if data_b.dtype == "float16":
        assert bool(transpose_b) is True, "Rhs must be transposed when dtype is float16."

    M, K = get_const_tuple(data_a.shape)
    if transpose_b:
        N = get_const_tuple(data_b.shape)[0]
    else:
        N = get_const_tuple(data_b.shape)[1]

    if not out_dtype:
        out_dtype = data_a.dtype

    tile_m = 2 * tvm.tir.get_vscale_expr(data_a.dtype)
    tile_k = tvm.tir.get_vscale_expr(data_a.dtype)
    if data_a.dtype == "float32":
        tile_k *= 2
    tile_n = 2 * tvm.tir.get_vscale_expr(data_a.dtype)

    if data_a.dtype == "float16":
        _, pad_M = pad_dim_to_multiple(M, tile_m)
        _, pad_K = pad_dim_to_multiple(K, tile_k)
        _, pad_N = pad_dim_to_multiple(N, tile_n)
        m_pad_after = (pad_M, pad_K)
        n_pad_after = (pad_N, pad_K) if transpose_b else (pad_K, pad_N)
        if pad_M != 0:
            data_a = nn.pad(data_a, pad_before=(0, 0), pad_after=m_pad_after)
        if pad_N != 0:
            data_b = nn.pad(data_b, pad_before=(0, 0), pad_after=n_pad_after)

    if out_dtype is None:
        out_dtype = data_a.dtype

    k = te.reduce_axis((0, K), name="k")

    def compute(*indices):
        i, j = indices[-2:]
        a_indices = (k, i) if transpose_a else (i, k)
        b_indices = (j, k) if transpose_b else (k, j)
        return te.sum(
            data_a[a_indices].astype(out_dtype) * data_b[b_indices].astype(out_dtype), axis=k
        )

    compute_name = {
        (True, True): "T_matmul_TT",
        (True, False): "T_matmul_TN",
        (False, True): "T_matmul_NT",
        (False, False): "T_matmul_NN",
    }[(transpose_a, transpose_b)]

    return te.compute(
        (M, N),
        compute,
        name=compute_name,
        attrs={"schedule_type": "sme"},
    )


def tir_schedule_matmul_sme(sch):
    """
    SME STIR Matmul schedule.
    """
    # pylint: disable=import-outside-toplevel
    from tvm.tir.tensor_intrin.arm_cpu import (
        ARM_SME_2SVLx2SVL_GEMM_INTERLEAVED_MOPA,
        ARM_SME_INIT,
        get_sme_gemm_interleaved_mopa_2svlx2svl_intrin,
        get_transpose_interleave_intrin_name,
    )

    main_func = sch.mod["main"]
    data_handle = main_func.params[0]
    in_dtype = main_func.buffer_map[data_handle].dtype
    out_dtype = "float32"

    block_infos = normalize_prim_func(sch)
    reduction_block_infos = [block_info for block_info in block_infos if block_info.is_reduction()]
    assert len(reduction_block_infos) == 1, "Expected a single gemm reduction block."
    gemm_block = reduction_block_infos[0].block_rv
    gemm_block_name = sch.get(gemm_block).name_hint
    transpose = gemm_block_name.split("_")[-1]
    transpose_b = transpose[1] == "T"

    m, n, k = sch.get_loops(gemm_block)

    extent_m = sch.get(m).extent
    extent_k = sch.get(k).extent
    extent_n = sch.get(n).extent

    if in_dtype == "float16":
        tile_m = T.cast(2 * tvm.tir.get_vscale_expr(in_dtype), extent_m.dtype)
        tile_k = T.cast(tvm.tir.get_vscale_expr(in_dtype), extent_k.dtype)
        tile_n = T.cast(2 * tvm.tir.get_vscale_expr(in_dtype), extent_n.dtype)
    else:
        tile_m = T.cast(2 * tvm.tir.get_vscale_expr(in_dtype), extent_m.dtype)
        tile_k = T.cast(2 * tvm.tir.get_vscale_expr(in_dtype), extent_k.dtype)
        tile_n = T.cast(2 * tvm.tir.get_vscale_expr(in_dtype), extent_n.dtype)

    # Interleave the input utilizing the matrix tile
    interleave_a_block = sch.cache_read(gemm_block, 0, "global")
    sch.transform_layout(interleave_a_block, ("write", 0), lambda m, k: (k, m))
    m, k = sch.get_loops(interleave_a_block)
    outer_m, inner_m = sch.split(m, factors=(None, tile_m), disable_predication=True)
    outer_k, inner_k = sch.split(k, factors=(None, tile_k), disable_predication=True)
    sch.reorder(outer_k, outer_m, inner_k, inner_m)
    sch.tensorize(
        inner_k, get_transpose_interleave_intrin_name(in_dtype, out_dtype, extent_m, extent_k)
    )

    # Interleave the weights utilizing the matrix tile
    if transpose_b:
        interleave_b_block = sch.cache_read(gemm_block, 1, "global")
        sch.transform_layout(interleave_b_block, ("write", 0), lambda n, k: (k, n))
        n, k = sch.get_loops(interleave_b_block)
        outer_k, inner_k = sch.split(k, factors=(None, tile_k), disable_predication=True)
        outer_n, inner_n = sch.split(n, factors=(None, tile_n), disable_predication=True)
        sch.reorder(outer_k, outer_n, inner_k, inner_n)
        sch.tensorize(
            inner_k, get_transpose_interleave_intrin_name(in_dtype, out_dtype, extent_k, extent_n)
        )

    # Split and reorder the loops of the GeMM for tensorization
    tile_m = T.cast(2 * tvm.tir.get_vscale_expr(out_dtype), extent_m.dtype)
    tile_n = T.cast(2 * tvm.tir.get_vscale_expr(out_dtype), extent_n.dtype)
    m, n, k = sch.get_loops(gemm_block)
    outer_m, inner_m = sch.split(m, factors=(None, tile_m), disable_predication=True)
    outer_n, inner_n = sch.split(n, factors=(None, tile_n), disable_predication=True)
    sch.reorder(outer_m, outer_n, inner_m, inner_n, k)

    # Tensorize the GeMM initialization
    init_block = sch.decompose_reduction(gemm_block, inner_m)
    sch.tensorize(sch.get_loops(init_block)[-2], ARM_SME_INIT)

    # Tensorize the GeMM update
    sme_gemm_interleaved_intrin_name = (
        ARM_SME_2SVLx2SVL_GEMM_INTERLEAVED_MOPA + f"_{extent_m}_{extent_k}_{in_dtype}"
    )
    tvm.tir.TensorIntrin.register(
        sme_gemm_interleaved_intrin_name,
        *get_sme_gemm_interleaved_mopa_2svlx2svl_intrin(extent_m, extent_k, in_dtype),
        override=True,
    )
    sch.tensorize(inner_m, sme_gemm_interleaved_intrin_name)

    # Add pstate annotations
    root_block = sch.get_block("root")
    sch.annotate(
        root_block, SMEAttributes.STREAMING_MODE, SMEAttributes.StreamingModeValues.ENABLED
    )
    sch.annotate(root_block, SMEAttributes.ZA_STORAGE, SMEAttributes.ZAStorageValues.NEW)
