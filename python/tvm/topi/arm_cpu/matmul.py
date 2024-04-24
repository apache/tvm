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


@autotvm.register_topi_compute("matmul.arm_cpu.sme")
def compute_matmul_sme(cfg, data_a, data_b, _, out_dtype, transpose_a=False, transpose_b=False):
    """
    SME Matmul compute definition.
    """
    assert (
        transpose_a == transpose_b == False
    ), "Compute definition currently does not support transposed inputs."

    M, K = get_const_tuple(data_a.shape)
    N = get_const_tuple(data_b.shape)[1]

    if not out_dtype:
        out_dtype = data_a.dtype

    tile_m = 2 * 4 * tvm.tir.vscale()
    tile_n = 2 * 4 * tvm.tir.vscale()

    M_padded, pad_M = pad_dim_to_multiple(M, tile_m)
    N_padded, pad_N = pad_dim_to_multiple(N, tile_n)
    if pad_M != 0:
        data_a = nn.pad(data_a, pad_before=(0, 0), pad_after=(pad_M, 0))
    if pad_N != 0:
        data_b = nn.pad(data_b, pad_before=(0, 0), pad_after=(0, pad_N))

    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M_padded, N_padded),
        lambda m, n: te.sum(
            data_a[m, k].astype(data_a.dtype) * data_b[k, n].astype(data_b.dtype),
            axis=k,
        ).astype(out_dtype),
        name="matmul_sme_gemm",
    )
    C = te.compute((M, N), lambda m, n: C[m, n])
    return C


def tir_schedule_matmul_sme(sch):
    """
    SME STIR Matmul schedule.
    """
    # pylint: disable=import-outside-toplevel
    from tvm.tir.tensor_intrin.arm_cpu import (
        ARM_SME_2SVLx2SVL_TRANSPOSE_INTERLEAVE,
        ARM_SME_2SVLx2SVL_GEMM_INTERLEAVED_MOPA,
        ARM_SME_INIT,
        get_sme_gemm_interleaved_mopa_2svlx2svl_intrin,
    )

    gemm_block = sch.get_block("matmul_sme_gemm")
    m, n, k = sch.get_loops(gemm_block)

    extent_m = sch.get(m).extent
    extent_k = sch.get(k).extent

    tile_m = T.cast(2 * 4 * T.vscale(), extent_m.dtype)
    tile_k = T.cast(2 * 4 * T.vscale(), extent_k.dtype)
    tile_n = T.cast(2 * 4 * T.vscale(), sch.get(n).extent.dtype)

    # Interleave the input utilizing the matrix tile
    interleave_a_block = sch.cache_read(gemm_block, 0, "global")
    sch.transform_layout(interleave_a_block, ("write", 0), lambda m, k: (k, m))
    m, k = sch.get_loops(interleave_a_block)
    outer_m, inner_m = sch.split(m, factors=(None, tile_m), disable_predication=True)
    outer_k, inner_k = sch.split(k, factors=(None, tile_k), disable_predication=True)
    sch.reorder(outer_k, outer_m, inner_k, inner_m)
    sch.tensorize(inner_k, ARM_SME_2SVLx2SVL_TRANSPOSE_INTERLEAVE)

    # Split and reorder the loops of the GeMM for tensorization
    m, n, k = sch.get_loops(gemm_block)
    outer_m, inner_m = sch.split(m, factors=(None, tile_m), disable_predication=True)
    outer_n, inner_n = sch.split(n, factors=(None, tile_n), disable_predication=True)
    sch.reorder(outer_m, outer_n, inner_m, inner_n, k)

    # Tensorize the GeMM initialization
    init_block = sch.decompose_reduction(gemm_block, inner_m)
    sch.tensorize(sch.get_loops(init_block)[-2], ARM_SME_INIT)

    # Tensorize the GeMM update
    sme_gemm_interleaved_intrin_name = ARM_SME_2SVLx2SVL_GEMM_INTERLEAVED_MOPA + f"_{extent_k}"
    tvm.tir.TensorIntrin.register(
        sme_gemm_interleaved_intrin_name,
        *get_sme_gemm_interleaved_mopa_2svlx2svl_intrin(extent_k),
        override=True,
    )
    sch.tensorize(inner_m, sme_gemm_interleaved_intrin_name)

    # Add pstate annotations
    root_block = sch.get_block("root")
    sch.annotate(
        root_block, SMEAttributes.STREAMING_MODE, SMEAttributes.StreamingModeValues.ENABLED
    )
    sch.annotate(root_block, SMEAttributes.ZA_STORAGE, SMEAttributes.ZAStorageValues.NEW)
