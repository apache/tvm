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

"""Implementation of gemm_async operator dispatch for CUDA targets.

Registered op: gemm_async (1 variant: "tcgen05").
See the @register_dispatch block below for detailed documentation with
before/after IR examples.
"""

import functools
import operator

import tvm
from tvm.arith.analyzer import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.tirx import PrimFunc
from tvm.tirx import op as tirx_op
from tvm.tirx.layout import (
    ComposeLayout,
    Iter,
    R,
    S,
    TCol,
    TileLayout,
    TLane,
    tmem_datapath_layout,
)
from tvm.tirx.operator.tile_primitive import DispatchContext, predicate, register_dispatch
from tvm.tirx.stmt import AllocBuffer, Evaluate, SeqStmt, TilePrimitiveCall

from ..common import get_st_extent, smem_desc_add_16B_offset
from ..exec_scope_utils import single_thread
from ..tma_utils import (
    SwizzleMode,
    get_swizzle_mode_from_layout,
    mma_atom_layout,
    mma_atom_shape,
)

# Mirror of ``format_map`` in the dense ``encode_instr_descriptor`` codegen
# (``python/tvm/tirx/operator/intrinsics/cuda/tcgen05.py``). Used to fold the
# runtime-encoded instruction descriptor into a compile-time uint32 when
# all parameters are dispatch-time constants.
_INSTR_DESC_FORMAT_MAP = {
    "float16": 0,
    "bfloat16": 1,
    "tensor_float32": 2,
    "tf32": 2,
    "float8_e4m3fn": 0,
    "float8_e4m3fnuz": 0,
    "float8_e5m2": 1,
    "float6_e2m3fn": 3,
    "float6_e3m2fn": 4,
    "float4_e2m1fn": 5,
    "uint8": 0,
    "int8": 1,
    "float32": 1,
    "int32": 2,
}


def _dtype_name(dtype) -> str:
    dtype_obj = getattr(dtype, "dtype", None)
    if dtype_obj is not None:
        return str(dtype_obj)
    return str(dtype)


def _encode_instr_descriptor_dense_uint32(
    M,
    N,
    d_dtype,
    a_dtype,
    b_dtype,
    trans_a,
    trans_b,
    neg_a=False,
    neg_b=False,
    sat_d=False,
    is_sparse=False,
):
    """Compile-time port of the dense ``InstrDescriptor`` bitfield packing.

    See ``python/tvm/tirx/operator/intrinsics/cuda/header.py:InstrDescriptor``
    for the bit layout. Lets the dispatcher pass a literal ``uint32`` to
    ``T.ptx.tcgen05.mma`` instead of allocating + encoding a per-dispatch
    local descriptor on every gemm_async call (which forces an inline ``asm``
    block that ptxas cannot hoist out of the i_kv loop body).
    """
    d_dtype = _dtype_name(d_dtype)
    a_dtype = _dtype_name(a_dtype)
    b_dtype = _dtype_name(b_dtype)
    d_format = _INSTR_DESC_FORMAT_MAP[d_dtype]
    a_format = _INSTR_DESC_FORMAT_MAP[a_dtype]
    b_format = _INSTR_DESC_FORMAT_MAP[b_dtype]
    desc = 0
    desc |= (int(is_sparse) & 0x1) << 2
    desc |= (int(sat_d) & 0x1) << 3
    desc |= (d_format & 0x3) << 4
    desc |= (a_format & 0x7) << 7
    desc |= (b_format & 0x7) << 10
    desc |= (int(neg_a) & 0x1) << 13
    desc |= (int(neg_b) & 0x1) << 14
    desc |= (int(trans_a) & 0x1) << 15
    desc |= (int(trans_b) & 0x1) << 16
    desc |= ((N >> 3) & 0x3F) << 17
    desc |= ((M >> 4) & 0x1F) << 24
    return desc & 0xFFFFFFFF


def sf_smem_layout(rows, SF_K, sf_per_mma, sf_reuse=1, pipe_depth=None):
    """SMEM-side layout for SF in tcgen05.cp scale-factor copy.

    The hardware reads SFs in 128-row super-blocks: 32 lanes x 16 bytes/lane.
    The 16 bytes per lane row decompose as
    ``M_SF_INNER (=4) x sf_per_mma x in_lane_K`` where
    ``in_lane_K = epc / sf_per_mma`` and ``epc = 4`` (32-bit TMEM cell / 8-bit
    SF). The remaining ``K_outer = SF_K / epc`` super-blocks march along K
    with stride 512. ``sf_reuse > 1`` appends a stride-0 broadcast dim.

    Buffer shape: ``(rows, SF_K * sf_reuse)`` (or ``(pipe_depth, rows,
    SF_K * sf_reuse)``). Mirrors :func:`sf_tmem_layout` parameterization.

    Args:
        rows:        Multiple of 128; M-direction rows.
        SF_K:        Number of unique SFs along K per row (multiple of 4).
        sf_per_mma:  Atom inner SFs per MMA-K step (must divide 4).
                     nvfp4=4, mxfp4=2, fp8=1.
        sf_reuse:    Broadcast factor (stride-0 dim). 1 = no broadcast.
        pipe_depth:  Optional pipeline depth as outermost dim.
    """
    epc = 4
    M_SUPER_ROWS = 128
    LANE = 32
    M_SF_INNER = M_SUPER_ROWS // LANE
    if rows % M_SUPER_ROWS != 0:
        raise ValueError(f"rows={rows} must be a multiple of {M_SUPER_ROWS}")
    if epc % sf_per_mma != 0:
        raise ValueError(f"sf_per_mma={sf_per_mma} must divide epc={epc}")
    if SF_K % epc != 0:
        raise ValueError(f"SF_K={SF_K} must be a multiple of epc={epc}")

    in_lane_K = epc // sf_per_mma
    K_outer = SF_K // epc
    M_super = rows // M_SUPER_ROWS
    LANE_BYTES = epc * M_SF_INNER  # 16
    SUPER_BYTES = LANE_BYTES * LANE  # 512
    K_TOTAL_BYTES = SUPER_BYTES * K_outer
    STAGE_BYTES = K_TOTAL_BYTES * M_super

    raw_shape = [M_super, M_SF_INNER, LANE, K_outer, sf_per_mma, in_lane_K]
    raw_strides = [K_TOTAL_BYTES, epc, LANE_BYTES, SUPER_BYTES, in_lane_K, 1]
    if sf_reuse > 1:
        raw_shape.append(sf_reuse)
        raw_strides.append(0)
    # Drop unit (extent-1) dims for cleaner canonical form.
    shape = [s for s in raw_shape if s != 1]
    strides = [st for s, st in zip(raw_shape, raw_strides) if s != 1]
    if pipe_depth is not None:
        shape = [pipe_depth, *shape]
        strides = [STAGE_BYTES, *strides]
    return TileLayout(S[tuple(shape) : tuple(strides)])


def sf_tmem_layout(rows, SF_K, sf_per_mma, sf_reuse=1, pipe_depth=None):
    """Create a TileLayout for SFA/SFB TMEM via atom direct_sum outer (+ optional reuse dim).

    Args:
        rows:        CTA M-direction row count (multiple of 32).
        SF_K:        Number of *unique* SFs along K per row (loaded from gmem).
        sf_per_mma:  Atom inner SFs — number of SFs one MMA reads in K.
                     Equals ``mma_k // sf_vec``: nvfp4=4 (mma_k=64,sf_vec=16),
                     mxfp4=2 (64,32), fp8=1 (32,32).
        sf_reuse:    Number of MMAs that reuse one physical SF group via a
                     stride-0 broadcast dim. Equals ``quant_size // mma_k``;
                     default 1 (no reuse). fp8 blockwise with quant=128 and
                     mma_k=32 → ``sf_reuse=4``.
        pipe_depth:  Optional outer pipe-depth dim for double-buffered TMEM SF
                     allocations. Stride is ``M*epc @ TCol`` (one stage spans
                     ``M*epc`` cols). When ``None`` no pipe dim is added.

    Buffer shape: ``(rows, SF_K * sf_reuse)`` (or ``(pipe_depth, rows,
    SF_K * sf_reuse)``). Gemm dispatch iterates the last dim
    ``SF_K * sf_reuse`` MMA times; only ``SF_K`` distinct SFs are physically
    stored due to broadcast. Scale factor dtype is assumed 8-bit (epc=4);
    all current SF formats (e8m0fnu, e4m3fn) fit.
    """
    if SF_K % sf_per_mma != 0:
        raise ValueError(f"SF_K={SF_K} must be a multiple of sf_per_mma={sf_per_mma}")
    K = SF_K // sf_per_mma  # outer K iterations of unique SFs

    M = rows // 32
    epc = 4  # 32-bit TMEM column / 8-bit SF

    # Atom: one 32-row chunk, one MMA's worth of SF.
    atom = TileLayout(S[(32, sf_per_mma) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])

    if K == 1:
        outer = TileLayout(S[M : epc @ TCol])
    else:
        # Pack consecutive ki's within one uint32 TMEM column when possible.
        pack_factor = epc // sf_per_mma
        while pack_factor > 1 and K % pack_factor != 0:
            pack_factor //= 2
        if pack_factor > 1:
            K_outer = K // pack_factor
            if K_outer == 1:
                outer = TileLayout(S[(M, pack_factor) : (epc @ TCol, sf_per_mma @ TCol)])
            else:
                outer = TileLayout(
                    S[(M, K_outer, pack_factor) : (epc @ TCol, M * epc @ TCol, sf_per_mma @ TCol)]
                )
        else:
            outer = TileLayout(S[(M, K) : (epc @ TCol, M * epc @ TCol)])

    base = atom.direct_sum(outer, left_shape=[M, K], right_shape=[32, sf_per_mma])
    if sf_reuse == 1 and pipe_depth is None:
        return base
    shard = list(base.shard)
    if sf_reuse > 1:
        # Append a stride-0 reuse dim on TCol for fp8 blockwise (vec_NX) mode.
        shard.append(Iter(sf_reuse, 0, shard[0].axis))
    if pipe_depth is not None:
        # Prepend a pipe-depth dim that strides one stage (M*epc TCols).
        shard.insert(0, Iter(pipe_depth, M * epc, shard[0].axis))
    return TileLayout.from_iters(shard, list(base.replica), dict(base.offset))


def _compute_sf_mma_k(data_dtype, sf_dtype):
    """Compute sf_mma_k (scale factor elements per MMA iteration) from dtypes.

    This is determined by hardware constraints:
    - fp8 data + e8m0fnu SF: MMA_K=32, one SF per MMA → sf_mma_k=1
    - fp4 data + e8m0fnu SF: MMA_K=64, SF_VEC=32 → sf_mma_k=2
    - fp4 data + e4m3fn SF (nvfp4): MMA_K=64, SF_VEC=16 → sf_mma_k=4
    """
    data_dtype = _dtype_name(data_dtype)
    sf_dtype = _dtype_name(sf_dtype)
    if data_dtype in ("float8_e4m3fn", "float8_e5m2"):
        return 1  # MMA_K=32, one SF per MMA
    elif data_dtype == "float4_e2m1fn":
        if sf_dtype == "float8_e8m0fnu":
            return 2  # MMA_K=64, SF_VEC=32
        elif sf_dtype == "float8_e4m3fn":
            return 4  # MMA_K=64, SF_VEC=16 (nvfp4)
    raise ValueError(f"Unsupported data_dtype={data_dtype}, sf_dtype={sf_dtype} for sf_mma_k")


def _validate_sf_tmem_layout(slice_layout, rows, sf_K_total, sf_mma_k, name):
    """Validate SFA/SFB TMEM sliced layout matches atom direct_sum outer pattern.

    Validates that slice_layout (already sliced to last 2D: rows x sf_K_total)
    matches the atom:
      shard = ([32, sf_mma_k], [1@TLane, 1@TCol])
      replica = ([4], [32@TLane])
    """
    assert isinstance(slice_layout, TileLayout), (
        f"{name}: sliced layout must be TileLayout, got {type(slice_layout)}"
    )
    M = rows // 32

    assert sf_K_total % sf_mma_k == 0, (
        f"{name}: sf_K_total={sf_K_total} must be divisible by sf_mma_k={sf_mma_k}"
    )
    K = sf_K_total // sf_mma_k

    atom = TileLayout(S[(32, sf_mma_k) : (1 @ TLane, 1 @ TCol)] + R[4 : 32 @ TLane])
    # interleaved_shape is the interleaved domain [M, 32, K, sf_mma_k]
    outer = atom.is_direct_sum_right(slice_layout, [M, 32, K, sf_mma_k], [32, sf_mma_k])
    assert outer is not None, f"{name}: layout does not match atom direct_sum outer pattern"


def _choose_mma_tile(M, N, cta_group, MMA_N_MIN):
    """Select per-instruction (M_mma, N_mma) for tcgen05 tile decomposition.

    M is per-CTA M.  valid_M lists valid *descriptor* M values (total across
    the CTA group).  We compute M_total = M * cta_group and pick the largest
    descriptor M that divides it, then return M_mma = M_desc // cta_group.

    N_mma: if N <= 256 and N % MMA_N_MIN == 0, use N directly.
      Otherwise, largest valid N_mma <= 256 that divides N and is divisible by MMA_N_MIN.
    """
    M_total = M * cta_group
    valid_M = [128, 64] if cta_group == 1 else [256, 128]
    M_desc = next((m for m in valid_M if M_total % m == 0), None)
    assert M_desc is not None, (
        f"tcgen05: M_total={M_total} (M={M}, cta_group={cta_group}) not divisible by "
        f"any valid descriptor M (valid: {valid_M})"
    )
    M_mma = M_desc // cta_group

    if N <= 256 and N % MMA_N_MIN == 0:
        N_mma = N
    else:
        N_mma = next((n for n in range(256, MMA_N_MIN - 1, -MMA_N_MIN) if N % n == 0), None)
        assert N_mma is not None, (
            f"tcgen05: No valid N_mma <= 256 that divides N={N} (MMA_N_MIN={MMA_N_MIN})"
        )

    return M_mma, N_mma


def _layout_matches_datapath_f(tmem_buf) -> bool:
    """Return True if ``tmem_buf.layout`` structurally equals Layout F (M=64
    scattered) over the buffer's full (64, X) shape — i.e. the buffer was
    allocated via ``tmem_pool.alloc((64, X), datapath="F")``.

    Used by the C-operand layout check to accept M=64 MMA writes into Layout
    F C buffers (the canonical pairing for M=64 outputs that are read back
    via ``.16x*b`` M=64; see PTX ISA §9.7.16.10.5).
    """
    if tmem_buf.layout is None or int(tmem_buf.shape[0]) != 64:
        return False
    try:
        expected = tmem_datapath_layout("F", 64, tmem_buf.shape[1]).canonicalize()
        tvm.ir.assert_structural_equal(tmem_buf.layout.canonicalize(), expected)
        return True
    except (AssertionError, ValueError):
        return False


def gemm_async_tcgen05_impl(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    """Schedule an asynchronous GEMM operation using tcgen05.mma (Blackwell Tensor Core).

    Computes C = A @ B (with optional transpose on A/B and accumulation).
    Supports both regular MMA and block-scaled MMA for low-precision dtypes.

    When called from warp scope, automatically wraps tcgen05.mma with elect_sync
    so that only one thread in the warp issues the MMA instruction.

    Args:
        op_call: The TilePrimitiveCall containing:
            Regular (6 args):
            - args[0:3]: C, A, B buffer regions
            - args[3:6]: transA, transB, accum flags
            Block-scaled (8 args):
            - args[0:3]: C, A, B buffer regions
            - args[3:5]: SFA, SFB buffer regions (scale factors in tmem)
            - args[5:8]: transA, transB, accum flags
            Config:
            - config["cta_group"]: CTA group in tcgen05 instructions (default 1)
            - config["descI"]: Optional pre-encoded instruction descriptor
        sctx: Schedule context (single-thread or warp execution scope)

    Returns:
        A PrimFunc implementing the tcgen05 MMA schedule.

    Raises:
        ValueError: If buffer scopes are invalid (C must be tmem, A must be shared or tmem,
            B must be shared).
        AssertionError: If shape/layout constraints are not satisfied.
    """
    warp_scope = sctx.is_warp
    op_call = TilePrimitiveCall.downcast(op_call)
    is_block_scaled = op_call.is_block_scaled

    C_buffer_region: tvm.tirx.BufferRegion = op_call.output
    A_buffer_region: tvm.tirx.BufferRegion = op_call.lhs
    B_buffer_region: tvm.tirx.BufferRegion = op_call.rhs
    C_buffer, A_buffer, B_buffer = (
        C_buffer_region.buffer,
        A_buffer_region.buffer,
        B_buffer_region.buffer,
    )

    C_scope, A_scope, B_scope = C_buffer.scope(), A_buffer.scope(), B_buffer.scope()
    a_is_tmem = A_scope == "tmem"
    if a_is_tmem:
        if not (C_scope == "tmem" and B_scope.startswith("shared")):
            raise ValueError(
                f"tcgen05 schedule expected C_scope=tmem, B_scope=shared when A is tmem, "
                f"got C_scope={C_scope}, B_scope={B_scope}"
            )
    elif not (C_scope == "tmem" and A_scope.startswith("shared") and B_scope.startswith("shared")):
        raise ValueError(
            f"tcgen05 schedule expected C_scope=tmem, A_scope=shared, B_scope=shared, got C_scope={C_scope}, A_scope={A_scope}, B_scope={B_scope}"  # noqa: E501
        )

    analyzer = Analyzer()

    C_type, A_type, B_type = (
        _dtype_name(C_buffer.dtype),
        _dtype_name(A_buffer.dtype),
        _dtype_name(B_buffer.dtype),
    )
    assert C_type == "float32", f"tcgen05 schedule expected C_type=float32, got {C_type}"

    # fp32/bf16 storage may still use tf32 MMA semantics via is_AB_tf32.
    is_AB_tf32 = op_call.config.get("is_AB_tf32", False)
    A_sem = "tf32" if is_AB_tf32 else A_type
    B_sem = "tf32" if is_AB_tf32 else B_type

    # Valid A/B dtypes for block-scaled MMA (low-precision with per-block scale factors)
    _BLOCK_SCALED_DTYPES = ["float4_e2m1fn", "float8_e4m3fn"]

    _SCALE_FACTOR_DTYPES = ["float8_e8m0fnu", "float8_e4m3fn"]

    if is_block_scaled:
        assert A_type in _BLOCK_SCALED_DTYPES, (
            f"tcgen05 block-scaled schedule expected A_type in {_BLOCK_SCALED_DTYPES}, got {A_type}"
        )
        assert B_type in _BLOCK_SCALED_DTYPES, (
            f"tcgen05 block-scaled schedule expected B_type in {_BLOCK_SCALED_DTYPES}, got {B_type}"
        )
    else:
        _DENSE_DTYPES = [
            "float16",
            "bfloat16",
            "float8_e4m3fn",
            "float8_e5m2",
            "tensor_float32",
            "tf32",
        ]
        assert A_sem in _DENSE_DTYPES, (
            f"tcgen05 schedule expected A dtype in {_DENSE_DTYPES}, got {A_sem}"
        )
        assert B_sem in _DENSE_DTYPES, (
            f"tcgen05 schedule expected B dtype in {_DENSE_DTYPES}, got {B_sem}"
        )
    assert A_sem == B_sem, (
        f"tcgen05 schedule expect A and B MMA dtype to be the same, got A={A_sem}, B={B_sem}"
    )

    # Parse SFA/SFB and transA/transB/accum based on arg layout
    if is_block_scaled:
        SFA_buffer_region, SFB_buffer_region = op_call.sfa, op_call.sfb
        transA, transB, accum = op_call.transA, op_call.transB, op_call.accum
        SFA_buffer: tvm.tirx.Buffer = SFA_buffer_region.buffer
        SFB_buffer: tvm.tirx.Buffer = SFB_buffer_region.buffer
        SFA_scope, SFB_scope = SFA_buffer.scope(), SFB_buffer.scope()
        if not (SFA_scope == "tmem" and SFB_scope == "tmem"):
            raise ValueError(
                f"tcgen05 block-scaled schedule expected SFA_scope=tmem, SFB_scope=tmem, "
                f"got SFA_scope={SFA_scope}, SFB_scope={SFB_scope}"
            )
        SFA_type, SFB_type = _dtype_name(SFA_buffer.dtype), _dtype_name(SFB_buffer.dtype)
        SFA_slice_layout = SFA_buffer.layout.slice(SFA_buffer.shape, SFA_buffer_region.region)
        SFB_slice_layout = SFB_buffer.layout.slice(SFB_buffer.shape, SFB_buffer_region.region)
        SFA_elem_per_col = 32 // DataType(SFA_type).bits
        SFB_elem_per_col = 32 // DataType(SFB_type).bits
        assert SFA_type in _SCALE_FACTOR_DTYPES, (
            f"tcgen05 block-scaled schedule expected SFA_type in {_SCALE_FACTOR_DTYPES}, got {SFA_type}"  # noqa: E501
        )
        assert SFB_type in _SCALE_FACTOR_DTYPES, (
            f"tcgen05 block-scaled schedule expected SFB_type in {_SCALE_FACTOR_DTYPES}, got {SFB_type}"  # noqa: E501
        )
        # Compute sf_mma_k from data/SF dtypes and validate layouts
        sfa_sf_mma_k = _compute_sf_mma_k(A_type, SFA_type)
        sfb_sf_mma_k = _compute_sf_mma_k(B_type, SFB_type)
        assert sfa_sf_mma_k == sfb_sf_mma_k, (
            f"SFA and SFB must have same sf_mma_k, got sfa={sfa_sf_mma_k}, sfb={sfb_sf_mma_k}"
        )
        SFA_rows = int(SFA_buffer_region.region[-2].extent)
        SFA_K_total = int(SFA_buffer_region.region[-1].extent)
        SFB_rows = int(SFB_buffer_region.region[-2].extent)
        SFB_K_total = int(SFB_buffer_region.region[-1].extent)
        _validate_sf_tmem_layout(SFA_slice_layout, SFA_rows, SFA_K_total, sfa_sf_mma_k, "SFA")
        _validate_sf_tmem_layout(SFB_slice_layout, SFB_rows, SFB_K_total, sfb_sf_mma_k, "SFB")
    else:
        transA, transB, accum = op_call.transA, op_call.transB, op_call.accum

    cta_group = op_call.config.get("cta_group", 1)
    assert cta_group in [1, 2], f"tcgen05 schedule expected cta_group=1 or 2, got {cta_group}"
    # descI: pre-encoded instruction descriptor (uint32), if None we encode it locally
    descI = op_call.config.get("descI", None)

    C_elem_size = DataType(C_type).bits
    C_elem_per_32b = 32 // C_elem_size
    C_st, C_extent = get_st_extent(C_buffer_region)
    _, A_extent = get_st_extent(A_buffer_region)
    _, B_extent = get_st_extent(B_buffer_region)
    A_slice_layout = A_buffer.layout.slice(A_buffer.shape, A_buffer_region.region)
    B_slice_layout = B_buffer.layout.slice(B_buffer.shape, B_buffer_region.region)
    C_slice_layout = C_buffer.layout.slice(C_buffer.shape, C_buffer_region.region)
    # Extract pre-swizzle tile layout for descriptor offset computation
    if not a_is_tmem:
        A_slice_tile = (
            A_slice_layout.tile_layout
            if isinstance(A_slice_layout, ComposeLayout)
            else A_slice_layout
        )
    B_slice_tile = (
        B_slice_layout.tile_layout if isinstance(B_slice_layout, ComposeLayout) else B_slice_layout
    )

    assert len(C_extent) == 2 and len(A_extent) >= 2 and len(B_extent) >= 2, (
        "Only 2D C, A, B are supported for gemm"
    )

    def _mat_dim_vals(extent, name):
        """Extract the two non-unit dimension values from a GEMM operand extent."""
        vals = [int(e) for e in extent if not analyzer.can_prove_equal(e, 1)]
        assert len(vals) == 2, (
            f"Expected exactly 2 non-unit dims in {name}_extent {[int(e) for e in extent]}"
        )
        return vals[0], vals[1]

    M = int(C_extent[-2])
    N = int(C_extent[-1])
    is_2x2 = M == 64 and cta_group == 2

    # Majorness (a_mn_major / b_mn_major) is determined later by
    # compute_canonical_params via dual-atom matching on the physical
    # SMEM layout.  Extract dim extents here for cross-validation.
    # Use non-unit dims (not last-2) to handle unit dims in the middle
    # (e.g. region shape [M, 1, K]).
    A_dim2, A_dim1 = _mat_dim_vals(A_extent, "A")
    B_dim2, B_dim1 = _mat_dim_vals(B_extent, "B")

    # Compute SMEM descriptor parameters (swizzle mode, ldo, sdo) and infer
    # majorness by matching the sliced layout against both K-major atom
    # [8, T*s] and MN-major atom [T*s, 8] via is_tile_inner.
    #
    # Priority: MN-major atom match → definitively MN-major (column-major SMEM).
    # K-major atom match → use extent matching to determine semantic majorness,
    # since mma_shared_layout creates K-major layouts for both [M,K] and [K,M].
    def compute_canonical_params(buf, buf_region, dtype, is_transposed):
        """Compute descriptor parameters from buffer layout.

        Uses is_transposed (from op's transA/transB) to determine which
        atom orientation corresponds to K-major for this buffer:
        - transposed=False: buffer is [MN, K], K-major atom = [8, T*s]
        - transposed=True:  buffer is [K, MN], K-major atom = [T*s, 8]

        Then tries both atom orientations with is_tile_inner.  Whichever
        matches determines the physical majorness.

        Strips unit dims and passes 2D shapes to is_tile_inner on the
        sliced layout — handles >2D regions like [1, M, K] or [1, 1, M, K].

        Returns:
            Tuple of (swizzle_mode, ldo, sdo, is_mn_major).
        """
        region = list(buf_region.region)

        def _match(slice_layout, shape_2d):
            """Match ``slice_layout`` (of ``shape_2d``) against the swizzle atoms.

            Returns ``(swizzle_mode, ldo, sdo, is_mn_major)`` or ``None``.
            """

            def _try_atom(atom, atom_shape):
                if any(s % a != 0 for s, a in zip(shape_2d, atom_shape)):
                    return None
                atom_size = functools.reduce(operator.mul, atom_shape, 1)
                tiler = atom.is_tile_inner(slice_layout, shape_2d, atom_shape)
                if tiler is None:
                    return None
                tiler_shape = [s // a for s, a in zip(shape_2d, atom_shape)]
                tiler_grouped, seps = tiler.canonicalize().group(tiler_shape)
                elem_per_128b = 128 // tvm.DataType(dtype).bits

                # extent==1 leading dim -> unused LBO/SBO offset.
                def _atom_off(dim):
                    if int(dim.extent) == 1:
                        return 0
                    return (dim.stride * atom_size) // elem_per_128b

                ldo = _atom_off(tiler_grouped.shard[-1])
                sdo = _atom_off(tiler_grouped.shard[-2])
                return mode, ldo, sdo

            for mode in (
                SwizzleMode.SWIZZLE_128B_ATOM,
                SwizzleMode.SWIZZLE_64B_ATOM,
                SwizzleMode.SWIZZLE_32B_ATOM,
            ):
                swizzle_atom = mma_atom_layout(dtype, mode)
                base_shape = mma_atom_shape(dtype, mode)  # [8, T*s]
                swapped_shape = [base_shape[1], base_shape[0]]  # [T*s, 8]

                # MN-major atom: compose SwizzleLayout with stride-reversed TileLayout
                # so the first dim (T*s) is contiguous instead of the second.
                # Needed when the penultimate dim is physically contiguous.
                mn_tile = TileLayout(S[tuple(swapped_shape) : (1, swapped_shape[0])])
                mn_atom = ComposeLayout(swizzle_atom, mn_tile)

                # Determine K-major vs MN-major based on which dim is contiguous.
                # K-major: K dim contiguous (last dim for [MN,K], first dim for [K,MN])
                # MN-major: MN dim contiguous
                #
                # The plain swizzle_atom has last dim contiguous.
                # The mn_atom has first dim contiguous.
                #
                # For non-transposed [MN, K]: K is last dim
                #   - K-major = swizzle_atom with [8, T*s] (K contiguous in last dim)
                #   - MN-major = mn_atom with [T*s, 8] (MN contiguous in first dim)
                # For transposed [K, MN]: MN is last dim
                #   - K-major = mn_atom with [T*s, 8] (K contiguous in first dim)
                #   - MN-major = swizzle_atom with [8, T*s] (MN contiguous in last dim)
                if is_transposed:
                    candidates = [
                        (False, mn_atom, swapped_shape),  # K-major: K in first dim
                        (True, swizzle_atom, base_shape),  # MN-major: MN in last dim
                    ]
                else:
                    candidates = [
                        (False, swizzle_atom, base_shape),  # K-major: K in last dim
                        (True, mn_atom, swapped_shape),  # MN-major: MN in first dim
                    ]

                for is_mn_major, atom, atom_shape in candidates:
                    result = _try_atom(atom, atom_shape)
                    if result is not None:
                        sw, ldo_val, sdo_val = result
                        # shard[-1] = last-dim groups, shard[-2] = first-dim groups.
                        # LBO strides MN-groups for MN-major, K-groups for K-major.
                        # Non-transposed [MN,K]: last=K, first=MN → swap for MN-major
                        # Transposed [K,MN]: last=MN, first=K → swap for K-major
                        if is_mn_major != is_transposed:
                            ldo_val, sdo_val = sdo_val, ldo_val
                        return sw, ldo_val, sdo_val, is_mn_major
            return None

        # The MMA SMEM descriptor describes the buffer's *physical* swizzle, which
        # spans whole atoms and is a property of the buffer -- not of any sub-tile.
        # Round the contiguous (innermost) axis up to a swizzle-atom multiple before
        # matching, so the descriptor is always derived from a whole number of atoms:
        #   * for an already atom-aligned region (full-K, stride-axis P@V slices,
        #     non-swizzled buffers) this is a no-op -- desc_region == region, so the
        #     derived (swizzle, ldo, sdo, is_mn_major) are identical to matching the
        #     region directly;
        #   * for a sub-atom contiguous slice (fine K-major split-K, e.g.
        #     Asmem[..., :, lo:hi]) it rounds up to the smallest atom count covering
        #     the slice and describes it from the buffer origin.
        # Either way the actual [lo:hi] range is addressed by K_iters + the per-MMA
        # -tile 16B offset (from the *sliced* region in _a_operand / _b_desc_val),
        # not by this descriptor. Verified hardware-correct for every MMA_K-aligned
        # contiguous slice -- this is what enables fine K-major split-K.
        cax = len(region) - 1  # innermost (contiguous) axis
        elem_per_16b = 128 // DataType(dtype).bits
        phys_mode = get_swizzle_mode_from_layout(buf.layout)
        desc_region = list(region)
        if phys_mode in (
            SwizzleMode.SWIZZLE_128B_ATOM,
            SwizzleMode.SWIZZLE_64B_ATOM,
            SwizzleMode.SWIZZLE_32B_ATOM,
        ):
            atom_inner = mma_atom_shape(dtype, phys_mode)[-1]
            contig = int(region[cax].extent)
            rounded = ((contig + atom_inner - 1) // atom_inner) * atom_inner
            if rounded != contig:
                # Sub-atom contiguous slice. The per-tile 16B offset that locates the
                # slice start must be exact, so the start has to sit on a 16B
                # (= elem_per_16b element) boundary; otherwise reject rather than
                # silently mis-address.
                if not analyzer.can_prove_equal(
                    tvm.tirx.floormod(region[cax].min, elem_per_16b), 0
                ):
                    raise ValueError(
                        f"gemm_async: contiguous-axis slice start {region[cax].min} is not "
                        f"16B-aligned (={elem_per_16b} elements, dtype {dtype}). A sub-atom "
                        f"contiguous slice must start on a 16B boundary; otherwise keep that "
                        f"axis full and split on a stride/outer axis instead (lay the operand "
                        f"out MN-major)."
                    )
                desc_region[cax] = tvm.ir.Range.from_min_extent(0, rounded)

        slice_layout = buf.layout.slice(buf.shape, desc_region)
        # Strip unit dims to get the 2D matrix shape.
        shape_2d = [int(r.extent) for r in desc_region if int(r.extent) != 1]
        assert len(shape_2d) == 2, (
            f"Expected exactly 2 non-unit dims in region {[int(r.extent) for r in desc_region]}"
        )
        result = _match(slice_layout, shape_2d)
        if result is not None:
            return result

        # Genuinely unsupported: the layout doesn't tile any swizzle atom even at
        # full atom width. Actionable error (the old generic "no swizzle mode"
        # message read like a hard limit and was mistaken for one).
        hint = ""
        if phys_mode in (
            SwizzleMode.SWIZZLE_128B_ATOM,
            SwizzleMode.SWIZZLE_64B_ATOM,
            SwizzleMode.SWIZZLE_32B_ATOM,
        ):
            atom_inner = mma_atom_shape(dtype, phys_mode)[-1]
            atom_bytes = atom_inner * DataType(dtype).bits // 8
            hint = (
                f" The buffer is physically {phys_mode.name}-swizzled (contiguous atom "
                f"= {atom_inner} elements / {atom_bytes} B); the region's layout does not "
                f"tile it."
            )
        raise ValueError(
            f"gemm_async: no MMA SMEM descriptor matches region shape {shape_2d} "
            f"for dtype {dtype}.{hint}"
        )

    if a_is_tmem:
        # TMEM A: hardware requires transA=False (no transpose from TMEM)
        assert not transA, "tcgen05 schedule: transA must be False when A is in tmem"
        a_mn_major = False
    else:
        A_swizzle_mode, A_ldo, A_sdo, a_mn_major = compute_canonical_params(
            A_buffer, A_buffer_region, A_type, transA
        )
    B_swizzle_mode, B_ldo, B_sdo, b_mn_major = compute_canonical_params(
        B_buffer, B_buffer_region, B_type, transB
    )

    # Extract K from A dims using transA (shape order).
    # transA tells us which dim is K; a_mn_major tells us the layout orientation.
    # transA=False [M, K]: K = dim[-1]; transA=True [K, M]: K = dim[-2]
    K = A_dim2 if transA else A_dim1

    # tcgen05 MMA hardware constraints (MMA_K keyed on semantic dtype A_sem).
    if A_sem == "float4_e2m1fn":
        MMA_K = 64
    elif A_sem in ["float8_e4m3fn", "float8_e5m2"]:
        MMA_K = 32
    elif A_sem in ["tensor_float32", "tf32"]:
        MMA_K = 8
    else:  # float16, bfloat16
        MMA_K = 16
    MMA_N_MIN = 8 if cta_group == 1 else 16  # Minimum N dimension

    M_mma, N_mma = _choose_mma_tile(M, N, cta_group, MMA_N_MIN)
    M_tiles = M // M_mma
    N_tiles = N // N_mma
    K_iters = K // MMA_K
    N_mma_per_cta = N_mma // cta_group
    assert K % MMA_K == 0, f"tcgen05 schedule expected K % {MMA_K} == 0, got {K}"

    # Cross-validate A dimensions (shape order from transA)
    A_M = A_dim1 if transA else A_dim2
    assert A_M == M, f"tcgen05: A_M={A_M} doesn't match M={M} from C region"

    # Cross-validate K between A and B
    B_K = B_dim1 if not transB else B_dim2
    assert K == B_K, f"tcgen05: A_K={K} doesn't match B_K={B_K}"

    # Cross-validate B's N with C's N and cta_group
    B_N = B_dim2 if not transB else B_dim1
    assert B_N * cta_group == N, (
        f"tcgen05: B_N={B_N} * cta_group={cta_group}={B_N * cta_group} doesn't match N={N}"
    )

    # Validate SFA/SFB region shapes
    if is_block_scaled:
        assert SFA_rows == M, f"tcgen05: SFA rows={SFA_rows} must equal M={M}"
        assert SFB_rows >= N, f"tcgen05: SFB rows={SFB_rows} must be >= N={N}"
        sfa_epc = 32 // DataType(SFA_type).bits
        sfb_epc = 32 // DataType(SFB_type).bits
        valid_sfa_K = {sfa_sf_mma_k, sfa_sf_mma_k * K_iters, sfa_sf_mma_k * K_iters * sfa_epc}
        valid_sfb_K = {sfb_sf_mma_k, sfb_sf_mma_k * K_iters, sfb_sf_mma_k * K_iters * sfb_epc}
        assert SFA_K_total in valid_sfa_K, (
            f"tcgen05: SFA K extent={SFA_K_total} must be in {valid_sfa_K}"
        )
        assert SFB_K_total in valid_sfb_K, (
            f"tcgen05: SFB K extent={SFB_K_total} must be in {valid_sfb_K}"
        )

    # Check C's sliced layout, allow offset.
    # 4x1 layout (Layout D, M=128 identity): (M, N):(1@TLane, 1@TCol)
    # 2x2 layout: (M, 2, N//2):(1@TLane, 64@TLane, 1@TCol)
    # Layout F (M=64 scatter): the full TMEM buffer is shape (64, X) with the
    # scattered row→lane mapping from tmem_datapath_layout("F", 64, X). When
    # the user allocates with ``tmem_pool.alloc(..., datapath="F")`` and slices
    # the full row range, the slice layout structurally matches Layout F over
    # (M=64, N) — assert against that base instead of the Layout D identity.
    if is_2x2:
        N_half = N // 2
        base = TileLayout(S[(M, 2, N_half) : (1 @ TLane, 64 @ TLane, 1 @ TCol)])
    elif (
        M == 64
        and int(C_buffer.shape[0]) == 64
        and C_buffer.layout is not None
        and _layout_matches_datapath_f(C_buffer)
    ):
        base = tmem_datapath_layout("F", 64, N)
    else:
        base = TileLayout(S[(M, N) : (1 @ TLane, 1 @ TCol)])
    expected_c_layout = TileLayout.from_iters(
        base.shard, base.replica, C_slice_layout.offset
    ).canonicalize()
    tvm.ir.assert_structural_equal(C_slice_layout.canonicalize(), expected_c_layout)
    assert C_buffer.allocated_addr is not None
    tmem_addr = C_buffer.allocated_addr[0]
    tmem_offset_32b = C_slice_layout.offset.get(TCol, 0)

    # Validate TMEM A layout: (A_dim2, A_dim1):(1@TLane, 1@TCol)
    if a_is_tmem:
        A_tmem_base = TileLayout(S[(A_dim2, A_dim1) : (1 @ TLane, 1 @ TCol)])
        expected_a_layout = TileLayout.from_iters(
            A_tmem_base.shard, A_tmem_base.replica, A_slice_layout.offset
        ).canonicalize()
        tvm.ir.assert_structural_equal(A_slice_layout.canonicalize(), expected_a_layout)
        assert A_buffer.allocated_addr is not None, "TMEM A buffer must have allocated_addr"
        A_tmem_addr = A_buffer.allocated_addr[0]
        A_elem_per_32b = 32 // DataType(A_type).bits
        # TCol offset is in element units (not 32-bit columns) for sub-32-bit dtypes.
        # Convert to 32-bit column units for get_tmem_addr.
        A_tmem_offset_32b = A_slice_layout.offset.get(TCol, 0) // A_elem_per_32b

    # Convert accum to TIR bool outside the macro (TIR AST evaluator doesn't
    # support short-circuit evaluation, so accum.dtype inside macro would fail
    # when accum is a Python bool).
    if isinstance(accum, bool):
        accum_expr = tvm.tirx.const(int(accum), "bool")
    elif tvm.ir.is_prim_expr(accum) and accum.ty.dtype != "bool":
        accum_expr = tvm.tirx.Cast("bool", accum)
    else:
        accum_expr = accum

    # 16B element count for descriptor offset computation
    B_elem_per_16B = 128 // DataType(B_type).bits
    if not a_is_tmem:
        A_elem_per_16B = 128 // DataType(A_type).bits

    elect_pred = T.ptx.elect_sync() if warp_scope else True

    _SWIZZLE_TO_LAYOUT = {0: 0, 1: 6, 2: 4, 3: 2, 4: 1}
    _krp = Evaluate(tirx_op.tvm_kernel_replace_point())

    def _make_lo_uniform(desc):
        func_name = "smem_desc_make_lo_uniform_"
        source_code = f"""
        __forceinline__ __device__ void {func_name}(uint64_t* desc) {{
            SmemDescriptor* d = reinterpret_cast<SmemDescriptor*>(desc);
            d->lo = __shfl_sync(0xffffffff, d->lo, 0);
        }}
        """
        return T.cuda.func_call(
            func_name, T.address_of(desc), source_code=source_code, return_type="void"
        )

    def _make_desc(smem_buf, ldo, sdo, swizzle_val, name):
        desc_buf = tvm.tirx.decl_buffer((1,), "uint64", name=name, scope="local")
        encode_call = tvm.tirx.call_intrin(
            "",
            "tirx.ptx.tcgen05_encode_matrix_descriptor",
            tvm.tirx.address_of(desc_buf[0]),
            smem_buf.ptr_to([0] * len(smem_buf.shape)),
            ldo,
            sdo,
            swizzle_val,
        )
        wrap = SeqStmt(
            [
                AllocBuffer(desc_buf),
                Evaluate(encode_call),
                Evaluate(_make_lo_uniform(desc_buf[0])),
                _krp,
            ]
        )
        sctx.add_post_buffer_def_stmt(smem_buf, wrap)
        return desc_buf

    def _uniform_desc(smem_buf, off16, ldo, sdo, swizzle):
        layout = _SWIZZLE_TO_LAYOUT[int(swizzle)]
        const_hi = (int(sdo) & 0x3FFF) | (1 << 14) | (layout << 29)
        lo_const = (int(ldo) & 0x3FFF) << 16
        base_ptr = smem_buf.ptr_to([0] * len(smem_buf.shape))
        addr = T.ptr_byte_offset(base_ptr, off16 * 16, smem_buf.dtype)
        sa = T.bitwise_and(
            T.shift_right(T.cuda.cvta_generic_to_shared(addr), T.uint32(4)),
            T.uint32(0x3FFF),
        )
        lo = T.bitwise_or(T.uint32(lo_const), sa) if lo_const else sa
        return T.bitwise_or(T.shift_left(T.uint64(const_hi), T.uint64(32)), T.cast(lo, "uint64"))

    def _b_offset(ni, ki):
        B_linear = (
            ki * MMA_K * B_extent[-1] + ni * N_mma_per_cta
            if transB
            else ni * N_mma_per_cta * B_extent[-1] + ki * MMA_K
        )
        return tvm.tirx.floordiv(B_slice_tile.apply(B_linear)["m"], B_elem_per_16B)

    def _a_offset(mi, ki):
        A_linear = (
            ki * MMA_K * A_extent[-1] + mi * M_mma
            if transA
            else mi * M_mma * A_extent[-1] + ki * MMA_K
        )
        return tvm.tirx.floordiv(A_slice_tile.apply(A_linear)["m"], A_elem_per_16B)

    # smem_desc: "hoist" (default) or "recompute" per-MMA descriptor update.
    use_add = op_call.config.get("smem_desc", "hoist") != "recompute"
    descB_buf = (
        _make_desc(B_buffer, B_ldo, B_sdo, B_swizzle_mode.value, "descB") if use_add else None
    )
    A_use_add = use_add and not a_is_tmem
    descA_buf = (
        _make_desc(A_buffer, A_ldo, A_sdo, A_swizzle_mode.value, "descA") if A_use_add else None
    )
    B_use_add = use_add

    # Helper: compute B descriptor value for a given (ni, ki) tile
    def _b_desc_val(descB_in, ni, ki):
        B_offset = _b_offset(ni, ki)
        if B_use_add:
            return smem_desc_add_16B_offset(descB_buf[0], B_offset)
        return _uniform_desc(B_buffer, B_offset, B_ldo, B_sdo, B_swizzle_mode.value)

    # Helper: compute A operand (TMEM address or SMEM descriptor) for a given (mi, ki) tile
    def _a_operand(mi, ki, descA_in=None):
        if a_is_tmem:
            # A is [M, K] non-transposed: M→TLane (rows), K→TCol (cols)
            a_row = mi * M_mma
            a_col = A_tmem_offset_32b + ki * (MMA_K // A_elem_per_32b)
            return T.cuda.get_tmem_addr(A_tmem_addr, a_row, a_col)
        A_offset = _a_offset(mi, ki)
        if A_use_add:
            return smem_desc_add_16B_offset(descA_buf[0], A_offset)
        return _uniform_desc(A_buffer, A_offset, A_ldo, A_sdo, A_swizzle_mode.value)

    if is_block_scaled:
        # Compute per-ki SF element steps from region extents
        sfa_elems_per_ki = SFA_K_total // K_iters if K_iters > 0 else 0
        sfb_elems_per_ki = SFB_K_total // K_iters if K_iters > 0 else 0

        sfa_base = SFA_buffer.allocated_addr[0]
        sfb_base = SFB_buffer.allocated_addr[0]

        # Compute initial SFA/SFB addresses (for ki=0)
        # apply(0)["TCol"] at row 0 gives physical TCol offset
        sfa_tcol_0 = SFA_slice_layout.apply(0).get("TCol", 0)
        sfb_tcol_0 = SFB_slice_layout.apply(0).get("TCol", 0)
        SFA_init_addr = analyzer.simplify(
            sfa_base + tvm.tirx.floordiv(sfa_tcol_0, SFA_elem_per_col)
        )
        SFB_init_addr = analyzer.simplify(
            sfb_base + tvm.tirx.floordiv(sfb_tcol_0, SFB_elem_per_col)
        )

        # Rotate sf_id per ki when multiple ki share one SF column.
        needs_sf_id = sfa_sf_mma_k < SFA_elem_per_col and sfa_elems_per_ki > 0

    else:
        needs_sf_id = False

    # Physical TMEM columns per MMA N tile.
    # 2x2 layout (Layout B): each MMA tile spans N_mma/2 physical columns
    # and uses rows 64-127 for the other half.
    N_mma_phys_cols = N_mma // 2 if is_2x2 else N_mma

    # Build main_impl: descA_in is None when A is in TMEM (ignored by _a_operand).
    # fmt: off
    if is_block_scaled:
        @T.inline
        def main_impl(descA_in, descB_in, descI_in):
            for mi in T.unroll(M_tiles):
              for ni in T.unroll(N_tiles):
                for ki in T.unroll(K_iters):
                    # meta_var inlines operands into mma.block_scale (avoids LMEM temps).
                    a_val = T.meta_var(_a_operand(mi, ki, descA_in))
                    descB_val = T.meta_var(_b_desc_val(descB_in, ni, ki))
                    should_accum = T.meta_var(tvm.tirx.any(ki != 0, accum_expr))
                    sfa_linear = mi * M_mma * SFA_K_total + ki * sfa_elems_per_ki
                    sfb_linear = ni * N_mma_per_cta * SFB_K_total + ki * sfb_elems_per_ki
                    # Fold tcol to constants shared by SF addr and sf_id.
                    sfa_tcol = T.meta_var(analyzer.simplify(SFA_slice_layout.apply(sfa_linear).get("TCol", 0)))  # noqa: E501
                    sfb_tcol = T.meta_var(analyzer.simplify(SFB_slice_layout.apply(sfb_linear).get("TCol", 0)))  # noqa: E501
                    sfa_addr = T.meta_var(
                        analyzer.simplify(sfa_base + tvm.tirx.floordiv(sfa_tcol, SFA_elem_per_col))
                    )
                    sfb_addr = T.meta_var(
                        analyzer.simplify(sfb_base + tvm.tirx.floordiv(sfb_tcol, SFB_elem_per_col))
                    )
                    if needs_sf_id:
                        sf_id = T.meta_var(analyzer.simplify(tvm.tirx.floormod(sfa_tcol, SFA_elem_per_col)))  # noqa: E501
                        T.cuda.runtime_instr_desc(T.address_of(descI_in), sf_id)
                    tmem_col = T.meta_var(
                        tmem_offset_32b + ni * (N_mma_phys_cols // C_elem_per_32b)
                    )
                    if elect_pred:
                        T.ptx.tcgen05.mma.block_scale(
                            T.cuda.get_tmem_addr(tmem_addr, mi * M_mma, tmem_col),
                            a_val, descB_val,
                            sfa_addr, sfb_addr,
                            descI_in,
                            d_dtype=C_type, a_dtype=A_type, b_dtype=B_type,
                            sfa_dtype=SFA_type, sfb_dtype=SFB_type,
                            use_a_tmem=a_is_tmem, cta_group=cta_group,
                            enable_input_d=should_accum,
                        )
    else:
        # Wrap each per-MMA operand in ``T.meta_var`` so the parser inlines
        # the value directly into the ``T.ptx.tcgen05.mma`` call instead of
        # materializing it into a fresh ``alignas(64) T x[1]; x[0] = expr``
        # local. Without this wrap each unrolled MMA emits 4 throw-away
        # 1-element local arrays (``a_val_ptr``, ``descB_val_ptr``,
        # ``should_accum_ptr``, ``tmem_col_ptr``) which ptxas cannot fold
        # back into the operand and the resulting LMEM round-trips show up
        # on the fa4 hot path.
        @T.inline
        def main_impl(descA_in, descB_in, descI_in):
            for mi in T.unroll(M_tiles):
              for ni in T.unroll(N_tiles):
                for ki in T.unroll(K_iters):
                    a_val = T.meta_var(_a_operand(mi, ki, descA_in))
                    descB_val = T.meta_var(_b_desc_val(descB_in, ni, ki))
                    should_accum = T.meta_var(tvm.tirx.any(ki != 0, accum_expr))
                    tmem_col = T.meta_var(
                        tmem_offset_32b + ni * (N_mma_phys_cols // C_elem_per_32b)
                    )
                    if elect_pred:
                        T.ptx.tcgen05.mma(
                            T.cuda.get_tmem_addr(tmem_addr, mi * M_mma, tmem_col),
                            a_val, descB_val, descI_in,
                            d_dtype="float32", a_dtype=A_sem, b_dtype=B_sem,
                            use_a_tmem=a_is_tmem, cta_group=cta_group,
                            enable_input_d=should_accum,
                        )

    descA_val = None  # descriptors built per-MMA from SMEM addr via _uniform_desc

    if descI is not None and not needs_sf_id:
        @T.prim_func(check_well_formed=False)
        def impl():
            main_impl(descA_val, None, descI)
    elif descI is not None:
        # Local copy: main_impl rotates descI in-place per ki.
        @T.prim_func(check_well_formed=False)
        def impl():
            descI_local: T.uint32
            descI_local = descI
            main_impl(descA_val, None, descI_local)
    elif is_block_scaled:
        @T.prim_func(check_well_formed=False)
        def impl():
            descI_local: T.uint32
            T.ptx.tcgen05.encode_instr_descriptor_block_scaled(T.address_of(descI_local), d_dtype=C_type, a_dtype=A_type, b_dtype=B_type, sfa_dtype=SFA_type, sfb_dtype=SFB_type,  # noqa: E501, F821
                                                               sfa_tmem_addr=SFA_init_addr, sfb_tmem_addr=SFB_init_addr,  # noqa: E501
                                                               M=M_mma * cta_group, N=N_mma, K=MMA_K, trans_a=a_mn_major, trans_b=b_mn_major, n_cta_groups=cta_group)  # noqa: E501
            main_impl(descA_val, None, descI_local)  # noqa: F821
    else:
        # Pre-compute the dense instruction descriptor at dispatcher time so
        # the MMA's 4th operand is a literal ``uint32`` instead of a per-call
        # ``alignas(64) uint descI_local[1]; encode_instr_descriptor(...)``
        # block. The encoded value depends only on (M, N, dtype, transA,
        # transB) which are all constants here.
        descI_value = _encode_instr_descriptor_dense_uint32(
            M=M_mma * cta_group,
            N=N_mma,
            d_dtype="float32",
            a_dtype=A_sem,
            b_dtype=B_sem,
            trans_a=a_mn_major,
            trans_b=b_mn_major,
        )
        descI_const = tvm.tirx.const(descI_value, "uint32")

        @T.prim_func(check_well_formed=False)
        def impl():
            main_impl(descA_val, None, descI_const)
    # fmt: on

    return impl


# === Variant: gemm_async/tcgen05 (priority=10) ===
#
# When: gemm_async op at single-thread exec scope on Blackwell (SM100+).
# Requires A in smem (with TMA-compatible swizzle layout) or tmem, B in smem, accum in tmem.
#
# Before (TilePrimitiveCall — regular MMA):
#     Tx.gemm_async(C_tmem[0:64, 0:256], A_smem[0:64, 0:64], B_smem[0:256, 0:64])
#     # A: shared float16, B: shared float16, C: tmem float32
#
# After (encodes instruction descriptor + calls tcgen05.mma):
#     descI_local: uint32
#     T.ptx.tcgen05.encode_instr_descriptor(
#         &descI_local, C_type="f32", A_type="f16", B_type="f16",
#         M=64, N=256, MMA_K=64, transA=False, transB=True, cta_group=1)
#     T.ptx.tcgen05.mma(descA_buf[0], descB_buf[0], descI_local)
#
# Before (TilePrimitiveCall — block-scaled fp8 MMA):
#     Tx.gemm_async(C_tmem, A_smem, B_smem,
#                   scale_A=SFA_tmem, scale_B=SFB_tmem)
#     # A/B: shared float8_e4m3, SFA/SFB: tmem float8_e8m0fnu
#
# After (adds scale factor descriptors):
#     T.ptx.tcgen05.mma(descA, descB, descI,
#                        scale_A=sfA_desc, scale_B=sfB_desc)
#
# Scale factor layout (sf_tmem_layout) must match tcgen05 hardware requirements:
# rows = M or N, sf_mma_k = ceil(MMA_K / sf_block_size), specific TileLayout
# structure with direct_sum atom tiling.
@register_dispatch(
    "gemm_async",
    "cuda",
    variant="tcgen05",
    priority=10,
    when=[
        predicate(
            "single_thread_or_warp",
            lambda op, sctx: (
                single_thread(op, sctx) or sctx.is_warp,
                f"unsupported exec_scope {sctx.exec_scope}, expected single thread or warp scope",
            ),
        )
    ],
)
def gemm_async_dispatch_tcgen05(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return gemm_async_tcgen05_impl(op_call, sctx)
