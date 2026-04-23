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
# ruff: noqa: E501, E731, RUF005
# fmt: off

"""Shared TIR helpers used by KV-cache / attention kernels in this package.

This module consolidates constructs reused by the prefill/decode/paged/tree
attention kernels so each kernel file can focus on its own specialised logic.

Contents:
- Thread-limit checks (``get_max_num_threads_per_block``, ``check_thread_limits``)
- KV-cache enums (``AttnKind``, ``RopeMode``)
- Small TVMScript helpers (``_var``, ``_var_cpu``, ``_causal_mask``, ``_rope``)
- Length-info accessors for sliding-window-aware indexing
- Buffer allocators for the tiled online-softmax state used by every prefill kernel
- ``_make_prefill_macros`` — the ``@T.macro`` bundle invoked by the prefill kernels
- Tiling config (``_get_prefill_kernel_config``) and scheduling (``_schedule_prefill_kernel``)
"""

# pylint: disable=too-many-statements,too-many-arguments,invalid-name,line-too-long
import enum
import math
from typing import Any

import tvm
from tvm import s_tir, tirx
from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.target import Target

from .position_embedding import switch_rope_freq_func


def _var(dtype):
    return T.sblock_alloc_buffer((1,), dtype, scope="local")


def _var_cpu(dtype):
    return T.sblock_alloc_buffer((1,), dtype)


def get_max_num_threads_per_block(target: Target) -> int:
    """
    max(max_num_threads, max_threads_per_block); if latter does not exist, return max_num_threads.
    We add this method since some targets have both fields and `max_threads_per_block` is larger.
    """
    max_num_threads = int(target.attrs["max_num_threads"])
    max_threads_per_block = target.attrs.get("max_threads_per_block", None)
    if max_threads_per_block is None:
        return max_num_threads
    return max(max_num_threads, max_threads_per_block)


def check_thread_limits(target: Target, bdx: int, bdy: int, bdz: int, gdz: int):
    """
    Check whether max num threads exceeded given a target.

    Parameters
    ----------
    bdx: threadIdx.x
    bdy: threadIdx.y
    bdz: threadIdx.z
    gdz: blockIdx.z
    """
    max_num_threads_per_block = get_max_num_threads_per_block(target)

    assert bdx * bdy * bdz <= max_num_threads_per_block, (
        f"{target.kind} max num threads exceeded: {bdx}*{bdy}*{bdz}>{max_num_threads_per_block}"
    )

    if target.kind.name == "webgpu":
        # https://gpuweb.github.io/gpuweb/#dom-supported-limits-maxcomputeworkgroupsizez
        assert bdz <= 64, f"webgpu's threadIdx.z cannot exceed 64, but got bdz={bdz}"
        assert gdz == 1, f"webgpu's blockIdx.z should be 1, but got gdz={gdz}"


class AttnKind(enum.IntEnum):
    """The attention kind class.
    MHA denotes multi-head attention, multi-query attention or grouped query attention.
    MLA denotes multi-head latent attention.
    """

    MHA = 0
    MLA = 1
    MHA_SLIDING = 3


class RopeMode(enum.IntEnum):
    """The RoPE mode of the Paged KV cache.
    If it is none, the KV cache will not apply RoPE to q and k.
    If it is normal, RoPE will be applied to k before adding k to cache.
    Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
    """

    NONE = 0
    NORMAL = 1
    INLINE = 2


def _rope(buffer: T.Buffer, offset: tirx.Var, rotary_dim: int, theta: tirx.Var, scale: tirx.Var, indices: tuple[tirx.Var, ...], qkv_dtype: str, rope_scaling: dict[str, Any]):
    d = indices[-1]
    cos_freq, sin_freq, var_map = switch_rope_freq_func(rope_scaling)(offset * scale, d, rotary_dim, theta, "float32")
    cos = cos_freq * buffer[indices].astype("float32")
    sin = sin_freq * tirx.if_then_else(
        d < rotary_dim // 2,
        -buffer[indices[:-1] + (d + rotary_dim // 2,)],
        buffer[indices[:-1] + (d - rotary_dim // 2,)],
    ).astype("float32")
    expr = (cos + sin).astype(qkv_dtype)
    for var, value in var_map.items():
        expr = tirx.Let(var, value, expr)
    return expr


def _causal_mask(causal, row, col, kv_len, qo_len):
    return T.if_then_else(
        causal > 0,
        col < kv_len - qo_len + row + 1,
        col < kv_len,
    )


def _declare_length_info(var_length_info, batch_size, sliding_window, elem_offset):
    return (
        T.match_buffer(var_length_info, (3, batch_size), "int32", elem_offset=elem_offset)
        if sliding_window
        else T.match_buffer(var_length_info, (batch_size,), "int32", elem_offset=elem_offset)
    )


def _get_kv_chunk_len(num_pages, page_size, seq_id, length_info, sliding_window):
    if not sliding_window:
        return (num_pages - 1) * page_size + length_info[seq_id]
    # ((num_pages - 1) * page_size + last_page_len) - sliding_window_offset + sink_size
    return (num_pages - 1) * page_size + length_info[0, seq_id] - length_info[1, seq_id] + length_info[2, seq_id]


def _get_seq_offset(pos, seq_id, length_info, sliding_window):
    if not sliding_window:
        return pos
    # pos if pos < sink_size else pos - sink_size + sliding_window_offset
    return T.if_then_else(
        pos < length_info[2, seq_id],
        pos,
        pos - length_info[2, seq_id] + length_info[1, seq_id],
    )


def _alloc_softmax_state_buffers(tile_x, tile_z, bdx, num_warps):
    """Allocate the shared/local online-softmax working state used by every tiled prefill kernel.

    Returns ``(S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new)``.
    """
    S_smem = T.sblock_alloc_buffer((tile_x, tile_z), "float32", scope="shared")
    S_local = T.sblock_alloc_buffer((tile_x, tile_z), "float32", scope="local")
    m_smem = T.sblock_alloc_buffer((tile_x,), "float32", scope="shared")
    m_prev_smem = T.sblock_alloc_buffer((tile_x,), "float32", scope="shared")
    d_smem = T.sblock_alloc_buffer((tile_x,), "float32", scope="shared")
    md_shape = (math.ceil(tile_x / (bdx * num_warps)),)
    m_new = T.sblock_alloc_buffer(md_shape, "float32", scope="local")
    m_prev = T.sblock_alloc_buffer(md_shape, "float32", scope="local")
    d_new = T.sblock_alloc_buffer(md_shape, "float32", scope="local")
    return S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new


def _alloc_mha_qkvo_buffers(tile_x, tile_z, d_qk, d_v, dtype):
    """Allocate Q/K/V shared + O local buffers for standard MHA/GQA prefill kernels."""
    Q_smem = T.sblock_alloc_buffer((tile_x, d_qk), dtype, scope="shared")
    K_smem = T.sblock_alloc_buffer((tile_z, d_qk), dtype, scope="shared")
    V_smem = T.sblock_alloc_buffer((tile_z, d_v), dtype, scope="shared")
    O_local = T.sblock_alloc_buffer((tile_x, d_v), "float32", scope="local")
    return Q_smem, K_smem, V_smem, O_local


def _alloc_mla_qkvo_buffers(tile_x, tile_z, d_qk, d_latent, dtype):
    """Allocate Q + combined KV shared + O local for MLA prefill (V reuses the KV buffer)."""
    Q_smem = T.sblock_alloc_buffer((tile_x, d_qk), dtype, scope="shared")
    KV_smem = T.sblock_alloc_buffer((tile_z, d_qk), dtype, scope="shared")
    O_local = T.sblock_alloc_buffer((tile_x, d_latent), "float32", scope="local")
    return Q_smem, KV_smem, O_local


def _alloc_tile_walk_state():
    """Return (tile_id, batch_idx, batch_tiles, batch_rows, iterator, kv_chunk_len) int32 scalars for the paged/ragged/MLA tile-walk state machine."""
    return _var("int32"), _var("int32"), _var("int32"), _var("int32"), _var("int32"), _var("int32")


def _make_prefill_macros(tile_x, tile_y, tile_z, tile_o, bdx, num_warps, group_size):
    """Build @T.macro helpers shared across tiled online-softmax prefill kernels.

    Parameters
    ----------
    tile_x : int  # query/output row tile
    tile_y : int  # QK reduction dim (head_dim for MHA, d_qk for MLA/ragged)
    tile_z : int  # key/value column tile
    tile_o : int  # output/V column dim (d for MHA/sequence, d_v for ragged, d_latent for MLA)
    """
    @T.macro
    def init_states(
        m_smem: T.Buffer, d_smem: T.Buffer, O_local: T.Buffer, ty: T.int32, tx: T.int32,
    ):
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
            if row < tile_x:
                m_smem[row] = -5e4
                d_smem[row] = 1.0
        for li, lj in T.grid(tile_x, tile_o):
            with T.sblock("O_init"):
                i, j = T.axis.remap("SS", [li, lj])
                O_local[i, j] = 0.0
        T.tvm_storage_sync("shared")

    @T.macro
    def compute_s_gemm(
        Q_smem: T.Buffer, K_smem: T.Buffer, S_local: T.Buffer, S_smem: T.Buffer, sm_scale: T.float32,
    ):
        with T.sblock():
            for li, lj, lk in T.grid(tile_x, tile_z, tile_y):
                with T.sblock("S_gemm"):
                    i, j, k = T.axis.remap("SSR", [li, lj, lk])
                    with T.init():
                        S_local[i, j] = 0.0
                    S_local[i, j] += T.cast(Q_smem[i, k], "float32") * T.cast(K_smem[j, k], "float32") * sm_scale * math.log2(math.exp(1))
        T.tvm_storage_sync("shared")
        for li, lj in T.grid(tile_x, tile_z):
            with T.sblock("S_store"):
                i, j = T.axis.remap("SS", [li, lj])
                S_smem[i, j] = S_local[i, j]
        T.tvm_storage_sync("shared")

    @T.macro
    def softmax_update_causal(
        S_smem: T.Buffer, m_smem: T.Buffer, d_smem: T.Buffer, m_prev_smem: T.Buffer,
        m_new: T.Buffer, m_prev: T.Buffer, d_new: T.Buffer,
        ty: T.int32, tx: T.int32, LH_start: T.int32, L_kv_start: T.int32,
        causal: T.int32, kv_len: T.int32, qo_len: T.int32,
    ):
        # Phase 1: compute m_new = max(masked S over kv tile), d_new = d_prev * exp2(m_prev - m_new)
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
            if row < tile_x:
                with T.sblock("update1"):
                    m_prev[i] = m_smem[row]
                    m_new[i] = m_smem[row]
                    row_: T.int32 = (LH_start + row) // group_size
                    for j in T.serial(tile_z):
                        if _causal_mask(causal, row=row_, col=L_kv_start + j, kv_len=kv_len, qo_len=qo_len):
                            m_new[i] = T.max(m_new[i], S_smem[row, j])
                    d_new[i] = d_smem[row] * T.exp2(m_prev[i] - m_new[i])
        # Phase 2: exp-and-scale S_smem; masked-out entries use -inf
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
            with T.sblock("update"):
                for j in T.serial(tile_z):
                    # predicate sits inside loop so sync stays outside conditional branches
                    if row < tile_x:
                        row_: T.int32 = (LH_start + row) // group_size
                        if _causal_mask(causal, row=row_, col=L_kv_start + j, kv_len=kv_len, qo_len=qo_len):
                            S_smem[row, j] = T.exp2(S_smem[row, j] - m_new[i])
                        else:
                            S_smem[row, j] = T.exp2(-5e4 - m_new[i])
        # Phase 3: d_new += sum(S_smem[row, :]); write m/d/m_prev back to smem
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
            if row < tile_x:
                with T.sblock("update"):
                    for j in T.serial(tile_z):
                        d_new[i] += S_smem[row, j]
                    m_smem[row] = m_new[i]
                    d_smem[row] = d_new[i]
                    m_prev_smem[row] = m_prev[i]
        T.tvm_storage_sync("shared")

    @T.macro
    def compute_o_gemm(
        S_smem: T.Buffer, V_smem: T.Buffer, O_local: T.Buffer,
        m_prev_smem: T.Buffer, m_smem: T.Buffer,
    ):
        with T.sblock():
            for li, lj, lk in T.grid(tile_x, tile_o, tile_z):
                with T.sblock("O_gemm"):
                    i, j, k = T.axis.remap("SSR", [li, lj, lk])
                    with T.init():
                        O_local[i, j] *= T.exp2(m_prev_smem[i] - m_smem[i])
                    O_local[i, j] += S_smem[i, k] * T.cast(V_smem[k, j], "float32")

    @T.macro
    def paged_store_output_lse(
        output: T.Buffer, lse: T.Buffer, O_local: T.Buffer, m_smem: T.Buffer, d_smem: T.Buffer,
        q_indptr: T.Buffer, b_idx: T.int32, by: T.int32, LH_start: T.int32,
    ):
        """Paged-style (q_indptr-based) O_store + lse_store epilogue.

        Used by paged prefill, ragged prefill and MLA prefill. MLA passes ``by=0`` so
        the ``by * group_size`` term drops to zero at compile time.
        """
        for li, lj in T.grid(tile_x, tile_o):
            with T.sblock("O_store"):
                i, j = T.axis.remap("SS", [li, lj])
                cur_L: T.int32 = q_indptr[b_idx] + (LH_start + i) // group_size
                cur_H_qo: T.int32 = by * group_size + (LH_start + i) % group_size
                if cur_L < q_indptr[b_idx + 1]:
                    output[cur_L, cur_H_qo, j] = O_local[i, j] / d_smem[i]
        for li in T.grid(tile_x):
            with T.sblock("lse_store"):
                i = T.axis.remap("S", [li])
                cur_L: T.int32 = q_indptr[b_idx] + (LH_start + i) // group_size
                cur_H_qo: T.int32 = by * group_size + (LH_start + i) % group_size
                if cur_L < q_indptr[b_idx + 1]:
                    lse[cur_L, cur_H_qo] = m_smem[i] + T.log2(d_smem[i])

    @T.macro
    def advance_tile_batch(
        tile_id: T.Buffer, batch_idx: T.Buffer, batch_tiles: T.Buffer, batch_rows: T.Buffer,
        q_indptr: T.Buffer, batch_size: T.int32,
    ):
        """Advance tile_id/batch_idx past exhausted batches.

        After the loop, either batch_idx[0] >= batch_size (all tiles consumed) or
        tile_id[0] < batch_tiles[0] (the current batch still has work to do).
        """
        while tile_id[0] >= batch_tiles[0] and batch_idx[0] < batch_size:
            tile_id[0] -= batch_tiles[0]
            batch_idx[0] += 1
            if batch_idx[0] < batch_size:
                b_idx: T.int32 = batch_idx[0]
                batch_rows[0] = (q_indptr[b_idx + 1] - q_indptr[b_idx]) * group_size
                batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)

    @T.macro
    def softmax_update_valid_length(
        S_smem: T.Buffer, m_smem: T.Buffer, d_smem: T.Buffer, m_prev_smem: T.Buffer,
        m_new: T.Buffer, m_prev: T.Buffer, d_new: T.Buffer,
        ty: T.int32, tx: T.int32, LH_start: T.int32, L_kv_start: T.int32,
        valid_len: T.int32, qo_len: T.int32,
    ):
        # Same three-phase online softmax as softmax_update_causal but with a
        # per-batch right-padding mask in place of causal masking.
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
            if row < tile_x:
                with T.sblock("update1"):
                    m_prev[i] = m_smem[row]
                    m_new[i] = m_smem[row]
                    row_: T.int32 = (LH_start + row) // group_size
                    for j in T.serial(tile_z):
                        if tirx.And(tirx.And(row_ < qo_len, row_ < valid_len), L_kv_start + j < valid_len):
                            m_new[i] = T.max(m_new[i], S_smem[row, j])
                    d_new[i] = d_smem[row] * T.exp2(m_prev[i] - m_new[i])
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
            with T.sblock("update"):
                for j in T.serial(tile_z):
                    if row < tile_x:
                        row_: T.int32 = (LH_start + row) // group_size
                        if tirx.And(tirx.And(row_ < qo_len, row_ < valid_len), L_kv_start + j < valid_len):
                            S_smem[row, j] = T.exp2(S_smem[row, j] - m_new[i])
                        else:
                            S_smem[row, j] = T.exp2(-5e4 - m_new[i])
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
            if row < tile_x:
                with T.sblock("update"):
                    for j in T.serial(tile_z):
                        d_new[i] += S_smem[row, j]
                    m_smem[row] = m_new[i]
                    d_smem[row] = d_new[i]
                    m_prev_smem[row] = m_prev[i]
        T.tvm_storage_sync("shared")

    @T.macro
    def softmax_update_causal_padded_left(
        S_smem: T.Buffer, m_smem: T.Buffer, d_smem: T.Buffer, m_prev_smem: T.Buffer,
        m_new: T.Buffer, m_prev: T.Buffer, d_new: T.Buffer,
        ty: T.int32, tx: T.int32, LH_start: T.int32, L_kv_start: T.int32,
        valid_len: T.int32, qo_len: T.int32,
    ):
        # Three-phase online softmax with left-padding + causal mask. Real
        # queries/keys occupy [qo_len - valid_len, qo_len); causal keeps
        # col <= row within that valid range. Self-attention only (qo_len
        # == kv_len); the same `pad` expression gates both row and col.
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
            if row < tile_x:
                with T.sblock("update1"):
                    m_prev[i] = m_smem[row]
                    m_new[i] = m_smem[row]
                    row_: T.int32 = (LH_start + row) // group_size
                    pad: T.int32 = qo_len - valid_len
                    for j in T.serial(tile_z):
                        col_: T.int32 = L_kv_start + j
                        if tirx.And(tirx.And(row_ < qo_len, row_ >= pad), tirx.And(col_ >= pad, col_ <= row_)):
                            m_new[i] = T.max(m_new[i], S_smem[row, j])
                    d_new[i] = d_smem[row] * T.exp2(m_prev[i] - m_new[i])
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
            with T.sblock("update"):
                for j in T.serial(tile_z):
                    if row < tile_x:
                        row_: T.int32 = (LH_start + row) // group_size
                        pad: T.int32 = qo_len - valid_len
                        col_: T.int32 = L_kv_start + j
                        if tirx.And(tirx.And(row_ < qo_len, row_ >= pad), tirx.And(col_ >= pad, col_ <= row_)):
                            S_smem[row, j] = T.exp2(S_smem[row, j] - m_new[i])
                        else:
                            S_smem[row, j] = T.exp2(-5e4 - m_new[i])
        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
            if row < tile_x:
                with T.sblock("update"):
                    for j in T.serial(tile_z):
                        d_new[i] += S_smem[row, j]
                    m_smem[row] = m_new[i]
                    d_smem[row] = d_new[i]
                    m_prev_smem[row] = m_prev[i]
        T.tvm_storage_sync("shared")

    return init_states, compute_s_gemm, softmax_update_causal, compute_o_gemm, softmax_update_valid_length, advance_tile_batch, paged_store_output_lse, softmax_update_causal_padded_left


def _get_prefill_kernel_config(h_kv, h_q, d, dtype, target: Target):
    NUM_BLKS = 16
    LOAD_VEC = 8 // ((DataType(dtype).bits + 7) // 8)  # 8 bytes
    group_size = h_q // h_kv

    bdx = 32
    num_warps = 4
    tile_x, tile_y, tile_z = (
        64 // ((DataType(dtype).bits + 7) // 8) // max(d // 128, 1),
        d,
        64 // ((DataType(dtype).bits + 7) // 8) // max(d // 128, 1),
    )
    original_tile_y = tile_y
    original_tile_z = tile_z
    while (tile_x * tile_z) % (bdx * num_warps) != 0:
        tile_z += original_tile_z
    while (tile_x * tile_y) % (bdx * num_warps) != 0:
        tile_y += original_tile_y

    # Otherwise we would exceed maxComputeWorkgroupStorageSize
    if (
        target.kind.name == "webgpu"
        and ((d + 127) // 128) * ((DataType(dtype).bits + 15) // 16) >= 4
    ):
        tile_z = 8
        num_warps = 2
    if target.kind.name == "opencl" and (
        ("android" in str(target.host)) or ("adreno" in str(target.attrs))
    ):
        LOAD_VEC = 16 // ((DataType(dtype).bits + 7) // 8)  # 16 bytes
        NUM_BLKS = group_size * 8

    check_thread_limits(target, bdx=bdx, bdy=num_warps, bdz=1, gdz=1)

    return NUM_BLKS, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z


def _schedule_prefill_kernel(sch: s_tir.Schedule, load_vec, bdx, num_warps, tile_x, tile_y, tile_z, transform_k_load: bool, merged_qk_load: bool) -> tvm.s_tir.Schedule:
    get_extent = lambda *lps: [int(sch.get(lp).extent) for lp in lps]

    def get_vecsize(extent):
        return min(load_vec, (extent & ~(extent - 1)))

    def getxy_vecsize(x, y, t):
        assert (x * y) % t == 0
        return min(get_vecsize(y), get_vecsize(x * y // t))

    def get_tile_size(x, y, t):
        cnt = (x * y) // t
        assert (x * y) % t == 0
        tile_y = math.ceil(math.sqrt(cnt))
        while (cnt % tile_y != 0 or y % tile_y != 0 or x % (cnt // tile_y) != 0) and tile_y <= cnt:
            tile_y += 1
        assert tile_y <= cnt
        tile_x = cnt // tile_y
        return tile_x, tile_y

    def apply_to_qkv_load(sch: s_tir.Schedule, block):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        x_extent, y_extent = get_extent(loop_x, loop_y)
        vec_size = getxy_vecsize(x_extent, y_extent, bdx * num_warps)
        yo, yv = sch.split(loop_y, [None, vec_size])
        yo_extent = y_extent // vec_size
        tile_x, tile_y = get_tile_size(x_extent, yo_extent, (bdx * num_warps))
        xo, xi = sch.split(loop_x, [tile_x, None])
        yo, yi = sch.split(yo, [tile_y, None])
        sch.reorder(xi, yi, xo, yo)
        t = sch.fuse(xi, yi)
        ty, tx = sch.split(t, [num_warps, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(yv)

    def apply_to_so_ewise(sch: s_tir.Schedule, block, tile):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        yiv_extent = get_vecsize(tile[1])
        yio, yiv = sch.split(yi, [None, yiv_extent])
        sch.unroll(yio)
        sch.vectorize(yiv)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    def apply_to_gemm(sch: s_tir.Schedule, block, tile, r_len=16, k_major=False):
        loop_x, loop_y, loop_z = sch.get_loops(block)[-3:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

        ko, ki = sch.split(loop_z, factors=[None, r_len])
        if k_major:
            sch.reorder(ko, xi, yi, ki)
        else:
            sch.reorder(ko, ki, xi, yi)
        yiv_extent = get_vecsize(tile[1])
        yio, yiv = sch.split(yi, [None, yiv_extent])
        sch.unroll(yio)
        sch.vectorize(yiv)
        sch.unroll(xi)
        sch.decompose_reduction(block, ty)

    def apply_to_md(sch, block):
        loop = sch.get_loops(block)[-1]
        _, ty, tx = sch.split(loop, factors=[None, num_warps, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    if transform_k_load and not merged_qk_load:
        sch.transform_layout("K_load", ("write", 0), lambda i, j: (j, i))
    tile_s = get_tile_size(tile_x, tile_z, bdx * num_warps)
    tile_o = get_tile_size(tile_x, tile_y, bdx * num_warps)
    apply_to_gemm(sch, sch.get_sblock("S_gemm"), tile_s, k_major=True)
    apply_to_gemm(sch, sch.get_sblock("O_gemm"), tile_o, k_major=False)
    apply_to_so_ewise(sch, sch.get_sblock("S_store"), tile_s)
    apply_to_so_ewise(sch, sch.get_sblock("O_init"), tile_o)
    apply_to_so_ewise(sch, sch.get_sblock("O_store"), tile_o)
    apply_to_qkv_load(sch, sch.get_sblock("Q_load"))
    if not merged_qk_load:
        apply_to_qkv_load(sch, sch.get_sblock("K_load"))
        apply_to_qkv_load(sch, sch.get_sblock("V_load"))
    else:
        apply_to_qkv_load(sch, sch.get_sblock("KV_load"))
    apply_to_md(sch, sch.get_sblock("lse_store"))
    return sch
