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
# ruff: noqa: E501
# fmt: off

"""Prefill attention kernels for (paged/ragged/MLA/dense) KV storage.

All of the ``@T.prim_func`` factories below share the same online-softmax
skeleton that is built up from ``@T.macro`` helpers in
``_kernel_common._make_prefill_macros``. Each kernel only supplies the
K/V loading path that is specific to its storage layout.
"""

# pylint: disable=too-many-statements,too-many-arguments,invalid-name,line-too-long
import math
from typing import Any, Literal

import tvm
from tvm import tirx
from tvm.script import tirx as T
from tvm.target import Target

from ._kernel_common import (
    _alloc_mha_qkvo_buffers,
    _alloc_mla_qkvo_buffers,
    _alloc_softmax_state_buffers,
    _alloc_tile_walk_state,
    _causal_mask,
    _declare_length_info,
    _get_kv_chunk_len,
    _get_prefill_kernel_config,
    _get_seq_offset,
    _make_prefill_macros,
    _rope,
    _schedule_prefill_kernel,
)


def _attention_prefill_cpu(
    h_kv, h_q, d, dtype, sliding_window: bool, rope_scaling: dict[str, Any], page_size: int = 16
):
    global_symbol = "batch_prefill_paged_kv_cpu"
    if sliding_window:
        global_symbol += "_sliding_window"

    group_size = h_q // h_kv

    # pylint: disable=too-many-branches
    @T.prim_func
    def batch_prefill_paged_kv_cpu(
        var_q: T.handle, # [total_len, h_q, d]
        var_q_indptr: T.handle, # [batch_size + 1]
        var_pages: T.handle, # [max_num_pages, 2, h_kv, page_size, d]
        var_page_indptr: T.handle, # [batch_size + 1]
        var_page_values: T.handle, # [nnz_pages]
        var_length_info: T.handle, # [b] when sliding window = False, or otherwise [3, b]
        var_k_rope_pos_offset: T.handle, # [b]
        var_q_rope_position: T.handle, # [total_len]
        var_output: T.handle, # [total_len, h_q, d]
        var_lse: T.handle, # [total_len, h_q]
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
    ):
        T.func_attr({"global_symbol": global_symbol})
        batch_size = T.int32(is_size_var=True)
        total_len = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)
        q_indptr_elem_offset = T.int32(is_size_var=True)
        page_indptr_elem_offset = T.int32(is_size_var=True)
        page_values_elem_offset = T.int32(is_size_var=True)
        k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        length_info_elem_offset = T.int32(is_size_var=True)

        q = T.match_buffer(var_q, (total_len, h_q, d), dtype)
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
        pages = T.match_buffer(var_pages, (max_num_pages, 2, h_kv, page_size, d), dtype)
        page_indptr = T.match_buffer(var_page_indptr, (batch_size + 1,), "int32", elem_offset=page_indptr_elem_offset)
        page_values = T.match_buffer(var_page_values, (nnz_pages,), "int32", elem_offset=page_values_elem_offset)
        k_rope_pos_offset = T.match_buffer(var_k_rope_pos_offset, (batch_size,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
        q_rope_position = T.match_buffer(var_q_rope_position, (total_len,), "int32", elem_offset=q_rope_position_elem_offset)
        output = T.match_buffer(var_output, (total_len, h_q, d), dtype)
        lse = T.match_buffer(var_lse, (total_len, h_q), "float32")  # pylint: disable=unused-variable
        # The length information of the sequences.
        # - It is in shape `(3, batch_size)` when sliding window is enabled.
        #   For a sequence "i", location
        #   - "(0, i)" is the number of KV slots used in the last page of the seq ("last_page_len"),
        #   - "(1, i)" is the starting offset of the sliding window in the seq,
        #   - "(2, i)" is the attn sink length of the sequence.
        # - It is in shape `(batch_size,)` when sliding window is disabled,
        #   denoting the "last_page_len".
        length_info = _declare_length_info(var_length_info, batch_size, sliding_window, length_info_elem_offset)


        for h_qo in T.serial(h_q):
            for b_idx in T.serial(batch_size):
                with T.sblock("attn"):
                    O_local = T.sblock_alloc_buffer((d, ), "float32")
                    Q_local = T.sblock_alloc_buffer((d, ), "float32")
                    K_local = T.sblock_alloc_buffer((d, ), "float32")
                    V_local = T.sblock_alloc_buffer((d, ), "float32")

                    kv_chunk_len = T.sblock_alloc_buffer((1, ), "int32")

                    m_val = T.sblock_alloc_buffer((1, ), "float32")
                    new_m = T.sblock_alloc_buffer((1, ), "float32")
                    d_val = T.sblock_alloc_buffer((1, ), "float32")
                    S_val = T.sblock_alloc_buffer((1, ), "float32")
                    scale_O = T.sblock_alloc_buffer((1, ), "float32")
                    factor = T.sblock_alloc_buffer((1, ), "float32")
                    cur_page_indptr_begin: T.int32 = page_indptr[b_idx]
                    cur_page_indptr_end: T.int32 = page_indptr[b_idx + 1]
                    #max_kv_len: T.int32 = max_num_pages * page_size
                    kv_chunk_len[0] = T.if_then_else(
                        cur_page_indptr_begin != cur_page_indptr_end,
                        _get_kv_chunk_len(cur_page_indptr_end - cur_page_indptr_begin, page_size, b_idx, length_info, sliding_window),
                        0
                    )


                    for q_idx in T.serial(q_indptr[b_idx + 1] - q_indptr[b_idx]):
                        #init m, d, O
                        m_val[0] = -5e4
                        d_val[0] = 1.0
                        for d_idx in T.serial(d):
                            O_local[d_idx] = 0.0
                        curl_q: T.int32 = q_indptr[b_idx] + q_idx

                        for d_idx in T.serial(d):

                            Q_local[d_idx] = T.if_then_else(
                                rotary_mode == 1,
                                _rope(q, q_rope_position[curl_q], d, rope_theta, rope_scale, (curl_q, h_qo, d_idx), dtype, rope_scaling),
                                q[curl_q, h_qo, d_idx]
                            )
                        for row_idx in T.serial(max_num_pages * page_size):
                            if row_idx < kv_chunk_len[0]:
                                # seq_offset: T.int32(is_size_var=True) = _get_seq_offset(row_idx, b_idx, length_info, sliding_window)
                                #seq_offset: T.int32(is_size_var=True) = row_idx
                                page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + (_get_seq_offset(row_idx, b_idx, length_info, sliding_window) // page_size)]
                                page_offset: T.int32(is_size_var=True) = _get_seq_offset(row_idx, b_idx, length_info, sliding_window) % page_size

                                # Load KV
                                for d_idx in T.serial(d):
                                    K_local[d_idx] = T.if_then_else(
                                        rotary_mode == 1,
                                        _rope(pages, k_rope_pos_offset[b_idx] + row_idx, d, rope_theta, rope_scale, (page_no, 0, h_qo // group_size, page_offset, d_idx), dtype, rope_scaling),
                                        pages[page_no, 0, h_qo // group_size, page_offset, d_idx]
                                    )
                                    V_local[d_idx] = pages[page_no, 1, h_qo // group_size, page_offset, d_idx]

                                # Compute S
                                # Q[i] * K[i] * sm_scale
                                S_val[0] = 0.0
                                for d_idx in T.serial(d):
                                    S_val[0] += Q_local[d_idx] * K_local[d_idx]
                                S_val[0] *= sm_scale * math.log2(math.exp(1))

                                # update m_val, d_val , O_local
                                if _causal_mask(causal,
                                    row=q_idx,
                                    col=row_idx,
                                    kv_len=kv_chunk_len[0],
                                    qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx]):
                                    new_m[0] = T.max(m_val[0], S_val[0])
                                else:
                                    S_val[0] = -5e4
                                # update d_val
                                d_val[0] *= T.exp2(m_val[0] - new_m[0])
                                d_val[0] += T.exp2(S_val[0] - new_m[0])

                                # restore O_local then update O_local
                                scale_O[0] = T.exp2(m_val[0] - new_m[0])
                                m_val[0] = new_m[0]
                                factor[0] = T.exp2(S_val[0] - m_val[0])
                                for d_idx in T.serial(d):
                                    O_local[d_idx] = O_local[d_idx] * scale_O[d_idx]


                                for d_idx in T.serial(d):
                                    O_local[d_idx] += V_local[d_idx] * factor[0]
                        # Store Output
                        for d_idx in T.serial(d):
                            O_local[d_idx] = O_local[d_idx] /d_val[0]
                            output[curl_q, h_qo, d_idx] = O_local[d_idx]
                        lse[curl_q, h_qo] = m_val[0] + T.log2(d_val[0])
    return batch_prefill_paged_kv_cpu


def _attention_prefill(h_kv, h_q, d, dtype, sliding_window: bool, rope_scaling: dict[str, Any], target: Target, page_size: int = 16):
    NUM_BLKS, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z = _get_prefill_kernel_config(h_kv, h_q, d, dtype, target)

    global_symbol = "batch_prefill_paged_kv"
    if sliding_window:
        global_symbol += "_sliding_window"

    init_states, compute_s_gemm, softmax_update_causal, compute_o_gemm, _, advance_tile_batch, paged_store_output_lse, *_ = _make_prefill_macros(tile_x, tile_y, tile_z, tile_y, bdx, num_warps, group_size)

    # pylint: disable=too-many-branches
    @T.prim_func
    def batch_prefill_paged_kv(
        var_q: T.handle, # [total_len, h_q, d]
        var_q_indptr: T.handle, # [batch_size + 1]
        var_pages: T.handle, # [max_num_pages, 2, h_kv, page_size, d]
        var_page_indptr: T.handle, # [batch_size + 1]
        var_page_values: T.handle, # [nnz_pages]
        var_length_info: T.handle, # [b] when sliding window = False, or otherwise [3, b]
        var_k_rope_pos_offset: T.handle, # [b]
        var_q_rope_position: T.handle, # [total_len]
        var_output: T.handle, # [total_len, h_q, d]
        var_lse: T.handle, # [total_len, h_q]
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
    ):
        T.func_attr({"global_symbol": global_symbol})
        batch_size = T.int32(is_size_var=True)
        total_len = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)
        pages_elem_offset = T.int64(is_size_var=True)
        q_indptr_elem_offset = T.int32(is_size_var=True)
        page_indptr_elem_offset = T.int32(is_size_var=True)
        page_values_elem_offset = T.int32(is_size_var=True)
        k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        length_info_elem_offset = T.int32(is_size_var=True)

        q = T.match_buffer(var_q, (total_len, h_q, d), dtype)
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
        pages = T.match_buffer(var_pages, (max_num_pages, 2, h_kv, page_size, d), dtype, elem_offset=pages_elem_offset)
        page_indptr = T.match_buffer(var_page_indptr, (batch_size + 1,), "int32", elem_offset=page_indptr_elem_offset)
        page_values = T.match_buffer(var_page_values, (nnz_pages,), "int32", elem_offset=page_values_elem_offset)
        k_rope_pos_offset = T.match_buffer(var_k_rope_pos_offset, (batch_size,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
        q_rope_position = T.match_buffer(var_q_rope_position, (total_len,), "int32", elem_offset=q_rope_position_elem_offset)
        output = T.match_buffer(var_output, (total_len, h_q, d), dtype)
        lse = T.match_buffer(var_lse, (total_len, h_q), "float32")  # pylint: disable=unused-variable
        # The length information of the sequences.
        # - It is in shape `(3, batch_size)` when sliding window is enabled.
        #   For a sequence "i", location
        #   - "(0, i)" is the number of KV slots used in the last page of the seq ("last_page_len"),
        #   - "(1, i)" is the starting offset of the sliding window in the seq,
        #   - "(2, i)" is the attn sink length of the sequence.
        # - It is in shape `(batch_size,)` when sliding window is disabled,
        #   denoting the "last_page_len".
        length_info = _declare_length_info(var_length_info, batch_size, sliding_window, length_info_elem_offset)

        # kernel code
        for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
            for lby in T.thread_binding(h_kv, thread="blockIdx.y"):
                for lty in T.thread_binding(num_warps, thread="threadIdx.y"):
                    for ltx in T.thread_binding(bdx, thread="threadIdx.x"):
                        with T.sblock("attn"):
                            bx, by, ty, tx = T.axis.remap("SSSS", [lbx, lby, lty, ltx])
                            T.reads()
                            T.writes()
                            tile_id, batch_idx, batch_tiles, batch_rows, iterator, kv_chunk_len = _alloc_tile_walk_state()
                            Q_smem, K_smem, V_smem, O_local = _alloc_mha_qkvo_buffers(tile_x, tile_z, d, d, dtype)
                            S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new = (
                                _alloc_softmax_state_buffers(tile_x, tile_z, bdx, num_warps)
                            )

                            tile_id[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = (q_indptr[1] - q_indptr[0]) * group_size
                            batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)
                            while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                advance_tile_batch(tile_id, batch_idx, batch_tiles, batch_rows, q_indptr, batch_size)

                                if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                    b_idx: T.int32 = batch_idx[0]
                                    LH_start: T.int32 = tile_id[0] * tile_x
                                    q_indptr_val: T.int32 = q_indptr[b_idx]

                                    cur_page_indptr_begin: T.int32 = page_indptr[b_idx]
                                    cur_page_indptr_end: T.int32 = page_indptr[b_idx + 1]
                                    kv_chunk_len[0] = T.if_then_else(
                                        cur_page_indptr_begin != cur_page_indptr_end,
                                        _get_kv_chunk_len(cur_page_indptr_end - cur_page_indptr_begin, page_size, b_idx, length_info, sliding_window),
                                        0
                                    )
                                    T.tvm_storage_sync("shared")

                                    init_states(m_smem, d_smem, O_local, ty, tx)

                                    # Load Q from gmem to smem
                                    for li, lj in T.grid(tile_x, tile_y):
                                        with T.sblock("Q_load"):
                                            i, j = T.axis.remap("SS", [li, lj])
                                            T.reads()
                                            T.writes()
                                            cur_L = q_indptr_val + (LH_start + i) // group_size
                                            cur_H_qo = by * group_size + (LH_start + i) % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                Q_smem[i, j] = T.if_then_else(
                                                    rotary_mode == 1,
                                                    _rope(q, q_rope_position[cur_L], d, rope_theta, rope_scale, (cur_L, cur_H_qo, j), dtype, rope_scaling),
                                                    q[cur_L, cur_H_qo, j]
                                                )
                                            else:
                                                Q_smem[i, j] = 0.0
                                    T.tvm_storage_sync("shared")

                                    for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_z)):
                                        L_kv_start: T.int32 = iterator * tile_z
                                        for lz, ly in T.grid(tile_z, tile_y):
                                            with T.sblock("K_load"):
                                                i, j = T.axis.remap("SS", [lz, ly])
                                                T.reads()
                                                T.writes()
                                                cur_L = L_kv_start + i
                                                if cur_L < kv_chunk_len[0]:
                                                    seq_offset: T.int32(is_size_var=True) = _get_seq_offset(cur_L, b_idx, length_info, sliding_window)  # type: ignore
                                                    page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + T.floordiv(seq_offset, page_size)]  # type: ignore
                                                    page_offset: T.int32(is_size_var=True) = T.floormod(seq_offset, page_size)  # type: ignore
                                                    K_smem[i, j] = T.if_then_else(
                                                        rotary_mode == 1,
                                                        _rope(pages, k_rope_pos_offset[b_idx] + cur_L, d, rope_theta, rope_scale, (page_no, 0, by, page_offset, j), dtype, rope_scaling),
                                                        pages[page_no, 0, by, page_offset, j]
                                                    )
                                                else:
                                                    K_smem[i, j] = 0.0
                                        T.tvm_storage_sync("shared")
                                        for lz, ly in T.grid(tile_z, tile_y):
                                            with T.sblock("V_load"):
                                                i, j = T.axis.remap("SS", [lz, ly])
                                                T.reads()
                                                T.writes()
                                                cur_L = L_kv_start + i
                                                if cur_L < kv_chunk_len[0]:
                                                    seq_offset: T.int32(is_size_var=True) = _get_seq_offset(cur_L, b_idx, length_info, sliding_window)  # type: ignore
                                                    page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + T.floordiv(seq_offset, page_size)]  # type: ignore
                                                    page_offset: T.int32(is_size_var=True) = T.floormod(seq_offset, page_size)  # type: ignore
                                                    V_smem[i, j] = pages[page_no, 1, by, page_offset, j]
                                                else:
                                                    V_smem[i, j] = 0.0
                                        T.tvm_storage_sync("shared")

                                        compute_s_gemm(Q_smem, K_smem, S_local, S_smem, sm_scale)
                                        softmax_update_causal(S_smem, m_smem, d_smem, m_prev_smem, m_new, m_prev, d_new, ty, tx, LH_start, L_kv_start, causal, kv_chunk_len[0], q_indptr[b_idx + 1] - q_indptr[b_idx])
                                        compute_o_gemm(S_smem, V_smem, O_local, m_prev_smem, m_smem)

                                    paged_store_output_lse(output, lse, O_local, m_smem, d_smem, q_indptr, b_idx, by, LH_start)

                                    # move to next tile
                                    tile_id[0] += NUM_BLKS
    # pylint: enable=too-many-branches
    sch = tvm.s_tir.Schedule(batch_prefill_paged_kv)
    sch = _schedule_prefill_kernel(
        sch, LOAD_VEC, bdx, num_warps, tile_x, tile_y, tile_z, False, False
    )
    return sch.mod["main"].with_attr("tirx.is_scheduled", True)



def _attention_sequence_prefill(h_kv, h_q, d, dtype, target: Target, causal=0, sm_scale=1.0):
    _, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z = _get_prefill_kernel_config(h_kv, h_q, d, dtype, target)
    init_states, compute_s_gemm, softmax_update_causal, compute_o_gemm, *_ = _make_prefill_macros(tile_x, tile_y, tile_z, tile_y, bdx, num_warps, group_size)

    @T.prim_func
    def batch_sequence_prefill_kv(  # pylint: disable=too-many-branches
        var_q: T.handle, # [total_len, h_q, d]
        var_k: T.handle, # [total_len, h_kv, d]
        var_v: T.handle, # [total_len, h_kv, d]
        var_output: T.handle, # [total_len, h_q, d]
        var_lse: T.handle # [total_len, h_q]
    ):
        batch_size = T.int32(is_size_var=True)
        qo_len = T.int32(is_size_var=True)
        kv_len = T.int32(is_size_var=True)
        q = T.match_buffer(var_q, (batch_size, qo_len, h_q, d), dtype)
        k = T.match_buffer(var_k, (batch_size, kv_len, h_kv, d), dtype)
        v = T.match_buffer(var_v, (batch_size, kv_len, h_kv, d), dtype)
        output = T.match_buffer(var_output, (batch_size, qo_len, h_q, d), dtype)
        lse = T.match_buffer(var_lse, (batch_size, qo_len, h_q), dtype)  # pylint: disable=unused-variable

        batch_tiles: T.int32 = T.ceildiv(qo_len * group_size, tile_x)

        # kernel code
        for lbx in T.thread_binding(T.cast(batch_size, "int32") * batch_tiles, thread="blockIdx.x"):
            for lby in T.thread_binding(h_kv, thread="blockIdx.y"):
                for lty in T.thread_binding(num_warps, thread="threadIdx.y"):
                    for ltx in T.thread_binding(bdx, thread="threadIdx.x"):
                        with T.sblock("attn"):
                            vbx, by, ty, tx = T.axis.remap("SSSS", [lbx, lby, lty, ltx])
                            T.reads()
                            T.writes()

                            Q_smem, K_smem, V_smem, O_local = _alloc_mha_qkvo_buffers(tile_x, tile_z, d, d, dtype)
                            S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new = (
                                _alloc_softmax_state_buffers(tile_x, tile_z, bdx, num_warps)
                            )

                            b_idx: T.int32 = vbx // batch_tiles
                            tile_id: T.int32 = vbx % batch_tiles
                            LH_start: T.int32 = tile_id * tile_x
                            T.tvm_storage_sync("shared")

                            init_states(m_smem, d_smem, O_local, ty, tx)

                            # Load Q from gmem to smem
                            for li, lj in T.grid(tile_x, tile_y):
                                with T.sblock("Q_load"):
                                    i, j = T.axis.remap("SS", [li, lj])
                                    T.reads()
                                    T.writes()
                                    cur_L = (LH_start + i) // group_size
                                    cur_H_qo = by * group_size + (LH_start + i) % group_size
                                    if cur_L < qo_len:
                                        Q_smem[i, j] = q[b_idx, cur_L, cur_H_qo, j]
                                    else:
                                        Q_smem[i, j] = 0.0
                            T.tvm_storage_sync("shared")

                            for iterator in T.serial(T.ceildiv(kv_len, tile_z)):
                                L_kv_start: T.int32 = iterator * tile_z
                                L_kv_base: T.int32 = 0
                                for lz, ly in T.grid(tile_z, tile_y):
                                    with T.sblock("K_load"):
                                        i, j = T.axis.remap("SS", [lz, ly])
                                        T.reads()
                                        T.writes()
                                        cur_L = L_kv_start + i
                                        if cur_L < kv_len:
                                            K_smem[i, j] = k[
                                                b_idx, L_kv_base + cur_L, by, j
                                            ]
                                        else:
                                            K_smem[i, j] = 0.0
                                T.tvm_storage_sync("shared")
                                for lz, ly in T.grid(tile_z, tile_y):
                                    with T.sblock("V_load"):
                                        i, j = T.axis.remap("SS", [lz, ly])
                                        T.reads()
                                        T.writes()
                                        cur_L = L_kv_start + i
                                        if cur_L < kv_len:
                                            V_smem[i, j] = v[b_idx, L_kv_base + cur_L, by, j]
                                        else:
                                            V_smem[i, j] = 0.0
                                T.tvm_storage_sync("shared")

                                compute_s_gemm(Q_smem, K_smem, S_local, S_smem, sm_scale)
                                softmax_update_causal(S_smem, m_smem, d_smem, m_prev_smem, m_new, m_prev, d_new, ty, tx, LH_start, L_kv_start, causal, kv_len, qo_len)
                                compute_o_gemm(S_smem, V_smem, O_local, m_prev_smem, m_smem)

                            # Store O from smem to gmem
                            for li, lj in T.grid(tile_x, tile_y):
                                with T.sblock("O_store"):
                                    i, j = T.axis.remap("SS", [li, lj])
                                    cur_L: T.int32 = 0 + (LH_start + i) // group_size
                                    cur_H_qo: T.int32 = by * group_size + (LH_start + i) % group_size
                                    if cur_L < qo_len:
                                        output[b_idx, cur_L, cur_H_qo, j] = O_local[i, j] / d_smem[i]

                            # Store LSE to gmem
                            for li in T.grid(tile_x):
                                with T.sblock("lse_store"):
                                    i = T.axis.remap("S", [li])
                                    cur_L: T.int32 = 0 + (LH_start + i) // group_size
                                    cur_H_qo: T.int32 = by * group_size + (LH_start + i) % group_size
                                    if cur_L < qo_len:
                                        lse[b_idx, cur_L, cur_H_qo] = m_smem[i] + T.log2(d_smem[i])

    # pylint: enable=too-many-branches
    sch = tvm.s_tir.Schedule(batch_sequence_prefill_kv)
    sch = _schedule_prefill_kernel(sch, LOAD_VEC, bdx, num_warps, tile_x, tile_y, tile_z, False, False)
    return sch.mod["main"].with_attr("tirx.is_scheduled", True)



def _attention_sequence_prefill_with_mask(
    h_kv, h_q, d, dtype, target: Target, sm_scale=1.0, *,
    mask_mode: Literal["padded", "causal_padded_left"] = "padded",
):
    """Tiled sequence prefill kernel with a per-batch padding mask.

    Supports two mask regimes selected by ``mask_mode``:

    * ``"padded"`` (default) — bidirectional attention with right-padding.
      For batch ``b``, positions ``[0, valid_lens[b])`` are real and
      positions ``[valid_lens[b], seq_len)`` are padding. This is the
      encoder-style batch regime.
    * ``"causal_padded_left"`` — causal attention with left-padding. For
      batch ``b``, positions ``[seq_len - valid_lens[b], seq_len)`` are
      real and positions ``[0, seq_len - valid_lens[b])`` are padding;
      the causal constraint additionally keeps ``col <= row`` within the
      valid range. This is the decoder-embedding batch regime, where
      last-token pooling is a cheap slice of the final row.

    In both modes the kernel takes an extra ``valid_lens`` buffer of shape
    ``(batch_size,)`` and applies the mask inside the QKV load path and the
    online softmax update, so no explicit mask tensor broadcast or additive
    bias is needed on the host side. Padding queries and keys/values are
    zeroed at load time; masked ``(row, col)`` pairs are excluded from the
    max/sum of the online softmax via a ``-inf`` slot. ``valid_len`` is the
    per-batch real token count shared by Q and K/V; cross-attention with
    independent Q/K validity is out of scope.
    """
    _, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z = _get_prefill_kernel_config(h_kv, h_q, d, dtype, target)
    (
        init_states, compute_s_gemm, _, compute_o_gemm, softmax_update_valid_length,
        _, _, softmax_update_causal_padded_left,
    ) = _make_prefill_macros(tile_x, tile_y, tile_z, tile_y, bdx, num_warps, group_size)

    softmax_update = (
        softmax_update_valid_length
        if mask_mode == "padded"
        else softmax_update_causal_padded_left
    )

    def _q_row_valid(row, valid_len, qo_len):
        # Row-validity predicate for Q load (TIR expression); mask_mode is
        # captured at closure time so the prim_func body stays specialised.
        if mask_mode == "padded":
            return tirx.And(row < qo_len, row < valid_len)
        pad = qo_len - valid_len
        return tirx.And(row < qo_len, row >= pad)

    def _kv_col_valid(col, valid_len, kv_len):
        # Column-validity predicate for K/V load (TIR expression).
        if mask_mode == "padded":
            return tirx.And(col < kv_len, col < valid_len)
        pad = kv_len - valid_len
        return tirx.And(col < kv_len, col >= pad)

    @T.prim_func
    def batch_sequence_prefill_kv_masked(  # pylint: disable=too-many-branches
        var_q: T.handle, # [batch_size, qo_len, h_q, d]
        var_k: T.handle, # [batch_size, kv_len, h_kv, d]
        var_v: T.handle, # [batch_size, kv_len, h_kv, d]
        var_valid_lens: T.handle, # [batch_size], int32
        var_output: T.handle, # [batch_size, qo_len, h_q, d]
        var_lse: T.handle # [batch_size, qo_len, h_q]
    ):
        batch_size = T.int32(is_size_var=True)
        qo_len = T.int32(is_size_var=True)
        kv_len = T.int32(is_size_var=True)
        q = T.match_buffer(var_q, (batch_size, qo_len, h_q, d), dtype)
        k = T.match_buffer(var_k, (batch_size, kv_len, h_kv, d), dtype)
        v = T.match_buffer(var_v, (batch_size, kv_len, h_kv, d), dtype)
        valid_lens = T.match_buffer(var_valid_lens, (batch_size,), "int32")
        output = T.match_buffer(var_output, (batch_size, qo_len, h_q, d), dtype)
        lse = T.match_buffer(var_lse, (batch_size, qo_len, h_q), dtype)

        batch_tiles: T.int32 = T.ceildiv(qo_len * group_size, tile_x)

        for lbx in T.thread_binding(T.cast(batch_size, "int32") * batch_tiles, thread="blockIdx.x"):
            for lby in T.thread_binding(h_kv, thread="blockIdx.y"):
                for lty in T.thread_binding(num_warps, thread="threadIdx.y"):
                    for ltx in T.thread_binding(bdx, thread="threadIdx.x"):
                        with T.sblock("attn"):
                            vbx, by, ty, tx = T.axis.remap("SSSS", [lbx, lby, lty, ltx])
                            T.reads()
                            T.writes()

                            Q_smem, K_smem, V_smem, O_local = _alloc_mha_qkvo_buffers(tile_x, tile_z, d, d, dtype)
                            S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new = (
                                _alloc_softmax_state_buffers(tile_x, tile_z, bdx, num_warps)
                            )

                            b_idx: T.int32 = vbx // batch_tiles
                            valid_len: T.int32 = valid_lens[b_idx]
                            tile_id: T.int32 = vbx % batch_tiles
                            LH_start: T.int32 = tile_id * tile_x
                            T.tvm_storage_sync("shared")

                            init_states(m_smem, d_smem, O_local, ty, tx)

                            # Load Q; rows outside the valid range are zeroed so they contribute nothing downstream.
                            for li, lj in T.grid(tile_x, tile_y):
                                with T.sblock("Q_load"):
                                    i, j = T.axis.remap("SS", [li, lj])
                                    T.reads()
                                    T.writes()
                                    cur_L = (LH_start + i) // group_size
                                    cur_H_qo = by * group_size + (LH_start + i) % group_size
                                    if _q_row_valid(cur_L, valid_len, qo_len):
                                        Q_smem[i, j] = q[b_idx, cur_L, cur_H_qo, j]
                                    else:
                                        Q_smem[i, j] = 0.0
                            T.tvm_storage_sync("shared")

                            for iterator in T.serial(T.ceildiv(kv_len, tile_z)):
                                L_kv_start: T.int32 = iterator * tile_z
                                L_kv_base: T.int32 = 0
                                for lz, ly in T.grid(tile_z, tile_y):
                                    with T.sblock("K_load"):
                                        i, j = T.axis.remap("SS", [lz, ly])
                                        T.reads()
                                        T.writes()
                                        cur_L = L_kv_start + i
                                        if _kv_col_valid(cur_L, valid_len, kv_len):
                                            K_smem[i, j] = k[b_idx, L_kv_base + cur_L, by, j]
                                        else:
                                            K_smem[i, j] = 0.0
                                T.tvm_storage_sync("shared")
                                for lz, ly in T.grid(tile_z, tile_y):
                                    with T.sblock("V_load"):
                                        i, j = T.axis.remap("SS", [lz, ly])
                                        T.reads()
                                        T.writes()
                                        cur_L = L_kv_start + i
                                        if _kv_col_valid(cur_L, valid_len, kv_len):
                                            V_smem[i, j] = v[b_idx, L_kv_base + cur_L, by, j]
                                        else:
                                            V_smem[i, j] = 0.0
                                T.tvm_storage_sync("shared")

                                compute_s_gemm(Q_smem, K_smem, S_local, S_smem, sm_scale)
                                softmax_update(S_smem, m_smem, d_smem, m_prev_smem, m_new, m_prev, d_new, ty, tx, LH_start, L_kv_start, valid_len, qo_len, kv_len)
                                compute_o_gemm(S_smem, V_smem, O_local, m_prev_smem, m_smem)

                            # Store O
                            for li, lj in T.grid(tile_x, tile_y):
                                with T.sblock("O_store"):
                                    i, j = T.axis.remap("SS", [li, lj])
                                    cur_L: T.int32 = 0 + (LH_start + i) // group_size
                                    cur_H_qo: T.int32 = by * group_size + (LH_start + i) % group_size
                                    if cur_L < qo_len:
                                        output[b_idx, cur_L, cur_H_qo, j] = O_local[i, j] / d_smem[i]

                            # Store LSE
                            for li in T.grid(tile_x):
                                with T.sblock("lse_store"):
                                    i = T.axis.remap("S", [li])
                                    cur_L: T.int32 = 0 + (LH_start + i) // group_size
                                    cur_H_qo: T.int32 = by * group_size + (LH_start + i) % group_size
                                    if cur_L < qo_len:
                                        lse[b_idx, cur_L, cur_H_qo] = m_smem[i] + T.log2(d_smem[i])

    sch = tvm.s_tir.Schedule(batch_sequence_prefill_kv_masked)
    sch = _schedule_prefill_kernel(sch, LOAD_VEC, bdx, num_warps, tile_x, tile_y, tile_z, False, False)
    return sch.mod["main"].with_attr("tirx.is_scheduled", True)



def _attention_prefill_ragged_cpu(h_kv, h_q, d_qk, d_v, dtype, rope_scaling: dict[str, Any]):
    group_size = h_q // h_kv

    @T.prim_func
    def batch_prefill_ragged_kv(  # pylint: disable=too-many-branches
        var_q: T.handle,  # [total_len, h_q, d_qk]
        var_q_indptr: T.handle,  # [batch_size + 1]
        var_k: T.handle,  # [total_len, h_kv, d_qk]
        var_v: T.handle,  # [total_len, h_kv, d_v]
        var_kv_indptr: T.handle,  # [batch_size + 1]
        var_q_rope_position: T.handle,  # [total_q_len]
        var_k_rope_pos_offset: T.handle,  # [b]
        var_output: T.handle,  # [total_len, h_q, d_v]
        var_lse: T.handle,  # [total_len, h_q]
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32,
    ):
        batch_size = T.int32(is_size_var=True)
        qo_len = T.int32(is_size_var=True)
        kv_len = T.int32(is_size_var=True)
        q_indptr_elem_offset = T.int32(is_size_var=True)
        kv_indptr_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)

        q = T.match_buffer(var_q, (qo_len, h_q, d_qk), dtype)
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
        k = T.match_buffer(var_k, (kv_len, h_kv, d_qk), dtype)
        v = T.match_buffer(var_v, (kv_len, h_kv, d_v), dtype)
        kv_indptr = T.match_buffer(var_kv_indptr, (batch_size + 1,), "int32", elem_offset=kv_indptr_elem_offset)
        q_rope_position = T.match_buffer(var_q_rope_position, (qo_len,), "int32", elem_offset=q_rope_position_elem_offset)
        k_rope_pos_offset = T.match_buffer(var_k_rope_pos_offset, (batch_size,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
        output = T.match_buffer(var_output, (qo_len, h_q, d_v), dtype)
        lse = T.match_buffer(var_lse, (qo_len, h_q), "float32")  # pylint: disable=unused-variable

        for b in T.serial(batch_size):
            with T.sblock("attn"):
                softmax_sum = T.sblock_alloc_buffer([h_q], "float32")
                m_prev = T.sblock_alloc_buffer([h_q], "float32")
                m_new = T.sblock_alloc_buffer([h_q], "float32")
                d_prev = T.sblock_alloc_buffer([h_q], "float32")
                d_new = T.sblock_alloc_buffer([h_q], "float32")
                p_sum = T.sblock_alloc_buffer([d_v], "float32")
                max_score = T.sblock_alloc_buffer([h_q], "float32")
                attention_scores = T.sblock_alloc_buffer([kv_len, h_q], "float32")
                exp_scores = T.sblock_alloc_buffer([kv_len, h_q], "float32")
                attention_score = T.sblock_alloc_buffer([1], "float32")
                query_val = T.sblock_alloc_buffer([1], "float32")
                key_val = T.sblock_alloc_buffer([1], "float32")
                result = T.sblock_alloc_buffer([1], "float32")

                for q_idx in T.serial(q_indptr[b + 1] - q_indptr[b]):
                    for i in T.serial(h_q):
                        max_score[i] = -5e4
                        m_prev[i] = -5e4
                        d_prev[i] = 1.0

                    for k_idx in T.serial(kv_indptr[b + 1] - kv_indptr[b]):
                        for h in T.serial(h_q):
                            h_kv_idx = h // group_size

                            if _causal_mask(
                                causal,
                                row=q_idx,
                                col=k_idx,
                                kv_len=kv_indptr[b + 1] - kv_indptr[b],
                                qo_len=q_indptr[b + 1] - q_indptr[b],
                            ):
                                result[0] = 0.0
                                for d_idx in T.serial(d_qk):
                                    query_val[0] = T.if_then_else(
                                        rotary_mode == 1,
                                        _rope(q, q_rope_position[q_indptr[b] + q_idx], d_qk, rope_theta, rope_scale, (q_indptr[b] + q_idx, h, d_idx), dtype, rope_scaling),
                                        q[q_indptr[b] + q_idx, h, d_idx],
                                    )

                                    key_val[0] = T.if_then_else(
                                        rotary_mode == 1,
                                        _rope(k, k_rope_pos_offset[b] + k_idx, d_qk, rope_theta, rope_scale, (kv_indptr[b] + k_idx, h_kv_idx, d_idx), dtype, rope_scaling),
                                        k[kv_indptr[b] + k_idx, h_kv_idx, d_idx],
                                    )

                                    result[0] += query_val[0] * key_val[0]
                                attention_score[0] = result[0] * math.log2(math.exp(1)) * sm_scale
                            else:
                                attention_score[0] = -5e4 * math.log2(math.exp(1)) * sm_scale
                            attention_scores[k_idx, h] = attention_score[0]
                            max_score[h] = T.max(max_score[h], attention_score[0])
                            m_new[h] = T.max(m_prev[h], max_score[h])

                    for h in T.serial(h_q):
                        d_new[h] = d_prev[h] * T.exp2(m_prev[h] - m_new[h])

                    for h in T.serial(h_q):
                        softmax_sum[h] = 0.0
                        for k_idx in T.serial(kv_indptr[b + 1] - kv_indptr[b]):
                            exp_scores[k_idx, h] = T.exp2(attention_scores[k_idx, h] - m_new[h])
                            softmax_sum[h] += exp_scores[k_idx, h]
                        d_new[h] += softmax_sum[h]
                    d_prev = d_new
                    m_prev = m_new

                    for h in T.serial(h_q):
                        h_kv_idx = h // group_size
                        for i in T.serial(d_v):
                            p_sum[i] = 0.0
                        for v_idx in T.serial(kv_indptr[b + 1] - kv_indptr[b]):
                            weight = exp_scores[v_idx, h] / d_new[h]
                            for i in T.serial(d_v):
                                p_sum[i] += v[kv_indptr[b] + v_idx, h_kv_idx, i] * weight
                        for i in T.serial(d_v):
                            output[q_indptr[b] + q_idx, h, i] = p_sum[i]
                        lse[q_indptr[b] + q_idx, h] = m_prev[h] + T.log2(d_prev[h])
    return batch_prefill_ragged_kv



def _attention_prefill_ragged(h_kv, h_q, d_qk, d_v, dtype, rope_scaling: dict[str, Any], target: Target):
    NUM_BLKS, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z = _get_prefill_kernel_config(h_kv, h_q, d_qk, dtype, target)
    init_states, compute_s_gemm, softmax_update_causal, compute_o_gemm, _, advance_tile_batch, paged_store_output_lse, *_ = _make_prefill_macros(tile_x, tile_y, tile_z, d_v, bdx, num_warps, group_size)

    @T.prim_func
    def batch_prefill_ragged_kv(  # pylint: disable=too-many-branches
        var_q: T.handle, # [total_len, h_q, d_qk]
        var_q_indptr: T.handle, # [batch_size + 1]
        var_k: T.handle, # [total_len, h_kv, d_qk]
        var_v: T.handle, # [total_len, h_kv, d_v]
        var_kv_indptr: T.handle, # [batch_size + 1]
        var_q_rope_position: T.handle, # [total_q_len]
        var_k_rope_pos_offset: T.handle, # [b]
        var_output: T.handle, # [total_len, h_q, d_v]
        var_lse: T.handle, # [total_len, h_q]
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        sm_scale: T.float32
    ):
        batch_size = T.int32(is_size_var=True)
        qo_len = T.int32(is_size_var=True)
        kv_len = T.int32(is_size_var=True)
        q_indptr_elem_offset = T.int32(is_size_var=True)
        kv_indptr_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)

        q = T.match_buffer(var_q, (qo_len, h_q, d_qk), dtype)
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
        k = T.match_buffer(var_k, (kv_len, h_kv, d_qk), dtype)
        v = T.match_buffer(var_v, (kv_len, h_kv, d_v), dtype)
        kv_indptr = T.match_buffer(var_kv_indptr, (batch_size + 1,), "int32", elem_offset=kv_indptr_elem_offset)
        q_rope_position = T.match_buffer(var_q_rope_position, (qo_len,), "int32", elem_offset=q_rope_position_elem_offset)
        k_rope_pos_offset = T.match_buffer(var_k_rope_pos_offset, (batch_size,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
        output = T.match_buffer(var_output, (qo_len, h_q, d_v), dtype)
        lse = T.match_buffer(var_lse, (qo_len, h_q), "float32")  # pylint: disable=unused-variable

        # kernel code
        for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
            for lby in T.thread_binding(h_kv, thread="blockIdx.y"):
                for lty in T.thread_binding(num_warps, thread="threadIdx.y"):
                    for ltx in T.thread_binding(bdx, thread="threadIdx.x"):
                        with T.sblock("attn"):
                            bx, by, ty, tx = T.axis.remap("SSSS", [lbx, lby, lty, ltx])
                            T.reads()
                            T.writes()
                            tile_id, batch_idx, batch_tiles, batch_rows, iterator, kv_chunk_len = _alloc_tile_walk_state()
                            Q_smem, K_smem, V_smem, O_local = _alloc_mha_qkvo_buffers(tile_x, tile_z, d_qk, d_v, dtype)
                            S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new = (
                                _alloc_softmax_state_buffers(tile_x, tile_z, bdx, num_warps)
                            )

                            tile_id[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = (q_indptr[1] - q_indptr[0]) * group_size
                            batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)
                            while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                advance_tile_batch(tile_id, batch_idx, batch_tiles, batch_rows, q_indptr, batch_size)

                                if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                    b_idx: T.int32 = batch_idx[0]
                                    q_indptr_val: T.int32 = q_indptr[b_idx]
                                    LH_start: T.int32 = tile_id[0] * tile_x

                                    kv_chunk_len[0] = kv_indptr[b_idx + 1] - kv_indptr[b_idx]
                                    T.tvm_storage_sync("shared")

                                    init_states(m_smem, d_smem, O_local, ty, tx)

                                    # Load Q from gmem to smem
                                    for li, lj in T.grid(tile_x, tile_y):
                                        with T.sblock("Q_load"):
                                            i, j = T.axis.remap("SS", [li, lj])
                                            T.reads()
                                            T.writes()
                                            cur_L = q_indptr_val + (LH_start + i) // group_size
                                            cur_H_qo = by * group_size + (LH_start + i) % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                Q_smem[i, j] = T.if_then_else(
                                                    rotary_mode == 1,
                                                    _rope(q, q_rope_position[cur_L], d_qk, rope_theta, rope_scale, (cur_L, cur_H_qo, j), dtype, rope_scaling),
                                                    q[cur_L, cur_H_qo, j]
                                                )
                                            else:
                                                Q_smem[i, j] = 0.0
                                    T.tvm_storage_sync("shared")

                                    for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_z)):
                                        L_kv_start: T.int32 = iterator * tile_z
                                        L_kv_base: T.int32 = kv_indptr[b_idx]
                                        for lz, ly in T.grid(tile_z, tile_y):
                                            with T.sblock("K_load"):
                                                i, j = T.axis.remap("SS", [lz, ly])
                                                cur_L = L_kv_start + i
                                                if cur_L < kv_chunk_len[0]:
                                                    K_smem[i, j] = T.if_then_else(
                                                        rotary_mode == 1,
                                                        _rope(k, k_rope_pos_offset[b_idx] + cur_L, d_qk, rope_theta, rope_scale, (L_kv_base + cur_L, by, j), dtype, rope_scaling),
                                                        k[L_kv_base + cur_L, by, j]
                                                    )
                                                else:
                                                    K_smem[i, j] = 0.0
                                        T.tvm_storage_sync("shared")
                                        for lz, ly in T.grid(tile_z, d_v):
                                            with T.sblock("V_load"):
                                                i, j = T.axis.remap("SS", [lz, ly])
                                                T.reads()
                                                T.writes()
                                                cur_L = L_kv_start + i
                                                if cur_L < kv_chunk_len[0]:
                                                    V_smem[i, j] = v[L_kv_base + cur_L, by, j]
                                                else:
                                                    V_smem[i, j] = 0.0
                                        T.tvm_storage_sync("shared")

                                        compute_s_gemm(Q_smem, K_smem, S_local, S_smem, sm_scale)
                                        softmax_update_causal(S_smem, m_smem, d_smem, m_prev_smem, m_new, m_prev, d_new, ty, tx, LH_start, L_kv_start, causal, kv_chunk_len[0], q_indptr[b_idx + 1] - q_indptr[b_idx])
                                        compute_o_gemm(S_smem, V_smem, O_local, m_prev_smem, m_smem)

                                    paged_store_output_lse(output, lse, O_local, m_smem, d_smem, q_indptr, b_idx, by, LH_start)

                                    # move to next tile
                                    tile_id[0] += NUM_BLKS
    # pylint: enable=too-many-branches
    sch = tvm.s_tir.Schedule(batch_prefill_ragged_kv)
    sch = _schedule_prefill_kernel(sch, LOAD_VEC, bdx, num_warps, tile_x, d_v, tile_z, True, False)
    return sch.mod["main"].with_attr("tirx.is_scheduled", True)



def _attention_prefill_mla(h_q, d_latent, d_rope, dtype, sliding_window: bool, target: Target, page_size: int = 16):
    d_qk = d_latent + d_rope
    NUM_BLKS, LOAD_VEC, group_size, bdx, num_warps, tile_x, tile_y, tile_z = _get_prefill_kernel_config(1, h_q, d_qk, dtype, target)
    init_states, compute_s_gemm, softmax_update_causal, compute_o_gemm, _, advance_tile_batch, paged_store_output_lse, *_ = _make_prefill_macros(tile_x, tile_y, tile_z, d_latent, bdx, num_warps, group_size)

    global_symbol = "batch_prefill_paged_kv_mla"
    if sliding_window:
        global_symbol += "_sliding_window"

    # pylint: disable=too-many-branches
    @T.prim_func
    def batch_prefill_paged_kv_mla(
        var_q: T.handle, # [total_len, h_q, d_qk]
        var_q_indptr: T.handle, # [batch_size + 1]
        var_pages: T.handle, # [max_num_pages, page_size, d_qk]
        var_page_indptr: T.handle, # [batch_size + 1]
        var_page_values: T.handle, # [nnz_pages]
        var_length_info: T.handle, # [b] when sliding window = False, or otherwise [3, b]
        var_output: T.handle, # [total_len, h_q, d_latent]
        var_lse: T.handle, # [total_len, h_q]
        causal: T.int32,
        sm_scale: T.float32,
    ):
        T.func_attr({"global_symbol": global_symbol})
        batch_size = T.int32(is_size_var=True)
        total_len = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)
        pages_elem_offset = T.int64(is_size_var=True)
        q_indptr_elem_offset = T.int32(is_size_var=True)
        page_indptr_elem_offset = T.int32(is_size_var=True)
        page_values_elem_offset = T.int32(is_size_var=True)
        length_info_elem_offset = T.int32(is_size_var=True)

        q = T.match_buffer(var_q, (total_len, h_q, d_qk), dtype)
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
        pages = T.match_buffer(var_pages, (max_num_pages, page_size, d_qk), dtype, elem_offset=pages_elem_offset)
        page_indptr = T.match_buffer(var_page_indptr, (batch_size + 1,), "int32", elem_offset=page_indptr_elem_offset)
        page_values = T.match_buffer(var_page_values, (nnz_pages,), "int32", elem_offset=page_values_elem_offset)
        output = T.match_buffer(var_output, (total_len, h_q, d_latent), dtype)
        lse = T.match_buffer(var_lse, (total_len, h_q), "float32")  # pylint: disable=unused-variable
        # The length information of the sequences.
        # - It is in shape `(3, batch_size)` when sliding window is enabled.
        #   For a sequence "i", location
        #   - "(0, i)" is the number of KV slots used in the last page of the seq ("last_page_len"),
        #   - "(1, i)" is the starting offset of the sliding window in the seq,
        #   - "(2, i)" is the attn sink length of the sequence.
        # - It is in shape `(batch_size,)` when sliding window is disabled,
        #   denoting the "last_page_len".
        length_info = _declare_length_info(var_length_info, batch_size, sliding_window, length_info_elem_offset)

        # kernel code
        for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
            for lty in T.thread_binding(num_warps, thread="threadIdx.y"):
                for ltx in T.thread_binding(bdx, thread="threadIdx.x"):
                    with T.sblock("attn"):
                        bx, ty, tx = T.axis.remap("SSS", [lbx, lty, ltx])
                        T.reads()
                        T.writes()
                        tile_id, batch_idx, batch_tiles, batch_rows, iterator, kv_chunk_len = _alloc_tile_walk_state()
                        Q_smem, KV_smem, O_local = _alloc_mla_qkvo_buffers(tile_x, tile_z, d_qk, d_latent, dtype)
                        S_smem, S_local, m_smem, m_prev_smem, d_smem, m_new, m_prev, d_new = (
                            _alloc_softmax_state_buffers(tile_x, tile_z, bdx, num_warps)
                        )

                        tile_id[0] = bx
                        batch_idx[0] = 0
                        batch_rows[0] = (q_indptr[1] - q_indptr[0]) * group_size
                        batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)
                        while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                            advance_tile_batch(tile_id, batch_idx, batch_tiles, batch_rows, q_indptr, batch_size)

                            if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                b_idx: T.int32 = batch_idx[0]
                                LH_start: T.int32 = tile_id[0] * tile_x
                                q_indptr_val: T.int32 = q_indptr[b_idx]

                                cur_page_indptr_begin: T.int32 = page_indptr[b_idx]
                                cur_page_indptr_end: T.int32 = page_indptr[b_idx + 1]
                                kv_chunk_len[0] = T.if_then_else(
                                    cur_page_indptr_begin != cur_page_indptr_end,
                                    _get_kv_chunk_len(cur_page_indptr_end - cur_page_indptr_begin, page_size, b_idx, length_info, sliding_window),
                                    0
                                )
                                T.tvm_storage_sync("shared")

                                init_states(m_smem, d_smem, O_local, ty, tx)

                                # Load Q from gmem to smem
                                for li, lj in T.grid(tile_x, tile_y):
                                    with T.sblock("Q_load"):
                                        i, j = T.axis.remap("SS", [li, lj])
                                        T.reads()
                                        T.writes()
                                        cur_L = q_indptr_val + (LH_start + i) // group_size
                                        cur_H_qo = (LH_start + i) % group_size
                                        if cur_L < q_indptr[b_idx + 1]:
                                            Q_smem[i, j] = q[cur_L, cur_H_qo, j]
                                        else:
                                            Q_smem[i, j] = 0.0
                                T.tvm_storage_sync("shared")

                                for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_z)):
                                    L_kv_start: T.int32 = iterator * tile_z
                                    for lz, ly in T.grid(tile_z, tile_y):
                                        with T.sblock("KV_load"):
                                            i, j = T.axis.remap("SS", [lz, ly])
                                            T.reads()
                                            T.writes()
                                            cur_L = L_kv_start + i
                                            if cur_L < kv_chunk_len[0]:
                                                seq_offset: T.int32(is_size_var=True) = _get_seq_offset(cur_L, b_idx, length_info, sliding_window)  # type: ignore
                                                page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + T.floordiv(seq_offset, page_size)]  # type: ignore
                                                page_offset: T.int32(is_size_var=True) = T.floormod(seq_offset, page_size)  # type: ignore
                                                KV_smem[i, j] = pages[page_no, page_offset, j]
                                            else:
                                                KV_smem[i, j] = 0.0
                                    T.tvm_storage_sync("shared")

                                    # MLA shares the same buffer for K and V (V = KV_smem[:, :d_latent])
                                    compute_s_gemm(Q_smem, KV_smem, S_local, S_smem, sm_scale)

                                    softmax_update_causal(
                                        S_smem, m_smem, d_smem, m_prev_smem,
                                        m_new, m_prev, d_new,
                                        ty, tx, LH_start, L_kv_start,
                                        causal, kv_chunk_len[0], q_indptr[b_idx + 1] - q_indptr[b_idx],
                                    )

                                    compute_o_gemm(S_smem, KV_smem, O_local, m_prev_smem, m_smem)

                                # MLA has no blockIdx.y binding; pass by=0 so the
                                # by*group_size term in the shared epilogue drops.
                                paged_store_output_lse(
                                    output, lse, O_local, m_smem, d_smem,
                                    q_indptr, b_idx, 0, LH_start,
                                )

                                # move to next tile
                                tile_id[0] += NUM_BLKS
    # pylint: enable=too-many-branches
    sch = tvm.s_tir.Schedule(batch_prefill_paged_kv_mla)
    sch = _schedule_prefill_kernel(sch, LOAD_VEC, bdx, num_warps, tile_x, d_latent, tile_z, False, True)
    return sch.mod["main"].with_attr("tirx.is_scheduled", True)
