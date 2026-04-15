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

"""TIR kernels that operate on paged KV-cache storage (without doing attention).

This module contains:
- Append helpers that transpose/write new K/V tokens into the paged layout
  (``_kv_cache_transpose_append`` and its MLA variant).
- Debug helpers that extract K/V from the paged layout for inspection
  (``_kv_cache_debug_get_kv``, ``_kv_cache_debug_get_kv_mla``).
- Copy helpers used by the cache runtime for forking/sharing pages
  (``_copy_single_page``, ``_copy_single_page_mla``, ``_copy_single_page_cpu``).
- Compact helpers that reorganise pages after removals
  (``_compact_kv_copy``, ``_compact_kv_copy_cpu``).
"""

# pylint: disable=too-many-statements,too-many-arguments,invalid-name,line-too-long
from tvm.script import tirx as T
from tvm.target import Target

from ._kernel_common import get_max_num_threads_per_block


def _kv_cache_transpose_append(num_key_value_heads, head_dim, dtype, page_size: int = 16):
    """Return the TIR function that appends new k/v data to PagedKVCache."""

    @T.prim_func
    def tir_kv_cache_transpose_append(
        var_pages: T.handle,
        var_k_data: T.handle,
        var_v_data: T.handle,
        var_position_map: T.handle,
    ):
        T.func_attr({"tirx.noalias": True})
        ntoken = T.SizeVar("num_tokens_excluding_cache", "int64")
        num_pages = T.int64()
        pages_elem_offset = T.int64()
        position_map_elem_offset = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_key_value_heads, page_size, head_dim), dtype, elem_offset=pages_elem_offset)
        k_data = T.match_buffer(var_k_data, (ntoken, num_key_value_heads, head_dim), dtype)
        v_data = T.match_buffer(var_v_data, (ntoken, num_key_value_heads, head_dim), dtype)
        position_map = T.match_buffer(var_position_map, (ntoken,), "int32", elem_offset=position_map_elem_offset)
        for global_pos, h, f in T.grid(ntoken, num_key_value_heads, head_dim):
            if position_map[global_pos] != T.int32(-1):
                with T.sblock("k_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                    T.writes(pages[position_map[vgpos] // page_size, 0, vh, position_map[vgpos] % page_size, vf])
                    position: T.int32 = position_map[vgpos]  # type: ignore
                    pages[T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vf] = k_data[vgpos, vh, vf]
                with T.sblock("v_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], v_data[vgpos, vh, vf])
                    T.writes(pages[position_map[vgpos] // page_size, 1, vh, position_map[vgpos] % page_size, vf])
                    position: T.int32 = position_map[vgpos] # type: ignore[name-defined,no-redef]
                    pages[T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vf] = v_data[vgpos, vh, vf]

    return tir_kv_cache_transpose_append


def _kv_cache_transpose_append_mla(d_qk: int, dtype, page_size: int = 16):
    """Return the TIR function that appends new compressed KV data to PagedKVCache for MLA."""

    @T.prim_func
    def tir_kv_cache_transpose_append_mla(
        var_pages: T.handle,
        var_kv_data: T.handle,
        var_position_map: T.handle,
    ):
        T.func_attr({"tirx.noalias": True})
        ntoken = T.SizeVar("num_tokens_excluding_cache", "int64")
        num_pages = T.int64()
        pages_elem_offset = T.int64()
        position_map_elem_offset = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, page_size, d_qk), dtype, elem_offset=pages_elem_offset)
        kv_data = T.match_buffer(var_kv_data, (ntoken, d_qk), dtype)
        position_map = T.match_buffer(var_position_map, (ntoken,), "int32", elem_offset=position_map_elem_offset)
        for global_pos, f in T.grid(ntoken, d_qk):
            if position_map[global_pos] != T.int32(-1):
                with T.sblock("k_transpose_append"):
                    vgpos, vf = T.axis.remap("SS", [global_pos, f])
                    T.reads(position_map[vgpos], kv_data[vgpos, vf])
                    T.writes(pages[position_map[vgpos] // page_size, position_map[vgpos] % page_size, vf])
                    position: T.int32 = position_map[vgpos]  # type: ignore
                    pages[T.floordiv(position, page_size), T.floormod(position, page_size), vf] = kv_data[vgpos, vf]

    return tir_kv_cache_transpose_append_mla


def _kv_cache_debug_get_kv(num_hidden_layers, num_key_value_heads, head_dim, dtype):
    """Return the TIR function that fetches the k/v data on given positions and layer."""

    @T.prim_func
    def tir_kv_cache_debug_get_kv(
        var_pages: T.handle,
        var_position_map: T.handle,
        var_k_data: T.handle,
        var_v_data: T.handle,
        layer_id: T.int64,
    ):
        T.func_attr({"tirx.noalias": True})
        seqlen = T.SizeVar("num_tokens_including_cache", "int64")
        page_size = T.SizeVar("page_size", "int64")
        num_pages = T.int64()
        pages_elem_offset = T.int64()
        position_map_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_key_value_heads, page_size, head_dim), dtype,elem_offset=pages_elem_offset)
        position_map = T.match_buffer(var_position_map, (seqlen,), "int32", elem_offset=position_map_elem_offset)
        k_data = T.match_buffer(var_k_data, (num_hidden_layers, seqlen, num_key_value_heads, head_dim), dtype)
        v_data = T.match_buffer(var_v_data, (num_hidden_layers, seqlen, num_key_value_heads, head_dim), dtype)
        for p, h, d in T.grid(seqlen, num_key_value_heads, head_dim):
            with T.sblock("copy0"):
                vp, vh, vd = T.axis.remap("SSS", [p, h, d])
                T.reads(position_map[vp], pages[position_map[vp] // page_size, 0:2, vh, position_map[vp] % page_size, vd])
                T.writes(k_data[layer_id, vp, vh, vd], v_data[layer_id, vp, vh, vd])
                position: T.int32 = position_map[vp] # type: ignore[name-defined]
                k_data[layer_id, vp, vh, vd] = pages[T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vd]
                v_data[layer_id, vp, vh, vd] = pages[T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vd]

    return tir_kv_cache_debug_get_kv


def _kv_cache_debug_get_kv_mla(num_hidden_layers, d_qk, dtype):
    """Return the TIR function that fetches the k/v data on given positions and layer."""

    @T.prim_func
    def tir_kv_cache_debug_get_kv_mla(
        var_pages: T.handle,
        var_position_map: T.handle,
        var_compressed_kv_with_k_pe_data: T.handle,
        layer_id: T.int64,
    ):
        T.func_attr({"tirx.noalias": True})
        seqlen = T.SizeVar("num_tokens_including_cache", "int64")
        page_size = T.SizeVar("page_size", "int64")
        num_pages = T.int64()
        pages_elem_offset = T.int64()
        position_map_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, page_size, d_qk), dtype, elem_offset=pages_elem_offset)
        position_map = T.match_buffer(var_position_map, (seqlen,), "int32", elem_offset=position_map_elem_offset)
        compressed_kv_with_k_pe_data = T.match_buffer(var_compressed_kv_with_k_pe_data, (num_hidden_layers, seqlen, d_qk), dtype)
        for p, d in T.grid(seqlen, d_qk):
            with T.sblock("copy0"):
                vp, vd = T.axis.remap("SS", [p, d])
                T.reads(position_map[vp], pages[position_map[vp] // page_size, position_map[vp] % page_size, vd])
                T.writes(compressed_kv_with_k_pe_data[layer_id, vp, vd])
                position: T.int32 = position_map[vp] # type: ignore[name-defined]
                compressed_kv_with_k_pe_data[layer_id, vp, vd] = pages[T.floordiv(position, page_size), T.floormod(position, page_size), vd]

    return tir_kv_cache_debug_get_kv_mla


def _copy_single_page(num_heads, page_size, head_dim, dtype, target: Target):
    tx = get_max_num_threads_per_block(target)

    @T.prim_func
    def copy_single_page(var_pages: T.handle, src_page_id: T.int64, tgt_page_id: T.int64, copy_length: T.int64):
        T.func_attr({"tirx.is_scheduled": True})
        num_pages = T.int32()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_heads, page_size, head_dim), dtype, elem_offset=pages_elem_offset)

        for b in T.thread_binding((copy_length * num_heads * head_dim + tx - 1) // tx, thread="blockIdx.x"):
            for t in T.thread_binding(tx, thread="threadIdx.x"):
                with T.sblock("copy"):
                    T.where(b * tx + t < copy_length * num_heads * head_dim)
                    vh = T.axis.spatial(num_heads, T.Cast("int32", (b * tx + t) // (copy_length * head_dim)))
                    vp = T.axis.spatial(copy_length, (b * tx + t) % (copy_length * head_dim) // head_dim)
                    vd = T.axis.spatial(head_dim, T.Cast("int32", (b * tx + t) % head_dim))
                    pages[tgt_page_id, 0, vh, vp, vd] = pages[src_page_id, 0, vh, vp, vd]
                    pages[tgt_page_id, 1, vh, vp, vd] = pages[src_page_id, 1, vh, vp, vd]

    return copy_single_page


def _copy_single_page_mla(page_size, head_dim, dtype, target: Target):
    tx = get_max_num_threads_per_block(target)

    @T.prim_func
    def copy_single_page_mla(var_pages: T.handle, src_page_id: T.int64, tgt_page_id: T.int64, copy_length: T.int64):
        T.func_attr({"tirx.is_scheduled": True})
        num_pages = T.int32()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, page_size, head_dim), dtype, elem_offset=pages_elem_offset)

        for b in T.thread_binding((copy_length * head_dim + tx - 1) // tx, thread="blockIdx.x"):
            for t in T.thread_binding(tx, thread="threadIdx.x"):
                with T.sblock("copy"):
                    T.where(b * tx + t < copy_length * head_dim)
                    vp = T.axis.spatial(copy_length, (b * tx + t) // head_dim)
                    vd = T.axis.spatial(head_dim, T.Cast("int32", (b * tx + t) % head_dim))
                    pages[tgt_page_id, vp, vd] = pages[src_page_id, vp, vd]

    return copy_single_page_mla


def _copy_single_page_cpu(num_heads, page_size, head_dim, dtype):
    tx = 1

    @T.prim_func
    def copy_single_page_cpu(var_pages: T.handle, src_page_id: T.int64, tgt_page_id: T.int64, copy_length: T.int64):
        T.func_attr({"tirx.is_scheduled": True})
        num_pages = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_heads, page_size, head_dim), dtype)

        for b in T.serial((copy_length * num_heads * head_dim + tx - 1) // tx):
            for t in T.serial(tx):
                with T.sblock("copy"):
                    T.where(b * tx + t < copy_length * num_heads * head_dim)
                    vh = T.axis.spatial(num_heads, T.Cast("int32", (b * tx + t) // (copy_length * head_dim)))
                    vp = T.axis.spatial(copy_length, (b * tx + t) % (copy_length * head_dim) // head_dim)
                    vd = T.axis.spatial(head_dim, T.Cast("int32", (b * tx + t) % head_dim))
                    pages[tgt_page_id, 0, vh, vp, vd] = pages[src_page_id, 0, vh, vp, vd]
                    pages[tgt_page_id, 1, vh, vp, vd] = pages[src_page_id, 1, vh, vp, vd]

    return copy_single_page_cpu


def _compact_kv_copy(num_heads, head_dim, dtype, target: Target, page_size: int = 16):
    tx = get_max_num_threads_per_block(target)

    @T.prim_func
    def compact_kv_copy(var_pages: T.handle, var_copy_length_indptr: T.handle, var_copy_src_dst_pos: T.handle, batch_size: T.int32):
        T.func_attr({"tirx.is_scheduled": True})
        num_pages = T.int32()
        total_copy_length = T.int32()
        copy_length_indptr_elem_offset = T.int32()
        copy_src_dst_pos_elem_offset = T.int32()
        pages_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_heads, page_size, head_dim), dtype, elem_offset=pages_elem_offset)
        copy_length_indptr = T.match_buffer(var_copy_length_indptr, (batch_size + 1,), "int32", elem_offset=copy_length_indptr_elem_offset)
        copy_src_dst_pos = T.match_buffer(var_copy_src_dst_pos, (2, total_copy_length), "int32", elem_offset=copy_src_dst_pos_elem_offset)

        with T.sblock("root"):
            for bhd_o in T.thread_binding((batch_size * num_heads * head_dim + tx - 1) // tx, thread="blockIdx.x"):
                for bhd_i in T.thread_binding(tx, thread="threadIdx.x"):
                    b: T.int32 = (bhd_o * tx + bhd_i) // (num_heads * head_dim)
                    h: T.int32 = (bhd_o * tx + bhd_i) // head_dim % num_heads
                    d: T.int32 = (bhd_o * tx + bhd_i) % head_dim
                    if (bhd_o * tx + bhd_i) < batch_size * num_heads * head_dim:
                        for i in T.serial(copy_length_indptr[b + 1] - copy_length_indptr[b]):
                            src_pos: T.int32 = copy_src_dst_pos[0, copy_length_indptr[b] + i]
                            dst_pos: T.int32 = copy_src_dst_pos[1, copy_length_indptr[b] + i]
                            pages[dst_pos // page_size, 0, h, dst_pos % page_size, d] = pages[src_pos // page_size, 0, h, src_pos % page_size, d]
                            pages[dst_pos // page_size, 1, h, dst_pos % page_size, d] = pages[src_pos // page_size, 1, h, src_pos % page_size, d]

    return compact_kv_copy


def _compact_kv_copy_cpu(num_heads, head_dim, dtype, page_size: int = 16):
    tx = 8

    @T.prim_func
    def compact_kv_copy_cpu(var_pages: T.handle, var_copy_length_indptr: T.handle, var_copy_src_dst_pos: T.handle, batch_size: T.int32):
        T.func_attr({"tirx.is_scheduled": True})
        num_pages = T.int32()
        total_copy_length = T.int32()
        copy_length_indptr_elem_offset = T.int32()
        copy_src_dst_pos_elem_offset = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_heads, page_size, head_dim), dtype)
        copy_length_indptr = T.match_buffer(var_copy_length_indptr, (batch_size + 1,), "int32", elem_offset=copy_length_indptr_elem_offset)
        copy_src_dst_pos = T.match_buffer(var_copy_src_dst_pos, (2, total_copy_length), "int32", elem_offset=copy_src_dst_pos_elem_offset)

        with T.sblock("root"):
            for bhd_o in T.serial((batch_size * num_heads * head_dim + tx - 1) // tx):
                for bhd_i in T.serial(tx):
                    b: T.int32 = (bhd_o * tx + bhd_i) // (num_heads * head_dim)
                    h: T.int32 = (bhd_o * tx + bhd_i) // head_dim % num_heads
                    d: T.int32 = (bhd_o * tx + bhd_i) % head_dim
                    if (bhd_o * tx + bhd_i) < batch_size * num_heads * head_dim:
                        for i in T.serial(copy_length_indptr[b + 1] - copy_length_indptr[b]):
                            src_pos: T.int32 = copy_src_dst_pos[0, copy_length_indptr[b] + i]
                            dst_pos: T.int32 = copy_src_dst_pos[1, copy_length_indptr[b] + i]
                            pages[dst_pos // page_size, 0, h, dst_pos % page_size, d] = pages[src_pos // page_size, 0, h, src_pos % page_size, d]
                            pages[dst_pos // page_size, 1, h, dst_pos % page_size, d] = pages[src_pos // page_size, 1, h, src_pos % page_size, d]

    return compact_kv_copy_cpu
