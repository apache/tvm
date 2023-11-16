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
from typing import List

import numpy as np
import tvm
import tvm.testing
from tvm.script import tir as T


reserved_nseq = 2
total_seq_len = 128
page_size = 8
nlayer = 4
nhead = 16
nfeat = 32
dtype = "float16"


fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
fadd_sequence = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_add_sequence")
freserve = tvm.get_global_func(
    "vm.builtin.paged_attention_kv_cache_reserve_extra_length_for_append"
)
fappend = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_append")
freset_append_length = tvm.get_global_func(
    "vm.builtin.paged_attention_kv_cache_reset_append_lengths"
)
fsync = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_sync_aux_array_to_device")
fremove = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_remove")
fpopn = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_popn")
fclear = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_clear")

# fmt: off
@T.prim_func
def transpose_append(
    var_pages: T.handle,
    var_k_data: T.handle,
    var_v_data: T.handle,
    var_page_table_indptr: T.handle,
    var_page_table_values: T.handle,
    var_last_page_offset: T.handle,
    var_append_length_indptr: T.handle,
    var_pos2seqidx: T.handle,
    layer_id: T.int32,
):
    nseq = T.int32()
    ntoken = T.int32()
    nhead = T.int32()
    nfeat = T.int32()
    nlayer = T.int32()
    npage = T.int32()
    page_size = T.int32()
    num_pages = T.int32()

    pages = T.match_buffer(var_pages, (num_pages, nlayer, 2, nhead, page_size, nfeat), "float16")
    k_data = T.match_buffer(var_k_data, (ntoken, nhead, nfeat), "float16")
    v_data = T.match_buffer(var_v_data, (ntoken, nhead, nfeat), "float16")
    last_page_offset = T.match_buffer(var_last_page_offset, (nseq,), "int32")
    page_table_indptr = T.match_buffer(var_page_table_indptr, (nseq + 1,), "int32")
    page_table_values = T.match_buffer(var_page_table_values, (npage,), "int32")
    append_length_indptr = T.match_buffer(var_append_length_indptr, (nseq + 1,), "int32")
    pos2seqidx = T.match_buffer(var_pos2seqidx, (ntoken,), "int32")

    for global_pos, h, f in T.grid(ntoken, nhead, nfeat):
        with T.block("k_transpose_append"):
            vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
            seq_idx = pos2seqidx[vgpos]
            seqlen: T.int32 = (page_table_indptr[seq_idx + 1] - page_table_indptr[seq_idx] - 1) * page_size + last_page_offset[seq_idx]
            pages[
                page_table_values[page_table_indptr[seq_idx] + T.floordiv(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size)],
                layer_id,
                0,
                vh,
                T.floormod(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size),
                vf,
            ] = k_data[vgpos, vh, vf]
        with T.block("v_transpose_append"):
            vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
            seq_idx = pos2seqidx[vgpos]
            seqlen: T.int32 = (page_table_indptr[seq_idx + 1] - page_table_indptr[seq_idx] - 1) * page_size + last_page_offset[seq_idx]
            pages[
                page_table_values[page_table_indptr[seq_idx] + T.floordiv(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size)],
                layer_id,
                1,
                vh,
                T.floormod(seqlen - (append_length_indptr[seq_idx + 1] - vgpos), page_size),
                vf,
            ] = v_data[vgpos, vh, vf]


@T.prim_func
def copy_cache(
    var_pages: T.handle,
    var_page_table_indptr: T.handle,
    var_page_table_values: T.handle,
    var_values: T.handle,
    seq_id: T.int32,
):
    nhead = T.int32()
    nfeat = T.int32()
    nlayer = T.int32()
    seqlen = T.int32()
    npage = T.int32()
    page_size = T.int32()
    num_pages = T.int32()
    num_total_seqs_plus_1 = T.int32()

    pages = T.match_buffer(var_pages, (num_pages, nlayer, 2, nhead, page_size, nfeat), "float16")
    page_table_indptr = T.match_buffer(var_page_table_indptr, (num_total_seqs_plus_1,), "int32")
    page_table_values = T.match_buffer(var_page_table_values, (npage,), "int32")
    values = T.match_buffer(var_values, (nlayer, 2, nhead, seqlen, nfeat), "float16")

    for l, kv_idx, h, pos, f in T.grid(nlayer, 2, nhead, seqlen, nfeat):
        with T.block("view"):
            vl, vi, vh, vp, vf = T.axis.remap("SSSSS", [l, kv_idx, h, pos, f])
            values[vl, vi, vh, vp, vf] = pages[
                page_table_values[page_table_indptr[seq_id] + T.floordiv(vp, page_size)],
                vl,
                vi,
                vh,
                T.floormod(vp, page_size),
                vf,
            ]
# fmt: on


def verify_cached_values(cache, expected, f_copy_cache):
    fview = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_debug_get_kv")

    actual = fview(cache, f_copy_cache)
    assert len(actual) == len(expected)
    for seq_actual, seq_expected in zip(actual, expected):
        tvm.testing.assert_allclose(np.transpose(seq_actual.numpy(), [0, 1, 3, 2, 4]), seq_expected)


def build_tir_func(tir_funcs: List[tvm.tir.PrimFunc], target="llvm"):
    return [tvm.build(tir_func, target=target).entry_func for tir_func in tir_funcs]


def test_paged_attention_kv_cache_append_prefill():
    f_transpose_append, f_copy_cache = build_tir_func([transpose_append, copy_cache])
    cache = fcreate(
        tvm.runtime.ShapeTuple([reserved_nseq, total_seq_len, page_size]),
        nlayer,
        nhead,
        nfeat,
        tvm.nd.empty((), dtype),
        True,
    )

    operation_seq = [[(0, 6)], [(1, 8)], [(2, 11)], [(3, 16)], [(4, 19), (5, 20)]]
    operation_seq += [[(6, 21), (7, 24)], [(2, 5), (4, 7), (8, 24)]]
    operation_seq += [[(6, 13)], [(8, 19)], [(0, 1)], [(1, 3), (3, 8), (5, 12), (7, 11)]]

    current_nseq = 0
    cached_values = []
    for batch in operation_seq:
        for seq_id, _ in batch:
            if seq_id >= current_nseq:
                seq_id_in_cache = fadd_sequence(cache)
                assert seq_id_in_cache == seq_id
                assert seq_id == current_nseq
                current_nseq += 1

        freset_append_length(cache)
        for seq_id, append_length in batch:
            freserve(cache, seq_id, append_length)
        fsync(cache)

        global_new_kv = np.zeros((nlayer, 2, 0, nhead, nfeat), dtype)
        for seq_id, new_len in batch:
            if seq_id >= len(cached_values):
                assert seq_id == len(cached_values)
                cached_values.append(np.zeros((nlayer, 2, 0, nhead, nfeat), dtype))

            new_kv = np.random.rand(nlayer, 2, new_len, nhead, nfeat).astype(dtype)
            cached_values[seq_id] = np.concatenate([cached_values[seq_id], new_kv], axis=2)
            global_new_kv = np.concatenate([global_new_kv, new_kv], axis=2)
        for layer_id in range(nlayer):
            keys = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 0], axis=0))
            values = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 1], axis=0))
            fappend(cache, f_transpose_append, keys, values, layer_id)

        # Verify
        verify_cached_values(cache, cached_values, f_copy_cache)


def test_paged_attention_kv_cache_append_decode():
    f_transpose_append, f_copy_cache = build_tir_func([transpose_append, copy_cache])
    cache = fcreate(
        tvm.runtime.ShapeTuple([reserved_nseq, total_seq_len, page_size]),
        nlayer,
        nhead,
        nfeat,
        tvm.nd.empty((), dtype),
        True,
    )

    cached_values = []
    initial_lengths = [31, 21, 16, 3, 8, 7, 3]
    nseq = len(initial_lengths)

    # Initial prefill
    freset_append_length(cache)
    for seq_id, append_length in enumerate(initial_lengths):
        seq_id_in_cache = fadd_sequence(cache)
        assert seq_id_in_cache == seq_id
        freserve(cache, seq_id, append_length)
    fsync(cache)

    global_new_kv = np.zeros((nlayer, 2, 0, nhead, nfeat), dtype)
    for length in initial_lengths:
        new_kv = np.random.rand(nlayer, 2, length, nhead, nfeat).astype(dtype)
        cached_values.append(new_kv)
        global_new_kv = np.concatenate([global_new_kv, new_kv], axis=2)
    for layer_id in range(nlayer):
        keys = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 0], axis=0))
        values = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 1], axis=0))
        fappend(cache, f_transpose_append, keys, values, layer_id)

    verify_cached_values(cache, cached_values, f_copy_cache)

    # Decode
    for _ in range(16):
        decode_new_kv = np.random.rand(nlayer, 2, nseq, 1, nhead, nfeat).astype(dtype)
        freset_append_length(cache)
        for seq_id in range(nseq):
            freserve(cache, seq_id, 1)
        fsync(cache)
        for seq_id in range(nseq):
            cached_values[seq_id] = np.concatenate(
                [cached_values[seq_id], decode_new_kv[:, :, seq_id, ...]], axis=2
            )
        for layer_id in range(nlayer):
            keys = tvm.nd.array(decode_new_kv[layer_id, 0])
            values = tvm.nd.array(decode_new_kv[layer_id, 1])
            fappend(cache, f_transpose_append, keys, values, layer_id)

        verify_cached_values(cache, cached_values, f_copy_cache)


def test_paged_attention_kv_cache_remove():
    f_transpose_append, f_copy_cache = build_tir_func([transpose_append, copy_cache])
    cache = fcreate(
        tvm.runtime.ShapeTuple([reserved_nseq, total_seq_len, page_size]),
        nlayer,
        nhead,
        nfeat,
        tvm.nd.empty((), dtype),
        True,
    )

    cached_values = []
    initial_lengths = [31, 21, 16, 3, 8, 7, 3]

    # Initial prefill
    freset_append_length(cache)
    for seq_id, append_length in enumerate(initial_lengths):
        seq_id_in_cache = fadd_sequence(cache)
        assert seq_id_in_cache == seq_id
        freserve(cache, seq_id, append_length)
    fsync(cache)

    global_new_kv = np.zeros((nlayer, 2, 0, nhead, nfeat), dtype)
    for length in initial_lengths:
        new_kv = np.random.rand(nlayer, 2, length, nhead, nfeat).astype(dtype)
        cached_values.append(new_kv)
        global_new_kv = np.concatenate([global_new_kv, new_kv], axis=2)
    for layer_id in range(nlayer):
        keys = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 0], axis=0))
        values = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 1], axis=0))
        fappend(cache, f_transpose_append, keys, values, layer_id)

    verify_cached_values(cache, cached_values, f_copy_cache)

    # Remove
    while len(cached_values) > 2:
        seq_id = np.random.randint(0, len(cached_values))
        fremove(cache, seq_id)
        cached_values.pop(seq_id)
    fsync(cache)
    verify_cached_values(cache, cached_values, f_copy_cache)

    # Append after removal
    seq_id = 2
    new_len = 29
    seq_id_in_cache = fadd_sequence(cache)
    assert seq_id_in_cache == seq_id
    freset_append_length(cache)
    freserve(cache, seq_id, new_len)
    fsync(cache)
    new_kv = np.random.rand(nlayer, 2, new_len, nhead, nfeat).astype(dtype)
    cached_values.append(new_kv)
    for layer_id in range(nlayer):
        keys = tvm.nd.array(np.expand_dims(new_kv[layer_id, 0], axis=0))
        values = tvm.nd.array(np.expand_dims(new_kv[layer_id, 1], axis=0))
        fappend(cache, f_transpose_append, keys, values, layer_id)

    verify_cached_values(cache, cached_values, f_copy_cache)


def test_paged_attention_kv_cache_popn():
    f_transpose_append, f_copy_cache = build_tir_func([transpose_append, copy_cache])
    cache = fcreate(
        tvm.runtime.ShapeTuple([reserved_nseq, total_seq_len, page_size]),
        nlayer,
        nhead,
        nfeat,
        tvm.nd.empty((), dtype),
        True,
    )

    cached_values = []
    initial_lengths = [20, 24, 26, 27]
    nseq = len(initial_lengths)

    # Initial prefill
    freset_append_length(cache)
    for seq_id, append_length in enumerate(initial_lengths):
        seq_id_in_cache = fadd_sequence(cache)
        assert seq_id_in_cache == seq_id
        freserve(cache, seq_id, append_length)
    fsync(cache)

    global_new_kv = np.zeros((nlayer, 2, 0, nhead, nfeat), dtype)
    for length in initial_lengths:
        new_kv = np.random.rand(nlayer, 2, length, nhead, nfeat).astype(dtype)
        cached_values.append(new_kv)
        global_new_kv = np.concatenate([global_new_kv, new_kv], axis=2)
    for layer_id in range(nlayer):
        keys = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 0], axis=0))
        values = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 1], axis=0))
        fappend(cache, f_transpose_append, keys, values, layer_id)

    verify_cached_values(cache, cached_values, f_copy_cache)

    # Pop n
    for pop_length in [3, 13]:
        for seq_id in range(nseq):
            fpopn(cache, seq_id, pop_length)
            cached_values[seq_id] = cached_values[seq_id][:, :, :-pop_length, ...]
    fsync(cache)
    verify_cached_values(cache, cached_values, f_copy_cache)

    # Decode after pop n
    for _ in range(5):
        decode_new_kv = np.random.rand(nlayer, 2, nseq, 1, nhead, nfeat).astype(dtype)
        freset_append_length(cache)
        for seq_id in range(nseq):
            freserve(cache, seq_id, 1)
        fsync(cache)

        for seq_id in range(nseq):
            cached_values[seq_id] = np.concatenate(
                [cached_values[seq_id], decode_new_kv[:, :, seq_id, ...]], axis=2
            )
        for layer_id in range(nlayer):
            keys = tvm.nd.array(decode_new_kv[layer_id, 0])
            values = tvm.nd.array(decode_new_kv[layer_id, 1])
            fappend(cache, f_transpose_append, keys, values, layer_id)

        verify_cached_values(cache, cached_values, f_copy_cache)


def test_paged_attention_kv_cache_clear():
    f_transpose_append, f_copy_cache = build_tir_func([transpose_append, copy_cache])
    cache = fcreate(
        tvm.runtime.ShapeTuple([reserved_nseq, total_seq_len, page_size]),
        nlayer,
        nhead,
        nfeat,
        tvm.nd.empty((), dtype),
        True,
    )

    cached_values = []
    initial_lengths = [20, 24, 26, 27]

    # Initial prefill
    freset_append_length(cache)
    for seq_id, append_length in enumerate(initial_lengths):
        seq_id_in_cache = fadd_sequence(cache)
        assert seq_id_in_cache == seq_id
        freserve(cache, seq_id, append_length)
    fsync(cache)

    global_new_kv = np.zeros((nlayer, 2, 0, nhead, nfeat), dtype)
    for length in initial_lengths:
        new_kv = np.random.rand(nlayer, 2, length, nhead, nfeat).astype(dtype)
        cached_values.append(new_kv)
        global_new_kv = np.concatenate([global_new_kv, new_kv], axis=2)
    for layer_id in range(nlayer):
        keys = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 0], axis=0))
        values = tvm.nd.array(np.expand_dims(global_new_kv[layer_id, 1], axis=0))
        fappend(cache, f_transpose_append, keys, values, layer_id)

    verify_cached_values(cache, cached_values, f_copy_cache)

    # Clear
    fclear(cache)
    verify_cached_values(cache, [], f_copy_cache)


if __name__ == "__main__":
    test_paged_attention_kv_cache_append_prefill()
    test_paged_attention_kv_cache_append_decode()
    test_paged_attention_kv_cache_remove()
    test_paged_attention_kv_cache_popn()
    test_paged_attention_kv_cache_clear()
    # Test for attention is not included at this moment
    # since we do not have TIR attention functions yet.
