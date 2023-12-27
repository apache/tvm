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
from typing import Dict, List, Tuple, Union

import numpy as np
import pytest
import scipy.special

import tvm
import tvm.testing
from tvm import dlight as dl
from tvm.runtime import ShapeTuple
from tvm.script import tir as T

reserved_nseq = 32
maximum_total_seq_length = 1024
page_size = 16
num_layers = 4
num_qo_heads = 32
num_kv_heads = 4
head_dim = 128
rope_scale = 1.0
rope_theta = 1e4
dtype = "float16"
device = tvm.cuda()

fclear = None
fcreate = None
fadd_sequence = None
fremove_sequence = None
ffork_sequence = None
fpopn = None
fbegin_forward = None
fend_forward = None
fattention = None
fdebug_get_kv = None

fattention_prefill = None
fattention_decode = None
fattention_prefill_ragged = None
fattention_prefill_begin_forward = None
fattention_prefill_end_forward = None
fattention_decode_begin_forward = None
fattention_decode_end_forward = None
fattention_prefill_ragged_begin_forward = None
fattention_prefill_ragged_end_forward = None
fattention_merge_state = None
fattention_rotary = None


@T.prim_func
def kv_cache_transpose_append(
    var_pages: T.handle,
    var_k_data: T.handle,
    var_v_data: T.handle,
    var_position_map: T.handle,
):
    ntoken = T.SizeVar("ntoken", "int64")
    page_size = T.SizeVar("page_size", "int64")
    num_pages = T.int64()

    pages = T.match_buffer(var_pages, (num_pages, 2, num_kv_heads, page_size, head_dim), dtype)
    k_data = T.match_buffer(var_k_data, (ntoken, num_kv_heads, head_dim), dtype)
    v_data = T.match_buffer(var_v_data, (ntoken, num_kv_heads, head_dim), dtype)
    position_map = T.match_buffer(var_position_map, (ntoken,), "int32")

    for global_pos, h, f in T.grid(ntoken, num_kv_heads, head_dim):
        with T.block("k_transpose_append"):
            vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
            position: T.int64 = T.Cast("int64", position_map[vgpos])
            pages[
                T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vf
            ] = k_data[vgpos, vh, vf]
        with T.block("v_transpose_append"):
            vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
            position: T.int64 = T.Cast("int64", position_map[vgpos])
            pages[
                T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vf
            ] = v_data[vgpos, vh, vf]


@T.prim_func
def copy_cache(
    var_pages: T.handle,
    var_position_map: T.handle,
    var_k_data: T.handle,
    var_v_data: T.handle,
    layer_id: T.int64,
):
    num_kv_heads = T.int64()
    head_dim = T.int64()
    seqlen = T.SizeVar("seqlen", "int64")
    page_size = T.int64()
    num_pages = T.int64()

    pages = T.match_buffer(var_pages, (num_pages, 2, num_kv_heads, page_size, head_dim), "float16")
    position_map = T.match_buffer(var_position_map, (seqlen,), "int32")
    k_data = T.match_buffer(var_k_data, (num_layers, seqlen, num_kv_heads, head_dim), "float16")
    v_data = T.match_buffer(var_v_data, (num_layers, seqlen, num_kv_heads, head_dim), "float16")

    for p, h, d in T.grid(seqlen, num_kv_heads, head_dim):
        with T.block("copy0"):
            vp, vh, vd = T.axis.remap("SSS", [p, h, d])
            position: T.int64 = T.Cast("int64", position_map[vp])
            k_data[layer_id, vp, vh, vd] = pages[
                T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vd
            ]
            v_data[layer_id, vp, vh, vd] = pages[
                T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vd
            ]


def set_global_func():
    global fclear, fcreate, fadd_sequence, fremove_sequence, ffork_sequence, fpopn
    global fbegin_forward, fend_forward, fattention, fdebug_get_kv
    global fattention_prefill, fattention_prefill_begin_forward, fattention_prefill_end_forward
    global fattention_decode, fattention_decode_begin_forward, fattention_decode_end_forward
    global fattention_prefill_ragged
    global fattention_prefill_ragged_begin_forward
    global fattention_prefill_ragged_end_forward
    global fattention_merge_state, fattention_rotary

    fclear = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_clear")
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    fadd_sequence = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_add_sequence")
    fremove_sequence = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_remove_sequence")
    ffork_sequence = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_fork_sequence")
    fpopn = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_popn")
    fbegin_forward = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_begin_forward")
    fend_forward = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_end_forward")
    fattention = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_attention")
    fdebug_get_kv = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_debug_get_kv")

    fattention_prefill = tvm.get_global_func("paged_kv_cache.attention_kernel_prefill")
    fattention_decode = tvm.get_global_func("paged_kv_cache.attention_kernel_decode")
    fattention_prefill_ragged = tvm.get_global_func(
        "flashinfer.attention_kernel_prefill_with_ragged_kv_cache"
    )
    fattention_prefill_begin_forward = tvm.get_global_func(
        "paged_kv_cache.attention_kernel_prefill_begin_forward"
    )
    fattention_prefill_end_forward = tvm.get_global_func(
        "paged_kv_cache.attention_kernel_prefill_end_forward"
    )
    fattention_decode_begin_forward = tvm.get_global_func(
        "paged_kv_cache.attention_kernel_decode_begin_forward"
    )
    fattention_decode_end_forward = tvm.get_global_func(
        "paged_kv_cache.attention_kernel_decode_end_forward"
    )
    fattention_prefill_ragged_begin_forward = tvm.get_global_func(
        "flashinfer.attention_kernel_prefill_with_ragged_kv_cache_begin_forward"
    )
    fattention_prefill_ragged_end_forward = tvm.get_global_func(
        "flashinfer.attention_kernel_prefill_with_ragged_kv_cache_end_forward"
    )
    fattention_merge_state = tvm.get_global_func("flashinfer.merge_state_in_place")
    fattention_rotary = tvm.get_global_func("flashinfer.batch_qk_apply_rotary_in_place")


def create_kv_cache():
    set_global_func()
    target = tvm.target.Target("nvidia/geforce-rtx-3090-ti")
    builts = []
    for tir_func in [kv_cache_transpose_append, copy_cache]:
        mod = tvm.IRModule({"main": tir_func})
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
        f = tvm.build(mod["main"], target=target)
        builts.append(f.entry_func)

    ftranspose_append, fcopy_cache = builts
    cache = fcreate(
        tvm.runtime.ShapeTuple([reserved_nseq, maximum_total_seq_length, page_size]),
        num_layers,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        rope_scale,
        rope_theta,
        tvm.nd.empty((), dtype, device=device),
        ftranspose_append,
        fattention_prefill,
        fattention_decode,
        fattention_prefill_ragged,
        fattention_prefill_ragged_begin_forward,
        fattention_prefill_ragged_end_forward,
        fattention_prefill_begin_forward,
        fattention_prefill_end_forward,
        fattention_decode_begin_forward,
        fattention_decode_end_forward,
        fattention_rotary,
        fattention_merge_state,
        fcopy_cache,
    )
    return cache


@pytest.fixture()
def kv_cache():
    return create_kv_cache()


def verify_cached_kv(kv_cache, seq_ids, expected_k, expected_v):
    for seq_id in seq_ids:
        keys_expected = expected_k[seq_id]
        values_expected = expected_v[seq_id]
        assert keys_expected.shape == values_expected.shape
        seq_length = expected_k[seq_id].shape[1]
        keys = tvm.nd.empty(keys_expected.shape, dtype=dtype, device=device)
        values = tvm.nd.empty(values_expected.shape, dtype=dtype, device=device)
        fdebug_get_kv(kv_cache, seq_id, 0, seq_length, keys, values)
        tvm.testing.assert_allclose(keys.numpy(), keys_expected, rtol=1e-3, atol=1e-3)
        tvm.testing.assert_allclose(values.numpy(), values_expected, rtol=1e-3, atol=1e-3)


def f_apply_rotary(x, offset, scale, theta):
    # x: (N, H, D)
    assert len(x.shape) == 3
    nfeat = x.shape[-1]
    nfeat_half = x.shape[-1] // 2
    x = x.astype("float32")
    y = np.concatenate([-x[:, :, nfeat_half:], x[:, :, :nfeat_half]], axis=-1)

    inv_freq = scale / (theta ** (np.arange(0, nfeat, 2).astype("float32") / nfeat))
    t = np.arange(offset, offset + x.shape[0], dtype=inv_freq.dtype)
    freqs = np.einsum("i,j->ij", t, inv_freq)
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos_values = np.cos(emb)
    sin_values = np.sin(emb)

    return np.einsum("ij,ikj->ikj", cos_values, x) + np.einsum("ij,ikj->ikj", sin_values, y)


def apply_attention(
    kv_cache,
    batch: List[Tuple[Union[int, Tuple[int, int]], int]],
    cached_k: Dict[int, np.ndarray],
    cached_v: Dict[int, np.ndarray],
) -> None:
    seq_ids = []
    append_lengths = []
    for i, (seq_id, append_length) in enumerate(batch):
        fork_parent_id = None
        if isinstance(seq_id, tuple):
            # Fork sequence
            seq_id, fork_parent_id = seq_id
            batch[i] = (seq_id, append_length)
        seq_ids.append(seq_id)
        append_lengths.append(append_length)
        if fork_parent_id is not None:
            assert fork_parent_id in cached_k
            assert seq_id not in cached_k
            ffork_sequence(kv_cache, fork_parent_id, seq_id)
            cached_k[seq_id] = cached_k[fork_parent_id]
            cached_v[seq_id] = cached_v[fork_parent_id]
        elif seq_id not in cached_k:
            fadd_sequence(kv_cache, seq_id)
            cached_k[seq_id] = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)
            cached_v[seq_id] = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)

    use_decode_shape = all(append_length == 1 for _, append_length in batch)
    fbegin_forward(kv_cache, ShapeTuple(seq_ids), ShapeTuple(append_lengths))

    global_new_q = np.zeros((num_layers, 0, num_qo_heads, head_dim), dtype)
    global_new_k = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)
    global_new_v = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)

    q_array = []
    for seq_id, append_length in batch:
        new_q = np.random.rand(num_layers, append_length, num_qo_heads, head_dim).astype(dtype)
        new_k = np.random.rand(num_layers, append_length, num_kv_heads, head_dim).astype(dtype)
        new_v = np.random.rand(num_layers, append_length, num_kv_heads, head_dim).astype(dtype)
        q_array.append(new_q)

        cached_k[seq_id] = np.concatenate(
            [
                cached_k[seq_id],
                np.stack(
                    [
                        f_apply_rotary(new_k[l], cached_k[seq_id].shape[1], rope_scale, rope_theta)
                        for l in range(num_layers)
                    ],
                    axis=0,
                ),
            ],
            axis=1,
        )
        cached_v[seq_id] = np.concatenate([cached_v[seq_id], new_v], axis=1)
        global_new_q = np.concatenate([global_new_q, new_q], axis=1)
        global_new_k = np.concatenate([global_new_k, new_k], axis=1)
        global_new_v = np.concatenate([global_new_v, new_v], axis=1)

    for layer_id in range(num_layers):
        queries_np = global_new_q[layer_id : layer_id + 1]
        keys_np = global_new_k[layer_id : layer_id + 1]
        values_np = global_new_v[layer_id : layer_id + 1]
        if use_decode_shape:
            queries_np = queries_np.transpose(1, 0, 2, 3)
            keys_np = keys_np.transpose(1, 0, 2, 3)
            values_np = values_np.transpose(1, 0, 2, 3)
        queries = tvm.nd.array(queries_np, device=device)
        keys = tvm.nd.array(keys_np, device=device)
        values = tvm.nd.array(values_np, device=device)
        outputs = tvm.nd.empty(queries.shape, dtype, device=device)
        fattention(kv_cache, layer_id, queries, keys, values, outputs)

        # Compute attention expected results.
        outputs = outputs.numpy()
        if use_decode_shape:
            outputs = outputs.transpose(1, 0, 2, 3)
        sum_length = 0
        for i, (seq_id, append_length) in enumerate(batch):
            assert cached_k[seq_id].shape[1] == cached_v[seq_id].shape[1] >= append_length

            rope_offset = cached_k[seq_id].shape[1] - append_length
            q_seq = f_apply_rotary(
                q_array[i][layer_id],
                rope_offset,
                rope_scale,
                rope_theta,
            ).transpose(1, 0, 2)
            # Todo(Zihao, Ruihang): fold RoPE into flashinfer attn kernel in multi-level cases.
            # so that k/v values in cache does not have RoPE applied.
            # k_seq = f_apply_rotary(cached_k[seq_id][layer_id], 0, rope_scale, rope_theta).transpose(
            #     1, 2, 0
            # )
            k_seq = cached_k[seq_id][layer_id].transpose(1, 2, 0)
            v_seq = cached_v[seq_id][layer_id].transpose(1, 0, 2)

            k_seq = np.repeat(k_seq, num_qo_heads // num_kv_heads, axis=0)
            v_seq = np.repeat(v_seq, num_qo_heads // num_kv_heads, axis=0)
            softmax_input = (q_seq.astype("float32") @ k_seq.astype("float32")) / np.sqrt(head_dim)
            softmax_shape = softmax_input.shape
            length_diff = softmax_shape[-1] - softmax_shape[-2]
            assert length_diff >= 0
            mask = np.tril(
                np.full_like(softmax_input, np.finfo("float32").max), k=length_diff
            ) + np.triu(np.full_like(softmax_input, np.finfo("float32").min), k=length_diff + 1)
            softmax_input = np.minimum(softmax_input, mask)

            results = np.expand_dims(
                (scipy.special.softmax(softmax_input, axis=-1) @ v_seq.astype("float32")).transpose(
                    1, 0, 2
                ),
                axis=0,
            ).astype(dtype)

            tvm.testing.assert_allclose(
                outputs[:, sum_length : sum_length + append_length, ...],
                results,
                rtol=1e-3,
                atol=1e-3,
            )
            sum_length += append_length
    fend_forward(kv_cache)

    # Verify
    verify_cached_kv(kv_cache, seq_ids, cached_k, cached_v)


@pytest.mark.skip(reason="Require FlashInfer enabled")
def test_paged_attention_kv_cache_prefill_and_decode(kv_cache):
    fclear(kv_cache)

    # Prefill.
    operation_seq = [[(0, 6)], [(1, 8)], [(2, 11)], [(3, 16)], [(4, 19), (5, 20)]]
    operation_seq += [[(6, 21), (7, 24)], [(2, 5), (4, 7), (8, 24)]]
    operation_seq += [[(6, 13)], [(8, 19)], [(0, 1)], [(1, 3), (3, 8), (5, 12), (7, 11)]]
    # Decode
    operation_seq += [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]]
    operation_seq += [[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]]
    operation_seq += [[(0, 1), (2, 1), (4, 1), (6, 1), (8, 1)]]
    operation_seq += [[(4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]]

    cached_k = {}
    cached_v = {}
    for batch in operation_seq:
        apply_attention(kv_cache, batch, cached_k, cached_v)


@pytest.mark.skip(reason="Require FlashInfer enabled")
def test_paged_attention_kv_cache_remove_sequence(kv_cache):
    fclear(kv_cache)

    num_sequences = 5
    batch = [(seq_id, 1) for seq_id in range(num_sequences)]
    cached_k = {}
    cached_v = {}
    for seq_id_to_remove in range(num_sequences):
        apply_attention(kv_cache, batch, cached_k, cached_v)
        # Remove sequence.
        fremove_sequence(kv_cache, seq_id_to_remove)
        cached_k.pop(seq_id_to_remove)
        cached_v.pop(seq_id_to_remove)
        verify_cached_kv(
            kv_cache,
            seq_ids=[seq_id for seq_id in range(num_sequences) if seq_id != seq_id_to_remove],
            expected_k=cached_k,
            expected_v=cached_v,
        )


@pytest.mark.skip(reason="Require FlashInfer enabled")
def test_paged_attention_kv_cache_fork_sequence(kv_cache):
    fclear(kv_cache)

    cached_k = {}
    cached_v = {}
    batch = [(0, 60), (1, 88), (2, 17), (3, 4)]
    apply_attention(kv_cache, batch, cached_k, cached_v)
    # Fork existing sequences.
    apply_attention(kv_cache, [((4, 3), 35)], cached_k, cached_v)
    apply_attention(kv_cache, [((5, 0), 20)], cached_k, cached_v)
    apply_attention(kv_cache, [((6, 5), 102)], cached_k, cached_v)
    apply_attention(kv_cache, [((7, 0), 3)], cached_k, cached_v)
    apply_attention(kv_cache, [((8, 5), 71)], cached_k, cached_v)
    apply_attention(kv_cache, [((9, 5), 20)], cached_k, cached_v)
    # Mixture of decode and prefill.
    operation_seq = [
        [(2, 1), (4, 1), (7, 1), (6, 1), (8, 1), (9, 1)],
        [(7, 1), (6, 1), (8, 1), (9, 1)],
        [(7, 1), (1, 1), (6, 1), (2, 1), (8, 1), (4, 1), (9, 1)],
        [(7, 10), (6, 2), (8, 3), (9, 4)],
    ]
    for batch in operation_seq:
        apply_attention(kv_cache, batch, cached_k, cached_v)


@pytest.mark.skip(reason="Require FlashInfer enabled")
def test_paged_attention_kv_cache_popn(kv_cache):
    fclear(kv_cache)

    cached_k = {}
    cached_v = {}
    batch = [(0, 35), (1, 88), (2, 17), (3, 4)]
    apply_attention(kv_cache, batch, cached_k, cached_v)
    apply_attention(kv_cache, [((4, 3), 35)], cached_k, cached_v)

    popn_operations = [(0, 17), (1, 57), (2, 16), (3, 0), (4, 19)]
    for seq_id, pop_length in popn_operations:
        fpopn(kv_cache, seq_id, pop_length)
        if pop_length != 0:
            cached_k[seq_id] = cached_k[seq_id][:, :-pop_length, ...]
            cached_v[seq_id] = cached_v[seq_id][:, :-pop_length, ...]
        verify_cached_kv(kv_cache, seq_ids=list(range(4)), expected_k=cached_k, expected_v=cached_v)


if __name__ == "__main__":
    cache = create_kv_cache()
    test_paged_attention_kv_cache_prefill_and_decode(cache)
    test_paged_attention_kv_cache_remove_sequence(cache)
    test_paged_attention_kv_cache_fork_sequence(cache)
    test_paged_attention_kv_cache_popn(cache)
