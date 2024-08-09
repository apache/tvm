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
import enum
import itertools
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import scipy.special

import tvm
import tvm.testing
from tvm import dlight as dl
from tvm.relax.frontend.nn.llm.kv_cache import (
    _attention_decode,
    _attention_prefill,
    _attention_prefill_ragged,
    _compact_kv_copy,
    _copy_single_page,
    _kv_cache_debug_get_kv,
    _kv_cache_transpose_append,
    _merge_state_inplace,
    llama_rope_with_position_map,
    tree_attn,
)
from tvm.runtime import ShapeTuple

reserved_nseq = 32
maximum_total_seq_length = 2048
prefill_chunk_size = 512
page_size = 16
num_layers = 4
num_qo_heads = 32
num_kv_heads = 4
head_dim = None
rope_scale = 1.0
rope_theta = 1e4
dtype = None
device = tvm.cuda()

fclear = None
fadd_sequence = None
fremove_sequence = None
ffork_sequence = None
fenable_sliding_window_for_seq = None
fpopn = None
fbegin_forward = None
fend_forward = None
fcommit_accepted_token_tree_nodes = None
fattention_with_fuse_qkv = None
fis_empty = None
fdebug_get_kv = None

ftranspose_append = None
fcopy_cache = None
fattn_prefill = None
fattn_decode = None
fattn_prefill_sliding_window = None
fattn_decode_sliding_window = None
fattn_prefill_ragged = None
fattn_prefill_with_tree_mask = None
fmerge_state = None
fsplit_rotary = None
fattention_rotary = None
fcopy_single_page = None
fcompact_copy = None


def set_global_func(head_dim, dtype):
    global fclear, fadd_sequence, fremove_sequence, ffork_sequence, fenable_sliding_window_for_seq
    global fpopn, fbegin_forward, fend_forward, fcommit_accepted_token_tree_nodes
    global fattention_with_fuse_qkv, fis_empty, fdebug_get_kv
    global ftranspose_append, fcopy_cache, fattn_prefill, fattn_decode
    global fattn_prefill_ragged, fattn_prefill_with_tree_mask
    global fattn_prefill_sliding_window, fattn_decode_sliding_window
    global fmerge_state, fsplit_rotary, fattention_rotary, fcopy_single_page, fcompact_copy

    fclear = tvm.get_global_func("vm.builtin.kv_state_clear")
    fadd_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    fremove_sequence = tvm.get_global_func("vm.builtin.kv_state_remove_sequence")
    ffork_sequence = tvm.get_global_func("vm.builtin.kv_state_fork_sequence")
    fenable_sliding_window_for_seq = tvm.get_global_func(
        "vm.builtin.attention_kv_cache_enable_sliding_window_for_seq"
    )
    fpopn = tvm.get_global_func("vm.builtin.kv_state_popn")
    fbegin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    fend_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    fcommit_accepted_token_tree_nodes = tvm.get_global_func(
        "vm.builtin.attention_kv_cache_commit_accepted_token_tree_nodes"
    )
    fattention_with_fuse_qkv = tvm.get_global_func(
        "vm.builtin.attention_kv_cache_attention_with_fused_qkv"
    )
    fis_empty = tvm.get_global_func("vm.builtin.attention_kv_cache_empty")
    fdebug_get_kv = tvm.get_global_func("vm.builtin.attention_kv_cache_debug_get_kv")

    target = tvm.target.Target("cuda")
    builts = []
    for tir_func in [
        _kv_cache_transpose_append(num_kv_heads, head_dim, dtype),
        _kv_cache_debug_get_kv(num_layers, num_kv_heads, head_dim, dtype),
        _attention_prefill(num_kv_heads, num_qo_heads, head_dim, dtype, False, target),
        _attention_decode(num_kv_heads, num_qo_heads, head_dim, dtype, False, target),
        _attention_prefill(num_kv_heads, num_qo_heads, head_dim, dtype, True, target),
        _attention_decode(num_kv_heads, num_qo_heads, head_dim, dtype, True, target),
        _attention_prefill_ragged(num_kv_heads, num_qo_heads, head_dim, dtype, target),
        tree_attn(num_kv_heads, num_qo_heads, head_dim, dtype, target),
        _merge_state_inplace(num_qo_heads, head_dim, dtype, target),
        llama_rope_with_position_map(
            rope_theta, rope_scale, head_dim, num_qo_heads, num_kv_heads, dtype
        ),
        _copy_single_page(num_kv_heads, page_size, head_dim, dtype, target),
        _compact_kv_copy(num_kv_heads, head_dim, dtype, target),
    ]:
        mod = tvm.IRModule({"main": tir_func})
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
        f = tvm.build(mod["main"], target=target)
        builts.append(f.entry_func)

    (
        ftranspose_append,
        fcopy_cache,
        fattn_prefill,
        fattn_decode,
        fattn_prefill_sliding_window,
        fattn_decode_sliding_window,
        fattn_prefill_ragged,
        fattn_prefill_with_tree_mask,
        fmerge_state,
        fsplit_rotary,
        fcopy_single_page,
        fcompact_copy,
    ) = builts


def create_kv_cache(head_dim, dtype, rope_mode, support_sliding_window):
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create_reduced")
    cache = fcreate(
        tvm.runtime.ShapeTuple(
            [
                reserved_nseq,
                maximum_total_seq_length,
                prefill_chunk_size,
                page_size,
                int(support_sliding_window),
            ]
        ),
        tvm.runtime.ShapeTuple([0, num_layers]),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        rope_mode,
        rope_scale,
        rope_theta,
        tvm.nd.empty((), dtype, device=device),
        ftranspose_append,
        fattn_prefill,
        fattn_decode,
        fattn_prefill_sliding_window,
        fattn_decode_sliding_window,
        fattn_prefill_ragged,
        fmerge_state,
        fsplit_rotary,
        fcopy_single_page,
        fcopy_cache,
        fcompact_copy,
        fattn_prefill_with_tree_mask,
    )
    return cache


class RopeMode(enum.IntEnum):
    """The RoPE mode of the Paged KV cache.
    If it is none, the KV cache will not apply RoPE to q and k.
    If it is normal, RoPE will be applied to k before adding k to cache.
    Otherwise, RoPE will be applied to q/k in attention kernel on-the-fly.
    """

    NONE = 0
    NORMAL = 1
    INLINE = 2


@pytest.fixture(
    params=itertools.chain(
        itertools.product(
            [64, 128],
            ["float16", "float32"],
            [RopeMode.NORMAL],
            [False],
        ),
        itertools.product(
            [128],
            ["float16"],
            [RopeMode.NONE, RopeMode.INLINE],
            [False, True],
        ),
    )
)
def kv_cache_and_config(request):
    global head_dim, dtype
    head_dim, dtype, rope_mode, support_sliding_window = request.param
    set_global_func(head_dim, dtype)
    return create_kv_cache(*request.param), rope_mode, support_sliding_window


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


def f_apply_rotary(x, offset, scale, theta, offset_list: Optional[List[int]] = None):
    # x: (N, H, D)
    assert len(x.shape) == 3
    nfeat = x.shape[-1]
    nfeat_half = x.shape[-1] // 2
    x = x.astype("float32")
    y = np.concatenate([-x[:, :, nfeat_half:], x[:, :, :nfeat_half]], axis=-1)

    inv_freq = scale / (theta ** (np.arange(0, nfeat, 2).astype("float32") / nfeat))
    t = (
        np.arange(offset, offset + x.shape[0], dtype=inv_freq.dtype)
        if offset_list is None
        else (np.array(offset_list, dtype=inv_freq.dtype) + offset)
    )
    freqs = np.einsum("i,j->ij", t, inv_freq)
    emb = np.concatenate((freqs, freqs), axis=-1)
    cos_values = np.cos(emb)
    sin_values = np.sin(emb)

    return np.einsum("ij,ikj->ikj", cos_values, x) + np.einsum("ij,ikj->ikj", sin_values, y)


def apply_attention(
    kv_cache,
    rope_mode: RopeMode,
    batch: List[Tuple[Union[int, Tuple[int, int, int]], int]],
    cached_k: Dict[int, np.ndarray],
    cached_v: Dict[int, np.ndarray],
    sliding_window_sizes: Optional[List[int]] = None,
    attn_sink_sizes: Optional[List[int]] = None,
    token_tree_parent_ptr_list: Optional[List[List[int]]] = None,
    accepted_leaf_indices: Optional[List[int]] = None,
) -> None:
    seq_ids = []
    append_lengths = []
    for i, (seq_id, append_length) in enumerate(batch):
        fork_parent_id = None
        if isinstance(seq_id, tuple):
            # Fork sequence
            seq_id, fork_parent_id, fork_pos = seq_id
            batch[i] = (seq_id, append_length)
        seq_ids.append(seq_id)
        append_lengths.append(append_length)
        if fork_parent_id is not None:
            assert fork_parent_id in cached_k
            assert seq_id not in cached_k
            ffork_sequence(kv_cache, fork_parent_id, seq_id, fork_pos)
            if fork_pos == -1:
                cached_k[seq_id] = cached_k[fork_parent_id]
                cached_v[seq_id] = cached_v[fork_parent_id]
            else:
                cached_k[seq_id] = cached_k[fork_parent_id][::, :fork_pos]
                cached_v[seq_id] = cached_v[fork_parent_id][::, :fork_pos]
        elif seq_id not in cached_k:
            fadd_sequence(kv_cache, seq_id)
            cached_k[seq_id] = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)
            cached_v[seq_id] = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)

    assert (token_tree_parent_ptr_list is None) == (accepted_leaf_indices is None)
    flattened_token_tree_parent_ptr = None
    token_tree_node_depths_list: List[Optional[List[int]]] = [None for _ in batch]
    if token_tree_parent_ptr_list:
        assert len(token_tree_node_depths_list) == len(seq_ids)
        assert len(accepted_leaf_indices) == len(seq_ids)
        flattened_token_tree_parent_ptr = []
        for i, (token_tree_parent_ptr, append_length) in enumerate(
            zip(token_tree_parent_ptr_list, append_lengths)
        ):
            assert len(token_tree_parent_ptr) == append_length
            flattened_token_tree_parent_ptr += token_tree_parent_ptr
            token_tree_node_depths = []
            for parent in token_tree_parent_ptr:
                token_tree_node_depths.append(
                    0 if parent == -1 else token_tree_node_depths[parent] + 1
                )
            token_tree_node_depths_list[i] = token_tree_node_depths

    fbegin_forward(
        kv_cache,
        ShapeTuple(seq_ids),
        ShapeTuple(append_lengths),
        (
            ShapeTuple(flattened_token_tree_parent_ptr)
            if flattened_token_tree_parent_ptr is not None
            else None
        ),
    )

    global_new_q = np.zeros((num_layers, 0, num_qo_heads, head_dim), dtype)
    global_new_k = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)
    global_new_v = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)

    q_array = []
    for i, (seq_id, append_length) in enumerate(batch):
        new_q = np.random.rand(num_layers, append_length, num_qo_heads, head_dim).astype(dtype)
        new_k = np.random.rand(num_layers, append_length, num_kv_heads, head_dim).astype(dtype)
        new_v = np.random.rand(num_layers, append_length, num_kv_heads, head_dim).astype(dtype)
        q_array.append(new_q)

        cached_k[seq_id] = np.concatenate(
            [
                cached_k[seq_id],
                np.stack(
                    [
                        (
                            new_k[l]
                            if rope_mode != RopeMode.NORMAL
                            else f_apply_rotary(
                                new_k[l],
                                cached_k[seq_id].shape[1],
                                rope_scale,
                                rope_theta,
                                token_tree_node_depths_list[i],
                            )
                        )
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
        queries_np = global_new_q[layer_id]
        keys_np = global_new_k[layer_id]
        values_np = global_new_v[layer_id]
        qkv = tvm.nd.array(np.concatenate([queries_np, keys_np, values_np], axis=1), device)
        outputs = tvm.nd.empty(queries_np.shape, dtype, device=device)
        fattention_with_fuse_qkv(kv_cache, layer_id, 1.0, qkv, outputs)

        # Compute attention expected results.
        outputs = np.expand_dims(outputs.numpy(), axis=0)
        sum_length = 0
        for i, (seq_id, append_length) in enumerate(batch):
            assert cached_k[seq_id].shape[1] == cached_v[seq_id].shape[1] >= append_length

            rope_offset = cached_k[seq_id].shape[1] - append_length
            q_seq = (
                q_array[i][layer_id]
                if rope_mode == RopeMode.NONE
                else f_apply_rotary(
                    q_array[i][layer_id],
                    rope_offset,
                    rope_scale,
                    rope_theta,
                    token_tree_node_depths_list[i],
                )
            ).transpose(1, 0, 2)
            k_seq = (
                cached_k[seq_id][layer_id]
                if rope_mode != RopeMode.INLINE
                else f_apply_rotary(
                    cached_k[seq_id][layer_id],
                    0,
                    rope_scale,
                    rope_theta,
                    (
                        (
                            list(range(rope_offset))
                            + [depth + rope_offset for depth in token_tree_node_depths_list[i]]
                        )
                        if token_tree_node_depths_list[i] is not None
                        else None
                    ),
                )
            ).transpose(1, 2, 0)
            v_seq = cached_v[seq_id][layer_id].transpose(1, 0, 2)

            k_seq = np.repeat(k_seq, num_qo_heads // num_kv_heads, axis=0)
            v_seq = np.repeat(v_seq, num_qo_heads // num_kv_heads, axis=0)
            softmax_input = (q_seq.astype("float32") @ k_seq.astype("float32")) / np.sqrt(head_dim)
            softmax_shape = softmax_input.shape
            assert softmax_shape[-2] == append_length
            length_diff = softmax_shape[-1] - softmax_shape[-2]
            assert length_diff >= 0
            mask = np.tril(
                np.full_like(softmax_input, np.finfo("float32").max), k=length_diff
            ) + np.triu(np.full_like(softmax_input, np.finfo("float32").min), k=length_diff + 1)
            if token_tree_parent_ptr_list is not None:
                tree_mask = np.full(
                    (append_length, append_length), np.finfo("float32").min, dtype="float32"
                )
                for i, parent in enumerate(token_tree_parent_ptr_list[i]):
                    if parent != -1:
                        tree_mask[i] = tree_mask[parent]
                    tree_mask[i, i] = np.finfo("float32").max
                tree_mask = np.broadcast_to(tree_mask, (num_qo_heads, *tree_mask.shape))
                mask[:, :, length_diff:] = tree_mask

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

    if accepted_leaf_indices is not None:
        seq_ids = [seq_id for seq_id, _ in batch]
        fcommit_accepted_token_tree_nodes(
            kv_cache, ShapeTuple(seq_ids), ShapeTuple(accepted_leaf_indices)
        )
        for i, (accepted_leaf_idx, (seq_id, append_length)) in enumerate(
            zip(accepted_leaf_indices, batch)
        ):
            tree_path = []
            node = accepted_leaf_idx
            while node != -1:
                tree_path.append(node)
                node = token_tree_parent_ptr_list[i][node]
            offset = cached_k[seq_id].shape[1] - append_length
            length_to_pop = append_length - len(tree_path)
            assert 0 <= length_to_pop <= append_length
            for dst_pos, src_pos in enumerate(reversed(tree_path)):
                if dst_pos == src_pos:
                    continue
                cached_k[seq_id][:, offset + dst_pos, ...] = cached_k[seq_id][
                    :, offset + src_pos, ...
                ]
                cached_v[seq_id][:, offset + dst_pos, ...] = cached_v[seq_id][
                    :, offset + src_pos, ...
                ]
            if length_to_pop > 0:
                cached_k[seq_id] = cached_k[seq_id][:, :-length_to_pop, ...]
                cached_v[seq_id] = cached_v[seq_id][:, :-length_to_pop, ...]

    for seq_id, _ in batch:
        if sliding_window_sizes is not None and len(sliding_window_sizes) > seq_id:
            assert len(sliding_window_sizes) > seq_id and len(attn_sink_sizes) > seq_id
            sliding_window_size = sliding_window_sizes[seq_id]
            attn_sink_size = attn_sink_sizes[seq_id]
            if sliding_window_size == 0:
                continue
            if cached_k[seq_id].shape[1] > sliding_window_size:
                # Apply sliding window and sink to cached kv.
                length_to_slide = cached_k[seq_id].shape[1] - sliding_window_size
                cached_k[seq_id] = np.concatenate(
                    [
                        cached_k[seq_id][:, :attn_sink_size, ...],
                        cached_k[seq_id][:, attn_sink_size + length_to_slide :, ...],
                    ],
                    axis=1,
                )
                cached_v[seq_id] = np.concatenate(
                    [
                        cached_v[seq_id][:, :attn_sink_size, ...],
                        cached_v[seq_id][:, attn_sink_size + length_to_slide :, ...],
                    ],
                    axis=1,
                )
                assert cached_k[seq_id].shape[1] == sliding_window_size

    # Verify
    verify_cached_kv(kv_cache, seq_ids, cached_k, cached_v)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_paged_attention_kv_cache_prefill_and_decode(kv_cache_and_config):
    kv_cache, rope_mode, support_sliding_window = kv_cache_and_config
    if support_sliding_window and rope_mode == RopeMode.NORMAL:
        # Normal RoPE mode under sliding window settings is not supported.
        return
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
        apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_paged_attention_kv_cache_remove_sequence(kv_cache_and_config):
    kv_cache, rope_mode, support_sliding_window = kv_cache_and_config
    if support_sliding_window and rope_mode == RopeMode.NORMAL:
        # Normal RoPE mode under sliding window settings is not supported.
        return
    fclear(kv_cache)

    num_sequences = 5
    batch = [(seq_id, 1) for seq_id in range(num_sequences)]
    cached_k = {}
    cached_v = {}
    for seq_id_to_remove in range(num_sequences):
        apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v)
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


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_paged_attention_kv_cache_fork_sequence(kv_cache_and_config):
    kv_cache, rope_mode, support_sliding_window = kv_cache_and_config
    if support_sliding_window and rope_mode == RopeMode.NORMAL:
        # Normal RoPE mode under sliding window settings is not supported.
        return
    fclear(kv_cache)

    cached_k = {}
    cached_v = {}
    batch = [(0, 60), (1, 88), (2, 17), (3, 4)]
    apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v)
    # Fork existing sequences.
    apply_attention(kv_cache, rope_mode, [((4, 3, -1), 35)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((5, 0, -1), 20)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((6, 5, -1), 102)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((7, 0, -1), 3)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((8, 5, -1), 71), ((9, 5, -1), 20)], cached_k, cached_v)
    # 0 <- 5 <- 6,8,9
    # 0 <- 7
    # 3 <- 4
    # Mixture of decode and prefill.
    operation_seq = [
        [(2, 1), (4, 1), (7, 1), (6, 1), (8, 1), (9, 1)],
        [(7, 1), (6, 1), (8, 1), (9, 1)],
        [(7, 1), (1, 1), (6, 1), (2, 1), (8, 1), (4, 1), (9, 1)],
        [(7, 10), (6, 2), (8, 3), (9, 4)],
    ]
    for batch in operation_seq:
        apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v)

    apply_attention(kv_cache, rope_mode, [((10, 1, 33), 11)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((11, 0, 60), 45), ((12, 0, 15), 14)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((13, 0, 16), 19), ((14, 0, 17), 19)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((15, 5, 60), 8), ((16, 5, 80), 10)], cached_k, cached_v)
    apply_attention(
        kv_cache,
        rope_mode,
        [((17, 5, 75), 11), ((18, 5, 76), 45), ((19, 5, 77), 14)],
        cached_k,
        cached_v,
    )

    operation_seq = [
        [(6, 1), (11, 1), (13, 1), (9, 1)],
        [(10, 1), (16, 1), (18, 1), (19, 1)],
        [(8, 1), (15, 1), (17, 1), (12, 1), (14, 1)],
        [(10, 10), (6, 2), (8, 3), (19, 4)],
    ]
    for batch in operation_seq:
        apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v)

    num_sequence = 20
    for i in range(num_sequence):
        fremove_sequence(kv_cache, i)
        cached_k.pop(i)
        cached_v.pop(i)
        verify_cached_kv(
            kv_cache,
            seq_ids=list(range(i + 1, num_sequence)),
            expected_k=cached_k,
            expected_v=cached_v,
        )

    assert fis_empty(kv_cache), "The KV cache is not empty after removing all sequences"

    # Test fork after page recycle
    apply_attention(kv_cache, rope_mode, [(0, 7), (1, 24)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((2, 1, -1), 10)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((3, 0, -1), 20)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [(2, 1), (3, 1)], cached_k, cached_v)

    apply_attention(kv_cache, rope_mode, [(10, 7), (11, 24)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((12, 11, -1), 200)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [(10, 1), (12, 1)], cached_k, cached_v)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_paged_attention_kv_cache_unlimited_depth(kv_cache_and_config):
    kv_cache, rope_mode, support_sliding_window = kv_cache_and_config
    if support_sliding_window and rope_mode == RopeMode.NORMAL:
        # Normal RoPE mode under sliding window settings is not supported.
        return
    fclear(kv_cache)

    cached_k = {}
    cached_v = {}
    apply_attention(kv_cache, rope_mode, [(0, 30)], cached_k, cached_v)
    # Fork existing sequences.
    apply_attention(kv_cache, rope_mode, [((1, 0, -1), 15)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((2, 1, -1), 5)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((3, 2, -1), 20)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((4, 3, -1), 26)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((5, 3, -1), 18)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((6, 5, -1), 22)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((7, 5, -1), 12)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((8, 7, -1), 29)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((9, 7, -1), 9)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((10, 9, -1), 31)], cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((11, 9, -1), 4)], cached_k, cached_v)
    # 0 <- 1 <- 2 <- 3 <- 5 <- 7 <- 9 <- 11
    #                |    |    |    |
    #                4    6    8    10
    # Decode.
    operation_seq = [
        [(3, 1), (6, 1), (9, 1)],
        [(4, 1), (8, 1), (10, 1)],
        [(5, 1), (7, 1), (11, 1)],
    ]
    for batch in operation_seq:
        apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v)

    num_sequence = 12
    for i in range(num_sequence):
        fremove_sequence(kv_cache, i)
        cached_k.pop(i)
        cached_v.pop(i)
        verify_cached_kv(
            kv_cache,
            seq_ids=list(range(i + 1, num_sequence)),
            expected_k=cached_k,
            expected_v=cached_v,
        )

    assert fis_empty(kv_cache), "The KV cache is not empty after removing all sequences"


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_paged_attention_kv_cache_popn(kv_cache_and_config):
    kv_cache, rope_mode, support_sliding_window = kv_cache_and_config
    if support_sliding_window and rope_mode == RopeMode.NORMAL:
        return
    fclear(kv_cache)

    cached_k = {}
    cached_v = {}
    batch = [(0, 35), (1, 88), (2, 17), (3, 4)]
    apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v)
    apply_attention(kv_cache, rope_mode, [((4, 3, -1), 35)], cached_k, cached_v)

    popn_operations = [(0, 17), (1, 57), (2, 16), (3, 0), (4, 37)]
    for seq_id, pop_length in popn_operations:
        fpopn(kv_cache, seq_id, pop_length)
        if pop_length != 0:
            cached_k[seq_id] = cached_k[seq_id][:, :-pop_length, ...]
            cached_v[seq_id] = cached_v[seq_id][:, :-pop_length, ...]
        verify_cached_kv(kv_cache, seq_ids=list(range(4)), expected_k=cached_k, expected_v=cached_v)

    num_sequence = 5
    for seq_id in range(num_sequence):
        fremove_sequence(kv_cache, seq_id)
        verify_cached_kv(
            kv_cache,
            seq_ids=list(range(seq_id + 1, num_sequence)),
            expected_k=cached_k,
            expected_v=cached_v,
        )

    assert fis_empty(kv_cache), "The KV cache is not empty after removing all sequences"


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_paged_attention_kv_cache_sliding_window(kv_cache_and_config):
    kv_cache, rope_mode, support_sliding_window = kv_cache_and_config
    if not support_sliding_window or rope_mode == RopeMode.NORMAL:
        return
    fclear(kv_cache)

    cached_k = {}
    cached_v = {}
    sliding_window_sizes = [20, 25, 30, 35, 40]
    attn_sink_sizes = [6, 4, 8, 3, 7]
    for seq_id, (sliding_window_size, attn_sink_size) in enumerate(
        zip(sliding_window_sizes, attn_sink_sizes)
    ):
        fadd_sequence(kv_cache, seq_id)
        fenable_sliding_window_for_seq(kv_cache, seq_id, sliding_window_size, attn_sink_size)
        cached_k[seq_id] = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)
        cached_v[seq_id] = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)

    # Prefill.
    operation_seq = [[(0, 4)], [(1, 6)], [(2, 6), (3, 7), (4, 7)]]
    operation_seq += [[(0, 20), (1, 19), (2, 30), (3, 35), (4, 40)]]
    operation_seq += [[(0, 6), (1, 5), (2, 4), (3, 3), (4, 2)]]
    for batch in operation_seq:
        apply_attention(
            kv_cache,
            rope_mode,
            batch,
            cached_k,
            cached_v,
            sliding_window_sizes,
            attn_sink_sizes,
        )
    # Decode
    batch = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
    for _ in range(20):
        apply_attention(
            kv_cache,
            rope_mode,
            batch,
            cached_k,
            cached_v,
            sliding_window_sizes,
            attn_sink_sizes,
        )


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_paged_attention_kv_cache_sliding_window_fork(kv_cache_and_config):
    kv_cache, rope_mode, support_sliding_window = kv_cache_and_config
    if not support_sliding_window or rope_mode == RopeMode.NORMAL:
        return
    fclear(kv_cache)

    cached_k = {}
    cached_v = {}
    sliding_window_sizes = [30, 35, 40]
    attn_sink_sizes = [15, 20, 25]
    for seq_id, (sliding_window_size, attn_sink_size) in enumerate(
        zip(sliding_window_sizes, attn_sink_sizes)
    ):
        fadd_sequence(kv_cache, seq_id)
        fenable_sliding_window_for_seq(kv_cache, seq_id, sliding_window_size, attn_sink_size)
        cached_k[seq_id] = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)
        cached_v[seq_id] = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)
    apply_attention(
        kv_cache,
        rope_mode,
        [(0, 12), (1, 18), (2, 28)],
        cached_k,
        cached_v,
        sliding_window_sizes,
        attn_sink_sizes,
    )
    # seq_len: [12, 18, 25+3]
    sliding_window_sizes += [0, 0, 0]
    attn_sink_sizes += [0, 0, 0]
    apply_attention(
        kv_cache,
        rope_mode,
        [((3, 0, 10), 8), ((4, 1, -1), 20), ((5, 2, 18), 18)],
        cached_k,
        cached_v,
        sliding_window_sizes,
        attn_sink_sizes,
    )
    # seq_len: [12, 18, 25+3, 18, 38, 36]
    apply_attention(
        kv_cache,
        rope_mode,
        [(0, 9), (1, 15), (2, 4), (3, 10), (4, 3), (5, 7)],
        cached_k,
        cached_v,
        sliding_window_sizes,
        attn_sink_sizes,
    )
    # seq_len: [15+6, 20+13, 25+7, 28, 41, 43]
    sliding_window_sizes += [25]
    attn_sink_sizes += [24]
    ffork_sequence(kv_cache, 3, 6, 18)
    fenable_sliding_window_for_seq(kv_cache, 6, sliding_window_sizes[-1], attn_sink_sizes[-1])
    cached_k[6] = cached_k[3][::, :18]
    cached_v[6] = cached_v[3][::, :18]
    apply_attention(
        kv_cache,
        rope_mode,
        [(3, 10), (6, 12)],
        cached_k,
        cached_v,
        sliding_window_sizes,
        attn_sink_sizes,
    )
    # seq_len: [15+6, 20+13, 25+7, 38, 41, 43, 24+6]


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_paged_attention_kv_cache_tree_attn(kv_cache_and_config):
    kv_cache, rope_mode, support_sliding_window = kv_cache_and_config
    if support_sliding_window and rope_mode == RopeMode.NORMAL:
        # Normal RoPE mode under sliding window settings is not supported.
        return
    fclear(kv_cache)

    cached_k = {}
    cached_v = {}
    # Prefill 4 sequences
    apply_attention(kv_cache, rope_mode, [(0, 10), (1, 20), (2, 30), (3, 40)], cached_k, cached_v)
    # Tree attention
    apply_attention(
        kv_cache,
        rope_mode,
        [(0, 7), (1, 15), (2, 10), (3, 14)],
        cached_k,
        cached_v,
        token_tree_parent_ptr_list=[
            [-1, 0, 0, 1, 1, 2, 2],  # complete binary tree of height 3
            [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],  # complete binary tree of height 4
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8],  # chain of length 10
            [-1, 0, 0, 1, 1, 2, 2, -1, 7, 7, 8, 8, 9, 9],  # two complete binary trees of height 3
        ],
        accepted_leaf_indices=[6, 11, 6, 13],
    )
    # Do 5 rounds of decode.
    for _ in range(5):
        apply_attention(kv_cache, rope_mode, [(0, 1), (1, 1), (2, 1), (3, 1)], cached_k, cached_v)

    # Test the cases where all trees are chains.
    fclear(kv_cache)
    cached_k = {}
    cached_v = {}
    # Prefill 4 sequences
    apply_attention(kv_cache, rope_mode, [(0, 10), (1, 20), (2, 30), (3, 40)], cached_k, cached_v)
    # Tree attention
    apply_attention(
        kv_cache,
        rope_mode,
        [(0, 7), (1, 15), (2, 10), (3, 14)],
        cached_k,
        cached_v,
        token_tree_parent_ptr_list=[
            [-1, 0, 1, 2, 3, 4, 5],  # complete binary tree of height 7
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],  # chain of length 15
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8],  # chain of length 10
            [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # chain of length 14
        ],
        accepted_leaf_indices=[2, 6, -1, 4],
    )
    # Do 5 rounds of decode.
    for _ in range(5):
        apply_attention(kv_cache, rope_mode, [(0, 1), (1, 1), (2, 1), (3, 1)], cached_k, cached_v)


if __name__ == "__main__":
    HEAD_DIMS = [64, 128]
    DTYPES = ["float16", "float32"]
    ROPE_MODES = [RopeMode.NONE, RopeMode.NORMAL, RopeMode.INLINE]
    SUPPORT_SLIDING_WINDOW = [False, True]
    for head_dim, dtype, rope_mode, support_sliding_window in itertools.product(
        HEAD_DIMS, DTYPES, ROPE_MODES, SUPPORT_SLIDING_WINDOW
    ):
        set_global_func(head_dim, dtype)
        cache = create_kv_cache(head_dim, dtype, rope_mode, support_sliding_window)
        cache_and_config = (cache, rope_mode, support_sliding_window)
        test_paged_attention_kv_cache_prefill_and_decode(cache_and_config)
        test_paged_attention_kv_cache_remove_sequence(cache_and_config)
        test_paged_attention_kv_cache_fork_sequence(cache_and_config)
        test_paged_attention_kv_cache_popn(cache_and_config)
        test_paged_attention_kv_cache_sliding_window(cache_and_config)
        test_paged_attention_kv_cache_tree_attn(cache_and_config)
        test_paged_attention_kv_cache_unlimited_depth(cache_and_config)
