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
import math
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytest
import scipy.special

import tvm
import tvm.testing
from tvm import DataType
from tvm import dlight as dl
from tvm import tir
from tvm.runtime import ShapeTuple
from tvm.script import tir as T
from tvm.target import Target

reserved_nseq = 32
maximum_total_seq_length = 1024
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
        kv_cache_transpose_append(head_dim, dtype),
        copy_cache(head_dim, dtype),
        _attention_prefill(num_kv_heads, num_qo_heads, head_dim, dtype, False, target),
        _attention_decode(num_kv_heads, num_qo_heads, head_dim, dtype, False, target),
        _attention_prefill(num_kv_heads, num_qo_heads, head_dim, dtype, True, target),
        _attention_decode(num_kv_heads, num_qo_heads, head_dim, dtype, True, target),
        _attention_prefill_ragged(num_kv_heads, num_qo_heads, head_dim, dtype, target),
        _attention_prefill_with_tree_mask(num_kv_heads, num_qo_heads, head_dim, dtype, target),
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
    num_storage = head_dim
    kv_storage_dtype = dtype

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
        num_layers,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        rope_mode,
        rope_scale,
        rope_theta,
        num_storage,
        tvm.nd.empty((), dtype, device=device),
        tvm.nd.empty((), kv_storage_dtype, device=device),
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


def kv_cache_transpose_append(head_dim, dtype):
    # undefined vars used
    @T.prim_func(check_well_formed=False)
    def _kv_cache_transpose_append(
        var_pages: T.handle,
        var_k_data: T.handle,
        var_v_data: T.handle,
        var_position_map: T.handle,
    ):
        ntoken = T.SizeVar("ntoken", "int32")
        num_pages = T.int32()
        position_map_elem_offset = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_kv_heads, 16, head_dim), dtype)
        k_data = T.match_buffer(var_k_data, (ntoken, num_kv_heads, head_dim), dtype)
        v_data = T.match_buffer(var_v_data, (ntoken, num_kv_heads, head_dim), dtype)
        position_map = T.match_buffer(
            var_position_map, (ntoken,), "int32", elem_offset=position_map_elem_offset
        )

        for global_pos, h, f in T.grid(ntoken, num_kv_heads, head_dim):
            if position_map[global_pos] != T.int32(-1):
                with T.block("k_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                    T.writes(pages[position_map[vgpos] // 16, 0, vh, position_map[vgpos] % 16, vf])
                    position: T.int64 = T.Cast("int64", position_map[vgpos])
                    pages[T.floordiv(position, 16), 0, vh, T.floormod(position, 16), vf] = k_data[
                        vgpos, vh, vf
                    ]
                with T.block("v_transpose_append"):
                    vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
                    T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
                    T.writes(pages[position_map[vgpos] // 16, 1, vh, position_map[vgpos] % 16, vf])
                    position: T.int64 = T.Cast("int64", position_map[vgpos])
                    pages[T.floordiv(position, 16), 1, vh, T.floormod(position, 16), vf] = v_data[
                        vgpos, vh, vf
                    ]

    return _kv_cache_transpose_append


def copy_cache(head_dim, dtype):
    # undefined vars used
    @T.prim_func(check_well_formed=False)
    def _copy_cache(
        var_pages: T.handle,
        var_position_map: T.handle,
        var_k_data: T.handle,
        var_v_data: T.handle,
        layer_id: T.int64,
    ):
        num_kv_heads = T.int64()
        seqlen = T.SizeVar("seqlen", "int64")
        page_size = T.int64()
        num_pages = T.int64()
        position_map_elem_offset = T.int64()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_kv_heads, page_size, head_dim), dtype)
        position_map = T.match_buffer(
            var_position_map, (seqlen,), "int32", elem_offset=position_map_elem_offset
        )
        k_data = T.match_buffer(var_k_data, (num_layers, seqlen, num_kv_heads, head_dim), dtype)
        v_data = T.match_buffer(var_v_data, (num_layers, seqlen, num_kv_heads, head_dim), dtype)

        for p, h, d in T.grid(seqlen, num_kv_heads, head_dim):
            with T.block("copy0"):
                vp, vh, vd = T.axis.remap("SSS", [p, h, d])
                T.reads(
                    position_map[vp],
                    pages[position_map[vp] // page_size, 0:2, vh, position_map[vp] % page_size, vd],
                )
                T.writes(k_data[layer_id, vp, vh, vd], v_data[layer_id, vp, vh, vd])
                position: T.int64 = T.Cast("int64", position_map[vp])
                k_data[layer_id, vp, vh, vd] = pages[
                    T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vd
                ]
                v_data[layer_id, vp, vh, vd] = pages[
                    T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vd
                ]

    return _copy_cache


def llama_rope_with_position_map(  # pylint: disable=too-many-arguments
    theta: float,
    scale: float,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: float = "float16",
    rotary_dim: int = None,
):
    fused_heads = num_q_heads + num_kv_heads * 2
    if rotary_dim is None:
        rotary_dim = head_dim
    scale = tir.const(scale, dtype)

    def _rope_freq(s: tir.Var, d: tir.Var, d_range: int, theta: float, dtype: str):
        freq = s / tir.power(theta, d * 2 % d_range / tir.const(d_range, "float32"))
        cos_freq = tir.cos(freq).astype(dtype)
        sin_freq = tir.sin(freq).astype(dtype)
        return cos_freq, sin_freq

    def _rope(  # pylint: disable=too-many-arguments
        x: T.Buffer,
        s: tir.Var,
        h: tir.Var,
        d: tir.Var,
        pos: tir.Var,
    ):
        cos_freq, sin_freq = _rope_freq(pos * scale, d, rotary_dim, theta, dtype)
        cos = cos_freq * x[s, h, d]
        sin = sin_freq * tir.if_then_else(
            d < rotary_dim // 2,
            -x[s, h, d + rotary_dim // 2],
            x[s, h, d - rotary_dim // 2],
        )
        return cos + sin

    # undefined vars used
    @T.prim_func(private=True, check_well_formed=False)
    def fused_rope(  # pylint: disable=too-many-locals
        var_qkv: T.handle,
        var_position_map: T.handle,
        var_q: T.handle,
        var_k: T.handle,
        var_v: T.handle,
        apply_rope: T.int32,
    ):
        T.func_attr(
            {
                "op_pattern": 8,  # 2 means injective, 8 means opaque
                "tir.noalias": T.bool(True),
            }
        )
        seq_len = T.int64()
        position_map_elem_offset = T.int64()
        qkv = T.match_buffer(var_qkv, (seq_len, fused_heads, head_dim), dtype)
        q = T.match_buffer(var_q, (seq_len, num_q_heads, head_dim), dtype)
        k = T.match_buffer(var_k, (seq_len, num_kv_heads, head_dim), dtype)
        v = T.match_buffer(var_v, (seq_len, num_kv_heads, head_dim), dtype)
        position_map = T.match_buffer(
            var_position_map, (seq_len,), "int32", elem_offset=position_map_elem_offset
        )
        for iters in T.grid(seq_len, fused_heads, head_dim):
            with T.block("llama_fused_rope"):
                s, h, d = T.axis.remap("SSS", iters)
                if h < num_q_heads:
                    q[s, h, d] = T.if_then_else(
                        apply_rope > 0 and d < rotary_dim,
                        _rope(qkv, s, h, d, position_map[s]),
                        qkv[s, h, d],
                    )
                elif h < num_q_heads + num_kv_heads:
                    k[s, h - num_q_heads, d] = T.if_then_else(
                        apply_rope > 0 and d < rotary_dim,
                        _rope(qkv, s, h, d, position_map[s]),
                        qkv[s, h, d],
                    )
                else:
                    v[s, h - (num_q_heads + num_kv_heads), d] = qkv[s, h, d]

    return fused_rope


def rope_freq(s: tir.Var, d: tir.Var, d_range: int, theta: float, dtype: str):
    """Compute the inverse frequency of RoPE and then return the cosine and sine of it.

    Parameters
    ----------
    s : tir.Var
        The position index.

    d : tir.Var
        The dimension index.

    d_range : int
        The maximum dimension index.

    theta : float
        The theta value in RoPE, which controls the frequency.

    dtype : str
        The data type of the output.

    Returns
    -------
    cos_freq : Tensor
        The cosine of the inverse frequency.

    sin_freq : Tensor
        The sine of the inverse frequency.
    """
    freq = s / tir.power(theta, d * 2 % d_range / tir.const(d_range, "float32"))
    cos_freq = tir.cos(freq).astype(dtype)
    sin_freq = tir.sin(freq).astype(dtype)
    return cos_freq, sin_freq


def _rope(  # pylint: disable=too-many-arguments
    buffer: T.Buffer,
    offset: tir.Var,
    rotary_dim: int,
    theta: tir.Var,
    scale: tir.Var,
    indices: Tuple[tir.Var, ...],
    qkv_dtype="float16",
):
    d = indices[-1]
    cos_freq, sin_freq = rope_freq(offset * scale, d, rotary_dim, theta, qkv_dtype)
    cos = cos_freq * buffer[indices]
    sin = sin_freq * tir.if_then_else(
        d < rotary_dim // 2,
        -buffer[indices[:-1] + (d + rotary_dim // 2,)],
        buffer[indices[:-1] + (d - rotary_dim // 2,)],
    )
    return cos + sin


def _var(dtype):
    return T.alloc_buffer((1,), dtype, scope="local")


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
    else:
        # ((num_pages - 1) * page_size + last_page_len) - sliding_window_offset + sink_size
        return (
            (num_pages - 1) * page_size
            + length_info[0, seq_id]
            - length_info[1, seq_id]
            + length_info[2, seq_id]
        )


def _get_seq_offset(pos, seq_id, length_info, sliding_window):
    if not sliding_window:
        return pos
    else:
        # pos if pos < sink_size else pos - sink_size + sliding_window_offset
        return T.if_then_else(
            pos < length_info[2, seq_id],
            pos,
            pos - length_info[2, seq_id] + length_info[1, seq_id],
        )


def get_max_num_threads_per_block(target: Target):
    """
    max(max_num_threads, max_threads_per_block); if latter does not exist, return max_num_threads.
    We add this method since some targets have both fields and `max_threads_per_block` is larger.
    """
    max_num_threads = target.max_num_threads
    max_threads_per_block = target.attrs.get("max_threads_per_block", None)
    if max_threads_per_block is None:
        return max_num_threads
    return max(max_num_threads, max_threads_per_block)


def _attention_prefill(
    h_kv, h_q, d, dtype, sliding_window: bool, target: Target
):  # pylint: disable=unused-argument
    # pylint: disable=invalid-name
    NUM_BLKS = 16
    LOAD_VEC = 8 // ((DataType(dtype).bits + 7) // 8)  # 8 bytes
    group_size = h_q // h_kv
    sm_scale = 1.0 / math.sqrt(float(d)) * math.log2(math.exp(1))

    bdx = 32
    num_warps = 4
    tile_x, tile_y, tile_z = 64 // ((DataType(dtype).bits + 7) // 8) // max(d // 128, 1), d, 16
    L_per_cta = tile_x // group_size

    # Otherwise we would exceed maxComputeWorkgroupStorageSize
    if (
        str(target.kind) == "webgpu"
        and ((d + 127) // 128) * ((DataType(dtype).bits + 15) // 16) >= 4
    ):
        tile_z = 8
        num_warps = 2

    # undefined vars used
    # pylint: disable=line-too-long,too-many-arguments,too-many-branches
    # fmt: off
    @T.prim_func(check_well_formed=False)
    def batch_prefill_paged_kv(
        _0: T.int32,  # pylint: disable=unused-argument
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
        attn_score_scaling_factor: T.float32,
    ):
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
        pages = T.match_buffer(var_pages, (max_num_pages, 2, h_kv, 16, d), dtype)
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
                        with T.block("attn"):
                            bx, by, ty, tx = T.axis.remap("SSSS", [lbx, lby, lty, ltx])
                            T.reads()
                            T.writes()
                            tile_id = _var("int32")
                            batch_idx = _var("int32")
                            batch_tiles = _var("int32")
                            batch_rows = _var("int32")
                            iterator = _var("int32")
                            kv_chunk_len = _var("int32")

                            Q_smem = T.alloc_buffer((tile_x, d), dtype, scope="shared")
                            K_smem = T.alloc_buffer((tile_z, d), dtype, scope="shared")
                            V_smem = T.alloc_buffer((tile_z, d), dtype, scope="shared")
                            S_smem = T.alloc_buffer((tile_x, tile_z), "float32", scope="shared")

                            S_local = T.alloc_buffer((tile_x, tile_z), "float32", scope="local")
                            O_local = T.alloc_buffer((tile_x, d), "float32", scope="local")

                            m_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")
                            m_prev_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")
                            d_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")

                            m_new = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")
                            m_prev = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")
                            d_new = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")

                            ## get tile_no, batch_idx, batch_tiles, batch_rows
                            tile_id[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = (q_indptr[1] - q_indptr[0]) * group_size
                            batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)
                            while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                # advance to next tile
                                while tile_id[0] >= batch_tiles[0] and batch_idx[0] < batch_size:
                                    tile_id[0] -= batch_tiles[0]
                                    batch_idx[0] += 1
                                    if batch_idx[0] < batch_size:
                                        b_idx: T.int32 = batch_idx[0]
                                        batch_rows[0] = (q_indptr[b_idx + 1] - q_indptr[b_idx]) * group_size
                                        batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)

                                if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                    b_idx: T.int32 = batch_idx[0]
                                    LH_start: T.int32 = tile_id[0] * tile_x
                                    q_indptr_val: T.int32 = q_indptr[b_idx]

                                    cur_page_indptr_begin: T.int32 = page_indptr[b_idx]
                                    cur_page_indptr_end: T.int32 = page_indptr[b_idx + 1]
                                    kv_chunk_len[0] = T.if_then_else(
                                        cur_page_indptr_begin != cur_page_indptr_end,
                                        _get_kv_chunk_len(cur_page_indptr_end - cur_page_indptr_begin, 16, b_idx, length_info, sliding_window),
                                        0
                                    )
                                    T.tvm_storage_sync("shared")

                                    # init states
                                    for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                        row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                        if row < tile_x:
                                            m_smem[row] = -5e4
                                            d_smem[row] = 1.0

                                    for li, lj in T.grid(tile_x, tile_y):
                                        with T.block("O_init"):
                                            i, j = T.axis.remap("SS", [li, lj])
                                            O_local[i, j] = 0.0
                                    T.tvm_storage_sync("shared")

                                    # Load Q from gmem to smem
                                    for li, lj in T.grid(tile_x, tile_y):
                                        with T.block("Q_load"):
                                            i, j = T.axis.remap("SS", [li, lj])
                                            T.reads()
                                            T.writes()
                                            cur_L = q_indptr_val + (LH_start + i) // group_size
                                            cur_H_qo = by * group_size + (LH_start + i) % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                Q_smem[i, j] = T.if_then_else(
                                                    rotary_mode == 1,
                                                    _rope(q, q_rope_position[cur_L], d, rope_theta, rope_scale, (cur_L, cur_H_qo, j), dtype),
                                                    q[cur_L, cur_H_qo, j]
                                                )
                                            else:
                                                Q_smem[i, j] = 0.0
                                    T.tvm_storage_sync("shared")

                                    for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_z)):
                                        L_kv_start: T.int32 = iterator * tile_z
                                        for lz, ly in T.grid(tile_z, tile_y):
                                            with T.block("K_load"):
                                                i, j = T.axis.remap("SS", [lz, ly])
                                                T.reads()
                                                T.writes()
                                                cur_L = L_kv_start + i
                                                if cur_L < kv_chunk_len[0]:
                                                    seq_offset: T.int32(is_size_var=True) = _get_seq_offset(cur_L, b_idx, length_info, sliding_window)  # type: ignore
                                                    page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + T.floordiv(seq_offset, 16)]  # type: ignore
                                                    page_offset: T.int32(is_size_var=True) = T.floormod(seq_offset, 16)  # type: ignore
                                                    K_smem[i, j] = T.if_then_else(
                                                        rotary_mode == 1,
                                                        _rope(pages, k_rope_pos_offset[b_idx] + cur_L, d, rope_theta, rope_scale, (page_no, 0, by, page_offset, j), dtype),
                                                        pages[page_no, 0, by, page_offset, j]
                                                    )
                                                else:
                                                    K_smem[i, j] = 0.0
                                        T.tvm_storage_sync("shared")
                                        for lz, ly in T.grid(tile_z, tile_y):
                                            with T.block("V_load"):
                                                i, j = T.axis.remap("SS", [lz, ly])
                                                T.reads()
                                                T.writes()
                                                cur_L = L_kv_start + i
                                                if cur_L < kv_chunk_len[0]:
                                                    seq_offset: T.int32(is_size_var=True) = _get_seq_offset(cur_L, b_idx, length_info, sliding_window)  # type: ignore
                                                    page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + T.floordiv(seq_offset, 16)]  # type: ignore
                                                    page_offset: T.int32(is_size_var=True) = T.floormod(seq_offset, 16)  # type: ignore
                                                    V_smem[i, j] = pages[page_no, 1, by, page_offset, j]
                                                else:
                                                    V_smem[i, j] = 0.0
                                        T.tvm_storage_sync("shared")

                                        # Compute S
                                        with T.block():
                                            for li, lj, lk in T.grid(tile_x, tile_z, tile_y):
                                                with T.block("S_gemm"):
                                                    i, j, k = T.axis.remap("SSR", [li, lj, lk])
                                                    with T.init():
                                                        S_local[i, j] = 0.0
                                                    S_local[i, j] += T.cast(Q_smem[i, k], "float32") * T.cast(K_smem[j, k], "float32") * attn_score_scaling_factor * sm_scale
                                        T.tvm_storage_sync("shared")
                                        for li, lj in T.grid(tile_x, tile_z):
                                            with T.block("S_store"):
                                                i, j = T.axis.remap("SS", [li, lj])
                                                S_smem[i, j] = S_local[i, j]
                                        T.tvm_storage_sync("shared")

                                        # Update S, m, d
                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                            if row < tile_x:
                                                with T.block("update1"):
                                                    m_prev[i] = m_smem[row]
                                                    m_new[i] = m_smem[row]
                                                    # mask out of kv_chunk_len S
                                                    row_: T.int32 = (LH_start + row) // group_size
                                                    for j in T.serial(tile_z):
                                                        if _causal_mask(causal,
                                                                row=row_,
                                                                col=L_kv_start + j,
                                                                kv_len=kv_chunk_len[0],
                                                                qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx]):
                                                            m_new[i] = T.max(m_new[i], S_smem[row, j])
                                                    d_new[i] = d_smem[row] * T.exp2(m_prev[i] - m_new[i])

                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                            with T.block("update"):
                                                for j in T.serial(tile_z):
                                                    # this is to avoid sync inside condition branch
                                                    if row < tile_x:
                                                        row_: T.int32 = (LH_start + row) // group_size
                                                        if _causal_mask(causal,
                                                                row=row_,
                                                                col=L_kv_start + j,
                                                                kv_len=kv_chunk_len[0],
                                                                qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx]):
                                                            S_smem[row, j] = T.exp2(S_smem[row, j] - m_new[i])
                                                        else:
                                                            S_smem[row, j] = T.exp2(-5e4 - m_new[i])

                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                            if row < tile_x:
                                                with T.block("update"):
                                                    for j in T.serial(tile_z):
                                                        d_new[i] += S_smem[row, j]
                                                    m_smem[row] = m_new[i]
                                                    d_smem[row] = d_new[i]
                                                    m_prev_smem[row] = m_prev[i]
                                        T.tvm_storage_sync("shared")

                                        # Update O
                                        with T.block():
                                            for li, lj, lk in T.grid(tile_x, tile_y, tile_z):
                                                with T.block("O_gemm"):
                                                    i, j, k = T.axis.remap("SSR", [li, lj, lk])
                                                    with T.init():
                                                        O_local[i, j] *= T.exp2(m_prev_smem[i] - m_smem[i])
                                                    O_local[i, j] += S_smem[i, k] * T.cast(V_smem[k, j], "float32")

                                    # Store O from smem to gmem
                                    for li, lj in T.grid(tile_x, tile_y):
                                        with T.block("O_store"):
                                            i, j = T.axis.remap("SS", [li, lj])
                                            cur_L: T.int32 = q_indptr[b_idx] + (LH_start + i) // group_size
                                            cur_H_qo: T.int32 = by * group_size + (LH_start + i) % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                output[cur_L, cur_H_qo, j] = O_local[i, j] / d_smem[i]

                                    # Store LSE to gmem
                                    for li in T.grid(tile_x):
                                        with T.block("lse_store"):
                                            i = T.axis.remap("S", [li])
                                            cur_L: T.int32 = q_indptr[b_idx] + (LH_start + i) // group_size
                                            cur_H_qo: T.int32 = by * group_size + (LH_start + i) % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                lse[cur_L, cur_H_qo] = m_smem[i] + T.log2(d_smem[i])

                                    # move to next tile
                                    tile_id[0] += NUM_BLKS
    # fmt: on
    # pylint: enable=line-too-long,invalid-name,too-many-arguments,too-many-branches
    sch = tir.Schedule(batch_prefill_paged_kv)

    def get_tile_size(x, y, t):
        cnt = (x * y) // t
        assert (x * y) % t == 0
        tile_y = (int)(math.ceil(math.sqrt(cnt)))
        while (cnt % tile_y != 0 or y % tile_y != 0) and tile_y <= cnt:
            tile_y += 1
        assert tile_y <= cnt
        tile_x = cnt // tile_y
        return tile_x, tile_y

    def apply_to_qkv_load(sch: tir.Schedule, block):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        loop = sch.fuse(loop_x, loop_y)
        _, ty, tx, vec = sch.split(
            loop, factors=[None, num_warps, bdx, LOAD_VEC], preserve_unit_iters=True
        )
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)

    def apply_to_so_ewise(sch: tir.Schedule, block, tile):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    def apply_to_gemm(  # pylint: disable=too-many-arguments,unused-argument
        sch: tir.Schedule, block, tile, read_0, read_1, r_len=8, k_major=False
    ):
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
        sch.decompose_reduction(block, ty)

    def apply_to_md(sch, block):
        loop = sch.get_loops(block)[-1]
        _, ty, tx = sch.split(loop, factors=[None, num_warps, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    tile_s = get_tile_size(tile_x, tile_z, bdx * num_warps)
    tile_o = get_tile_size(tile_x, tile_y, bdx * num_warps)
    apply_to_gemm(sch, sch.get_block("S_gemm"), tile_s, 0, 1, k_major=True)
    apply_to_gemm(sch, sch.get_block("O_gemm"), tile_o, 2, 3, k_major=False)
    apply_to_so_ewise(sch, sch.get_block("S_store"), tile_s)
    apply_to_so_ewise(sch, sch.get_block("O_init"), tile_o)
    apply_to_so_ewise(sch, sch.get_block("O_store"), tile_o)
    apply_to_qkv_load(sch, sch.get_block("Q_load"))
    apply_to_qkv_load(sch, sch.get_block("K_load"))
    apply_to_qkv_load(sch, sch.get_block("V_load"))
    apply_to_md(sch, sch.get_block("lse_store"))
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


def _attention_decode(
    num_kv_heads,
    num_qo_heads,
    head_dim,
    qkv_dtype,
    sliding_window: bool,
    target: Target,  # pylint: disable=unused-argument
):
    # pylint: disable=invalid-name
    qkv_dtype_bytes = 2
    H_qo = num_qo_heads
    H_kv = num_kv_heads
    D = head_dim

    THREAD_LIMIT = 512
    TILE_SIZE_PER_BDX = 2
    if target.kind.name == "opencl" and "android" in str(target.host):
        THREAD_LIMIT = 64
        TILE_SIZE_PER_BDX = 1
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    thread_limit = min(max_num_threads_per_block, THREAD_LIMIT)

    GROUP_SIZE = H_qo // H_kv
    VEC_SIZE = min(max(8 // qkv_dtype_bytes, D // 32), 4)
    bdx = D // VEC_SIZE
    bdy = GROUP_SIZE
    while bdx * bdy > thread_limit and bdy > 1:
        bdy //= 2
    gdz = GROUP_SIZE // bdy
    threads_per_CTA = max(thread_limit, bdx * bdy)
    bdz = threads_per_CTA // (bdx * bdy)
    tile_size_per_bdx = TILE_SIZE_PER_BDX if GROUP_SIZE == 1 else 1
    log2e = math.log2(math.exp(1))

    # undefined vars used
    # pylint: disable=line-too-long,too-many-arguments,too-many-branches
    # fmt: off
    @T.prim_func(check_well_formed=False)
    def batch_decode_paged_kv(
        _0: T.int32,  # pylint: disable=unused-argument
        Q_handle: T.handle,
        pages_handle: T.handle,
        page_table_indptr_handle: T.handle,
        page_table_values_handle: T.handle,
        var_length_info: T.handle, # [b] when sliding window = False, or otherwise [3, b]
        k_rope_pos_offset_handle: T.handle,
        q_rope_position_handle: T.handle,
        output_handle: T.handle,
        lse_handle: T.handle,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        attn_score_scaling_factor: T.float32,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        B = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)
        page_indptr_elem_offset = T.int32(is_size_var=True)
        page_values_elem_offset = T.int32(is_size_var=True)
        k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        length_info_elem_offset = T.int32(is_size_var=True)

        Q = T.match_buffer(Q_handle, (B, H_qo, D), qkv_dtype)
        pages = T.match_buffer(
            pages_handle, (max_num_pages, 2, H_kv, 16, D), qkv_dtype
        )
        page_table_indptr = T.match_buffer(page_table_indptr_handle, (B + 1,), "int32", elem_offset=page_indptr_elem_offset)
        page_table_values = T.match_buffer(page_table_values_handle, (nnz_pages,), "int32", elem_offset=page_values_elem_offset)
        k_rope_pos_offset = T.match_buffer(k_rope_pos_offset_handle, (B,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
        q_rope_position = T.match_buffer(q_rope_position_handle, (B,), "int32", elem_offset=q_rope_position_elem_offset)
        output = T.match_buffer(output_handle, (B, H_qo, D), qkv_dtype)
        lse = T.match_buffer(lse_handle, (B, H_qo), "float32")  # pylint: disable=unused-variable
        # The length information of the sequences.
        # - It is in shape `(3, batch_size)` when sliding window is enabled.
        #   For a sequence "i", location
        #   - "(0, i)" is the number of KV slots used in the last page of the seq ("last_page_len"),
        #   - "(1, i)" is the starting offset of the sliding window in the seq,
        #   - "(2, i)" is the attn sink length of the sequence.
        # - It is in shape `(batch_size,)` when sliding window is disabled,
        #   denoting the "last_page_len".
        length_info = _declare_length_info(var_length_info, B, sliding_window, length_info_elem_offset)

        sm_scale = 1.0 / math.sqrt(float(D)) * log2e

        for bx in T.thread_binding(B, thread="blockIdx.x"):
            for fused_by_bz in T.thread_binding(H_kv * gdz, thread="blockIdx.y"):
                for ty in T.thread_binding(bdy, thread="threadIdx.y"):
                    for tx in T.thread_binding(bdx, thread="threadIdx.x"):
                        for tz in T.thread_binding(bdz, thread="threadIdx.z"):
                            with T.block("attn"):
                                Q_local = T.alloc_buffer((VEC_SIZE,), qkv_dtype, scope="local")
                                kv_chunk_len = T.alloc_buffer((1,), "int32", scope="local")
                                K_smem = T.alloc_buffer((bdz * bdy * tile_size_per_bdx, D), qkv_dtype, scope="shared")
                                V_smem = T.alloc_buffer((bdz * bdy * tile_size_per_bdx, D), qkv_dtype, scope="shared")
                                O_allreduce = T.alloc_buffer((bdz, bdy, D), "float32", scope="shared")
                                md_allreduce = T.alloc_buffer((bdz, bdy, 2), "float32", scope="shared")
                                S_reduce_local = T.alloc_buffer((1,), "float32", scope="local")
                                t0 = T.alloc_buffer((1,), "float32", scope="local")

                                S_local = T.alloc_buffer((bdy * tile_size_per_bdx), "float32", scope="local")
                                K_local = T.alloc_buffer((VEC_SIZE,), qkv_dtype, scope="local")
                                V_local = T.alloc_buffer((VEC_SIZE,), qkv_dtype, scope="local")
                                m_prev = T.alloc_buffer((1,), "float32", scope="local")
                                d_prev = T.alloc_buffer((1,), "float32", scope="local")
                                other_m = T.alloc_buffer((1,), "float32", scope="local")
                                other_d = T.alloc_buffer((1,), "float32", scope="local")
                                other_o = T.alloc_buffer((VEC_SIZE,), "float32", scope="local")
                                st_m = T.alloc_buffer((1,), "float32", scope="local")
                                st_d = T.alloc_buffer((1,), "float32", scope="local")
                                O_local = T.alloc_buffer((VEC_SIZE,), "float32", scope="local")

                                by: T.int32 = fused_by_bz % H_kv
                                bz: T.int32 = fused_by_bz // H_kv
                                batch_idx: T.int32 = bx
                                cur_page_indptr_begin: T.int32 = page_table_indptr[batch_idx]
                                cur_page_indptr_end: T.int32 = page_table_indptr[batch_idx + 1]
                                kv_chunk_len[0] = T.if_then_else(
                                    cur_page_indptr_begin != cur_page_indptr_end,
                                    _get_kv_chunk_len(cur_page_indptr_end - cur_page_indptr_begin, 16, batch_idx, length_info, sliding_window),
                                    0
                                )

                                # init states
                                st_m[0] = -5e4
                                st_d[0] = 1.0
                                for vec in T.vectorized(VEC_SIZE):
                                    O_local[vec] = 0.0

                                # load q
                                for vec in T.vectorized(VEC_SIZE):
                                    Q_local[vec] = T.if_then_else(
                                        rotary_mode == 1,
                                        _rope(Q, q_rope_position[batch_idx], head_dim, rope_theta, rope_scale, (bx, by * GROUP_SIZE + bz * bdy + ty, tx * VEC_SIZE + vec), qkv_dtype),
                                        Q[bx, by * GROUP_SIZE + bz * bdy + ty, tx * VEC_SIZE + vec]
                                    )

                                for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_size_per_bdx * bdy * bdz)):
                                    tile_start_s: T.int32(is_size_var=True) = (tz * bdy + ty) * tile_size_per_bdx  # type: ignore
                                    tile_start_g: T.int32(is_size_var=True) = ((iterator * bdz + tz) * bdy + ty) * tile_size_per_bdx  # type: ignore
                                    # load K from global memory to shared memory
                                    for j in T.serial(tile_size_per_bdx):
                                        with T.block("K_load"):
                                            T.reads()
                                            T.writes()
                                            row_g: T.int32(is_size_var=True) = tile_start_g + j  # type: ignore
                                            if row_g < kv_chunk_len[0]:
                                                seq_offset: T.int32(is_size_var=True) = _get_seq_offset(row_g, batch_idx, length_info, sliding_window)  # type: ignore
                                                page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + T.floordiv(seq_offset, 16)]  # type: ignore
                                                page_offset: T.int32(is_size_var=True) = T.floormod(seq_offset, 16)  # type: ignore
                                                for vec in T.vectorized(VEC_SIZE):
                                                    K_smem[tile_start_s + j, tx * VEC_SIZE + vec] = T.if_then_else(
                                                        rotary_mode == 1,
                                                        _rope(pages, k_rope_pos_offset[batch_idx] + row_g, head_dim, rope_theta, rope_scale, (page_no, 0, by, page_offset, tx * VEC_SIZE + vec), qkv_dtype),
                                                        pages[page_no, 0, by, page_offset, tx * VEC_SIZE + vec]
                                                    )
                                            else:
                                                for vec in T.vectorized(VEC_SIZE):
                                                    K_smem[tile_start_s + j, tx * VEC_SIZE + vec] = 0.0
                                    T.tvm_storage_sync("shared")
                                    # load V from global memory to shared memory
                                    for j in T.serial(tile_size_per_bdx):
                                        with T.block("V_load"):
                                            T.reads()
                                            T.writes()
                                            row_g: T.int32(is_size_var=True) = tile_start_g + j  # type: ignore
                                            if row_g < kv_chunk_len[0]:
                                                seq_offset: T.int32(is_size_var=True) = _get_seq_offset(row_g, batch_idx, length_info, sliding_window)  # type: ignore
                                                page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + T.floordiv(seq_offset, 16)]  # type: ignore
                                                page_offset: T.int32(is_size_var=True) = T.floormod(seq_offset, 16)  # type: ignore
                                                for vec in T.vectorized(VEC_SIZE):
                                                    V_smem[tile_start_s + j, tx * VEC_SIZE + vec] = pages[page_no, 1, by, page_offset, tx * VEC_SIZE + vec]
                                            else:
                                                for vec in T.vectorized(VEC_SIZE):
                                                    V_smem[tile_start_s + j, tx * VEC_SIZE + vec] = 0.0
                                    T.tvm_storage_sync("shared")
                                    # compute QK
                                    m_prev[0] = st_m[0]
                                    for j in T.serial(bdy * tile_size_per_bdx):
                                        # load K from shared memory to local memory
                                        for vec in T.vectorized(VEC_SIZE):
                                            K_local[vec] = K_smem[tz * bdy * tile_size_per_bdx + j, tx * VEC_SIZE + vec]
                                        # compute S = Q * K * sm_scale
                                        S_reduce_local[0] = 0
                                        for vec in T.serial(VEC_SIZE):
                                            S_reduce_local[0] += T.cast(Q_local[vec], "float32") * T.cast(K_local[vec], "float32") * attn_score_scaling_factor * sm_scale

                                        with T.block("block_cross_thread"):
                                            T.reads(S_reduce_local[0])
                                            T.writes(t0[0])
                                            T.attr(
                                                T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
                                                "reduce_scope",
                                                T.reinterpret("handle", T.uint64(0)),
                                            )
                                            T.tvm_thread_allreduce(T.uint32(1), S_reduce_local[0], True, t0[0], tx, dtype="handle")

                                        S_local[j] = -5e4
                                        if (iterator * bdz + tz) * bdy * tile_size_per_bdx + j < kv_chunk_len[0]:
                                            S_local[j] = t0[0]
                                        # update st_m
                                        st_m[0] = T.max(st_m[0], S_local[j])

                                    # update st_d, st_O
                                    o_scale: T.float32 = T.exp2(m_prev[0] - st_m[0])
                                    st_d[0] *= o_scale
                                    for j in T.serial(bdy * tile_size_per_bdx):
                                        S_local[j] = T.exp2(S_local[j] - st_m[0])
                                        st_d[0] += S_local[j]
                                    for j in T.vectorized(VEC_SIZE):
                                        O_local[j] *= o_scale

                                    # load V from shared memory to local memory
                                    # compute O
                                    for j in T.serial(bdy * tile_size_per_bdx):
                                        for vec in T.vectorized(VEC_SIZE):
                                            V_local[vec] = V_smem[tz * bdy * tile_size_per_bdx + j, tx * VEC_SIZE + vec]
                                        for vec in T.vectorized(VEC_SIZE):
                                            O_local[vec] += T.cast(V_local[vec], "float32") * S_local[j]

                                if bdz > 1:
                                    # allreduce over bdz
                                    for vec in T.vectorized(VEC_SIZE):
                                        O_allreduce[tz, ty, tx * VEC_SIZE + vec] = O_local[vec]
                                    md_allreduce[tz, ty, 0] = st_m[0]
                                    md_allreduce[tz, ty, 1] = st_d[0]
                                    T.tvm_storage_sync("shared")

                                    st_m[0] = -5e4
                                    st_d[0] = 1.0
                                    for vec in T.vectorized(VEC_SIZE):
                                        O_local[vec] = 0.0

                                    for j in T.serial(bdz):
                                        m_prev[0] = st_m[0]
                                        d_prev[0] = st_d[0]
                                        other_m[0] = md_allreduce[j, ty, 0]
                                        other_d[0] = md_allreduce[j, ty, 1]
                                        for vec in T.vectorized(VEC_SIZE):
                                            other_o[vec] = O_allreduce[j, ty, tx * VEC_SIZE + vec]
                                        st_m[0] = T.max(st_m[0], other_m[0])
                                        st_d[0] = d_prev[0] * T.exp2(m_prev[0] - st_m[0]) + other_d[0] * T.exp2(other_m[0] - st_m[0])
                                        for vec in T.serial(VEC_SIZE):
                                            O_local[vec] = O_local[vec] * T.exp2(m_prev[0] - st_m[0]) + other_o[vec] * T.exp2(other_m[0] - st_m[0])

                                # normalize O
                                for vec in T.serial(VEC_SIZE):
                                    O_local[vec] /= st_d[0]

                                # store O to global memory
                                for vec in T.vectorized(VEC_SIZE):
                                    output[batch_idx, by * GROUP_SIZE + bz * bdy + ty, tx * VEC_SIZE + vec] = O_local[vec]

                                # store lse to global memory
                                lse[batch_idx, by * GROUP_SIZE + bz * bdy + ty] = st_m[0] + T.log2(st_d[0])
    # fmt: on
    # pylint: enable=line-too-long,invalid-name,too-many-arguments,too-many-branches
    return batch_decode_paged_kv


def _attention_prefill_ragged(
    h_kv, h_q, d, dtype, target: Target
):  # pylint: disable=unused-argument
    # pylint: disable=invalid-name,line-too-long
    NUM_BLKS = 16
    LOAD_VEC = 8 // ((DataType(dtype).bits + 7) // 8)  # 8 bytes
    group_size = h_q // h_kv
    sm_scale = 1.0 / math.sqrt(float(d)) * math.log2(math.exp(1))

    bdx = 32
    num_warps = 4
    tile_x, tile_y, tile_z = 64 // ((DataType(dtype).bits + 7) // 8) // max(d // 128, 1), d, 16

    # Otherwise we would exceed maxComputeWorkgroupStorageSize
    if (
        str(target.kind) == "webgpu"
        and ((d + 127) // 128) * ((DataType(dtype).bits + 15) // 16) >= 4
    ):
        tile_z = 8
        num_warps = 2

    # undefined vars used
    # fmt: off
    @T.prim_func(check_well_formed=False)
    def batch_prefill_ragged_kv( # pylint: disable=too-many-arguments,too-many-branches
        var_q: T.handle, # [total_len, h_q, d]
        var_q_indptr: T.handle, # [batch_size + 1]
        var_k: T.handle, # [total_len, h_kv, d]
        var_v: T.handle, # [total_len, h_kv, d]
        var_kv_indptr: T.handle, # [batch_size + 1]
        var_q_rope_position: T.handle, # [total_q_len]
        var_k_rope_pos_offset: T.handle, # [b]
        var_output: T.handle, # [total_len, h_q, d]
        var_lse: T.handle, # [total_len, h_q]
        causal: T.int32,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        attn_score_scaling_factor: T.float32
    ):
        batch_size = T.int32(is_size_var=True)
        qo_len = T.int32(is_size_var=True)
        kv_len = T.int32(is_size_var=True)
        q_indptr_elem_offset = T.int32(is_size_var=True)
        kv_indptr_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        k_rope_pos_offset_elem_offset = T.int32(is_size_var=True)

        q = T.match_buffer(var_q, (qo_len, h_q, d), dtype)
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
        k = T.match_buffer(var_k, (kv_len, h_kv, d), dtype)
        v = T.match_buffer(var_v, (kv_len, h_kv, d), dtype)
        kv_indptr = T.match_buffer(var_kv_indptr, (batch_size + 1,), "int32", elem_offset=kv_indptr_elem_offset)
        q_rope_position = T.match_buffer(var_q_rope_position, (qo_len,), "int32", elem_offset=q_rope_position_elem_offset)
        k_rope_pos_offset = T.match_buffer(var_k_rope_pos_offset, (batch_size,), "int32", elem_offset=k_rope_pos_offset_elem_offset)
        output = T.match_buffer(var_output, (qo_len, h_q, d), dtype)
        lse = T.match_buffer(var_lse, (qo_len, h_q), "float32")  # pylint: disable=unused-variable

        # kernel code
        for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
            for lby in T.thread_binding(h_kv, thread="blockIdx.y"):
                for lty in T.thread_binding(num_warps, thread="threadIdx.y"):
                    for ltx in T.thread_binding(bdx, thread="threadIdx.x"):
                        with T.block("attn"):
                            bx, by, ty, tx = T.axis.remap("SSSS", [lbx, lby, lty, ltx])
                            T.reads()
                            T.writes()
                            tile_id = _var("int32")
                            batch_idx = _var("int32")
                            batch_tiles = _var("int32")
                            batch_rows = _var("int32")
                            iterator = _var("int32")
                            kv_chunk_len = _var("int32")

                            Q_smem = T.alloc_buffer((tile_x, d), dtype, scope="shared")
                            K_smem = T.alloc_buffer((tile_z, d), dtype, scope="shared")
                            V_smem = T.alloc_buffer((tile_z, d), dtype, scope="shared")
                            S_smem = T.alloc_buffer((tile_x, tile_z), "float32", scope="shared")

                            S_local = T.alloc_buffer((tile_x, tile_z), "float32", scope="local")
                            O_local = T.alloc_buffer((tile_x, d), "float32", scope="local")

                            m_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")
                            m_prev_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")
                            d_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")

                            m_new = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")
                            m_prev = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")
                            d_new = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")

                            ## get tile_no, batch_idx, batch_tiles, batch_rows
                            tile_id[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = (q_indptr[1] - q_indptr[0]) * group_size
                            batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)
                            while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                # advance to next tile
                                while tile_id[0] >= batch_tiles[0] and batch_idx[0] < batch_size:
                                    tile_id[0] -= batch_tiles[0]
                                    batch_idx[0] += 1
                                    if batch_idx[0] < batch_size:
                                        b_idx: T.int32 = batch_idx[0]
                                        batch_rows[0] = (q_indptr[b_idx + 1] - q_indptr[b_idx]) * group_size
                                        batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)

                                if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                    b_idx: T.int32 = batch_idx[0]
                                    LH_start: T.int32 = tile_id[0] * tile_x
                                    q_indptr_val: T.int32 = q_indptr[b_idx]

                                    kv_chunk_len[0] = kv_indptr[b_idx + 1] - kv_indptr[b_idx]
                                    T.tvm_storage_sync("shared")

                                    # init states
                                    for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                        row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                        if row < tile_x:
                                            m_smem[row] = -5e4
                                            d_smem[row] = 1.0

                                    for li, lj in T.grid(tile_x, tile_y):
                                        with T.block("O_init"):
                                            i, j = T.axis.remap("SS", [li, lj])
                                            O_local[i, j] = 0.0
                                    T.tvm_storage_sync("shared")

                                    # Load Q from gmem to smem
                                    for li, lj in T.grid(tile_x, tile_y):
                                        with T.block("Q_load"):
                                            i, j = T.axis.remap("SS", [li, lj])
                                            T.reads()
                                            T.writes()
                                            cur_L = q_indptr_val + (LH_start + i) // group_size
                                            cur_H_qo = by * group_size + (LH_start + i) % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                Q_smem[i, j] = T.if_then_else(
                                                    rotary_mode == 1,
                                                    _rope(q, q_rope_position[cur_L], d, rope_theta, rope_scale, (cur_L, cur_H_qo, j), dtype),
                                                    q[cur_L, cur_H_qo, j]
                                                )
                                            else:
                                                Q_smem[i, j] = 0.0
                                    T.tvm_storage_sync("shared")

                                    for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_z)):
                                        L_kv_start: T.int32 = iterator * tile_z
                                        L_kv_base: T.int32 = kv_indptr[b_idx]
                                        for lz, ly in T.grid(tile_z, tile_y):
                                            with T.block("K_load"):
                                                i, j = T.axis.remap("SS", [lz, ly])
                                                T.reads()
                                                T.writes()
                                                cur_L = L_kv_start + i
                                                if cur_L < kv_chunk_len[0]:
                                                    K_smem[i, j] = T.if_then_else(
                                                        rotary_mode == 1,
                                                        _rope(k, k_rope_pos_offset[b_idx] + cur_L, d, rope_theta, rope_scale, (L_kv_base + cur_L, by, j), dtype),
                                                        k[L_kv_base + cur_L, by, j]
                                                    )
                                                else:
                                                    K_smem[i, j] = 0.0
                                        T.tvm_storage_sync("shared")
                                        for lz, ly in T.grid(tile_z, tile_y):
                                            with T.block("V_load"):
                                                i, j = T.axis.remap("SS", [lz, ly])
                                                T.reads()
                                                T.writes()
                                                cur_L = L_kv_start + i
                                                if cur_L < kv_chunk_len[0]:
                                                    V_smem[i, j] = v[L_kv_base + cur_L, by, j]
                                                else:
                                                    V_smem[i, j] = 0.0
                                        T.tvm_storage_sync("shared")

                                        # Compute S
                                        with T.block():
                                            for li, lj, lk in T.grid(tile_x, tile_z, tile_y):
                                                with T.block("S_gemm"):
                                                    i, j, k = T.axis.remap("SSR", [li, lj, lk])
                                                    with T.init():
                                                        S_local[i, j] = 0.0
                                                    S_local[i, j] += T.cast(Q_smem[i, k], "float32") * T.cast(K_smem[j, k], "float32") * attn_score_scaling_factor * sm_scale
                                        T.tvm_storage_sync("shared")
                                        for li, lj in T.grid(tile_x, tile_z):
                                            with T.block("S_store"):
                                                i, j = T.axis.remap("SS", [li, lj])
                                                S_smem[i, j] = S_local[i, j]
                                        T.tvm_storage_sync("shared")

                                        # Update S, m, d
                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                            if row < tile_x:
                                                with T.block("update1"):
                                                    m_prev[i] = m_smem[row]
                                                    m_new[i] = m_smem[row]
                                                    # mask out of kv_chunk_len S
                                                    row_: T.int32 = (LH_start + row) // group_size
                                                    for j in T.serial(tile_z):
                                                        if _causal_mask(causal,
                                                                row=row_,
                                                                col=L_kv_start + j,
                                                                kv_len=kv_chunk_len[0],
                                                                qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx]):
                                                            m_new[i] = T.max(m_new[i], S_smem[row, j])
                                                    d_new[i] = d_smem[row] * T.exp2(m_prev[i] - m_new[i])

                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                            with T.block("update"):
                                                for j in T.serial(tile_z):
                                                    # this is to avoid sync inside condition branch
                                                    if row < tile_x:
                                                        row_: T.int32 = (LH_start + row) // group_size
                                                        if _causal_mask(causal,
                                                                row=row_,
                                                                col=L_kv_start + j,
                                                                kv_len=kv_chunk_len[0],
                                                                qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx]):
                                                            S_smem[row, j] = T.exp2(S_smem[row, j] - m_new[i])
                                                        else:
                                                            S_smem[row, j] = T.exp2(-5e4 - m_new[i])

                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                            if row < tile_x:
                                                with T.block("update"):
                                                    for j in T.serial(tile_z):
                                                        d_new[i] += S_smem[row, j]
                                                    m_smem[row] = m_new[i]
                                                    d_smem[row] = d_new[i]
                                                    m_prev_smem[row] = m_prev[i]
                                        T.tvm_storage_sync("shared")

                                        # Update O
                                        with T.block():
                                            for li, lj, lk in T.grid(tile_x, tile_y, tile_z):
                                                with T.block("O_gemm"):
                                                    i, j, k = T.axis.remap("SSR", [li, lj, lk])
                                                    with T.init():
                                                        O_local[i, j] *= T.exp2(m_prev_smem[i] - m_smem[i])
                                                    O_local[i, j] += S_smem[i, k] * T.cast(V_smem[k, j], "float32")

                                    # Store O from smem to gmem
                                    for li, lj in T.grid(tile_x, tile_y):
                                        with T.block("O_store"):
                                            i, j = T.axis.remap("SS", [li, lj])
                                            cur_L: T.int32 = q_indptr[b_idx] + (LH_start + i) // group_size
                                            cur_H_qo: T.int32 = by * group_size + (LH_start + i) % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                output[cur_L, cur_H_qo, j] = O_local[i, j] / d_smem[i]

                                    # Store LSE to gmem
                                    for li in T.grid(tile_x):
                                        with T.block("lse_store"):
                                            i = T.axis.remap("S", [li])
                                            cur_L: T.int32 = q_indptr[b_idx] + (LH_start + i) // group_size
                                            cur_H_qo: T.int32 = by * group_size + (LH_start + i) % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                lse[cur_L, cur_H_qo] = m_smem[i] + T.log2(d_smem[i])

                                    # move to next tile
                                    tile_id[0] += NUM_BLKS
    # fmt: on
    # pylint: enable=line-too-long,invalid-name,too-many-arguments,too-many-branches
    sch = tir.Schedule(batch_prefill_ragged_kv)

    def get_tile_size(x, y, t):
        cnt = (x * y) // t
        assert (x * y) % t == 0
        tile_y = (int)(math.ceil(math.sqrt(cnt)))
        while (cnt % tile_y != 0 or y % tile_y != 0) and tile_y <= cnt:
            tile_y += 1
        assert tile_y <= cnt
        tile_x = cnt // tile_y
        return tile_x, tile_y

    def apply_to_qkv_load(sch: tir.Schedule, block):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        loop = sch.fuse(loop_x, loop_y)
        _, ty, tx, vec = sch.split(
            loop, factors=[None, num_warps, bdx, LOAD_VEC], preserve_unit_iters=True
        )
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)

    def apply_to_so_ewise(sch: tir.Schedule, block, tile):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    def apply_to_gemm(  # pylint: disable=too-many-arguments,unused-argument
        sch: tir.Schedule, block, tile, read_0, read_1, r_len=8, k_major=False
    ):
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
        sch.decompose_reduction(block, ty)

    def apply_to_md(sch, block):
        loop = sch.get_loops(block)[-1]
        _, ty, tx = sch.split(loop, factors=[None, num_warps, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    tile_s = get_tile_size(tile_x, tile_z, bdx * num_warps)
    tile_o = get_tile_size(tile_x, tile_y, bdx * num_warps)
    apply_to_gemm(sch, sch.get_block("S_gemm"), tile_s, 0, 1, k_major=True)
    apply_to_gemm(sch, sch.get_block("O_gemm"), tile_o, 2, 3, k_major=False)
    apply_to_so_ewise(sch, sch.get_block("S_store"), tile_s)
    apply_to_so_ewise(sch, sch.get_block("O_init"), tile_o)
    apply_to_so_ewise(sch, sch.get_block("O_store"), tile_o)
    apply_to_qkv_load(sch, sch.get_block("Q_load"))
    apply_to_qkv_load(sch, sch.get_block("K_load"))
    apply_to_qkv_load(sch, sch.get_block("V_load"))

    apply_to_md(sch, sch.get_block("lse_store"))
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


def _tree_mask(row, col, mask_ptr, offset, stride, kv_len):
    return tir.all(col < kv_len, mask_ptr[offset + row * stride + col] == 1)


def _attention_prefill_with_tree_mask(
    h_kv, h_q, d, dtype, target: Target
):  # pylint: disable=unused-argument
    # pylint: disable=invalid-name,line-too-long
    NUM_BLKS = 16
    LOAD_VEC = 8 // ((DataType(dtype).bits + 7) // 8)  # 8 bytes
    group_size = h_q // h_kv
    sm_scale = 1.0 / math.sqrt(float(d)) * math.log2(math.exp(1))

    bdx = 32
    num_warps = 4
    tile_x, tile_y, tile_z = 64 // ((DataType(dtype).bits + 7) // 8) // max(d // 128, 1), d, 16
    L_per_cta = tile_x // group_size

    # Otherwise we would exceed maxComputeWorkgroupStorageSize
    if (
        str(target.kind) == "webgpu"
        and ((d + 127) // 128) * ((DataType(dtype).bits + 15) // 16) >= 4
    ):
        tile_z = 8
        num_warps = 2

    # fmt: off
    @T.prim_func
    def batch_tree_attn(  # pylint: disable=too-many-branches
        var_q: T.handle, # [total_len, h_q, d]
        var_q_indptr: T.handle, # [batch_size + 1]
        var_k: T.handle, # [total_len, h_kv, d]
        var_v: T.handle, # [total_len, h_kv, d]
        var_kv_indptr: T.handle, # [batch_size + 1], kv_indptr should be the same as q_indptr in this case
        var_q_rope_position: T.handle, # [total_q_len]
        var_mn_indptr: T.handle, # [batch_size + 1]
        var_mask: T.handle, # [mn_indptr[batch_size]]
        var_output: T.handle, # [total_len, h_q, d]
        var_lse: T.handle, # [total_len, h_q]
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
        attn_score_scaling_factor: T.float32,
        batch_size: T.int32,
    ):
        qo_len = T.int32(is_size_var=True)
        kv_len = T.int32(is_size_var=True)
        q_indptr_elem_offset = T.int32(is_size_var=True)
        kv_indptr_elem_offset = T.int32(is_size_var=True)
        q_rope_position_elem_offset = T.int32(is_size_var=True)
        mn_indptr_elem_offset = T.int32(is_size_var=True)
        mask_elem_offset = T.int32(is_size_var=True)
        tree_size = T.int32(is_size_var=True)

        q = T.match_buffer(var_q, (qo_len, h_q, d), dtype)
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32", elem_offset=q_indptr_elem_offset)
        k = T.match_buffer(var_k, (kv_len, h_kv, d), dtype)
        v = T.match_buffer(var_v, (kv_len, h_kv, d), dtype)
        kv_indptr = T.match_buffer(var_kv_indptr, (batch_size + 1,), "int32", elem_offset=kv_indptr_elem_offset)
        q_rope_position = T.match_buffer(var_q_rope_position, (qo_len,), "int32", elem_offset=q_rope_position_elem_offset)
        mn_indptr = T.match_buffer(var_mn_indptr, (batch_size + 1,), "int32", elem_offset=mn_indptr_elem_offset)
        mask = T.match_buffer(var_mask, (tree_size,), "int32", elem_offset=mask_elem_offset)
        output = T.match_buffer(var_output, (qo_len, h_q, d), dtype)
        lse = T.match_buffer(var_lse, (qo_len, h_q), "float32")  # pylint: disable=unused-variable

        # kernel code
        for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
            for lby in T.thread_binding(h_kv, thread="blockIdx.y"):
                for lty in T.thread_binding(num_warps, thread="threadIdx.y"):
                    for ltx in T.thread_binding(bdx, thread="threadIdx.x"):
                        with T.block("attn"):
                            bx, by, ty, tx = T.axis.remap("SSSS", [lbx, lby, lty, ltx])
                            T.reads()
                            T.writes()
                            tile_id = _var("int32")
                            batch_idx = _var("int32")
                            batch_tiles = _var("int32")
                            batch_rows = _var("int32")
                            iterator = _var("int32")
                            kv_chunk_len = _var("int32")

                            Q_smem = T.alloc_buffer((tile_x, d), dtype, scope="shared")
                            K_smem = T.alloc_buffer((tile_z, d), dtype, scope="shared")
                            V_smem = T.alloc_buffer((tile_z, d), dtype, scope="shared")
                            S_smem = T.alloc_buffer((tile_x, tile_z), "float32", scope="shared")

                            S_local = T.alloc_buffer((tile_x, tile_z), "float32", scope="local")
                            O_local = T.alloc_buffer((tile_x, d), "float32", scope="local")

                            m_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")
                            m_prev_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")
                            d_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")

                            m_new = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")
                            m_prev = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")
                            d_new = T.alloc_buffer((math.ceil(tile_x / (bdx * num_warps)),), "float32", scope="local")

                            ## get tile_no, batch_idx, batch_tiles, batch_rows
                            tile_id[0] = bx
                            batch_idx[0] = 0
                            batch_rows[0] = (q_indptr[1] - q_indptr[0]) * group_size
                            batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)
                            while T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                # advance to next tile
                                while tile_id[0] >= batch_tiles[0] and batch_idx[0] < batch_size:
                                    tile_id[0] -= batch_tiles[0]
                                    batch_idx[0] += 1
                                    if batch_idx[0] < batch_size:
                                        b_idx: T.int32 = batch_idx[0]
                                        batch_rows[0] = (q_indptr[b_idx + 1] - q_indptr[b_idx]) * group_size
                                        batch_tiles[0] = T.ceildiv(batch_rows[0], tile_x)

                                if T.tvm_thread_invariant(batch_idx[0] < batch_size):
                                    b_idx: T.int32 = batch_idx[0]
                                    LH_start: T.int32 = tile_id[0] * tile_x
                                    q_indptr_val: T.int32 = q_indptr[b_idx]

                                    kv_chunk_len[0] = kv_indptr[b_idx + 1] - kv_indptr[b_idx]
                                    T.tvm_storage_sync("shared")

                                    # init states
                                    for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                        row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                        if row < tile_x:
                                            m_smem[row] = -5e4
                                            d_smem[row] = 1.0

                                    for li, lj in T.grid(tile_x, tile_y):
                                        with T.block("O_init"):
                                            i, j = T.axis.remap("SS", [li, lj])
                                            O_local[i, j] = 0.0
                                    T.tvm_storage_sync("shared")

                                    # Load Q from gmem to smem
                                    for li, lj in T.grid(tile_x, tile_y):
                                        with T.block("Q_load"):
                                            i, j = T.axis.remap("SS", [li, lj])
                                            T.reads()
                                            T.writes()
                                            cur_L = q_indptr_val + (LH_start + i) // group_size
                                            cur_H_qo = by * group_size + (LH_start + i) % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                Q_smem[i, j] = T.if_then_else(
                                                    rotary_mode == 1,
                                                    _rope(q, q_rope_position[cur_L], d, rope_theta, rope_scale, (cur_L, cur_H_qo, j), dtype),
                                                    q[cur_L, cur_H_qo, j]
                                                )
                                            else:
                                                Q_smem[i, j] = 0.0
                                    T.tvm_storage_sync("shared")

                                    for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_z)):
                                        L_kv_start: T.int32 = iterator * tile_z
                                        L_kv_base: T.int32 = kv_indptr[b_idx]
                                        for lz, ly in T.grid(tile_z, tile_y):
                                            with T.block("KV_load"):
                                                i, j = T.axis.remap("SS", [lz, ly])
                                                T.reads()
                                                T.writes()
                                                cur_L = L_kv_base + L_kv_start + i
                                                if L_kv_start + i < kv_chunk_len[0]:
                                                    K_smem[i, j] = T.if_then_else(
                                                        rotary_mode == 1,
                                                        _rope(k, q_rope_position[cur_L], d, rope_theta, rope_scale, (cur_L, by, j), dtype),
                                                        k[cur_L, by, j]
                                                    )
                                                    V_smem[i, j] = v[cur_L, by, j]
                                                else:
                                                    K_smem[i, j] = 0.0
                                                    V_smem[i, j] = 0.0
                                        T.tvm_storage_sync("shared")

                                        # Compute S
                                        with T.block():
                                            for li, lj, lk in T.grid(tile_x, tile_z, tile_y):
                                                with T.block("S_gemm"):
                                                    i, j, k = T.axis.remap("SSR", [li, lj, lk])
                                                    with T.init():
                                                        S_local[i, j] = 0.0
                                                    S_local[i, j] += T.cast(Q_smem[i, k], "float32") * T.cast(K_smem[j, k], "float32") * attn_score_scaling_factor * sm_scale
                                        T.tvm_storage_sync("shared")
                                        for li, lj in T.grid(tile_x, tile_z):
                                            with T.block("S_store"):
                                                i, j = T.axis.remap("SS", [li, lj])
                                                S_smem[i, j] = S_local[i, j]
                                        T.tvm_storage_sync("shared")

                                        # Update S, m, d
                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                            if row < tile_x:
                                                with T.block("update1"):
                                                    m_prev[i] = m_smem[row]
                                                    m_new[i] = m_smem[row]
                                                    # mask out of kv_chunk_len S
                                                    row_: T.int32 = (LH_start + row) // group_size
                                                    for j in T.serial(tile_z):
                                                        if _tree_mask(
                                                            row=row_,
                                                            col=L_kv_start + j,
                                                            mask_ptr=mask,
                                                            offset=mn_indptr[b_idx],
                                                            stride=q_indptr[b_idx + 1] - q_indptr[b_idx],
                                                            kv_len=kv_chunk_len[0]):
                                                            m_new[i] = T.max(m_new[i], S_smem[row, j])
                                                    d_new[i] = d_smem[row] * T.exp2(m_prev[i] - m_new[i])

                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                            with T.block("update"):
                                                for j in T.serial(tile_z):
                                                    # this is to avoid sync inside condition branch
                                                    if row < tile_x:
                                                        row_: T.int32 = (LH_start + row) // group_size
                                                        if _tree_mask(
                                                            row=row_,
                                                            col=L_kv_start + j,
                                                            mask_ptr=mask,
                                                            offset=mn_indptr[b_idx],
                                                            stride=q_indptr[b_idx + 1] - q_indptr[b_idx],
                                                            kv_len=kv_chunk_len[0]):
                                                            S_smem[row, j] = T.exp2(S_smem[row, j] - m_new[i])
                                                        else:
                                                            S_smem[row, j] = T.exp2(-5e4 - m_new[i])

                                        for i in T.serial(T.ceildiv(tile_x, bdx * num_warps)):
                                            row: T.int32 = i * bdx * num_warps + ty * bdx + tx
                                            if row < tile_x:
                                                with T.block("update"):
                                                    for j in T.serial(tile_z):
                                                        d_new[i] += S_smem[row, j]
                                                    m_smem[row] = m_new[i]
                                                    d_smem[row] = d_new[i]
                                                    m_prev_smem[row] = m_prev[i]
                                        T.tvm_storage_sync("shared")

                                        # Update O
                                        with T.block():
                                            for li, lj, lk in T.grid(tile_x, tile_y, tile_z):
                                                with T.block("O_gemm"):
                                                    i, j, k = T.axis.remap("SSR", [li, lj, lk])
                                                    with T.init():
                                                        O_local[i, j] *= T.exp2(m_prev_smem[i] - m_smem[i])
                                                    O_local[i, j] += S_smem[i, k] * T.cast(V_smem[k, j], "float32")

                                    # Store O from smem to gmem
                                    for li, lj in T.grid(tile_x, tile_y):
                                        with T.block("O_store"):
                                            i, j = T.axis.remap("SS", [li, lj])
                                            cur_L: T.int32 = q_indptr[b_idx] + (LH_start + i) // group_size
                                            cur_H_qo: T.int32 = by * group_size + (LH_start + i) % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                output[cur_L, cur_H_qo, j] = O_local[i, j] / d_smem[i]

                                    # Store LSE to gmem
                                    for li in T.grid(tile_x):
                                        with T.block("lse_store"):
                                            i = T.axis.remap("S", [li])
                                            cur_L: T.int32 = q_indptr[b_idx] + (LH_start + i) // group_size
                                            cur_H_qo: T.int32 = by * group_size + (LH_start + i) % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                lse[cur_L, cur_H_qo] = m_smem[i] + T.log2(d_smem[i])

                                    # move to next tile
                                    tile_id[0] += NUM_BLKS
    # fmt: on
    # pylint: enable=line-too-long,invalid-name,too-many-branches
    sch = tir.Schedule(batch_tree_attn)

    def get_tile_size(x, y, t):
        cnt = (x * y) // t
        assert (x * y) % t == 0
        tile_y = (int)(math.ceil(math.sqrt(cnt)))
        while (cnt % tile_y != 0 or y % tile_y != 0) and tile_y <= cnt:
            tile_y += 1
        assert tile_y <= cnt
        tile_x = cnt // tile_y
        return tile_x, tile_y

    def apply_to_qkv_load(sch: tir.Schedule, block):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        loop = sch.fuse(loop_x, loop_y)
        _, ty, tx, vec = sch.split(
            loop, factors=[None, num_warps, bdx, LOAD_VEC], preserve_unit_iters=True
        )
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)

    def apply_to_so_ewise(sch: tir.Schedule, block, tile):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[None, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    def apply_to_gemm(  # pylint: disable=unused-argument
        sch: tir.Schedule, block, tile, read_0, read_1, r_len=8, k_major=False
    ):
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
        sch.decompose_reduction(block, ty)

    def apply_to_md(sch, block):
        loop = sch.get_loops(block)[-1]
        _, ty, tx = sch.split(loop, factors=[None, num_warps, bdx])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

    tile_s = get_tile_size(tile_x, tile_z, bdx * num_warps)
    tile_o = get_tile_size(tile_x, tile_y, bdx * num_warps)
    apply_to_gemm(sch, sch.get_block("S_gemm"), tile_s, 0, 1, k_major=True)
    apply_to_gemm(sch, sch.get_block("O_gemm"), tile_o, 2, 3, k_major=False)
    apply_to_so_ewise(sch, sch.get_block("S_store"), tile_s)
    apply_to_so_ewise(sch, sch.get_block("O_init"), tile_o)
    apply_to_so_ewise(sch, sch.get_block("O_store"), tile_o)
    apply_to_qkv_load(sch, sch.get_block("Q_load"))
    apply_to_qkv_load(sch, sch.get_block("KV_load"))

    apply_to_md(sch, sch.get_block("lse_store"))
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


def _merge_state_inplace(
    num_heads, head_dim, v_dtype, target: Target
):  # pylint: disable=unused-argument
    # pylint: disable=invalid-name
    v_dtype_bytes = 2
    VEC_SIZE = min(max(8 // v_dtype_bytes, head_dim // 32), 4)
    bdx = head_dim // VEC_SIZE
    bdy = num_heads
    max_num_threads_per_block = get_max_num_threads_per_block(target)
    while bdx * bdy > max_num_threads_per_block and bdy > 1:
        bdy //= 2
    gdy = num_heads // bdy

    # undefined vars used
    @T.prim_func(check_well_formed=False)
    def merge_state_inplace(
        v: T.handle,
        s: T.handle,
        v_other: T.handle,
        s_other: T.handle,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        N = T.int32(is_size_var=True)
        H = T.int32(is_size_var=True)
        D = T.int32(is_size_var=True)

        V = T.match_buffer(v, (N, H, D), v_dtype)
        S = T.match_buffer(s, (N, H), "float32")
        V_other = T.match_buffer(v_other, (N, H, D), v_dtype)
        S_other = T.match_buffer(s_other, (N, H), "float32")

        for bx in T.thread_binding(N, thread="blockIdx.x"):
            for by in T.thread_binding(gdy, thread="blockIdx.y"):
                for ty in T.thread_binding(bdy, thread="threadIdx.y"):
                    for tx in T.thread_binding(bdx, thread="threadIdx.x"):
                        with T.block("merge"):
                            s_val = _var("float32")
                            s_other_val = _var("float32")
                            s_max = _var("float32")
                            scale = _var("float32")
                            other_scale = _var("float32")

                            v_vec = T.alloc_buffer((VEC_SIZE,), v_dtype, scope="local")
                            v_other_vec = T.alloc_buffer((VEC_SIZE,), v_dtype, scope="local")

                            s_val[0] = S[bx, ty + by * bdy]
                            s_other_val[0] = S_other[bx, ty + by * bdy]
                            s_max[0] = T.max(s_val[0], s_other_val[0])
                            s_val[0] = T.exp2(s_val[0] - s_max[0])
                            s_other_val[0] = T.exp2(s_other_val[0] - s_max[0])
                            scale[0] = s_val[0] / (s_val[0] + s_other_val[0])
                            other_scale[0] = s_other_val[0] / (s_val[0] + s_other_val[0])

                            # load v
                            for vec in T.vectorized(VEC_SIZE):
                                v_vec[vec] = V[bx, ty + by * bdy, tx * VEC_SIZE + vec]
                            # load v_other
                            for vec in T.vectorized(VEC_SIZE):
                                v_other_vec[vec] = V_other[bx, ty + by * bdy, tx * VEC_SIZE + vec]

                            # merge
                            for vec in T.serial(VEC_SIZE):
                                v_vec[vec] = (
                                    v_vec[vec] * scale[0] + v_other_vec[vec] * other_scale[0]
                                )

                            # store v
                            for vec in T.vectorized(VEC_SIZE):
                                V[bx, ty + by * bdy, tx * VEC_SIZE + vec] = v_vec[vec]

                            # store s
                            S[bx, ty + by * bdy] = T.log2(s_val[0] + s_other_val[0]) + s_max[0]

    # pylint: enable=invalid-name
    return merge_state_inplace


def _copy_single_page(num_heads, page_size, head_dim, dtype, target: Target):
    tx = 256 if str(target.kind) == "webgpu" else 1024

    @T.prim_func
    def copy_single_page(
        pages: T.handle,
        src_page_id: T.int64,
        tgt_page_id: T.int64,
        copy_length: T.int64,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        num_pages = T.int32()
        P = T.match_buffer(pages, (num_pages, 2, num_heads, page_size, head_dim), dtype)

        for b in T.thread_binding(
            (copy_length * num_heads * head_dim + tx - 1) // tx, thread="blockIdx.x"
        ):
            for t in T.thread_binding(tx, thread="threadIdx.x"):
                with T.block("copy"):
                    vh = T.axis.spatial(
                        num_heads,
                        T.Cast("int32", (b * tx + t) // (copy_length * head_dim)),
                    )
                    vp = T.axis.spatial(
                        copy_length,
                        (b * tx + t) % (copy_length * head_dim) // head_dim,
                    )
                    vd = T.axis.spatial(
                        head_dim,
                        T.Cast(
                            "int32",
                            (b * tx + t) % head_dim,
                        ),
                    )
                    P[tgt_page_id, 0, vh, vp, vd] = P[src_page_id, 0, vh, vp, vd]
                    P[tgt_page_id, 1, vh, vp, vd] = P[src_page_id, 1, vh, vp, vd]

    return copy_single_page


def _compact_kv_copy(num_heads, head_dim, dtype, target: Target):
    tx = 256 if str(target.kind) == "webgpu" else 1024

    @T.prim_func
    def compact_kv_copy(
        var_pages: T.handle,
        var_copy_length_indptr: T.handle,
        var_copy_src_dst_pos: T.handle,
        batch_size: T.int32,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        num_pages = T.int32()
        total_copy_length = T.int32()
        copy_length_indptr_elem_offset = T.int32()
        copy_src_dst_pos_elem_offset = T.int32()
        pages = T.match_buffer(var_pages, (num_pages, 2, num_heads, 16, head_dim), dtype)
        copy_length_indptr = T.match_buffer(
            var_copy_length_indptr,
            (batch_size + 1,),
            "int32",
            elem_offset=copy_length_indptr_elem_offset,
        )
        copy_src_dst_pos = T.match_buffer(
            var_copy_src_dst_pos,
            (2, total_copy_length),
            "int32",
            elem_offset=copy_src_dst_pos_elem_offset,
        )

        for bhd_o in T.thread_binding(
            (batch_size * num_heads * head_dim + tx - 1) // tx, thread="blockIdx.x"
        ):
            for bhd_i in T.thread_binding(tx, thread="threadIdx.x"):
                b: T.int32 = (bhd_o * tx + bhd_i) // (num_heads * head_dim)
                h: T.int32 = (bhd_o * tx + bhd_i) // head_dim % num_heads
                d: T.int32 = (bhd_o * tx + bhd_i) % head_dim
                if (bhd_o * tx + bhd_i) < batch_size * num_heads * head_dim:
                    for i in T.serial(copy_length_indptr[b + 1] - copy_length_indptr[b]):
                        src_pos: T.int32 = copy_src_dst_pos[0, copy_length_indptr[b] + i]
                        dst_pos: T.int32 = copy_src_dst_pos[1, copy_length_indptr[b] + i]
                        pages[dst_pos // 16, 0, h, dst_pos % 16, d] = pages[
                            src_pos // 16, 0, h, src_pos % 16, d
                        ]
                        pages[dst_pos // 16, 1, h, dst_pos % 16, d] = pages[
                            src_pos // 16, 1, h, src_pos % 16, d
                        ]

    return compact_kv_copy


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
