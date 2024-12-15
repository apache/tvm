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
    tree_attn_with_paged_kv_cache,
)
from tvm.runtime import ShapeTuple


def get_comm_rank():
    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        return comm, rank
    except ImportError:
        return None, 0


comm, rank = get_comm_rank()

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
rope_scaling = {}
dtype = None
device = tvm.cuda(rank)

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
fnvshmem_get_uid = None
fnvshmem_init = None
fdisagg_mark_send = None
fdisagg_prepare_recv = None

ftranspose_append = None
fcopy_cache = None
fattn_prefill = None
fattn_decode = None
fattn_prefill_sliding_window = None
fattn_decode_sliding_window = None
fattn_prefill_ragged = None
fattn_prefill_with_tree_mask = None
fattn_prefill_with_tree_mask_paged_kv_cache = None
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
    global fattn_prefill_ragged, fattn_prefill_with_tree_mask, fattn_prefill_with_tree_mask_paged_kv_cache
    global fattn_prefill_sliding_window, fattn_decode_sliding_window
    global fmerge_state, fsplit_rotary, fattention_rotary, fcopy_single_page, fcompact_copy
    global fnvshmem_get_uid, fnvshmem_init, fdisagg_mark_send, fdisagg_prepare_recv

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

    fnvshmem_get_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
    fnvshmem_init = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    fdisagg_mark_send = tvm.get_global_func("vm.builtin.kv_cache_disagg_mark_send")
    fdisagg_prepare_recv = tvm.get_global_func("vm.builtin.kv_cache_disagg_prepare_recv")

    target = tvm.target.Target.from_device(device)
    builts = []
    for tir_func in [
        _kv_cache_transpose_append(num_kv_heads, head_dim, dtype),
        _kv_cache_debug_get_kv(num_layers, num_kv_heads, head_dim, dtype),
        _attention_prefill(
            num_kv_heads, num_qo_heads, head_dim, dtype, False, rope_scaling, target
        ),
        _attention_decode(num_kv_heads, num_qo_heads, head_dim, dtype, False, rope_scaling, target),
        _attention_prefill(num_kv_heads, num_qo_heads, head_dim, dtype, True, rope_scaling, target),
        _attention_decode(num_kv_heads, num_qo_heads, head_dim, dtype, True, rope_scaling, target),
        _attention_prefill_ragged(
            num_kv_heads, num_qo_heads, head_dim, dtype, rope_scaling, target
        ),
        tree_attn(num_kv_heads, num_qo_heads, head_dim, dtype, rope_scaling, target),
        tree_attn_with_paged_kv_cache(
            num_kv_heads, num_qo_heads, head_dim, dtype, rope_scaling, target
        ),
        _merge_state_inplace(num_qo_heads, head_dim, dtype, target),
        llama_rope_with_position_map(
            rope_theta, rope_scale, head_dim, num_qo_heads, num_kv_heads, dtype, rope_scaling
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
        fattn_prefill_with_tree_mask_paged_kv_cache,
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
        fattn_prefill_with_tree_mask_paged_kv_cache,
        None,
        True,
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
            ["float32", "float16"],
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
    only_update_host=False,
    skip_add_sequence=False,
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
            if not only_update_host:
                ffork_sequence(kv_cache, fork_parent_id, seq_id, fork_pos)
            if fork_pos == -1:
                cached_k[seq_id] = cached_k[fork_parent_id]
                cached_v[seq_id] = cached_v[fork_parent_id]
            else:
                cached_k[seq_id] = cached_k[fork_parent_id][::, :fork_pos]
                cached_v[seq_id] = cached_v[fork_parent_id][::, :fork_pos]
        elif seq_id not in cached_k:
            if not only_update_host and not skip_add_sequence:
                fadd_sequence(kv_cache, seq_id)
            cached_k[seq_id] = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)
            cached_v[seq_id] = np.zeros((num_layers, 0, num_kv_heads, head_dim), dtype)

    flattened_token_tree_parent_ptr = None
    token_tree_node_depths_list: List[Optional[List[int]]] = [None for _ in batch]
    if token_tree_parent_ptr_list:
        assert len(token_tree_node_depths_list) == len(seq_ids)
        if accepted_leaf_indices is not None:
            assert len(accepted_leaf_indices) == len(seq_ids)
        flattened_token_tree_parent_ptr = []
        for i, (token_tree_parent_ptr, append_length) in enumerate(
            zip(token_tree_parent_ptr_list, append_lengths)
        ):
            assert len(token_tree_parent_ptr) >= append_length
            # parent pointer for the last `append_length` nodes (the new tokens)
            append_token_tree_parent_ptr = token_tree_parent_ptr[-append_length:]
            flattened_token_tree_parent_ptr += append_token_tree_parent_ptr
            token_tree_node_depths = []
            for parent in token_tree_parent_ptr:
                token_tree_node_depths.append(
                    0 if parent == -1 else token_tree_node_depths[parent] + 1
                )
            # depth of each node in the tree (this contains more than the last `append_length` nodes)
            token_tree_node_depths_list[i] = token_tree_node_depths
    if not only_update_host:
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

        rope_offset = cached_k[seq_id].shape[1]
        if token_tree_parent_ptr_list is not None:
            prev_tree_size = len(token_tree_parent_ptr_list[i]) - append_length
            assert prev_tree_size >= 0
            rope_offset -= prev_tree_size
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
                                rope_offset,
                                rope_scale,
                                rope_theta,
                                (
                                    token_tree_node_depths_list[i][-append_length:]
                                    if token_tree_node_depths_list[i] is not None
                                    else None
                                ),
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
        if not only_update_host:
            fattention_with_fuse_qkv(kv_cache, layer_id, 1.0, qkv, outputs)

        # Compute attention expected results.
        outputs = np.expand_dims(outputs.numpy(), axis=0)
        sum_length = 0
        for i, (seq_id, append_length) in enumerate(batch):
            assert cached_k[seq_id].shape[1] == cached_v[seq_id].shape[1] >= append_length

            rope_offset = cached_k[seq_id].shape[1]
            if token_tree_parent_ptr_list is not None:
                rope_offset -= len(token_tree_parent_ptr_list[i])
            else:
                rope_offset -= append_length
            q_seq = (
                q_array[i][layer_id]
                if rope_mode == RopeMode.NONE
                else f_apply_rotary(
                    q_array[i][layer_id],
                    rope_offset,
                    rope_scale,
                    rope_theta,
                    (
                        token_tree_node_depths_list[i][-append_length:]
                        if token_tree_node_depths_list[i] is not None
                        else None
                    ),
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
                tree_size = len(token_tree_parent_ptr_list[i])
                tree_mask = np.full(
                    (tree_size, tree_size), np.finfo("float32").min, dtype="float32"
                )
                for i, parent in enumerate(token_tree_parent_ptr_list[i]):
                    if parent != -1:
                        tree_mask[i] = tree_mask[parent]
                    tree_mask[i, i] = np.finfo("float32").max
                tree_mask = np.broadcast_to(tree_mask, (num_qo_heads, *tree_mask.shape))
                mask[:, :, -tree_size:] = tree_mask[:, -append_length:, :]

            softmax_input = np.minimum(softmax_input, mask)

            results = np.expand_dims(
                (scipy.special.softmax(softmax_input, axis=-1) @ v_seq.astype("float32")).transpose(
                    1, 0, 2
                ),
                axis=0,
            ).astype(dtype)
            if not only_update_host:
                tvm.testing.assert_allclose(
                    outputs[:, sum_length : sum_length + append_length, ...],
                    results,
                    rtol=1e-3,
                    atol=1e-3,
                )
            sum_length += append_length
    if not only_update_host:
        fend_forward(kv_cache)

    if accepted_leaf_indices is not None:
        seq_ids = [seq_id for seq_id, _ in batch]
        if not only_update_host:
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
    if not only_update_host:
        verify_cached_kv(kv_cache, seq_ids, cached_k, cached_v)


@pytest.mark.skip(reason="Require NVSHMEM")
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


@pytest.mark.skip(reason="Require NVSHMEM")
def test_paged_attention_kv_cache_transfer(kv_cache_and_config):
    kv_cache, rope_mode, support_sliding_window = kv_cache_and_config
    if support_sliding_window:
        # Normal RoPE mode under sliding window settings is not supported.
        return
    np.random.seed(0)
    fclear(kv_cache)
    # Prefill.
    prefill_operation_seq = [[(0, 6)], [(1, 8)], [(2, 11)], [(3, 16)], [(4, 19), (5, 20)]]
    prefill_operation_seq += [[(6, 21), (7, 24)], [(2, 5), (4, 7), (8, 24)]]
    prefill_operation_seq += [[(6, 13)], [(8, 19)], [(0, 1)], [(1, 3), (3, 8), (5, 12), (7, 11)]]
    prefill_len = {i: 0 for i in range(9)}
    for batch in prefill_operation_seq:
        for seq_id, append_length in batch:
            prefill_len[seq_id] += append_length
    # Decode
    decode_operation_seq = [
        [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]
    ]
    decode_operation_seq += [
        [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]
    ]
    decode_operation_seq += [[(0, 1), (2, 1), (4, 1), (6, 1), (8, 1)]]
    decode_operation_seq += [[(4, 1), (5, 1), (6, 1), (7, 1), (8, 1)]]

    cached_k = {}
    cached_v = {}
    if rank == 0:
        for seq_id, _ in prefill_len.items():
            fadd_sequence(kv_cache, seq_id)
        remote_pos_maps = None
        remote_pos_maps = comm.bcast(remote_pos_maps, root=1)
        comm.Barrier()
        for seq_id in prefill_len.keys():
            fdisagg_mark_send(kv_cache, seq_id, 0, ShapeTuple(remote_pos_maps[seq_id]), 1)
        for batch in prefill_operation_seq:
            apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v, skip_add_sequence=True)
        device.sync()
        comm.Barrier()
    else:
        remote_pos_maps = []
        for seq_id, len in prefill_len.items():
            fadd_sequence(kv_cache, seq_id)
            compressed_pos_map = list(fdisagg_prepare_recv(kv_cache, seq_id, len))
            remote_pos_maps.append(compressed_pos_map)
        remote_pos_maps = comm.bcast(remote_pos_maps, root=1)
        comm.Barrier()
        for batch in prefill_operation_seq:
            apply_attention(
                kv_cache,
                rope_mode,
                batch,
                cached_k,
                cached_v,
                only_update_host=True,
                skip_add_sequence=True,
            )
        comm.Barrier()
        for batch in decode_operation_seq:
            apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v, skip_add_sequence=True)


def init_nvshmem(num_workers, pe_offset):
    if rank == 0:
        f_init_nvshmem_uid = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem_uid")
        uid = f_init_nvshmem_uid()
    else:
        uid = None
    uid = comm.bcast(uid, root=0)
    init_func = tvm.get_global_func("runtime.disco.nvshmem.init_nvshmem")
    init_func(uid, num_workers, pe_offset)


if __name__ == "__main__":
    # To run this test, install mpi4py first, and then run
    # mpirun -np 2 python tests/python/relax/nvshmem/test_runtime_builtin_kv_cache_transfer.py
    HEAD_DIMS = [128]
    DTYPES = ["float16"]
    ROPE_MODES = [RopeMode.NONE]
    SUPPORT_SLIDING_WINDOW = [False]
    init_nvshmem(2, rank)
    for head_dim, dtype, rope_mode, support_sliding_window in itertools.product(
        HEAD_DIMS, DTYPES, ROPE_MODES, SUPPORT_SLIDING_WINDOW
    ):
        set_global_func(head_dim, dtype)
        cache = create_kv_cache(head_dim, dtype, rope_mode, support_sliding_window)
        cache_and_config = (cache, rope_mode, support_sliding_window)
        test_paged_attention_kv_cache_transfer(cache_and_config)
