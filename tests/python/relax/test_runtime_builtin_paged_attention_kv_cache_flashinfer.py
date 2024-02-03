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
from tvm import tir
from tvm.runtime import ShapeTuple
from tvm.script import tir as T

reserved_nseq = 32
maximum_total_seq_length = 1024
prefill_chunk_size = 512
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
fattention_with_fuse_qkv = None
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

ftranspose_append = None
fsplit_rotary = None
fcopy_cache = None


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
            T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
            T.writes(
                pages[position_map[vgpos] // page_size, 0, vh, position_map[vgpos] % page_size, vf]
            )
            position: T.int64 = T.Cast("int64", position_map[vgpos])
            pages[
                T.floordiv(position, page_size), 0, vh, T.floormod(position, page_size), vf
            ] = k_data[vgpos, vh, vf]
        with T.block("v_transpose_append"):
            vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
            T.reads(position_map[vgpos], k_data[vgpos, vh, vf])
            T.writes(
                pages[position_map[vgpos] // page_size, 1, vh, position_map[vgpos] % page_size, vf]
            )
            position: T.int64 = T.Cast("int64", position_map[vgpos])
            pages[
                T.floordiv(position, page_size), 1, vh, T.floormod(position, page_size), vf
            ] = v_data[vgpos, vh, vf]


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

    @T.prim_func(private=True)
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
        qkv = T.match_buffer(var_qkv, (seq_len, fused_heads, head_dim), dtype)
        q = T.match_buffer(var_q, (seq_len, num_q_heads, head_dim), dtype)
        k = T.match_buffer(var_k, (seq_len, num_kv_heads, head_dim), dtype)
        v = T.match_buffer(var_v, (seq_len, num_kv_heads, head_dim), dtype)
        position_map = T.match_buffer(var_position_map, (seq_len,), "int32")
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


def set_global_func():
    global fclear, fcreate, fadd_sequence, fremove_sequence, ffork_sequence, fpopn
    global fbegin_forward, fend_forward, fattention, fattention_with_fuse_qkv, fdebug_get_kv
    global fattention_prefill, fattention_prefill_begin_forward, fattention_prefill_end_forward
    global fattention_decode, fattention_decode_begin_forward, fattention_decode_end_forward
    global fattention_prefill_ragged
    global fattention_prefill_ragged_begin_forward
    global fattention_prefill_ragged_end_forward
    global fattention_merge_state, fsplit_rotary, fattention_rotary
    global ftranspose_append, fcopy_cache

    fclear = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_clear")
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
    fadd_sequence = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_add_sequence")
    fremove_sequence = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_remove_sequence")
    ffork_sequence = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_fork_sequence")
    fpopn = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_popn")
    fbegin_forward = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_begin_forward")
    fend_forward = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_end_forward")
    fattention = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_attention")
    fattention_with_fuse_qkv = tvm.get_global_func(
        "vm.builtin.paged_attention_kv_cache_attention_with_fused_qkv"
    )
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

    target = tvm.target.Target("nvidia/geforce-rtx-3090-ti")
    builts = []
    for tir_func in [
        kv_cache_transpose_append,
        llama_rope_with_position_map(
            rope_theta, rope_scale, head_dim, num_qo_heads, num_kv_heads, dtype
        ),
        copy_cache,
    ]:
        mod = tvm.IRModule({"main": tir_func})
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
        f = tvm.build(mod["main"], target=target)
        builts.append(f.entry_func)

    ftranspose_append, fsplit_rotary, fcopy_cache = builts


def create_kv_cache(rope_mode):
    cache = fcreate(
        tvm.runtime.ShapeTuple(
            [reserved_nseq, maximum_total_seq_length, prefill_chunk_size, page_size]
        ),
        num_layers,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        rope_mode,
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
        fattention_merge_state,
        fsplit_rotary,
        fattention_rotary,
        fcopy_cache,
    )
    return cache


@pytest.fixture(params=[0, 1])
def kv_cache_and_rope_mode(request):
    set_global_func()
    return create_kv_cache(request.param), request.param


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
    rope_mode: int,
    batch: List[Tuple[Union[int, Tuple[int, int]], int]],
    cached_k: Dict[int, np.ndarray],
    cached_v: Dict[int, np.ndarray],
    fuse_qkv: bool,
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
                        new_k[l]
                        if rope_mode == 1
                        else f_apply_rotary(
                            new_k[l], cached_k[seq_id].shape[1], rope_scale, rope_theta
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
        if not fuse_qkv:
            queries = tvm.nd.array(queries_np, device=device)
            keys = tvm.nd.array(keys_np, device=device)
            values = tvm.nd.array(values_np, device=device)
            outputs = tvm.nd.empty(queries.shape, dtype, device=device)
            fattention(kv_cache, layer_id, queries, keys, values, outputs)
        else:
            qkv = tvm.nd.array(np.concatenate([queries_np, keys_np, values_np], axis=1), device)
            outputs = tvm.nd.empty(queries_np.shape, dtype, device=device)
            fattention_with_fuse_qkv(kv_cache, layer_id, qkv, outputs)

        # Compute attention expected results.
        outputs = np.expand_dims(outputs.numpy(), axis=0)
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
            k_seq = (
                cached_k[seq_id][layer_id]
                if rope_mode == 0
                else f_apply_rotary(cached_k[seq_id][layer_id], 0, rope_scale, rope_theta)
            ).transpose(1, 2, 0)
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
@pytest.mark.parametrize("fuse_qkv", [False, True])
def test_paged_attention_kv_cache_prefill_and_decode(kv_cache_and_rope_mode, fuse_qkv):
    kv_cache, rope_mode = kv_cache_and_rope_mode
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
        apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v, fuse_qkv)


@pytest.mark.skip(reason="Require FlashInfer enabled")
@pytest.mark.parametrize("fuse_qkv", [False, True])
def test_paged_attention_kv_cache_remove_sequence(kv_cache_and_rope_mode, fuse_qkv):
    kv_cache, rope_mode = kv_cache_and_rope_mode
    fclear(kv_cache)

    num_sequences = 5
    batch = [(seq_id, 1) for seq_id in range(num_sequences)]
    cached_k = {}
    cached_v = {}
    for seq_id_to_remove in range(num_sequences):
        apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v, fuse_qkv)
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
@pytest.mark.parametrize("fuse_qkv", [False, True])
def test_paged_attention_kv_cache_fork_sequence(kv_cache_and_rope_mode, fuse_qkv):
    kv_cache, rope_mode = kv_cache_and_rope_mode
    fclear(kv_cache)

    cached_k = {}
    cached_v = {}
    batch = [(0, 60), (1, 88), (2, 17), (3, 4)]
    apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v, fuse_qkv)
    # Fork existing sequences.
    apply_attention(kv_cache, rope_mode, [((4, 3), 35)], cached_k, cached_v, fuse_qkv)
    apply_attention(kv_cache, rope_mode, [((5, 0), 20)], cached_k, cached_v, fuse_qkv)
    apply_attention(kv_cache, rope_mode, [((6, 5), 102)], cached_k, cached_v, fuse_qkv)
    apply_attention(kv_cache, rope_mode, [((7, 0), 3)], cached_k, cached_v, fuse_qkv)
    apply_attention(kv_cache, rope_mode, [((8, 5), 71)], cached_k, cached_v, fuse_qkv)
    apply_attention(kv_cache, rope_mode, [((9, 5), 20)], cached_k, cached_v, fuse_qkv)
    # Mixture of decode and prefill.
    operation_seq = [
        [(2, 1), (4, 1), (7, 1), (6, 1), (8, 1), (9, 1)],
        [(7, 1), (6, 1), (8, 1), (9, 1)],
        [(7, 1), (1, 1), (6, 1), (2, 1), (8, 1), (4, 1), (9, 1)],
        [(7, 10), (6, 2), (8, 3), (9, 4)],
    ]
    for batch in operation_seq:
        apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v, fuse_qkv)


@pytest.mark.skip(reason="Require FlashInfer enabled")
@pytest.mark.parametrize("fuse_qkv", [False, True])
def test_paged_attention_kv_cache_popn(kv_cache_and_rope_mode, fuse_qkv):
    kv_cache, rope_mode = kv_cache_and_rope_mode
    fclear(kv_cache)

    cached_k = {}
    cached_v = {}
    batch = [(0, 35), (1, 88), (2, 17), (3, 4)]
    apply_attention(kv_cache, rope_mode, batch, cached_k, cached_v, fuse_qkv)
    apply_attention(kv_cache, rope_mode, [((4, 3), 35)], cached_k, cached_v, fuse_qkv)

    popn_operations = [(0, 17), (1, 57), (2, 16), (3, 0), (4, 19)]
    for seq_id, pop_length in popn_operations:
        fpopn(kv_cache, seq_id, pop_length)
        if pop_length != 0:
            cached_k[seq_id] = cached_k[seq_id][:, :-pop_length, ...]
            cached_v[seq_id] = cached_v[seq_id][:, :-pop_length, ...]
        verify_cached_kv(kv_cache, seq_ids=list(range(5)), expected_k=cached_k, expected_v=cached_v)


if __name__ == "__main__":
    set_global_func()
    for rope_mode in [0, 1]:
        cache = create_kv_cache(rope_mode)
        for fuse_qkv in [False, True]:
            test_paged_attention_kv_cache_prefill_and_decode((cache, rope_mode), fuse_qkv)
            test_paged_attention_kv_cache_remove_sequence((cache, rope_mode), fuse_qkv)
            test_paged_attention_kv_cache_fork_sequence((cache, rope_mode), fuse_qkv)
            test_paged_attention_kv_cache_popn((cache, rope_mode), fuse_qkv)
