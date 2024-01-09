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
import math
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
fadd_sequence = None
fremove_sequence = None
ffork_sequence = None
fpopn = None
fbegin_forward = None
fend_forward = None
fattention = None
fdebug_get_kv = None


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
    global fclear, fadd_sequence, fremove_sequence, ffork_sequence, fpopn
    global fbegin_forward, fend_forward, fattention, fdebug_get_kv

    fclear = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_clear")
    fadd_sequence = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_add_sequence")
    fremove_sequence = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_remove_sequence")
    ffork_sequence = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_fork_sequence")
    fpopn = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_popn")
    fbegin_forward = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_begin_forward")
    fend_forward = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_end_forward")
    fattention = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_attention")
    fdebug_get_kv = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_debug_get_kv")


def create_kv_cache():
    set_global_func()
    target = tvm.target.Target("cuda")
    builts = []
    for tir_func in [
        kv_cache_transpose_append,
        copy_cache,
        _attention_prefill(num_kv_heads, num_qo_heads, head_dim, dtype),
        _attention_decode(num_kv_heads, num_qo_heads, head_dim, dtype),
        _inplace_rope(rope_theta, rope_scale, head_dim, num_qo_heads, num_kv_heads, dtype),
    ]:
        mod = tvm.IRModule({"main": tir_func})
        with target:
            mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(mod)
        f = tvm.build(mod["main"], target=target)
        builts.append(f.entry_func)

    ftranspose_append, fcopy_cache, fattn_prefill, fattn_decode, fbatch_rotary = builts
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create_reduced")
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
        fattn_prefill,
        fattn_decode,
        fbatch_rotary,
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


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
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


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
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


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_paged_attention_kv_cache_popn(kv_cache):
    fclear(kv_cache)

    cached_k = {}
    cached_v = {}
    batch = [(0, 35), (1, 88), (2, 17), (3, 4)]
    apply_attention(kv_cache, batch, cached_k, cached_v)

    popn_operations = [(0, 17), (1, 57), (2, 16), (3, 0)]
    for seq_id, pop_length in popn_operations:
        fpopn(kv_cache, seq_id, pop_length)
        if pop_length != 0:
            cached_k[seq_id] = cached_k[seq_id][:, :-pop_length, ...]
            cached_v[seq_id] = cached_v[seq_id][:, :-pop_length, ...]
        verify_cached_kv(kv_cache, seq_ids=list(range(4)), expected_k=cached_k, expected_v=cached_v)


def _inplace_rope(
    theta: float,
    scale: float,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    dtype: str,
):
    assert head_dim <= 128, "Rotary embedding currently only supports head_dim <= 128"
    rotary_dim = head_dim

    def _rope(
        x: T.Buffer,
        s: tir.Var,
        h: tir.Var,
        d: tir.Var,
        rope_offset: tir.Var,
        instance_offset: tir.Var,
    ):
        cos_freq, sin_freq = rope_freq((s + rope_offset) * scale, d, rotary_dim, theta, dtype)
        cos = cos_freq * x[s + instance_offset, h, d]
        sin = sin_freq * tir.if_then_else(
            d < rotary_dim // 2,
            -x[s + instance_offset, h, d + rotary_dim // 2],
            x[s + instance_offset, h, d - rotary_dim // 2],
        )
        return cos + sin

    # fmt: off
    @T.prim_func
    def tir_rotary(
        var_q: T.handle,
        var_k: T.handle,
        var_append_len_indptr: T.handle,
        var_rope_offsets: T.handle,
        _0: T.int32,
        _1: T.int32,
        _2: T.int32,
        _3: T.int32,
        _4: T.int32,
        _5: T.float32,
        _6: T.float32,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        total_len = T.int32()
        batch_size = T.int32()
        q = T.match_buffer(var_q, (total_len, num_q_heads, head_dim), dtype)
        k = T.match_buffer(var_k, (total_len, num_kv_heads, head_dim), dtype)
        rope_offsets = T.match_buffer(var_rope_offsets, (batch_size,), "int32")
        append_len_indptr = T.match_buffer(var_append_len_indptr, (batch_size + 1,), "int32")
        for b_h in T.thread_binding(batch_size * (num_q_heads + num_kv_heads), thread="blockIdx.x"):
            b: T.int32 = b_h // (num_q_heads + num_kv_heads)
            h: T.int32 = b_h % (num_q_heads + num_kv_heads)
            instance_offset: T.int32 = append_len_indptr[b]
            rope_offset: T.int32 = rope_offsets[b]
            append_len: T.int32 = append_len_indptr[b + 1] - append_len_indptr[b]
            for s0 in range(T.ceildiv(append_len, 32)):
                for s1 in T.thread_binding(32, thread="threadIdx.y"):
                    for d0 in T.thread_binding(T.ceildiv(head_dim, 4), thread="threadIdx.x"):
                        for d1 in T.vectorized(4):
                            s: T.int32 = s0 * 32 + s1
                            d: T.int32 = d0 * 4 + d1
                            if s < append_len and d < head_dim:
                                if h < num_q_heads:
                                    q[s + instance_offset, h, d] = _rope(q, s, h, d, rope_offset, instance_offset)
                                else:
                                    k[s + instance_offset, h - num_q_heads, d] = _rope(k, s, h - num_q_heads, d, rope_offset, instance_offset)
    return tir_rotary


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


def _attention_prefill(h_kv, h_q, d, dtype):
    assert dtype == "float16", f"TIR attention kernel does not support dtype {dtype} right now"
    # pylint: disable=invalid-name
    NUM_BLKS = 16
    LOAD_VEC = 8 // ((tvm.runtime.DataType(dtype).bits + 7) // 8)  # 8 bytes
    group_size = h_q // h_kv
    sm_scale = 1.0 / math.sqrt(float(d)) * math.log2(math.exp(1))

    num_warps = 4
    tile_x, tile_y, tile_z = 32, d, 16
    L_per_cta = tile_x // group_size

    def mask(causal, row, col, kv_len, qo_len):
        return T.if_then_else(
            causal > 0,
            col < kv_len - qo_len + row + 1,
            col < kv_len,
        )

    # pylint: disable=line-too-long,too-many-arguments,too-many-branches
    # fmt: off
    @T.prim_func
    def batch_prefill_paged_kv(
        _0: T.int32,  # pylint: disable=unused-argument
        var_q: T.handle, # [total_len, h_q, d]
        var_q_indptr: T.handle, # [batch_size + 1]
        var_pages: T.handle, # [max_num_pages, 2, h_kv, page_size, d]
        var_page_indptr: T.handle, # [batch_size + 1]
        var_page_values: T.handle, # [nnz_pages]
        var_last_page_len: T.handle, # [b]
        var_output: T.handle, # [total_len, h_q, d]
        var_lse: T.handle, # [total_len, h_q]
        causal: T.int32,
        _1: T.int32,
        _2: T.float32,
        _3: T.float32,
    ):
        batch_size = T.int32(is_size_var=True)
        total_len = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)
        page_size = T.int32(is_size_var=True)

        q = T.match_buffer(var_q, (total_len, h_q, d), dtype)
        q_indptr = T.match_buffer(var_q_indptr, (batch_size + 1,), "int32")
        pages = T.match_buffer(var_pages, (max_num_pages, 2, h_kv, page_size, d), dtype)
        page_indptr = T.match_buffer(var_page_indptr, (batch_size + 1,), "int32")
        page_values = T.match_buffer(var_page_values, (nnz_pages,), "int32")
        last_page_len = T.match_buffer(var_last_page_len, (batch_size,), "int32")
        output = T.match_buffer(var_output, (total_len, h_q, d), dtype)
        lse = T.match_buffer(var_lse, (total_len, h_q), "float32")  # pylint: disable=unused-variable

        # kernel code
        for lbx in T.thread_binding(NUM_BLKS, thread="blockIdx.x"):
            for lby in T.thread_binding(h_kv, thread="blockIdx.y"):
                for lty in T.thread_binding(num_warps, thread="threadIdx.y"):
                    for ltx in T.thread_binding(32, thread="threadIdx.x"):
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
                            m_new = _var("float32")
                            m_prev = _var("float32")
                            d_new = _var("float32")

                            Q_smem = T.alloc_buffer((tile_x, d), dtype, scope="shared")
                            K_smem = T.alloc_buffer((tile_z, d), dtype, scope="shared")
                            V_smem = T.alloc_buffer((tile_z, d), dtype, scope="shared")
                            S_smem = T.alloc_buffer((tile_x, tile_z), "float32", scope="shared")

                            S_local = T.alloc_buffer((tile_x, tile_z), "float32", scope="local")
                            O_local = T.alloc_buffer((tile_x, d), "float32", scope="local")

                            m_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")
                            m_prev_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")
                            d_smem = T.alloc_buffer((tile_x, ), "float32", scope="shared")

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
                                    L_start: T.int32 = q_indptr[b_idx] + tile_id[0] * L_per_cta
                                    H_qo_start: T.int32 = by * group_size

                                    cur_page_indptr_begin: T.int32 = page_indptr[b_idx]
                                    cur_page_indptr_end: T.int32 = page_indptr[b_idx + 1]
                                    cur_last_page_len: T.int32 = last_page_len[b_idx]
                                    kv_chunk_len[0] = T.if_then_else(
                                        cur_page_indptr_begin != cur_page_indptr_end,
                                        (cur_page_indptr_end - cur_page_indptr_begin - 1) * page_size + cur_last_page_len,
                                        0
                                    )
                                    T.tvm_storage_sync("shared")

                                    # init states
                                    for i in T.serial(T.ceildiv(tile_x, 32 * num_warps)):
                                        row: T.int32 = i * 32 * num_warps + ty * 32 + tx
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
                                            cur_L = L_start + i // group_size
                                            cur_H_qo = H_qo_start + i % group_size
                                            if cur_L < q_indptr[b_idx + 1]:
                                                Q_smem[i, j] = q[cur_L, cur_H_qo, j]
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
                                                    page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + T.floordiv(cur_L, page_size)]
                                                    page_offset: T.int32(is_size_var=True) = T.floormod(cur_L, page_size)
                                                    K_smem[i, j] = pages[page_no, 0, by, page_offset, j]
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
                                                    page_no: T.int32(is_size_var=True) = page_values[cur_page_indptr_begin + T.floordiv(cur_L, page_size)]
                                                    page_offset: T.int32(is_size_var=True) = T.floormod(cur_L, page_size)
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
                                                    S_local[i, j] += Q_smem[i, k] * K_smem[j, k] * sm_scale
                                        T.tvm_storage_sync("shared")
                                        for li, lj in T.grid(tile_x, tile_z):
                                            with T.block("S_store"):
                                                i, j = T.axis.remap("SS", [li, lj])
                                                S_smem[i, j] = S_local[i, j]
                                        T.tvm_storage_sync("shared")

                                        # Update S, m, d
                                        for i in T.serial(T.ceildiv(tile_x, 32 * num_warps)):
                                            row: T.int32 = i * 32 * num_warps + ty * 32 + tx
                                            if row < tile_x:
                                                with T.block("update1"):
                                                    m_prev[0] = m_smem[row]
                                                    m_new[0] = m_smem[row]
                                                    # mask out of kv_chunk_len S
                                                    for j in T.serial(tile_z):
                                                        if mask(causal,
                                                                row=tile_id[0] * L_per_cta + row // group_size,
                                                                col=L_kv_start + j,
                                                                kv_len=kv_chunk_len[0],
                                                                qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx]):
                                                            m_new[0] = T.max(m_new[0], S_smem[row, j])
                                                    d_new[0] = d_smem[row] * T.exp2(m_prev[0] - m_new[0])

                                        for i in T.serial(T.ceildiv(tile_x, 32 * num_warps)):
                                            row: T.int32 = i * 32 * num_warps + ty * 32 + tx
                                            with T.block("update"):
                                                for j in T.serial(tile_z):
                                                    # this is to avoid sync inside condition branch
                                                    if row < tile_x:
                                                        if mask(causal,
                                                                row=tile_id[0] * L_per_cta + row // group_size,
                                                                col=L_kv_start + j,
                                                                kv_len=kv_chunk_len[0],
                                                                qo_len=q_indptr[b_idx + 1] - q_indptr[b_idx]):
                                                            S_smem[row, j] = T.exp2(S_smem[row, j] - m_new[0])
                                                        else:
                                                            S_smem[row, j] = T.exp2(-5e4 - m_new[0])

                                        for i in T.serial(T.ceildiv(tile_x, 32 * num_warps)):
                                            row: T.int32 = i * 32 * num_warps + ty * 32 + tx
                                            if row < tile_x:
                                                with T.block("update"):
                                                    for j in T.serial(tile_z):
                                                        d_new[0] += S_smem[row, j]
                                                    m_smem[row] = m_new[0]
                                                    d_smem[row] = d_new[0]
                                                    m_prev_smem[row] = m_prev[0]
                                        T.tvm_storage_sync("shared")

                                        # Update O
                                        with T.block():
                                            for li, lj, lk in T.grid(tile_x, tile_y, tile_z):
                                                with T.block("O_gemm"):
                                                    i, j, k = T.axis.remap("SSR", [li, lj, lk])
                                                    with T.init():
                                                        O_local[i, j] *= T.exp2(m_prev_smem[i] - m_smem[i])
                                                    O_local[i, j] += S_smem[i, k] * V_smem[k, j]

                                    # Store O from smem to gmem
                                    for li, lj in T.grid(tile_x, tile_y):
                                        with T.block("O_store"):
                                            i, j = T.axis.remap("SS", [li, lj])
                                            if L_start + i // group_size < q_indptr[b_idx + 1]:
                                                output[L_start + i // group_size, H_qo_start + i % group_size, j] = O_local[i, j] / d_smem[i]

                                    # move to next tile
                                    tile_id[0] += NUM_BLKS
    # fmt: on
    # pylint: enable=line-too-long,invalid-name,too-many-arguments,too-many-branches
    sch = tir.Schedule(batch_prefill_paged_kv)

    def get_tile_size(x, y, t):
        cnt = (x * y) // t
        assert (x * y) % t == 0
        tile_y = (int)(math.ceil(math.sqrt(cnt)))
        while cnt % tile_y != 0 and y % tile_y != 0 and tile_y <= cnt:
            tile_y += 1
        assert tile_y <= cnt
        tile_x = cnt // tile_y
        return tile_x, tile_y

    def apply_to_qkv_load(sch: tir.Schedule, block):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        loop = sch.fuse(loop_x, loop_y)
        _, ty, tx, vec = sch.split(
            loop, factors=[None, num_warps, 32, LOAD_VEC], preserve_unit_iters=True
        )
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)

    def apply_to_so_ewise(sch: tir.Schedule, block, tile, vec_len=4):
        loop_x, loop_y = sch.get_loops(block)[-2:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[num_warps, 32])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        if tile[1] % vec_len == 0:
            yi, vec = sch.split(yi, factors=[None, vec_len])
            sch.vectorize(vec)
        elif tile[1] in [2, 4]:
            sch.vectorize(yi)

    def apply_to_gemm(  # pylint: disable=too-many-arguments,unused-argument
        sch: tir.Schedule, block, tile, read_0, read_1, r_len=8, k_major=False
    ):
        loop_x, loop_y, loop_z = sch.get_loops(block)[-3:]
        xo, xi = sch.split(loop_x, factors=[None, tile[0]])
        yo, yi = sch.split(loop_y, factors=[None, tile[1]])
        sch.reorder(xo, yo, xi, yi)
        t = sch.fuse(xo, yo)
        ty, tx = sch.split(t, factors=[num_warps, 32])
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")

        ko, ki = sch.split(loop_z, factors=[None, r_len])
        if k_major:
            sch.reorder(ko, xi, yi, ki)
        else:
            sch.reorder(ko, ki, xi, yi)
        sch.decompose_reduction(block, ty)

    tile_s = get_tile_size(tile_x, tile_z, 32 * num_warps)
    tile_o = get_tile_size(tile_x, tile_y, 32 * num_warps)
    apply_to_gemm(sch, sch.get_block("S_gemm"), tile_s, 0, 1, k_major=True)
    apply_to_gemm(sch, sch.get_block("O_gemm"), tile_o, 2, 3, k_major=False)
    apply_to_so_ewise(sch, sch.get_block("S_store"), tile_s)
    apply_to_so_ewise(sch, sch.get_block("O_init"), tile_o)
    apply_to_so_ewise(sch, sch.get_block("O_store"), tile_o)
    apply_to_qkv_load(sch, sch.get_block("Q_load"))
    apply_to_qkv_load(sch, sch.get_block("K_load"))
    apply_to_qkv_load(sch, sch.get_block("V_load"))
    return sch.mod["main"].with_attr("tir.is_scheduled", 1)


def _attention_decode(num_kv_heads, num_qo_heads, head_dim, qkv_dtype):
    assert (
        qkv_dtype == "float16"
    ), f"TIR attention kernel does not support dtype {qkv_dtype} right now"
    # pylint: disable=invalid-name
    qkv_dtype_bytes = 2
    H_qo = num_qo_heads
    H_kv = num_kv_heads
    D = head_dim

    GROUP_SIZE = H_qo // H_kv
    VEC_SIZE = max(8 // qkv_dtype_bytes, D // 32)
    bdx = D // VEC_SIZE
    bdy = GROUP_SIZE
    threads_per_CTA = max(128, bdx * bdy)
    bdz = threads_per_CTA // (bdx * bdy)
    tile_size_per_bdx = 4 if GROUP_SIZE == 1 else 1
    log2e = math.log2(math.exp(1))

    # pylint: disable=line-too-long,too-many-arguments,too-many-branches
    # fmt: off
    @T.prim_func
    def batch_decode_paged_kv(
        handler_id: T.int32,  # pylint: disable=unused-argument
        Q_handle: T.handle,
        pages_handle: T.handle,
        page_table_indptr_handle: T.handle,
        page_table_values_handle: T.handle,
        last_page_len_handle: T.handle,
        output_handle: T.handle,
        lse_handle: T.handle,
        rotary_mode: T.int32,
        rope_scale: T.float32,
        rope_theta: T.float32,
    ):
        T.func_attr({"tir.is_scheduled": 1})
        B = T.int32(is_size_var=True)
        page_size = T.int32(is_size_var=True)
        nnz_pages = T.int32(is_size_var=True)
        max_num_pages = T.int32(is_size_var=True)

        Q = T.match_buffer(Q_handle, (B, H_qo, D), qkv_dtype)
        pages = T.match_buffer(
            pages_handle, (max_num_pages, 2, H_kv, page_size, D), qkv_dtype
        )
        page_table_indptr = T.match_buffer(page_table_indptr_handle, (B + 1,), "int32")
        page_table_values = T.match_buffer(page_table_values_handle, (nnz_pages,), "int32")
        last_page_len = T.match_buffer(last_page_len_handle, (B,), "int32")
        output = T.match_buffer(output_handle, (B, H_qo, D), qkv_dtype)
        lse = T.match_buffer(lse_handle, (B, H_qo), "float32")  # pylint: disable=unused-variable

        sm_scale = 1.0 / math.sqrt(float(D)) * log2e

        for bx in T.thread_binding(B, thread="blockIdx.x"):
            for by in T.thread_binding(H_kv, thread="blockIdx.y"):
                for ty in T.thread_binding(bdy, thread="threadIdx.y"):
                    for tx in T.thread_binding(bdx, thread="threadIdx.x"):
                        for tz in T.thread_binding(bdz, thread="threadIdx.z"):
                            with T.block("attn"):
                                Q_local = T.alloc_buffer((VEC_SIZE,), qkv_dtype, scope="local")
                                kv_chunk_len = T.alloc_buffer((1,), "int32", scope="local")
                                K_smem = T.alloc_buffer((bdz * bdy * tile_size_per_bdx, D), qkv_dtype, scope="shared")
                                V_smem = T.alloc_buffer((bdz * bdy * tile_size_per_bdx, D), qkv_dtype, scope="shared")
                                S_allreduce = T.alloc_buffer((bdz, bdy, bdx), "float32", scope="shared")
                                O_allreduce = T.alloc_buffer((bdz, bdy, D), "float32", scope="shared")
                                md_allreduce = T.alloc_buffer((bdz, bdy, 2), "float32", scope="shared")

                                S_local = T.alloc_buffer((bdy * tile_size_per_bdx), "float32", scope="local")
                                K_local = T.alloc_buffer((VEC_SIZE,), qkv_dtype, scope="local")
                                V_local = T.alloc_buffer((VEC_SIZE,), qkv_dtype, scope="local")
                                offset = T.alloc_buffer((1,), "int32", scope="local")
                                m_prev = T.alloc_buffer((1,), "float32", scope="local")
                                d_prev = T.alloc_buffer((1,), "float32", scope="local")
                                other_m = T.alloc_buffer((1,), "float32", scope="local")
                                other_d = T.alloc_buffer((1,), "float32", scope="local")
                                other_o = T.alloc_buffer((VEC_SIZE,), "float32", scope="local")
                                st_m = T.alloc_buffer((1,), "float32", scope="local")
                                st_d = T.alloc_buffer((1,), "float32", scope="local")
                                O_local = T.alloc_buffer((VEC_SIZE,), "float32", scope="local")

                                batch_idx: T.int32 = bx
                                cur_page_indptr_begin: T.int32 = page_table_indptr[batch_idx]
                                cur_page_indptr_end: T.int32 = page_table_indptr[batch_idx + 1]
                                cur_last_page_len: T.int32 = last_page_len[batch_idx]
                                kv_chunk_len[0] = T.if_then_else(
                                    cur_page_indptr_begin != cur_page_indptr_end,
                                    (cur_page_indptr_end - cur_page_indptr_begin - 1) * page_size + cur_last_page_len,
                                    0
                                )

                                # init states
                                st_m[0] = -5e4
                                st_d[0] = 1.0
                                for vec in T.vectorized(VEC_SIZE):
                                    O_local[vec] = 0.0

                                # load q
                                for vec in T.vectorized(VEC_SIZE):
                                    Q_local[vec] = T.if_then_else(rotary_mode == 1,
                                                                  _rope(Q, kv_chunk_len[0]-1, head_dim, rope_theta, rope_scale, (bx, by * GROUP_SIZE + ty, tx * VEC_SIZE + vec)),
                                                                  Q[bx, by * GROUP_SIZE + ty, tx * VEC_SIZE + vec])

                                for iterator in T.serial(T.ceildiv(kv_chunk_len[0], tile_size_per_bdx * bdy * bdz)):
                                    tile_start_s: T.int32(is_size_var=True) = (tz * bdy + ty) * tile_size_per_bdx
                                    tile_start_g: T.int32(is_size_var=True) = ((iterator * bdz + tz) * bdy + ty) * tile_size_per_bdx
                                    # load K from global memory to shared memory
                                    for j in T.serial(tile_size_per_bdx):
                                        row_g: T.int32(is_size_var=True) = tile_start_g + j
                                        if row_g < kv_chunk_len[0]:
                                            page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + T.floordiv(row_g, page_size)]
                                            page_offset: T.int32(is_size_var=True) = T.floormod(row_g, page_size)
                                            for vec in T.vectorized(VEC_SIZE):
                                                K_smem[tile_start_s + j, tx * VEC_SIZE + vec] = T.if_then_else(rotary_mode == 1,
                                                                                                               _rope(pages, row_g, head_dim, rope_theta, rope_scale, (page_no, 0, by, page_offset, tx * VEC_SIZE + vec)),
                                                                                                               pages[page_no, 0, by, page_offset, tx * VEC_SIZE + vec])
                                        else:
                                            for vec in T.vectorized(VEC_SIZE):
                                                K_smem[tile_start_s + j, tx * VEC_SIZE + vec] = 0.0
                                    T.tvm_storage_sync("shared")
                                    # load V from global memory to shared memory
                                    for j in T.serial(tile_size_per_bdx):
                                        row_g: T.int32(is_size_var=True) = tile_start_g + j
                                        if row_g < kv_chunk_len[0]:
                                            page_no: T.int32(is_size_var=True) = page_table_values[cur_page_indptr_begin + T.floordiv(row_g, page_size)]
                                            page_offset: T.int32(is_size_var=True) = T.floormod(row_g, page_size)
                                            for vec in T.vectorized(VEC_SIZE):
                                                V_smem[tile_start_s + j, tx * VEC_SIZE + vec] = pages[page_no, 1, by, page_offset, tx * VEC_SIZE + vec]
                                        else:
                                            for vec in T.vectorized(VEC_SIZE):
                                                V_smem[tile_start_s + j, tx * VEC_SIZE + vec] = 0.0
                                    T.tvm_storage_sync("shared")
                                    # compute QK
                                    m_prev[0] = st_m[0]
                                    for j in T.serial(bdy * tile_size_per_bdx):
                                        if (iterator * bdz + tz) * bdy * tile_size_per_bdx + j >= kv_chunk_len[0]:
                                            S_local[j] = -5e4
                                        else:
                                            # load K from shared memory to local memory
                                            for vec in T.vectorized(VEC_SIZE):
                                                K_local[vec] = K_smem[tz * bdy * tile_size_per_bdx + j, tx * VEC_SIZE + vec]
                                            # compute S = Q * K * sm_scale
                                            S_local[j] = 0
                                            for vec in T.serial(VEC_SIZE):
                                                S_local[j] += Q_local[vec] * K_local[vec] * sm_scale
                                            # allreduce over bdx
                                            S_allreduce[tz, ty, tx] = S_local[j]
                                            T.tvm_storage_sync("shared")
                                            offset[0] = bdx // 2
                                            while offset[0] > 0:
                                                if tx < offset[0]:
                                                    S_allreduce[tz, ty, tx] += S_allreduce[tz, ty, tx + offset[0]]
                                                T.tvm_storage_sync("shared")
                                                offset[0] = offset[0] >> 1
                                            S_local[j] = S_allreduce[tz, ty, 0]
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
                                            O_local[vec] += V_local[vec] * S_local[j]

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
                                    output[batch_idx, by * GROUP_SIZE + ty, tx * VEC_SIZE + vec] = O_local[vec]
    # fmt: on
    # pylint: enable=line-too-long,invalid-name,too-many-arguments,too-many-branches
    return batch_decode_paged_kv


if __name__ == "__main__":
    cache = create_kv_cache()
    test_paged_attention_kv_cache_prefill_and_decode(cache)
    test_paged_attention_kv_cache_remove_sequence(cache)
    test_paged_attention_kv_cache_popn(cache)
