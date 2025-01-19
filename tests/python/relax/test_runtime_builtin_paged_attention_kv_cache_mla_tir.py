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
import itertools
from typing import Dict, List, Tuple, Union

import numpy as np
import pytest
import scipy.special

import tvm
import tvm.testing
from tvm import dlight as dl
from tvm.relax.frontend.nn.llm.kv_cache import (
    AttnKind,
    RopeMode,
    _attention_decode_mla,
    _attention_prefill_mla,
    _attention_prefill_ragged_mla_absorbed,
    _copy_single_page_mla,
    _kv_cache_debug_get_kv_mla,
    _kv_cache_transpose_append_mla,
    _merge_state_inplace,
)
from tvm.runtime import ShapeTuple

reserved_nseq = 32
maximum_total_seq_length = 2048
prefill_chunk_size = 512
page_size = 16
num_layers = 4
num_attention_heads = 128
qk_nope_head_dim = 128
qk_rope_head_dim = 64
kv_lora_rank = 512
dtype = None
device = tvm.cuda()

fclear = None
fadd_sequence = None
fremove_sequence = None
ffork_sequence = None
fpopn = None
fbegin_forward = None
fend_forward = None
fmla_absorbed = None
fis_empty = None
fdebug_get_kv = None

ftranspose_append = None
fcopy_cache = None
fattn_prefill = None
fattn_decode = None
fattn_prefill_ragged_absorbed = None
fmerge_state = None
fcopy_single_page = None


# Register a dumb function for testing purpose.
@tvm.register_func("test.dumb_function")
def _dumb_function():
    pass


def set_global_func(dtype):
    global fclear, fadd_sequence, fremove_sequence, ffork_sequence
    global fpopn, fbegin_forward, fend_forward
    global fmla_absorbed, fis_empty, fdebug_get_kv
    global ftranspose_append, fcopy_cache, fattn_prefill, fattn_decode
    global fattn_prefill_ragged_absorbed
    global fmerge_state, fcopy_single_page

    fclear = tvm.get_global_func("vm.builtin.kv_state_clear")
    fadd_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    fremove_sequence = tvm.get_global_func("vm.builtin.kv_state_remove_sequence")
    ffork_sequence = tvm.get_global_func("vm.builtin.kv_state_fork_sequence")
    fpopn = tvm.get_global_func("vm.builtin.kv_state_popn")
    fbegin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    fend_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    fmla_absorbed = tvm.get_global_func("vm.builtin.attention_kv_cache_mla_absorbed")
    fis_empty = tvm.get_global_func("vm.builtin.attention_kv_cache_empty")
    fdebug_get_kv = tvm.get_global_func("vm.builtin.attention_kv_cache_debug_get_kv_mla")

    target = tvm.target.Target.from_device(device)
    builts = []
    for tir_func in [
        _kv_cache_transpose_append_mla(kv_lora_rank, qk_rope_head_dim, dtype),
        _kv_cache_debug_get_kv_mla(num_layers, kv_lora_rank + qk_rope_head_dim, dtype),
        _attention_prefill_mla(
            num_attention_heads, kv_lora_rank, qk_rope_head_dim, dtype, False, target
        ),
        _attention_decode_mla(
            num_attention_heads, kv_lora_rank, qk_rope_head_dim, dtype, False, target
        ),
        _attention_prefill_ragged_mla_absorbed(
            num_attention_heads, kv_lora_rank, qk_rope_head_dim, dtype, target
        ),
        _merge_state_inplace(num_attention_heads, kv_lora_rank, dtype, target),
        _copy_single_page_mla(page_size, kv_lora_rank + qk_rope_head_dim, dtype, target),
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
        fattn_prefill_ragged_absorbed,
        fmerge_state,
        fcopy_single_page,
    ) = builts


def create_kv_cache(dtype):
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create_reduced_mla")
    fdumb = tvm.get_global_func("test.dumb_function")
    cache = fcreate(
        tvm.runtime.ShapeTuple(
            [
                reserved_nseq,
                maximum_total_seq_length,
                prefill_chunk_size,
                page_size,
                0,
            ]
        ),
        tvm.runtime.ShapeTuple([0, num_layers]),
        num_attention_heads,
        1,
        kv_lora_rank + qk_rope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        tvm.runtime.ShapeTuple([int(AttnKind.MLA) for _ in range(num_layers)]),
        RopeMode.NONE,
        1,
        10000,
        tvm.nd.empty((), dtype, device=device),
        fdumb,
        ftranspose_append,
        fdumb,
        fdumb,
        fdumb,
        fdumb,
        fdumb,
        0,
        0,
        0,
        0,
        0,
        0,
        fattn_prefill,
        fattn_decode,
        fdumb,
        fattn_prefill_ragged_absorbed,
        fmerge_state,
        fdumb,
        fcopy_single_page,
        fcopy_cache,
        fdumb,
        fdumb,
        fdumb,
        None,
        False,
    )
    return cache


@pytest.fixture(params=itertools.product(["float16"]))
def kv_cache_and_config(request):
    global dtype
    (dtype,) = request.param
    set_global_func(dtype)
    return (create_kv_cache(dtype),)


def verify_cached_kv(kv_cache, seq_ids, expected_kv):
    for seq_id in seq_ids:
        kv_expected = expected_kv[seq_id]
        seq_length = expected_kv[seq_id].shape[1]
        kv_actual = tvm.nd.empty(kv_expected.shape, dtype=dtype, device=device)
        fdebug_get_kv(kv_cache, seq_id, 0, seq_length, kv_actual)
        tvm.testing.assert_allclose(kv_actual.numpy(), kv_expected, rtol=1e-3, atol=1e-3)


def apply_attention(
    kv_cache,
    batch: List[Tuple[Union[int, Tuple[int, int, int]], int]],
    cached_kv: Dict[int, np.ndarray],
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
            assert fork_parent_id in cached_kv
            assert seq_id not in cached_kv
            ffork_sequence(kv_cache, fork_parent_id, seq_id, fork_pos)
            if fork_pos == -1:
                cached_kv[seq_id] = cached_kv[fork_parent_id]
            else:
                cached_kv[seq_id] = cached_kv[fork_parent_id][::, :fork_pos]
        elif seq_id not in cached_kv:
            fadd_sequence(kv_cache, seq_id)
            cached_kv[seq_id] = np.zeros((num_layers, 0, kv_lora_rank + qk_rope_head_dim), dtype)

    fbegin_forward(kv_cache, ShapeTuple(seq_ids), ShapeTuple(append_lengths), None)

    global_new_q = np.zeros(
        (num_layers, 0, num_attention_heads, kv_lora_rank + qk_rope_head_dim), dtype
    )
    global_new_kv = np.zeros((num_layers, 0, kv_lora_rank + qk_rope_head_dim), dtype)

    q_array = []
    for i, (seq_id, append_length) in enumerate(batch):
        new_q = np.random.rand(
            num_layers, append_length, num_attention_heads, kv_lora_rank + qk_rope_head_dim
        ).astype(dtype)
        new_kv = np.random.rand(num_layers, append_length, kv_lora_rank + qk_rope_head_dim).astype(
            dtype
        )
        q_array.append(new_q)

        cached_kv[seq_id] = np.concatenate([cached_kv[seq_id], new_kv], axis=1)
        global_new_q = np.concatenate([global_new_q, new_q], axis=1)
        global_new_kv = np.concatenate([global_new_kv, new_kv], axis=1)

    for layer_id in range(num_layers):
        queries_np = global_new_q[layer_id]
        queries = tvm.nd.array(queries_np, device)
        compressed_kv = tvm.nd.array(global_new_kv[layer_id][:, :kv_lora_rank], device)
        k_pe = tvm.nd.array(global_new_kv[layer_id][:, kv_lora_rank:], device)
        outputs = tvm.nd.empty(
            (queries_np.shape[0], queries_np.shape[1], kv_lora_rank), dtype, device=device
        )
        fmla_absorbed(kv_cache, layer_id, 1.0, queries, compressed_kv, k_pe, outputs)

        # Compute attention expected results.
        outputs = np.expand_dims(outputs.numpy(), axis=0)
        sum_length = 0
        for i, (seq_id, append_length) in enumerate(batch):
            assert cached_kv[seq_id].shape[1] >= append_length

            q_seq = q_array[i][layer_id].transpose(1, 0, 2)
            k_seq = np.expand_dims(cached_kv[seq_id][layer_id], axis=1).transpose(1, 2, 0)
            v_seq = np.expand_dims(cached_kv[seq_id][layer_id], axis=1).transpose(1, 0, 2)[
                :, :, :kv_lora_rank
            ]

            k_seq = np.repeat(k_seq, num_attention_heads, axis=0)
            v_seq = np.repeat(v_seq, num_attention_heads, axis=0)
            softmax_input = q_seq.astype("float32") @ k_seq.astype("float32")
            softmax_shape = softmax_input.shape
            assert softmax_shape[-2] == append_length
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
    verify_cached_kv(kv_cache, seq_ids, cached_kv)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_paged_attention_kv_cache_prefill_and_decode(kv_cache_and_config):
    (kv_cache,) = kv_cache_and_config
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

    cached_kv = {}
    for batch in operation_seq:
        apply_attention(kv_cache, batch, cached_kv)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_paged_attention_kv_cache_remove_sequence(kv_cache_and_config):
    (kv_cache,) = kv_cache_and_config
    fclear(kv_cache)

    num_sequences = 5
    batch = [(seq_id, 1) for seq_id in range(num_sequences)]
    cached_kv = {}
    for seq_id_to_remove in range(num_sequences):
        apply_attention(kv_cache, batch, cached_kv)
        # Remove sequence.
        fremove_sequence(kv_cache, seq_id_to_remove)
        cached_kv.pop(seq_id_to_remove)
        verify_cached_kv(
            kv_cache,
            seq_ids=[seq_id for seq_id in range(num_sequences) if seq_id != seq_id_to_remove],
            expected_kv=cached_kv,
        )


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_paged_attention_kv_cache_fork_sequence(kv_cache_and_config):
    (kv_cache,) = kv_cache_and_config
    fclear(kv_cache)

    cached_kv = {}
    batch = [(0, 60), (1, 88), (2, 17), (3, 4)]
    apply_attention(kv_cache, batch, cached_kv)
    # Fork existing sequences.
    apply_attention(kv_cache, [((4, 3, -1), 35)], cached_kv)
    apply_attention(kv_cache, [((5, 0, -1), 20)], cached_kv)
    apply_attention(kv_cache, [((6, 5, -1), 102)], cached_kv)
    apply_attention(kv_cache, [((7, 0, -1), 3)], cached_kv)
    apply_attention(kv_cache, [((8, 5, -1), 71), ((9, 5, -1), 20)], cached_kv)
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
        apply_attention(kv_cache, batch, cached_kv)

    apply_attention(kv_cache, [((10, 1, 33), 11)], cached_kv)
    apply_attention(kv_cache, [((11, 0, 60), 45), ((12, 0, 15), 14)], cached_kv)
    apply_attention(kv_cache, [((13, 0, 16), 19), ((14, 0, 17), 19)], cached_kv)
    apply_attention(kv_cache, [((15, 5, 60), 8), ((16, 5, 80), 10)], cached_kv)
    apply_attention(
        kv_cache,
        [((17, 5, 75), 11), ((18, 5, 76), 45), ((19, 5, 77), 14)],
        cached_kv,
    )

    operation_seq = [
        [(6, 1), (11, 1), (13, 1), (9, 1)],
        [(10, 1), (16, 1), (18, 1), (19, 1)],
        [(8, 1), (15, 1), (17, 1), (12, 1), (14, 1)],
        [(10, 10), (6, 2), (8, 3), (19, 4)],
    ]
    for batch in operation_seq:
        apply_attention(kv_cache, batch, cached_kv)

    num_sequence = 20
    for i in range(num_sequence):
        fremove_sequence(kv_cache, i)
        cached_kv.pop(i)
        verify_cached_kv(
            kv_cache,
            seq_ids=list(range(i + 1, num_sequence)),
            expected_kv=cached_kv,
        )

    assert fis_empty(kv_cache), "The KV cache is not empty after removing all sequences"

    # Test fork after page recycle
    apply_attention(kv_cache, [(0, 7), (1, 24)], cached_kv)
    apply_attention(kv_cache, [((2, 1, -1), 10)], cached_kv)
    apply_attention(kv_cache, [((3, 0, -1), 20)], cached_kv)
    apply_attention(kv_cache, [(2, 1), (3, 1)], cached_kv)

    apply_attention(kv_cache, [(10, 7), (11, 24)], cached_kv)
    apply_attention(kv_cache, [((12, 11, -1), 200)], cached_kv)
    apply_attention(kv_cache, [(10, 1), (12, 1)], cached_kv)


@tvm.testing.requires_gpu
@tvm.testing.requires_cuda
def test_paged_attention_kv_cache_popn(kv_cache_and_config):
    (kv_cache,) = kv_cache_and_config
    fclear(kv_cache)

    cached_kv = {}
    batch = [(0, 35), (1, 88), (2, 17), (3, 4)]
    apply_attention(kv_cache, batch, cached_kv)
    apply_attention(kv_cache, [((4, 3, -1), 35)], cached_kv)

    popn_operations = [(0, 17), (1, 57), (2, 16), (3, 0), (4, 37)]
    for seq_id, pop_length in popn_operations:
        fpopn(kv_cache, seq_id, pop_length)
        if pop_length != 0:
            cached_kv[seq_id] = cached_kv[seq_id][:, :-pop_length, ...]
        verify_cached_kv(kv_cache, seq_ids=list(range(4)), expected_kv=cached_kv)

    num_sequence = 5
    for seq_id in range(num_sequence):
        fremove_sequence(kv_cache, seq_id)
        verify_cached_kv(
            kv_cache,
            seq_ids=list(range(seq_id + 1, num_sequence)),
            expected_kv=cached_kv,
        )

    assert fis_empty(kv_cache), "The KV cache is not empty after removing all sequences"


if __name__ == "__main__":
    DTYPES = ["float16"]
    for (dtype,) in itertools.product(DTYPES):
        set_global_func(dtype)
        cache = create_kv_cache(dtype)
        cache_and_config = (cache,)
        test_paged_attention_kv_cache_prefill_and_decode(cache_and_config)
        test_paged_attention_kv_cache_remove_sequence(cache_and_config)
        test_paged_attention_kv_cache_fork_sequence(cache_and_config)
        test_paged_attention_kv_cache_popn(cache_and_config)
