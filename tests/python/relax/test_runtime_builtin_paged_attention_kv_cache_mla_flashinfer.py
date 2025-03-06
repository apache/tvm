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
import torch

import tvm
import tvm.testing
from tvm import dlight as dl
from tvm import relax
from tvm.contrib import utils
from tvm.relax.frontend.nn.llm.kv_cache import (
    AttnKind,
    RopeMode,
    _copy_single_page_mla,
    _kv_cache_debug_get_kv_mla,
    _kv_cache_transpose_append_mla,
    _merge_state_inplace,
)
from tvm.runtime import ShapeTuple

np.random.seed(0)

reserved_nseq = 32
maximum_total_seq_length = 2048
prefill_chunk_size = 512
page_size = 16
num_layers = 4
num_attention_heads = 128
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = qk_nope_head_dim
sm_scale = (qk_nope_head_dim + qk_rope_head_dim) ** (-0.5)
kv_lora_rank = 512
dtype = "float16"
dtype_torch = getattr(torch, dtype)
device = tvm.cuda()
device_torch = torch.device("cuda")

fclear = None
fadd_sequence = None
fremove_sequence = None
ffork_sequence = None
fpopn = None
fbegin_forward = None
fend_forward = None
fself_attn = None
fcross_attn = None
fappend_mla_kv = None
fkv_merge_attn_output = None
fis_empty = None
fdebug_get_kv = None

ftranspose_append = None
fcopy_cache = None
fmla_prefill = None
fmla_prefill_plan = None
fattn_prefill_ragged = None
fattn_prefill_ragged_plan = None
fmerge_state = None
fmerge_state_additional = None
fcopy_single_page = None

w_kv = None
w_uk = None
w_uv = None


# Register a dumb function for testing purpose.
@tvm.register_func("test.dumb_function", override=True)
def _dumb_function():
    raise RuntimeError("Dumb function isn't supposed to be accessed.")


def set_global_func(dtype):
    global fclear, fadd_sequence, fremove_sequence, ffork_sequence
    global fpopn, fbegin_forward, fend_forward
    global fself_attn, fcross_attn, fappend_mla_kv, fkv_merge_attn_output
    global fis_empty, fdebug_get_kv
    global ftranspose_append, fcopy_cache, fmla_prefill, fmla_prefill_plan
    global fattn_prefill_ragged, fattn_prefill_ragged_plan
    global fmerge_state, fmerge_state_additional, fcopy_single_page
    global w_kv, w_uk, w_uv

    fclear = tvm.get_global_func("vm.builtin.kv_state_clear")
    fadd_sequence = tvm.get_global_func("vm.builtin.kv_state_add_sequence")
    fremove_sequence = tvm.get_global_func("vm.builtin.kv_state_remove_sequence")
    ffork_sequence = tvm.get_global_func("vm.builtin.kv_state_fork_sequence")
    fpopn = tvm.get_global_func("vm.builtin.kv_state_popn")
    fbegin_forward = tvm.get_global_func("vm.builtin.kv_state_begin_forward")
    fend_forward = tvm.get_global_func("vm.builtin.kv_state_end_forward")
    fself_attn = tvm.get_global_func("vm.builtin.attention_kv_cache_self_attention")
    fcross_attn = tvm.get_global_func("vm.builtin.attention_kv_cache_cross_attention")
    fappend_mla_kv = tvm.get_global_func("vm.builtin.attention_kv_cache_append_mla_kv")
    fkv_merge_attn_output = tvm.get_global_func(
        "vm.builtin.attention_kv_cache_merge_attn_output_inplace"
    )
    fis_empty = tvm.get_global_func("vm.builtin.attention_kv_cache_empty")
    fdebug_get_kv = tvm.get_global_func("vm.builtin.attention_kv_cache_debug_get_kv_mla")

    def load_module(name: str, static_modules: List[tvm.runtime.Module]):
        assert len(static_modules) > 0
        if len(static_modules) == 1:
            return static_modules[0]
        static_mod = static_modules[0]
        for mod in static_modules[1:]:
            static_mod.import_module(mod)
        temp = utils.tempdir()
        mod_path = temp.relpath(f"{name}.so")
        static_mod.export_library(mod_path)
        return tvm.runtime.load_module(mod_path)

    target = tvm.target.Target.from_device(device)
    flashinfer_prefill_mod = load_module(
        "flashinfer_prefill",
        relax.backend.cuda.flashinfer.gen_flashinfer_prefill_module(
            dtype_q=dtype,
            dtype_kv=dtype,
            dtype_o=dtype,
            qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
            v_head_dim=v_head_dim,
            target=target,
            enable_inline_rope=False,
        ),
    )
    flashinfer_mla_mod = load_module(
        "flashinfer_mla",
        relax.backend.cuda.flashinfer.gen_flashinfer_mla_module(
            dtype_q=dtype,
            dtype_kv=dtype,
            dtype_o=dtype,
            head_dim_ckv=kv_lora_rank,
            head_dim_kpe=qk_rope_head_dim,
            target=target,
        ),
    )

    fattn_prefill_ragged = flashinfer_prefill_mod["batch_prefill_with_ragged_kv_cache_run"]
    fattn_prefill_ragged_plan = flashinfer_prefill_mod["batch_prefill_with_kv_cache_plan"]
    fmla_prefill = flashinfer_mla_mod["batch_mla_paged_attention_run"]
    fmla_prefill_plan = flashinfer_mla_mod["batch_mla_paged_attention_plan"]

    builts = []
    for tir_func in [
        _kv_cache_transpose_append_mla(kv_lora_rank + qk_rope_head_dim, dtype),
        _kv_cache_debug_get_kv_mla(num_layers, kv_lora_rank + qk_rope_head_dim, dtype),
        _merge_state_inplace(num_attention_heads, kv_lora_rank, dtype, target),
        _merge_state_inplace(num_attention_heads, v_head_dim, dtype, target),
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
        fmerge_state,
        fmerge_state_additional,
        fcopy_single_page,
    ) = builts

    w_kv = torch.empty(
        (kv_lora_rank, num_attention_heads * (qk_nope_head_dim + v_head_dim)),
        device=device_torch,
        dtype=dtype_torch,
    )
    w_kv.uniform_(-0.1, 0.1)
    w_uk, w_uv = torch.split(
        w_kv.view(kv_lora_rank, num_attention_heads, qk_nope_head_dim + v_head_dim),
        [qk_nope_head_dim, v_head_dim],
        dim=2,
    )
    w_uk = w_uk.permute(1, 2, 0)
    w_uv = w_uv.permute(1, 0, 2)


def create_kv_cache(dtype):
    fcreate = tvm.get_global_func("vm.builtin.paged_attention_kv_cache_create")
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
        1,  # num_kv_heads
        kv_lora_rank + qk_rope_head_dim,
        kv_lora_rank,
        tvm.runtime.ShapeTuple([int(AttnKind.MLA) for _ in range(num_layers)]),
        False,  # enable_kv_transfer
        RopeMode.NONE,
        1,
        10000,
        None,  # rope_ext_factors
        tvm.nd.empty((), dtype, device=device),
        None,  # f_transpose_append_mha
        ftranspose_append,
        ["flashinfer", fattn_prefill_ragged, fattn_prefill_ragged_plan],  # fattn_prefill_ragged
        [],  # fattn_prefill
        [],  # fattn_decode
        [],  # fattn_prefill_sliding_window
        [],  # fattn_decode_sliding_window
        [],  # fattn_prefill_with_tree_mask_paged_kv_cache
        [],  # fattn_prefill_with_tree_mask
        ["flashinfer", fmla_prefill, fmla_prefill_plan],
        [fmerge_state, fmerge_state_additional],
        fdumb,  # fsplit_rotary
        fcopy_single_page,
        fcopy_cache,
        fdumb,  # fcompact_copy
    )
    return cache


@pytest.fixture(params=itertools.product(["float16"]))
def kv_cache_and_config(request):
    global dtype, dtype_torch
    (dtype,) = request.param
    dtype_torch = getattr(torch, dtype)
    set_global_func(dtype)
    return (create_kv_cache(dtype),)


def verify_cached_kv(kv_cache, seq_ids, expected_kv):
    for seq_id in seq_ids:
        kv_expected = expected_kv[seq_id]
        seq_length = expected_kv[seq_id].shape[1]
        kv_actual = tvm.nd.empty(kv_expected.shape, dtype=dtype, device=device)
        fdebug_get_kv(kv_cache, seq_id, 0, seq_length, kv_actual)
        torch.testing.assert_close(
            torch.from_numpy(kv_actual.numpy()).to(device_torch), kv_expected, rtol=1e-3, atol=1e-3
        )


def apply_attention(
    kv_cache,
    batch: List[Tuple[Union[int, Tuple[int, int, int]], int]],
    cached_kv: Dict[int, torch.Tensor],
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
            cached_kv[seq_id] = torch.zeros(
                (num_layers, 0, kv_lora_rank + qk_rope_head_dim),
                dtype=dtype_torch,
                device=device_torch,
            )

    fbegin_forward(kv_cache, ShapeTuple(seq_ids), ShapeTuple(append_lengths), None)

    global_new_q = torch.zeros(
        (num_layers, 0, num_attention_heads, qk_nope_head_dim + qk_rope_head_dim),
        dtype=dtype_torch,
        device=device_torch,
    )
    global_new_kv = torch.zeros(
        (num_layers, 0, kv_lora_rank + qk_rope_head_dim),
        dtype=dtype_torch,
        device=device_torch,
    )

    q_array = []
    is_decode_request = True
    all_new_sequences = True
    for i, (seq_id, append_length) in enumerate(batch):
        new_q_np = np.random.uniform(
            -0.1,
            0.1,
            size=(
                num_layers,
                append_length,
                num_attention_heads,
                qk_nope_head_dim + qk_rope_head_dim,
            ),
        ).astype(dtype)
        new_kv_np = np.random.uniform(
            -0.1, 0.1, size=(num_layers, append_length, kv_lora_rank + qk_rope_head_dim)
        ).astype(dtype)
        q_array.append(new_q_np)

        # Convert the numpy arrays to torch tensors on device.
        new_q_tensor = torch.from_numpy(new_q_np).to(device_torch)
        new_kv_tensor = torch.from_numpy(new_kv_np).to(device_torch)

        all_new_sequences = all_new_sequences and cached_kv[seq_id].shape[1] == 0
        cached_kv[seq_id] = torch.cat([cached_kv[seq_id], new_kv_tensor], dim=1)
        global_new_q = torch.cat([global_new_q, new_q_tensor], dim=1)
        global_new_kv = torch.cat([global_new_kv, new_kv_tensor], dim=1)

        if append_length > 1:
            is_decode_request = False

    for layer_id in range(num_layers):
        queries = tvm.nd.array(global_new_q[layer_id].cpu().numpy(), device)
        key_value = tvm.nd.array(global_new_kv[layer_id].cpu().numpy(), device)
        total_seq_length = global_new_q[layer_id].shape[0]
        outputs1 = tvm.nd.empty(
            (total_seq_length, num_attention_heads, v_head_dim), dtype, device=device
        )
        lse1 = tvm.nd.empty((total_seq_length, num_attention_heads), "float32", device=device)
        outputs2 = tvm.nd.empty(
            (total_seq_length, num_attention_heads, kv_lora_rank), dtype, device=device
        )
        lse2 = tvm.nd.empty((total_seq_length, num_attention_heads), "float32", device=device)

        fappend_mla_kv(kv_cache, layer_id, key_value)
        if not is_decode_request:
            # Part 1. self-attention
            latent, k_pe = torch.split(
                global_new_kv[layer_id], [kv_lora_rank, qk_rope_head_dim], dim=1
            )
            keys, values = torch.split(
                (latent @ w_kv).to(dtype_torch).reshape(total_seq_length, num_attention_heads, -1),
                [qk_nope_head_dim, v_head_dim],
                dim=2,
            )
            k_pe_expanded = torch.unsqueeze(k_pe, 1).expand(
                total_seq_length, num_attention_heads, qk_rope_head_dim
            )
            keys = torch.cat([keys, k_pe_expanded], dim=2)
            keys_tvm = tvm.nd.array(keys.cpu().numpy(), device)
            values_tvm = tvm.nd.array(values.cpu().numpy(), device)
            fself_attn(kv_cache, layer_id, sm_scale, queries, keys_tvm, values_tvm, outputs1, lse1)

        if not all_new_sequences or is_decode_request:
            # Part 2. cross-attention
            queries_lora_np, q_pe = torch.split(
                global_new_q[layer_id], [qk_nope_head_dim, qk_rope_head_dim], dim=2
            )
            queries_lora_np = torch.cat(
                [torch.bmm(queries_lora_np.permute(1, 0, 2), w_uk).permute(1, 0, 2), q_pe], dim=2
            )
            queries_lora = tvm.nd.array(queries_lora_np.cpu().numpy(), device)
            fcross_attn(kv_cache, layer_id, sm_scale, queries_lora, outputs2, lse2)
            cross_attn_output = tvm.nd.array(
                torch.bmm(
                    torch.from_numpy(outputs2.numpy()).to(device_torch).permute(1, 0, 2), w_uv
                )
                .permute(1, 0, 2)
                .cpu()
                .numpy(),
                device,
            )

        if not is_decode_request:
            if not all_new_sequences:
                fkv_merge_attn_output(kv_cache, outputs1, lse1, cross_attn_output, lse2)
        else:
            outputs1 = cross_attn_output

        # Compute attention expected results.
        outputs = torch.unsqueeze(torch.tensor(outputs1.numpy()).to(device_torch), 0)
        sum_length = 0
        for i, (seq_id, append_length) in enumerate(batch):
            assert cached_kv[seq_id].shape[1] >= append_length

            q_seq = torch.from_numpy(q_array[i][layer_id]).to(device_torch).permute(1, 0, 2)
            latent_seq, k_pe_seq = torch.split(
                torch.unsqueeze(cached_kv[seq_id][layer_id], 1),
                [kv_lora_rank, qk_rope_head_dim],
                dim=2,
            )
            k_seq, v_seq = torch.split(
                (latent_seq @ w_kv).reshape(k_pe_seq.shape[0], num_attention_heads, -1),
                [qk_nope_head_dim, v_head_dim],
                dim=2,
            )
            k_pe_seq = k_pe_seq.expand(k_pe_seq.shape[0], num_attention_heads, qk_rope_head_dim)
            k_seq = torch.cat([k_seq, k_pe_seq], dim=2).permute(1, 2, 0)
            v_seq = v_seq.permute(1, 0, 2)

            softmax_input = (q_seq.to(torch.float32) @ k_seq.to(torch.float32)) / torch.sqrt(
                torch.tensor(qk_nope_head_dim + qk_rope_head_dim, dtype=torch.float32)
            )
            softmax_shape = softmax_input.shape
            assert softmax_shape[-2] == append_length
            length_diff = softmax_shape[-1] - softmax_shape[-2]
            assert length_diff >= 0
            # Create a mask similar to np.tril and np.triu.
            mask = torch.tril(
                torch.full_like(softmax_input, float(np.finfo(np.float32).max)),
                diagonal=length_diff,
            ) + torch.triu(
                torch.full_like(softmax_input, float(np.finfo(np.float32).min)),
                diagonal=length_diff + 1,
            )
            softmax_input = torch.minimum(softmax_input, mask)

            results = torch.nn.functional.softmax(softmax_input, dim=-1) @ v_seq.to(torch.float32)
            results = results.permute(1, 0, 2).unsqueeze(0).to(dtype_torch)

            torch.testing.assert_close(
                outputs[:, sum_length : sum_length + append_length, ...],
                results,
                rtol=1e-3,
                atol=1e-3,
            )
            sum_length += append_length
    fend_forward(kv_cache)

    # Verify
    verify_cached_kv(kv_cache, seq_ids, cached_kv)


@pytest.mark.skip(reason="Require FlashInfer enabled")
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


@pytest.mark.skip(reason="Require FlashInfer enabled")
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


@pytest.mark.skip(reason="Require FlashInfer enabled")
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


@pytest.mark.skip(reason="Require FlashInfer enabled")
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
    set_global_func(dtype)
    cache = create_kv_cache(dtype)
    cache_and_config = (cache,)
    test_paged_attention_kv_cache_prefill_and_decode(cache_and_config)
    test_paged_attention_kv_cache_remove_sequence(cache_and_config)
    test_paged_attention_kv_cache_fork_sequence(cache_and_config)
    test_paged_attention_kv_cache_popn(cache_and_config)
