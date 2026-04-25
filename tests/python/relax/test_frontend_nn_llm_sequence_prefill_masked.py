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
"""Focused correctness tests for ``_attention_sequence_prefill_with_mask``.

The masked variant supports two regimes selected by ``mask_mode``:

* ``"padded"`` — encoder-style right-padded bidirectional attention.
* ``"causal_padded_left"`` — decoder-embedding-style left-padded causal
  attention. Real tokens occupy ``[seq_len - valid_len, seq_len)`` and
  the causal constraint keeps ``col <= row`` within the valid range.

In both regimes each sample in a padded batch carries its own
``valid_len`` and the kernel applies the mask inside the QKV load path
and the online softmax update. These tests cover the four shape / mask
regimes that can break each kernel independently of any scheduler
tuning:

* ``valid_len == 0``       — entire batch row is padding
* ``valid_len == seq_len`` — full-length row, must match the unmasked kernel
* mixed ``valid_lens``     — typical padded batch
* grouped-query attention  — ``h_q > h_kv`` with ``group_size > 1``

The references are float32 NumPy implementations of masked softmax
attention restricted to the valid prefix/suffix, so the kernel is only
compared on the unpadded positions (padded positions are intentionally
free to contain arbitrary garbage).
"""
# ruff: noqa: E501
import math

import numpy as np

import tvm
import tvm.testing
from tvm.relax.frontend.nn.llm.kv_cache import _attention_sequence_prefill_with_mask


def _reference_masked_attention(q, k, v, valid_lens, sm_scale):
    """Right-pad bidirectional reference. Only the first ``valid_lens[b]`` rows are written."""
    batch, seq_q, h_q, d = q.shape
    _, seq_kv, h_kv, _ = k.shape
    group_size = h_q // h_kv
    out = np.zeros_like(q, dtype=np.float32)
    q32 = q.astype(np.float32)
    k32 = k.astype(np.float32)
    v32 = v.astype(np.float32)
    for b in range(batch):
        L = int(valid_lens[b])
        if L == 0:
            continue
        for h in range(h_q):
            hk = h // group_size
            qh = q32[b, :L, h, :]  # [L, d]
            kh = k32[b, :L, hk, :]  # [L, d]
            vh = v32[b, :L, hk, :]  # [L, d]
            s = (qh @ kh.T) * sm_scale  # [L, L]
            m = s.max(axis=-1, keepdims=True)
            e = np.exp(s - m)
            p = e / e.sum(axis=-1, keepdims=True)
            out[b, :L, h, :] = p @ vh
    return out


def _reference_masked_attention_causal_padded_left(q, k, v, valid_lens, sm_scale):
    """Left-pad causal reference.

    Real tokens occupy ``[seq_q - valid_len, seq_q)`` for queries and
    ``[seq_kv - valid_len, seq_kv)`` for keys/values. Only the valid query
    suffix rows are written; padded rows stay zeroed.
    """
    batch, seq_q, h_q, d = q.shape
    _, seq_kv, h_kv, _ = k.shape
    group_size = h_q // h_kv
    out = np.zeros_like(q, dtype=np.float32)
    q32 = q.astype(np.float32)
    k32 = k.astype(np.float32)
    v32 = v.astype(np.float32)
    for b in range(batch):
        L = int(valid_lens[b])
        if L == 0:
            continue
        pad_q = seq_q - L
        pad_kv = seq_kv - L
        for h in range(h_q):
            hk = h // group_size
            qh = q32[b, pad_q:, h, :]  # [L, d]
            kh = k32[b, pad_kv:, hk, :]  # [L, d]
            vh = v32[b, pad_kv:, hk, :]  # [L, d]
            s = (qh @ kh.T) * sm_scale  # [L, L]
            # Causal on the LxL valid block: mask upper triangle to -inf.
            s = s + np.triu(np.full((L, L), -np.inf), k=1)
            m = s.max(axis=-1, keepdims=True)
            e = np.exp(s - m)
            p = e / e.sum(axis=-1, keepdims=True)
            out[b, pad_q:, h, :] = p @ vh
    return out


def _build_masked_prefill(h_kv, h_q, d, dtype, target, mask_mode="padded"):
    sm_scale = 1.0 / math.sqrt(d)
    tir_func = _attention_sequence_prefill_with_mask(
        h_kv=h_kv,
        h_q=h_q,
        d=d,
        dtype=dtype,
        target=target,
        sm_scale=sm_scale,
        mask_mode=mask_mode,
    )
    mod = tvm.IRModule({"main": tir_func})
    return tvm.tirx.build(mod["main"], target=target), sm_scale


def _run_case(
    *,
    target,
    dev,
    h_kv,
    h_q,
    d,
    batch,
    seq,
    valid_lens,
    seq_kv=None,
    dtype="float16",
    seed=0,
    mask_mode="padded",
):
    target = tvm.target.Target(target)
    built, sm_scale = _build_masked_prefill(h_kv, h_q, d, dtype, target, mask_mode=mask_mode)

    if seq_kv is None:
        seq_kv = seq
    np_dtype = {"float16": np.float16, "float32": np.float32}[dtype]
    rng = np.random.default_rng(seed)
    q_np = (rng.standard_normal((batch, seq, h_q, d)) * 0.1).astype(np_dtype)
    k_np = (rng.standard_normal((batch, seq_kv, h_kv, d)) * 0.1).astype(np_dtype)
    v_np = (rng.standard_normal((batch, seq_kv, h_kv, d)) * 0.1).astype(np_dtype)
    valid_np = np.asarray(valid_lens, dtype=np.int32)
    out_np = np.zeros((batch, seq, h_q, d), dtype=np_dtype)
    lse_np = np.zeros((batch, seq, h_q), dtype=np_dtype)

    q_nd = tvm.runtime.tensor(q_np, device=dev)
    k_nd = tvm.runtime.tensor(k_np, device=dev)
    v_nd = tvm.runtime.tensor(v_np, device=dev)
    valid_nd = tvm.runtime.tensor(valid_np, device=dev)
    out_nd = tvm.runtime.tensor(out_np, device=dev)
    lse_nd = tvm.runtime.tensor(lse_np, device=dev)

    built.main(q_nd, k_nd, v_nd, valid_nd, out_nd, lse_nd)

    got = out_nd.numpy().astype(np.float32)
    if mask_mode == "padded":
        ref = _reference_masked_attention(q_np, k_np, v_np, valid_np, sm_scale)
    else:
        ref = _reference_masked_attention_causal_padded_left(q_np, k_np, v_np, valid_np, sm_scale)

    # Only compare valid rows. Padding rows are undefined by design.
    rtol, atol = (2e-2, 2e-2) if dtype == "float16" else (1e-4, 1e-4)
    for b in range(batch):
        L = int(valid_np[b])
        if L == 0:
            continue
        if mask_mode == "padded":
            np.testing.assert_allclose(got[b, :L], ref[b, :L], rtol=rtol, atol=atol)
        else:
            pad_q = seq - L
            np.testing.assert_allclose(got[b, pad_q:], ref[b, pad_q:], rtol=rtol, atol=atol)


@tvm.testing.requires_gpu
@tvm.testing.parametrize_targets("cuda", "metal")
def test_valid_len_zero(target, dev):
    """All samples are fully padded: kernel must not crash and must stay bounded."""
    _run_case(
        target=target,
        dev=dev,
        h_kv=4,
        h_q=4,
        d=64,
        batch=2,
        seq=16,
        valid_lens=[0, 0],
    )


@tvm.testing.requires_gpu
@tvm.testing.parametrize_targets("cuda", "metal")
def test_valid_len_full(target, dev):
    """All samples are fully valid: must match a plain unmasked attention."""
    _run_case(
        target=target,
        dev=dev,
        h_kv=4,
        h_q=4,
        d=64,
        batch=2,
        seq=32,
        valid_lens=[32, 32],
    )


@tvm.testing.requires_gpu
@tvm.testing.parametrize_targets("cuda", "metal")
def test_valid_len_mixed(target, dev):
    """Typical encoder batch with different valid lengths per sample."""
    _run_case(
        target=target,
        dev=dev,
        h_kv=4,
        h_q=4,
        d=64,
        batch=4,
        seq=64,
        valid_lens=[10, 64, 5, 33],
    )


@tvm.testing.requires_gpu
@tvm.testing.parametrize_targets("cuda", "metal")
def test_valid_len_mixed_gqa(target, dev):
    """Grouped-query attention: ``group_size = h_q / h_kv > 1``."""
    _run_case(
        target=target,
        dev=dev,
        h_kv=2,
        h_q=4,
        d=64,
        batch=3,
        seq=32,
        valid_lens=[8, 32, 17],
    )


@tvm.testing.requires_gpu
@tvm.testing.parametrize_targets("cuda", "metal")
def test_causal_padded_left_valid_len_zero(target, dev):
    """Causal left-pad: all samples are fully padded."""
    _run_case(
        target=target,
        dev=dev,
        h_kv=4,
        h_q=4,
        d=64,
        batch=2,
        seq=16,
        valid_lens=[0, 0],
        mask_mode="causal_padded_left",
    )


@tvm.testing.requires_gpu
@tvm.testing.parametrize_targets("cuda", "metal")
def test_causal_padded_left_valid_len_full(target, dev):
    """Causal left-pad: all samples are fully valid — degenerates to plain causal attention."""
    _run_case(
        target=target,
        dev=dev,
        h_kv=4,
        h_q=4,
        d=64,
        batch=2,
        seq=32,
        valid_lens=[32, 32],
        mask_mode="causal_padded_left",
    )


@tvm.testing.requires_gpu
@tvm.testing.parametrize_targets("cuda", "metal")
def test_causal_padded_left_valid_len_mixed(target, dev):
    """Causal left-pad: typical decoder-embedding batch with mixed lengths."""
    _run_case(
        target=target,
        dev=dev,
        h_kv=4,
        h_q=4,
        d=64,
        batch=4,
        seq=64,
        valid_lens=[10, 64, 5, 33],
        mask_mode="causal_padded_left",
    )


@tvm.testing.requires_gpu
@tvm.testing.parametrize_targets("cuda", "metal")
def test_causal_padded_left_valid_len_mixed_gqa(target, dev):
    """Causal left-pad: grouped-query attention with mixed lengths."""
    _run_case(
        target=target,
        dev=dev,
        h_kv=2,
        h_q=4,
        d=64,
        batch=3,
        seq=32,
        valid_lens=[8, 32, 17],
        mask_mode="causal_padded_left",
    )


@tvm.testing.requires_gpu
@tvm.testing.parametrize_targets("cuda", "metal")
def test_causal_padded_left_qo_len_differs_from_kv_len(target, dev):
    """Causal left-pad: Q and K/V may have different padded lengths."""
    _run_case(
        target=target,
        dev=dev,
        h_kv=2,
        h_q=4,
        d=64,
        batch=3,
        seq=32,
        seq_kv=48,
        valid_lens=[8, 32, 17],
        mask_mode="causal_padded_left",
    )


if __name__ == "__main__":
    tvm.testing.main()
