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
"""Metal prefill kernels split each softmax row across SIMD lanes."""

import math

import numpy as np
import pytest
import tvm_ffi

import tvm
import tvm_tirx
import tvm_tirx.testing
from tvm.relax.frontend.nn.llm._prefill_kernels import _attention_prefill, _attention_prefill_ragged
from tvm.testing import device_enabled

METAL = {
    "kind": "metal",
    "max_shared_memory_per_block": 32768,
    "max_threads_per_block": 1024,
    "thread_warp_size": 32,
}


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_simd_softmax_only_on_metal(dtype):
    for kind, uses_shuffle in (
        (METAL, True),
        ({"kind": "cuda", "arch": "sm_80"}, False),
        ("vulkan", False),
    ):
        target = tvm_tirx.target.Target(kind)
        for func in (
            _attention_prefill(2, 8, 64, dtype, False, {}, target),
            _attention_prefill_ragged(2, 8, 64, 64, dtype, {}, target),
        ):
            assert ("gpu_warp_shuffle" in func.script()) == uses_shuffle, (kind, dtype)


def _reference(q, k, v, q_lens, kv_lens, causal, scale, group):
    """Base-2 attention over per-sequence (q, k, v) slices, as the kernels compute it."""
    out = np.zeros((q.shape[0], q.shape[1], v.shape[2]))
    lse = np.zeros(q.shape[:2])
    q_off = kv_off = 0
    for n, m in zip(q_lens, kv_lens):
        for head in range(q.shape[1]):
            scores = (
                q[q_off : q_off + n, head].astype("float64")
                @ k[kv_off : kv_off + m, head // group].astype("float64").T
            )
            scores *= scale / np.log(2)
            if causal:
                # Query row r sees keys up to r + m - n, the last n keys being its own tokens.
                scores[np.arange(m)[None, :] > np.arange(n)[:, None] + m - n] = -np.inf
            peak = scores.max(axis=1, keepdims=True)
            w = np.exp2(scores - peak)
            out[q_off : q_off + n, head] = (
                w @ v[kv_off : kv_off + m, head // group] / w.sum(axis=1, keepdims=True)
            )
            lse[q_off : q_off + n, head] = peak[:, 0] + np.log2(w.sum(axis=1))
        q_off += n
        kv_off += m
    return out, lse


@pytest.mark.gpu
@pytest.mark.parametrize("kernel", ["ragged", "paged"])
@pytest.mark.parametrize("causal", [0, 1])
@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_prefill_runtime_metal(kernel, causal, dtype):
    if not device_enabled("metal"):
        pytest.skip("metal not enabled")
    device = tvm_ffi.device("metal")
    target = tvm_tirx.target.Target.from_device(device)
    h_kv, h_q, d, page_size = 2, 8, 64, 16
    # Several query and KV tiles, a partial last tile, and a one-token sequence.
    q_lens = [1, 37, 129]
    kv_lens = q_lens if kernel == "ragged" else [40, 37, 200]
    rng = np.random.default_rng(3)
    q = rng.normal(0, 0.5, (sum(q_lens), h_q, d)).astype(dtype)
    k = rng.normal(0, 0.5, (sum(kv_lens), h_kv, d)).astype(dtype)
    v = rng.normal(size=(sum(kv_lens), h_kv, d)).astype(dtype)
    scale = d**-0.5
    ref, ref_lse = _reference(q, k, v, q_lens, kv_lens, causal, scale, h_q // h_kv)
    q_indptr = np.array([0, *np.cumsum(q_lens)], dtype="int32")
    zeros = np.zeros(len(q_lens), dtype="int32")
    if kernel == "ragged":
        func = _attention_prefill_ragged(h_kv, h_q, d, d, dtype, {}, target)
        inputs = [q, q_indptr, k, v, q_indptr, np.arange(q_indptr[-1], dtype="int32"), zeros]
    else:
        func = _attention_prefill(h_kv, h_q, d, dtype, False, {}, target, page_size=page_size)
        num_pages = [math.ceil(m / page_size) for m in kv_lens]
        pages = np.zeros((sum(num_pages), 2, h_kv, page_size, d), dtype=dtype)
        page, kv_off = 0, 0
        for m, p in zip(kv_lens, num_pages):
            for t in range(m):
                pages[page + t // page_size, 0, :, t % page_size] = k[kv_off + t]
                pages[page + t // page_size, 1, :, t % page_size] = v[kv_off + t]
            page, kv_off = page + p, kv_off + m
        last_page_len = np.array(
            [m - (p - 1) * page_size for m, p in zip(kv_lens, num_pages)], dtype="int32"
        )
        page_indptr = np.array([0, *np.cumsum(num_pages)], dtype="int32")
        inputs = [
            q,
            q_indptr,
            pages,
            page_indptr,
            np.arange(page_indptr[-1], dtype="int32"),
            last_page_len,
            zeros,
            np.zeros(q_indptr[-1], dtype="int32"),
        ]
    module = tvm.compile(func.with_attr("global_symbol", "main"), target=target)

    def run_and_check():
        args = [tvm_tirx.runtime.tensor(a, device) for a in inputs]
        output = tvm_tirx.runtime.tensor(np.full(ref.shape, np.nan, dtype=dtype), device)
        lse = tvm_tirx.runtime.tensor(np.full(ref_lse.shape, np.nan, dtype="float32"), device)
        module(*args, output, lse, causal, 0, 1.0, 10000.0, scale)
        rtol, atol = (2e-3, 2e-3) if dtype == "float16" else (2e-4, 2e-5)
        np.testing.assert_allclose(output.numpy(), ref, rtol=rtol, atol=atol)
        np.testing.assert_allclose(lse.numpy(), ref_lse, rtol=0, atol=1e-4)

    tvm_tirx.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm_tirx.testing.main()
