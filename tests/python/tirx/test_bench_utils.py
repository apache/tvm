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
"""Tests for tvm.tirx.bench utilities."""

import pytest
import torch

pytest.importorskip("triton")  # tvm.tirx.bench imports triton.profiler

from tvm.testing import env
from tvm.tirx.bench import (
    _compute_group_count,
    _parse_proton_tree,
    bench,
    bench_tk,
    tensor_bytes,
)

# ── _parse_proton_tree ──────────────────────────────────────────────────────


SAMPLE_TREE = """\
├─ 1.500 tir
│  ├─ 1.500 my_kernel_fn
│  └─ 0.001 vectorized_elementwise_kernel
└─ 0.800 cublas
   └─ 0.800 sm90_xmma_gemm_f16f16
"""


def test_parse_proton_tree_basic():
    impls, errors = _parse_proton_tree(SAMPLE_TREE)
    assert impls == {"tir": 1.5, "cublas": 0.8}
    assert errors == {}


def test_parse_proton_tree_filters_elementwise():
    """vectorized_elementwise_kernel and elementwise_kernel_with_index are skipped."""
    tree = """\
├─ 0.500 tir
│  ├─ 0.500 real_kernel
│  └─ 0.001 elementwise_kernel_with_index
"""
    impls, _ = _parse_proton_tree(tree)
    assert impls == {"tir": 0.5}


def test_parse_proton_tree_slowest_child():
    """Takes the slowest depth-2 child per impl."""
    tree = """\
├─ 2.000 tir
│  ├─ 0.300 kernel_a
│  └─ 0.700 kernel_b
"""
    impls, _ = _parse_proton_tree(tree)
    assert impls == {"tir": 0.7}


def test_parse_proton_tree_baseline_errors():
    tree = """\
BASELINE_ERROR: cublas: CUDA OOM
├─ 1.000 tir
│  └─ 1.000 my_kernel
"""
    impls, errors = _parse_proton_tree(tree)
    assert impls == {"tir": 1.0}
    assert errors == {"cublas": "CUDA OOM"}


def test_parse_proton_tree_ansi_stripped():
    """ANSI color codes are stripped before parsing."""
    tree = "\x1b[32m├─ 1.000 tir\x1b[0m\n│  └─ 1.000 k\n"
    impls, _ = _parse_proton_tree(tree)
    assert impls == {"tir": 1.0}


def test_parse_proton_tree_empty():
    impls, errors = _parse_proton_tree("")
    assert impls == {}
    assert errors == {}


# ── bench_tk (ThunderKittens group-input protocol) ──────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_tk_basic():
    """bench_tk returns positive times for each impl."""
    M, N = 256, 256

    funcs = {"matmul": lambda case: torch.mm(case[0], case[1])}

    def make_input():
        A = torch.randn(M, N, device="cuda", dtype=torch.float16)
        B = torch.randn(M, N, device="cuda", dtype=torch.float16)
        return (A, B), tensor_bytes(A, B)

    results = bench_tk(funcs, make_input, warmup=5, repeat=10, cooldown_s=0.0, timer="event")
    assert "matmul" in results["impls"]
    assert results["impls"]["matmul"] > 0


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_tk_multiple_impls():
    """Multiple impls each get their own timing."""
    M, N = 128, 128
    funcs = {
        "mm": lambda case: torch.mm(case[0], case[1]),
        "addmm": lambda case: torch.addmm(
            torch.zeros(M, N, device="cuda", dtype=torch.float16), case[0], case[1]
        ),
    }

    def make_input():
        A = torch.randn(M, N, device="cuda", dtype=torch.float16)
        B = torch.randn(M, N, device="cuda", dtype=torch.float16)
        return (A, B), tensor_bytes(A, B)

    results = bench_tk(funcs, make_input, warmup=5, repeat=10, cooldown_s=0.0, timer="event")
    assert set(results["impls"].keys()) == {"mm", "addmm"}
    assert all(v > 0 for v in results["impls"].values())


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_tk_multiple_input_groups():
    """Multiple input groups cycle correctly (L2 eviction)."""
    M, N = 128, 128
    call_count = [0]

    def make_input():
        call_count[0] += 1
        A = torch.randn(M, N, device="cuda", dtype=torch.float16)
        B = torch.randn(M, N, device="cuda", dtype=torch.float16)
        return (A, B), tensor_bytes(A, B)

    funcs = {"mm": lambda case: torch.mm(case[0], case[1])}
    results = bench_tk(
        funcs, make_input, warmup=5, repeat=20, cooldown_s=0.0, timer="event", l2_bytes=64 * 1024
    )
    assert results["impls"]["mm"] > 0
    assert call_count[0] > 1


# ── bench (Triton-standard, pure-launch) ─────────────────────────────────────


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_event_pure_launch():
    """New Triton-standard bench(): no-arg launch closures, event timer."""
    M, N = 256, 256
    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = torch.randn(M, N, device="cuda", dtype=torch.float16)

    funcs = {"mm": lambda: torch.mm(A, B)}
    results = bench(funcs, warmup=5, repeat=10, timer="event")
    assert "mm" in results["impls"]
    assert results["impls"]["mm"] > 0
    assert results["timer"] == "event"


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_default_timer_is_proton():
    """Omitting timer resolves to the proton default (recorded as timer='proton').

    Under pytest _do_bench_proton falls back to event timing, but bench() still
    records the resolved timer name, so this pins the timer=None -> 'proton' default.
    """
    M, N = 256, 256
    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = torch.randn(M, N, device="cuda", dtype=torch.float16)

    funcs = {"mm": lambda: torch.mm(A, B)}
    results = bench(funcs, warmup=5, repeat=10)
    assert results["timer"] == "proton"
    assert results["impls"]["mm"] > 0


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_cudagraph_proton_pure_launch():
    """New Triton-standard bench(): cudagraph_proton timer.

    Under pytest, _do_bench_cudagraph_proton falls back to event-based cudagraph
    timing (no CUPTI/Proton required in CI), so this exercises the mode wiring and
    the fallback path rather than the real Proton attribution.
    """
    M, N = 256, 256
    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = torch.randn(M, N, device="cuda", dtype=torch.float16)

    funcs = {"mm": lambda: torch.mm(A, B)}
    results = bench(funcs, warmup=5, repeat=10, timer="cudagraph_proton")
    assert results["impls"]["mm"] > 0
    assert results["timer"] == "cudagraph_proton"


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_references_pure_launch():
    """New bench(): reference builders return no-arg callables and get timed."""
    M, N = 128, 128
    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = torch.randn(M, N, device="cuda", dtype=torch.float16)

    funcs = {"tir": lambda: torch.mm(A, B)}

    def _addmm():
        C = torch.zeros(M, N, device="cuda", dtype=torch.float16)
        return lambda: torch.addmm(C, A, B)

    results = bench(funcs, warmup=5, repeat=10, timer="event", references={"addmm": _addmm})
    assert set(results["impls"].keys()) == {"tir", "addmm"}
    assert all(v > 0 for v in results["impls"].values())


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_rejects_unknown_timer():
    """New bench() only accepts event / proton / cudagraph_proton (plain cudagraph
    was removed)."""
    A = torch.randn(8, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError):
        bench({"mm": lambda: torch.mm(A, A)}, timer="cudagraph")


# ── _compute_group_count ───────────────────────────────────────────────────


def test_compute_groups_small_tensors():
    """Small tensors need many groups to fill 3x L2."""
    # 128x128 fp16 = 32KB.  3*128MB / 32KB = 12288, +1 = 12289
    input_bytes = tensor_bytes(torch.empty(128, 128, dtype=torch.float16))
    n = _compute_group_count(input_bytes, l2_bytes=128 * 1024 * 1024)
    assert n == 12289


def test_compute_groups_large_tensors():
    """Inputs >= 3x L2 need only 1 group."""
    # 16384x16384 fp32 = 1GB >> 3*128MB = 384MB
    input_bytes = tensor_bytes(torch.empty(16384, 16384, dtype=torch.float32))
    n = _compute_group_count(input_bytes, l2_bytes=128 * 1024 * 1024)
    assert n == 1


def test_compute_groups_moderate_tensors():
    """Moderate tensors: floor(3*L2 / input) + 1."""
    # 8192x8192 bf16 = 128MB.  floor(384M / 128M) + 1 = 4
    input_bytes = tensor_bytes(torch.empty(8192, 8192, dtype=torch.bfloat16))
    n = _compute_group_count(input_bytes, l2_bytes=128 * 1024 * 1024)
    assert n == 4


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_tk_legacy_callable_api():
    """bench_tk still accepts the existing single-callable API used by TIRx tests."""
    M, N = 128, 128
    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = torch.randn(M, N, device="cuda", dtype=torch.float16)

    result = bench_tk(
        lambda: torch.mm(A, B), warmup=1, repeat=2, proton_name="legacy", flush_l2_size=1
    )
    assert result > 0


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_tk_callable_inputs():
    """bench_tk accepts a factory callable and auto-computes groups."""
    M, N = 256, 256

    call_count = [0]

    def make_input():
        call_count[0] += 1
        case = (
            torch.randn(M, N, device="cuda", dtype=torch.float16),
            torch.randn(M, N, device="cuda", dtype=torch.float16),
        )
        return case, tensor_bytes(*case)

    funcs = {"mm": lambda case: torch.mm(case[0], case[1])}
    results = bench_tk(funcs, make_input, warmup=5, repeat=10, cooldown_s=0.0, timer="event")
    assert "mm" in results["impls"]
    assert results["impls"]["mm"] > 0
    assert call_count[0] >= 2  # at least 2 groups created


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
