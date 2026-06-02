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
# pylint: disable=missing-function-docstring
"""Tests for the priority=0 ``copy/fallback`` dispatch — scalar single-thread
emit picked when every higher-priority variant rejects.

The cases here are *intentionally* shaped so ``gmem_smem`` rejects (region
element count doesn't divide ``thread_cnt``) and ``reg`` / ``ld_stmatrix`` /
... don't apply (scope pair mismatch). The dispatcher should land on
fallback, the emit should pick one active thread, and the round-trip
``A_gmem → A_smem → B_gmem`` should match.
"""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as Tx
from tvm.tirx.layout import S, TileLayout

# Force the fallback dispatch to register before any test compiles a kernel.
# Without this import, in fresh pytest workers the `copy/fallback` variant
# isn't yet registered when the dispatcher snapshots its registry.
from tvm.tirx.operator.tile_primitive.cuda.copy import fallback as _fallback_module  # noqa: F401


def _round_trip_shapes_and_threads():
    """Cases where ``gmem_smem`` rejects on ``n_elements % thread_cnt``.

    Per task: ``(scope, n_threads, shape, why_fallback)``. ``shape`` is small
    enough that scalar emit is fine, and chosen so the higher-priority
    variants can't accept it (size doesn't divide thread_cnt).
    """
    return [
        # warp scope, 32 threads, 24 elements (4x6) → 24 % 32 != 0.
        ("warp", 32, (4, 6), "4*6=24 ∤ 32"),
        # warp scope, 32 threads, 8 elements (1x8) → 8 % 32 != 0.
        ("warp", 32, (1, 8), "1*8=8 ∤ 32"),
        # warpgroup scope, 128 threads, 24 elements (4x6) → 24 % 128 != 0.
        ("warpgroup", 128, (4, 6), "4*6=24 ∤ 128"),
        # warpgroup scope, 128 threads, 32 elements (4x8) → 32 % 128 != 0.
        ("warpgroup", 128, (4, 8), "4*8=32 ∤ 128"),
        # cta scope, 256 threads, 32 elements (4x8) → 32 % 256 != 0.
        ("cta", 256, (4, 8), "4*8=32 ∤ 256"),
        # cta scope, 1024 threads, 64 elements (8x8) → 64 % 1024 != 0.
        # Mimics the test_partial_reduction sparse-write-back pattern.
        ("cta", 1024, (8, 8), "8*8=64 ∤ 1024"),
    ]


def _build_round_trip_kernel(scope, n_threads, shape, dtype):
    """``Tx.copy(A_smem, A); Tx.copy(B, A_smem)`` at the given scope. Both
    copies hit the same predicates; both should fall to fallback."""
    s_layout = TileLayout(S[shape])
    full = tuple(slice(0, d) for d in shape)

    # Each scope variant inserts an explicit ``cta_sync`` between the two
    # copies — fallback's emit no longer sneaks one in, so the writer/reader
    # pair on ``A_smem`` would otherwise race.
    if scope == "warp":

        @Tx.prim_func
        def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, shape, dtype)
            B = Tx.match_buffer(B_ptr, shape, dtype)
            Tx.device_entry()
            Tx.cta_id([1])
            Tx.lane_id([32])
            Tx.thread_id([n_threads])
            with Tx.cta():
                A_smem = Tx.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
                with Tx.warp():
                    Tx.copy(A_smem[full], A[full])
                    Tx.cuda.cta_sync()
                    Tx.copy(B[full], A_smem[full])

    elif scope == "warpgroup":

        @Tx.prim_func
        def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, shape, dtype)
            B = Tx.match_buffer(B_ptr, shape, dtype)
            Tx.device_entry()
            Tx.cta_id([1])
            Tx.warpgroup_id([n_threads // 128])
            Tx.warp_id_in_wg([4])
            Tx.lane_id([32])
            Tx.thread_id_in_wg([128])
            Tx.thread_id([n_threads])
            with Tx.cta():
                A_smem = Tx.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
                with Tx.warpgroup():
                    Tx.copy(A_smem[full], A[full])
                    Tx.cuda.cta_sync()
                    Tx.copy(B[full], A_smem[full])

    elif scope == "cta":

        @Tx.prim_func
        def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
            A = Tx.match_buffer(A_ptr, shape, dtype)
            B = Tx.match_buffer(B_ptr, shape, dtype)
            Tx.device_entry()
            Tx.cta_id([1])
            Tx.warp_id([n_threads // 32])
            Tx.lane_id([32])
            Tx.thread_id([n_threads])
            with Tx.cta():
                A_smem = Tx.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
                Tx.copy(A_smem[full], A[full])
                Tx.cuda.cta_sync()
                Tx.copy(B[full], A_smem[full])

    else:
        raise ValueError(f"unsupported scope {scope!r}")

    return kernel


@pytest.mark.parametrize(
    "scope,n_threads,shape,why",
    [
        pytest.param(s, n, sh, w, id=f"{s}-{n}-{'x'.join(map(str, sh))}")
        for s, n, sh, w in _round_trip_shapes_and_threads()
    ],
)
def test_fallback_round_trip(scope, n_threads, shape, why):
    """End-to-end: compile + run + compare. Failure means either the
    dispatcher didn't pick fallback (silent crash earlier) or fallback's
    emit is wrong (mismatch on B vs A)."""
    del why
    dtype = "float32"
    kernel = _build_round_trip_kernel(scope, n_threads, shape, dtype)

    dev = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    with target, pytest.warns(UserWarning, match="copy/fallback"):
        mod = tvm.IRModule({"main": kernel})
        compiled = tvm.compile(mod, target=target, tir_pipeline="tirx")

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    A_np = tvm.testing.generate_random_array(dtype, shape)
    B_np = np.zeros(shape, dtype=np_dtype)
    A = tvm.runtime.tensor(A_np, dev)
    B = tvm.runtime.tensor(B_np, dev)
    compiled(A, B)
    np.testing.assert_array_equal(B.numpy(), A_np)


def test_fallback_thread_scope():
    """``Tx.thread()`` — single thread, no gate. Either ``gmem_smem`` picks
    it up (n_elements % 1 == 0) or ``fallback`` does — both end up emitting
    a sensible single-thread copy. We only check the round trip is correct,
    not which variant fired."""
    shape = (4, 6)
    dtype = "float32"
    s_layout = TileLayout(S[shape])
    full = tuple(slice(0, d) for d in shape)

    @Tx.prim_func
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, shape, dtype)
        B = Tx.match_buffer(B_ptr, shape, dtype)
        Tx.device_entry()
        Tx.cta_id([1])
        Tx.thread_id([1])
        with Tx.thread():
            A_smem = Tx.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
            Tx.copy(A_smem[full], A[full])
            Tx.cuda.cta_sync()
            Tx.copy(B[full], A_smem[full])

    dev = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        compiled = tvm.compile(mod, target=target, tir_pipeline="tirx")

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    A_np = tvm.testing.generate_random_array(dtype, shape)
    B_np = np.zeros(shape, dtype=np_dtype)
    A = tvm.runtime.tensor(A_np, dev)
    B = tvm.runtime.tensor(B_np, dev)
    compiled(A, B)
    np.testing.assert_array_equal(B.numpy(), A_np)


def test_fallback_emits_gate():
    """Compiled CUDA source must contain a single-thread gate so only one
    active thread executes the scalar copy (not all of them, which would
    work but be racy + wasteful and indicate gate elision)."""
    shape = (4, 6)
    dtype = "float32"
    s_layout = TileLayout(S[shape])
    full = tuple(slice(0, d) for d in shape)

    @Tx.prim_func
    def kernel(A_ptr: Tx.handle, B_ptr: Tx.handle) -> None:
        A = Tx.match_buffer(A_ptr, shape, dtype)
        B = Tx.match_buffer(B_ptr, shape, dtype)
        Tx.device_entry()
        Tx.cta_id([1])
        Tx.warp_id([8])  # 256 threads => 8 warps
        Tx.lane_id([32])
        Tx.thread_id([256])
        with Tx.cta():
            A_smem = Tx.alloc_buffer(shape, dtype, scope="shared", layout=s_layout)
            Tx.copy(A_smem[full], A[full])
            Tx.copy(B[full], A_smem[full])

    target = tvm.target.Target("cuda")
    with target, pytest.warns(UserWarning, match="copy/fallback"):
        mod = tvm.IRModule({"main": kernel})
        compiled = tvm.compile(mod, target=target, tir_pipeline="tirx")

    src = "".join(im.inspect_source() for im in compiled.mod.imports)
    # The gate compiles to something like ``if (((int)threadIdx.x) == 0)``.
    # We don't pin the exact spelling; just require an equality predicate
    # against threadIdx.x somewhere in the source.
    assert "threadIdx.x" in src
    # At least one ``== 0`` (or ``== <literal>``) comparison must exist for
    # the single-thread gate.
    assert "== 0" in src, "fallback emit didn't produce a tid==0 gate; src:\n" + src[:2000]


if __name__ == "__main__":
    tvm.testing.main()
