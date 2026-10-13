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
"""Round-trip tests for the ``vec_auto`` global ↔ shared path.

Pipeline: A_gmem --G2S--> A_smem --S2G--> B_gmem. If either direction is
wrong the round trip leaves B mismatched against A.
"""

import math

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.testing import env
from tvm.tirx.layout import ComposeLayout, S, TileLayout


def _build_kernel(scope, n_threads, shape, dtype):
    s_layout = TileLayout(S[shape])
    full_slices = tuple(slice(0, d) for d in shape)

    if scope == "warpgroup":

        @T.function
        def kernel(A: T.Tensor(shape, dtype), B: T.Tensor(shape, dtype)) -> None:
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(n_threads,)))
            _lane = T.cuda.lane_id()
            copy_thread_id = T.cuda.thread_idx("x")
            A_smem = T.alloc_tensor(
                ty_args=[T.Tensor(shape, dtype, scope="shared", layout=s_layout)]
            )
            copy_src_6 = T.meta_var(A[full_slices])
            copy_src_tensor_6 = T.meta_var(copy_src_6.source)
            copy_dst_6 = T.meta_var(A_smem[full_slices])
            copy_dst_tensor_6 = T.meta_var(copy_dst_6.source)
            for copy_step_6 in T.serial(
                T.ceildiv(math.prod([int(r.extent) for r in copy_src_6.region]), 128)
            ):
                copy_index_6 = copy_step_6 * (128) + (copy_thread_id % 128)
                if copy_index_6 < math.prod([int(r.extent) for r in copy_src_6.region]):
                    copy_value_6 = copy_src_tensor_6[
                        tuple(
                            [
                                copy_src_6.region[k].min
                                + copy_index_6
                                // math.prod([int(s.extent) for s in copy_src_6.region[k + 1 :]])
                                % copy_src_6.region[k].extent
                                for k in range(len(copy_src_6.region))
                            ]
                        )
                    ]
                    copy_dst_index_6 = T.meta_var(
                        tuple(
                            [
                                copy_dst_6.region[k].min
                                + copy_index_6
                                // math.prod([int(s.extent) for s in copy_dst_6.region[k + 1 :]])
                                % copy_dst_6.region[k].extent
                                for k in range(len(copy_dst_6.region))
                            ]
                        )
                    )
                    copy_dst_tensor_6[copy_dst_index_6] = copy_value_6
            T.cuda.cta_sync()
            copy_src_7 = T.meta_var(A_smem[full_slices])
            copy_src_tensor_7 = T.meta_var(copy_src_7.source)
            copy_dst_7 = T.meta_var(B[full_slices])
            copy_dst_tensor_7 = T.meta_var(copy_dst_7.source)
            for copy_step_7 in T.serial(
                T.ceildiv(math.prod([int(r.extent) for r in copy_src_7.region]), 128)
            ):
                copy_index_7 = copy_step_7 * (128) + (copy_thread_id % 128)
                if copy_index_7 < math.prod([int(r.extent) for r in copy_src_7.region]):
                    copy_value_7 = copy_src_tensor_7[
                        tuple(
                            [
                                copy_src_7.region[k].min
                                + copy_index_7
                                // math.prod([int(s.extent) for s in copy_src_7.region[k + 1 :]])
                                % copy_src_7.region[k].extent
                                for k in range(len(copy_src_7.region))
                            ]
                        )
                    ]
                    copy_dst_index_7 = T.meta_var(
                        tuple(
                            [
                                copy_dst_7.region[k].min
                                + copy_index_7
                                // math.prod([int(s.extent) for s in copy_dst_7.region[k + 1 :]])
                                % copy_dst_7.region[k].extent
                                for k in range(len(copy_dst_7.region))
                            ]
                        )
                    )
                    copy_dst_tensor_7[copy_dst_index_7] = copy_value_7

    elif scope == "warp":

        @T.function
        def kernel(A: T.Tensor(shape, dtype), B: T.Tensor(shape, dtype)) -> None:
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(n_threads,)))
            _lane = T.cuda.lane_id()
            copy_thread_id = T.cuda.thread_idx("x")
            A_smem = T.alloc_tensor(
                ty_args=[T.Tensor(shape, dtype, scope="shared", layout=s_layout)]
            )
            copy_src_8 = T.meta_var(A[full_slices])
            copy_src_tensor_8 = T.meta_var(copy_src_8.source)
            copy_dst_8 = T.meta_var(A_smem[full_slices])
            copy_dst_tensor_8 = T.meta_var(copy_dst_8.source)
            for copy_step_8 in T.serial(
                T.ceildiv(math.prod([int(r.extent) for r in copy_src_8.region]), 32)
            ):
                copy_index_8 = copy_step_8 * (32) + (copy_thread_id % 32)
                if copy_index_8 < math.prod([int(r.extent) for r in copy_src_8.region]):
                    copy_value_8 = copy_src_tensor_8[
                        tuple(
                            [
                                copy_src_8.region[k].min
                                + copy_index_8
                                // math.prod([int(s.extent) for s in copy_src_8.region[k + 1 :]])
                                % copy_src_8.region[k].extent
                                for k in range(len(copy_src_8.region))
                            ]
                        )
                    ]
                    copy_dst_index_8 = T.meta_var(
                        tuple(
                            [
                                copy_dst_8.region[k].min
                                + copy_index_8
                                // math.prod([int(s.extent) for s in copy_dst_8.region[k + 1 :]])
                                % copy_dst_8.region[k].extent
                                for k in range(len(copy_dst_8.region))
                            ]
                        )
                    )
                    copy_dst_tensor_8[copy_dst_index_8] = copy_value_8
            T.cuda.cta_sync()
            copy_src_9 = T.meta_var(A_smem[full_slices])
            copy_src_tensor_9 = T.meta_var(copy_src_9.source)
            copy_dst_9 = T.meta_var(B[full_slices])
            copy_dst_tensor_9 = T.meta_var(copy_dst_9.source)
            for copy_step_9 in T.serial(
                T.ceildiv(math.prod([int(r.extent) for r in copy_src_9.region]), 32)
            ):
                copy_index_9 = copy_step_9 * (32) + (copy_thread_id % 32)
                if copy_index_9 < math.prod([int(r.extent) for r in copy_src_9.region]):
                    copy_value_9 = copy_src_tensor_9[
                        tuple(
                            [
                                copy_src_9.region[k].min
                                + copy_index_9
                                // math.prod([int(s.extent) for s in copy_src_9.region[k + 1 :]])
                                % copy_src_9.region[k].extent
                                for k in range(len(copy_src_9.region))
                            ]
                        )
                    ]
                    copy_dst_index_9 = T.meta_var(
                        tuple(
                            [
                                copy_dst_9.region[k].min
                                + copy_index_9
                                // math.prod([int(s.extent) for s in copy_dst_9.region[k + 1 :]])
                                % copy_dst_9.region[k].extent
                                for k in range(len(copy_dst_9.region))
                            ]
                        )
                    )
                    copy_dst_tensor_9[copy_dst_index_9] = copy_value_9

    elif scope == "cta":

        @T.function
        def kernel(A: T.Tensor(shape, dtype), B: T.Tensor(shape, dtype)) -> None:
            T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(n_threads,)))
            _lane = T.cuda.lane_id()
            copy_thread_id = T.cuda.thread_idx("x")
            A_smem = T.alloc_tensor(
                ty_args=[T.Tensor(shape, dtype, scope="shared", layout=s_layout)]
            )
            copy_src_10 = T.meta_var(A[full_slices])
            copy_src_tensor_10 = T.meta_var(copy_src_10.source)
            copy_dst_10 = T.meta_var(A_smem[full_slices])
            copy_dst_tensor_10 = T.meta_var(copy_dst_10.source)
            for copy_step_10 in T.serial(
                T.ceildiv(math.prod([int(r.extent) for r in copy_src_10.region]), n_threads)
            ):
                copy_index_10 = copy_step_10 * (n_threads) + (copy_thread_id)
                if copy_index_10 < math.prod([int(r.extent) for r in copy_src_10.region]):
                    copy_value_10 = copy_src_tensor_10[
                        tuple(
                            [
                                copy_src_10.region[k].min
                                + copy_index_10
                                // math.prod([int(s.extent) for s in copy_src_10.region[k + 1 :]])
                                % copy_src_10.region[k].extent
                                for k in range(len(copy_src_10.region))
                            ]
                        )
                    ]
                    copy_dst_index_10 = T.meta_var(
                        tuple(
                            [
                                copy_dst_10.region[k].min
                                + copy_index_10
                                // math.prod([int(s.extent) for s in copy_dst_10.region[k + 1 :]])
                                % copy_dst_10.region[k].extent
                                for k in range(len(copy_dst_10.region))
                            ]
                        )
                    )
                    copy_dst_tensor_10[copy_dst_index_10] = copy_value_10
            T.cuda.cta_sync()
            copy_src_11 = T.meta_var(A_smem[full_slices])
            copy_src_tensor_11 = T.meta_var(copy_src_11.source)
            copy_dst_11 = T.meta_var(B[full_slices])
            copy_dst_tensor_11 = T.meta_var(copy_dst_11.source)
            for copy_step_11 in T.serial(
                T.ceildiv(math.prod([int(r.extent) for r in copy_src_11.region]), n_threads)
            ):
                copy_index_11 = copy_step_11 * (n_threads) + (copy_thread_id)
                if copy_index_11 < math.prod([int(r.extent) for r in copy_src_11.region]):
                    copy_value_11 = copy_src_tensor_11[
                        tuple(
                            [
                                copy_src_11.region[k].min
                                + copy_index_11
                                // math.prod([int(s.extent) for s in copy_src_11.region[k + 1 :]])
                                % copy_src_11.region[k].extent
                                for k in range(len(copy_src_11.region))
                            ]
                        )
                    ]
                    copy_dst_index_11 = T.meta_var(
                        tuple(
                            [
                                copy_dst_11.region[k].min
                                + copy_index_11
                                // math.prod([int(s.extent) for s in copy_dst_11.region[k + 1 :]])
                                % copy_dst_11.region[k].extent
                                for k in range(len(copy_dst_11.region))
                            ]
                        )
                    )
                    copy_dst_tensor_11[copy_dst_index_11] = copy_value_11
    else:
        raise ValueError(f"unsupported scope {scope!r}")

    return kernel


# (scope, n_threads, shape) — shape chosen so total / T / vec_len > 1 with at
# least one outer round, and total is divisible by T*vec_len.
TASKS = [
    ("warp", 32, (32, 32)),  # 1024 total, T=32, vec 8 → outer 4
    ("warp", 32, (32, 64)),  # 2048 total, outer 8
    ("warpgroup", 128, (128, 32)),  # 4096, outer 4
    ("warpgroup", 128, (128, 64)),  # 8192, outer 8
    ("warpgroup", 128, (256, 16)),  # 4096, outer 4
    ("cta", 256, (256, 32)),  # 8192, T=256, vec 8 → outer 4
    ("cta", 256, (512, 16)),  # 8192, outer 4
]


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize(
    "scope,n_threads,shape",
    [pytest.param(*t, id=f"{t[0]}-{t[1]}-{'x'.join(map(str, t[2]))}") for t in TASKS],
)
@pytest.mark.parametrize("dtype", ["float16", "float32", "uint8"])
def test_gmem_smem_roundtrip(scope, n_threads, shape, dtype):
    kernel = _build_kernel(scope, n_threads, shape, dtype)

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": kernel})
        compiled = tvm.tirx.compile(mod, target=target, pipeline="tirx")

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    A_np = tvm.testing.generate_random_array(dtype, shape)
    B_np = np.zeros(shape, dtype=np_dtype)

    def run_and_check():
        dev = tvm.cuda(0)
        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        compiled(A, B)
        np.testing.assert_array_equal(B.numpy(), A_np)

    tvm.testing.run_with_gpu_lock(run_and_check)


# ----------------------------------------------------------------------------
# Migrated from test_copy_sync.py: sync G↔S copy via the user-facing
# Tx.copy() (which dispatches to the vec_auto global ↔ shared path).
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task",
    [
        # A[0:128, 0:32] -> A_smem[0:128, 0:32] -> B[0:128, 0:32]
        (
            (128, 32),
            (128, 32),
            ((0, 128), (0, 32)),
            32,
            TileLayout(S[128, 32]),
            TileLayout(S[128, 32]),
            TileLayout(S[128, 32]),
        ),
        # A[32:64, 32:64] -> A_smem[0:32, 0:32] -> B[32:64, 32:64]
        (
            (64, 64),
            (32, 32),
            ((32, 64), (32, 64)),
            32,
            TileLayout(S[64, 64]),
            TileLayout(S[64, 64]),
            TileLayout(S[32, 32]),
        ),
        # A[0:1, 0:32, 0:32] -> A_smem[0:32, 0:32] -> B[0:1, 0:32, 0:32]
        (
            (4, 32, 32),
            (32, 32),
            ((0, 1), (0, 32), (0, 32)),
            32,
            TileLayout(S[4, 32, 32]),
            TileLayout(S[4, 32, 32]),
            TileLayout(S[32, 32]),
        ),
        # A[0:8, 0:8] -> A_smem[0:8, 0:8] -> B[0:8, 0:8]
        (
            (16, 16),
            (8, 8),
            ((0, 8), (0, 8)),
            32,
            TileLayout(S[16, 16]),
            TileLayout(S[16, 16]),
            TileLayout(S[8, 8]),
        ),
        # A[32:96, 256:512] -> A_smem[0:32, 0:256] -> B[32:96, 256:512] (swizzled)
        (
            (96, 512),
            (32, 256),
            ((16, 48), (256, 512)),
            32,
            TileLayout(S[96, 512]),
            TileLayout(S[96, 512]),
            ComposeLayout(3, 3, 3, TileLayout(S[8, 64]))
            .tile_to((16, 128), (8, 64))
            .tile_to((32, 256), (16, 128)),
        ),
    ],
)
@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize(
    "dtype", ["int8", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16", "float32"]
)
@pytest.mark.parametrize("scope", ["cta", "thread"])
def test_copy_g2s_s2g(task, dtype, scope):
    g_shape, s_shape, g_region, thread_cnt, layoutA, layoutB, layoutS = task

    r_smem = tuple(slice(None) for _ in range(len(s_shape)))
    r_gmem = tuple(slice(g_region[i][0], g_region[i][1]) for i in range(len(g_shape)))

    if scope == "thread":
        thread_cnt = 1

    @T.function
    def copy_sync(
        A: T.Tensor(g_shape, dtype, layout=layoutA), B: T.Tensor(g_shape, dtype, layout=layoutB)
    ) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(2,), block=(thread_cnt,)))
        copy_thread_id = T.cuda.thread_idx("x")

        A_smem = T.alloc_tensor(ty_args=[T.Tensor(s_shape, dtype, scope="shared", layout=layoutS)])
        # `scope` is parametrized at runtime; select the scope namespace
        # dynamically (T.cta / T.thread) instead of a literal prefix.
        copy_src_1 = T.meta_var(A[r_gmem])
        copy_src_tensor_1 = T.meta_var(copy_src_1.source)
        copy_dst_1 = T.meta_var(A_smem[r_smem])
        copy_dst_tensor_1 = T.meta_var(copy_dst_1.source)
        for copy_step_1 in T.serial(
            T.ceildiv(math.prod([int(r.extent) for r in copy_src_1.region]), thread_cnt)
        ):
            copy_index_1 = copy_step_1 * (thread_cnt) + (copy_thread_id)
            if copy_index_1 < math.prod([int(r.extent) for r in copy_src_1.region]):
                copy_value_1 = copy_src_tensor_1[
                    tuple(
                        [
                            copy_src_1.region[k].min
                            + copy_index_1
                            // math.prod([int(s.extent) for s in copy_src_1.region[k + 1 :]])
                            % copy_src_1.region[k].extent
                            for k in range(len(copy_src_1.region))
                        ]
                    )
                ]
                copy_dst_index_1 = T.meta_var(
                    tuple(
                        [
                            copy_dst_1.region[k].min
                            + copy_index_1
                            // math.prod([int(s.extent) for s in copy_dst_1.region[k + 1 :]])
                            % copy_dst_1.region[k].extent
                            for k in range(len(copy_dst_1.region))
                        ]
                    )
                )
                copy_dst_tensor_1[copy_dst_index_1] = copy_value_1
        T.cuda.cta_sync()
        copy_src_2 = T.meta_var(A_smem[r_smem])
        copy_src_tensor_2 = T.meta_var(copy_src_2.source)
        copy_dst_2 = T.meta_var(B[r_gmem])
        copy_dst_tensor_2 = T.meta_var(copy_dst_2.source)
        for copy_step_2 in T.serial(
            T.ceildiv(math.prod([int(r.extent) for r in copy_src_2.region]), thread_cnt)
        ):
            copy_index_2 = copy_step_2 * (thread_cnt) + (copy_thread_id)
            if copy_index_2 < math.prod([int(r.extent) for r in copy_src_2.region]):
                copy_value_2 = copy_src_tensor_2[
                    tuple(
                        [
                            copy_src_2.region[k].min
                            + copy_index_2
                            // math.prod([int(s.extent) for s in copy_src_2.region[k + 1 :]])
                            % copy_src_2.region[k].extent
                            for k in range(len(copy_src_2.region))
                        ]
                    )
                ]
                copy_dst_index_2 = T.meta_var(
                    tuple(
                        [
                            copy_dst_2.region[k].min
                            + copy_index_2
                            // math.prod([int(s.extent) for s in copy_dst_2.region[k + 1 :]])
                            % copy_dst_2.region[k].extent
                            for k in range(len(copy_dst_2.region))
                        ]
                    )
                )
                copy_dst_tensor_2[copy_dst_index_2] = copy_value_2

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_sync})
        mod = tvm.tirx.compile(mod, target=target, pipeline="tirx")

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        B_ref = B_np.copy()
        B_ref[r_gmem] = A_np[r_gmem]

        def run_and_check():
            dev = tvm.cuda(0)
            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            np.testing.assert_allclose(B_ref, B.numpy())

        tvm.testing.run_with_gpu_lock(run_and_check)


def test_swizzled_smem_emit_must_be_swizzle_aware():
    """Codegen-level: emitted S address should go through the swizzle's
    Apply so the XOR scrambling is honored. Currently emit uses
    ``s_buf.ptr_to([0,..,0]) + linear_offset`` which only matches a
    non-swizzled storage layout."""
    import tvm
    from tvm.script import tirx as T
    from tvm.tirx.layout import ComposeLayout, S, TileLayout

    shape = (128, 32)
    s_layout = ComposeLayout(3, 3, 3, TileLayout(S[shape]))

    @T.function
    def kernel(A: T.Tensor(shape, "float16")) -> None:
        T.device_entry(launch=T.cuda.LaunchConfig(grid=(1,), block=(128,)))
        _lane = T.cuda.lane_id()
        copy_thread_id = T.cuda.thread_idx("x")
        A_smem = T.alloc_tensor(
            ty_args=[T.Tensor(shape, "float16", scope="shared", layout=s_layout)]
        )
        copy_src_3 = T.meta_var(A[0:128, 0:32])
        copy_src_tensor_3 = T.meta_var(copy_src_3.source)
        copy_dst_3 = T.meta_var(A_smem[0:128, 0:32])
        copy_dst_tensor_3 = T.meta_var(copy_dst_3.source)
        for copy_step_3 in T.serial(
            T.ceildiv(math.prod([int(r.extent) for r in copy_src_3.region]), 128)
        ):
            copy_index_3 = copy_step_3 * (128) + (copy_thread_id % 128)
            if copy_index_3 < math.prod([int(r.extent) for r in copy_src_3.region]):
                copy_value_3 = copy_src_tensor_3[
                    tuple(
                        [
                            copy_src_3.region[k].min
                            + copy_index_3
                            // math.prod([int(s.extent) for s in copy_src_3.region[k + 1 :]])
                            % copy_src_3.region[k].extent
                            for k in range(len(copy_src_3.region))
                        ]
                    )
                ]
                copy_dst_index_3 = T.meta_var(
                    tuple(
                        [
                            copy_dst_3.region[k].min
                            + copy_index_3
                            // math.prod([int(s.extent) for s in copy_dst_3.region[k + 1 :]])
                            % copy_dst_3.region[k].extent
                            for k in range(len(copy_dst_3.region))
                        ]
                    )
                )
                copy_dst_tensor_3[copy_dst_index_3] = copy_value_3

    # NB: pin sm_90 explicitly — the default cuda target falls back to sm_50
    # when no GPU is detected, which nvcc 13+ rejects. Codegen happens before
    # nvcc; if the whole tvm.compile pipeline fails, we never see the source.
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    with target:
        mod = tvm.IRModule({"main": kernel})
        compiled = tvm.tirx.compile(mod, target=target, pipeline="tirx")
    src = "".join(im.inspect_source() for im in compiled.mod.imports)

    # If emit is swizzle-aware, two ways it shows up in the generated
    # CUDA:
    #   1. fallback path emits ``swizzle.apply(linear)``, which lowers
    #      to a ``^`` (XOR) somewhere in the S-offset computation
    #      (typically on a separate ``s_off_ptr[0] = ...`` line, not on
    #      the ``tvm_builtin_pointer_offset`` line itself).
    #   2. fast path computes the swizzled base once and XORs compile-time
    #      constants per iter — also containing a ``^`` somewhere in the
    #      S-offset computation.
    # XOR-less code means swizzle was silently dropped.
    assert "^" in src, "emitted s_ptr address shows no swizzle handling — no XOR anywhere"


# ----------------------------------------------------------------------------
# Structured ComposeLayout lowering test. The transformed S TileLayout is
# recomposed with the original swizzle and applied directly to (f, tid, 0).
# ----------------------------------------------------------------------------


if __name__ == "__main__":
    tvm.testing.main()
