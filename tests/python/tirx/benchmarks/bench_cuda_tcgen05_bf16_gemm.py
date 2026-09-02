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
"""End-to-end BF16 tcgen05 GEMM benchmark for Blackwell CUDA GPUs.

The kernel computes C = A @ B.T.  One warp produces K tiles with TMA while
another warp consumes them with tcgen05 MMA through a staged shared-memory
ring.  Unlike ``bench_cuda_tcgen05_bf16.py``, every MMA reads a different K
tile and the timing includes global-memory input loads and output stores.
For stable Jetson measurements, select MAXN mode and lock clocks before
running the benchmark.
"""

import argparse
import json

import ml_dtypes
import numpy as np

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.tile_primitive.tma_utils import SwizzleMode, mma_shared_layout
from tvm.tirx.layout import S, TCol, TileLayout, TLane
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg


def build_kernel(m_dim: int, n_dim: int, k_dim: int, stages: int, group_m: int, block_n: int):
    """Build a role-specialized, staged BF16 tcgen05 GEMM kernel."""
    block_m = 128
    block_k = 64
    if m_dim % block_m:
        raise ValueError(f"M must be a multiple of {block_m}, but got {m_dim}")
    if block_n not in (64, 128, 256) or n_dim % block_n:
        raise ValueError(f"N must be divisible by block_n in {{64, 128, 256}}, got {n_dim=}")
    if k_dim % block_k:
        raise ValueError(f"K must be a multiple of {block_k}, but got {k_dim}")
    if stages < 1:
        raise ValueError(f"stages must be positive, but got {stages}")
    if group_m < 1:
        raise ValueError(f"group_m must be positive, but got {group_m}")

    grid_m = m_dim // block_m
    grid_n = n_dim // block_n
    k_tiles = k_dim // block_k
    dtype = "bfloat16"
    a_base = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (block_m, block_k))
    b_base = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (block_n, block_k))
    a_layout = a_base.tile_to((stages, block_m, block_k), (1, block_m, block_k))
    b_layout = b_base.tile_to((stages, block_n, block_k), (1, block_n, block_k))
    tile_bytes = (block_m * block_k + block_n * block_k) * 2
    output_chunk = min(block_n, 128)
    output_chunks = block_n // output_chunk

    @T.prim_func
    def gemm(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (m_dim, k_dim), dtype)
        B = T.match_buffer(B_ptr, (n_dim, k_dim), dtype)
        C = T.match_buffer(C_ptr, (m_dim, n_dim), "float32")
        T.device_entry()

        # Group nearby M tiles so multiple CTAs can reuse the same B tile in L2.
        block = T.cta_id([grid_n * grid_m])
        group_width: T.let = group_m * grid_n
        group: T.let = block // group_width
        first_m: T.let = group * group_m
        actual_group_m: T.let = T.min(group_m, grid_m - first_m)
        block_m_idx: T.let = first_m + block % group_width % actual_group_m
        block_n_idx: T.let = block % group_width // actual_group_m

        warp_id = T.warp_id([4])
        lane_id = T.lane_id([32])
        wg_id = T.warpgroup_id([1])
        tid = T.thread_id_in_wg([128])
        A_smem = T.alloc_buffer(
            (stages, block_m, block_k),
            dtype,
            scope="shared",
            layout=a_layout,
            align=1024,
        )
        B_smem = T.alloc_buffer(
            (stages, block_n, block_k),
            dtype,
            scope="shared",
            layout=b_layout,
            align=1024,
        )
        tmem_addr = T.alloc_shared([1], "uint32")
        full_mbar = T.alloc_shared([stages], "uint64")
        empty_mbar = T.alloc_shared([stages], "uint64")
        if tid == 0:
            for stage in T.unroll(stages):
                T.ptx.mbarrier.init.shared.b64(full_mbar.ptr_to([stage]), T.uint32(1))
                T.ptx.mbarrier.init.shared.b64(empty_mbar.ptr_to([stage]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(block_n)
            )
        T.cuda.cta_sync()
        tmem = T.decl_buffer(
            (block_m, block_n),
            "float32",
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(block_m, block_n) : (1 @ TLane, 1 @ TCol)]),
        )

        # The empty barrier uses the opposite phase: every stage is initially
        # available to the producer before tcgen05 has signaled it once.
        if warp_id == 0 and lane_id == 0:
            for ko in T.serial(k_tiles):
                stage: T.let = ko % stages
                phase: T.let = (ko // stages) % 2
                T.cuda.mbarrier_wait(empty_mbar.ptr_to([stage]), phase ^ 1)
                Tx.copy_async(
                    A_smem[stage, :, :],
                    A[
                        block_m_idx * block_m : (block_m_idx + 1) * block_m,
                        ko * block_k : (ko + 1) * block_k,
                    ],
                    dispatch="tma_auto",
                    mbar=full_mbar.ptr_to([stage]),
                )
                Tx.copy_async(
                    B_smem[stage, :, :],
                    B[
                        block_n_idx * block_n : (block_n_idx + 1) * block_n,
                        ko * block_k : (ko + 1) * block_k,
                    ],
                    dispatch="tma_auto",
                    mbar=full_mbar.ptr_to([stage]),
                )
                T.ptx.mbarrier.arrive.expect_tx.shared.b64(
                    full_mbar.ptr_to([stage]), T.uint32(tile_bytes)
                )

        if warp_id == 1 and lane_id == 0:
            for ko in T.serial(k_tiles):
                stage: T.let = ko % stages
                phase: T.let = (ko // stages) % 2
                T.cuda.mbarrier_wait(full_mbar.ptr_to([stage]), phase)
                Tx.gemm_async(
                    tmem[:, :],
                    A_smem[stage, :, :],
                    B_smem[stage, :, :],
                    dispatch="tcgen05",
                    accum=ko != 0,
                )
                T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                    empty_mbar.ptr_to([stage])
                )
            last_stage: T.let = (k_tiles - 1) % stages
            last_phase: T.let = ((k_tiles - 1) // stages) % 2
            T.cuda.mbarrier_wait(empty_mbar.ptr_to([last_stage]), last_phase)

        T.cuda.cta_sync()
        T.ptx.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(output_chunk, dtype="float32")
        C_view = C_reg.view(
            block_m,
            output_chunk,
            layout=TileLayout(S[(block_m, output_chunk) : (1 @ axis_tid_in_wg, 1)]),
        )
        for chunk in T.unroll(output_chunks):
            if wg_id == 0:
                Tx.wg.copy_async(
                    C_view[:, :],
                    tmem[:, chunk * output_chunk : (chunk + 1) * output_chunk],
                )
                T.ptx.tcgen05.wait__ld.sync.aligned()
            T.cuda.cta_sync()
            Tx.copy(
                C[
                    block_m_idx * block_m + tid,
                    block_n_idx * block_n + chunk * output_chunk : block_n_idx * block_n
                    + (chunk + 1) * output_chunk,
                ],
                C_reg[:],
            )
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(block_n))

    target = tvm.target.Target.from_device(tvm.cuda(0))
    with target:
        executable = tvm.compile(tvm.IRModule({"main": gemm}), target=target, tir_pipeline="tirx")
    return executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--stages", type=int, choices=[1, 2, 3, 4], default=2)
    parser.add_argument("--group-m", type=int, default=16)
    parser.add_argument("--block-n", type=int, choices=[64, 128, 256], default=256)
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--min-repeat-ms", type=int, default=100)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Compare the full output with NumPy; use small dimensions for a quick check",
    )
    args = parser.parse_args()

    dev = tvm.cuda(0)
    if not dev.exist:
        raise RuntimeError("A CUDA device is required")
    if int(dev.compute_version.split(".")[0]) < 10:
        raise RuntimeError("tcgen05 requires CUDA compute capability 10.0 or newer")

    executable = build_kernel(args.m, args.n, args.k, args.stages, args.group_m, args.block_n)
    rng = np.random.default_rng(0)
    a_np = (rng.standard_normal((args.m, args.k)) * 0.01).astype(ml_dtypes.bfloat16)
    b_np = (rng.standard_normal((args.n, args.k)) * 0.01).astype(ml_dtypes.bfloat16)
    a = tvm.runtime.tensor(a_np, dev)
    b = tvm.runtime.tensor(b_np, dev)
    c = tvm.runtime.tensor(np.zeros((args.m, args.n), dtype="float32"), dev)

    executable(a, b, c)
    if args.verify:
        expected = a_np.astype("float32") @ b_np.astype("float32").T
        np.testing.assert_allclose(c.numpy(), expected, atol=5e-2, rtol=2e-2)

    timer = executable.jit().time_evaluator(
        "main",
        dev,
        number=args.number,
        repeat=args.repeat,
        min_repeat_ms=args.min_repeat_ms,
    )
    samples_s = np.asarray(timer(a, b, c).results, dtype="float64")
    flops = 2 * args.m * args.n * args.k
    output = {
        "measurement_scope": "end-to-end BF16 GEMM, C = A @ B.T",
        "device": dev.device_name,
        "compute_version": dev.compute_version,
        "target": str(tvm.target.Target.from_device(dev)),
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "block_m": 128,
        "block_n": args.block_n,
        "block_k": 64,
        "stages": args.stages,
        "group_m": args.group_m,
        "verified": args.verify,
        "flops_per_launch": flops,
        "time_ms_median": float(np.median(samples_s) * 1e3),
        "time_ms_min": float(np.min(samples_s) * 1e3),
        "tflops_median": float(flops / np.median(samples_s) / 1e12),
        "tflops_best": float(flops / np.min(samples_s) / 1e12),
        "samples_ms": (samples_s * 1e3).tolist(),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
