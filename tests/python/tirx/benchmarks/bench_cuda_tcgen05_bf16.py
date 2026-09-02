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
"""Compute-heavy BF16 tcgen05 throughput benchmark for Blackwell CUDA GPUs.

The kernel loads one A/B tile per CTA and repeatedly accumulates the same MMA
into TMEM.  It measures sustained tcgen05 instruction throughput, not an
end-to-end GEMM with unique operands for every K tile.
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


def build_kernel(num_ctas: int, mma_repeats: int, n: int):
    """Build a multi-CTA BF16 tcgen05 kernel."""
    m = 128
    k = 64
    dtype = "bfloat16"
    a_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (m, k))
    b_layout = mma_shared_layout(dtype, SwizzleMode.SWIZZLE_128B_ATOM, (n, k))
    total_bytes = (m * k + n * k) * 2

    @T.prim_func
    def bf16_tcgen05(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (m, k), dtype)
        B = T.match_buffer(B_ptr, (n, k), dtype)
        C = T.match_buffer(C_ptr, (num_ctas, m, n), "float32")
        T.device_entry()
        cta = T.cta_id([num_ctas])
        warp_id = T.warp_id([4])
        wg_id = T.warpgroup_id([1])
        tid = T.thread_id_in_wg([128])
        A_smem = T.alloc_buffer((m, k), dtype, scope="shared", layout=a_layout)
        B_smem = T.alloc_buffer((n, k), dtype, scope="shared", layout=b_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        tma_mbar = T.alloc_shared([1], "uint64")
        mma_mbar = T.alloc_shared([1], "uint64")
        if tid == 0:
            T.ptx.mbarrier.init.shared.b64(tma_mbar.ptr_to([0]), T.uint32(1))
            T.ptx.mbarrier.init.shared.b64(mma_mbar.ptr_to([0]), T.uint32(1))
        T.ptx.fence.proxy.async_.shared__cta()
        T.cuda.cta_sync()
        if warp_id == 0:
            T.ptx.tcgen05.alloc.cta_group__1.sync.aligned.shared__cta.b32(
                T.address_of(tmem_addr), T.uint32(n)
            )
        T.cuda.cta_sync()
        tmem = T.decl_buffer(
            (m, n),
            "float32",
            scope="tmem",
            allocated_addr=tmem_addr[0],
            layout=TileLayout(S[(m, n) : (1 @ TLane, 1 @ TCol)]),
        )
        if tid == 0:
            Tx.copy_async(A_smem[:, :], A[:, :], dispatch="tma_auto", mbar=tma_mbar.ptr_to([0]))
            Tx.copy_async(B_smem[:, :], B[:, :], dispatch="tma_auto", mbar=tma_mbar.ptr_to([0]))
            T.ptx.mbarrier.arrive.expect_tx.shared.b64(tma_mbar.ptr_to([0]), T.uint32(total_bytes))
        T.cuda.mbarrier_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        if tid == 0:
            for repeat in T.serial(mma_repeats):
                Tx.gemm_async(
                    tmem[:, :],
                    A_smem[:, :],
                    B_smem[:, :],
                    dispatch="tcgen05",
                    accum=repeat != 0,
                )
            T.ptx.tcgen05.commit.cta_group__1.mbarrier__arrive__one.shared__cluster.b64(
                mma_mbar.ptr_to([0])
            )
        T.cuda.mbarrier_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()
        T.ptx.tcgen05.fence__after_thread_sync()
        C_reg = T.alloc_local(n, dtype="float32")
        C_view = C_reg.view(m, n, layout=TileLayout(S[(m, n) : (1 @ axis_tid_in_wg, 1)]))
        if wg_id == 0:
            for chunk in T.unroll(n // 128):
                Tx.wg.copy_async(
                    C_view[:, chunk * 128 : (chunk + 1) * 128],
                    tmem[:, chunk * 128 : (chunk + 1) * 128],
                )
            T.ptx.tcgen05.wait__ld.sync.aligned()
        T.cuda.cta_sync()
        Tx.copy(C[cta, tid, 0:n], C_reg[:])
        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit.cta_group__1.sync.aligned()
            T.ptx.tcgen05.dealloc.cta_group__1.sync.aligned.b32(tmem_addr[0], T.uint32(n))

    target = tvm.target.Target.from_device(tvm.cuda(0))
    with target:
        executable = tvm.compile(
            tvm.IRModule({"main": bf16_tcgen05}), target=target, tir_pipeline="tirx"
        )
    return executable, (m, n, k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ctas", type=int, default=None)
    parser.add_argument("--mma-repeats", type=int, default=2048)
    parser.add_argument("--n", type=int, choices=[128, 256], default=128)
    parser.add_argument("--number", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--min-repeat-ms", type=int, default=100)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    dev = tvm.cuda(0)
    if not dev.exist:
        raise RuntimeError("A CUDA device is required")
    if int(dev.compute_version.split(".")[0]) < 10:
        raise RuntimeError("tcgen05 requires CUDA compute capability 10.0 or newer")
    num_ctas = args.ctas or int(dev.multi_processor_count) * 16

    executable, (m, n, k) = build_kernel(num_ctas, args.mma_repeats, args.n)
    rng = np.random.default_rng(0)
    a_np = (rng.standard_normal((m, k)) * 0.01).astype(ml_dtypes.bfloat16)
    b_np = (rng.standard_normal((n, k)) * 0.01).astype(ml_dtypes.bfloat16)
    a = tvm.runtime.tensor(a_np, dev)
    b = tvm.runtime.tensor(b_np, dev)
    c = tvm.runtime.tensor(np.zeros((num_ctas, m, n), dtype="float32"), dev)

    executable(a, b, c)
    if args.verify:
        expected = (a_np.astype("float32") @ b_np.astype("float32").T) * args.mma_repeats
        np.testing.assert_allclose(c.numpy()[0], expected, atol=5e-2, rtol=2e-2)

    timer = executable.jit().time_evaluator(
        "main",
        dev,
        number=args.number,
        repeat=args.repeat,
        min_repeat_ms=args.min_repeat_ms,
    )
    samples_s = np.asarray(timer(a, b, c).results, dtype="float64")
    flops = 2 * m * n * k * num_ctas * args.mma_repeats
    output = {
        "measurement_scope": "repeated same-tile BF16 tcgen05 instruction throughput",
        "device": dev.device_name,
        "compute_version": dev.compute_version,
        "multi_processor_count": int(dev.multi_processor_count),
        "cuda_runtime_max_clock_rate_khz": int(dev.max_clock_rate),
        "target": str(tvm.target.Target.from_device(dev)),
        "ctas": num_ctas,
        "mma_repeats": args.mma_repeats,
        "m": m,
        "n": n,
        "k": k,
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
