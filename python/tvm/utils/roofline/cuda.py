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
"""Estimation of peak flops and memory bandwidth for cuda devices"""
from typing import Optional
from ...script import tir as T
from ... import nd, build, transform
from ...runtime import Device
from ...target import Target
from ...rpc.base import RPC_SESS_MASK
from ...rpc.client import RPCSession
from . import registry
from ...contrib import utils, nvcc


@registry.estimate_peak_flops.register("cuda")
def estimate_peak_flops_tensorcore(
    target: Target,
    dev: Device,
    remote: Optional[RPCSession],
    mat_dtype: str = "float16",
    acc_dtype: str = "float32",
) -> float:
    """Estimate the peak FLOP/s of a cuda device with tensorcores.

    This estimate should only be used to compare with operators that can use
    dense tensorcore mma instructions.

    References
    ----------
    Wei Sun, Ang Li, Tong Geng, Sander Stuijk, Henk Corporaal: "Dissecting
    Tensor Cores via Microbenchmarks: Latency, Throughput and Numerical
    Behaviors", 2022; http://arxiv.org/abs/2206.02874
    https://www.nvidia.com/content/PDF/nvidia-ampere-ga-102-gpu-architecture-whitepaper-v2.1.pdf

    Parameters
    ----------
    target : Target
        Target to run on. This should be as specific to the actual hardware as
        possible.
    dev : Device
        Device to run on.
    remote : Optional[RPCSession]
      Remote session used to upload artifacts for runtime evaluation. Must be
      the same session used to create `dev`.
    mat_dtype : str
        Dtype of matrices passed to mma instructions.
    acc_dtype : str
        Dtype of accumulator to use with mma instructions. Should be compatible
        with `mat_dtype`.

    Returns
    -------
    float
        Approximate sustained FLOP/s of this target/device combo assuming
        mma instructions. Addition and multiplications are each counted as
        separate FLOPs.
    """
    assert str(target.kind) == "cuda", "Only CUDA devices have tensorcores"

    @T.prim_func
    def peak_flops_tensorcore_tir(
        inp: T.Buffer((16, 16), mat_dtype),
        out: T.Buffer((16, 16), acc_dtype),
        n: T.int32,
        sms: T.int32,
    ):
        # pylint: disable=invalid-name, missing-function-docstring
        A = T.alloc_buffer((16, 16), dtype=mat_dtype, scope="wmma.matrix_a")
        B = T.alloc_buffer((16, 16), dtype=mat_dtype, scope="wmma.matrix_b")
        C = T.alloc_buffer((16, 16), dtype=acc_dtype, scope="wmma.accumulator")
        for _ in T.thread_binding(sms, thread="blockIdx.x"):
            for _ in T.thread_binding(
                8, thread="threadIdx.y"
            ):  # need 8 warps to get enough in-SM parallelism
                for _ in T.thread_binding(32, thread="threadIdx.x"):
                    T.evaluate(
                        T.tvm_load_matrix_sync(
                            A.data,
                            16,
                            16,
                            16,
                            0,
                            T.tvm_access_ptr(
                                T.type_annotation(dtype=mat_dtype),
                                inp.data,
                                0,
                                16,
                                1,
                                dtype="handle",
                            ),
                            16,
                            "row_major",
                            dtype="handle",
                        )
                    )
                    T.evaluate(T.tvm_fill_fragment(B.data, 16, 16, 16, 0, 0, dtype="handle"))
                    T.evaluate(T.tvm_fill_fragment(C.data, 16, 16, 16, 0, 0, dtype="handle"))
                    for _ in range(n):
                        T.evaluate(
                            T.tvm_mma_sync(
                                C.data, 0, A.data, 0, B.data, 0, C.data, 0, dtype="handle"
                            )
                        )
                    T.evaluate(
                        T.tvm_store_matrix_sync(
                            C.data,
                            16,
                            16,
                            16,
                            0,
                            T.tvm_access_ptr(
                                T.type_annotation(dtype=acc_dtype),
                                out.data,
                                0,
                                16,
                                2,
                                dtype="handle",
                            ),
                            16,
                            "row_major",
                            dtype="handle",
                        )
                    )

    n = 100000
    sms = dev.multi_processor_count
    specialized = peak_flops_tensorcore_tir.specialize(
        {peak_flops_tensorcore_tir.params[2]: n, peak_flops_tensorcore_tir.params[3]: sms}
    )
    with transform.PassContext(opt_level=3):
        f = build(specialized, target=target)

    # upload to remote if running over rpc
    if dev.device_type >= RPC_SESS_MASK:
        if remote is None:
            raise RuntimeError("A RPCSession must be provided when using a remote device.")
        temp = utils.tempdir()
        path = temp.relpath("peak_fma_flops.tar")
        f.export_library(path)
        remote.upload(path)
        f = remote.load_module("peak_fma_flops.tar")

    x = nd.empty((16, 16), dtype=mat_dtype, device=dev)
    y = nd.empty((16, 16), dtype=acc_dtype, device=dev)
    times = f.time_evaluator(f.entry_name, dev, repeat=10, number=1)(x, y)
    # each mma operation computes 16 x 16 x 16 FLOPs
    return n * 16 * 16 * 16 * 2 * sms * 8 / times.min


@T.prim_func
def peak_bandwidth_tir(a: T.handle, b: T.handle, blocks: T.int32, warp_size: T.int32) -> None:
    # pylint: disable=invalid-name, missing-function-docstring
    N = T.var("int32")
    A = T.match_buffer(a, [blocks, N, 4, warp_size], "float32")
    B = T.match_buffer(b, [blocks, 4, warp_size], "float32")
    for i in T.thread_binding(blocks, "blockIdx.x"):
        for k in T.serial(N):
            for l in T.unroll(4):
                # vectorized load is necessary to hit peak bandwidth
                for j in T.thread_binding(warp_size, "threadIdx.x"):
                    # += is necessary to introduce a data dependency for all
                    # elements of A, preventing the backend from removing the
                    # `k` loop and setting `k` to the loop extent.
                    B[i, l, j] += A[i, k, l, j]


@registry.estimate_peak_bandwidth.register("cuda")
def estimate_peak_bandwidth(
    target: Target,
    dev: Device,
    remote: Optional[RPCSession] = None,
) -> float:
    """Estimate peak memory bandwidth of a target/device combo.

    Peak bandwidth is estimated by running a small experiment on the underlying
    hardware. The peak bandwidth measurement assumes that vector instructions
    are being used to load the data.

    Parameters
    ----------
    target : Target
        Target to use for measurement. This target should be as specific to the
        underlying hardware as possible.
    dev : Device
        Device to measure peak bandwidth on.
    remote : Optional[RPCSession]
      Remote session used to upload artifacts for runtime evaluation. Must be
      the same session used to create `dev`.

    Returns
    -------
    float
        Peak memory bandwidth in bytes/seconds.
    """
    assert nvcc.have_tensorcore(
        dev.compute_version
    ), "CUDA roofline only works with devices that have tensorcores"
    warp_size = dev.warp_size
    # These sizes seem large enough to give the card time to hit a fixpoint on memory bandwidth
    blocks = 1024
    size = 1024

    specialized = peak_bandwidth_tir.specialize(
        {peak_bandwidth_tir.params[2]: blocks, peak_bandwidth_tir.params[3]: warp_size}
    )
    with transform.PassContext(opt_level=3):
        f = build(specialized, target=target)

    # upload to remote if running over rpc
    if dev.device_type >= RPC_SESS_MASK:
        if remote is None:
            raise RuntimeError("A RPCSession must be provided when using a remote device.")
        temp = utils.tempdir()
        path = temp.relpath("peak_bandwidth.tar")
        f.export_library(path)
        remote.upload(path)
        f = remote.load_module("peak_bandwidth.tar")

    a = nd.empty((blocks, size, 4, warp_size), dtype="float32", device=dev)
    b = nd.empty((blocks, 4, warp_size), dtype="float32", device=dev)
    times = f.time_evaluator(f.entry_name, dev, repeat=10, number=1)(a, b)
    return a.numpy().size * 4 / times.min  # 4 bytes per float32
