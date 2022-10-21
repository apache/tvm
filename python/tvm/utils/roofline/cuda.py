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
import functools
import re
from typing import Dict, Optional, Tuple

import numpy as np

from ... import build, nd, transform
from ...contrib import nvcc, utils
from ...rpc.base import RPC_SESS_MASK
from ...rpc.client import RPCSession
from ...runtime import Device
from ...script import tir as T
from ...target import Target
from ...tir import PrimFunc
from . import registry


@functools.lru_cache(maxsize=None)
def estimate_peak_flops_tensorcore(
    target: Target,
    dev: Device,
    remote: Optional[RPCSession],
    mat_dtype: str = "float16",
    acc_dtype: str = "float32",
) -> Tuple[float, float, str]:
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
    peak_flops : float
        Approximate sustained FLOP/s of this target/device combo assuming
        mma instructions. Addition and multiplications are each counted as
        separate FLOPs.
    """

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


@registry.estimate_peak_flops.register("cuda")
def estimate_peak_flops(
    func: PrimFunc,  # pylint: disable=unused-argument
    features: Dict[str, np.ndarray],
    target: Target,
    dev: Device,
    remote: Optional[RPCSession],
) -> Tuple[float, float, str]:
    """Estimate the peak FLOP/s of a cuda device.

    Parameters
    ----------
    func : PrimFunc
        Function to estimate peak flops for. Used to check if a specific kind
        intrinsic or dtype could be used with this function.
    features : Dict[str, np.ndarry]
        Features extracted from `func`. Used to check if a specific kind
        intrinsic or dtype could be used with this function.
    target : Target
        Target to run on. This should be as specific to the actual hardware as
        possible.
    dev : Device
        Device to run on.
    remote : Optional[RPCSession]
      Remote session used to upload artifacts for runtime evaluation. Must be
      the same session used to create `dev`.

    Returns
    -------
    flops : float
        Estimated number of flops used by `func`.
    peak_flops : float
        Approximate sustained FLOP/s of this target/device combo. Addition and
        multiplications are each counted as separate FLOPs.
    name : str
        Dtype/intrinsic used by `func` to achieve peak flops.
    """
    assert nvcc.have_tensorcore(
        dev.compute_version
    ), "CUDA roofline only works with devices that have tensorcores"
    flops = np.sum(
        features["float_addsub"]
        + features["float_mul"]
        + features["float_mad"] * 2
        + features["float_divmod"]
    )
    peak_flops = estimate_peak_flops_tensorcore(target, dev, remote)
    return flops, peak_flops, "float16 tensorcore"


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


@functools.lru_cache(maxsize=None)
def estimate_peak_bandwidth_global_mem(
    target: Target,
    dev: Device,
    remote: Optional[RPCSession] = None,
) -> Tuple[float, float, str]:
    """Estimate peak bandwidth of global memory. See estimate_peak_bandwidth"""
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


@registry.estimate_peak_bandwidth.register("cuda")
def estimate_peak_bandwidth(
    func: PrimFunc,  # pylint: disable=unused-argument
    features: Dict[str, np.ndarray],
    target: Target,
    dev: Device,
    remote: Optional[RPCSession] = None,
) -> Tuple[float, float, str]:
    """Estimate peak memory bandwidth of a target/device combo.

    Peak bandwidth is estimated by running a small experiment on the underlying
    hardware. The peak bandwidth measurement assumes that vector instructions
    are being used to load the data.

    Parameters
    ----------
    func : PrimFunc
        Function to estimate peak bandwidth for. Used to check if a specific
        kind of memory could be used with this function.
    features : Dict[str, np.ndarry]
        Features extracted from `func`. Used to check if a specific kind of
        memory could be used with this function.
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
    loaded_bytes : float
        Estimated bytes loaded by `func`.
    peak_bandwidth : float
        Peak memory bandwidth in bytes/seconds.
    name : str
        Name of the memory being used.
    """
    # autoscheduler features do not take into account that 1.
    # global and shared memory have very different performance
    # characteristics -- both are included in the same bytes
    # touched count 2. multiple threads accessing the same byte
    # of memory does not use the same amount of bandwidth as
    # multiple threads accessing different bytes of memory. We
    # use unique bytes accessed here to avoid these two issues,
    # but this does bias results towards being more compute
    # bound.
    loaded_bytes = sum(
        [
            np.sum(x)
            for (k, x) in features.items()
            if re.match(r"^B[0-9]+\.unique_bytes$", k) is not None
        ]
    )
    peak_bandwidth = estimate_peak_bandwidth_global_mem(target, dev, remote)
    return loaded_bytes, peak_bandwidth, "global"
