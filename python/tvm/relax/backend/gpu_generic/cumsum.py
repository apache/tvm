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
# pylint: disable=invalid-name, too-many-nested-blocks
"""Backend kernels for cumsum operator."""

import math
from typing import Optional

from tvm.script import tir as T
from tvm.tir import PrimFunc


def _is_power_of_two(n: int):
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def gpu_2d_continuous_cumsum(
    ty_len: int = 4,
    tx_len: int = 32,
    thread_elem: int = 4,
    in_dtype: str = "int32",
    out_dtype: Optional[str] = None,
) -> PrimFunc:
    """Generate GPU kernel for 2D continuous cumsum, i.e. The cumsum axis is -1

    Parameters
    ----------
    ty_len : int
        The length of `threadIdx.y`

    tx_len : int
        The length of `threadIdx.x`

    thread_elem : int
        The number of elements processed by single thread

    in_dtype : str
        The input data type

    out_dtype : Optional[str]
        The output data type, if None, it will be the same as in_dtype

    Returns
    -------
    cumsum : PrimFunc
        The generated cumsum kernel
    """

    out_dtype = out_dtype or in_dtype

    # Configuration for GPU kernel
    TX = T.int64(tx_len)  # threadIdx.x
    TY = T.int64(ty_len)  # threadIdx.y
    N = T.int64(thread_elem)  # number of elements in single thread

    if not _is_power_of_two(TX) or not _is_power_of_two(TY) or not _is_power_of_two(N):
        raise ValueError("Configuration of TX, TY, N must be power of 2")

    # number of elements to be processed by single warp
    warp_elem = T.int64(tx_len * thread_elem)
    # number of elements to be processed by single block(SM)
    block_elem = T.int64(tx_len * ty_len * thread_elem)

    LOG_TX = T.int64(int(math.log2(tx_len)))
    LOG_BLOCK_N = T.int64(int(math.log2(tx_len * ty_len * thread_elem)))

    @T.macro
    def block_inclusive_inside_block(
        batch: T.int64,
        cur_len: T.int64,
        source: T.Buffer,
        output: T.Buffer,
        tmp_buf: T.Buffer,
        src_offset: T.int64,
        tmp_offset: T.int64,
    ):
        for by in T.thread_binding(batch, thread="blockIdx.y"):
            for bx in T.thread_binding(T.ceildiv(cur_len, block_elem), thread="blockIdx.x"):
                with T.block():
                    local_buf = T.alloc_buffer((thread_elem,), out_dtype, scope="local")
                    shared_buf = T.alloc_buffer((block_elem,), out_dtype, scope="shared")
                    for ty in T.thread_binding(TY, thread="threadIdx.y"):
                        for tx in T.thread_binding(TX, thread="threadIdx.x"):
                            tx_idx = bx * block_elem + ty * warp_elem + tx * thread_elem
                            # Load data from global memory
                            for i in T.vectorized(N):
                                local_buf[i] = T.if_then_else(
                                    tx_idx + i < cur_len,
                                    T.Cast(out_dtype, source[by, src_offset + tx_idx + i]),
                                    T.Cast(out_dtype, 0),
                                )
                            # Inclusive scan inside thread
                            for i in T.unroll(1, N):
                                local_buf[i] += local_buf[i - 1]
                            # Store data to shared memory
                            for i in T.vectorized(N):
                                shared_buf[ty * warp_elem + tx * thread_elem + i] = local_buf[i]
                            # Inclusive scan inside warp
                            for i in T.unroll(LOG_TX):
                                for j in T.vectorized(N):
                                    idx: T.int64 = ty * warp_elem + tx * thread_elem
                                    if tx >= (1 << i):
                                        shared_buf[idx + j] += shared_buf[
                                            idx - (1 << i) * thread_elem + N - 1
                                        ]
                            # Inclusive scan inside block
                            for i in T.unroll(1, TY):
                                for j in T.vectorized(N):
                                    if ty == 0:
                                        idx: T.int64 = i * warp_elem + tx * thread_elem
                                        shared_buf[idx + j] += shared_buf[i * warp_elem - 1]
                            # Write sum of block to global memory
                            for i in T.vectorized(N):
                                idx: T.int64 = ty * warp_elem + tx * thread_elem + i
                                if bx * block_elem + idx < cur_len:
                                    output[by, src_offset + bx * block_elem + idx] = shared_buf[idx]
                            if tx == 0 and ty == 0:
                                for i in T.vectorized(N):
                                    tmp_buf[by, tmp_offset + bx] = shared_buf[block_elem - 1]

    @T.macro
    def update_cross_block(
        batch: T.int64,
        cur_len: T.int64,
        source: T.Buffer,
        output: T.Buffer,
        src_offset: T.int64,
        out_offset: T.int64,
    ):
        for by in T.thread_binding(batch, thread="blockIdx.y"):
            for bx in T.thread_binding(T.ceildiv(cur_len, block_elem), thread="blockIdx.x"):
                for ty in T.thread_binding(TY, thread="threadIdx.y"):
                    for tx in T.thread_binding(TX, thread="threadIdx.x"):
                        for i in T.serial(N):
                            idx: T.int64 = bx * block_elem + ty * warp_elem + i * TX + tx
                            if idx < cur_len:
                                output[by, out_offset + idx] += T.if_then_else(
                                    bx > 0, source[by, src_offset + bx - 1], 0
                                )

    @T.prim_func(private=True)
    def cumsum(var_a: T.handle, var_out: T.handle):
        T.func_attr({"tir.is_scheduled": 1})  # prevent further scheduling
        m, n = T.int64(), T.int64()
        A = T.match_buffer(var_a, [m, n], dtype=in_dtype)
        Out = T.match_buffer(var_out, [m, n], dtype=out_dtype)
        Tmp = T.alloc_buffer([m, n], dtype=out_dtype)
        ceil_log2 = T.Cast("int64", T.ceil(T.log2(T.Cast("float32", n))))
        total_rounds = ceil_log2 // LOG_BLOCK_N

        block_inclusive_inside_block(
            m, n, A, Out, Tmp, src_offset=T.int64(0), tmp_offset=T.int64(0)
        )
        for i in range(total_rounds):
            cur_len = T.ceildiv(n, 1 << (LOG_BLOCK_N * (i + 1)))
            block_inclusive_inside_block(
                m,
                cur_len,
                Tmp,
                Tmp,
                Tmp,
                src_offset=i * T.ceildiv(n, block_elem),
                tmp_offset=(i + 1) * T.ceildiv(n, block_elem),
            )
        for i in range(total_rounds - 1):
            real_idx = total_rounds - 1 - i - 1
            cur_len = T.ceildiv(n, 1 << (LOG_BLOCK_N * (real_idx + 1)))
            update_cross_block(
                m,
                cur_len,
                Tmp,
                Tmp,
                src_offset=(real_idx + 1) * T.ceildiv(n, block_elem),
                out_offset=real_idx * T.ceildiv(n, block_elem),
            )
        update_cross_block(m, n, Tmp, Out, src_offset=0, out_offset=0)

    return cumsum
