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
# pylint: disable=invalid-name,no-else-return
"""MLAS operators"""
import tvm
from tvm import te
from tvm.topi.utils import get_const_tuple


def mlas_packb(B, K, N, transb_size, transb=True):
    """Pre-pack B matrix if it is constant for mlas_matmul, C = A * B^T.
    It only supports float32 datatype.

    Parameters
    ----------
    B : tvm.te.Tensor
        The second input of mlas_matmul.

    K : int
        The number of colums of A.

    N : int
        The number of colums of output C.

    transb_size : int
        The size (in bytes) of the output pre-packed B matrix.

    transb : bool
        Whether the B matrix is transposed.
    Returns
    -------
    PackedB: tvm.te.Tensor
        The pre-packed B matrix.
    """
    return te.extern(
        (transb_size),
        [B],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.mlas.gemm_packb",
            N,
            K,
            K if transb else N,
            transb,
            ins[0],
            outs[0],
        ),
        name="PackedB",
    )


def mlas_matmul(A, B, packb=False, K=0, N=0):
    """Computes matrix multiplication of `A` and `B`, C = A * B^T.
    It supports both batch_matmul and dense mode.

    Parameters
    ----------
    A : tvm.te.Tensor
        The first input.

    B : tvm.te.Tensor
        The second input.

    packb : bool
        Specify whether the B is pre-packed.

    K : int
        The number of colums of A.

    N : int
        The number of colums of output C.

    Returns
    -------
    C: tvm.te.Tensor
        The computed result.
    """
    if len(A.shape) == 3:
        # batch_matmul mode
        batch_A, M_A, K_A = get_const_tuple(A.shape)
        if packb:
            # when B is packed, the batch_size must be 1
            batch_B, N_B, K_B = 1, N, K
        else:
            batch_B, N_B, K_B = get_const_tuple(B.shape)
        assert K_A == K_B
        assert batch_B in (batch_A, 1)
        M, N, K = M_A, N_B, K_A
        return te.extern(
            (batch_A, M, N),
            [A, B],
            lambda ins, outs: tvm.tir.call_packed(
                "tvm.contrib.mlas.batch_sgemm",
                batch_A,
                batch_B,
                M,
                N,
                K,
                packb,
                ins[0],
                ins[1],
                outs[0],
            ),
            name="C",
        )
    else:
        # dense mode
        M_A, K_A = get_const_tuple(A.shape)
        if packb:
            N_B, K_B = N, K
        else:
            N_B, K_B = get_const_tuple(B.shape)
        assert K_A == K_B
        M, N, K = M_A, N_B, K_A
        return te.extern(
            (M, N),
            [A, B],
            lambda ins, outs: tvm.tir.call_packed(
                "tvm.contrib.mlas.batch_sgemm",
                1,
                1,
                M,
                N,
                K,
                packb,
                ins[0],
                ins[1],
                outs[0],
            ),
            name="C",
        )
