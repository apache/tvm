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
# pylint: disable=invalid-name
"""MLAS relay operators"""
from tvm import _ffi
from . import _make


def mlas_matmul(A, B, packb=False, K=-1, N=-1):
    r"""Computes batch matrix multiplication of `A` and `B` when `A` and `B` are data
    in batch.

    .. math::

        C[i, :, :] = \mbox{matmul}(A[i, :, :], B[i, :, :]^T)

    Parameters
    ----------
    A : tvm.relay.Expr
        The first input.

    B : tvm.relay.Expr
        The second input.

    packb : bool
        Specify whether the B is pre-packed.

    K : int
        The number of colums of A.

    N : int
        The number of colums of output C.

    Returns
    -------
    result: tvm.relay.Expr
        The computed result.
    """
    return _make.mlas_matmul(A, B, packb, K, N)


def mlas_packb(B, K, N, transb=True):
    r"""Pre-pack B matrix if it is constant for mlas_matmul, C = A * B^T.

    Parameters
    ----------
    B : tvm.relay.Expr
        The second input of mlas_matmul.

    K : int
        The number of colums of A.

    N : int
        The number of colums of output C.

    transb : bool
        Whether the B matrix is transposed.
    Returns
    -------
    result: tvm.relay.Expr
        The pre-packed B matrix.
    """
    get_packb_size = _ffi.get_global_func("tvm.contrib.mlas.gemm_packb_size")
    packb_size = get_packb_size(N, K)
    # only support 4 bytes float32 datatype
    arr_size = int(packb_size / 4)
    return _make.mlas_packb(B, K, N, arr_size, transb)
