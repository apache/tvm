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

"""Sparse operators"""
from __future__ import absolute_import
import tvm

from ..util import get_const_tuple


@tvm.target.generic_func
def sparse_dense(data, weight_data, weight_indices, weight_indptr):
    """
    Computes sparse-dense matrix multiplication of `data` and
    `(weight_data, weight_indices, weight_indptr).T`

    Parameters
    ----------
    x : tvm.Tensor
        2-D with shape [M, K], float32

    weight_data : tvm.Tensor
        1-D with shape [nnz] (CSR) or
        3-D with shape [num_blocks, bs_r, bs_c] (BSR)

    weight_indices : tvm.Tensor
        1-D with shape [nnz] (CSR) or
        1-D with shape [num_blocks] (BSR)

    weight_indptr : tvm.Tensor
        1-D with shape [N + 1] (CSR) or
        1-D with shape [(N + 1) // bs_r] (BSR)

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [M, N]
    """
    assert len(weight_data.shape) in (1, 3)
    if len(weight_data.shape) == 1:
        func = _sparse_dense_csrmm
    if len(weight_data.shape) == 3:
        func = _sparse_dense_bsrmm
    return func(data, weight_data, weight_indices, weight_indptr)


def _sparse_dense_csrmm(data, weight_data, weight_indices, weight_indptr):
    oshape = (
        get_const_tuple(data.shape)[0],
        get_const_tuple(weight_indptr.shape)[0] - 1)

    def f(i, row):
        row_start = weight_indptr[row]
        row_end = weight_indptr[row + 1]
        row_elems = row_end - row_start
        elem_idx = tvm.reduce_axis((0, row_elems), name="elem_idx")
        elem = row_start + elem_idx
        a_val = weight_data[elem]
        weight_val = data[i, weight_indices[elem]]
        return tvm.sum(a_val * weight_val, axis=elem_idx)
    return tvm.compute(oshape, f, tag="sparse_dense_csrmm")


def _sparse_dense_bsrmm(data, weight_data, weight_indices, weight_indptr):
    (m, _) = get_const_tuple(data.shape)
    (_, bs_r, bs_c) = get_const_tuple(weight_data.shape)
    (num_blocks_plus_1, ) = get_const_tuple(weight_indptr.shape)
    num_blocks = num_blocks_plus_1 - 1

    def _compute_block(i, nb_j, j):
        row_start = weight_indptr[nb_j]
        row_end = weight_indptr[nb_j + 1]
        row_elems = row_end - row_start
        elem_idx = tvm.reduce_axis(
            (0, row_elems), name="elem_idx")
        block_offset = row_start + elem_idx
        c = tvm.reduce_axis((0, bs_c), name="c")
        block_j = weight_indices[block_offset]
        block_ij_val = weight_data[block_offset][j][c]
        x_val = data[i, bs_c * block_j + c]
        return tvm.sum(block_ij_val * x_val, axis=[elem_idx, c])

    bsrmm_block = tvm.compute(
        (m, num_blocks, bs_r), _compute_block,
        tag="sparse_dense_bsrmm_block")
    return tvm.compute(
        (m, num_blocks * bs_r),
        lambda m, n: bsrmm_block[m, n // bs_r, n % bs_r],
        tag="sparse_dense_bsrmm")
