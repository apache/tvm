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
"""TVM operator compute SpMV in BSR format."""
from __future__ import absolute_import
import tvm
from tvm import te
from .. import tag
from ...tir.generic import cast


def bsrmv_default(data, indices, indptr, blocksize, weight, bias=None):
    """The default implementation of bsrmv in topi.

    Parameters
    ----------
    data : tvm.te.Tensor
        1-D with shape [nonzeros]

    indices : tvm.te.Tensor
        1-D with shape [nonzeros]

    indptr : tvm.te.Tensor
        1-D with shape [m+1]

    blocksize: tvm.te.Tensor
        Block size of the matrix
        2-D with shape [bs_r, bs_c]

    weight : tvm.te.Tensor
        2-D with shape [k, 1]

    bias : tvm.te.Tensor, optional
        1-D with shape [1]

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [m, 1]
    """
    assert len(data.shape) == 1 and len(weight.shape) == 2, "only support 2-dim bsrmv"
    assert isinstance(
        weight, te.tensor.Tensor
    ), "weight matrix is assumed to be tvm.te.Tensor, but weight is `%s`" % (type(weight))
    assert (
        data.dtype == weight.dtype
    ), "Data and weight must have the same dtype, but they have %s and %s" % (
        data.dtype,
        weight.dtype,
    )
    if bias is not None:
        assert len(bias.shape) == 1
    batch = indptr.shape[0] - 1

    def bsrmv_default_ir(data, indices, indptr, blocksize, weight, out):
        """define ir for bsrmv"""
        irb = tvm.tir.ir_builder.create()
        data_ptr = irb.buffer_ptr(data)
        indices_ptr = irb.buffer_ptr(indices)
        indptr_ptr = irb.buffer_ptr(indptr)
        block_r = blocksize[0]
        weight_ptr = irb.buffer_ptr(weight)
        out_ptr = irb.buffer_ptr(out)
        num_rows = indptr.shape[0] - 1
        block_r_num = int(num_rows/block_r)

        with irb.for_range(0, block_r_num, kind="vectorize", name="block_r_num") as block_r_loop:
            with irb.for_range(0, block_r, kind="parallel", name="row") as row:
                dot = irb.allocate(data.dtype, (1,), name="dot", scope="local")
                out_ptr[row] = cast(0, data.dtype)
                dot[0] = cast(0, data.dtype)
                row_start = indptr_ptr[row]
                row_end = indptr_ptr[row + 1]
                row_elems = row_end - row_start
                with irb.for_range(0, row_elems, name="elemidx") as elemidx:
                    elem = row_start + elemidx
                    dot[0] += data_ptr[elem] * weight_ptr[indices_ptr[elem]]
                out_ptr[row] += dot[0]

        return irb.get()

    oshape = (batch, 1)
    matmul = te.extern(
        oshape,
        [data, indices, indptr, blocksize, weight],
        lambda ins, outs: bsrmv_default_ir(ins[0], ins[1], ins[2], ins[3], ins[4], outs[0]),
        tag="bsrmv",
        dtype=data.dtype,
        name="bsrmv",
    )
    if bias is not None:
        matmul = te.compute((batch, 1), lambda i, j: matmul[i, 0] + bias[i], tag=tag.BROADCAST)
    return matmul


def bsrmv(a, blocksize, x, y=None):
    """The `bsrmv` routine performs a matrix-vector operation defined as :math:`y := A*x + y`,
    where `x` and `y` are vectors, `A` is an m-by-k sparse matrix in the BSR format.

    Parameters
    ----------
    a : tvm.contrib.sparse.BSRNDArray
        2-D sparse matrix with shape [m, k]

    blocksize: tvm.te.Tensor
        2-D bsr block size with shape [block_r, block_c]

    x : tvm.te.Tensor
        2-D dense matrix with shape [k, 1]

    y : tvm.te.Tensor, optional
        1-D dense vector with shape [1]

    Returns
    -------
    output : tvm.te.Tensor
        2-D dense matrix with shape [m, 1]
    """
    return bsrmv_default(a.data, a.indices, a.indptr, blocksize, x, y)
