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
"""TVM operator compute SpMM in CSR format."""
from __future__ import absolute_import
import tvm
from tvm import te
from .. import tag
from ..utils import simplify
from ...tir.generic import cast


def csrmm_default(data, indices, indptr, weight, bias=None):
    # pylint: disable=invalid-name
    """The default implementation of csrmm in topi.

    Parameters
    ----------
    data : tvm.te.Tensor
        1-D with shape [nonzeros]

    indices : tvm.te.Tensor
        1-D with shape [nonzeros]

    indptr : tvm.te.Tensor
        1-D with shape [m+1]

    weight : tvm.te.Tensor
        2-D with shape [k, n]

    bias : tvm.te.Tensor, optional
        1-D with shape [m]

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [m, n]
    """
    assert (
        len(data.shape) == 1
        and len(indices.shape) == 1
        and len(indptr.shape) == 1
        and len(weight.shape) == 2
    ), "only support 2-dim csrmm"
    assert isinstance(
        weight, te.tensor.Tensor
    ), f"weight matrix is assumed to be tvm.te.Tensor, but weight is `{type(weight)}`"
    assert (
        data.dtype == weight.dtype
    ), f"Data and weight must have the same dtype, but they have {data.dtype} and {weight.dtype}"
    if bias is not None:
        assert len(bias.shape) == 1
    M = simplify(indptr.shape[0] - 1)
    _, N = weight.shape

    def csrmm_default_ir(data, indices, indptr, weight, out):
        """define ir for csrmm"""
        irb = tvm.tir.ir_builder.create()
        data_ptr = irb.buffer_ptr(data)
        indices_ptr = irb.buffer_ptr(indices)
        indptr_ptr = irb.buffer_ptr(indptr)
        weight_ptr = irb.buffer_ptr(weight)
        out_ptr = irb.buffer_ptr(out)
        M = simplify(indptr.shape[0] - 1)
        _, N = weight.shape
        with irb.for_range(0, N, kind="vectorize", name="n") as n:
            with irb.for_range(0, M, kind="parallel", name="row") as row:
                dot = irb.allocate(data.dtype, (1,), name="dot", scope="local")
                out_ptr[row * N + n] = cast(0, data.dtype)
                dot[0] = cast(0, data.dtype)
                row_start = indptr_ptr[row]
                row_end = indptr_ptr[row + 1]
                row_elems = row_end - row_start
                with irb.for_range(0, row_elems, name="idx") as idx:
                    elem = row_start + idx
                    dot[0] += data_ptr[elem] * weight_ptr[indices_ptr[elem] * N + n]
                out_ptr[row * N + n] += dot[0]
        return irb.get()

    oshape = (M, N)
    matmul = te.extern(
        oshape,
        [data, indices, indptr, weight],
        lambda ins, outs: csrmm_default_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
        tag="csrmm",
        dtype=data.dtype,
        name="out",
    )
    if bias is not None:
        matmul = te.compute(oshape, lambda i, j: matmul[i, j] + bias[i], tag=tag.BROADCAST)
    return matmul


def csrmm(a, b, c=None):
    """The `csrmm` routine performs a matrix-matrix operation defined as :math:`C := A*B + C`,
    where `B` and `C` are dense matrices, `A` is an m-by-k sparse matrix in the CSR format.

    Parameters
    ----------
    a : tvm.contrib.sparse.CSRNDArray
        2-D sparse matrix with shape [m, k]

    b : tvm.te.Tensor
        2-D dense matrix with shape [k, n]

    c : tvm.te.Tensor, optional
        1-D dense vector with shape [n]

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [m, n]
    """
    return csrmm_default(a.data, a.indices, a.indptr, b, c)
