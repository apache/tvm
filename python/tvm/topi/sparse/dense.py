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
"""TVM operator compute Dense in CSR format."""
from __future__ import absolute_import
import tvm
from tvm import te
from .. import tag
from ..utils import simplify


def dense_si(data, indices, indptr, weight, bias=None):
    # pylint: disable=invalid-name
    """The implementation of dense in topi, assuming sparse input.

    Parameters
    ----------
    data : tvm.te.Tensor
        1-D with shape [num_nonzeros]

    indices : tvm.te.Tensor
        1-D with shape [num_nonzeros]

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
    ), "only support 2-dim dense"
    assert isinstance(
        weight, te.tensor.Tensor
    ), "weight matrix is assumed to be tvm.te.Tensor, but weight is `%s`" % (type(weight))
    if bias is not None:
        assert len(bias.shape) == 1
    dtype = data.dtype
    M = simplify(indptr.shape[0] - 1)
    N, _ = weight.shape

    def dense_default_ir(data, indices, indptr, weight, out):
        """Define IR for Dense"""
        dtype = data.dtype
        irb = tvm.tir.ir_builder.create()
        data_ptr = irb.buffer_ptr(data)
        indices_ptr = irb.buffer_ptr(indices)
        indptr_ptr = irb.buffer_ptr(indptr)
        weight_ptr = irb.buffer_ptr(weight)
        out_ptr = irb.buffer_ptr(out)
        M = simplify(indptr.shape[0] - 1)
        N, K = weight.shape
        with irb.for_range(0, N, kind="vectorize", name="n") as n:
            with irb.for_range(0, M, kind="parallel", name="m") as m:
                dot = irb.allocate(dtype, (1,), name="dot", scope="local")
                out_ptr[m * N + n] = tvm.tir.const(0, dtype)
                dot[0] = tvm.tir.const(0, dtype)
                row_start = indptr_ptr[m]
                row_elems = indptr_ptr[m + 1] - row_start
                with irb.for_range(0, row_elems, name="k") as k:
                    elem = row_start + k
                    dot[0] += data_ptr[elem] * weight_ptr[indices_ptr[elem] + n * K]
                out_ptr[m * N + n] += dot[0]
        return irb.get()

    oshape = (M, N)
    matmul = te.extern(
        oshape,
        [data, indices, indptr, weight],
        lambda ins, outs: dense_default_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
        tag="dense",
        dtype=dtype,
        name="out",
    )
    if bias is not None:
        matmul = te.compute(oshape, lambda i, j: matmul[i, j] + bias[j], tag=tag.BROADCAST)
    return matmul


def dense_sw(data, w_data, w_indices, w_indptr, bias=None):
    # pylint: disable=invalid-name
    """The implementation of dense in topi, assuming sparse weight.

    Parameters
    ----------
    data : tvm.te.Tensor
        2-D with shape [m, k]

    w_data : tvm.te.Tensor
        1-D with shape [nonzeros]

    w_indices : tvm.te.Tensor
        1-D with shape [nonzeros]

    w_indptr : tvm.te.Tensor
        1-D with shape [n+1]

    bias : tvm.te.Tensor, optional
        1-D with shape [n]

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [m, n]
    """
    assert (
        len(w_data.shape) == 1
        and len(w_indices.shape) == 1
        and len(w_indptr.shape) == 1
        and len(data.shape) == 2
    ), "only support 2-dim dense"
    assert isinstance(
        data, te.tensor.Tensor
    ), "data matrix is assumed to be tvm.te.Tensor, but weight is `%s`" % (type(data))
    if bias is not None:
        assert len(bias.shape) == 1
    dtype = data.dtype
    M, _ = data.shape
    N = simplify(w_indptr.shape[0] - 1)

    def dense_default_ir(data, w_data, w_indices, w_indptr, out):
        """Define IR for Dense"""
        dtype = data.dtype
        irb = tvm.tir.ir_builder.create()
        data_ptr = irb.buffer_ptr(data)
        w_data_ptr = irb.buffer_ptr(w_data)
        w_indices_ptr = irb.buffer_ptr(w_indices)
        w_indptr_ptr = irb.buffer_ptr(w_indptr)
        out_ptr = irb.buffer_ptr(out)
        M, K = data.shape
        N = simplify(w_indptr.shape[0] - 1)
        with irb.for_range(0, M, kind="vectorize", name="m") as m:
            with irb.for_range(0, N, kind="parallel", name="n") as n:
                dot = irb.allocate(dtype, (1,), name="dot", scope="local")
                out_ptr[m * N + n] = tvm.tir.const(0, dtype)
                dot[0] = tvm.tir.const(0, dtype)
                row_start = w_indptr_ptr[n]
                row_elems = w_indptr_ptr[n + 1] - row_start
                with irb.for_range(0, row_elems, name="k") as k:
                    elem = row_start + k
                    dot[0] += w_data_ptr[elem] * data_ptr[w_indices_ptr[elem] + m * K]
                out_ptr[m * N + n] += dot[0]
        return irb.get()

    oshape = (M, N)
    matmul = te.extern(
        oshape,
        [data, w_data, w_indices, w_indptr],
        lambda ins, outs: dense_default_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
        tag="dense",
        dtype=dtype,
        name="out",
    )
    if bias is not None:
        matmul = te.compute(oshape, lambda i, j: matmul[i, j] + bias[j], tag=tag.BROADCAST)
    return matmul


def dense(data, weight, bias=None):
    """Applies a linear transformation: :math:`Y = XW^T + b`.
    Either data or weight should be tvm.contrib.sparse.CSRNDArray.

    Parameters
    ----------
    data : tvm.contrib.sparse.CSRNDArray or te.tensor.Tensor
        2-D with shape [batch, in_dim]

    weight : te.tensor.Tensor or tvm.contrib.sparse.CSRNDArray
        2-D with shape [out_dim, in_dim]

    bias : te.tensor.Tensor, optional
        1-D with shape [out_dim]

    Returns
    -------
    output : tvm.te.Tensor
        2-D with shape [batch, out_dim]
    """
    ret = None
    if isinstance(data, tvm.contrib.sparse.CSRPlaceholderOp) and isinstance(
        weight, te.tensor.Tensor
    ):
        ret = dense_si(data.data, data.indices, data.indptr, weight, bias)
    elif isinstance(data, te.tensor.Tensor) and isinstance(
        weight, tvm.contrib.sparse.CSRPlaceholderOp
    ):
        ret = dense_sw(data, weight.data, weight.indices, weight.indptr, bias)
    else:
        raise NotImplementedError(
            "implementation for %s as data and %s as weights, "
            "is not supported yet."
            % (
                type(data),
                type(weight),
            )
        )
    return ret
