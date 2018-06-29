"""TVM operator compute Dense in CSR format."""
from __future__ import absolute_import
import tvm
from .. import tag
from ..util import simplify

def dense_default(data, indices, indptr, weight, bias=None):
    # pylint: disable=invalid-name
    """The default implementation of dense in topi.

    Parameters
    ----------
    data : tvm.Tensor
        1-D with shape [num_nonzeros]

    indices : tvm.Tensor
        1-D with shape [num_nonzeros]

    indptr : tvm.Tensor
        1-D with shape [M+1]

    weight : tvm.Tensor
        2-D with shape [K, N]

    bias : tvm.Tensor, optional
        1-D with shape [M]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [M, N]
    """
    assert len(data.shape) == 1 and len(indices.shape) == 1 and len(indptr.shape) == 1 \
        and len(weight.shape) == 2, "only support 2-dim dense"
    assert isinstance(weight, tvm.tensor.Tensor), \
        "weight matrix is assumed to be tvm.Tensor, but weight is `%s`" % (type(weight))
    if bias is not None:
        assert len(bias.shape) == 1
    dtype = data.dtype
    M = simplify(indptr.shape[0]-1)
    N, _ = weight.shape
    def dense_default_ir(data, indices, indptr, weight, out):
        """Define IR for Dense"""
        dtype = data.dtype
        irb = tvm.ir_builder.create()
        data_ptr = irb.buffer_ptr(data)
        indices_ptr = irb.buffer_ptr(indices)
        indptr_ptr = irb.buffer_ptr(indptr)
        weight_ptr = irb.buffer_ptr(weight)
        out_ptr = irb.buffer_ptr(out)
        M = simplify(indptr.shape[0]-1)
        N, K = weight.shape
        with irb.for_range(0, N, for_type="vectorize", name='n') as n:
            with irb.for_range(0, M, for_type="parallel", name='m') as m:
                dot = irb.allocate(dtype, (1,), name='dot', scope='local')
                out_ptr[m*N+n] = tvm.const(0, dtype)
                dot[0] = tvm.const(0, dtype)
                row_start = indptr_ptr[m]
                row_elems = indptr_ptr[m+1]-row_start
                with irb.for_range(0, row_elems, name='k') as k:
                    elem = row_start+k
                    dot[0] += data_ptr[elem] * weight_ptr[indices_ptr[elem]+n*K]
                out_ptr[m*N+n] += dot[0]
        return irb.get()
    oshape = (M, N)
    matmul = tvm.extern(oshape, [data, indices, indptr, weight],
                        lambda ins, outs: dense_default_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
                        tag="dense", dtype=dtype, name='out')
    if bias is not None:
        matmul = tvm.compute(oshape, lambda i, j: matmul[i, j] + bias[j], \
                             tag=tag.BROADCAST)
    return matmul


def dense(data, weight, bias=None):
    """Applies a linear transformation: :math:`Y = XW^T + b`.

    Parameters
    ----------
    data : tvm.contrib.CSRTensor
        2-D with shape [batch, in_dim]

    weight : tvm.Tensor
        2-D with shape [out_dim, in_dim]

    bias : tvm.Tensor, optional
        1-D with shape [out_dim]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [batch, out_dim]
    """
    return dense_default(data.data, data.indices, data.indptr, weight, bias)
