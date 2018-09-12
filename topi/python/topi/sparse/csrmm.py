"""TVM operator compute SpMM in CSR format."""
from __future__ import absolute_import
import tvm
from .. import tag
from ..util import simplify

def csrmm_default(data, indices, indptr, weight, bias=None):
    # pylint: disable=invalid-name
    """The default implementation of csrmm in topi.

    Parameters
    ----------
    data : tvm.Tensor
        1-D with shape [nonzeros]

    indices : tvm.Tensor
        1-D with shape [nonzeros]

    indptr : tvm.Tensor
        1-D with shape [m+1]

    weight : tvm.Tensor
        2-D with shape [k, n]

    bias : tvm.Tensor, optional
        1-D with shape [m]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [m, n]
    """
    assert len(data.shape) == 1 and len(indices.shape) == 1 and len(indptr.shape) == 1 \
        and len(weight.shape) == 2, "only support 2-dim csrmm"
    assert isinstance(weight, tvm.tensor.Tensor), \
        "weight matrix is assumed to be tvm.Tensor, but weight is `%s`" % (type(weight))
    if bias is not None:
        assert len(bias.shape) == 1
    M = simplify(indptr.shape[0]-1)
    _, N = weight.shape
    def csrmm_default_ir(data, indices, indptr, weight, out):
        """define ir for csrmm"""
        irb = tvm.ir_builder.create()
        data_ptr = irb.buffer_ptr(data)
        indices_ptr = irb.buffer_ptr(indices)
        indptr_ptr = irb.buffer_ptr(indptr)
        weight_ptr = irb.buffer_ptr(weight)
        out_ptr = irb.buffer_ptr(out)
        M = simplify(indptr.shape[0]-1)
        _, N = weight.shape
        with irb.for_range(0, N, for_type="vectorize", name='n') as n:
            with irb.for_range(0, M, for_type="parallel", name='row') as row:
                dot = irb.allocate('float32', (1,), name='dot', scope='local')
                out_ptr[row*N+n] = 0.
                dot[0] = 0.
                row_start = indptr_ptr[row]
                row_end = indptr_ptr[row+1]
                row_elems = row_end-row_start
                with irb.for_range(0, row_elems, name='idx') as idx:
                    elem = row_start+idx
                    dot[0] += data_ptr[elem] * weight_ptr[indices_ptr[elem]*N+n]
                out_ptr[row*N+n] += dot[0]
        return irb.get()
    oshape = (M, N)
    matmul = tvm.extern(oshape, [data, indices, indptr, weight],
                        lambda ins, outs: csrmm_default_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
                        tag="csrmm", dtype='float32', name='out')
    if bias is not None:
        matmul = tvm.compute(oshape, lambda i, j: matmul[i, j] + bias[i], \
                             tag=tag.BROADCAST)
    return matmul


def csrmm(a, b, c=None):
    """The `csrmm` routine performs a matrix-matrix operation defined as :math:`C := A*B + C`,
    where `B` and `C` are dense matrices, `A` is an m-by-k sparse matrix in the CSR format.

    Parameters
    ----------
    a : tvm.contrib.sparse.CSRNDArray
        2-D sparse matrix with shape [m, k]

    b : tvm.Tensor
        2-D dense matrix with shape [k, n]

    c : tvm.Tensor, optional
        1-D dense vector with shape [n]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [m, n]
    """
    return csrmm_default(a.data, a.indices, a.indptr, b, c)
