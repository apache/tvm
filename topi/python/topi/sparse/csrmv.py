"""TVM operator compute SpMV in CSR format."""
from __future__ import absolute_import
import tvm
from .. import tag

def csrmv_default(data, indices, indptr, weight, bias=None):
    """The default implementation of csrmv in topi.

    Parameters
    ----------
    data : tvm.Tensor
        1-D with shape [nonzeros]

    indices : tvm.Tensor
        1-D with shape [nonzeros]

    indptr : tvm.Tensor
        1-D with shape [m+1]

    weight : tvm.Tensor
        2-D with shape [k, 1]

    bias : tvm.Tensor, optional
        1-D with shape [1]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [m, 1]
    """
    assert len(data.shape) == 1 and len(weight.shape) == 2, \
        "only support 2-dim csrmv"
    assert isinstance(weight, tvm.tensor.Tensor), \
        "weight matrix is assumed to be tvm.Tensor, but weight is `%s`" % (type(weight))
    if bias is not None:
        assert len(bias.shape) == 1
    batch = indptr.shape[0]-1
    def csrmv_default_ir(data, indices, indptr, weight, out):
        """define ir for csrmv"""
        irb = tvm.ir_builder.create()
        data_ptr = irb.buffer_ptr(data)
        indices_ptr = irb.buffer_ptr(indices)
        indptr_ptr = irb.buffer_ptr(indptr)
        weight_ptr = irb.buffer_ptr(weight)
        out_ptr = irb.buffer_ptr(out)
        num_rows = indptr.shape[0]-1
        with irb.for_range(0, num_rows, for_type="parallel", name='row') as row:
            dot = irb.allocate('float32', (1,), name='dot', scope='local')
            out_ptr[row] = 0.
            dot[0] = 0.
            row_start = indptr_ptr[row]
            row_end = indptr_ptr[row+1]
            row_elems = row_end-row_start
            with irb.for_range(0, row_elems, name='elemidx') as elemidx:
                elem = row_start+elemidx
                dot[0] += data_ptr[elem] * weight_ptr[indices_ptr[elem]]
            out_ptr[row] += dot[0]
        return irb.get()
    oshape = (batch, 1)
    matmul = tvm.extern(oshape, [data, indices, indptr, weight],
                        lambda ins, outs: csrmv_default_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
                        tag="csrmv", dtype='float32', name='csrmv')
    if bias is not None:
        matmul = tvm.compute((batch, 1), lambda i, j: matmul[i, 0] + bias[i], \
                             tag=tag.BROADCAST)
    return matmul


def csrmv(a, x, y=None):
    """The `csrmv` routine performs a matrix-vector operation defined as :math:`y := A*x + y`,
    where `x` and `y` are vectors, `A` is an m-by-k sparse matrix in the CSR format.

    Parameters

    ----------
    a : tvm.contrib.sparse.CSRNDArray
        2-D sparse matrix with shape [m, k]

    x : tvm.Tensor
        2-D dense matrix with shape [k, 1]

    y : tvm.Tensor, optional
        1-D dense vector with shape [1]

    Returns
    -------
    output : tvm.Tensor
        2-D dense matrix with shape [m, 1]
    """
    return csrmv_default(a.data, a.indices, a.indptr, x, y)
