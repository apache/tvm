"""TVM operator compute SpMV in CSR format."""
from __future__ import absolute_import
import tvm
from .. import tag

def csrmm_default(data, indices, indptr, weight, bias=None):
    # pylint: disable=invalid-name
    """The default implementation of csrmm in topi.

    Parameters
    ----------
    data : tvm.Tensor
        1-D with shape [num_nonzeros]

    indices : tvm.Tensor
        1-D with shape [num_nonzeros]

    indptr : tvm.Tensor
        1-D with shape [num_rows+1]

    weight : tvm.Tensor
        2-D with shape [num_cols, 1]

    bias : tvm.Tensor, optional
        1-D with shape [num_rows]

    Returns
    -------
    output : tvm.Tensor
        2-D with shape [num_rows, 1]
    """
    assert len(data.shape) == 1 and len(indices.shape) == 1 and len(indptr.shape) == 1 \
        and len(weight.shape) == 2, "only support 2-dim csrmm"
    assert isinstance(weight, tvm.tensor.Tensor), \
        "weight matrix is assumed to be tvm.Tensor, but weight is `%s`" % (type(weight))
    if bias is not None:
        assert len(bias.shape) == 1
    M = indptr.shape[0]-1
    _, N = weight.shape
    def csrmm_default_ir(data, indices, indptr, weight, out):
        """Define IR for SpMV"""
        irb = tvm.ir_builder.create()
        data_ptr = irb.buffer_ptr(data)
        indices_ptr = irb.buffer_ptr(indices)
        indptr_ptr = irb.buffer_ptr(indptr)
        weight_ptr = irb.buffer_ptr(weight)
        out_ptr = irb.buffer_ptr(out)
        M = indptr.shape[0]-1
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


def csrmm(data, weight, bias=None):
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
    return csrmm_default(data.data, data.indices, data.indptr, weight, bias)
