"""TVM operator fully connected compute."""
from __future__ import absolute_import
import tvm
from .. import tag

def dense_default(data, indices, indptr, weight, bias=None):
    # pylint: disable=unused-argument
    """The default implementation of dense in topi.

    Parameters
    ----------
    data : tvm.Tensor
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
    assert len(data.shape) == 2 and len(weight.shape) == 2, \
        "only support 2-dim dense"
    assert isinstance(weight, tvm.tensor.Tensor), \
        "weight matrix is assumed to be tvm.Tensor, but weight is `%s`" % (type(weight))
    if bias is not None:
        assert len(bias.shape) == 1
    batch = 1
    out_dim, _ = weight.shape
    def dense_default_ir(data, indices, indptr, weight, out):
        # pylint: disable=invalid-name
        """Define IR for SpMM"""
        ib = tvm.ir_builder.create()
        data_ptr = ib.buffer_ptr(data)
        indices_ptr = ib.buffer_ptr(indices)
        indptr_ptr = ib.buffer_ptr(indptr)
        weight_ptr = ib.buffer_ptr(weight)
        out_ptr = ib.buffer_ptr(out)
        num_rows = indptr.shape[0]-1
        with ib.for_range(0, num_rows, name='row') as row:
            dot = ib.allocate('float32', (1,), name='dot', scope='local')
            dot[0] = 0.
            row_start = indptr_ptr[row]
            row_end = indptr_ptr[row+1]
            with ib.for_range(row_start, row_end, name='elem') as elem:
                dot[0] += data_ptr[elem] * weight_ptr[indices_ptr[elem]]
            out_ptr[row] += dot[0]
        return ib.get()
    oshape = (out_dim, 1)
    matmul = tvm.extern(oshape, [data, indices, indptr, weight],
                        lambda ins, outs: dense_default_ir(ins[0], ins[1], ins[2], ins[3], outs[0]),
                        tag="dense", dtype='float32')
    print(matmul.op.body)
    if bias is not None:
        matmul = tvm.compute((batch, out_dim), \
                             lambda i, j: matmul[i, j] + bias[j], \
                             tag=tag.BROADCAST)
    return matmul


def dense(data, weight, bias=None):
    """Applies a linear transformation: :math:`Y = XW^T + b`.

    Parameters
    ----------
    data : tvm.Tensor
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
