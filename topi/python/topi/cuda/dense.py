# pylint: disable=invalid-name, unused-variable
"""Schedule for dense operator"""
from __future__ import absolute_import as _abs
import tvm
from tvm.contrib import cublas
from ..nn.dense import dense, dense_default
from .. import tag
from .. import generic

@dense.register("cuda")
def dense_cuda(data, weight, bias=None):
    """Dense operator for cuda backend.

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
    if bias is not None:
        assert len(bias.shape) == 1
    batch, in_dim = data.shape
    out_dim, _ = weight.shape
    target = tvm.target.current_target()
    if "cublas" in target.libs:
        matmul = cublas.matmul(data, weight, False, True)
        if bias is not None:
            matmul = tvm.compute((batch, out_dim), \
                                 lambda i, j: matmul[i, j] + bias[j], \
                                 tag=tag.BROADCAST)
        return matmul
    return dense_default(data, weight, bias)


@generic.schedule_dense.register(["cuda", "gpu"])
def schedule_dense(outs):
    """Schedule for dense operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for dense.
    """
    target = tvm.target.current_target()
    if target.target_name == "cuda" and "cublas" in target.libs:
        return generic.schedule_extern(outs)

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _schedule(Dense):
        num_thread = 64
        k = Dense.op.reduce_axis[0]
        ko, kf = s[Dense].split(k, factor=num_thread)
        DenseF = s.rfactor(Dense, kf)

        if Dense.op in s.outputs:
            Out = Dense
        else:
            Out = outs[0].op.output(0)
            s[Dense].compute_at(s[Out], s[Out].op.axis[1])
        s[Out].bind(s[Out].op.axis[0], tvm.thread_axis("blockIdx.y"))
        s[Out].bind(s[Out].op.axis[1], tvm.thread_axis("blockIdx.x"))

        tx = s[Dense].op.reduce_axis[0]
        thread_x = tvm.thread_axis("threadIdx.x")
        s[Dense].bind(tx, thread_x)
        s[DenseF].compute_at(s[Dense], tx)
        s[Dense].set_store_predicate(thread_x.var.equal(0))
        s[Out].set_store_predicate(thread_x.var.equal(0))

    scheduled_ops = []

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule dense
        elif OP.tag == 'dense':
            Dense = OP.output(0)
            _schedule(Dense)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s
