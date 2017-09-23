# pylint: disable=invalid-name, unused-variable
"""Schedule for dense operator"""
from __future__ import absolute_import as _abs
import tvm
from .. import tag

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
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _schedule(Dense):
        num_thread = 16
        k = Dense.op.reduce_axis[0]
        ko, kf = s[Dense].split(k, factor=num_thread)
        DenseF = s.rfactor(Dense, kf)

        if Dense.op in s.outputs:
            Out = Dense
            bx, ty = s[Out].split(s[Out].op.axis[0], factor=num_thread)
        else:
            Out = outs[0].op.output(0)
            bx, ty = s[Out].split(s[Out].op.axis[0], factor=num_thread)
            s[Dense].compute_at(s[Out], ty)
        s[Out].bind(bx, tvm.thread_axis("blockIdx.x"))
        s[Out].bind(ty, tvm.thread_axis("threadIdx.y"))

        tx = s[Dense].op.reduce_axis[0]
        thread_x = tvm.thread_axis("threadIdx.x")
        s[Dense].bind(tx, thread_x)
        s[DenseF].compute_at(s[Dense], tx)
        s[Dense].set_store_predicate(thread_x.var.equal(0))
        s[Out].set_store_predicate(thread_x.var.equal(0))

    def traverse(OP):
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule dense
        if OP.tag == 'dense':
            Dense = OP.output(0)
            _schedule(Dense)

    traverse(outs[0].op)
    return s
