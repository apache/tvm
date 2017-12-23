# pylint: disable=invalid-name, unused-variable
"""Schedule for dense operator"""
from __future__ import absolute_import as _abs
import tvm
from .. import tag
from .. import generic

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

    def traverse(OP):
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule dense
        elif OP.tag == 'dense':
            Dense = OP.output(0)
            _schedule(Dense)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)
    return s
