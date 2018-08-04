# pylint: disable=invalid-name, unused-variable, unused-argument
"""Schedule for binary dense operator."""
from __future__ import absolute_import as _abs
import tvm
from .. import tag
from .. import generic


@generic.schedule_binary_dense.register(["cpu"])
def schedule_binary_dense(outs):
    """Schedule for binary_dense.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of binary_dense
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for binary_dense.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def _schedule(A, B, C):
        s[C].split(s[C].op.reduce_axis[0], factor=8)
        s[C].parallel(s[C].op.axis[0])
        if C.op in s.outputs:
            Out = C
        else:
            Out = outs[0].op.output(0)
        xo, xi = s[Out].split(Out.op.axis[1], factor=8)
        s[Out].vectorize(xi)

    def traverse(OP):
        """Internal travserse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule binary_dense
        elif OP.tag == 'binary_dense':
            output = OP.output(0)
            data = OP.input_tensors[0]
            weight = OP.input_tensors[1]
            _schedule(data, weight, output)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s
