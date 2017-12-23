# pylint: disable=invalid-name, unused-variable, unused-argument
"""Schedule for binary dense operator."""
from __future__ import absolute_import as _abs
import tvm
from .. import tag
from .. import generic


@generic.schedule_binary_dense.register(["x86", "cpu"])
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

    def _schedule(A, B, C):
        if C.op in s.outputs:
            Out = C
        else:
            Out = outs[0].op.output(0)

        bn = 8
        yo, xo, yi, xi = s[Out].tile(Out.op.axis[1], Out.op.axis[0], bn, bn)
        s[Out].parallel(yo)
        s[Out].vectorize(yi)

        if C.op not in s.outputs:
            s[C].compute_at(s[Out], xi)

    def traverse(OP):
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        # schedule binary_dense
        elif OP.tag == 'binary_dense':
            output = OP.output(0)
            data = OP.input_tensors[0]
            weight = OP.input_tensors[1]
            _schedule(data, weight, output)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

    traverse(outs[0].op)
    return s
