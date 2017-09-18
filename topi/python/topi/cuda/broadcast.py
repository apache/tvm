# pylint: disable=invalid-name,unused-variable
"""Schedule for broadcast operators"""
from __future__ import absolute_import as _abs
import tvm

from .elemwise import _schedule_elemwise


def schedule_broadcast(outs):
    """Schedule for broadcasting ops (broadcast_to + broadcast binary) + element-wise ops.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of broadcast_to in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    sch = tvm.create_schedule([x.op for x in outs])
    def traverse(operator):
        if operator.tag == 'ewise' or operator.tag == 'scale_shift':
            if operator not in sch.outputs:
                sch[operator].compute_inline()
            for tensor in operator.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        elif operator.tag == 'broadcast_to' or operator.tag == 'broadcast_binary_op':
            _schedule_elemwise(operator, sch)
        else:
            raise RuntimeError("Unsupported operator: %s" % operator.tag)

    traverse(outs[0].op)
    return sch
