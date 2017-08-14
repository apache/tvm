# pylint: disable=invalid-name,unused-variable
"""Schedule for broadcast operators"""
from __future__ import absolute_import as _abs
import tvm

def _schedule_broadcast_to(op, sch):
    data_in = op.input_tensors[0]
    data_out = op.output(0)

    num_thread = 512
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")

    xo, vi = sch[data_out].split(sch[data_out].op.axis[len(sch[data_out].op.axis) - 1],
                                 factor=4)
    sch[data_out].vectorize(vi)
    fused_axis = sch[data_out].fuse(*[sch[data_out].op.axis[i]
                                      for i in range(len(sch[data_out].op.axis) - 1)] + [xo])
    bx, tx = sch[data_out].split(fused_axis, factor=num_thread)

    sch[data_out].bind(bx, block_x)
    sch[data_out].bind(tx, thread_x)
    return sch


def schedule_broadcast_map(op):
    """Schedule for broadcast map ops + ewise + scale_shift ops.

    Parameters
    ----------
    op: tvm.tensor.Operation
        The symbolic description of the operation, should be reduce or
        reduce followed by a sequence of one-to-one-mapping operators.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    def traverse(operator):
        if operator.tag == 'ewise' or operator.tag == 'scale_shift':
            if operator not in sch.outputs:
                sch[operator].compute_inline()
            for tensor in operator.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        elif operator.tag == 'broadcast_to':
            _schedule_broadcast_to(operator, sch)
        else:
            raise RuntimeError("Unsupported operator: %s" % operator.tag)

    sch = tvm.create_schedule(op)
    traverse(op)
    return sch
