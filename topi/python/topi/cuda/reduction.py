# pylint: disable=invalid-name,unused-variable,too-many-locals,len-as-condition
"""Schedule for reduce operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import tag

def _schedule_reduce(op, sch):
    data_in = op.input_tensors[0]
    data_out = op.output(0)
    assert len(sch[data_out].op.reduce_axis) > 0, "reduce_axis must be bigger than zero!"

    if len(sch[data_out].op.axis) > 0:
        all_reduce = False
        num_thread = 16
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
        thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
    else:
        all_reduce = True
        num_thread = 512
        thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")

    # Fuse and refactor the reduce axis
    fused_reduce = sch[data_out].fuse(*[sch[data_out].op.reduce_axis[i]
                                        for i in range(len(sch[data_out].op.reduce_axis))])
    ko, ki = sch[data_out].split(fused_reduce, factor=num_thread)
    data_out_rf = sch.rfactor(data_out, ki)
    sch[data_out_rf].compute_at(sch[data_out], sch[data_out].op.reduce_axis[0])
    if not all_reduce:
        # Fuse and split the axis
        fused_outer = sch[data_out].fuse(*[sch[data_out].op.axis[i]
                                           for i in range(len(sch[data_out].op.axis))])
        bx, outer_in = sch[data_out].split(fused_outer, factor=num_thread)

        # Bind the axes to threads and blocks
        sch[data_out].bind(sch[data_out].op.reduce_axis[0], thread_x)
        sch[data_out].set_store_predicate(thread_x.equal(0))
        sch[data_out].bind(outer_in, thread_y)
        sch[data_out].bind(bx, block_x)
    else:
        sch[data_out].bind(sch[data_out].op.reduce_axis[0], thread_x)
    return sch


def schedule_reduce(outs):
    """Schedule for inject->reduce->bcast ops.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of reduce in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    sch = tvm.create_schedule([x.op for x in outs])

    def traverse_before_reduce(operator):
        if isinstance(operator, tvm.tensor.PlaceholderOp):
            return
        elif tag.is_injective(operator.tag):
            sch[operator].compute_inline()
            for tensor in operator.input_tensors:
                traverse_before_reduce(tensor.op)
        else:
            raise RuntimeError("Unsupported operator: %s" % operator.tag)

    def traverse_after_reduce(operator):
        if tag.is_broadcast(operator.tag):
            raise RuntimeError("Not yet support ewise after reduce")
        elif operator.tag == 'comm_reduce':
            _schedule_reduce(operator, sch)
            for tensor in operator.input_tensors:
                traverse_before_reduce(tensor.op)
        else:
            raise RuntimeError("Unsupported operator: %s" % operator.tag)

    traverse_after_reduce(outs[0].op)
    return sch
