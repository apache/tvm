# pylint: disable=invalid-name, unused-variable,
"""Schedule for cudnn and miopen extern op"""
import tvm
from .. import generic

def _schedule_output(op, sch):
    x = op.output(0)
    fused = sch[x].fuse(*sch[x].op.axis)
    num_thread = tvm.target.current_target(allow_none=False).max_num_threads
    bx, tx = sch[x].split(fused, factor=num_thread)
    sch[x].bind(bx, tvm.thread_axis("blockIdx.x"))
    sch[x].bind(tx, tvm.thread_axis("threadIdx.x"))
    return sch


@generic.schedule_extern.register(["cuda", "gpu"])
def schedule_extern(outs):
    """Schedule for an extern op followed by injective operations.
       For example, cudnn kernel + bias add + relu.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of extern plus injective ops in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    tvm.schedule.AutoInlineInjective(s)
    for out in outs:
        if isinstance(out.op, tvm.tensor.ExternOp):
            continue
        _schedule_output(out.op, s)
    return s
