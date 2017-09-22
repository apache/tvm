# pylint: disable=invalid-name, unused-variable,
"""Schedule for composition of injective operator"""
import tvm

def _schedule_injective(op, sch):
    x = op.output(0)
    fused = sch[x].fuse(*sch[x].op.axis)
    num_thread = 512
    bx, tx = sch[x].split(fused, factor=num_thread)
    sch[x].bind(bx, tvm.thread_axis("blockIdx.x"))
    sch[x].bind(tx, tvm.thread_axis("threadIdx.x"))
    return sch


def schedule_injective(outs):
    """Schedule for injective op.

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
    s = tvm.create_schedule([x.op for x in outs])

    tvm.schedule.AutoInlineInjective(s)
    return _schedule_injective(outs[0].op, s)

schedule_elemwise = schedule_injective
schedule_broadcast = schedule_injective
