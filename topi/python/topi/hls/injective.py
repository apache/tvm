# pylint: disable=invalid-name, unused-variable,
"""Schedule for composition of injective operator"""
import tvm
from .. import generic

@generic.schedule_injective.register(["hls"])
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
    for out in outs:
        fused = s[out].fuse(*s[out].op.axis)
        px, x = s[out].split(fused, nparts=1)
        s[out].bind(px, tvm.thread_axis("pipeline"))
    return s

schedule_elemwise = schedule_injective
schedule_broadcast = schedule_injective
