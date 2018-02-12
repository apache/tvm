# pylint: disable=invalid-name, unused-variable,
"""Schedule for composition of injective operator"""
import tvm
from .. import generic

def _schedule_injective(op, sch):
    x = op.output(0)
    sch[x].opengl()
    return sch


@generic.schedule_injective.register(["opengl"])
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
        _schedule_injective(out.op, s)
    return s

schedule_elemwise = schedule_injective
schedule_broadcast = schedule_injective
