# pylint: disable=invalid-name
"""generic declaration and schedules."""
from __future__ import absolute_import as _abs
import tvm
from .. import generic

@generic.schedule_injective.register(["cpu"])
def schedule_injective(outs):
    """X86 schedule for injective op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of injective in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)
    s[x].parallel(s[x].op.axis[0])
    return s

schedule_elemwise = schedule_injective
schedule_broadcast = schedule_injective
