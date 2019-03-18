# pylint: disable=invalid-name, unused-variable
"""Schedule for pooling operators"""
import tvm
from .. import generic

@generic.schedule_injective.register(["arm_cpu"])
def schedule_injective(outs):
    """ARM CPU schedule for injective op.

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
    s = tvm.create_schedule([x.op for x in outs])
    x = outs[0]
    if list(s[x].op.axis):
        # do not vectorize for broadcast
        (io, ii) = s[x].split(list(s[x].op.axis)[-1], 8)
        s[x].vectorize(ii)
    tvm.schedule.AutoInlineInjective(s)
    if len(s[x].op.axis) >= 4:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1], s[x].op.axis[2])
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 3:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1])
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 2:
        s[x].parallel(s[x].op.axis[0])
    return s
