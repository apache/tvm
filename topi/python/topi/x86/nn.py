"""x86 nn operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import generic

def _default_schedule(outs, auto_inline):
    """Default schedule for x86."""
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    if auto_inline:
        tvm.schedule.AutoInlineInjective(s)
        s[x].fuse(s[x].op.axis)
        return s
    if len(s[x].op.axis) == 4:
        n, c, _, _ = s[x].op.axis
        fused = s[x].fuse(n, c) # for nhwc layout, fuse n and h
        s[x].parallel(fused)
    else:
        s[x].parallel(s[x].op.axis[0])
    return s


@generic.schedule_softmax.register(["cpu"])
def schedule_softmax(outs):
    """Schedule for softmax

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of softmax
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


@generic.schedule_pool.register(["cpu"])
def schedule_pool(outs):
    """Schedule for pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)
