# pylint: disable=invalid-name
"""generic declaration and schedules."""
from __future__ import absolute_import as _abs

import tvm

@tvm.target.override_native_generic_func("schedule_injective")
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
    target = tvm.target.current_target(allow_none=False)
    if target.target_name != "llvm":
        raise RuntimeError("schedule_injective not registered for '%s'" % target)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)
    s[x].fuse(s[x].op.axis)
    return s

schedule_elemwise = schedule_injective
schedule_broadcast = schedule_injective
