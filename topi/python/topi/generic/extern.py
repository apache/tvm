# pylint: disable=invalid-name
"""generic declaration and schedules."""
from __future__ import absolute_import as _abs

import tvm

@tvm.target.generic_func
def schedule_extern(outs):
    """Schedule for an extern op followed by injective operations.

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
    target = tvm.target.current_target(allow_none=False)
    if target.target_name != "llvm":
        raise RuntimeError("schedule_injective not registered for '%s'" % target)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    return tvm.create_schedule([x.op for x in outs])
