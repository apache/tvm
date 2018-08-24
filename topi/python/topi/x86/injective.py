# pylint: disable=invalid-name
"""x86 declaration and schedules."""
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
    if len(s[x].op.axis) >= 5:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1], s[x].op.axis[2])
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 3:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1])
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 1:
        s[x].parallel(s[x].op.axis[0])
    return s

@generic.schedule_concatenate.register(["cpu"])
def schedule_concatenate(outs):
    """X86 schedule for concatenate op.

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
    def vectorize(sch, tensor, vectorize_limit):
        """Internal vectorization function for concatenate."""
        inner_axis = s[tensor].op.axis[len(s[tensor].op.axis) - 1]
        inner_length = tensor.shape[len(tensor.shape) - 1].value
        if inner_length <= vectorize_limit:
            sch[tensor].vectorize(inner_axis)
        else:
            split_factor = 1
            for i in range(vectorize_limit, 1, -1):
                if inner_length % i == 0:
                    split_factor = i
                    break
            if split_factor > 1:
                _, inner_i = sch[tensor].split(inner_axis, split_factor)
                sch[tensor].vectorize(inner_i)

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineInjective(s)
    if len(s[x].op.axis) >= 5:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1], s[x].op.axis[2])
        vectorize(s, x, 64)
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 3:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1])
        s[x].parallel(fused)
    else:
        s[x].parallel(s[x].op.axis[0])
    return s

schedule_elemwise = schedule_injective
schedule_broadcast = schedule_injective
