# pylint: disable=invalid-name, unused-variable,
"""Schedule for cudnn and miopen extern op"""
import tvm
from .. import generic
from .injective import _schedule_injective


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
        _schedule_injective(out.op, s)
    return s
