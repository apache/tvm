# pylint: disable=invalid-name, unused-variable,
"""Schedule for upsampling operator"""
from .. import generic
from .injective import schedule_injective

@generic.schedule_upsampling.register(["cuda", "gpu"])
def schedule_upsampling(outs):
    """Schedule for upsampling op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of upsampling in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return schedule_injective(outs)
