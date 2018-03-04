# pylint: disable=invalid-name, unused-variable
"""Schedule for vision operator"""
from __future__ import absolute_import as _abs
import topi
from .. import generic

@generic.schedule_reorg.register(["rocm"])
def schedule_reorg(outs):
    """Schedule for reorg operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of reorg
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for reorg.
    """
    return topi.cuda.schedule_reorg(outs)

@generic.schedule_shortcut.register(["rocm"])
def schedule_shortcut(outs):
    """Schedule for shortcut operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of shortcut
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for shortcut.
    """
    return topi.cuda.schedule_shortcut(outs)

@generic.schedule_region.register(["rocm"])
def schedule_region(outs):
    """Schedule for region operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of region
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for region.
    """
    return topi.cuda.schedule_region(outs)
