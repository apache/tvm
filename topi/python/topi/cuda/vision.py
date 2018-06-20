# pylint: disable=invalid-name, unused-variable, unused-argument
"""Schedule for vision operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import generic
from .. import cpp

@generic.schedule_reorg.register(["cuda", "gpu"])
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
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.cuda.schedule_injective(cpp_target, outs)

@generic.schedule_region.register(["cuda", "gpu"])
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
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.cuda.schedule_region(cpp_target, outs)

@generic.schedule_multibox_prior.register(["cuda", "gpu"])
def schedule_multibox_prior(out):
    """Schedule for multibox_prior operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of multibox_prior
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for multibox_prior.
    """
    raise RuntimeError("Currently multibox_prior only supports CPU.")

@generic.schedule_multibox_detection.register(["cuda", "gpu"])
def schedule_multibox_detection(out):
    """Schedule for multibox_detection operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of multibox_detection
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for multibox_detection.
    """
    raise RuntimeError("Currently multibox_detection only supports CPU.")
