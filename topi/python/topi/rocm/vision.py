# pylint: disable=invalid-name, unused-variable
"""Schedule for vision operator"""
from __future__ import absolute_import as _abs
import tvm
from .. import generic
from .. import cpp

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
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.rocm.schedule_region(cpp_target, outs)
