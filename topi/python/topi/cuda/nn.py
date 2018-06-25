# pylint: disable=invalid-name
"""scheduler functions for cuda backend"""
from __future__ import absolute_import as _abs

import tvm
from .. import generic
from .. import cpp

@generic.schedule_lrn.register(["cuda"])
def schedule_lrn(outs):
    """Schedule for LRN

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of LRN
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.cuda.schedule_lrn(cpp_target, outs)

@generic.schedule_l2_normalize.register(["cuda"])
def schedule_l2_normalize(outs):
    """Schedule for L2 normalize

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of L2 normalize
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.cuda.schedule_l2_normalize(cpp_target, outs)
