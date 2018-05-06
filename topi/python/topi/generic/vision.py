# pylint: disable=invalid-name, no-member
"""Generic vision operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import cpp

def _default_schedule(outs, auto_inline):
    """Default schedule for llvm."""
    target = tvm.target.current_target(allow_none=False)
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    if target.target_name != "llvm":
        raise RuntimeError("schedule not registered for '%s'" % target)
    s = tvm.create_schedule([x.op for x in outs])
    if auto_inline:
        x = outs[0]
        tvm.schedule.AutoInlineInjective(s)
        s[x].fuse(s[x].op.axis)
    return s

@tvm.target.generic_func
def schedule_shortcut(outs):
    """Schedule for shortcut

    Parameters
    ----------
    outs: Array of Tensor
      The computation graph description of shortcut
      in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    return _default_schedule(outs, False)

@tvm.target.generic_func
def schedule_reorg(outs):
    """Schedule for reorg

    Parameters
    ----------
    outs: Array of Tensor
      The computation graph description of reorg
      in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.generic.default_schedule(cpp_target, outs, False)

@tvm.target.generic_func
def schedule_region(outs):
    """Schedule for region

    Parameters
    ----------
    outs: Array of Tensor
      The computation graph description of region
      in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.generic.default_schedule(cpp_target, outs, False)

@tvm.target.generic_func
def schedule_nms(outs):
    """Schedule for non-maximum suppression

    Parameters
    ----------
    outs: Array of Tensor
      The computation graph description of nms
      in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    return _default_schedule(outs, False)

@tvm.target.generic_func
def schedule_multibox_prior(outs):
    """Schedule for multibox_prior

    Parameters
    ----------
    outs: Array of Tensor
      The computation graph description of multibox_prior
      in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    return _default_schedule(outs, False)

@tvm.target.generic_func
def schedule_multibox_transform_loc(outs):
    """Schedule for multibox_transform_loc

    Parameters
    ----------
    outs: Array of Tensor
      The computation graph description of
      multibox_transform_loc in the format
      of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    return _default_schedule(outs, False)

@tvm.target.generic_func
def schedule_multibox_detection(outs):
    """Schedule for multibox_detection

    Parameters
    ----------
    outs: Array of Tensor
      The computation graph description of multibox_detection
      in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    return _default_schedule(outs, False)
