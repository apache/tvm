"""Generic nn operators"""
from __future__ import absolute_import as _abs
import tvm


def _default_schedule(outs, auto_inline):
    """Default schedule for llvm."""
    target = tvm.target.current_target(allow_none=False)
    if target.target_name != "llvm":
        raise RuntimeError("schedule_pool not registered for '%s'" % target)
    s = tvm.create_schedule([x.op for x in outs])
    if auto_inline:
        x = outs[0]
        tvm.schedule.AutoInlineInjective(s)
        s[x].fuse(s[x].op.axis)
    return s


@tvm.target.generic_func
def schedule_conv2d_nchw(outs):
    """Schedule for conv2d nchow

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
    return _default_schedule(outs, False)


@tvm.target.generic_func
def schedule_depthwise_conv2d_nchw(outs):
    """Schedule for conv2d nchow

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
    return _default_schedule(outs, False)


@tvm.target.generic_func
def schedule_reduce(outs):
    """Schedule for reduction

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
    return _default_schedule(outs, True)


@tvm.target.generic_func
def schedule_softmax(outs):
    """Schedule for softmax

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
    return _default_schedule(outs, False)


@tvm.target.generic_func
def schedule_dense(outs):
    """Schedule for dense

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
    return _default_schedule(outs, False)


@tvm.target.generic_func
def schedule_pool(outs):
    """Schedule for pool

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
    return _default_schedule(outs, False)


@tvm.target.generic_func
def schedule_global_pool(outs):
    """Schedule for global pool

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
    return _default_schedule(outs, False)
