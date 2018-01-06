"""Generic nn operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import generic

def _default_schedule(outs, auto_inline):
    """Default schedule for x86."""
    x = outs[0]
    s = tvm.create_schedule([x.op for x in outs])
    if auto_inline:
        tvm.schedule.AutoInlineInjective(s)
        s[x].fuse(s[x].op.axis)
        return s
    if len(s[x].op.axis) == 4:
        n, c, h, w = s[x].op.axis
        fused = s[x].fuse(n, c)
        s[x].parallel(fused)
    elif len(s[x].op.axis) == 1:
        axis = s[x].op.axis[0]
        s[x].parallel(axis)
    else:
        s[x].fuse(s[x].op.axis)
    return s


@generic.schedule_conv2d_transpose_nchw.register(["cpu"])
def schedule_conv2d_transpose_nchw(outs):
    """Schedule for conv2d_transpose_nchw

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_transpose_nchw
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


@generic.schedule_reduce.register(["cpu"])
def schedule_reduce(outs):
    """Schedule for reduction

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of reduce
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, True)


@generic.schedule_softmax.register(["cpu"])
def schedule_softmax(outs):
    """Schedule for softmax

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of softmax
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


@generic.schedule_dense.register(["cpu"])
def schedule_dense(outs):
    """Schedule for dense

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of dense
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


@generic.schedule_pool.register(["cpu"])
def schedule_pool(outs):
    """Schedule for pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


@generic.schedule_global_pool.register(["cpu"])
def schedule_global_pool(outs):
    """Schedule for global pool

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of global pool
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)
