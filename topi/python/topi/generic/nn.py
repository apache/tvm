"""Generic nn operators"""
from __future__ import absolute_import as _abs
import tvm


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
def schedule_conv2d_nchw(outs):
    """Schedule for conv2d_nchw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


@tvm.target.generic_func
def schedule_conv2d_nhwc(outs):
    """Schedule for conv2d_nhwc

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


@tvm.target.generic_func
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


@tvm.target.generic_func
def schedule_depthwise_conv2d_nchw(outs):
    """Schedule for depthwise_conv2d_nchw

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of depthwise_conv2d_nchw
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


@tvm.target.generic_func
def schedule_depthwise_conv2d_nhwc(outs):
    """Schedule for depthwise_conv2d_nhwc
    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of depthwise_conv2d_nhwc
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


@tvm.target.override_native_generic_func("schedule_reduce")
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


@tvm.target.override_native_generic_func("schedule_softmax")
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


@tvm.target.override_native_generic_func("schedule_dense")
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


@tvm.target.override_native_generic_func("schedule_pool")
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


@tvm.target.override_native_generic_func("schedule_global_pool")
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

@tvm.target.override_native_generic_func("schedule_binarize_pack")
def schedule_binarize_pack(outs):
    """Schedule for binarize_pack

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of binarize_pack
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


@tvm.target.override_native_generic_func("schedule_binary_dense")
def schedule_binary_dense(outs):
    """Schedule for binary_dense

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of binary_dense
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)
