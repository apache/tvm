#pylint: disable=invalid-name, no-member, too-many-locals, too-many-statements, too-many-arguments, too-many-branches, line-too-long
"""Schedule for rocm conv2d_nchw with auto fusion"""
import tvm
import topi
from .. import generic


@generic.schedule_conv2d_nchw.register(["rocm"])
def schedule_conv2d_nchw(outs):
    """Schedule for conv2d_nchw with rocm backend.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_nchw
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d_nchw.
    """
    target = tvm.target.current_target()
    if target and "miopen" in target.libs:
        return topi.generic.schedule_extern(outs)
    return topi.cuda.schedule_conv2d_nchw(outs)
