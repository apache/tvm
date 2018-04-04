# pylint: disable=invalid-name, no-member, too-many-locals, too-many-statements, too-many-arguments, too-many-branches, line-too-long
"""Compute definition for elemwise ops with cuda backend"""
import tvm
from tvm.contrib import cudnn
import topi
from .. import generic
from ..nn.elemwise import relu

@relu.register(["cuda"])
def relu_cuda(x):
    """ReLU operator for cuda backend.

    Parameters
    ----------
    input : tvm.Tensor
        4-D with shape [batch, in_channel, in_height, in_width]

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [batch, out_channel, out_height, out_width]
    """
    target = tvm.target.current_target()
    if target and "cudnn" in target.libs:
        return cudnn.relu(x)
    return tvm.compute(x.shape, lambda *i: tvm.max(x(*i), tvm.const(0, x.dtype)))


@generic.schedule_relu.register(["cuda"])
def schedule_relu_cuda(outs):
    """Schedule for ReLU.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of relu
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for relu.
    """
    target = tvm.target.current_target()
    if target and "cudnn" in target.libs:
        return topi.generic.schedule_extern(outs)
    return topi.generic.injective.schedule_elemwise(outs)
