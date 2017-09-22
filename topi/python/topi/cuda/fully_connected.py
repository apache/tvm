# pylint: disable=invalid-name
"""Schedule for fully connected operators"""
from __future__ import absolute_import as _abs
import tvm

def schedule_fully_connected(outs):
    """Schedule for fully_connected.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of fully_connected
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for fully_connected.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    Out = outs[0]

    num_thread = 64
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")

    k = Out.op.reduce_axis[0]
    ko, ki = s[Out].split(k, factor=num_thread)
    OutF = s.rfactor(Out, ki)
    s[Out].bind(s[Out].op.axis[0], block_x)
    s[Out].bind(s[Out].op.axis[1], block_y)
    s[Out].bind(s[Out].op.reduce_axis[0], thread_x)
    s[OutF].compute_at(s[Out], s[Out].op.reduce_axis[0])

    return s


def schedule_fully_connected_with_bias(outs):
    """Schedule for fully_connected with bias.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of fully_connected with bias
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for fully_connected with bias.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    temp = outs[0].op.input_tensors[0]
    Out = outs[0]

    s[temp].set_scope("local")

    num_thread = 8
    block_x = tvm.thread_axis("blockIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")

    i, j = s[Out].op.axis
    k = s[temp].op.reduce_axis
    by, ty = s[Out].split(i, factor=num_thread)
    bx, tx = s[Out].split(j, factor=num_thread)
    s[Out].reorder(by, bx, ty, tx)
    s[Out].bind(ty, thread_y)
    s[Out].bind(tx, thread_x)
    s[Out].bind(by, block_y)
    s[Out].bind(bx, block_x)
    s[temp].compute_at(s[Out], tx)

    return s
