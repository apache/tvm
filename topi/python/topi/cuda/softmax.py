# pylint: disable=invalid-name, unused-variable, trailing-whitespace
"""Schedule for softmax operator"""
import tvm
import topi
from .. import generic
from ..nn.softmax import (softmax, compute_softmax,
                          log_softmax, compute_log_softmax)
from tvm.contrib import cudnn

@softmax.register("cuda")
def softmax_cuda(data, axis):
    """Softmax activation operator for cuda backend

    Parameters
    ----------
    data : tvm.Tensor
        can be any dimension

    axis : int
        channel axis

    Returns
    -------
    output : tvm.Tensor
        output shape is the same as input
    """
    target = tvm.target.current_target()
    if "cudnn" in target.libs:
        if (axis != -1):
            warnings.warn("Softmax with axis (> -1) is not supported on CuDNN")
            return topi.nn.compute_softmax(data, axis)

        if data.ndim == 2:
            return cudnn.softmax_forward(data, "accurate", "instance")
        else:
            return cudnn.softmax_forward(data, "accurate", "channel")
    else:
        return topi.nn.compute_softmax(data, axis)

@generic.schedule_softmax.register(["cuda", "gpu"])
def schedule_softmax(outs):
    """Schedule for softmax op.

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
    target = tvm.target.current_target()
    if target.target_name == "cuda" and "cudnn" in target.libs:
        return topi.generic.schedule_extern(outs)

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    softmax = outs[0]
    max_elem = softmax.op.input_tensors[1]
    expsum = softmax.op.input_tensors[2]

    num_thread = 64
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")

    s[max_elem].bind(max_elem.op.axis[0], block_x)

    k = expsum.op.reduce_axis[0]
    ko, ki = s[expsum].split(k, factor=num_thread)
    EF = s.rfactor(expsum, ki)
    s[expsum].bind(s[expsum].op.axis[0], block_x)
    s[expsum].bind(s[expsum].op.reduce_axis[0], thread_x)
    s[EF].compute_at(s[expsum], s[expsum].op.reduce_axis[0])
    s[expsum].set_store_predicate(thread_x.var.equal(0))
    tx, xi = s[softmax].split(softmax.op.axis[1], nparts=num_thread)
    s[softmax].bind(softmax.op.axis[0], block_x)
    s[softmax].bind(tx, thread_x)

    return s

@log_softmax.register("cuda")
def log_softmax(x):
    """Perform log softmax activation on the data

    Parameters
    ----------
    data : tvm.Tensor
        2-D input data

    Returns
    -------
    output : tvm.Tensor
        2-D output with same shape
    """

    assert len(x.shape) == 2, "only support 2-dim log softmax"
    if "cudnn" in target.libs:
        return cudnn.softmax_forward(data, "log", "instance")

    return topi.nn.compute_log_softmax(x)
