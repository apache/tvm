# pylint: disable=invalid-name, unused-variable, trailing-whitespace
"""Schedule for softmax operator"""
import tvm
from .. import generic

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
