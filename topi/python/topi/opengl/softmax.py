# pylint: disable=invalid-name, unused-variable, trailing-whitespace
"""Schedule for softmax operator"""
import tvm
from .. import generic

@generic.schedule_softmax.register(["opengl"])
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
    s[max_elem].opengl()
    s[expsum].opengl()
    s[softmax].opengl()
    return s
