# pylint: disable=invalid-name, unused-variable
"""Schedule for vision operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import tag
from .. import generic

@generic.schedule_region.register(["cuda", "gpu"])
def schedule_region(outs):
    """Schedule for region operator.
    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of region
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for region.
    """

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    output = outs[0].op.output(0)
    #thread = 64 for higher size tensors, give resource_unavailable error for higher values
    num_thread = 64
    def _schedule_softmax(softmax_op):
        softmax = softmax_op.input_tensors[0]
        max_elem = softmax_op.input_tensors[1]
        expsum = softmax_op.input_tensors[2]
        block_x = tvm.thread_axis("blockIdx.x")
        thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
        s[max_elem].bind(max_elem.op.axis[0], block_x)
        k = expsum.op.reduce_axis[0]
        ko, ki = s[expsum].split(k, factor=num_thread)
        ef = s.rfactor(expsum, ki)
        s[expsum].bind(s[expsum].op.axis[0], block_x)
        s[expsum].bind(s[expsum].op.reduce_axis[0], thread_x)
        s[ef].compute_at(s[expsum], s[expsum].op.reduce_axis[0])
        s[expsum].set_store_predicate(thread_x.var.equal(0))
        tx, xi = s[softmax_op].split(softmax_op.axis[1], nparts=num_thread)
        s[softmax_op].bind(softmax_op.axis[0], block_x)
        s[softmax_op].bind(tx, thread_x)
        return max_elem.op.input_tensors[0]

    def _traverse(op):
        if tag.is_injective(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    _traverse(tensor.op)
        elif op.tag == 'softmax_output':
            tensor = _schedule_softmax(op)
            if tensor.op.input_tensors:
                _traverse(tensor.op)
        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)
    _traverse(outs[0].op)
    k = output.op.axis[0]
    bx, tx = s[output].split(k, factor=num_thread)
    s[output].bind(bx, tvm.thread_axis("blockIdx.x"))
    s[output].bind(tx, tvm.thread_axis("threadIdx.x"))
    return s
