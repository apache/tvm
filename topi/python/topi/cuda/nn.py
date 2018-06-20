# pylint: disable=invalid-name
"""scheduler functions for cuda backend"""
from __future__ import absolute_import as _abs

import tvm
from .. import generic
from .. import tag
from .reduction import _schedule_reduce

@generic.schedule_lrn.register(["cuda"])
def schedule_lrn(outs):
    """Schedule for LRN

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of LRN
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    num_thread = 64
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")

    lrn = outs[0]
    sqr_sum_up = lrn.op.input_tensors[1]
    sqr_sum = sqr_sum_up.op.input_tensors[0]
    set_pad = sqr_sum.op.input_tensors[0]
    s[set_pad].bind(set_pad.op.axis[0], block_x)
    rxk = sqr_sum.op.reduce_axis[0]
    _, xki = s[sqr_sum].split(rxk, factor=num_thread)
    srf = s.rfactor(sqr_sum, xki)
    s[sqr_sum].bind(s[sqr_sum].op.axis[0], block_x)
    s[sqr_sum].bind(s[sqr_sum].op.reduce_axis[0], thread_x)
    s[srf].compute_at(s[sqr_sum], s[sqr_sum].op.reduce_axis[0])
    s[sqr_sum_up].bind(sqr_sum_up.op.axis[0], block_x)
    xto, _ = s[lrn].split(lrn.op.axis[1], nparts=num_thread)
    s[lrn].bind(lrn.op.axis[0], block_x)
    s[lrn].bind(xto, thread_x)
    return s

@generic.schedule_l2norm.register(["cuda"])
def schedule_l2norm(outs):
    """Schedule for L2norm

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of L2norm
          in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def traverse(OP):
        '''inline all one-to-one-mapping operators
        except the last stage (output)'''
        if tag.is_injective(OP.tag) or OP.tag == 'l2norm':
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)
        elif OP.tag == 'comm_reduce':
            _schedule_reduce(OP, s, is_idx_reduce=False)
            for tensor in OP.input_tensors:
                traverse(tensor.op)
        else:
            raise RuntimeError("Unsupported operator tag: %s" % OP.tag)
    traverse(outs[0].op)

    num_thread = 64
    l2norm = outs[0]
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    xto, _ = s[l2norm].split(l2norm.op.axis[1], nparts=num_thread)
    s[l2norm].bind(l2norm.op.axis[0], block_x)
    s[l2norm].bind(xto, thread_x)

    return s
