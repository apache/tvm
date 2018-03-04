# pylint: disable=invalid-name, unused-variable
"""Schedule for vision operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import tag
from .. import generic

@generic.schedule_reorg.register(["cuda", "gpu"])
def schedule_reorg(outs):
    """Schedule for reorg operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of reorg
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for reorg.
    """
    target = tvm.target.current_target()
    if target.target_name == "cuda" and "cublas" in target.libs:
        return generic.schedule_extern(outs)

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def _schedule(Reorg):
        num_thread = 64#tvm.target.current_target(allow_none=False).max_num_threads
        bx, tx = s[Reorg].split(Reorg.op.axis[0], factor=num_thread)
        s[Reorg].bind(bx, tvm.thread_axis("blockIdx.x"))
        s[Reorg].bind(tx, tvm.thread_axis((0, num_thread), "threadIdx.x"))
        Out = outs[0].op.output(0)
        s[Reorg].compute_at(s[Out], s[Out].op.axis[1])
        tx, xi = s[Out].split(Out.op.axis[0], nparts=num_thread)
        s[Out].bind(tx, tvm.thread_axis((0, num_thread), "threadIdx.x"))

    def _traverse(OP):
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_injective(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    _traverse(tensor.op)
        # schedule reorg
        elif OP.tag == 'reorg':
            Reorg = OP.output(0)
            _schedule(Reorg)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)
    _traverse(outs[0].op)
    return s

def schedule_shortcut(outs):
    """Schedule for shortcut operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of shortcut
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for shortcut.
    """

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _schedule(Shortcut):
        num_thread = 64#tvm.target.current_target(allow_none=False).max_num_threads
        if Shortcut.op in s.outputs:
            Out = Shortcut
        else:
            Out = outs[0].op.output(0)
            s[Shortcut].compute_at(s[Out], s[Out].op.axis[1])
        k = Out.op.axis[0]
        bx, tx = s[Out].split(k, factor=num_thread)
        s[Out].bind(bx, tvm.thread_axis("blockIdx.x"))
        s[Out].bind(tx, tvm.thread_axis("threadIdx.x"))

    def _traverse(OP):
        print("Traverse OPs:", OP, OP.tag, OP.input_tensors)

        # inline all one-to-one-mapping operators except the last stage (output)
        # schedule shortcut
        if tag.is_injective(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    _traverse(tensor.op)
        elif OP.tag == 'shortcut':
            Shortcut = OP.output(0)
            _schedule(Shortcut)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)
    _traverse(outs[0].op)
    return s

def schedule_region(outs):
    """Schedule for region operator.
    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of region
        in the format of an array of tensors.
    -------
    s: Schedule
        The computation schedule for region.
    """

    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    def _schedule(Softmax):
        num_thread = 64#tvm.target.current_target(allow_none=False).max_num_threads
        softmax = Softmax.input_tensors[0]
        max_elem = Softmax.input_tensors[1]
        expsum = Softmax.input_tensors[2]
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
        tx, xi = s[Softmax].split(Softmax.axis[1], nparts=num_thread)
        s[Softmax].bind(Softmax.axis[0], block_x)
        s[Softmax].bind(tx, thread_x)
        return max_elem.op.input_tensors[0]

    def _sigmoid(OP):
        sigmoid_out = OP.input_tensors[0]
        num_thread = 64
        k = OP.axis[0]
        bx, tx = s[OP].split(k, factor=num_thread)
        s[OP].bind(bx, tvm.thread_axis("blockIdx.x"))
        s[OP].bind(tx, tvm.thread_axis("threadIdx.x"))

    def _traverse(OP):
        if tag.is_injective(OP.tag):
            if OP not in s.outputs:
                if OP.tag in ("elemwise",):
                    _sigmoid(OP)
                else:
                    s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if tensor.op.input_tensors:
                    _traverse(tensor.op)
        elif OP.tag == 'softmax_output':
            tensor = _schedule(OP)
            if tensor.op.input_tensors:
                _traverse(tensor.op)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)
    _traverse(outs[0].op)
    return s
