# pylint: disable=invalid-name, unused-variable, unused-argument
"""Schedule for vision operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import generic
from .. import cpp
from .. import tag
import topi

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
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.cuda.schedule_injective(cpp_target, outs)

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
    target = tvm.target.current_target(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.target_name)
    return cpp.cuda.schedule_region(cpp_target, outs)

@generic.schedule_multibox_prior.register(["cuda", "gpu"])
def schedule_multibox_prior(outs):
    """Schedule for multibox_prior operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of multibox_prior
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for multibox_prior.
    """
    target = tvm.target.current_target()
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    def traverse(op):
        """inline all one-to-one-mapping operators except the last stage (output)"""
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            #TODO: should be injected automatically
            else:
                x = op.output(0)
                fused = s[x].fuse(*s[x].op.axis)
                num_thread = tvm.target.current_target(allow_none=False).max_num_threads
                bx, tx = s[x].split(fused, factor=num_thread)
                s[x].bind(bx, tvm.thread_axis("blockIdx.x"))
                s[x].bind(tx, tvm.thread_axis("threadIdx.x"))
            for tensor in op.input_tensors:
                if tensor.op.input_tensors:
                    traverse(tensor.op)

    traverse(outs[0].op)
    return s

@generic.schedule_multibox_detection.register(["cuda", "gpu"])
def schedule_multibox_detection(outs):
    """Schedule for multibox_detection operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of multibox_detection
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for multibox_detection.
    """
    raise RuntimeError("Currently multibox_detection only supports CPU.")
