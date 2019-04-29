# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-variable, unused-argument, no-member
"""Schedule for vision operators"""
from __future__ import absolute_import as _abs
import tvm
from .. import generic
from .. import cpp
from .. import tag
from .pooling import schedule_pool

def _default_schedule(outs):
    """Default schedule for gpu."""
    target = tvm.target.current_target()
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        """inline all one-to-one-mapping operators except the last stage (output)"""
        if op.tag in ["nms", "invalid_to_bottom"]:
            if op.tag == "nms":
                sort = op.input_tensors[1]
            else:
                out = op.input_tensors[0]
                sort = s[out].op.input_tensors[1]
            score = s[sort].op.input_tensors[0]
            fused = s[score].fuse(*s[score].op.axis)
            num_thread = int(tvm.target.current_target(allow_none=False).max_num_threads)
            bx, tx = s[score].split(fused, factor=num_thread)
            s[score].bind(bx, tvm.thread_axis("blockIdx.x"))
            s[score].bind(tx, tvm.thread_axis("threadIdx.x"))
        if tag.is_broadcast(op.tag):
            if op not in s.outputs:
                s[op].compute_inline()
            else:
                x = op.output(0)
                fused = s[x].fuse(*s[x].op.axis)
                num_thread = tvm.target.current_target(allow_none=False).max_num_threads
                bx, tx = s[x].split(fused, factor=num_thread)
                s[x].bind(bx, tvm.thread_axis("blockIdx.x"))
                s[x].bind(tx, tvm.thread_axis("threadIdx.x"))
            for tensor in op.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return s

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

@generic.schedule_nms.register(["cuda", "gpu"])
def schedule_nms(outs):
    """Schedule for non-maximum suppression

    Parameters
    ----------
    outs: Array of Tensor
      The computation graph description of nms
      in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    return _default_schedule(outs)

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
    return _default_schedule(outs)

@generic.schedule_multibox_transform_loc.register(["cuda", "gpu"])
def schedule_multibox_transform_loc(outs):
    """Schedule for multibox_transform_loc

    Parameters
    ----------
    outs: Array of Tensor
      The computation graph description of
      multibox_transform_loc in the format
      of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    return _default_schedule(outs)

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
    return _default_schedule(outs)

@generic.schedule_roi_align.register(["cuda", "gpu"])
def schedule_roi_align(outs):
    return schedule_pool(outs, 'NCHW')

@generic.schedule_roi_pool.register(["cuda", "gpu"])
def schedule_roi_pool(outs):
    return schedule_pool(outs, 'NCHW')

@generic.schedule_proposal.register(["cuda", "gpu"])
def schedule_proposal(outs):
    """Schedule for proposal operator.

    Parameters
    ----------
    outs: Array of Tensor
      The computation graph description of proposal
      in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []
    from .injective import _schedule_injective
    def traverse(op):
        if op.tag in ['bbox_score', 'sorted_bbox']:
            _schedule_injective(op, s)
        for tensor in op.input_tensors:
            if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                traverse(tensor.op)
        scheduled_ops.append(op)
    traverse(outs[0].op)
    return s

@generic.schedule_get_valid_counts.register(["cuda", "gpu"])
def schedule_get_valid_counts(outs):
    """Schedule for get_valid_counts operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of get_valid_counts
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    return _default_schedule(outs)

@generic.schedule_argsort.register(["cuda", "gpu"])
def schedule_argsort(outs):
    """Schedule for argsort operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of argsort
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []
    from .injective import _schedule_injective
    def traverse(op):
        for tensor in op.input_tensors:
            if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                traverse(tensor.op)
        scheduled_ops.append(op)
    traverse(outs[0].op)
    return s
