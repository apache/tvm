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
# pylint: disable=invalid-name, unused-variable, unused-argument, no-member, import-outside-toplevel
"""Schedule for vision operators"""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
from .. import cpp
from .. import tag
from .pooling import schedule_pool
from .injective import schedule_injective_from_existing


def _default_schedule(outs):
    """Default schedule for gpu."""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        if tag.is_injective(op.tag) or op.tag in ["bbox_score", "sorted_bbox"]:
            schedule_injective_from_existing(s, op.output(0))
        for tensor in op.input_tensors:
            if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                traverse(tensor.op)
        scheduled_ops.append(op)

    for o in outs:
        traverse(o.op)

    return s


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
    target = tvm.target.Target.current(allow_none=False)
    cpp_target = cpp.TEST_create_target(target.kind.name)
    return cpp.cuda.schedule_injective(cpp_target, outs)


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


def schedule_roi_align(outs):
    return schedule_pool(outs, "NCHW")


def schedule_roi_pool(outs):
    return schedule_pool(outs, "NCHW")


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
    return _default_schedule(outs)


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
