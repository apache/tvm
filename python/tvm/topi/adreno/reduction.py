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
# pylint: disable=invalid-name,unused-variable,too-many-locals,len-as-condition
"""Schedule for reduce operators"""
import numpy
import tvm
from tvm import te
from .. import tag
from ..utils import get_const_tuple
from .injective import schedule_injective_from_existing
from .utils import get_div


def _schedule_reduce_adreno(op, sch, is_idx_reduce=False):
    if is_idx_reduce:
        real_output = op.output(0)
        temp_idx_input = op.input_tensors[0].op.output(0)
        temp_val_input = op.input_tensors[0].op.output(1)
    else:
        real_output = op.output(0)
    shape = get_const_tuple(real_output.shape)
    latest4 = shape[-1] == 4
    div4 = numpy.prod(shape) % 4 == 0

    # Fuse and split the axis
    if latest4:
        fused_outer = sch[real_output].fuse(
            *[sch[real_output].op.axis[i] for i in range(len(sch[real_output].op.axis) - 1)]
        )
    else:
        fused_outer = sch[real_output].fuse(
            *[sch[real_output].op.axis[i] for i in range(len(sch[real_output].op.axis))]
        )

    ftc = numpy.prod(shape)
    a = fused_outer
    if latest4:
        sch[real_output].vectorize(sch[real_output].op.axis[-1])
    elif div4 and not is_idx_reduce:
        a, b = sch[real_output].split(fused_outer, factor=4)
        sch[real_output].vectorize(b)
        ftc = ftc / 4

    num_thread = get_div(ftc, 128)

    bx, outer_in = sch[real_output].split(a, factor=num_thread)

    sch[real_output].bind(bx, te.thread_axis("blockIdx.x"))
    sch[real_output].bind(outer_in, te.thread_axis("threadIdx.y"))
    if is_idx_reduce:
        sch[temp_idx_input].compute_at(sch[real_output], outer_in)
        sch[temp_val_input].compute_at(sch[real_output], outer_in)


def _enable_auto_inline(sch):
    def is_scheduled(stage):
        # auto inline requires the attach type is AttachType.kGroupRoot
        conds = [
            len(stage.relations) == 0,
            stage.attach_type == 1,
            stage.all_iter_vars == stage.leaf_iter_vars,
        ]
        if not all(conds):
            return True
        return False

    for s in sch.stages:
        if not s.is_output and isinstance(s.op, tvm.te.ComputeOp):
            if is_scheduled(s) or len(s.op.reduce_axis) != 0:
                return False
    return True


def schedule_reduce(outs):
    """Schedule for inject->reduce->bcast ops.

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
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    sch = te.create_schedule([x.op for x in outs])
    scheduled_ops = []
    enable_auto_inline = _enable_auto_inline(sch)

    def traverse_before_reduce(operator):
        """Internal traverse function"""
        if isinstance(operator, tvm.te.PlaceholderOp):
            return
        if tag.is_injective(operator.tag):
            sch[operator].compute_inline()
            for tensor in operator.input_tensors:
                if tensor.op not in scheduled_ops:
                    traverse_before_reduce(tensor.op)
        else:
            raise RuntimeError("Unsupported operator: %s" % operator.tag)

        scheduled_ops.append(operator)

    def traverse_after_reduce(operator):
        """Internal traverse function"""
        if tag.is_broadcast(operator.tag):
            if operator not in scheduled_ops:
                schedule_injective_from_existing(sch, operator.output(0))
            for tensor in operator.input_tensors:
                if tensor.op not in scheduled_ops:
                    if enable_auto_inline:
                        traverse_before_reduce(tensor.op)
                    else:
                        traverse_after_reduce(tensor.op)
        elif operator.tag == "comm_reduce":
            if operator not in scheduled_ops:
                _schedule_reduce_adreno(operator, sch)
            for tensor in operator.input_tensors:
                if tensor.op not in scheduled_ops:
                    traverse_before_reduce(tensor.op)
        elif operator.tag == "comm_reduce_idx":
            if operator not in scheduled_ops:
                _schedule_reduce_adreno(operator, sch, is_idx_reduce=True)
            input_tensors = operator.input_tensors[0].op.input_tensors
            for tensor in input_tensors:
                if tensor.op not in scheduled_ops:
                    traverse_before_reduce(tensor.op)
        elif isinstance(operator, tvm.te.PlaceholderOp):
            pass
        else:
            raise RuntimeError("Unsupported operator: %s" % operator.tag)

        scheduled_ops.append(operator)

    for out in outs:
        traverse_after_reduce(out.op)
    return sch
