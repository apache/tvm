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
from __future__ import absolute_import as _abs
from operator import mul
from functools import reduce
import tvm
from tvm import te
from .. import tag
from .injective import schedule_injective_from_existing


def _schedule_reduce(op, sch, is_idx_reduce=False):
    if is_idx_reduce:
        data_out = op.input_tensors[0]
    else:
        data_in = op.input_tensors[0]
        data_out = op.output(0)

    if not sch[data_out].op.reduce_axis:
        return schedule_injective_from_existing(sch, op.output(0))

    if len(sch[data_out].op.axis) > 0:
        all_reduce = False
        num_thread = 32
        target = tvm.target.Target.current()
        if target and (target.kind.name == "opencl" or target.kind.name == "metal"):
            # without it, CL_INVALID_WORK_GROUP_SIZE occurred when running test_topi_reduce.py
            # don't know why
            num_thread = 16
        block_x = te.thread_axis("blockIdx.x")
        thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
        thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
    else:
        all_reduce = True
        num_thread = tvm.target.Target.current(allow_none=False).max_num_threads
        thread_x = te.thread_axis((0, num_thread), "threadIdx.x")

    # Fuse and refactor the reduce axis
    fused_reduce = sch[data_out].fuse(
        *[sch[data_out].op.reduce_axis[i] for i in range(len(sch[data_out].op.reduce_axis))]
    )
    ko, ki = sch[data_out].split(fused_reduce, factor=num_thread)
    if is_idx_reduce:
        data_out_rf, _ = sch.rfactor(data_out, ki)
    else:
        data_out_rf = sch.rfactor(data_out, ki)
    tx = sch[data_out].op.reduce_axis[0]
    sch[data_out].bind(tx, thread_x)
    sch[data_out_rf].compute_at(sch[data_out], tx)
    if is_idx_reduce:
        real_output = op.output(0)
        temp_idx_input = data_out.op.output(0)
        temp_val_input = data_out.op.output(1)
    else:
        real_output = data_out
    if not all_reduce:
        # Fuse and split the axis
        fused_outer = sch[real_output].fuse(
            *[sch[real_output].op.axis[i] for i in range(len(sch[real_output].op.axis))]
        )
        bx, outer_in = sch[real_output].split(fused_outer, factor=num_thread)

        # Bind the axes to threads and blocks
        sch[real_output].bind(outer_in, thread_y)
        sch[real_output].bind(bx, block_x)
        if is_idx_reduce:
            sch[temp_idx_input].compute_at(sch[real_output], outer_in)
            sch[temp_val_input].compute_at(sch[real_output], outer_in)
        sch[real_output].set_store_predicate(
            tvm.tir.all(
                thread_x.equal(0), block_x * num_thread + thread_y < reduce(mul, real_output.shape)
            )
        )
    else:
        if is_idx_reduce:
            spatial_axis = sch[real_output].fuse(*(sch[real_output].op.axis))
            sch[real_output].bind(spatial_axis, te.thread_axis("blockIdx.x"))
            sch[temp_idx_input].compute_at(sch[real_output], spatial_axis)
            sch[temp_val_input].compute_at(sch[real_output], spatial_axis)
        sch[real_output].set_store_predicate(thread_x.equal(0))
    return sch


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


def schedule_reduce_impl(
    outs, schedule_reduce_stage, schedule_injective_stage, inline_postops=False
):
    """Schedule for inject->reduce->bcast ops.
    Traverse over the stages in the schedule and schedule separate stages depending
    on the position of the stage. Injecteve post-ops of reduction will be scheduled using
    injection schedule, injective pre-ops of reduction will be inlined, reduction stage
    will be scheduled using reduction schedule

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of reduce in the format
          of an array of tensors.
    schedule_reduce_stage: Function responsible for scheduling the reduction
          stage
    schedule_injective_stage: Function responsible for scheduling the
          standalone injection stage

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
            raise RuntimeError(f"Unsupported operator: {operator.tag}")

        scheduled_ops.append(operator)

    def traverse_after_reduce(operator):
        """Internal traverse function"""
        if tag.is_broadcast(operator.tag):
            if operator not in scheduled_ops and not inline_postops:
                schedule_injective_stage(sch, operator.output(0))
            for tensor in operator.input_tensors:
                if tensor.op not in scheduled_ops:
                    if enable_auto_inline:
                        traverse_before_reduce(tensor.op)
                    else:
                        traverse_after_reduce(tensor.op)
        elif operator.tag == "comm_reduce":
            if operator not in scheduled_ops:
                schedule_reduce_stage(operator, sch, is_idx_reduce=False)
            for tensor in operator.input_tensors:
                if tensor.op not in scheduled_ops:
                    traverse_before_reduce(tensor.op)
        elif operator.tag == "comm_reduce_idx":
            if operator not in scheduled_ops:
                schedule_reduce_stage(operator, sch, is_idx_reduce=True)
            input_tensors = operator.input_tensors[0].op.input_tensors
            for tensor in input_tensors:
                if tensor.op not in scheduled_ops:
                    traverse_before_reduce(tensor.op)
        elif isinstance(operator, tvm.te.PlaceholderOp):
            pass
        else:
            raise RuntimeError(f"Unsupported operator: {operator.tag}")

        scheduled_ops.append(operator)

    for out in outs:
        traverse_after_reduce(out.op)
    return sch


def schedule_reduce(outs):
    return schedule_reduce_impl(outs, _schedule_reduce, schedule_injective_from_existing)
