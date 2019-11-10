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
# pylint: disable=invalid-name
"""x86 declaration and schedules."""
from __future__ import absolute_import as _abs
import tvm
from .. import tag
from .. import generic


def _schedule_reduce(sch, out):
    if len(sch[out].op.axis) >= 5:
        # avoid too many parallelism
        fused = sch[out].fuse(sch[out].op.axis[0], sch[out].op.axis[1], sch[out].op.axis[2])
        sch[out].parallel(fused)
    else:
        fused = sch[out].fuse(*sch[out].op.axis)
    

@generic.schedule_reduce.register(["cpu"])
def schedule_reduce(outs):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    sch = tvm.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse_before_reduce(operator):
        """Internal travserse function"""
        if isinstance(operator, tvm.tensor.PlaceholderOp):
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
        """Internal travserse function"""
        if tag.is_broadcast(operator.tag):
            if operator not in scheduled_ops:
                generic.schedule_injective_from_existing(sch, operator)
            for tensor in operator.input_tensors:
                traverse_after_reduce(tensor.op)
        elif operator.tag == 'comm_reduce':
            _schedule_reduce(sch, operator, is_idx_reduce=False)
            for tensor in operator.input_tensors:
                if tensor.op not in scheduled_ops:
                    traverse_before_reduce(tensor.op)
        elif operator.tag == 'comm_reduce_idx':
            _schedule_reduce(sch, operator, is_idx_reduce=True)
            input_tensors = operator.input_tensors[0].op.input_tensors
            for tensor in input_tensors:
                if tensor.op not in scheduled_ops:
                    traverse_before_reduce(tensor.op)
        else:
            raise RuntimeError("Unsupported operator: %s" % operator.tag)

        scheduled_ops.append(operator)

    traverse_after_reduce(outs[0].op)
    return sch

