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
# pylint: disable=invalid-name, unused-variable
"""Schedule for pooling operators"""

import logging
from tvm import topi, te
from tvm.target import Target
from .. import tag


def schedule_pool(outs, layout):
    """Create schedule for avgpool/maxpool"""

    if layout != "NHWC":
        logger = logging.getLogger("topi")
        logger.warning(
            """We currently only support NHWC target specific pools on arm_cpu,
               falling back on generic pool scheduling"""
        )
        return topi.generic.schedule_pool(outs, layout)

    return schedule_pool_2d(outs)


def schedule_pool_2d(outs):
    """Create arm_cpu specific 2D schedule for avgpool/maxpool"""

    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    schedule_ops = [x.op for x in outs]
    schedule = te.create_schedule(schedule_ops)
    scheduled_ops = []

    def traverse(op):
        # Recursively inline any injective operation that isn't the pooling
        # operation or hasn't already been scheduled.
        if tag.is_injective(op.tag):
            if op not in schedule.outputs:
                schedule[op].compute_inline()
            for tensor in op.input_tensors:
                if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule the actual pooling operation
        elif op.tag.startswith("pool"):
            n, height, width, channel = schedule[op].op.axis
            # Average pool consists of two parts; a sum then a division.
            # We can schedule the division loop to parallelize across height and
            # vectorize across width.
            enable_explicit_vectorization = not Target.current(allow_none=False).features.has_sve
            if op != outs[0].op:
                output = outs[0]
                output_fused = schedule[output].fuse(output.op.axis[1], output.op.axis[2])
                schedule[output].parallel(output_fused)
                vectorization_factor = (
                    8 if enable_explicit_vectorization else output.op.axis[3].dom.extent
                )
                _, inner = schedule[output].split(output.op.axis[3], vectorization_factor)
                schedule[output].vectorize(inner)

            padded_input = op.input_tensors[0]
            if isinstance(padded_input.op, te.tensor.ComputeOp):
                schedule[padded_input].compute_inline()

            # For targets without SVE try explicitly vectorizing the channel
            # loop, For SVE targets leave the loop in place for LLVM to convert
            # into a scalable vector loop.
            vectorization_factor = 8 if enable_explicit_vectorization else channel.dom.extent
            channel_outer, channel_inner = schedule[op].split(channel, vectorization_factor)
            schedule[op].vectorize(channel_inner)
            schedule[op].parallel(height)
            if len(schedule[op].op.reduce_axis) > 0:
                filter_height, filter_width = schedule[op].op.reduce_axis
                # We consider any filter of area < 10 to be small enough to
                # unroll; 3x3 filters have shown better performance when
                # unrolled.
                if filter_height.dom.extent * filter_width.dom.extent <= 9:
                    # For small filters, unrolling the filter loops allows us to
                    # vectorize over channels without reordering anything.
                    schedule[op].unroll(filter_width)
                    schedule[op].unroll(filter_height)
                else:
                    # Reordering so that channels is the fastest moving axis allows
                    # LLVM to vectorize across contiguous memory in the NHWC
                    # ordering.
                    schedule[op].reorder(
                        n, height, width, filter_height, filter_width, channel_outer, channel_inner
                    )
            else:
                schedule[op].reorder(n, height, width, channel_outer, channel_inner)
        else:
            raise RuntimeError("Unsupported operator: %s" % op.tag)

        scheduled_ops.append(op)

    traverse(outs[0].op)
    return schedule
