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
# pylint: disable=invalid-name, unused-variable,
"""Schedule for composition of injective operator"""
import numpy as np

import tvm
from tvm import te
from .. import utils


def schedule_injective_from_existing(sch, out):
    """Schedule for injective op from existing schedule.

    Parameters
    ----------
    sch: Schedule
         The schedule to update.
    out: Tensor
         The tensor representing the injective op.

    Returns
    -------
    sch: Schedule
         The updated schedule.
    """

    def find_nearest_small_factor(num, target):
        """Find the nearest factor of the given number that is smaller than the target."""
        for i in range(target, 0, -1):
            if num % i == 0:
                return i
        # Unreachable because i=1 must hold.
        return -1

    fused = sch[out].fuse(*sch[out].op.axis)
    num_thread = tvm.target.Target.current(allow_none=False).max_num_threads
    max_block = 256

    # Vectorize on fp16 data type to enable half2 for better memory bandwidth utilization.
    vector_width = 2 if out.dtype == "float16" else 1

    is_dynamic_output = False
    for dim in out.shape:
        if not isinstance(dim, tvm.tir.IntImm):
            is_dynamic_output = True
            break

    out_len = utils.prod(out.shape)

    try:
        const_size = utils.get_const_int(out_len)

        # Adjust block and thread to make sure they are dividable so that vectorize can be
        # correctly applied.
        if vector_width > 1 and const_size % vector_width == 0:
            remain_total_size = const_size // vector_width
            cand_sizes = []
            for max_size in [num_thread, max_block]:
                cand_sizes.append(
                    max_size
                    if remain_total_size % max_size == 0
                    else find_nearest_small_factor(remain_total_size, max_size)
                )
                remain_total_size //= cand_sizes[-1]

            # If the product of candidate dividable (block * thread) is too small,
            # then the performance may be worse even half2 is enabled. Note that 0.7
            # is just a heuristic ratio and may not be optimal for all workloads.
            if np.prod(cand_sizes) / (max_block * num_thread) >= 0.7:
                num_thread, max_block = cand_sizes

        need_block_split = const_size > max_block * num_thread * vector_width
    except ValueError:
        need_block_split = False
        const_size = 0

    if vector_width > 1:
        fused, v = sch[out].split(fused, vector_width)
        sch[out].vectorize(v)

    if need_block_split:
        xo, xi = sch[out].split(fused, factor=num_thread * max_block)
        bx, tx = sch[out].split(xi, factor=num_thread)
        sch[out].reorder(bx, tx, xo)
        sch[out].bind(bx, te.thread_axis("blockIdx.x"))
        sch[out].bind(tx, te.thread_axis("threadIdx.x"))
    else:
        # Use less threads for dynamic shape ops to avoid runtime error.
        if is_dynamic_output:
            num_thread //= 2
        if const_size != 0 and const_size < num_thread:
            bx, tx = sch[out].split(fused, factor=const_size)
        else:
            bx, tx = sch[out].split(fused, factor=num_thread)
        sch[out].bind(tx, te.thread_axis("threadIdx.x"))
        sch[out].bind(bx, te.thread_axis("blockIdx.x"))

    return sch


def schedule_injective(outs):
    """Schedule for injective op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of injective in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    tvm.te.schedule.AutoInlineInjective(s)
    for out in outs:
        if not utils.is_empty_shape(out.shape):
            schedule_injective_from_existing(s, out)
    return s


schedule_elemwise = schedule_injective
schedule_broadcast = schedule_injective
