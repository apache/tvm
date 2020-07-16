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
from tvm import te, tir, target, runtime
from ..util import is_empty_shape, get_const_int

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
    max_concurrency = runtime._ffi_api.max_concurrency()
    max_vectorization = 32

    if len(sch[out].op.axis) == 0:
        return sch

    # Try to fuse and parallel outer loops until we have enough parallelsim
    fused_len = 1
    has_unknown = False
    to_fuse = []
    for axis in sch[out].op.axis:
        to_fuse.append(axis)

        if isinstance(axis.dom.extent, tir.IntImm):
            fused_len *= axis.dom.extent.value
        else:
            has_unknown = True
            break

        if fused_len > max_concurrency:
            break

    if len(to_fuse) == 1:
        to_parallel = to_fuse[0]
    elif len(to_fuse) > 1:
        to_parallel = sch[out].fuse(*to_fuse)

    # Try to vectorize the inner loop
    if not has_unknown:
        num_remaining = len(sch[out].op.axis) - len(to_fuse)

        if num_remaining >= 1:
            # if there are remaining axes, directly vectroize the inner most one
            to_vectorize = sch[out].op.axis[-1]
            if isinstance(to_vectorize.dom.extent, tir.IntImm):
                to_vectorize_len = max_factor = to_vectorize.dom.extent.value
            else:
                to_vectorize = max_factor = 0
        else:  # otherwise, split out one axis from the fused outer axis
            to_vectorize = to_parallel
            to_vectorize_len = fused_len
            max_factor = fused_len // max_concurrency

        for factor in range(min(max_factor, max_vectorization), -1, -1):
            if factor <= 0 or to_vectorize_len % factor == 0:
                break

        if factor > 1:
            outer, inner = sch[out].split(to_vectorize, factor)
            sch[out].vectorize(inner)

            if to_vectorize == to_parallel:
                to_parallel = outer

    sch[out].parallel(to_parallel)
    return sch

def schedule_injective(outs):
    """X86 schedule for injective op.

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
    x = outs[0]
    s = te.create_schedule([x.op for x in outs])
    te.schedule.AutoInlineInjective(s)

    if not is_empty_shape(x.shape):
        schedule_injective_from_existing(s, x)
    return s

def schedule_concatenate(outs):
    """X86 schedule for concatenate op.

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
    def vectorize(sch, tensor, vectorize_limit):
        """Internal vectorization function for concatenate."""
        inner_axis = s[tensor].op.axis[len(s[tensor].op.axis) - 1]
        inner_length = tensor.shape[len(tensor.shape) - 1].value
        if inner_length <= vectorize_limit:
            sch[tensor].vectorize(inner_axis)
        else:
            split_factor = 1
            for i in range(vectorize_limit, 1, -1):
                if inner_length % i == 0:
                    split_factor = i
                    break
            if split_factor > 1:
                _, inner_i = sch[tensor].split(inner_axis, split_factor)
                sch[tensor].vectorize(inner_i)

    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    x = outs[0]
    s = te.create_schedule([x.op for x in outs])
    te.schedule.AutoInlineInjective(s)
    if len(s[x].op.axis) >= 5:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1], s[x].op.axis[2])
        vectorize(s, x, 64)
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 3:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1])
        s[x].parallel(fused)
    else:
        s[x].parallel(s[x].op.axis[0])
    return s

schedule_elemwise = schedule_injective
schedule_broadcast = schedule_injective
