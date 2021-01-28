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
from tvm import te
from tvm.tir import IntImm
from ..utils import is_empty_shape


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
    if len(sch[out].op.axis) >= 5:
        fused = sch[out].fuse(sch[out].op.axis[0], sch[out].op.axis[1], sch[out].op.axis[2])
        sch[out].parallel(fused)
    elif len(sch[out].op.axis) >= 3:
        fused = sch[out].fuse(sch[out].op.axis[0], sch[out].op.axis[1])
        sch[out].parallel(fused)
    elif len(sch[out].op.axis) >= 1:
        sch[out].parallel(sch[out].op.axis[0])

    # Vectorize the inner most for loop. Tiling first to get a const extent
    if len(sch[out].op.axis) >= 1:
        l = sch[out].op.axis[-1]
        lo, li = sch[out].split(l, factor=16)
        sch[out].vectorize(li)

        # for 1D loop, the above split will break the parallel axis
        # Need to make the outer loop parallel again
        if len(sch[out].op.axis) == 1:
            sch[out].parallel(lo)

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
        # Check that the tensor shape is static. Otherwise skip vectorization.
        if isinstance(tensor.shape[len(tensor.shape) - 1], IntImm):
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
