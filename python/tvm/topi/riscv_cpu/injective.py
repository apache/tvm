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
from tvm import te
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
    """RISCV schedule for injective op.
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
    te.schedule.AutoInlineInjective(s)
    for x in outs:
        # Vectorize "ADD32" operation.
        if "add" in x.name:
            is_int32 = x.op.input_tensors[0].dtype == 'int32' and x.op.input_tensors[1].dtype == 'int32'
            is_even = x.shape[-1] % 2 == 0
            if is_int32 and is_even:
                outer, inner = s[x].split(x.op.axis[-1], 2)
                s[x].vectorize(inner)
        elif not is_empty_shape(x.shape):
            schedule_injective_from_existing(s, x)

    return s
