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
import tvm
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
    if len(sch[out].op.axis) >= 4:
        fused = sch[out].fuse(sch[out].op.axis[0], sch[out].op.axis[1], sch[out].op.axis[2])
        sch[out].parallel(fused)
    elif len(sch[out].op.axis) >= 3:
        fused = sch[out].fuse(sch[out].op.axis[0], sch[out].op.axis[1])
        sch[out].parallel(fused)
    elif len(sch[out].op.axis) >= 2:
        sch[out].parallel(sch[out].op.axis[0])
    return sch


def schedule_injective(outs):
    """ARM CPU schedule for injective op.

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
    x = outs[0]

    if list(s[x].op.axis):
        # do not vectorize for broadcast
        dtype = "uint16" if x.dtype == "bfloat16" else x.dtype
        itemsize = max(1, tvm.DataType(dtype).bits // 8)
        (io, ii) = s[x].split(list(s[x].op.axis)[-1], 16 // itemsize)
        s[x].vectorize(ii)
    tvm.te.schedule.AutoInlineInjective(s)

    if not is_empty_shape(x.shape):
        schedule_injective_from_existing(s, x)
    return s


def schedule_concatenate(outs):
    """Schedule for concatenate op.

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of concatenate in the format
          of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    x = outs[0]
    tvm.te.schedule.AutoInlineInjective(s)
    if len(s[x].op.axis) >= 4:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1], s[x].op.axis[2])
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 3:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1])
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 2:
        s[x].parallel(s[x].op.axis[0])
    return s
