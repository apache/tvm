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
from .. import generic

@generic.schedule_injective.register(["arm_cpu"])
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
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    x = outs[0]
    if list(s[x].op.axis):
        # do not vectorize for broadcast
        (io, ii) = s[x].split(list(s[x].op.axis)[-1], 8)
        s[x].vectorize(ii)
    tvm.schedule.AutoInlineInjective(s)
    if len(s[x].op.axis) >= 4:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1], s[x].op.axis[2])
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 3:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1])
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 2:
        s[x].parallel(s[x].op.axis[0])
    return s

@generic.schedule_concatenate.register(["arm_cpu"])
def schedule_concatenate(outs):
    """Schedule for concatenate op.

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
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])
    x = outs[0]
    tvm.schedule.AutoInlineInjective(s)
    if len(s[x].op.axis) >= 4:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1], s[x].op.axis[2])
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 3:
        fused = s[x].fuse(s[x].op.axis[0], s[x].op.axis[1])
        s[x].parallel(fused)
    elif len(s[x].op.axis) >= 2:
        s[x].parallel(s[x].op.axis[0])
    return s
