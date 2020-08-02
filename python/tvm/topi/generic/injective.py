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
"""generic declaration and schedules."""
from __future__ import absolute_import as _abs

import tvm
from tvm import te

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
    sch[out].fuse(*sch[out].op.axis)
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
    target = tvm.target.Target.current(allow_none=False)
    if target.id.name != "llvm":
        raise RuntimeError("schedule_injective not registered for '%s'" % target)
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    x = outs[0]
    s = te.create_schedule([x.op for x in outs])
    te.schedule.AutoInlineInjective(s)
    schedule_injective_from_existing(s, x)
    return s

schedule_elemwise = schedule_injective
schedule_broadcast = schedule_injective
