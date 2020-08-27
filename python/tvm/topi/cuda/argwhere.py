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
# pylint: disable=too-many-arguments
"""Argwhere operator"""

import tvm
from tvm import te
from ..util import traverse_inline

def schedule_argwhere(outs):
    """Schedule for argwhere on cuda.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of argwhere
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for argwhere
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    sch = te.create_schedule([x.op for x in outs])

    def _schedule_argwhere(op):
        if op in sch.outputs:
            out = op
        else:
            out = outs[0].op.output(0)
        fused = sch[out].fuse(*sch[out].op.axis)
        num_thread = tvm.target.Target.current(allow_none=False).max_num_threads
        bx, tx = sch[out].split(fused, factor=num_thread)
        sch[out].bind(bx, te.thread_axis("blockIdx.x"))
        sch[out].bind(tx, te.thread_axis("threadIdx.x"))

    traverse_inline(sch, outs[0].op, _schedule_argwhere)

    return sch
