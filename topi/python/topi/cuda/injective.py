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
import tvm
from .. import generic, util

def _schedule_injective(op, sch):
    x = op.output(0)
    fused = sch[x].fuse(*sch[x].op.axis)
    num_thread = tvm.target.current_target(allow_none=False).max_num_threads
    max_block = 256

    try:
        const_size = util.get_const_int(util.prod(x.shape))
        max_block = 256
        need_block_split = const_size > max_block * num_thread
    except ValueError:
        need_block_split = False

    if need_block_split:
        xo, xi = sch[x].split(fused, factor=num_thread * max_block)
        bx, tx = sch[x].split(xi, factor=num_thread)
        sch[x].reorder(bx, tx, xo)
        sch[x].bind(bx, tvm.thread_axis("blockIdx.x"))
        sch[x].bind(tx, tvm.thread_axis("threadIdx.x"))
    else:
        bx, tx = sch[x].split(fused, factor=num_thread)
        sch[x].bind(tx, tvm.thread_axis("threadIdx.x"))
        sch[x].bind(bx, tvm.thread_axis("blockIdx.x"))

    return sch


@generic.schedule_injective.register(["cuda", "gpu"])
def schedule_injective(outs):
    """Schedule for injective op.

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

    tvm.schedule.AutoInlineInjective(s)
    for out in outs:
        _schedule_injective(out.op, s)
    return s

schedule_elemwise = schedule_injective
schedule_broadcast = schedule_injective
