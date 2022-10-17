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
# pylint: disable=invalid-name,unused-variable,unused-argument,no-else-return
"""pooling schedules for Qualcomm Adreno GPU"""
import tvm
from tvm import te
from .. import tag


def schedule_pool(outs, layout):
    """Schedule for various pooling operators.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of pool
        in the format of an array of tensors.

    layout: str
        Data layout.

    Returns
    -------
    s: Schedule
        The computation schedule for pool.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule(PaddedInput, Pool):
        if isinstance(PaddedInput.op, tvm.te.ComputeOp):
            s[PaddedInput].compute_inline()
        num_thread = tvm.target.Target.current(allow_none=False).max_num_threads
        num_thread = int(num_thread * 2)
        if Pool.op in s.outputs:
            Out = Pool
            OL = s.cache_write(Pool, "local")
        else:
            Out = outs[0].op.output(0)
            s[Pool].set_scope("local")
        fused = s[Out].fuse(*s[Out].op.axis[:-1])
        bx, tx = s[Out].split(fused, factor=num_thread)
        s[Out].bind(bx, te.thread_axis("blockIdx.x"))
        s[Out].bind(tx, te.thread_axis("threadIdx.x"))
        s[Out].vectorize(s[Out].op.axis[-1])
        if Pool.op in s.outputs:
            s[OL].compute_at(s[Out], tx)
            s[OL].vectorize(s[OL].op.axis[-1])
        else:
            s[Pool].compute_at(s[Out], tx)
            s[Pool].vectorize(s[Pool].op.axis[-1])

    scheduled_ops = []

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_broadcast(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule pool
        elif OP.tag.startswith("pool"):
            PaddedInput = OP.input_tensors[0]
            Pool = OP.output(0)
            _schedule(PaddedInput, Pool)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s
