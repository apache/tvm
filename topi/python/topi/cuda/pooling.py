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
# pylint: disable=invalid-name, unused-variable, unused-argument
"""Schedule for pooling operators"""
import tvm
from tvm import te
from .. import tag
from ..util import traverse_inline


def schedule_adaptive_pool(outs, layout='NCHW'):
    """Schedule for adaptive_pool.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of adaptive_pool
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for adaptive_pool.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule(Pool):
        num_thread = 8
        block_x = te.thread_axis("blockIdx.x")
        block_y = te.thread_axis("blockIdx.y")
        thread_x = te.thread_axis((0, num_thread), "threadIdx.x")
        thread_y = te.thread_axis((0, num_thread), "threadIdx.y")
        if Pool.op in s.outputs:
            Out = Pool
            OL = s.cache_write(Pool, "local")
        else:
            Out = outs[0].op.output(0)
            s[Pool].set_scope("local")

        by, ty = s[Out].split(s[Out].op.axis[0], factor=num_thread)
        if layout == 'NHWC':
            bx, tx = s[Out].split(s[Out].op.axis[3], factor=num_thread)
        else:
            bx, tx = s[Out].split(s[Out].op.axis[1], factor=num_thread)
        s[Out].reorder(by, bx, ty, tx)
        s[Out].bind(ty, thread_y)
        s[Out].bind(tx, thread_x)
        s[Out].bind(by, block_y)
        s[Out].bind(bx, block_x)
        if Pool.op in s.outputs:
            s[OL].compute_at(s[Out], tx)
        else:
            s[Pool].compute_at(s[Out], tx)

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
        # schedule global_pool
        elif OP.tag.startswith('adaptive_pool'):
            Pool = OP.output(0)
            _schedule(Pool)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s


def schedule_pool(outs, layout):
    """Schedule for pool.

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
        if Pool.op in s.outputs:
            Out = Pool
            OL = s.cache_write(Pool, "local")
        else:
            Out = outs[0].op.output(0)
            s[Pool].set_scope("local")
        fused = s[Out].fuse(*s[Out].op.axis)
        bx, tx = s[Out].split(fused, factor=num_thread)
        s[Out].bind(bx, te.thread_axis("blockIdx.x"))
        s[Out].bind(tx, te.thread_axis("threadIdx.x"))
        if Pool.op in s.outputs:
            s[OL].compute_at(s[Out], tx)
        else:
            s[Pool].compute_at(s[Out], tx)

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
        elif OP.tag.startswith('pool'):
            PaddedInput = OP.input_tensors[0]
            Pool = OP.output(0)
            _schedule(PaddedInput, Pool)
        else:
            raise RuntimeError("Unsupported operator: %s" % OP.tag)

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s


def schedule_pool_grad(outs):
    """Schedule for pool_grad on CUDA

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of pool_grad
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for pool_grad.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule_pool_grad(op):
        if op in s.outputs:
            out = op
        else:
            out = outs[0].op.output(0)
        fused = s[out].fuse(*s[out].op.axis)
        num_thread = tvm.target.Target.current(allow_none=False).max_num_threads
        bx, tx = s[out].split(fused, factor=num_thread)
        s[out].bind(bx, te.thread_axis("blockIdx.x"))
        s[out].bind(tx, te.thread_axis("threadIdx.x"))

        if tag.COMM_REDUCE_IDX in op.input_tensors[0].op.tag:
            max_pool_index = op.input_tensors[0]
            s[max_pool_index].compute_at(s[out], tx)

            pool_input = max_pool_index.op.input_tensors[0]
            if isinstance(pool_input.op, tvm.te.ComputeOp):
                # handle padding
                s[pool_input].compute_inline()
        if op not in s.outputs:
            s[op].compute_at(s[out], tx)

    def _callback(op):
        if op.tag.startswith('pool_grad'):
            _schedule_pool_grad(op)

    traverse_inline(s, outs[0].op, _callback)

    return s
