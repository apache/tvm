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
from .utils import get_div


def schedule_adaptive_pool(outs, layout="NCHW"):
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

    def _schedule_global(Pool, layout):
        # examples of latest pool op is global max pool and non latest is global avg pooling
        # OL - an Expr will be used for rfactor
        # Out - programming of the parallelizm on the global level
        # shared is not required, local could be enough but shared scope gives quite significant
        # perf boost
        if Pool.op in s.outputs:
            Out = Pool
            OL = s.cache_write(Pool, "shared")
        else:
            Out = outs[0].op.output(0)
            s[Pool].set_scope("shared")
            OL = Pool

        PaddedInput = Pool.op.input_tensors[0]

        # detect axis for later reorder and binding of batch/channel to blocks and
        # spatial to threads
        if layout in ("NCHW", "NCHW4c"):
            channel_index = 1
            height_index = 2
            width_index = 3
        else:
            channel_index = 3
            height_index = 1
            width_index = 2

        if isinstance(PaddedInput.op, tvm.te.ComputeOp):
            s[PaddedInput].compute_inline()

        fused_reduce = s[OL].fuse(*s[OL].op.reduce_axis)

        spatial = PaddedInput.shape[height_index].value * PaddedInput.shape[width_index].value
        # below values were selected empirically assuming that we should have some work in each
        # thread (currently from 25-49) and number of threads not exceeding some threshold that
        # was selected as 256 from performance point of view after experiments on Adreno 660
        max_threads = spatial // 25 if spatial > 25 else 1
        max_threads = 256 if max_threads > 256 else max_threads
        num_thread = get_div(spatial, max_threads)

        thread_y = te.thread_axis((0, num_thread), "threadIdx.y")

        _, ki = s[OL].split(fused_reduce, factor=num_thread)
        data_out_rf = s.rfactor(OL, ki)
        s[data_out_rf].compute_at(s[OL], s[OL].op.reduce_axis[0])
        s[OL].bind(s[OL].op.reduce_axis[0], thread_y)

        naxis = s[Out].op.axis[0]
        caxis = s[Out].op.axis[channel_index]
        haxis = s[Out].op.axis[height_index]
        waxis = s[Out].op.axis[width_index]

        if layout in ("NHWC4c", "NCHW4c"):
            texture_axis = s[Out].op.axis[-1]
            s[Out].reorder(naxis, caxis, haxis, waxis, texture_axis)
            s[Out].vectorize(texture_axis)
        else:
            texture_axis = None
            s[Out].reorder(naxis, caxis, haxis, waxis)

        bx = s[Out].fuse(naxis, caxis, haxis, waxis)
        s[Out].bind(bx, te.thread_axis("blockIdx.x"))

        s[OL].compute_at(s[Out], bx)

    scheduled_ops = []

    def traverse(OP):
        """Internal traverse function"""
        # inline all one-to-one-mapping operators except the last stage (output)
        if tag.is_injective(OP.tag):
            if OP not in s.outputs:
                s[OP].compute_inline()
            for tensor in OP.input_tensors:
                if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        # schedule global_pool
        elif OP.tag.startswith("adaptive_pool"):
            Pool = OP.output(0)
            _schedule_global(Pool, layout)
        else:
            raise RuntimeError(f"Unsupported operator: {OP.tag}")

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s


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
            raise RuntimeError(f"Unsupported operator: {OP.tag}")

        scheduled_ops.append(OP)

    traverse(outs[0].op)
    return s
