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
# pylint: disable=invalid-name, too-many-locals, too-many-statements
"""Schedule for conv2d_hwcn with auto fusion"""
import tvm
from .. import tag

def schedule_conv2d_hwcn(outs):
    """Schedule for conv2d_hwcn and any element-wise operations.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of conv2d_hwcn in the format
        of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for conv2d_hwcn.
    """
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    sch = tvm.create_schedule([x.op for x in outs])
    def schedule(Apad, W, B):
        """Schedule conv2d_hwcn"""
        sch[Apad].compute_inline()
        AA = sch.cache_read(Apad, "shared", [B])
        WW = sch.cache_read(W, "shared", [B])
        AL = sch.cache_read(AA, "local", [B])
        WL = sch.cache_read(WW, "local", [B])

        if B.op in sch.outputs:
            Out = B
            BL = sch.cache_write(Out, "local")
        else:
            Out = sch.outputs[0].output(0)
            sch[B].set_scope("local")
            BL = B

        tile = 8
        num_thread = 8
        block_factor = tile * num_thread
        step = 8
        vthread = 2

        block_x = tvm.thread_axis("blockIdx.x")
        block_y = tvm.thread_axis("blockIdx.y")
        block_z = tvm.thread_axis("blockIdx.z")
        thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
        thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
        thread_xz = tvm.thread_axis((0, vthread), "vthread", name="vx")
        thread_yz = tvm.thread_axis((0, vthread), "vthread", name="vy")

        hi, wi, fi, ni = sch[Out].op.axis
        bz = sch[Out].fuse(hi, wi)
        by, fi = sch[Out].split(fi, factor=block_factor)
        bx, ni = sch[Out].split(ni, factor=block_factor)
        tyz, fi = sch[Out].split(fi, nparts=vthread)
        txz, ni = sch[Out].split(ni, nparts=vthread)
        ty, fi = sch[Out].split(fi, nparts=num_thread)
        tx, ni = sch[Out].split(ni, nparts=num_thread)
        sch[Out].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)
        sch[Out].bind(bz, block_z)
        sch[Out].bind(by, block_y)
        sch[Out].bind(bx, block_x)
        sch[Out].bind(tyz, thread_yz)
        sch[Out].bind(txz, thread_xz)
        sch[Out].bind(ty, thread_y)
        sch[Out].bind(tx, thread_x)

        # Schedule BL local write
        sch[BL].compute_at(sch[Out], tx)
        yi, xi, fi, ni = sch[BL].op.axis
        ry, rx, rc = sch[BL].op.reduce_axis
        rco, rci = sch[BL].split(rc, factor=step)
        sch[BL].reorder(rco, ry, rx, rci, fi, ni)
        fuse_index = sch[BL].fuse(ry, rx)
        fuse_index = sch[BL].fuse(fuse_index, rco)
        rx = fuse_index

        sch[AA].compute_at(sch[BL], rx)
        sch[WW].compute_at(sch[BL], rx)
        sch[AL].compute_at(sch[BL], rci)
        sch[WL].compute_at(sch[BL], rci)
        # Schedule for A's shared memory load
        yi, xi, ci, ni = sch[AA].op.axis
        ty, ci = sch[AA].split(ci, nparts=num_thread)
        tx, ni = sch[AA].split(ni, nparts=num_thread)
        _, ni = sch[AA].split(ni, factor=4)
        sch[AA].reorder(ty, tx, yi, xi, ci, ni)
        sch[AA].bind(ty, thread_y)
        sch[AA].bind(tx, thread_x)
        sch[AA].vectorize(ni)
        # Schedule for W's shared memory load
        yi, xi, ci, fi = sch[WW].op.axis
        ty, ci = sch[WW].split(ci, nparts=num_thread)
        tx, fi = sch[WW].split(fi, nparts=num_thread)
        _, fi = sch[WW].split(fi, factor=4)
        sch[WW].reorder(ty, tx, yi, xi, ci, fi)
        sch[WW].bind(ty, thread_y)
        sch[WW].bind(tx, thread_x)
        sch[WW].vectorize(fi)

    scheduled_ops = []

    def traverse(operator):
        """Traverse operators from computation graph"""
        if tag.is_broadcast(operator.tag):
            if operator not in sch.outputs:
                sch[operator].compute_inline()
            for tensor in operator.input_tensors:
                if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        elif operator.tag == 'conv2d_hwcn':
            Apad = operator.input_tensors[0]
            W = operator.input_tensors[1]
            if isinstance(W.op, tvm.tensor.ComputeOp) and 'dilate' in W.op.tag:
                sch[W].compute_inline()
            B = operator.output(0)
            schedule(Apad, W, B)
        else:
            raise RuntimeError("Unsupported operator: %s" % operator.tag)

        scheduled_ops.append(operator)

    traverse(outs[0].op)
    return sch
