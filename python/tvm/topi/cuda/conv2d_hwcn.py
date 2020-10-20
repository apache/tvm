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
# pylint: disable=invalid-name, too-many-locals, too-many-statements, unused-argument
"""Schedule for conv2d_hwcn with auto fusion"""
import tvm
from tvm import te
from tvm import autotvm

from tvm.autotvm.task.space import SplitEntity

from .. import nn, tag


@autotvm.register_topi_compute("conv2d_hwcn.cuda")
def conv2d_hwcn(cfg, data, kernel, strides, padding, dilation, out_dtype="float32"):
    """Compute conv2d with HWCN layout on CUDA"""
    return nn.conv2d_hwcn(data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv2d_hwcn.cuda")
def schedule_conv2d_hwcn(cfg, outs):
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
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    sch = te.create_schedule([x.op for x in outs])

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

        hi, wi, fi, ni = sch[Out].op.axis

        # Create tuning space
        n_thread_cand = [1, 2, 4, 8, 16, 32]
        vthread_cand = [1, 2, 4, 8]

        cfg.define_split(
            "tile_fi",
            fi,
            num_outputs=4,
            filter=lambda x: (x.size[1] in vthread_cand and x.size[2] in n_thread_cand),
        )
        cfg.define_split(
            "tile_ni",
            ni,
            num_outputs=4,
            filter=lambda x: (x.size[1] in vthread_cand and x.size[2] in n_thread_cand),
        )

        if cfg.is_fallback:
            cfg["tile_fi"] = SplitEntity([-1, 2, 8, 4])
            cfg["tile_ni"] = SplitEntity([-1, 2, 8, 4])

        # Scheduling
        step = 8

        bz = sch[Out].fuse(hi, wi)
        by, tyz, ty, fi = cfg["tile_fi"].apply(sch, Out, fi)
        bx, txz, tx, ni = cfg["tile_ni"].apply(sch, Out, ni)
        sch[Out].reorder(bz, by, bx, tyz, txz, ty, tx, fi, ni)

        sch[Out].bind(bz, te.thread_axis("blockIdx.z"))
        sch[Out].bind(by, te.thread_axis("blockIdx.y"))
        sch[Out].bind(bx, te.thread_axis("blockIdx.x"))
        sch[Out].bind(tyz, te.thread_axis("vthread"))
        sch[Out].bind(txz, te.thread_axis("vthread"))
        sch[Out].bind(ty, te.thread_axis("threadIdx.y"))
        sch[Out].bind(tx, te.thread_axis("threadIdx.x"))

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
        ty, ci = sch[AA].split(ci, nparts=cfg["tile_fi"].size[2])
        tx, ni = sch[AA].split(ni, nparts=cfg["tile_ni"].size[2])
        _, ni = sch[AA].split(ni, factor=4)
        sch[AA].reorder(ty, tx, yi, xi, ci, ni)
        sch[AA].bind(ty, te.thread_axis("threadIdx.y"))
        sch[AA].bind(tx, te.thread_axis("threadIdx.x"))
        sch[AA].vectorize(ni)
        # Schedule for W's shared memory load
        yi, xi, ci, fi = sch[WW].op.axis
        ty, ci = sch[WW].split(ci, nparts=cfg["tile_fi"].size[2])
        tx, fi = sch[WW].split(fi, nparts=cfg["tile_ni"].size[2])
        _, fi = sch[WW].split(fi, factor=4)
        sch[WW].reorder(ty, tx, yi, xi, ci, fi)
        sch[WW].bind(ty, te.thread_axis("threadIdx.y"))
        sch[WW].bind(tx, te.thread_axis("threadIdx.x"))
        sch[WW].vectorize(fi)

    scheduled_ops = []

    def traverse(operator):
        """Traverse operators from computation graph"""
        if tag.is_broadcast(operator.tag):
            if operator not in sch.outputs:
                sch[operator].compute_inline()
            for tensor in operator.input_tensors:
                if isinstance(tensor.op, te.tensor.ComputeOp) and tensor.op not in scheduled_ops:
                    traverse(tensor.op)
        elif operator.tag == "conv2d_hwcn":
            Apad = operator.input_tensors[0]
            W = operator.input_tensors[1]
            if isinstance(W.op, tvm.te.ComputeOp) and "dilate" in W.op.tag:
                sch[W].compute_inline()
            B = operator.output(0)
            schedule(Apad, W, B)
        else:
            raise RuntimeError("Unsupported operator: %s" % operator.tag)

        scheduled_ops.append(operator)

    traverse(outs[0].op)
    return sch
