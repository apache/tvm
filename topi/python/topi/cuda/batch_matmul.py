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
# pylint: disable=invalid-name,too-many-locals,unused-variable,unused-argument
"""cuda batch_matmul operators"""
import tvm
from tvm import autotvm
from tvm import te
from tvm.contrib import cublas
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity
from .. import nn
from ..util import traverse_inline, get_const_tuple, get_max_power2_factor

@autotvm.register_topi_compute("batch_matmul.cuda")
def batch_matmul(cfg, x, y):
    """Compute conv2d with NCHW layout"""
    return nn.batch_matmul(x, y)


@autotvm.register_topi_schedule("batch_matmul.cuda")
def schedule_batch_matmul(cfg, outs):
    """Schedule for batch_matmul

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of batch_matmul
          in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _schedule(cfg, op):
        C = op.output(0)
        A, B = s[C].op.input_tensors
        _, M, N = get_const_tuple(C.shape)
        AA = s.cache_read(A, "shared", [C])
        AL = s.cache_read(AA, "local", [C])
        BB = s.cache_read(B, "shared", [C])
        BL = s.cache_read(BB, "local", [C])
        CC = s.cache_write(C, "local")
        if op not in s.outputs:
            s[C].compute_inline()
            C = s.outputs[0].output(0)

        b, y, x = s[C].op.axis
        k, = s[CC].op.reduce_axis

        cfg.define_split("tile_y", y, num_outputs=3)
        cfg.define_split("tile_x", x, num_outputs=3)
        cfg.define_split("tile_k", k, num_outputs=2)
        cfg.define_knob("auto_unroll_max_step", [8, 16, 32, 64])
        target = tvm.target.Target.current()
        if target.id.name in ['nvptx', 'rocm']:
            # llvm-based backends cannot do non-explicit unrolling
            cfg.define_knob("unroll_explicit", [1])
        else:
            cfg.define_knob("unroll_explicit", [0, 1])

        if cfg.is_fallback:
            y_bn = get_max_power2_factor(M, 64)
            x_bn = get_max_power2_factor(N, 64)
            y_nthreads = min(y_bn, 8)
            x_nthreads = min(x_bn, 8)
            cfg['tile_x'] = SplitEntity([-1, x_nthreads, x_bn // x_nthreads])
            cfg['tile_y'] = SplitEntity([-1, y_nthreads, y_bn // y_nthreads])
            cfg['tile_k'] = SplitEntity([-1, 8])
            cfg['auto_unroll_max_step'] = OtherOptionEntity(16)

        by, ty, yi = cfg["tile_y"].apply(s, C, y)
        bx, tx, xi = cfg["tile_x"].apply(s, C, x)

        thread_x = te.thread_axis("threadIdx.x")
        thread_y = te.thread_axis("threadIdx.y")

        s[C].reorder(b, by, bx, ty, tx, yi, xi)
        s[C].bind(b, te.thread_axis("blockIdx.z"))
        s[C].bind(by, te.thread_axis("blockIdx.y"))
        s[C].bind(bx, te.thread_axis("blockIdx.x"))
        s[C].bind(ty, thread_y)
        s[C].bind(tx, thread_x)
        s[C].pragma(yi, "auto_unroll_max_step", cfg['auto_unroll_max_step'].val)
        s[C].pragma(yi, 'unroll_explicit', cfg['unroll_explicit'].val)

        s[CC].compute_at(s[C], tx)
        _, yi, xi = s[CC].op.axis
        ko, ki = cfg["tile_k"].apply(s, CC, k)
        s[CC].reorder(ko, ki, yi, xi)
        s[CC].pragma(ki, "auto_unroll_max_step", cfg['auto_unroll_max_step'].val)
        s[CC].pragma(ki, 'unroll_explicit', cfg['unroll_explicit'].val)

        s[AA].compute_at(s[CC], ko)
        s[AL].compute_at(s[CC], ki)
        s[BB].compute_at(s[CC], ko)
        s[BL].compute_at(s[CC], ki)
        _, y, k = s[AA].op.axis
        ty, yi = s[AA].split(y, nparts=cfg["tile_y"].size[1])
        tx, ki = s[AA].split(k, nparts=cfg["tile_x"].size[1])
        s[AA].reorder(ty, tx, yi, ki)
        s[AA].bind(ty, thread_y)
        s[AA].bind(tx, thread_x)
        s[AA].pragma(yi, "auto_unroll_max_step", cfg['auto_unroll_max_step'].val)
        s[AA].pragma(yi, 'unroll_explicit', cfg['unroll_explicit'].val)

        _, x, k = s[BB].op.axis
        ty, xi = s[BB].split(x, nparts=cfg["tile_y"].size[1])
        tx, ki = s[BB].split(k, nparts=cfg["tile_x"].size[1])
        s[BB].bind(ty, thread_y)
        s[BB].bind(tx, thread_x)
        s[BB].reorder(ty, tx, xi, ki)
        s[BB].pragma(xi, "auto_unroll_max_step", cfg['auto_unroll_max_step'].val)
        s[BB].pragma(xi, 'unroll_explicit', cfg['unroll_explicit'].val)

    def _callback(op):
        if "batch_matmul" in op.tag:
            _schedule(cfg, op)

    traverse_inline(s, outs[0].op, _callback)
    return s

def batch_matmul_cublas(x, y):
    """Computes batch matrix multiplication of `x` and `y` when `x` and `y` are
    data in batch.

    Parameters
    ----------
    x : tvm.te.Tensor
        3-D with shape [batch, M, K]

    y : tvm.te.Tensor
        3-D with shape [batch, N, K]

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    return cublas.batch_matmul(x, y, False, True)
