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
"""Direct conv2d in NHWC layout"""
import tvm
from tvm import te
from tvm import autotvm
from ..util import get_const_tuple


def schedule_conv2d_nhwc_direct(cfg, s, Conv):
    """schedule optimized for NHWC direct conv2d"""
    pad_data, kernel = s[Conv].op.input_tensors
    s[pad_data].compute_inline()

    if isinstance(kernel.op, tvm.te.ComputeOp) and 'dilate' in kernel.op.tag:
        s[kernel].compute_inline()

    if Conv.op in s.outputs:
        output = Conv
        OL = s.cache_write(Conv, 'local')
    else:
        output = s.outputs[0].output(0)
        s[Conv].set_scope('local')
        OL = Conv
    # create cache stage
    AA = s.cache_read(pad_data, 'shared', [OL])
    WW = s.cache_read(kernel, "shared", [OL])
    AL = s.cache_read(AA, "local", [OL])
    WL = s.cache_read(WW, "local", [OL])

    # Schedule for autotvm
    cfg.define_knob("tile_n", [2, 4, 8])
    cfg.define_knob("tile_c", [2, 4, 8])
    cfg.define_knob("num_thread_n", [4, 8, 16])
    cfg.define_knob("num_thread_c", [4, 8, 16])
    cfg.define_knob("vthread_n", [1, 2])
    cfg.define_knob("vthread_c", [1, 2])
    cfg.define_knob("step", [16, 3, 32, 64])

    # fallback support
    target = tvm.target.Target.current()
    if cfg.is_fallback:
        ref_log = autotvm.tophub.load_reference_log(
            target.id.name, target.model, 'conv2d_nhwc.cuda')
        cfg.fallback_with_reference_log(ref_log)

    tile_n = cfg["tile_n"].val
    tile_c = cfg["tile_c"].val
    num_thread_n = cfg["num_thread_n"].val
    num_thread_c = cfg["num_thread_c"].val
    vthread_n = cfg["vthread_n"].val
    vthread_c = cfg["vthread_c"].val
    step = cfg["step"].val
    block_factor_c = tile_c * num_thread_c * vthread_c

    offset = 8
    A_align = step + offset
    W_align = block_factor_c + offset

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis((0, num_thread_c), "threadIdx.x")
    thread_y = te.thread_axis((0, num_thread_n), "threadIdx.y")
    thread_xz = te.thread_axis((0, vthread_c), "vthread", name="vx")
    thread_yz = te.thread_axis((0, vthread_n), "vthread", name="vy")

    # Schedule for output
    ni, hi, wi, fi = s[output].op.axis
    bz = s[output].fuse(hi, wi)
    tx, fi = s[output].split(fi, factor=tile_c)
    txz, tx = s[output].split(tx, factor=num_thread_c)
    bx, txz = s[output].split(txz, factor=vthread_c)
    ty, ni = s[output].split(ni, factor=tile_n)
    tyz, ty = s[output].split(ty, factor=num_thread_n)
    by, tyz = s[output].split(tyz, factor=vthread_n)
    s[output].reorder(bz, by, bx, tyz, txz, ty, tx, ni, fi)
    s[output].bind(bz, block_z)
    s[output].bind(by, block_y)
    s[output].bind(bx, block_x)
    s[output].bind(tyz, thread_yz)
    s[output].bind(txz, thread_xz)
    s[output].bind(ty, thread_y)
    s[output].bind(tx, thread_x)
    # Schedule local computation
    s[OL].compute_at(s[output], tx)
    ni, yi, xi, fi = s[OL].op.axis
    ry, rx, rc = s[OL].op.reduce_axis
    rco, rci = s[OL].split(rc, factor=step)
    s[OL].reorder(rco, ry, rx, rci, ni, fi)

    s[AA].compute_at(s[OL], rx)
    s[WW].compute_at(s[OL], rx)
    s[AL].compute_at(s[OL], rci)
    s[WL].compute_at(s[OL], rci)
    # Schedule for data's share memory
    ni, yi, xi, ci = s[AA].op.axis
    s[AA].reorder(yi, xi, ni, ci)
    s[AA].storage_align(xi, A_align - 1, A_align)
    t = s[AA].fuse(ni, ci)
    ty, tx = s[AA].split(t, factor=num_thread_c)
    _, ty = s[AA].split(ty, factor=num_thread_n)
    s[AA].bind(tx, thread_x)
    s[AA].bind(ty, thread_y)
    # Schedule for kernel's share memory
    _, _, ic, o = s[WW].op.axis
    t = s[WW].fuse(ic, o)
    s[WW].storage_align(ic, W_align - 1, W_align)
    ty, tx = s[WW].split(t, factor=num_thread_c)
    _, ty = s[WW].split(ty, factor=num_thread_n)
    s[WW].bind(tx, thread_x)
    s[WW].bind(ty, thread_y)

    N, OH, OW, CO = get_const_tuple(output.shape)
    KH, KW, CI, _ = get_const_tuple(kernel.shape)
    cfg.add_flop(2 * N * OH * OW * CO * CI * KH * KW)
