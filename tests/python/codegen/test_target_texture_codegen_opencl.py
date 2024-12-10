# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# 'License'); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# 'AS IS' BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import sys

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import autotvm
from tvm import te
from tvm.topi import testing
from tvm.topi.utils import get_const_tuple, simplify
from tvm.topi import nn


def compute_plus_one_rank3(shape):
    X = te.placeholder(shape, name="X", dtype="float32")
    Y = te.compute(shape, lambda i, j, k: X[i, j, k] + 1, name="Compute_Y")
    return X, Y


def schedule_plus_one_rank3(X, Y):
    s = te.create_schedule(Y.op)
    # Xt = s.cache_read(X, "texture", [Y])
    # Xt = s.cache_read(X, "global", [Y])
    Xt = s.cache_read(X, "global.texture", [Y])

    # copy to texture stage
    x, y, c = s[Xt].op.axis
    s[Xt].bind(x, te.thread_axis("blockIdx.x"))
    s[Xt].bind(y, te.thread_axis("threadIdx.x"))
    s[Xt].vectorize(c)

    # the compute stage
    x, y, c = s[Y].op.axis
    xo, yo, xi, yi = s[Y].tile(x, y, 4, 4)
    s[Y].bind(xo, te.thread_axis("blockIdx.x"))
    s[Y].bind(yo, te.thread_axis("threadIdx.x"))
    s[Y].vectorize(c)
    return s


def compute_plus_one_rank5(shape):
    X = te.placeholder(shape, name="X", dtype="float32")
    Y = te.compute(shape, lambda i, j, k, l, m: X[i, j, k, l, m] + 1, name="Compute_Y")
    return X, Y


def schedule_plus_one_rank5(X, Y):
    s = te.create_schedule(Y.op)
    Xt = s.cache_read(X, "global.texture", [Y])

    # copy to texture stage
    a, b, c, d, e = s[Xt].op.axis
    abc = s[Xt].fuse(a, b, c)
    s[Xt].bind(abc, te.thread_axis("blockIdx.x"))
    s[Xt].bind(d, te.thread_axis("threadIdx.x"))
    s[Xt].vectorize(e)

    # the compute stage
    a, b, c, d, e = s[Y].op.axis
    abc = s[Y].fuse(a, b, c)
    xo, yo, xi, yi = s[Y].tile(abc, d, 4, 4)
    s[Y].bind(xo, te.thread_axis("blockIdx.x"))
    s[Y].bind(yo, te.thread_axis("threadIdx.x"))
    s[Y].vectorize(e)
    return s


def compute_matmul(shape):
    A = te.placeholder(shape, name="A", dtype="float32")
    B = te.placeholder(shape, name="B", dtype="float32")
    k = te.reduce_axis((0, shape[1]), name="k")
    C = te.compute(
        (shape[0] * shape[2], shape[0] * shape[2]),
        lambda i, j: te.sum(
            A[i // shape[2], k, i % shape[2]].astype("float32")
            * B[j // shape[2], k, j % shape[2]].astype("float32"),
            axis=[k],
        ),
        name="Compute_MatMul",
    )
    return A, B, C


def schedule_matmul(A, B, C, local=False):
    s = te.create_schedule(C.op)
    At = s.cache_read(A, "global.texture", [C])
    Bt = s.cache_read(B, "global.texture", [C])
    if local:
        Al = s.cache_read(At, "local", [C])
        Bl = s.cache_read(Bt, "local", [C])
    Cl = s.cache_write(C, "local")

    bx = te.thread_axis("blockIdx.x")
    tx = te.thread_axis("threadIdx.x")

    def copy_to_texture(stage):
        _io, _k, _ii = s[stage].op.axis
        s[stage].vectorize(_ii)
        s[stage].bind(_io, bx)
        s[stage].bind(_k, tx)

    copy_to_texture(At)
    copy_to_texture(Bt)

    # copy to global stage
    _i, _j = s[C].op.axis
    xo, yo, xi, yi = s[C].tile(_i, _j, 4, 4)
    s[C].unroll(xi)
    s[C].vectorize(yi)
    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(yo, te.thread_axis("threadIdx.x"))

    # the compute stage
    s[Cl].compute_at(s[C], yo)
    (_k,) = Cl.op.reduce_axis
    _x, _y = s[Cl].op.axis
    s[Cl].reorder(_k, _x, _y)
    s[Cl].unroll(_x)
    s[Cl].vectorize(_y)

    if local:
        s[Al].compute_at(s[Cl], _k)
        s[Al].vectorize(s[Al].op.axis[-1])
        s[Bl].compute_at(s[Cl], _k)
        s[Bl].vectorize(s[Bl].op.axis[-1])

    return s


def compute_matmul_inner(shape):
    A = te.placeholder(shape, name="A", dtype="float32")
    B = te.placeholder(shape, name="B", dtype="float32")
    k = te.reduce_axis((0, shape[1] * shape[2]), name="k")
    # (M, K) x (N, K)
    # (32, 256) x (32, 256)
    # (32, 64, 4) x (32, 64, 4)
    C = te.compute(
        (shape[0], shape[0]),
        lambda i, j: te.sum(
            A[i, k // shape[2], k % shape[2]].astype("float32")
            * B[j, k // shape[2], k % shape[2]].astype("float32"),
            axis=[k],
        ),
        name="Compute_MatMul",
    )
    return A, B, C


def schedule_matmul_inner(A, B, C, local=False):
    s = te.create_schedule(C.op)
    At = s.cache_read(A, "global.texture", [C])
    Bt = s.cache_read(B, "global.texture", [C])
    if local:
        Al = s.cache_read(At, "local", [C])
        Bl = s.cache_read(Bt, "local", [C])
    Cl = s.cache_write(C, "local")

    bx = te.thread_axis("blockIdx.x")
    tx = te.thread_axis("threadIdx.x")

    def copy_to_texture(stage):
        _i, _ko, _ki = s[stage].op.axis
        s[stage].vectorize(_ki)
        s[stage].bind(_i, bx)
        s[stage].bind(_ko, tx)

    copy_to_texture(At)
    copy_to_texture(Bt)

    # copy to global stage
    _i, _j = s[C].op.axis
    xo, yo, xi, yi = s[C].tile(_i, _j, 4, 4)
    s[C].unroll(xi)
    s[C].vectorize(yi)
    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(yo, te.thread_axis("threadIdx.x"))

    # the compute stage
    s[Cl].compute_at(s[C], yo)
    (_k,) = Cl.op.reduce_axis
    _x, _y = s[Cl].op.axis
    s[Cl].reorder(_x, _y, _k)
    s[Cl].unroll(_x)
    # TODO(csullivan): consider whether the below error is worth resolving
    # s[Cl].vectorize(_y) # error

    if local:
        s[Al].compute_at(s[Cl], _x)
        s[Al].vectorize(s[Al].op.axis[-1])
        s[Bl].compute_at(s[Cl], _x)
        s[Bl].vectorize(s[Bl].op.axis[-1])

    return s


def compute_matmul_vector_accumulator(shapeA, shapeB):
    # A x B
    # (K/4, M, K%4) x (K, N/4, N%4) = (M, N)
    # (32, 64, 4) x (128, 16, 4) = (64, 64)
    A = te.placeholder(shapeA, name="A", dtype="float32")
    B = te.placeholder(shapeB, name="B", dtype="float32")
    k = te.reduce_axis((0, shapeB[0]), name="k")
    C = te.compute(
        (shapeA[1], shapeB[1] * shapeB[2]),
        lambda i, j: te.sum(
            A[k // shapeA[-1], i, k % shapeA[-1]].astype("float32")
            * B[k, j // shapeB[-1], j % shapeB[-1]].astype("float32"),
            axis=[k],
        ),
        name="Compute_MatMul",
    )
    return A, B, C


def schedule_matmul_vector_accumulator(A, B, C, local=False):
    s = te.create_schedule(C.op)
    At = s.cache_read(A, "global.texture", [C])
    Bt = s.cache_read(B, "global.texture", [C])
    if local:
        Al = s.cache_read(At, "local", [C])
        Bl = s.cache_read(Bt, "local", [C])
    Cl = s.cache_write(C, "local")

    def copy_to_texture(stage):
        _y, _x, _v = s[stage].op.axis
        # TODO(csullivan): removing this vectorize results in numerical errors, autovectorize
        s[stage].vectorize(_v)
        s[stage].bind(_y, te.thread_axis("blockIdx.x"))
        s[stage].bind(_x, te.thread_axis("threadIdx.x"))

    copy_to_texture(At)
    copy_to_texture(Bt)

    # copy to global stage
    _i, _j = s[C].op.axis
    xo, yo, xi, yi = s[C].tile(_i, _j, 4, 4)
    s[C].unroll(xi)
    s[C].vectorize(yi)
    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(yo, te.thread_axis("threadIdx.x"))

    # the compute stage
    s[Cl].compute_at(s[C], yo)
    (_k,) = Cl.op.reduce_axis
    _a, _b = s[Cl].op.axis
    _ko, _ki = s[Cl].split(_k, factor=4)
    s[Cl].reorder(_ko, _a, _ki, _b)
    s[Cl].unroll(_ki)
    s[Cl].unroll(_a)
    s[Cl].vectorize(_b)

    if local:
        s[Al].compute_at(s[Cl], _a)
        _aa, _ka, _ba = s[Al].op.axis
        # TODO(csullivan)[BEFORE PR]: removing this vectorize command causes a crash. This needs to be autovectorized.
        s[Al].vectorize(_ba)
        s[Bl].compute_at(s[Cl], _ko)
        _ab, _kb, _bb = s[Bl].op.axis
        s[Bl].vectorize(_bb)
        s[Bl].unroll(_ab)

    return s


def compute_conv2d_1x1_NCHWc_RSCKk(input_shape, filter_shape):
    # conv2d( [N, C, H, W, c] , [1, 1, C, K, k]
    data = te.placeholder(input_shape, name="data", dtype="float32")
    filt = te.placeholder(filter_shape, name="filter", dtype="float32")
    c = te.reduce_axis((0, input_shape[1]), name="C")
    c4 = te.reduce_axis((0, input_shape[-1]), name="c4")
    kh = te.reduce_axis((0, filter_shape[0]), name="kh")
    kw = te.reduce_axis((0, filter_shape[1]), name="kw")
    conv = te.compute(
        (input_shape[0], filter_shape[-2], input_shape[2], input_shape[3], filter_shape[-1]),
        lambda n, ko, i, j, ki: te.sum(
            data[n, c, i, j, c4].astype("float32")
            * filt[kh, kw, c * input_shape[-1] + c4, ko, ki].astype("float32"),
            axis=[kh, kw, c, c4],
        ),
        # name="Compute_conv2d_1x1_NCHWc_RSCKk",
        name="conv2d_1x1",
    )
    return data, filt, conv


def schedule_conv2d_1x1_NCHWc_RSCKk(data, filt, conv):
    # inputs: (1, 128//4, 56, 56, 4), (1, 1, 128, 128//4, 4)
    # outputs:
    s = te.create_schedule(conv.op)
    A, B, C = data, filt, conv
    At = s.cache_read(A, "global.texture", [C])
    Bt = s.cache_read(B, "global.texture", [C])
    Al = s.cache_read(At, "local", [C])
    Bl = s.cache_read(Bt, "local", [C])
    Cl = s.cache_write(C, "local")

    def copy_to_texture(stage):
        axes = s[stage].op.axis
        fused = s[stage].fuse(*axes[:-1])
        block, thread = s[stage].split(fused, factor=32)
        s[stage].vectorize(axes[-1])
        s[stage].bind(block, te.thread_axis("blockIdx.x"))
        s[stage].bind(thread, te.thread_axis("threadIdx.x"))

    copy_to_texture(At)
    copy_to_texture(Bt)

    _n, _ko, _h, _w, _ki = s[C].op.axis
    s[C].vectorize(_ki)
    s[C].bind(_n, te.thread_axis("blockIdx.x"))
    s[C].bind(_ko, te.thread_axis("threadIdx.x"))

    s[Cl].compute_at(s[C], _w)
    _nl, _kol, _hl, _wl, _kil = s[Cl].op.axis
    _khl, _kwl, _cl, _cl4 = s[Cl].op.reduce_axis
    _clo, _cli = s[Cl].split(_cl, factor=4)
    s[Cl].reorder(_clo, _cli, _cl4, _kil)
    s[Cl].unroll(_cli)
    s[Cl].unroll(_cl4)
    s[Cl].vectorize(_kil)

    s[Al].compute_at(s[Cl], _cli)
    s[Al].vectorize(s[Al].op.axis[-1])
    s[Bl].compute_at(s[Cl], _kwl)
    s[Bl].vectorize(s[Bl].op.axis[-1])

    return s


def compute_conv2d_1x1_WCHNc_CRSKk(input_shape, filter_shape):
    # input_shape = [W, C, H, N, c] -> [W, C, H*N, c]
    # filter_shape = [C, R, S, K, k] -> [C, R*S*K, k]
    # output_shape: [WK, HN, k] -> [W, K, H, N, k]
    data = te.placeholder(input_shape, name="data", dtype="float32")
    filt = te.placeholder(filter_shape, name="filter", dtype="float32")

    packed_data = te.compute(
        (input_shape[0], input_shape[1], input_shape[2] * input_shape[3], input_shape[4]),
        lambda i, j, k, l: data[i, j, k // input_shape[3], k % input_shape[3], l],
        name="packed_data",
    )

    # Logical transformation of Nd -> 3d tensor
    # CRSKk -> C|RSK|k
    # r = rsk // SK
    # sk = rsk % SK
    # s = sk // K == (rsk % SK) // K == (rsk // K) % S
    # k = sk % K == (rsk % SK) % K == rsk % K
    packed_filter = te.compute(
        (filter_shape[0], filter_shape[1] * filter_shape[2] * filter_shape[3], filter_shape[4]),
        lambda i, j, k: filt[
            i,
            j // (filter_shape[3] * filter_shape[2]),
            (j // filter_shape[3]) % filter_shape[2],
            j % filter_shape[3],
            k,
        ],
        name="packed_filter",
    )

    c = te.reduce_axis((0, input_shape[1]), name="C")
    c4 = te.reduce_axis((0, input_shape[-1]), name="c4")
    r = te.reduce_axis((0, filter_shape[1]), name="r")
    s = te.reduce_axis((0, filter_shape[2]), name="s")

    conv = te.compute(
        (input_shape[0], filter_shape[3], input_shape[2], input_shape[3], filter_shape[4]),
        lambda w, ko, h, n, ki: te.sum(
            packed_data[w, c, h * input_shape[3] + n, c4].astype("float32")
            * packed_filter[
                c * input_shape[-1] + c4, ((r * filter_shape[2]) + s) * filter_shape[3] + ko, ki
            ].astype("float32"),
            axis=[r, s, c, c4],
        ),
        name="conv2d_1x1",
    )
    return data, filt, packed_data, packed_filter, conv


def schedule_conv2d_1x1_WCHNc_CRSKk(data, filt, packed_data, packed_filter, conv):
    # data: [W, C, H*N, c]
    # filter: [C, R*S*K, k]
    # output: [W, K, H, N, k]

    # conv2d( [N, C, H, W, c] , [1, 1, C, K, k]
    # inputs: (1, 128//4, 56, 56, 4), (1, 1, 128, 128//4, 4)

    # data: (56, 128//4, 56*1, 4) = (56, 32, 56, 4)
    # filt: (128, 1*1*128//4, 4) = (128, 32, 4)
    # conv: (56, 32, 56, 1, 4)

    s = te.create_schedule(conv.op)
    cfg = autotvm.get_config()

    s[packed_data].compute_inline()
    s[packed_filter].compute_inline()
    A, B, C = packed_data, packed_filter, conv
    At = s.cache_read(A, "global.texture", [C])
    Bt = s.cache_read(B, "global.texture", [C])
    Al = s.cache_read(At, "local", [C])
    Bl = s.cache_read(Bt, "local", [C])
    Cl = s.cache_write(C, "local")

    def copy_to_texture(stage):
        axes = s[stage].op.axis
        fused = s[stage].fuse(*axes[:-1])
        block, thread = s[stage].split(fused, factor=32)
        s[stage].vectorize(axes[-1])
        s[stage].bind(block, te.thread_axis("blockIdx.x"))
        s[stage].bind(thread, te.thread_axis("threadIdx.x"))

    copy_to_texture(At)
    copy_to_texture(Bt)

    _w, _ko, _h, _n, _ki = s[C].op.axis
    kernel_scope, _n = s[C].split(_n, nparts=1)

    cfg.define_split("tile_f", _ko, num_outputs=4)
    cfg.define_split("tile_w", _w, num_outputs=4)
    cfg.define_split("tile_h", _h, num_outputs=4)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    bk, vk, tk, ki = cfg["tile_f"].apply(s, C, _ko)
    bw, vw, tw, wi = cfg["tile_w"].apply(s, C, _w)
    bh, vh, th, hi = cfg["tile_h"].apply(s, C, _h)
    s[C].reorder(bh, _n, vh, th, hi)
    bhn = s[C].fuse(bh, _n)

    s[C].bind(bk, te.thread_axis("blockIdx.z"))
    s[C].bind(bhn, te.thread_axis("blockIdx.y"))
    s[C].bind(bw, te.thread_axis("blockIdx.x"))
    s[C].bind(vk, te.thread_axis("vthread"))
    s[C].bind(vh, te.thread_axis("vthread"))
    s[C].bind(vw, te.thread_axis("vthread"))
    s[C].bind(tk, te.thread_axis("threadIdx.z"))
    s[C].bind(th, te.thread_axis("threadIdx.y"))
    s[C].bind(tw, te.thread_axis("threadIdx.x"))
    s[C].reorder(bw, bk, bhn, vw, vk, vh, tw, tk, th, ki, hi, wi, _ki)
    s[C].vectorize(_ki)

    # TODO(csullivan): Try uneven workgroup split
    # _wo, _wi = s[C].split(_w, factor=4)
    # #_hno, _hni = s[C].split(_hn, factor=8)
    # #s[C].reorder(_wo, _wi, _ko, _hno, _hni, _ki)
    # s[C].reorder(_wo, _ko, _hn, _ki, _wi)
    # s[C].unroll(_wi)

    # # mace:
    # # const int out_ch_blk = get_global_id(0);
    # # const int out_w_blk = get_global_id(1);
    # # const int out_hb = get_global_id(2);

    # bx = te.thread_axis("blockIdx.x")
    # by = te.thread_axis("blockIdx.y")
    # bz = te.thread_axis("blockIdx.z")
    # s[C].bind(_ko, bx)
    # s[C].bind(_wo, by)
    # s[C].bind(_hn, bz)

    # s[Cl].compute_at(s[C], _hn)
    s[Cl].compute_at(s[C], th)

    _wl, _kol, _hl, _nl, _kil = s[Cl].op.axis
    _khl, _kwl, _cl, _cl4 = s[Cl].op.reduce_axis

    cfg.define_split("tile_c", _cl, num_outputs=2)
    cfg.define_split("tile_kh", _khl, num_outputs=2)
    cfg.define_split("tile_kw", _kwl, num_outputs=2)

    _clo, _cli = cfg["tile_c"].apply(s, Cl, _cl)
    _khlo, _khli = cfg["tile_kh"].apply(s, Cl, _khl)
    _kwlo, _kwli = cfg["tile_kw"].apply(s, Cl, _kwl)
    # s[OL].reorder(rco, ryo, rxo, rci, ryi, rxi, n, f, y, x)
    s[Cl].reorder(_clo, _khlo, _kwlo, _cli, _cl4, _khli, _kwli, _kol, _hl, _nl, _kil, _wl)
    # s[Cl].reorder(_clo, _khlo, _kwlo, _cli, _cl4, _khli, _kwli)
    # s[Cl].reorder(_cl, _cl4, _kil, _wl)
    s[Cl].unroll(_cl4)
    s[Cl].unroll(_wl)
    s[Cl].vectorize(_kil)

    _wla, _cla, _hnla, _cl4a = s[Al].op.axis
    s[Al].compute_at(s[Cl], _cli)
    s[Al].vectorize(_cl4a)
    s[Al].unroll(_wla)

    _clb, _rskolb, _kilb = s[Bl].op.axis
    s[Bl].compute_at(s[Cl], _cli)
    s[Bl].vectorize(_kilb)
    s[Bl].unroll(_clb)

    s[C].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)

    WO, K, HO, N, K4 = get_const_tuple(C.shape)
    RSC, _, _ = get_const_tuple(B.shape)
    cfg.add_flop(2 * N * K * K4 * HO * WO * RSC)

    return s


def compute_conv2d_NCHWc_KCRSk(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Convolution operator in NCHWc layout."""

    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel_chunk, in_height, in_width, in_channel_block = Input.shape
    num_filter_chunk, channel, kernel_h, kernel_w, num_filter_block = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = nn.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )

    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left, 0]
    pad_after = [0, 0, pad_down, pad_right, 0]
    temp = nn.pad(Input, pad_before, pad_after, name="pad_temp")

    rcc = te.reduce_axis((0, in_channel_chunk), name="rc")
    rcb = te.reduce_axis((0, in_channel_block), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    # NCHWc x KCRSk
    # texture: NCH|W|c
    # texture: K|CRS|k
    # c = crs//RS
    # rs = crs % RS
    # r = rs // W == (crs // S) % R
    # s = rs % W == crs % S
    Filter = te.compute(
        (num_filter_chunk, channel * kernel_h * kernel_w, num_filter_block),
        lambda ffc, crs, ffb: Filter[
            ffc, crs // (kernel_h * kernel_w), (crs // kernel_w) % kernel_h, crs % kernel_w, ffb
        ],
        name="packed_filter",
    )
    return te.compute(
        (batch, num_filter_chunk, out_height, out_width, num_filter_block),
        lambda nn, ffc, yy, xx, ffb: te.sum(
            temp[
                nn, rcc, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, rcb
            ].astype(out_dtype)
            * Filter[
                ffc, ((rcc * in_channel_block + rcb) * kernel_h + ry) * kernel_w + rx, ffb
            ].astype(out_dtype),
            axis=[rcc, rcb, ry, rx],
        ),
        tag="conv2d_nchwc_kcrsk_texture",
    )


def schedule_conv2d_NCHWc_KCRSk(cfg, s, conv):
    """schedule optimized for batch size = 1"""

    ##### space definition begin #####
    n, fc, y, x, fb = s[conv].op.axis
    rcc, rcb, ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_fc", fc, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rcc", rcc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    pad_data, flattened_kernel = s[conv].op.input_tensors
    kernel = s[flattened_kernel].op.input_tensors[0]
    s[flattened_kernel].compute_inline()

    s[pad_data].compute_inline()
    if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()
    kernel = flattened_kernel

    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, "local")
    else:
        output = s.outputs[0].output(0)
        s[conv].set_scope("local")
        OL = conv

    # create cache stage
    AT = s.cache_read(pad_data, "global.texture", [OL])
    WT = s.cache_read(kernel, "global.texture", [OL])

    def copy_to_texture(stage):
        axes = s[stage].op.axis
        fused = s[stage].fuse(*axes[:-1])
        block, thread = s[stage].split(fused, factor=32)
        s[stage].vectorize(axes[-1])
        s[stage].bind(block, te.thread_axis("blockIdx.x"))
        s[stage].bind(thread, te.thread_axis("threadIdx.x"))

    copy_to_texture(AT)
    copy_to_texture(WT)

    AA = s.cache_read(AT, "shared", [OL])
    WW = s.cache_read(WT, "shared", [OL])

    # tile and bind spatial axes
    n, fc, y, x, fb = s[output].op.axis

    kernel_scope, n = s[output].split(n, nparts=1)

    bf, vf, tf, fi = cfg["tile_fc"].apply(s, output, fc)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    bf = s[output].fuse(n, bf)
    s[output].bind(bf, te.thread_axis("blockIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi, fb)
    s[output].vectorize(fb)
    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, fc, y, x, fb = s[OL].op.axis

    rcc, rcb, ry, rx = s[OL].op.reduce_axis
    rco, rci = cfg["tile_rcc"].apply(s, OL, rcc)
    ryo, ryi = cfg["tile_ry"].apply(s, OL, ry)
    rxo, rxi = cfg["tile_rx"].apply(s, OL, rx)

    # TODO(csullivan): check position of rcb
    s[OL].reorder(rco, ryo, rxo, rci, ryi, rxi, rcb, n, fc, y, x, fb)
    s[OL].vectorize(fb)
    s[OL].unroll(rcb)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)
    # cooperative fetching
    for load in [AA, WW]:
        if load == WW:
            n, fyx, v = s[load].op.axis
            fused = s[load].fuse(n, fyx)
        else:
            n, f, y, x, v = s[load].op.axis
            fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_fc"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))
        s[load].vectorize(v)

    # unroll
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)

    N, OCC, OH, OW, OCB = get_const_tuple(output.shape)
    _, ICKHKW, _ = get_const_tuple(kernel.shape)

    if isinstance(N, int):
        cfg.add_flop(2 * N * OH * OW * OCC * OCB * ICKHKW)


def compute_conv2d_NCHWc_KCRSk_acc32(Input, Filter, stride, padding, dilation, out_dtype=None):
    """Convolution operator in NCHWc layout."""

    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_channel_chunk, in_height, in_width, in_channel_block = Input.shape
    num_filter_chunk, channel, kernel_h, kernel_w, num_filter_block = Filter.shape
    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = nn.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )

    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left, 0]
    pad_after = [0, 0, pad_down, pad_right, 0]
    temp = nn.pad(Input, pad_before, pad_after, name="pad_temp")

    rcc = te.reduce_axis((0, in_channel_chunk), name="rc")
    rcb = te.reduce_axis((0, in_channel_block), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    # NCHWc x KCRSk
    # texture: NCH|W|c
    # texture: K|CRS|k
    # c = crs//RS
    # rs = crs % RS
    # r = rs // W == (crs // S) % R
    # s = rs % W == crs % S
    Filter = te.compute(
        (num_filter_chunk, channel * kernel_h * kernel_w, num_filter_block),
        lambda ffc, crs, ffb: Filter[
            ffc, crs // (kernel_h * kernel_w), (crs // kernel_w) % kernel_h, crs % kernel_w, ffb
        ],
        name="packed_filter",
    )
    conv = te.compute(
        (batch, num_filter_chunk, out_height, out_width, num_filter_block),
        lambda nn, ffc, yy, xx, ffb: te.sum(
            (
                temp[nn, rcc, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, rcb]
                * Filter[ffc, ((rcc * in_channel_block + rcb) * kernel_h + ry) * kernel_w + rx, ffb]
            ).astype(out_dtype),
            axis=[rcc, rcb, ry, rx],
        ),
        tag="conv2d_nchwc_kcrsk_texture",
    )
    output = te.compute(conv.shape, lambda n, fc, y, x, fb: conv[n, fc, y, x, fb].astype("float32"))
    return output


def schedule_conv2d_NCHWc_KCRSk_acc32(cfg, s, output):
    """schedule optimized for batch size = 1"""

    conv = output.op.input_tensors[0]

    ##### space definition begin #####
    n, fc, y, x, fb = s[conv].op.axis
    rcc, rcb, ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_fc", fc, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rcc", rcc, num_outputs=2)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    pad_data, flattened_kernel = s[conv].op.input_tensors
    kernel = s[flattened_kernel].op.input_tensors[0]
    s[flattened_kernel].compute_inline()

    s[pad_data].compute_inline()
    if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()
    kernel = flattened_kernel

    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, "local")
    else:
        output = s.outputs[0].output(0)
        s[conv].set_scope("local")
        OL = conv

    # create cache stage
    AT = s.cache_read(pad_data, "global.texture", [OL])
    WT = s.cache_read(kernel, "global.texture", [OL])

    def copy_to_texture(stage):
        axes = s[stage].op.axis
        fused = s[stage].fuse(*axes[:-1])
        block, thread = s[stage].split(fused, factor=32)
        s[stage].vectorize(axes[-1])
        s[stage].bind(block, te.thread_axis("blockIdx.x"))
        s[stage].bind(thread, te.thread_axis("threadIdx.x"))

    copy_to_texture(AT)
    copy_to_texture(WT)

    AA = s.cache_read(AT, "shared", [OL])
    WW = s.cache_read(WT, "shared", [OL])

    # tile and bind spatial axes
    n, fc, y, x, fb = s[output].op.axis

    kernel_scope, n = s[output].split(n, nparts=1)

    bf, vf, tf, fi = cfg["tile_fc"].apply(s, output, fc)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    bf = s[output].fuse(n, bf)
    s[output].bind(bf, te.thread_axis("blockIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi, fb)
    s[output].vectorize(fb)

    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, fc, y, x, fb = s[OL].op.axis

    rcc, rcb, ry, rx = s[OL].op.reduce_axis
    rco, rci = cfg["tile_rcc"].apply(s, OL, rcc)
    ryo, ryi = cfg["tile_ry"].apply(s, OL, ry)
    rxo, rxi = cfg["tile_rx"].apply(s, OL, rx)

    # TODO(csullivan): check position of rcb
    s[OL].reorder(rco, ryo, rxo, rci, ryi, rxi, rcb, n, fc, y, x, fb)
    s[OL].vectorize(fb)
    s[OL].unroll(rcb)

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)
    # cooperative fetching
    for load in [AA, WW]:
        if load == WW:
            n, fyx, v = s[load].op.axis
            fused = s[load].fuse(n, fyx)
        else:
            n, f, y, x, v = s[load].op.axis
            fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_fc"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))
        s[load].vectorize(v)

    # unroll
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)

    N, OCC, OH, OW, OCB = get_const_tuple(output.shape)
    _, ICKHKW, _ = get_const_tuple(kernel.shape)

    if isinstance(N, int):
        cfg.add_flop(2 * N * OH * OW * OCC * OCB * ICKHKW)


def compute_depthwise_conv2d_NCHWc_KCRSk_acc32(
    Input, Filter, stride, padding, dilation, out_dtype=None
):
    """Depthwise convolution operator in NCHWc layout."""
    if out_dtype is None:
        out_dtype = Input.dtype
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, channel_chunk, in_height, in_width, channel_block = Input.shape
    _, channel_multiplier, kernel_h, kernel_w, _ = Filter.shape

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = nn.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel_chunk = simplify(channel_chunk * channel_multiplier)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    # compute graph
    pad_before = [0, 0, pad_top, pad_left, 0]
    pad_after = [0, 0, pad_down, pad_right, 0]
    temp = nn.pad(Input, pad_before, pad_after, name="pad_temp")

    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    # NCHWc x CMRSc = [N,(C//4)M,OH,OW, 4c]
    # NCHWc x CMRS
    # texture: NCH|W|c
    # texture: C|MRS|c
    # output: N
    # m = mrs//RS
    # rs = mrs % RS
    # r = rs // W == (mrs // S) % R
    # s = rs % W == mrs % S
    Filter = te.compute(
        (channel_chunk, channel_multiplier * kernel_h * kernel_w, channel_block),
        lambda ffc, mrs, ffb: Filter[
            ffc, mrs // (kernel_h * kernel_w), (mrs // kernel_w) % kernel_h, mrs % kernel_w, ffb
        ],
        name="packed_filter",
    )

    conv = te.compute(
        (batch, out_channel_chunk, out_height, out_width, channel_block),
        lambda nn, ffc, yy, xx, ffb: te.sum(
            (
                temp[
                    nn,
                    ffc // channel_multiplier,
                    yy * stride_h + ry * dilation_h,
                    xx * stride_w + rx * dilation_w,
                    ffb,
                ]
                * Filter[
                    ffc // channel_multiplier,
                    ((ffc % channel_multiplier) * kernel_h + ry) * kernel_w + rx,
                    ffb,
                ]
            ).astype(out_dtype),
            axis=[ry, rx],
        ),
        tag="depthwise_conv2d_nchwc_kcrsk_texture",
    )
    return te.compute(
        conv.shape, lambda n, ffc, y, x, ffb: conv[n, ffc, y, x, ffb].astype("float32")
    )


def schedule_depthwise_conv2d_NCHWc_KCRSk_acc32(cfg, s, output):
    """schedule optimized for batch size = 1"""

    conv = output.op.input_tensors[0]

    ##### space definition begin #####
    n, fc, y, x, fb = s[conv].op.axis
    ry, rx = s[conv].op.reduce_axis
    cfg.define_split("tile_fc", fc, num_outputs=4)
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_ry", ry, num_outputs=2)
    cfg.define_split("tile_rx", rx, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    pad_data, flattened_kernel = s[conv].op.input_tensors
    kernel = s[flattened_kernel].op.input_tensors[0]
    s[flattened_kernel].compute_inline()

    s[pad_data].compute_inline()
    if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()
    kernel = flattened_kernel

    if conv.op in s.outputs:
        output = conv
        OL = s.cache_write(conv, "local")
    else:
        output = s.outputs[0].output(0)
        s[conv].set_scope("local")
        OL = conv

    # create cache stage
    AT = s.cache_read(pad_data, "global.texture", [OL])
    WT = s.cache_read(kernel, "global.texture", [OL])

    def copy_to_texture(stage):
        axes = s[stage].op.axis
        fused = s[stage].fuse(*axes[:-1])
        block, thread = s[stage].split(fused, factor=32)
        s[stage].vectorize(axes[-1])
        s[stage].bind(block, te.thread_axis("blockIdx.x"))
        s[stage].bind(thread, te.thread_axis("threadIdx.x"))

    copy_to_texture(AT)
    copy_to_texture(WT)

    AA = s.cache_read(AT, "shared", [OL])
    WW = s.cache_read(WT, "shared", [OL])

    # tile and bind spatial axes
    n, fc, y, x, fb = s[output].op.axis

    kernel_scope, n = s[output].split(n, nparts=1)

    bf, vf, tf, fi = cfg["tile_fc"].apply(s, output, fc)
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    bf = s[output].fuse(n, bf)
    s[output].bind(bf, te.thread_axis("blockIdx.z"))
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vf, te.thread_axis("vthread"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(tf, te.thread_axis("threadIdx.z"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    s[output].reorder(bf, by, bx, vf, vy, vx, tf, ty, tx, fi, yi, xi, fb)
    s[output].vectorize(fb)

    s[OL].compute_at(s[output], tx)

    # tile reduction axes
    n, fc, y, x, fb = s[OL].op.axis

    ry, rx = s[OL].op.reduce_axis
    ryo, ryi = cfg["tile_ry"].apply(s, OL, ry)
    rxo, rxi = cfg["tile_rx"].apply(s, OL, rx)

    s[OL].reorder(ryo, rxo, ryi, rxi, n, fc, y, x, fb)
    s[OL].vectorize(fb)
    # s[OL].unroll()

    s[AA].compute_at(s[OL], rxo)
    s[WW].compute_at(s[OL], rxo)
    # cooperative fetching
    for load in [AA, WW]:
        if load == WW:
            n, fyx, v = s[load].op.axis
            fused = s[load].fuse(n, fyx)
        else:
            n, f, y, x, v = s[load].op.axis
            fused = s[load].fuse(n, f, y, x)
        tz, fused = s[load].split(fused, nparts=cfg["tile_fc"].size[2])
        ty, fused = s[load].split(fused, nparts=cfg["tile_y"].size[2])
        tx, fused = s[load].split(fused, nparts=cfg["tile_x"].size[2])
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))
        s[load].vectorize(v)

    # unroll
    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)

    N, OCC, OH, OW, OCB = get_const_tuple(output.shape)
    ICC, MKHKW, ICB = get_const_tuple(kernel.shape)
    M = (OCC * OCB) // (ICC * ICB)
    KHKW = MKHKW // M

    if isinstance(N, int):
        cfg.add_flop(2 * N * OH * OW * OCC * OCB * KHKW)


def scheduler(compute, schedule, *args, **kwargs):
    placeholders = compute(*args)
    s = schedule(*placeholders, **kwargs)
    return s, placeholders


def conv2d_1x1_NCHWc_RSCKk(input_shape, filter_shape):
    placeholders = compute_conv2d_1x1_NCHWc_RSCKk(input_shape, filter_shape)
    s = schedule_conv2d_1x1_NCHWc_RSCKk(*placeholders)
    return s, placeholders


def conv2d_1x1_WCHNc_CRSKk(input_shape, filter_shape):
    placeholders = compute_conv2d_1x1_WCHNc_CRSKk(input_shape, filter_shape)
    s = schedule_conv2d_1x1_WCHNc_CRSKk(*placeholders)
    return s, (placeholders[0], placeholders[1], placeholders[-1])


def conv2d_NCHWc_KCRSk(input_shape, filter_shape):
    data = te.placeholder(input_shape, name="data", dtype="float32")
    filt = te.placeholder(filter_shape, name="filter", dtype="float32")
    conv = compute_conv2d_NCHWc_KCRSk(data, filt, [1, 1], [0, 0], [1, 1], "float32")
    cfg = autotvm.get_config()
    s = te.create_schedule([x.op for x in [conv]])
    schedule_conv2d_NCHWc_KCRSk(cfg, s, conv)
    return s, (data, filt, conv)


def conv2d_NCHWc_KCRSk_fp32_acc(input_shape, filter_shape):
    data = te.placeholder(input_shape, name="data", dtype="float32")
    filt = te.placeholder(filter_shape, name="filter", dtype="float32")
    output = compute_conv2d_NCHWc_KCRSk_acc32(data, filt, [1, 1], [0, 0], [1, 1], "float32")
    cfg = autotvm.get_config()
    s = te.create_schedule([x.op for x in [output]])
    schedule_conv2d_NCHWc_KCRSk_acc32(cfg, s, output)
    return s, (data, filt, output)


def depthwise_conv2d_NCHWc_KCRSk_acc32(input_shape, filter_shape):
    data = te.placeholder(input_shape, name="data", dtype="float32")
    filt = te.placeholder(filter_shape, name="filter", dtype="float32")
    output = compute_depthwise_conv2d_NCHWc_KCRSk_acc32(
        data, filt, [1, 1], [0, 0], [1, 1], "float32"
    )
    cfg = autotvm.get_config()
    s = te.create_schedule([x.op for x in [output]])
    schedule_depthwise_conv2d_NCHWc_KCRSk_acc32(cfg, s, output)
    return s, (data, filt, output)


def ref_convolution(data, kernel, stride, pad):
    import mxnet as mx

    groups = 1
    kernel_size = (kernel.shape[2], kernel.shape[3])
    num_filter = kernel.shape[0]
    ref_res = mx.nd.Convolution(
        data=mx.nd.array(data),
        weight=mx.nd.array(kernel),
        bias=None,
        no_bias=True,
        kernel=kernel_size,
        stride=stride,
        pad=pad,
        num_filter=num_filter,
        num_group=groups,
    )
    return ref_res.asnumpy()


def ref_depthwise_convolution(data, kernel, stride, pad):
    import mxnet as mx

    groups = kernel.shape[0]
    kernel_size = (kernel.shape[2], kernel.shape[3])
    num_filter = kernel.shape[0]
    multiplier = kernel.shape[1]
    ref_res = mx.nd.Convolution(
        data=mx.nd.array(data),
        weight=mx.nd.array(kernel),
        bias=None,
        no_bias=True,
        kernel=kernel_size,
        stride=stride,
        pad=pad,
        num_filter=num_filter,
        num_group=groups,
    )
    return ref_res.asnumpy()


def validate(workload, target, dev, input_shapes, *args, **kwargs):
    s, placeholders = workload(*input_shapes, *args, **kwargs)
    func = tvm.driver.build(s, [*placeholders], target=target, name="TestFunction")

    args_tvm = []
    args_np = []
    for var in placeholders[:-1]:
        var_np = np.random.uniform(size=[i.value for i in var.shape]).astype(var.dtype)
        args_np.append(var_np)
        args_tvm.append(tvm.nd.array(var_np, dev))
    args_tvm.append(
        tvm.nd.array(
            np.zeros([i.value for i in placeholders[-1].shape], dtype=placeholders[-1].dtype), dev
        )
    )
    func(*args_tvm)

    if "plus_one" in workload.__name__:
        np_result = args_np[0] + 1.0
    elif "matmul" in workload.__name__:
        if "inner" in workload.__name__:
            np_result = np.matmul(
                args_np[0].reshape(32, 256), args_np[1].reshape(32, 256).transpose(1, 0)
            )
        elif "accum" in workload.__name__:
            np_result = np.matmul(
                args_np[0].transpose((1, 0, 2)).reshape(64, 128), args_np[1].reshape(128, 64)
            )
        else:
            np_result = np.matmul(
                args_np[0].transpose((0, 2, 1)).reshape(128, 64),
                args_np[1].transpose(1, 0, 2).reshape(64, 128),
            )
    elif "conv2d_1x1_NCHWc_RSCKk" in workload.__name__:
        vec_length = args_np[1].shape[-1]
        # nchwc -> nchw
        args_np[0] = (
            args_np[0]
            .transpose((0, 1, 4, 2, 3))
            .reshape(
                args_np[0].shape[0],
                args_np[0].shape[1] * args_np[0].shape[-1],
                args_np[0].shape[2],
                args_np[0].shape[3],
            )
        )
        # rsckk -> rsck -> kcrs
        args_np[1] = (
            args_np[1]
            .reshape(
                args_np[1].shape[0],
                args_np[1].shape[1],
                args_np[1].shape[2],
                args_np[1].shape[3] * args_np[1].shape[4],
            )
            .transpose((3, 2, 0, 1))
        )
        np_result = testing.conv2d_nchw_python(args_np[0], args_np[1], 1, 0)
        # nkhw -> nkhwk
        np_result = np_result.reshape(
            np_result.shape[0],
            np_result.shape[1] // vec_length,
            vec_length,
            np_result.shape[2],
            np_result.shape[3],
        ).transpose(0, 1, 3, 4, 2)
    elif "conv2d_1x1_WCHNc_CRSKk" in workload.__name__:
        vec_length = args_np[1].shape[-1]
        # wchnc -> nchw
        args_np[0] = (
            args_np[0]
            .transpose((3, 1, 4, 2, 0))
            .reshape(
                args_np[0].shape[3],
                args_np[0].shape[1] * args_np[0].shape[-1],
                args_np[0].shape[2],
                args_np[0].shape[0],
            )
        )
        # crskk -> crsk -> kcrs
        args_np[1] = (
            args_np[1]
            .reshape(
                args_np[1].shape[0],
                args_np[1].shape[1],
                args_np[1].shape[2],
                args_np[1].shape[3] * args_np[1].shape[4],
            )
            .transpose((3, 0, 1, 2))
        )
        np_result = testing.conv2d_nchw_python(args_np[0], args_np[1], 1, 0)
        # nkhw -> nkkhw -> wkhnk
        np_result = np_result.reshape(
            np_result.shape[0],
            np_result.shape[1] // vec_length,
            vec_length,
            np_result.shape[2],
            np_result.shape[3],
        ).transpose(4, 1, 3, 0, 2)
    elif "NCHW_KCRS" in workload.__name__:
        np_result = testing.conv2d_nchw_python(args_np[0], args_np[1], 1, 0)
    elif "NCHWc_KCRSk" in workload.__name__:
        vec_length = args_np[1].shape[-1]
        # nchwc -> nchw
        args_np[0] = (
            args_np[0]
            .transpose((0, 1, 4, 2, 3))
            .reshape(
                args_np[0].shape[0],
                args_np[0].shape[1] * args_np[0].shape[-1],
                args_np[0].shape[2],
                args_np[0].shape[3],
            )
        )
        # kcrsk/cmrsc -> kcrs/cmrs
        args_np[1] = (
            args_np[1]
            .transpose((0, 4, 1, 2, 3))
            .reshape(
                args_np[1].shape[0] * args_np[1].shape[4],
                args_np[1].shape[1],
                args_np[1].shape[2],
                args_np[1].shape[3],
            )
        )
        if "depthwise" in workload.__name__:
            # np_result = testing.depthwise_conv2d_python_nchw(args_np[0], args_np[1], 1, "VALID")
            np_result = ref_depthwise_convolution(args_np[0], args_np[1], [], [])
        else:
            # np_result = testing.conv2d_nchw_python(args_np[0], args_np[1], 1, 0)
            np_result = ref_convolution(args_np[0], args_np[1], [], [])
        # nkhw -> nkhwk
        np_result = np_result.reshape(
            np_result.shape[0],
            np_result.shape[1] // vec_length,
            vec_length,
            np_result.shape[2],
            np_result.shape[3],
        ).transpose(0, 1, 3, 4, 2)
    np.testing.assert_allclose(args_tvm[-1].asnumpy(), np_result, rtol=1e-2, atol=1e-2)


class BaseSingleShapeValidator:
    @tvm.testing.parametrize_targets("opencl")
    def test_unary(self, test_func, input_shape, target, dev):
        validate(test_func, target, dev, [input_shape])


class TestPlusOneRank3(BaseSingleShapeValidator):
    input_shape = tvm.testing.parameter((32, 32, 4))

    def plus_one(input_shape):
        return scheduler(compute_plus_one_rank3, schedule_plus_one_rank3, input_shape)

    test_func = tvm.testing.parameter(plus_one)


class TestPlusOneRank5(BaseSingleShapeValidator):
    input_shape = tvm.testing.parameter((32, 2, 4, 4, 4))

    def plus_one(input_shape):
        return scheduler(compute_plus_one_rank5, schedule_plus_one_rank5, input_shape)

    test_func = tvm.testing.parameter(plus_one)


class TestMatmul:
    input_shape = tvm.testing.parameter((32, 64, 4))
    local = tvm.testing.parameter(False, True)

    def matmul(input_shape, local):
        return scheduler(compute_matmul, schedule_matmul, input_shape, local=local)

    def matmul_inner(input_shape, local):
        return scheduler(compute_matmul_inner, schedule_matmul_inner, input_shape, local=local)

    test_func = tvm.testing.parameter(matmul, matmul_inner)

    @tvm.testing.parametrize_targets("opencl")
    def test_matmul(self, test_func, input_shape, local, target, dev):
        validate(test_func, target, dev, [input_shape], local=local)


class TestMatmulVectorAccumulator:
    shapeA = tvm.testing.parameter((32, 64, 4))
    shapeB = tvm.testing.parameter((128, 16, 4))
    local = tvm.testing.parameter(False, True)

    def matmul_vector_accumulator(shapeA, shapeB, local):
        return scheduler(
            compute_matmul_vector_accumulator,
            schedule_matmul_vector_accumulator,
            shapeA,
            shapeB,
            local=local,
        )

    test_func = tvm.testing.parameter(matmul_vector_accumulator)

    @tvm.testing.parametrize_targets("opencl")
    def test_matmul_vec_acc(self, test_func, shapeA, shapeB, local, target, dev):
        validate(test_func, target, dev, [shapeA, shapeB], local=local)


class BaseConv2DValidator:
    @tvm.testing.parametrize_targets("opencl")
    def test_conv2d(self, test_func, input_shapes, target, dev):
        validate(test_func, target, dev, input_shapes)


class TestConv2dNCHWcRSCKk(BaseConv2DValidator):
    input_shapes = tvm.testing.parameter([(1, 32, 56, 56, 4), (1, 1, 128, 32, 4)])
    test_func = tvm.testing.parameter(conv2d_1x1_NCHWc_RSCKk)


class TestConv2dWCHNcCRSKk(BaseConv2DValidator):
    input_shapes = tvm.testing.parameter([(56, 32, 56, 1, 4), (128, 1, 1, 32, 4)])
    test_func = tvm.testing.parameter(conv2d_1x1_WCHNc_CRSKk)


@pytest.mark.skip("AttributeError: module 'numpy' has no attribute 'bool' raised from mxnet")
class TestConv2dNCHWcKCRSk(BaseConv2DValidator):
    input_shapes = tvm.testing.parameter(
        [(1, 32, 56, 56, 4), (32, 128, 1, 1, 4)], [(1, 32, 112, 112, 4), (32, 128, 3, 3, 4)]
    )
    test_func = tvm.testing.parameter(conv2d_NCHWc_KCRSk, conv2d_NCHWc_KCRSk_fp32_acc)


@pytest.mark.skip("AttributeError: module 'numpy' has no attribute 'bool' raised from mxnet")
class TestDepthwiseConv2dNCHWcKCRSk(BaseConv2DValidator):
    input_shapes = tvm.testing.parameter([(1, 24, 257, 257, 4), (24, 1, 3, 3, 4)])
    test_func = tvm.testing.parameter(depthwise_conv2d_NCHWc_KCRSk_acc32)


def simple_texture_to_scalar_common(
    target, input_info, output_info, find_patterns, dtype, cast_type
):
    def _compute():
        p0 = te.placeholder(input_info[1], name="p0", dtype=dtype)
        p0_comp = te.compute(input_info[1], lambda *i: p0(*i), name="p0_comp")
        if len(output_info[1]) == 4 and len(input_info[1]) == 5:
            out = te.compute(
                output_info[1],
                lambda n, c, h, w: p0_comp[n][c // 4][h][w][c % 4].astype(cast_type),
                name="out",
            )
        elif len(output_info[1]) == 5 and len(input_info[1]) == 5:
            out = te.compute(
                output_info[1],
                lambda n, c, h, w, cb: p0_comp[n][c][h][w][cb].astype(cast_type),
                name="out",
            )
        else:
            raise Exception("Impossible case")
        dummy_out = te.compute(output_info[1], lambda *i: out(*i), name="dummy_out")
        return p0, dummy_out

    def _schedule(dummy_out):
        from tvm.topi.adreno.utils import bind_data_copy

        s = te.create_schedule(dummy_out.op)
        out = s[dummy_out].op.input_tensors[0]
        p0_comp = s[out].op.input_tensors[0]
        s[p0_comp].set_scope(input_info[0])
        bind_data_copy(s[p0_comp])
        s[out].set_scope(output_info[0])
        bind_data_copy(s[out])
        bind_data_copy(s[dummy_out])
        return s

    p0, dummy_out = _compute()
    s = _schedule(dummy_out)

    fun = tvm.build(s, [p0, dummy_out], target)
    dev = tvm.device(target, 0)
    opencl_source = fun.imported_modules[0].get_source()
    start_idx = 0
    for pattern in find_patterns:
        start_idx = opencl_source.find(pattern, start_idx)
        assert start_idx > -1

    input_np = np.random.uniform(size=[i for i in input_info[1]]).astype(dtype)
    input_tvm = tvm.nd.array(input_np, dev)
    c = tvm.nd.empty(output_info[1], dtype, dev)
    # Doesn't run OpenCL code for FP16 because GPUs in CI don't support FP16 inference
    if cast_type == "float32":
        fun(input_tvm, c)
    # For output len == 5 it makes no sense to check the accuracy
    if cast_type == "float32" and len(output_info[1]) == 4:
        np_result = input_np.transpose(0, 2, 3, 1, 4)  # NCHW4c -> NHWC4c
        np_result = np.squeeze(np_result, axis=3)
        np_result = np_result.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        np.testing.assert_allclose(c.asnumpy(), np_result, rtol=1e-2, atol=1e-2)


class TestSimpleTextureToScalarFP16:
    # (input [scope, shape], output [scope, shape], [find_patterns])
    input_info, output_info, find_patterns = tvm.testing.parameters(
        # 1. Texture (NCHW4c) -> Cast(FP16) -> Buffer (NCHW)
        (
            ["global.texture", (1, 1, 40, 40, 4)],
            ["", (1, 4, 40, 40)],
            [
                "float4 v_ = READ_IMAGEF(p0_comp, image_sampler, ((int2)(((convert_int(get_local_id(0))) % 40), ((((convert_int(get_group_id(0))) & 1) * 20) + ((convert_int(get_local_id(0))) / 40)))));",
                "out[(((convert_int(get_group_id(0))) * 800) + (convert_int(get_local_id(0))))] = (convert_half(((float*)&v_)[((convert_int(get_group_id(0))) >> 1)]));",
            ],
        ),
        # 2. Buffer (NCHW4c) -> Cast(FP16) -> Buffer (NCHW)
        (
            ["", (1, 1, 40, 40, 4)],
            ["", (1, 4, 40, 40)],
            [
                "out[(((convert_int(get_group_id(0))) * 800) + (convert_int(get_local_id(0))))] = (convert_half(p0_comp[(((((convert_int(get_group_id(0))) & 1) * 3200) + ((convert_int(get_local_id(0))) * 4)) + ((convert_int(get_group_id(0))) >> 1))]));"
            ],
        ),
        # 3. Texture (NCHW4c) -> Cast(FP16) -> Texture (NCHW4c)
        (
            ["global.texture", (1, 1, 40, 40, 4)],
            ["global.texture", (1, 1, 40, 40, 4)],
            [
                "float4 v_ = READ_IMAGEF(p0_comp, image_sampler, ((int2)(((((convert_int(get_group_id(0))) * 24) + (convert_int(get_local_id(0)))) % 40), ((((convert_int(get_group_id(0))) * 8) + ((convert_int(get_local_id(0))) >> 3)) / 5))));",
                "write_imageh(out, (int2)(((((convert_int(get_group_id(0))) * 24) + (convert_int(get_local_id(0)))) % 40), ((((convert_int(get_group_id(0))) * 8) + ((convert_int(get_local_id(0))) >> 3)) / 5)), (convert_half4(v_)));",
            ],
        ),
    )
    dtype = tvm.testing.parameter("float32")

    @tvm.testing.parametrize_targets("opencl")
    def test_simple_texture_to_scalar_fp16(
        self, input_info, output_info, find_patterns, dtype, target
    ):
        simple_texture_to_scalar_common(
            target, input_info, output_info, find_patterns, dtype, "float16"
        )


class TestSimpleTextureToScalarFP32:
    # (input [scope, shape], output [scope, shape], [find_patterns])
    input_info, output_info, find_patterns = tvm.testing.parameters(
        # 1. Texture (NCHW4c) -> Buffer (NCHW)
        (
            ["global.texture", (1, 1, 40, 40, 4)],
            ["", (1, 4, 40, 40)],
            [
                "float4 v_ = READ_IMAGEF(p0_comp, image_sampler, ((int2)(((convert_int(get_local_id(0))) % 40), ((((convert_int(get_group_id(0))) & 1) * 20) + ((convert_int(get_local_id(0))) / 40)))));",
                "out[(((convert_int(get_group_id(0))) * 800) + (convert_int(get_local_id(0))))] = ((float*)&v_)[((convert_int(get_group_id(0))) >> 1)];",
            ],
        ),
        # 2. Buffer (NCHW4c) -> Buffer (NCHW)
        (
            ["", (1, 1, 40, 40, 4)],
            ["", (1, 4, 40, 40)],
            [
                "out[(((convert_int(get_group_id(0))) * 800) + (convert_int(get_local_id(0))))] = p0_comp[(((((convert_int(get_group_id(0))) & 1) * 3200) + ((convert_int(get_local_id(0))) * 4)) + ((convert_int(get_group_id(0))) >> 1))];"
            ],
        ),
    )
    dtype = tvm.testing.parameter("float32")

    @tvm.testing.parametrize_targets("opencl")
    def test_simple_texture_to_scalar_fp32(
        self, input_info, output_info, find_patterns, dtype, target
    ):
        simple_texture_to_scalar_common(
            target, input_info, output_info, find_patterns, dtype, "float32"
        )


def texture_to_scalar_reuse_ssa_common(
    target, input_info, output_info, find_patterns, dtype, cast_type
):
    def _compute():
        p0 = te.placeholder(input_info[1], name="p0", dtype=dtype)
        p0_comp = te.compute(input_info[1], lambda *i: p0(*i), name="p0_comp")
        if len(output_info[1]) == 4 and len(input_info[1]) == 5:
            out = te.compute(
                output_info[1],
                lambda n, c, h, w: p0_comp[n][c // 4][h][w][c % 4].astype(cast_type),
                name="out",
            )
            out2 = te.compute(
                output_info[1],
                lambda n, c, h, w: out[n][c][h][w]
                + p0_comp[n][c // 4][h][w][c % 4].astype(cast_type),
                name="out",
            )
        elif len(output_info[1]) == 5 and len(input_info[1]) == 5:
            out = te.compute(
                output_info[1],
                lambda n, c, h, w, cb: p0_comp[n][c][h][w][cb].astype(cast_type),
                name="out",
            )
            out2 = te.compute(
                output_info[1],
                lambda n, c, h, w, cb: out[n][c][h][w][cb]
                + p0_comp[n][c][h][w][cb].astype(cast_type),
                name="out",
            )
        else:
            raise Exception("Impossible case")
        out_sum = te.compute(output_info[1], lambda *i: out(*i) + out2(*i), name="out_sum")
        dummy_out = te.compute(output_info[1], lambda *i: out_sum(*i), name="dummy_out")
        return p0, dummy_out

    def _schedule(dummy_out):
        from tvm.topi.adreno.utils import bind_data_copy

        s = te.create_schedule(dummy_out.op)
        out_sum = s[dummy_out].op.input_tensors[0]
        out, out2 = s[out_sum].op.input_tensors
        p0_comp = s[out].op.input_tensors[0]
        s[p0_comp].set_scope(input_info[0])
        bind_data_copy(s[p0_comp])
        s[out].set_scope(output_info[0])
        s[out2].set_scope(output_info[0])
        s[out2].compute_inline()
        s[out].compute_inline()
        s[out_sum].set_scope(output_info[0])
        bind_data_copy(s[out_sum])
        bind_data_copy(s[dummy_out])
        return s

    p0, dummy_out = _compute()
    s = _schedule(dummy_out)

    fun = tvm.build(s, [p0, dummy_out], target)
    dev = tvm.device(target, 0)
    opencl_source = fun.imported_modules[0].get_source()
    start_idx = 0
    for pattern in find_patterns:
        start_idx = opencl_source.find(pattern, start_idx)
        assert start_idx > -1

    input_np = np.random.uniform(size=[i for i in input_info[1]]).astype(dtype)
    input_tvm = tvm.nd.array(input_np, dev)
    c = tvm.nd.empty(output_info[1], dtype, dev)
    # Doesn't run OpenCL code for FP16 because GPUs in CI don't support FP16 inference
    if cast_type == "float32":
        fun(input_tvm, c)
    # For output len == 5 it makes no sense to check the accuracy
    if cast_type == "float32" and len(output_info[1]) == 4:
        np_result = input_np * 3
        np_result = np_result.transpose(0, 2, 3, 1, 4)  # NCHW4c -> NHWC4c
        np_result = np.squeeze(np_result, axis=3)
        np_result = np_result.transpose(0, 3, 1, 2)  # NHWC -> NCHW
        np.testing.assert_allclose(c.asnumpy(), np_result, rtol=1e-2, atol=1e-2)


class TestTextureToScalarReuseSSAFP16:
    # (input [scope, shape], output [scope, shape], [find_patterns])
    input_info, output_info, find_patterns = tvm.testing.parameters(
        # 1. Texture (NCHW4c) -> Cast(FP16) -> Buffer (NCHW)
        (
            ["global.texture", (1, 1, 40, 40, 4)],
            ["", (1, 4, 40, 40)],
            [
                "float4 v_ = READ_IMAGEF(p0_comp, image_sampler, ((int2)(((convert_int(get_local_id(0))) % 40), ((((convert_int(get_group_id(0))) & 1) * 20) + ((convert_int(get_local_id(0))) / 40)))));",
                "out_sum[(((convert_int(get_group_id(0))) * 800) + (convert_int(get_local_id(0))))] = ((convert_half(((float*)&v_)[((convert_int(get_group_id(0))) >> 1)])) + ((convert_half(((float*)&v_)[((convert_int(get_group_id(0))) >> 1)])) + (convert_half(((float*)&v_)[((convert_int(get_group_id(0))) >> 1)]))));",
            ],
        ),
        # 2. Buffer (NCHW4c) -> Cast(FP16) -> Buffer (NCHW)
        (
            ["", (1, 1, 40, 40, 4)],
            ["", (1, 4, 40, 40)],
            [
                " out_sum[(((convert_int(get_group_id(0))) * 800) + (convert_int(get_local_id(0))))] = ((convert_half(p0_comp[(((((convert_int(get_group_id(0))) & 1) * 3200) + ((convert_int(get_local_id(0))) * 4)) + ((convert_int(get_group_id(0))) >> 1))])) + ((convert_half(p0_comp[(((((convert_int(get_group_id(0))) & 1) * 3200) + ((convert_int(get_local_id(0))) * 4)) + ((convert_int(get_group_id(0))) >> 1))])) + (convert_half(p0_comp[(((((convert_int(get_group_id(0))) & 1) * 3200) + ((convert_int(get_local_id(0))) * 4)) + ((convert_int(get_group_id(0))) >> 1))]))));"
            ],
        ),
        # 3. Texture (NCHW4c) -> Cast(FP16) -> Texture (NCHW4c)
        (
            ["global.texture", (1, 1, 40, 40, 4)],
            ["global.texture", (1, 1, 40, 40, 4)],
            [
                "float4 v_ = READ_IMAGEF(p0_comp, image_sampler, ((int2)(((((convert_int(get_group_id(0))) * 24) + (convert_int(get_local_id(0)))) % 40), ((((convert_int(get_group_id(0))) * 8) + ((convert_int(get_local_id(0))) >> 3)) / 5))));",
                "write_imageh(out_sum, (int2)(((((convert_int(get_group_id(0))) * 24) + (convert_int(get_local_id(0)))) % 40), ((((convert_int(get_group_id(0))) * 8) + ((convert_int(get_local_id(0))) >> 3)) / 5)), ((convert_half4(v_)) + ((convert_half4(v_)) + (convert_half4(v_)))));",
            ],
        ),
    )
    dtype = tvm.testing.parameter("float32")

    @tvm.testing.parametrize_targets("opencl")
    def test_texture_to_scalar_reuse_ssa_fp16(
        self, input_info, output_info, find_patterns, dtype, target
    ):
        texture_to_scalar_reuse_ssa_common(
            target, input_info, output_info, find_patterns, dtype, "float16"
        )


class TestTextureToScalarReuseSSAFP32:
    # (input [scope, shape], output [scope, shape], [find_patterns])
    input_info, output_info, find_patterns = tvm.testing.parameters(
        # 1. Texture (NCHW4c) -> Buffer (NCHW)
        (
            ["global.texture", (1, 1, 40, 40, 4)],
            ["", (1, 4, 40, 40)],
            [
                "float4 v_ = READ_IMAGEF(p0_comp, image_sampler, ((int2)(((convert_int(get_local_id(0))) % 40), ((((convert_int(get_group_id(0))) & 1) * 20) + ((convert_int(get_local_id(0))) / 40)))));",
                "out_sum[(((convert_int(get_group_id(0))) * 800) + (convert_int(get_local_id(0))))] = (((float*)&v_)[((convert_int(get_group_id(0))) >> 1)] + (((float*)&v_)[((convert_int(get_group_id(0))) >> 1)] + ((float*)&v_)[((convert_int(get_group_id(0))) >> 1)]));",
            ],
        ),
        # 2. Buffer (NCHW4c) -> Buffer (NCHW)
        (
            ["", (1, 1, 40, 40, 4)],
            ["", (1, 4, 40, 40)],
            [
                "out_sum[(((convert_int(get_group_id(0))) * 800) + (convert_int(get_local_id(0))))] = (p0_comp[(((((convert_int(get_group_id(0))) & 1) * 3200) + ((convert_int(get_local_id(0))) * 4)) + ((convert_int(get_group_id(0))) >> 1))] + (p0_comp[(((((convert_int(get_group_id(0))) & 1) * 3200) + ((convert_int(get_local_id(0))) * 4)) + ((convert_int(get_group_id(0))) >> 1))] + p0_comp[(((((convert_int(get_group_id(0))) & 1) * 3200) + ((convert_int(get_local_id(0))) * 4)) + ((convert_int(get_group_id(0))) >> 1))]));"
            ],
        ),
    )
    dtype = tvm.testing.parameter("float32")

    @tvm.testing.parametrize_targets("opencl")
    def test_texture_to_scalar_reuse_ssa_fp32(
        self, input_info, output_info, find_patterns, dtype, target
    ):
        texture_to_scalar_reuse_ssa_common(
            target, input_info, output_info, find_patterns, dtype, "float32"
        )


class TestLocalArrayToTexture:
    # 1. conv2d(Texture(NCHW4c), Texture(OIHW4o)) -> local_array[4] -> Texture (NCHW4c)
    input_shape1, input_shape2, output_shape, find_patterns = tvm.testing.parameters(
        (
            (1, 1, 40, 40, 4),
            (2, 4, 3, 3, 4),
            (1, 2, 38, 38, 4),
            [
                "float out_local[4];",
                "float4 v_ = READ_IMAGEF(p1_comp, image_sampler, ((int2)(((((convert_int(get_group_id(0))) * 14) + (convert_int(get_local_id(0)))) % 38), (((((convert_int(get_group_id(0))) * 64) + ((convert_int(get_local_id(0))) >> 1)) % 722) / 19))));",
                "float4 v__1 = READ_IMAGEF(p2_comp, image_sampler, ((int2)(rw, (((((((convert_int(get_group_id(0))) * 32) + ((convert_int(get_local_id(0))) >> 2)) / 361) * 12) + (rcb * 3)) + rh))));",
                "out_local[cb_c] = (out_local[cb_c] + (((float*)&v_)[rcb] * ((float*)&v__1)[cb_c]));",
                "write_imagef(out, (int2)(((((convert_int(get_group_id(0))) * 14) + (convert_int(get_local_id(0)))) % 38), ((((convert_int(get_group_id(0))) * 64) + ((convert_int(get_local_id(0))) >> 1)) / 19)), vload4(0, out_local + 0));",
            ],
        ),
    )
    dtype = tvm.testing.parameter("float32")

    @tvm.testing.parametrize_targets("opencl")
    def test_local_array_to_texture(
        self, input_shape1, input_shape2, output_shape, find_patterns, dtype, target
    ):
        def _compute():
            p1 = te.placeholder(input_shape1, name="p1", dtype=dtype)
            p1_comp = te.compute(input_shape1, lambda *i: p1(*i), name="p1_comp")
            p2 = te.placeholder(input_shape2, name="p2", dtype=dtype)
            p2_comp = te.compute(input_shape2, lambda *i: p2(*i), name="p2_comp")
            KH, KW = input_shape2[2], input_shape2[3]
            IC, ICB = input_shape1[1], input_shape1[4]
            rh = te.reduce_axis((0, KH), name="rh")
            rw = te.reduce_axis((0, KW), name="rw")
            rc = te.reduce_axis((0, IC), name="rc")
            rcb = te.reduce_axis((0, ICB), name="rcb")
            out = te.compute(
                output_shape,
                lambda n, c, h, w, cb: te.sum(
                    (p1_comp[n, rc, h, w, rcb] * p2_comp[c, rc * ICB + rcb, rh, rw, cb]).astype(
                        dtype
                    ),
                    axis=[rh, rw, rc, rcb],
                ),
                name="out",
            )
            dummy_out = te.compute(output_shape, lambda *i: out(*i), name="dummy_out")
            return p1, p2, dummy_out

        def _schedule(dummy_out):
            from tvm.topi.adreno.utils import bind_data_copy

            s = te.create_schedule(dummy_out.op)
            out = s[dummy_out].op.input_tensors[0]
            p1_comp, p2_comp = s[out].op.input_tensors
            bind_data_copy(s[p1_comp])
            s[p1_comp].set_scope("global.texture")
            bind_data_copy(s[p2_comp])
            s[p2_comp].set_scope("global.texture")
            OL = s.cache_write(out, "local")
            n, c, h, w, cb = s[out].op.axis
            fused = s[out].fuse(n, c, h, w)
            bx, tx = s[out].split(fused, 128)
            s[out].reorder(bx, tx, cb)
            s[out].vectorize(cb)
            s[out].set_scope("global.texture")
            s[out].bind(bx, te.thread_axis("blockIdx.x"))
            s[out].bind(tx, te.thread_axis("threadIdx.x"))
            s[OL].compute_at(s[out], tx)
            bind_data_copy(s[dummy_out])
            return s

        p1, p2, dummy_out = _compute()
        s = _schedule(dummy_out)

        fun = tvm.build(s, [p1, p2, dummy_out], target)
        dev = tvm.device(target, 0)
        opencl_source = fun.imported_modules[0].get_source()
        start_idx = 0
        for pattern in find_patterns:
            start_idx = opencl_source.find(pattern, start_idx)
            assert start_idx > -1

        input_np1 = np.random.uniform(size=[i for i in input_shape1]).astype(dtype)
        input_np2 = np.random.uniform(size=[i for i in input_shape2]).astype(dtype)
        input_tvm1 = tvm.nd.array(input_np1, dev)
        input_tvm2 = tvm.nd.array(input_np2, dev)
        c = tvm.nd.empty(output_shape, dtype, dev)
        fun(input_tvm1, input_tvm2, c)


if __name__ == "__main__":
    tvm.testing.main()
