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

"""sparse_dense schedule on x86"""
from tvm import te, tir, autotvm
from functools import partial, reduce

from ..transform import reshape
from ..utils import traverse_inline, get_const_int
from .utils import get_fp32_len


def schedule_sparse_dense(outs):
    """Create schedule for sparse dense"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        simd_width = get_fp32_len()
        if op.tag == "sparse_dense_sp_lhs_csrmm" or op.tag == "sparse_dense_sp_lhs_csrmm":
            (y_o, y_i) = s[op].split(s[op].op.axis[1], 2)
            fused = s[op].fuse(s[op].op.axis[0], y_o)
            s[op].parallel(fused)
            s[op].vectorize(y_i)
        elif op.tag == "sparse_dense_sp_rhs_bsrmm" or op.tag == "sparse_dense_sp_rhs_bsrmm":
            y_bsrmm = op.input_tensors[0]
            assert (
                y_bsrmm.op.tag == "sparse_dense_sp_rhs_bsrmm_block"
                or y_bsrmm.op.tag == "sparse_dense_sp_lhs_bsrmm_block"
            )
            y_reshape = op
            (m, num_blocks, b_r) = s[y_bsrmm].op.axis
            bs_r = get_const_int(b_r.dom.extent)
            (elem_idx, c) = s[y_bsrmm].op.reduce_axis
            s[y_bsrmm].reorder(num_blocks, m, elem_idx, b_r, c)
            s[y_bsrmm].vectorize(b_r)
            (m_o, n_o) = s[y_reshape].op.axis
            (noo, noi) = s[y_reshape].split(n_o, bs_r)
            s[y_bsrmm].compute_at(s[y_reshape], noi)
            s[y_reshape].vectorize(noi)
            if op != s[outs[0]].op:
                (y_o, y_i) = s[outs[0].op].split(s[outs[0].op].op.axis[1], 2 * simd_width)
                s[y_reshape].compute_at(s[outs[0]], y_o)
                s[outs[0].op].parallel(y_o)
                s[outs[0].op].vectorize(y_i)
            else:
                m_o_noo = s[y_reshape].fuse(m_o, noo)
                s[y_reshape].parallel(m_o_noo)

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv3x3_spNHWC.x86")
def spconv2d_3x3_nhwc(cfg, Data, Wdat, Wind, Wptr, layout="NHWC"):
    N, H, W, CI = [i.value for i in Data.shape]
    nElems, bsrR, bsrC = [i.value for i in Wdat.shape]
    CO = (Wptr.shape[0].value - 1) * bsrR

    Y, X, K = N * H * W, CO, 9 * CI
    cfg.define_split("tile_y", Y, num_outputs=3)
    cfg.define_split("tile_x", X // bsrR, num_outputs=2)
    cfg.add_flop(Y * (nElems * bsrC * bsrR * 2 - X))
    if cfg.is_fallback:
        cfg["tile_y"] = autotvm.task.space.SplitEntity([-1, 160, 8])
        cfg["tile_x"] = autotvm.task.space.SplitEntity([-1, 4])

    idxsplit = lambda x, y: reduce(lambda a, b: a[:-1] + [a[-1] % b, a[-1] // b], y, [x])

    @partial(te.compute, (Y, K), name="Im2Col")
    def Im2Col(row, col):
        jw, jh, jn = idxsplit(row, [W, H])
        jc, kw, kh = idxsplit(col, [CI, 3])
        ih, iw = jh + kh - 1, jw + kw - 1
        return tir.if_then_else(tir.all(0 <= ih, ih < H, 0 <= iw, iw < W), Data[jn, ih, iw, jc], 0)

    @partial(te.compute, (Y, X // bsrR, bsrR, bsrC), name="CC")
    def CC(drow, wrow, brow, bcol):
        row_start, row_end = Wptr[wrow], Wptr[wrow + 1]
        elem_idx = te.reduce_axis((0, row_end - row_start), name="elem_idx")
        elem = row_start + elem_idx
        return te.sum(
            Im2Col[drow, Wind[elem] * bsrC + bcol] * Wdat[elem, brow, bcol], axis=elem_idx
        )

    k = te.reduce_axis((0, bsrC), name="k")
    C = te.compute(
        (Y, X),
        lambda y, x: te.sum(CC[y, x // bsrR, x % bsrR, k], axis=k),
        name="C",
        tag="conv3x3_spNHWC",
    )
    return reshape(C, (N, H, W, CO))


@autotvm.register_topi_schedule("conv3x3_spNHWC.x86")
def schedule_spconv2d_3x3_nhwc(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv3x3_spNHWC":
            C = op
            (CC,) = op.input_tensors
            Wptr, Wind, Im2Col, Wdat = CC.op.input_tensors
            (Data,) = Im2Col.op.input_tensors
            bsrR = CC.shape[-2].value
            CI = Data.shape[-1].value

            y, x = s[C].op.axis
            yt, yo, yi = cfg["tile_y"].apply(s, C, y)
            xo, xi = s[C].split(x, factor=bsrR)
            xt, xo = cfg["tile_x"].apply(s, C, xo)
            (k,) = s[C].op.reduce_axis
            s[C].reorder(yt, xt, yo, xo, yi, xi, k)
            s[C].unroll(k)
            s[C].vectorize(xi)
            s[C].unroll(yi)

            s[CC].compute_at(s[C], xo)
            yi, xi, r, c = s[CC].op.axis
            (k,) = s[CC].op.reduce_axis
            s[CC].reorder(xi, k, yi, r, c)
            s[CC].unroll(c)
            s[CC].vectorize(r)
            s[CC].unroll(yi)

            s[Im2Col].compute_at(s[C], yo)
            yi, k = s[Im2Col].op.axis
            ko, ki = s[Im2Col].split(k, factor=CI)
            s[Im2Col].vectorize(ki)

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv3x3_spNCHW.x86")
def spconv2d_3x3_nchw(cfg, Data, Wdat, Wind, Wptr, layout="NCHW"):
    N, CI, H, W = [i.value for i in Data.shape]
    NNZ, VL, bsrC = [i.value for i in Wdat.shape]
    CO = (Wptr.shape[0].value - 1) * VL
    assert bsrC == 1

    cfg.add_flop(N * H * W * (NNZ * VL * bsrC * 2 - CO))
    cfg.define_split("tile_hw", H * W, num_outputs=3)
    cfg.define_split("tile_ckk", CI * 9, num_outputs=3)

    @partial(te.compute, (N, CI * 3 * 3, H * W), name="im2col")
    def Im2Col(n, ckk, hw):
        jh, jw = hw // W, hw % W
        ic, kh, kw = ckk // 9, ckk // 3 % 3, ckk % 3
        ih, iw = jh + kh - 1, jw + kw - 1
        return tir.if_then_else(tir.all(0 <= ih, ih < H, 0 <= iw, iw < W), Data[n, ic, ih, iw], 0)

    @partial(te.compute, (N, CO // VL, VL, bsrC, H * W), name="CC", tag="conv3x3_spNCHW")
    def CC(n, fo, fi, k, hw):
        row_start, row_end = Wptr[fo], Wptr[fo + 1]
        elem_idx = te.reduce_axis((0, row_end - row_start), name="elem_idx")
        elem = row_start + elem_idx
        return te.sum(Im2Col[n, Wind[elem] * bsrC + k, hw] * Wdat[elem, fi, k], axis=elem_idx)

    return reshape(CC, [N, CO, H, W])


@autotvm.register_topi_schedule("conv3x3_spNCHW.x86")
def schedule_spconv2d_3x3_nchw(cfg, outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv3x3_spNCHW":
            CC = op
            Wptr, Wind, im2col, Wdat = op.input_tensors
            (Data,) = im2col.op.input_tensors

            n, fo, fi, bc, hw = s[CC].op.axis
            (kk,) = s[CC].op.reduce_axis
            hw1, hw2, hw3 = cfg["tile_hw"].apply(s, CC, hw)
            s[CC].reorder(n, hw1, fo, hw2, kk, fi, bc, hw3)
            s[CC].unroll(fi)
            s[CC].unroll(bc)
            s[CC].vectorize(hw3)

            s[im2col].compute_at(s[CC], hw1)
            n, ckk, hw = s[im2col].op.axis
            ckk1, ckk2, ckk3 = cfg["tile_ckk"].apply(s, im2col, ckk)
            hw2, hw3 = s[im2col].split(hw, factor=cfg["tile_hw"].size[-1])
            s[im2col].reorder(n, ckk1, ckk2, hw2, ckk3, hw3)
            s[im2col].unroll(ckk3)
            s[im2col].vectorize(hw3)

    traverse_inline(s, outs[0].op, _callback)
    return s