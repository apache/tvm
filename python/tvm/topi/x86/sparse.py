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
from functools import partial, reduce
from tvm import te, tir, autotvm

from ..transform import reshape
from ..utils import traverse_inline, get_const_int
from .utils import get_simd_32bit_lanes


def schedule_sparse_dense(outs):
    """Create schedule for sparse dense"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        simd_width = get_simd_32bit_lanes()
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
def spconv2d_3x3_nhwc(cfg, data, wdat, wind, wptr, layout="NHWC"):
    """Sparse Conv2d 3x3 compute (NHWC)."""
    assert layout == "NHWC"
    nsamples, imh, imw, chanin = [i.value for i in data.shape]
    nelems, bsrr, bsrc = [i.value for i in wdat.shape]
    chanout = (wptr.shape[0].value - 1) * bsrr

    imglen, chanlen = nsamples * imh * imw, 9 * chanin
    cfg.define_split("tile_y", imglen, num_outputs=3)
    cfg.define_split("tile_x", chanout // bsrr, num_outputs=2)
    cfg.add_flop(imglen * (nelems * bsrc * bsrr * 2 - chanout))
    if cfg.is_fallback:
        cfg["tile_y"] = autotvm.task.space.SplitEntity([-1, 160, 8])
        cfg["tile_x"] = autotvm.task.space.SplitEntity([-1, 4])

    idxsplit = lambda x, y: reduce(lambda a, b: a[:-1] + [a[-1] % b, a[-1] // b], y, [x])

    @partial(te.compute, (imglen, chanlen), name="Im2Col")
    def im2col(row, col):
        j_w, j_h, j_n = idxsplit(row, [imw, imh])
        j_c, k_w, k_h = idxsplit(col, [chanin, 3])
        i_h, i_w = j_h + k_h - 1, j_w + k_w - 1
        return tir.if_then_else(
            tir.all(i_h >= 0, i_h < imh, i_w >= 0, i_w < imw), data[j_n, i_h, i_w, j_c], 0
        )

    @partial(te.compute, (imglen, chanout // bsrr, bsrr, bsrc), name="CC")
    def matmul(drow, wrow, brow, bcol):
        row_start, row_end = wptr[wrow], wptr[wrow + 1]
        elem_idx = te.reduce_axis((0, row_end - row_start), name="elem_idx")
        elem = row_start + elem_idx
        return te.sum(
            im2col[drow, wind[elem] * bsrc + bcol] * wdat[elem, brow, bcol], axis=elem_idx
        )

    sum_bsrc = te.reduce_axis((0, bsrc), name="k")
    ret = te.compute(
        (imglen, chanout),
        lambda y, x: te.sum(matmul[y, x // bsrr, x % bsrr, sum_bsrc], axis=sum_bsrc),
        name="C",
        tag="conv3x3_spNHWC",
    )
    return reshape(ret, (nsamples, imh, imw, chanout))


@autotvm.register_topi_schedule("conv3x3_spNHWC.x86")
def schedule_spconv2d_3x3_nhwc(cfg, outs):
    """Sparse Conv2d 3x3 schedule (NHWC)."""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv3x3_spNHWC":
            (matmul,) = op.input_tensors
            # wptr, wind, im2col, wdat
            _, _, im2col, _ = matmul.op.input_tensors
            (data,) = im2col.op.input_tensors
            bsrr = matmul.shape[-2].value
            chanin = data.shape[-1].value

            mm_y, mm_x = s[op].op.axis
            y_t, y_o, y_i = cfg["tile_y"].apply(s, op, mm_y)
            x_o, x_i = s[op].split(mm_x, factor=bsrr)
            x_t, x_o = cfg["tile_x"].apply(s, op, x_o)
            (sum_ax,) = s[op].op.reduce_axis
            s[op].reorder(y_t, x_t, y_o, x_o, y_i, x_i, sum_ax)
            s[op].unroll(sum_ax)
            s[op].vectorize(x_i)
            s[op].unroll(y_i)

            s[matmul].compute_at(s[op], x_o)
            y_i, x_i, bsrr, bsrc = s[matmul].op.axis
            (sum_ax,) = s[matmul].op.reduce_axis
            s[matmul].reorder(x_i, sum_ax, y_i, bsrr, bsrc)
            s[matmul].unroll(bsrc)
            s[matmul].vectorize(bsrr)
            s[matmul].unroll(y_i)

            s[im2col].compute_at(s[op], y_o)
            y_i, sum_ax = s[im2col].op.axis
            _, k_i = s[im2col].split(sum_ax, factor=chanin)
            s[im2col].vectorize(k_i)

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv3x3_spNCHW.x86")
def spconv2d_3x3_nchw(cfg, data, wdat, wind, wptr, layout="NCHW"):
    """Sparse Conv2d 3x3 compute (NCHW)."""
    nsamples, chanin, imgh, imgw = [i.value for i in data.shape]
    nelems, veclen, bsrc = [i.value for i in wdat.shape]
    chanout = (wptr.shape[0].value - 1) * veclen
    assert bsrc == 1 and layout == "NCHW"

    cfg.add_flop(nsamples * imgh * imgw * (nelems * veclen * bsrc * 2 - chanout))
    cfg.define_split("tile_hw", imgh * imgw, num_outputs=3)
    cfg.define_split("tile_ckk", chanin * 9, num_outputs=3)

    @partial(te.compute, (nsamples, chanin * 3 * 3, imgh * imgw), name="im2col")
    def im2col(nsamples, ckk, imglen):
        j_h, j_w = imglen // imgw, imglen % imgw
        i_c, k_h, k_w = ckk // 9, ckk // 3 % 3, ckk % 3
        i_h, i_w = j_h + k_h - 1, j_w + k_w - 1
        return tir.if_then_else(
            tir.all(i_h >= 0, i_h < imgh, i_w >= 0, i_w < imgw), data[nsamples, i_c, i_h, i_w], 0
        )

    @partial(
        te.compute,
        (nsamples, chanout // veclen, veclen, bsrc, imgh * imgw),
        name="CC",
        tag="conv3x3_spNCHW",
    )
    def matmul(nsamples, f_o, f_i, bsrk, imglen):
        row_start, row_end = wptr[f_o], wptr[f_o + 1]
        elem_idx = te.reduce_axis((0, row_end - row_start), name="elem_idx")
        elem = row_start + elem_idx
        return te.sum(
            im2col[nsamples, wind[elem] * bsrc + bsrk, imglen] * wdat[elem, f_i, bsrk],
            axis=elem_idx,
        )

    return reshape(matmul, [nsamples, chanout, imgh, imgw])


@autotvm.register_topi_schedule("conv3x3_spNCHW.x86")
def schedule_spconv2d_3x3_nchw(cfg, outs):
    """Sparse Conv2d 3x3 schedule (NCHW)."""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "conv3x3_spNCHW":
            # wptr, wind, im2col, wdat
            _, _, im2col, _ = op.input_tensors

            n_samples, f_o, f_i, b_c, imglen = s[op].op.axis
            (sum_ax,) = s[op].op.reduce_axis
            hw1, hw2, hw3 = cfg["tile_hw"].apply(s, op, imglen)
            s[op].reorder(n_samples, hw1, f_o, hw2, sum_ax, f_i, b_c, hw3)
            s[op].unroll(f_i)
            s[op].unroll(b_c)
            s[op].vectorize(hw3)

            s[im2col].compute_at(s[op], hw1)
            n_samples, ckk, imglen = s[im2col].op.axis
            ckk1, ckk2, ckk3 = cfg["tile_ckk"].apply(s, im2col, ckk)
            hw2, hw3 = s[im2col].split(imglen, factor=cfg["tile_hw"].size[-1])
            s[im2col].reorder(n_samples, ckk1, ckk2, hw2, ckk3, hw3)
            s[im2col].unroll(ckk3)
            s[im2col].vectorize(hw3)

    traverse_inline(s, outs[0].op, _callback)
    return s
