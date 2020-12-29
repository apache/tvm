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
# pylint: disable=invalid-name,unused-variable,unused-argument
# pylint: disable=too-many-arguments,too-many-locals
# pylint: disable=too-many-statements
"""Winograd template for cuda backend"""

import tvm
from tvm import te
from tvm import autotvm
from .. import nn
from ..utils import get_const_int, get_const_tuple, traverse_inline
from ..nn.winograd_util import winograd_transform_matrices
from .tensor_intrin import intrin_wmma_load_matrix_A
from .tensor_intrin import intrin_wmma_load_matrix_W
from .tensor_intrin import intrin_wmma_store_matrix
from .tensor_intrin import intrin_wmma_gemm


def _infer_tile_size(data, kernel):
    """Compute the tile size"""
    N, H, W, CI = get_const_tuple(data.shape)
    if H % 8 == 0:
        return 4
    return 2


def schedule_bgemm_tensorcore(cfg, s, bgemm, data_pack, kernel_pack):
    """Schedule for bgemm tensorcore"""
    A = data_pack
    B = kernel_pack
    C = bgemm
    _, _, P, out_dim = get_const_tuple(C.shape)
    out_dtype = C.dtype

    # Explicit memory access
    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AF = s.cache_read(AS, "wmma.matrix_a", [C])
    BF = s.cache_read(BS, "wmma.matrix_b", [C])
    CF = s.cache_write(C, "wmma.accumulator")
    CS = s.cache_read(CF, "shared", [C])

    # Create tuning space
    cfg.define_knob("block_row_warps", [1, 2, 4])
    cfg.define_knob("block_col_warps", [1, 2, 4])
    cfg.define_knob("warp_row_tiles", [1, 2, 4, 8])
    cfg.define_knob("warp_col_tiles", [1, 2, 4, 8])
    cfg.define_knob("chunk", [1, 2, 4, 8])
    cfg.define_knob("offset", [0, 1, 2, 4, 8])
    cfg.define_knob("offsetCS", [0, 1, 2, 4, 8])
    cfg.define_knob("vec", [1, 2, 4, 8])

    # Ensure that the default parameters are applicable when autotvm is not in use
    if P % 16 == 0 and out_dim % 16 == 0:
        cfg.define_knob("wmma_m", [16, 8, 32])
    elif P % 32 == 0 and out_dim % 8 == 0:
        cfg.define_knob("wmma_m", [32, 16, 8])
    elif P % 8 == 0 and out_dim % 32 == 0:
        cfg.define_knob("wmma_m", [8, 16, 32])

    warp_size = 32
    wmma_k = 16
    block_row_warps = cfg["block_row_warps"].val
    block_col_warps = cfg["block_col_warps"].val
    warp_row_tiles = cfg["warp_row_tiles"].val
    warp_col_tiles = cfg["warp_col_tiles"].val
    chunk = cfg["chunk"].val
    offsetAB = cfg["offset"].val
    offsetCS = cfg["offsetCS"].val
    wmma_m = cfg["wmma_m"].val
    vec = cfg["vec"].val

    if wmma_m == 16:
        wmma_n = 16
    elif wmma_m == 8:
        wmma_n = 32
    elif wmma_m == 32:
        wmma_n = 8

    # Define the stride of intrin functions
    AS_align = chunk * wmma_k + offsetAB
    BS_align = warp_col_tiles * block_col_warps * wmma_n + offsetAB
    CS_align = warp_col_tiles * block_col_warps * wmma_n + offsetCS
    AS_stride = [AS_align, 1]
    BS_stride = [BS_align, 1]
    AF_stride = [wmma_k, 1]
    BF_stride = [wmma_n * warp_col_tiles, 1]
    CF_stride = [warp_col_tiles * wmma_n, 1]
    CS_stride = [CS_align, 1]
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    # Schedule for computation
    block_factor_b = wmma_m * warp_row_tiles * block_row_warps
    block_factor_o = wmma_n * warp_col_tiles * block_col_warps
    alpha_1, alpha_2, b, o = C.op.axis
    block_k = s[C].fuse(alpha_1, alpha_2)
    block_i, bc = s[C].split(b, factor=block_factor_b)
    block_j, oc = s[C].split(o, factor=block_factor_o)
    s[C].reorder(block_k, block_i, block_j, bc, oc)
    t = s[C].fuse(bc, oc)
    t, vi = s[C].split(t, factor=vec)
    t, tx = s[C].split(t, factor=warp_size)
    t, ty = s[C].split(t, factor=block_row_warps)
    t, tz = s[C].split(t, factor=block_col_warps)
    s[C].bind(block_k, block_z)
    s[C].bind(block_i, block_x)
    s[C].bind(block_j, block_y)
    s[C].bind(tz, thread_z)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].vectorize(vi)

    # Schedule for wmma store
    s[CS].compute_at(s[C], block_j)
    _, _, bb, oo = CS.op.axis
    s[CS].storage_align(bb, CS_align - 1, CS_align)
    bb, bbi = s[CS].split(bb, factor=wmma_m)
    oo, ooi = s[CS].split(oo, factor=wmma_n)
    bb, bbii = s[CS].split(bb, factor=warp_row_tiles)
    oo, ooii = s[CS].split(oo, factor=warp_col_tiles)
    s[CS].reorder(bb, oo, bbii, ooii, bbi, ooi)

    # Schedule for wmma computation
    s[CF].compute_at(s[CS], oo)
    _, _, warp_i, warp_j = CF.op.axis
    warp_i, _ii = s[CF].split(warp_i, factor=wmma_m)
    warp_j, _jj = s[CF].split(warp_j, factor=wmma_n)
    (k,) = CF.op.reduce_axis
    k, _k = s[CF].split(k, factor=wmma_k)
    ko, ki = s[CF].split(k, factor=chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)

    # Schedule for  wmma_matrix_a load
    s[AF].compute_at(s[CF], ki)
    _, _, b, i = AF.op.axis
    b, b_ii = s[AF].split(b, factor=wmma_m)
    i, i_jj = s[AF].split(i, factor=wmma_k)
    s[AF].reorder(b, i, b_ii, i_jj)

    # Schedule for  wmma_matrix_b load
    s[BF].compute_at(s[CF], ki)
    _, _, i, o = BF.op.axis
    o, o_ii = s[BF].split(o, factor=wmma_n)
    i, i_ii = s[BF].split(i, factor=wmma_k)
    s[BF].reorder(i, o, i_ii, o_ii)

    # Schedule for A's(B's) shared memory load
    def shared_shedule(stage, strides):
        s[stage].compute_at(s[CF], ko)
        _, _, xo, yo = stage.op.axis
        s[stage].storage_align(xo, strides - 1, strides)
        t = s[stage].fuse(xo, yo)
        t, vi = s[stage].split(t, factor=vec)
        t, tx = s[stage].split(t, factor=warp_size)
        t, ty = s[stage].split(t, factor=block_row_warps)
        _, tz = s[stage].split(t, factor=block_col_warps)
        s[stage].bind(ty, thread_y)
        s[stage].bind(tz, thread_z)
        s[stage].bind(tx, thread_x)
        s[stage].vectorize(vi)

    shared_shedule(AS, AS_align)
    shared_shedule(BS, BS_align)

    shape = (wmma_m, wmma_n, wmma_k)
    in_dtype = "float16"
    AL_gemm = te.placeholder((wmma_m, wmma_k), name="AL_gemm", dtype=in_dtype)
    BL_gemm = te.placeholder((wmma_k, wmma_n), name="BL_gemm", dtype=in_dtype)
    k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
    CL_compute = te.compute(
        (wmma_m, wmma_n),
        lambda ii, jj: te.sum(
            AL_gemm[ii, k_gemm].astype(out_dtype) * BL_gemm[k_gemm, jj].astype(out_dtype),
            axis=k_gemm,
        ),
        name="CL_compute",
    )

    # Lower the computation loops down to TensorCore hardware intrinsics
    # by mapping the tensorcore to tensor intrinsics
    s[AF].tensorize(
        b_ii,
        intrin_wmma_load_matrix_A(
            AF_stride, AS_stride, shape, "row_major", (wmma_m, wmma_k), (wmma_m, wmma_k), "float16"
        ),
    )
    s[BF].tensorize(
        i_ii,
        intrin_wmma_load_matrix_W(
            BF_stride, BS_stride, shape, "row_major", (wmma_k, wmma_n), (wmma_k, wmma_n), "float16"
        ),
    )
    s[CF].tensorize(
        _ii, intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride, CF_stride, shape)
    )
    s[CS].tensorize(
        bbi,
        intrin_wmma_store_matrix(
            CS_stride, CF_stride, shape, out_dtype, (wmma_m, wmma_n), (wmma_m, wmma_n)
        ),
    )


def schedule_bgemm_direct(cfg, s, bgemm, data_pack, kernel_pack):
    """Schedule for bgemm direct"""
    b1, b2, y, x = s[bgemm].op.axis
    rc = s[bgemm].op.reduce_axis[0]
    alpha = get_const_int(b1.dom.extent)

    # Create tuning space
    cfg.define_split(
        "tile_b", cfg.axis(alpha * alpha), num_outputs=4, filter=lambda x: x.size[-3:] == [1, 1, 1]
    )
    cfg.define_split("tile_y", y, num_outputs=4)
    cfg.define_split("tile_x", x, num_outputs=4)
    cfg.define_split("tile_rc", rc, num_outputs=2)
    cfg.define_knob("offset_bgemm", [0, 1, 2, 4, 8])
    cfg.define_knob("vector_bgemm", [1, 2, 4, 8])
    offset_bgemm = cfg["offset_bgemm"].val
    vector_bgemm = cfg["vector_bgemm"].val

    C = bgemm
    A0, B0 = kernel_pack, data_pack

    # Designate the memory hierarchy
    OL = s.cache_write(C, "local")
    AA = s.cache_read(A0, "shared", [OL])
    BB = s.cache_read(B0, "shared", [OL])

    # Tile and bind spatial axes
    b = s[bgemm].fuse(b1, b2)
    bgemm_scope, b = s[bgemm].split(b, nparts=1)
    bz, vz, tz, zi = cfg["tile_b"].apply(s, C, b)
    by, vy, ty, yi = cfg["tile_y"].apply(s, C, y)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, C, x)
    s[C].bind(bz, te.thread_axis("blockIdx.z"))
    s[C].bind(by, te.thread_axis("blockIdx.y"))
    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(vz, te.thread_axis("vthread"))
    s[C].bind(vy, te.thread_axis("vthread"))
    s[C].bind(vx, te.thread_axis("vthread"))
    s[C].bind(tz, te.thread_axis("threadIdx.z"))
    s[C].bind(ty, te.thread_axis("threadIdx.y"))
    s[C].bind(tx, te.thread_axis("threadIdx.x"))
    s[C].reorder(bgemm_scope, bz, by, bx, vz, vy, vx, tz, ty, tx, zi, yi, xi)

    # Tile reduction axes
    s[OL].compute_at(s[C], tx)
    b1, b2, y, x = s[OL].op.axis
    b = s[OL].fuse(b1, b2)
    (rc,) = s[OL].op.reduce_axis
    rco, rci = cfg["tile_rc"].apply(s, OL, rc)
    s[OL].reorder(rco, b, y, x, rci)

    s[AA].compute_at(s[OL], rco)
    _, _, k, n = s[AA].op.axis
    AA_align = offset_bgemm + cfg["tile_x"].size[1] * cfg["tile_x"].size[2] * cfg["tile_x"].size[3]
    s[AA].storage_align(k, AA_align - 1, AA_align)

    s[BB].compute_at(s[OL], rco)
    _, _, m, k = s[BB].op.axis
    BB_align = offset_bgemm + cfg["tile_rc"].size[1]
    s[BB].storage_align(m, BB_align - 1, BB_align)

    # Schedule for A and B shared memory load
    for load in [AA, BB]:
        fused = s[load].fuse(*list(s[load].op.axis))
        fused, ti = s[load].split(fused, factor=vector_bgemm)
        fused, tx = s[load].split(fused, cfg["tile_x"].size[2])
        fused, ty = s[load].split(fused, cfg["tile_y"].size[2])
        fused, tz = s[load].split(fused, cfg["tile_b"].size[2])
        s[load].bind(tz, te.thread_axis("threadIdx.z"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))
        s[load].bind(tx, te.thread_axis("threadIdx.x"))
        s[load].vectorize(ti)


def nhwc_winograd_cuda(
    cfg, data, kernel, strides, padding, dilation, out_dtype, use_tensorcore, pre_computed
):
    """Compute declaration for winograd"""
    tile_size = _infer_tile_size(data, kernel)
    N, H, W, CI = get_const_tuple(data.shape)

    if isinstance(N, tvm.tir.Any):
        N = tvm.te.size_var("n")

    if not isinstance(H, int) or not isinstance(W, int):
        raise RuntimeError(
            "cuda winograd nhwc conv2d doesn't support dynamic \
                           input height or width."
        )

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    HSTR, WSTR = (strides, strides) if isinstance(strides, int) else strides

    if not pre_computed:  # Kernel tensor is raw tensor, do strict check
        if dilation_h != 1 or dilation_w != 1:
            kernel = nn.dilate(kernel, (dilation_h, dilation_w, 1, 1))
        KH, KW, CI, CO = get_const_tuple(kernel.shape)
        alpha = KW + tile_size - 1
        assert HSTR == 1 and WSTR == 1 and KH == KW
    else:
        # Kernel tensor is pre-transfomred. This op is created by conv2d_alter_op.
        # Dilation is not supported
        alpha, _, CI, CO = get_const_tuple(kernel.shape)
        KH = KW = alpha + 1 - tile_size
        assert HSTR == 1 and WSTR == 1 and dilation_h == 1 and dilation_w == 1

    pt, pl, pb, pr = nn.get_pad_tuple(padding, (KH, KW))
    data_pad = nn.pad(data, (0, pt, pl, 0), (0, pb, pr, 0), name="data_pad")

    r = KW
    m = tile_size
    H = (H + pt + pb - KH) // HSTR + 1
    W = (W + pl + pr - KW) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m
    P = N * nH * nW if isinstance(N, int) else nH * nW

    # Determine whether the shape is available with tensorcore
    shape_judge = (
        (P % 16 == 0 and CI % 16 == 0 and CO % 16 == 0)
        or (P % 8 == 0 and CI % 16 == 0 and CO % 32 == 0)
        or (P % 32 == 0 and CI % 16 == 0 and CO % 8 == 0)
    )

    if shape_judge and use_tensorcore:
        trans_type = "float16"
    else:
        trans_type = data.dtype

    # Compute transform matrix
    A, _, _ = winograd_transform_matrices(m, r, out_dtype)
    _, B, G = winograd_transform_matrices(m, r, data.dtype)

    # Transform kernel
    if not pre_computed:
        # Check if we are currently tuning, if so we want to avoid counting
        # prepacking in time costs. Just use a placeholder with the packed shape instead.
        if autotvm.GLOBAL_SCOPE.in_tuning:
            kernel_pack = te.placeholder(
                (alpha, alpha, CI, CO), dtype=kernel.dtype, name="kernel_pack"
            )
        else:
            r_kh = te.reduce_axis((0, KH), name="r_kh")
            r_kw = te.reduce_axis((0, KW), name="r_kw")
            kernel_pack = te.compute(
                (alpha, alpha, CI, CO),
                lambda eps, nu, ci, co: te.sum(
                    (kernel[r_kh][r_kw][ci][co]) * G[eps][r_kh] * G[nu][r_kw], axis=[r_kh, r_kw]
                ),
                name="kernel_pack",
            )
    else:
        kernel_pack = kernel

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod

    # Pack input tile
    input_tile = te.compute(
        (P, CI, alpha, alpha),
        lambda p, c, eps, nu: data_pad[
            idxdiv(p, (nH * nW)), idxmod(idxdiv(p, nW), nH) * m + eps, idxmod(p, nW) * m + nu, c
        ],
        name="d",
    )

    # Transform data
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_b")
    data_pack = te.compute(
        (alpha, alpha, P, CI),
        lambda eps, nu, p, ci: te.sum(
            input_tile[p][ci][r_a][r_b] * B[r_a][eps] * B[r_b][nu], axis=[r_a, r_b]
        ),
        name="data_pack",
    )

    # Convert data type of input feature maps and weights for tensorcore
    Transdata = te.compute(
        data_pack.shape, lambda eps, nu, p, ci: data_pack[eps, nu, p, ci].astype(trans_type)
    )
    TransFilter = te.compute(
        kernel_pack.shape, lambda eps, nu, ci, co: kernel_pack[eps, nu, ci, co].astype(trans_type)
    )

    # Do batch gemm
    ci = te.reduce_axis((0, CI), name="ci")
    bgemm = te.compute(
        (alpha, alpha, P, CO),
        lambda eps, nu, p, co: te.sum(
            (Transdata[eps][nu][p][ci]).astype(out_dtype)
            * (TransFilter[eps][nu][ci][co]).astype(out_dtype),
            axis=[ci],
        ),
        name="bgemm",
    )

    # Inverse transform
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_a")
    inverse = te.compute(
        (P, CO, m, m),
        lambda p, co, vh, vw: te.sum(
            bgemm[r_a][r_b][p][co] * A[r_a][vh] * A[r_b][vw], axis=[r_a, r_b]
        ),
        name="inverse",
    )

    # Output
    output = te.compute(
        (N, H, W, CO),
        lambda n, h, w, co: inverse[
            n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m), co, idxmod(h, m), idxmod(w, m)
        ],
        name="output",
        tag="conv2d_nhwc_winograd",
    )
    if isinstance(N, int):
        cfg.add_flop(2 * N * CO * H * W * CI * KH * KW)
    return output


def data_weight_transform(s, data_trans, input_tile, thread_num_trans, offset_trans, trans_tag):
    """Schedule for data or kernel transform"""
    kernel_align = thread_num_trans + offset_trans
    indata_s = s.cache_read(input_tile, "shared", [data_trans])
    data_l = s.cache_write(data_trans, "local")
    # Schedule for data or kernel transform
    eps, nu, p, c = s[data_trans].op.axis

    block_x, thread_x = s[data_trans].split(c, thread_num_trans)
    block_x = s[data_trans].fuse(p, block_x)
    s[data_trans].reorder(block_x, thread_x, eps, nu)
    s[data_trans].bind(thread_x, te.thread_axis("threadIdx.x"))
    s[data_trans].bind(block_x, te.thread_axis("blockIdx.x"))

    s[data_l].compute_at(s[data_trans], thread_x)
    eps_l, nu_l, p_l, c_l = s[data_l].op.axis
    r_a, r_b = s[data_l].op.reduce_axis
    block_x_l, thread_x_l = s[data_l].split(c_l, thread_num_trans)
    block_x_l = s[data_l].fuse(p_l, block_x_l)

    s[data_l].reorder(block_x_l, thread_x_l, eps_l, nu_l, r_a, r_b)

    for axis in [eps_l, nu_l, r_a, r_b]:
        s[data_l].unroll(axis)

    # Schedule for share memory load
    s[indata_s].compute_at(s[data_l], block_x_l)
    if trans_tag == "data":
        p_is, c_is, eps_is, nu_is = s[indata_s].op.axis
        data_align = (
            get_const_int(eps_is.dom.extent) * get_const_int(nu_is.dom.extent) + offset_trans
        )
        s[indata_s].storage_align(c_is, data_align - 1, data_align)
        block_x_is, thread_x_is = s[indata_s].split(c_is, thread_num_trans)
        s[indata_s].bind(thread_x_is, te.thread_axis("threadIdx.x"))
    else:
        eps_is, nu_is, ci_is, co_is = s[indata_s].op.axis
        s[indata_s].storage_align(nu_is, kernel_align - 1, kernel_align)
        block_x_is, thread_x_is = s[indata_s].split(co_is, thread_num_trans)
        s[indata_s].reorder(ci_is, block_x_is, eps_is, nu_is, thread_x_is)
        s[indata_s].bind(thread_x_is, te.thread_axis("threadIdx.x"))


def schedule_nhwc_winograd_cuda(cfg, s, output, use_tensorcore, pre_computed):
    """Schedule winograd template"""
    # Get stages
    inverse = s[output].op.input_tensors[0]
    bgemm, A = s[inverse].op.input_tensors
    Transdata, TransFilter = s[bgemm].op.input_tensors
    data_pack = s[Transdata].op.input_tensors[0]
    kernel_pack = s[TransFilter].op.input_tensors[0]
    s[Transdata].compute_inline()
    s[TransFilter].compute_inline()

    input_tile, B = s[data_pack].op.input_tensors
    pad_data = s[input_tile].op.input_tensors[0]

    # Define the stride of intrin functions
    cfg.define_knob("thread_num_inverse", [1, 32, 64, 128, 256])
    cfg.define_knob("thread_num_data", [1, 32, 64, 128, 256])
    cfg.define_knob("thread_num_kernel", [1, 32, 64, 128, 256])
    cfg.define_knob("offset_inverse", [0, 2, 4])
    cfg.define_knob("offset_data", [0, 1, 2, 4])
    cfg.define_knob("offset_kernel", [0, 1, 2, 4])
    cfg.define_knob("inverse_in_vector", [1, 2, 4])

    thread_num_data = cfg["thread_num_data"].val
    thread_num_kernel = cfg["thread_num_kernel"].val
    thread_num_inverse = cfg["thread_num_inverse"].val
    offset_data = cfg["offset_data"].val
    offset_kernel = cfg["offset_kernel"].val
    offset_inverse = cfg["offset_inverse"].val
    inverse_in_vector = cfg["inverse_in_vector"].val

    # Data transform
    s[B].compute_inline()
    data_weight_transform(s, data_pack, input_tile, thread_num_data, offset_data, trans_tag="data")
    s[input_tile].compute_inline()
    s[pad_data].compute_inline()

    # Kernel transform
    if not pre_computed and not autotvm.GLOBAL_SCOPE.in_tuning:
        kernel, G = s[kernel_pack].op.input_tensors
        s[G].compute_inline()
        data_weight_transform(
            s, kernel_pack, kernel, thread_num_kernel, offset_kernel, trans_tag="kernel"
        )
    else:
        kernel = kernel_pack

    if isinstance(kernel.op, tvm.te.ComputeOp) and "dilate" in kernel.op.tag:
        s[kernel].compute_inline()

    b1, b2, y, x = s[bgemm].op.axis
    alpha = get_const_int(b1.dom.extent)
    _, _, P, CI = get_const_tuple(Transdata.shape)
    _, _, _, CO = get_const_tuple(TransFilter.shape)

    # Determine whether the shape is available with tensorcore
    shape_judge = (
        (P % 16 == 0 and CI % 16 == 0 and CO % 16 == 0)
        or (P % 8 == 0 and CI % 16 == 0 and CO % 32 == 0)
        or (P % 32 == 0 and CI % 16 == 0 and CO % 8 == 0)
    )

    if shape_judge and use_tensorcore:
        schedule_bgemm_tensorcore(cfg, s, bgemm, Transdata, TransFilter)
    else:
        schedule_bgemm_direct(cfg, s, bgemm, Transdata, TransFilter)

    # Schedule inverse, output and fusion
    if output.op in s.outputs:
        OL = None
    else:
        OL = output
        s[OL].set_scope("local")
        output = s.outputs[0]

    s[A].compute_inline()
    inverse_s = s.cache_read(bgemm, "shared", [inverse])

    m = alpha - 3 + 1
    offset_inverse_in = offset_inverse
    vector_width_inverse_in = inverse_in_vector

    # Schedule for output
    n, h, w, co = s[output].op.axis
    ho, wo, hi, wi = s[output].tile(h, w, m, m)
    s[output].reorder(n, ho, wo, co, hi, wi)
    fused = s[output].fuse(n, ho, wo)

    block_x_s, thread_x_s = s[output].split(co, thread_num_inverse)
    block_x_s = s[output].fuse(fused, block_x_s)
    s[output].reorder(block_x_s, thread_x_s, hi, wi)

    if OL is not None:
        s[OL].compute_inline()

    # Schedule for inverse
    s[inverse].compute_at(s[output], thread_x_s)
    p_inv, co_inv, eps_inv, nu_inv = s[inverse].op.axis
    block_x_inv, thread_x_inv = s[inverse].split(co_inv, thread_num_inverse)
    r_a, r_b = s[inverse].op.reduce_axis
    for axis in [eps_inv, nu_inv, r_a, r_b]:
        s[inverse].unroll(axis)

    # Schedule for share memory load
    s[inverse_s].compute_at(s[output], block_x_s)
    eps_inv_s, nu_inv_s, p_inv_s, co_inv_s = s[inverse_s].op.axis
    inverse_in_align = offset_inverse_in + thread_num_inverse
    s[inverse_s].storage_align(p_inv_s, inverse_in_align - 1, inverse_in_align)
    block_x_inv_s, thread_x_inv_s = s[inverse_s].split(co_inv_s, thread_num_inverse)
    block_x_inv_s = s[inverse_s].fuse(p_inv_s, block_x_inv_s)
    s[inverse_s].reorder(block_x_inv_s, eps_inv_s, nu_inv_s, thread_x_inv_s)
    t = s[inverse_s].fuse(eps_inv_s, nu_inv_s, thread_x_inv_s)
    t, ti = s[inverse_s].split(t, factor=vector_width_inverse_in)
    t, tx = s[inverse_s].split(t, factor=thread_num_inverse)
    s[inverse_s].bind(tx, te.thread_axis("threadIdx.x"))
    s[inverse_s].vectorize(ti)

    s[output].bind(thread_x_s, te.thread_axis("threadIdx.x"))
    s[output].bind(block_x_s, te.thread_axis("blockIdx.x"))
    return s


@autotvm.register_topi_compute("conv2d_nhwc_winograd_direct.cuda")
def conv2d_nhwc_winograd_direct(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d with winograd for NHWC layout"""
    return nhwc_winograd_cuda(
        cfg,
        data,
        kernel,
        strides,
        padding,
        dilation,
        out_dtype,
        use_tensorcore=False,
        pre_computed=False,
    )


@autotvm.register_topi_schedule("conv2d_nhwc_winograd_direct.cuda")
def schedule_conv2d_nhwc_winograd_direct(cfg, outs):
    """TOPI schedule callback"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_nhwc_winograd" in op.tag:
            schedule_nhwc_winograd_cuda(
                cfg, s, op.output(0), use_tensorcore=False, pre_computed=False
            )

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_nhwc_winograd_tensorcore.cuda")
def conv2d_nhwc_winograd_tensorcore(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv2d with winograd for NHWC layout"""
    return nhwc_winograd_cuda(
        cfg,
        data,
        kernel,
        strides,
        padding,
        dilation,
        out_dtype,
        use_tensorcore=True,
        pre_computed=False,
    )


@autotvm.register_topi_schedule("conv2d_nhwc_winograd_tensorcore.cuda")
def schedule_conv2d_nhwc_winograd_tensorcore(cfg, outs):
    """TOPI schedule callback"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_nhwc_winograd" in op.tag:
            schedule_nhwc_winograd_cuda(
                cfg, s, op.output(0), use_tensorcore=True, pre_computed=False
            )

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_nhwc_winograd_direct_without_weight_transform.cuda")
def conv2d_nhwc_winograd_direct_without_weight_transform(
    cfg, data, kernel, strides, padding, dilation, out_dtype
):
    """Compute conv2d with winograd for NHWC layout"""
    return nhwc_winograd_cuda(
        cfg,
        data,
        kernel,
        strides,
        padding,
        dilation,
        out_dtype,
        use_tensorcore=False,
        pre_computed=True,
    )


@autotvm.register_topi_schedule("conv2d_nhwc_winograd_direct_without_weight_transform.cuda")
def schedule_conv2d_nhwc_winograd_direct_without_weight_transform(cfg, outs):
    """TOPI schedule callback"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_nhwc_winograd" in op.tag:
            schedule_nhwc_winograd_cuda(
                cfg, s, op.output(0), use_tensorcore=False, pre_computed=True
            )

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("conv2d_nhwc_winograd_tensorcore_without_weight_transform.cuda")
def conv2d_nhwc_winograd_tensorcore_without_weight_transform(
    cfg, data, kernel, strides, padding, dilation, out_dtype
):
    """Compute conv2d with winograd for NHWC layout"""
    return nhwc_winograd_cuda(
        cfg,
        data,
        kernel,
        strides,
        padding,
        dilation,
        out_dtype,
        use_tensorcore=True,
        pre_computed=True,
    )


@autotvm.register_topi_schedule("conv2d_nhwc_winograd_tensorcore_without_weight_transform.cuda")
def schedule_conv2d_nhwc_winograd_tensorcore_without_weight_transform(cfg, outs):
    """TOPI schedule callback"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv2d_nhwc_winograd" in op.tag:
            schedule_nhwc_winograd_cuda(
                cfg, s, op.output(0), use_tensorcore=True, pre_computed=True
            )

    traverse_inline(s, outs[0].op, _callback)
    return s
