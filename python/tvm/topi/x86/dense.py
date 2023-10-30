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
# pylint: disable=invalid-name,too-many-locals,unused-argument
# pylint: disable=no-value-for-parameter,unused-variable
"""x86 dense operators"""
from __future__ import absolute_import as _abs

import tvm
from tvm import autotvm, te
from tvm.autotvm.task.space import SplitEntity
from tvm.contrib import cblas, dnnl, mkl
from tvm.target.x86 import get_simd_32bit_lanes
from tvm.target.codegen import target_has_features

from .. import generic, tag
from ..utils import get_const_tuple, traverse_inline
from .tensor_intrin import (
    acc_32x32_int32_sapphirerapids,
    dot_16x1x16_uint8_int8_int32,
    dot_32x128x32_u8s8s32_sapphirerapids,
)


def _schedule_dense_pack_template(cfg, s, C, O):
    A, packedB = s[C].op.input_tensors

    CC = s.cache_write(C, "global")
    y, x = s[C].op.axis
    (k,) = s[CC].op.reduce_axis

    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(xt, yt, yo, xo, yi, xi)
    xyt = s[C].fuse(xt, yt)
    if C == O:
        s[C].parallel(xyt)
    xyo = s[C].fuse(yo, xo)
    s[C].unroll(yi)
    s[C].vectorize(xi)

    s[CC].compute_at(s[C], xyo)
    y, x = s[CC].op.axis
    ko, ki = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(ko, ki, y, x)
    s[CC].vectorize(x)

    tile_inner = cfg["tile_inner"].size[-1]
    if tile_inner > 1:
        yo, yi = s[CC].split(y, tile_inner)
        s[CC].reorder(ko, yo, ki, yi, x)
        s[CC].unroll(yo)
        s[CC].unroll(ki)
        s[CC].unroll(yi)
    else:
        s[CC].unroll(ki)
        s[CC].unroll(y)

    if C != O:
        y, x = s[O].op.axis
        yt, yo, yi = cfg["tile_y"].apply(s, O, y)
        xt, xo, xi = cfg["tile_x"].apply(s, O, x)
        s[O].reorder(xt, yt, yo, xo, yi, xi)
        xyt = s[O].fuse(xt, yt)
        s[C].compute_at(s[O], xyt)
        s[O].vectorize(xi)
        s[O].parallel(xyt)
    return s


def _schedule_dense_nopack_template(cfg, s, C):
    y, x = s[C].op.axis
    (kk,) = s[C].op.reduce_axis
    yo, yi = cfg["tile_y"].apply(s, C, y)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(yo, xo, yi, xi)
    xyo = s[C].fuse(yo, xo)
    s[C].parallel(xyo)
    s[C].unroll(kk)

    (CC,) = s[C].op.input_tensors
    s[CC].compute_at(s[C], xyo)
    z, y, x = s[CC].op.axis
    (k,) = s[CC].op.reduce_axis
    yz = s[CC].fuse(z, y)
    s[CC].reorder(k, yz, x)
    s[CC].unroll(yz)
    s[CC].vectorize(x)
    return s


def _default_dense_pack_config(cfg, M, N, K):
    # Generate default schedule for dynamic shape.
    if isinstance(M, (tvm.tir.Var, tvm.tir.Any)):
        M = 16
    if isinstance(N, (tvm.tir.Var, tvm.tir.Any)):
        N = 16
    if isinstance(K, (tvm.tir.Var, tvm.tir.Any)):
        K = 16

    vec_width = get_simd_32bit_lanes()
    tilex_ii = 1
    for bn in range(vec_width * 2, 0, -1):
        if N % bn == 0:
            tilex_ii = bn
            break
    NN = N // tilex_ii
    tilex_oi = 1
    while NN // tilex_oi > 4:
        if (NN // tilex_oi) % 2 == 1:
            break
        tilex_oi *= 2

    tiley_ii = 8
    while M % tiley_ii != 0:
        tiley_ii //= 2
    MM = M // tiley_ii
    tiley_oi = 1
    while MM // tiley_oi > 4:
        if (MM // tiley_oi) % 2 == 1:
            break
        tiley_oi *= 2

    cfg["tile_y"] = SplitEntity([MM // tiley_oi, tiley_oi, tiley_ii])
    cfg["tile_x"] = SplitEntity([NN // tilex_oi, tilex_oi, tilex_ii])
    cfg["tile_k"] = SplitEntity([K, 1])
    cfg["tile_inner"] = SplitEntity([M // tiley_ii, tiley_ii])


def _default_dense_nopack_config(cfg, M, N, K):
    # Generate default schedule for dynamic shape.
    if isinstance(M, (tvm.tir.Var, tvm.tir.Any)):
        M = 16
    if isinstance(N, (tvm.tir.Var, tvm.tir.Any)):
        N = 16
    if isinstance(K, (tvm.tir.Var, tvm.tir.Any)):
        K = 16

    vec_width = get_simd_32bit_lanes()
    tilek_bn = 1
    for bn in range(vec_width * 2, 0, -1):
        if K % bn == 0:
            tilek_bn = bn
            break
    cfg["tile_k"] = SplitEntity([K // tilek_bn, tilek_bn])
    cfg["tile_x"] = SplitEntity([N, 1])
    cfg["tile_y"] = SplitEntity([1, M])


@autotvm.register_topi_compute("dense_nopack.x86")
def dense_nopack(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense without packing"""
    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape)
    N, _ = get_const_tuple(weight.shape)
    # create tuning space
    cfg.define_split(
        "tile_y", 32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M, num_outputs=2
    )
    cfg.define_split(
        "tile_x", 32 if isinstance(N, (tvm.tir.Var, tvm.tir.Any)) else N, num_outputs=2
    )
    cfg.define_split(
        "tile_k", 32 if isinstance(K, (tvm.tir.Var, tvm.tir.Any)) else K, num_outputs=2
    )
    if cfg.is_fallback:
        _default_dense_nopack_config(cfg, M, N, K)

    vec = cfg["tile_k"].size[-1]
    k = te.reduce_axis((0, K // vec), "k")
    CC = te.compute(
        (M, N, vec),
        lambda z, y, x: te.sum(
            data[z, k * vec + x].astype(out_dtype) * weight[y, k * vec + x].astype(out_dtype),
            axis=k,
        ),
    )

    kk = te.reduce_axis((0, vec), "kk")
    C = te.compute((M, N), lambda y, x: te.sum(CC[y, x, kk], axis=kk), tag="dense_nopack")
    if bias is not None:
        C = te.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    return C


@autotvm.register_topi_schedule("dense_nopack.x86")
def schedule_dense_nopack(cfg, outs):
    """Create the schedule for dense_nopack"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense_nopack" in op.tag:
            _schedule_dense_nopack_template(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("dense_pack.x86")
def dense_pack(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense with transformed weight."""
    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape)  # batch, in_dim
    if len(weight.shape) == 3:
        N, _, packw_bn = get_const_tuple(weight.shape)  # out_dim
        N = N * packw_bn
    else:
        N, _ = get_const_tuple(weight.shape)  # out_dim
    # create tuning space
    cfg.define_split(
        "tile_y", 32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M, num_outputs=3
    )
    cfg.define_split(
        "tile_x", 32 if isinstance(N, (tvm.tir.Var, tvm.tir.Any)) else N, num_outputs=3
    )
    cfg.define_split(
        "tile_k", 32 if isinstance(K, (tvm.tir.Var, tvm.tir.Any)) else K, num_outputs=2
    )
    cfg.define_split(
        "tile_inner",
        32 if isinstance(M, (tvm.tir.Var, tvm.tir.Any)) else M,
        num_outputs=2,
        filter=lambda y: y.size[-1] <= 16,
    )
    if cfg.is_fallback:
        _default_dense_pack_config(cfg, M, N, K)

    if len(weight.shape) == 2:
        packw_bn = cfg["tile_x"].size[-1]
        packw_shape = (N // packw_bn, K, packw_bn)
        if autotvm.GLOBAL_SCOPE.in_tuning:
            # Directly use modified data layout placeholder.
            packw = tvm.te.placeholder(packw_shape, weight.dtype, name="packed_weight")
        else:
            packw = te.compute(
                packw_shape, lambda z, y, x: weight[z * packw_bn + x, y], name="packed_weight"
            )
    else:
        packw = weight

    idxdiv = tvm.tir.indexdiv
    idxmod = tvm.tir.indexmod
    k = te.reduce_axis((0, K), name="k")
    C = te.compute(
        (M, N),
        lambda y, x: te.sum(
            data[y, k].astype(out_dtype)
            * packw[idxdiv(x, packw_bn), k, idxmod(x, packw_bn)].astype(out_dtype),
            axis=k,
        ),
        tag="dense_pack",
    )
    if bias is not None:
        C = te.compute((M, N), lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    return C


@autotvm.register_topi_schedule("dense_pack.x86")
def schedule_dense_pack(cfg, outs):
    """Create the schedule for dense_pack"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense_pack" in op.tag:
            _schedule_dense_pack_template(cfg, s, op.output(0), outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("dense_int8.x86")
def dense_int8(cfg, data, weight, bias=None, out_dtype=None):
    """Compute for uint8 x int8 -> int32 dense"""
    if out_dtype is None:
        out_dtype = data.dtype
    assert len(weight.shape) == 4
    assert data.dtype == "uint8" and weight.dtype == "int8"
    _, _, n_inner, k_inner = get_const_tuple(weight.shape)  # out_dim
    assert n_inner == 16 and k_inner == 4
    return dense_int8_compute(cfg, data, weight, bias)


@autotvm.register_topi_schedule("dense_int8.x86")
def schedule_dense_int8(cfg, outs):
    """Create a schedule for dense__int8"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense_int8" in op.tag:
            if target_has_features("amx-int8"):
                dense_amx_int8_schedule(cfg, s, op.output(0), outs[0])
            elif target_has_features(["avx512bw", "avx512f"]):
                dense_int8_schedule(cfg, s, op.output(0), outs[0])

    traverse_inline(s, outs[0].op, _callback)
    return s


def dense_int8_compute(cfg, X, packed_w, bias=None):
    """Compute for uint8 x int8 -> int32 dense"""
    m, k = X.shape
    n_o, _, n_i, _ = packed_w.shape
    ak = te.reduce_axis((0, k), name="k")
    if target_has_features(["avx512bw", "avx512f"]):
        target_attr = {"schedule_rule": "meta_schedule.x86.dense_int8"}
    else:
        target_attr = None

    C = te.compute(
        (m, n_o * n_i),
        lambda i, j: te.sum(
            X[i, ak].astype("int32")
            * packed_w[tvm.tir.indexdiv(j, 16), tvm.tir.indexdiv(ak, 4), j % 16, ak % 4].astype(
                "int32"
            ),
            axis=ak,
        ),
        tag="dense_int8",
        attrs=target_attr,
    )

    if bias is not None:
        C = te.compute(C.shape, lambda i, j: C[i, j] + bias[j], tag=tag.BROADCAST)

    return C


def dense_int8_schedule(cfg, s, C, O, do_parallel=True):
    """Schedule dense compute using avx512 or lower instructions
    including VNNI vpdpbusd instruction if possible"""
    # C: The output of GEMM
    # O: The output of the fused op
    def split_y(out):
        default_y_split_factor = 32
        a_y = out.op.axis[-2]

        if cfg.is_fallback:
            return s[out].split(a_y, factor=default_y_split_factor)

        cfg.define_split("tile_y", a_y, num_outputs=2)
        return cfg["tile_y"].apply(s, out, a_y)

    (a_k,) = C.op.reduce_axis

    a_yo, a_yi = split_y(C)
    a_xo, a_xi = s[C].split(C.op.axis[-1], factor=16)
    a_ko, a_ki = s[C].split(a_k, factor=4)

    s[C].reorder(a_yo, a_xo, a_yi, a_ko, a_xi, a_ki)

    pc = dot_16x1x16_uint8_int8_int32()
    s[C].tensorize(a_xi, pc)

    if C == O:
        fused = s[O].fuse(a_yo, a_xo)
    else:
        a_yo, a_yi = split_y(O)
        a_xo, a_xi = s[O].split(O.op.axis[-1], factor=16)

        s[O].reorder(a_yo, a_xo, a_yi, a_xi)
        s[O].vectorize(a_xi)
        s[C].compute_at(s[O], a_yi)

        fused = s[O].fuse(a_yo, a_xo)

    if do_parallel:
        s[O].parallel(fused)

    return s, fused


def dense_amx_int8_schedule(cfg, s, C, O, do_parallel=True):
    """Schedule dense compute using AMX TMUL instruction"""
    # C: The output of GEMM
    # O: The output of the fused op
    def split_x(out):
        default_x_split_factor1 = 32
        default_x_split_factor2 = 2
        default_x_split_factor3 = 2
        default_x_split_factor4 = 2
        a_x = s[out].op.axis[-2]

        if cfg.is_fallback:
            a_xo, a_xi = s[out].split(a_x, factor=default_x_split_factor1)
            a_xo2, a_xo1 = s[out].split(a_xo, factor=default_x_split_factor2)
            a_xo3, a_xo2 = s[out].split(a_xo2, factor=default_x_split_factor3)
            a_xo4, a_xo3 = s[out].split(a_xo3, factor=default_x_split_factor4)
            return [a_xo4, a_xo3, a_xo2, a_xo1, a_xi]

        cfg.define_split("tile_x", a_x, num_outputs=5, filter=lambda x: x.size[-1] == 32)
        return cfg["tile_x"].apply(s, out, a_x)

    def split_y(out):
        default_y_split_factor1 = 32
        default_y_split_factor2 = 4
        default_y_split_factor3 = 4
        default_y_split_factor4 = 4
        a_y = s[out].op.axis[-1]

        if cfg.is_fallback:
            a_yo1, a_yo = s[out].split(a_y, factor=default_y_split_factor1)
            a_yo2, a_yo1 = s[out].split(a_yo1, factor=default_y_split_factor2)
            a_yo3, a_yo2 = s[out].split(a_yo2, factor=default_y_split_factor3)
            a_yo4, a_yo3 = s[out].split(a_yo3, factor=default_y_split_factor4)
            return [a_yo4, a_yo3, a_yo2, a_yo1, a_yo]

        cfg.define_split("tile_y", a_y, num_outputs=5, filter=lambda y: y.size[-1] == 32)
        return cfg["tile_y"].apply(s, out, a_y)

    def split_k(out, rd_axis):
        default_k_split_factor1 = 128
        default_k_split_factor2 = 2
        default_k_split_factor3 = 2
        default_k_split_factor4 = 2

        if cfg.is_fallback:
            a_ko, a_ki = s[out].split(rd_axis, factor=default_k_split_factor1)
            a_ko2, a_ko1 = s[out].split(a_ko, factor=default_k_split_factor2)
            a_ko3, a_ko2 = s[out].split(a_ko2, factor=default_k_split_factor3)
            a_ko4, a_ko3 = s[out].split(a_ko3, factor=default_k_split_factor4)
            return [a_ko4, a_ko3, a_ko2, a_ko1, a_ki]

        cfg.define_split("tile_k", rd_axis, num_outputs=5, filter=lambda y: y.size[-1] == 128)
        return cfg["tile_k"].apply(s, out, rd_axis)

    a_x, a_y = C.op.axis[-2:]
    (a_k,) = C.op.reduce_axis
    CF = s.cache_write(C, "amx.tmm")

    a_x3, a_x2, a_x1, a_xo, a_xi = split_x(C)
    a_y3, a_y2, a_y1, a_yo, a_yi = split_y(C)
    s[C].reorder(a_x3, a_y3, a_x2, a_y2, a_x1, a_y1, a_xo, a_yo, a_xi, a_yi)

    s[CF].compute_at(s[C], a_yo)

    (a_k_f,) = CF.op.reduce_axis
    a_x_f, a_y_f = CF.op.axis[-2:]

    a_xo_f, a_xi_f = s[CF].split(a_x_f, factor=32)

    a_yo_f, a_yi_f = s[CF].split(a_y_f, factor=32)
    a_k3_f, a_k2_f, a_k1_f, a_ko_f, a_ki_f = split_k(CF, a_k_f)
    s[CF].reorder(a_k3_f, a_k2_f, a_k1_f, a_ko_f, a_xo_f, a_yo_f, a_ki_f, a_xi_f, a_yi_f)

    (m, k) = CF.op.input_tensors[0].shape[-2:]
    (n, c, n_i, c_i) = CF.op.input_tensors[1].shape[-4:]
    n = n * n_i

    s[CF].tensorize(a_ki_f, dot_32x128x32_u8s8s32_sapphirerapids(LDA=int(k)))
    s[C].tensorize(a_xi, acc_32x32_int32_sapphirerapids(LDC=int(n)))

    if C == O:
        fused = s[O].fuse(a_x3, a_y3)
    else:
        a_y3, a_y2, a_y1, a_yr, a_yi = split_y(O)
        a_x3, a_x2, a_x1, a_xr, a_xi = split_x(O)

        s[O].reorder(a_y3, a_x3, a_y2, a_x2, a_y1, a_x1, a_yr, a_xr, a_yi, a_xi)
        s[O].vectorize(a_xi)

        fused = s[O].fuse(a_x3, a_y3)

    if do_parallel:
        s[O].parallel(fused)

    return s, fused


def matmul_blas_common(cfg, tensor_a, tensor_b, bias, out_dtype, transpose_a, transpose_b, lib):
    """Compute matmul/dense using a BLAS library"""
    M, K = get_const_tuple(tensor_a.shape)
    N, _ = get_const_tuple(tensor_b.shape)
    if isinstance(M, int) and isinstance(K, int) and isinstance(N, int):
        cfg.add_flop(M * K * N * 2)
    if tensor_a.dtype == "uint8" and tensor_b.dtype == "int8" and out_dtype == "int32":
        if not hasattr(lib, "matmul_u8s8s32"):
            raise NotImplementedError(
                f"Matmul/Dense with {lib.__name__} for {tensor_a.dtype} is not supported "
                "(matmulu8s8s32 not imlemented)"
            )
        C = lib.matmul_u8s8s32(tensor_a, tensor_b, transpose_a, transpose_b, dtype=out_dtype)
    elif tensor_a.dtype == "float32" or tensor_a.dtype == "float64":
        C = lib.matmul(tensor_a, tensor_b, transpose_a, transpose_b)
    else:
        raise NotImplementedError(
            f"Matmul/Dense with {lib.__name__} for {tensor_a.dtype} is not supported"
        )

    if bias is not None:
        C = te.compute(C.shape, lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    return C


@autotvm.register_topi_compute("dense_cblas.x86")
def dense_cblas(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense using cblas. This is an alias of matmul_nt operator."""
    return matmul_blas_common(cfg, data, weight, bias, out_dtype, False, True, cblas)


@autotvm.register_topi_schedule("dense_cblas.x86")
def schedule_dense_cblas(_, outs):
    """Create schedule for dense_cblas. This is an alias of matmul_nt operator."""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("dense_mkl.x86")
def dense_mkl(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense using mkl. This is an alias of matmul_nt operator."""
    return matmul_blas_common(cfg, data, weight, bias, out_dtype, False, True, mkl)


@autotvm.register_topi_schedule("dense_mkl.x86")
def schedule_dense_mkl(_, outs):
    """Create schedule for dense_mkl. This is an alias of matmul_nt operator."""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("dense_dnnl.x86")
def dense_dnnl(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense using dnnl. This is an alias of matmul_nt operator."""
    return matmul_blas_common(cfg, data, weight, bias, out_dtype, False, True, dnnl)


@autotvm.register_topi_schedule("dense_dnnl.x86")
def schedule_dense_dnnl(_, outs):
    """Create schedule for dense_dnnl. This is an alias of matmul_nt operator."""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("matmul_cblas.x86")
def matmul_cblas(
    cfg, tensor_a, tensor_b, bias=None, out_dtype=None, transpose_a=False, transpose_b=False
):
    """Compute matmul using cblas."""
    return matmul_blas_common(
        cfg, tensor_a, tensor_b, bias, out_dtype, transpose_a, transpose_b, cblas
    )


@autotvm.register_topi_schedule("matmul_cblas.x86")
def schedule_matmul_cblas(_, outs):
    """Create schedule for matmul_cblas."""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("matmul_mkl.x86")
def matmul_mkl(
    cfg, tensor_a, tensor_b, bias=None, out_dtype=None, transpose_a=False, transpose_b=False
):
    """Compute matmul using mkl."""
    return matmul_blas_common(
        cfg, tensor_a, tensor_b, bias, out_dtype, transpose_a, transpose_b, mkl
    )


@autotvm.register_topi_schedule("matmul_mkl.x86")
def schedule_matmul_mkl(_, outs):
    """Create schedule for matmul_mkl."""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("matmul_dnnl.x86")
def matmul_dnnl(
    cfg, tensor_a, tensor_b, bias=None, out_dtype=None, transpose_a=False, transpose_b=False
):
    """Compute matmul using dnnl."""
    return matmul_blas_common(
        cfg, tensor_a, tensor_b, bias, out_dtype, transpose_a, transpose_b, dnnl
    )


@autotvm.register_topi_schedule("matmul_dnnl.x86")
def schedule_matmul_dnnl(_, outs):
    """Create schedule for matmul_dnnl."""
    return generic.schedule_extern(outs)


def dense_dynamic(A, B, bias, dtype):
    """Compute for dense with dynamic shape"""

    assert A.shape[0] == 1, "Only dynamic matrix vector multiplication with vector LHS is supported"

    # Right now we only support matrix-vector multiplication with lhs as the
    # vector. We don't need to do much optimization here because the access
    # pattern and parallelization are straight forward.
    def gen_ir(a, b, c):
        ib = tvm.tir.ir_builder.create()
        A = ib.buffer_ptr(a)
        B = ib.buffer_ptr(b)
        C = ib.buffer_ptr(c)
        with ib.for_range(0, b.shape[0], name="j", kind="parallel") as j:
            C[0, j] = 0.0
            with ib.for_range(0, b.shape[1], name="k") as k:
                C[0, j] += A[0, k] * B[j, k]
        return ib.get()

    def gen_ir_bias(a, b, bias, c):
        ib = tvm.tir.ir_builder.create()
        A = ib.buffer_ptr(a)
        B = ib.buffer_ptr(b)
        C = ib.buffer_ptr(c)
        with ib.for_range(0, b.shape[0], name="j", kind="parallel") as j:
            C[0, j] = bias[j]
            with ib.for_range(0, b.shape[1], name="k") as k:
                C[0, j] += A[0, k] * B[j, k]
        return ib.get()

    out_shape = (A.shape[0], B.shape[0])
    out_buf = tvm.tir.decl_buffer(out_shape, dtype, "out_buf")
    if bias is None:
        out = te.extern(
            [out_shape],
            [A, B],
            lambda ins, outs: gen_ir(*ins, *outs),
            dtype=dtype,
            out_buffers=[out_buf],
            name="dense_dynamic_cpu",
            tag="dense_dynamic_cpu",
        )
    else:
        out = te.extern(
            [out_shape],
            [A, B, bias],
            lambda ins, outs: gen_ir_bias(*ins, *outs),
            dtype=dtype,
            out_buffers=[out_buf],
            name="dense_dynamic_cpu",
            tag="dense_dynamic_cpu",
        )
    return out


def schedule_dense_dynamic(outs):
    """Create schedule for dense_dynamic."""
    return generic.schedule_extern(outs)
