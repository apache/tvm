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
# pylint: disable=invalid-name,too-many-locals,unused-variable
# pylint: disable=no-value-for-parameter
"""x86 dense operators"""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity
from tvm.contrib import cblas
from tvm.contrib import mkl
from tvm.contrib import mkldnn

from .utils import get_fp32_len
from .injective import schedule_injective_from_existing
from .. import generic, tag
from ..utils import traverse_inline, get_const_tuple


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

    vec_width = get_fp32_len()
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

    vec_width = get_fp32_len()
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


def dense_blas_common(cfg, data, weight, bias, out_dtype, lib):
    """Compute dense using a BLAS library"""
    M, K = get_const_tuple(data.shape)
    N, _ = get_const_tuple(weight.shape)
    if isinstance(M, int) and isinstance(K, int) and isinstance(N, int):
        cfg.add_flop(M * K * N * 2)
    if data.dtype == "uint8" and weight.dtype == "int8" and out_dtype == "int32":
        if not hasattr(lib, "matmul_u8s8s32"):
            raise NotImplementedError(
                f"Dense with {lib.__name__} for {data.dtype} is not supported "
                "(matmulu8s8s32 not imlemented)"
            )
        C = lib.matmul_u8s8s32(data, weight, False, True, dtype=out_dtype)
    elif data.dtype == "float32" or data.dtype == "float64":
        C = lib.matmul(data, weight, False, True)
    else:
        raise NotImplementedError(f"Dense with {lib.__name__} for {data.dtype} is not supported")

    if bias is not None:
        C = te.compute(C.shape, lambda i, j: C[i, j] + bias[j].astype(out_dtype), tag=tag.BROADCAST)
    return C


@autotvm.register_topi_compute("dense_cblas.x86")
def dense_cblas(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense using a cblas"""
    return dense_blas_common(cfg, data, weight, bias, out_dtype, cblas)


@autotvm.register_topi_schedule("dense_cblas.x86")
def schedule_dense_cblas(_, outs):
    """Create schedule for dense_cblas"""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("dense_mkl.x86")
def dense_mkl(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense using mkl"""
    return dense_blas_common(cfg, data, weight, bias, out_dtype, mkl)


@autotvm.register_topi_schedule("dense_mkl.x86")
def schedule_dense_mkl(_, outs):
    """Create schedule for dense_mkl"""
    # return generic.schedule_extern(outs)
    s = te.create_schedule([x.op for x in outs])
    te.schedule.AutoInlineInjective(s)

    def _callback(op):
        if "broadcast" in op.tag or "injective" in op.tag or "elemwise" in op.tag:
            schedule_injective_from_existing(s, op.output(0))

    # traverse_inline(s, outs[0].op, _callback)
    for out in outs:
        if "dense" not in out.op.name:
            schedule_injective_from_existing(s, out)
    return s


@autotvm.register_topi_compute("dense_mkldnn.x86")
def dense_mkldnn(cfg, data, weight, bias=None, out_dtype=None):
    """Compute dense using mkldnn"""
    return dense_blas_common(cfg, data, weight, bias, out_dtype, mkldnn)


@autotvm.register_topi_schedule("dense_mkldnn.x86")
def schedule_dense_mkldnn(_, outs):
    """Create schedule for dense_mkldnn"""
    return generic.schedule_extern(outs)
