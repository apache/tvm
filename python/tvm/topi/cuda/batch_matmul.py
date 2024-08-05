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
from .. import nn, generic
from ..utils import traverse_inline, get_const_tuple, get_max_power2_factor
from .tensor_intrin import dp4a


@autotvm.register_topi_compute("batch_matmul.cuda")
def batch_matmul(cfg, x, y, out_shape=None, out_dtype=None, transpose_a=False, transpose_b=True):
    """Compute batch matrix multiplication of `tensor_a` and `tensor_b`.

    Both `tensor_a` and `tensor_b` can be transposed. For legacy reason, we use NT format
    (transpose_a=False, transpose_b=True) by default.

    Parameters
    ----------
    cfg : ConfigSpace
        Autotvm tuning space config file.

    tensor_a : tvm.te.Tensor
        3-D with shape [batch, M, K] or [batch, K, M].

    tensor_b : tvm.te.Tensor
        3-D with shape [batch, K, N] or [batch, N, K].

    out_shape : List[Optional]
        Explicit intended output shape of the computation. Can be useful in cases
        with dynamic input shapes.

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision batch matmul.

    transpose_a : Optional[bool] = False
        Whether the first tensor is in transposed format.

    transpose_b : Optional[bool] = True
        Whether the second tensor is in transposed format.

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    return nn.batch_matmul(
        x,
        y,
        oshape=out_shape,
        out_dtype=out_dtype,
        transpose_a=transpose_a,
        transpose_b=transpose_b,
    )


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
        if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
            s[B].compute_inline()
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
        (k,) = s[CC].op.reduce_axis

        cfg.define_split("tile_y", y, num_outputs=3)
        cfg.define_split("tile_x", x, num_outputs=3)
        cfg.define_split("tile_k", k, num_outputs=2)
        cfg.define_knob("auto_unroll_max_step", [8, 16, 32, 64])
        target = tvm.target.Target.current()
        if target.kind.name in ["nvptx", "rocm"]:
            # llvm-based backends cannot do non-explicit unrolling
            cfg.define_knob("unroll_explicit", [1])
        else:
            cfg.define_knob("unroll_explicit", [0, 1])

        if cfg.is_fallback:
            y_bn = get_max_power2_factor(M, 64)
            x_bn = get_max_power2_factor(N, 64)
            y_nthreads = min(y_bn, 8)
            x_nthreads = min(x_bn, 8)
            cfg["tile_x"] = SplitEntity([-1, x_nthreads, x_bn // x_nthreads])
            cfg["tile_y"] = SplitEntity([-1, y_nthreads, y_bn // y_nthreads])
            cfg["tile_k"] = SplitEntity([-1, 8])
            cfg["auto_unroll_max_step"] = OtherOptionEntity(16)

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
        s[C].pragma(yi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[C].pragma(yi, "unroll_explicit", cfg["unroll_explicit"].val)

        s[CC].compute_at(s[C], tx)
        _, yi, xi = s[CC].op.axis
        ko, ki = cfg["tile_k"].apply(s, CC, k)
        s[CC].reorder(ko, ki, yi, xi)
        s[CC].pragma(ki, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[CC].pragma(ki, "unroll_explicit", cfg["unroll_explicit"].val)

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
        s[AA].pragma(yi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[AA].pragma(yi, "unroll_explicit", cfg["unroll_explicit"].val)

        _, x, k = s[BB].op.axis
        ty, xi = s[BB].split(x, nparts=cfg["tile_y"].size[1])
        tx, ki = s[BB].split(k, nparts=cfg["tile_x"].size[1])
        s[BB].bind(ty, thread_y)
        s[BB].bind(tx, thread_x)
        s[BB].reorder(ty, tx, xi, ki)
        s[BB].pragma(xi, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
        s[BB].pragma(xi, "unroll_explicit", cfg["unroll_explicit"].val)

    def _callback(op):
        if "batch_matmul" in op.tag:
            _schedule(cfg, op)

    traverse_inline(s, outs[0].op, _callback)
    return s


@autotvm.register_topi_compute("batch_matmul_cublas.cuda")
def batch_matmul_cublas(
    cfg, x, y, out_shape=None, out_dtype=None, transpose_a=False, transpose_b=True
):
    """Compute batch matrix multiplication of `x` and `y`.

    Both `x` and `y` can be transposed. For legacy reason, we use NT format
    (transpose_a=False, transpose_b=True) by default.

    Parameters
    ----------
    cfg : ConfigSpace
        Autotvm tuning space config file.

    x : tvm.te.Tensor
        3-D with shape [batch, M, K] or [batch, K, M].

    y : tvm.te.Tensor
        3-D with shape [batch, K, N] or [batch, N, K].

    out_shape : List[Optional]
        Explicit intended output shape of the computation. Can be useful in cases
        with dynamic input shapes.

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision batch matmul.

    transpose_a : Optional[bool] = False
        Whether the first tensor is in transposed format.

    transpose_b : Optional[bool] = True
        Whether the second tensor is in transposed format.

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    if transpose_a:
        b, k, m = get_const_tuple(x.shape)
    else:
        b, m, k = get_const_tuple(x.shape)
    if transpose_b:
        b, n, k = get_const_tuple(y.shape)
    else:
        b, k, n = get_const_tuple(y.shape)
    if all([isinstance(s, int) for s in [b, m, n, k]]):
        cfg.add_flop(b * m * k * n * 2)
    return cublas.batch_matmul(x, y, transa=transpose_a, transb=transpose_b, dtype=out_dtype)


@autotvm.register_topi_schedule("batch_matmul_cublas.cuda")
def schedule_batch_matmul_cublas(_, outs):
    """Schedule batch_matmul operator using CUBLAS"""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("batch_matmul_int8.cuda")
def batch_matmul_int8(
    cfg, x, y, out_shape=None, out_dtype=None, transpose_a=False, transpose_b=True
):
    """Batch Matmul operator for int8 on CUDA.

    Parameters
    ----------
    cfg : ConfigSpace
        Autotvm tuning space config file.

    x : tvm.te.Tensor
        3-D with shape [batch, M, K] or [batch, K, M].

    y : tvm.te.Tensor
        3-D with shape [batch, K, N] or [batch, N, K].

    out_shape : List[Optional]
        Explicit intended output shape of the computation. Can be useful in cases
        with dynamic input shapes.

    out_dtype : Optional[str]
        Specifies the output data type for mixed precision batch matmul.

    transpose_a : Optional[bool] = False
        Whether the first tensor is in transposed format.

    transpose_b : Optional[bool] = True
        Whether the second tensor is in transposed format.

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    del out_shape
    # TODO(jcf94): Deal with different transpose combinations
    assert not transpose_a and transpose_b
    if out_dtype is None:
        out_dtype = x.dtype

    x_shape = get_const_tuple(x.shape)
    y_shape = get_const_tuple(y.shape)
    assert len(x_shape) == 3 and len(y_shape) == 3, "only support 3-dim batch_matmul"

    XB, M, XK = x.shape
    YB, N, YK = y.shape
    assert XB == YB or XB == 1 or YB == 1, "batch dimension doesn't match"
    assert XK == YK, "shapes of x and y is inconsistent"

    nB = tvm.te.max(XB, YB)
    nK = ((XK + 3) // 4) * 4
    reduce_k = te.reduce_axis((0, nK), name="k")

    # pad for _dp4a vectorize
    pad_x = te.compute(
        (XB, M, nK),
        lambda b, i, j: tvm.te.if_then_else(j >= XK, tvm.tir.const(0, x.dtype), x[b, i, j]),
    )
    pad_y = te.compute(
        (YB, N, nK),
        lambda b, i, j: tvm.te.if_then_else(j >= YK, tvm.tir.const(0, y.dtype), y[b, i, j]),
    )

    out = te.compute(
        (nB, M, N),
        lambda b, i, j: te.sum(
            pad_x[b if XB != 1 else 0, i, reduce_k].astype(out_dtype)
            * pad_y[b if YB != 1 else 0, j, reduce_k].astype(out_dtype),
            axis=[reduce_k],
        ),
        tag="batch_matmul_int8",
    )
    cfg.add_flop(XB * M * N * nK * 2)
    return out


@autotvm.register_topi_schedule("batch_matmul_int8.cuda")
def schedule_batch_matmul_int8(cfg, outs):
    """Batch Matmul schedule for int8 on CUDA"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "batch_matmul_int8" in op.tag:
            _schedule_batch_matmul_int8(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_batch_matmul_int8(cfg, s, output):
    input_x, input_y = s[output].op.input_tensors
    if len(input_y.op.input_tensors) == 1 and input_y.op.input_tensors[0] == input_x:
        s[input_y].compute_inline()

    B, M, K = get_const_tuple(input_x.shape)
    _, N, _ = get_const_tuple(input_y.shape)

    k_factor = 4
    assert K % k_factor == 0, f"Input dimension must divide {k_factor}"
    if K % 16 == 0:
        k_factor = 16

    cfg.define_split("tile_f", B, num_outputs=4)
    cfg.define_split("tile_m", M, num_outputs=4)
    cfg.define_split("tile_n", N, num_outputs=4)
    cfg.define_split("tile_k", K // k_factor, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 256, 512, 1024])

    batch_matmul_op = s[output].op
    s[input_x].compute_inline()
    s[input_y].compute_inline()

    x_cache = s.cache_read(input_x, "shared", [batch_matmul_op])
    y_cache = s.cache_read(input_y, "shared", [batch_matmul_op])
    batch_matmul_cache = s.cache_write(batch_matmul_op.output(0), "local")

    # tile reduce axis
    ko = batch_matmul_cache.op.reduce_axis[0]
    ko, ki = s[batch_matmul_cache].split(ko, factor=4)
    ko, kt = cfg["tile_k"].apply(s, batch_matmul_cache, ko)
    # dp4a tensorize

    target = tvm.target.Target.current(allow_none=False)
    do_tensorize = "+dotprod" in target.mattr or target.supports_integer_dot_product

    if do_tensorize:
        dtypes = (input_x.dtype, input_y.dtype)
        s[batch_matmul_cache].tensorize(ki, dp4a("shared", "shared", "local", dtypes))

    if batch_matmul_op not in s.outputs:
        s[output].compute_inline()
        batch_matmul_op = s.outputs[0]

    # tile axis
    f, m, n = batch_matmul_op.axis
    kernel_scope, f = s[batch_matmul_op].split(f, nparts=1)

    bf, vf, tf, fi = cfg["tile_f"].apply(s, batch_matmul_op, f)
    bm, vm, tm, mi = cfg["tile_m"].apply(s, batch_matmul_op, m)
    bn, vn, tn, ni = cfg["tile_n"].apply(s, batch_matmul_op, n)
    s[batch_matmul_op].reorder(bf, bm, bn, vf, vm, vn, tf, tm, tn, fi, mi, ni)

    # bind axis
    s[batch_matmul_op].bind(bf, tvm.te.thread_axis("blockIdx.z"))
    s[batch_matmul_op].bind(bm, tvm.te.thread_axis("blockIdx.y"))
    s[batch_matmul_op].bind(bn, tvm.te.thread_axis("blockIdx.x"))

    s[batch_matmul_op].bind(vf, tvm.te.thread_axis("vthread"))
    s[batch_matmul_op].bind(vm, tvm.te.thread_axis("vthread"))
    s[batch_matmul_op].bind(vn, tvm.te.thread_axis("vthread"))

    s[batch_matmul_op].bind(tf, tvm.te.thread_axis("threadIdx.z"))
    s[batch_matmul_op].bind(tm, tvm.te.thread_axis("threadIdx.y"))
    s[batch_matmul_op].bind(tn, tvm.te.thread_axis("threadIdx.x"))

    # cache compute at
    s[batch_matmul_cache].compute_at(s[batch_matmul_op], tn)
    fo, mo, no = batch_matmul_cache.op.axis[:3]
    s[batch_matmul_cache].reorder(ko, kt, fo, mo, no, ki)

    # for load in [splited_x_op, splited_y_op]
    for load in [x_cache, y_cache]:
        s[load].compute_at(s[batch_matmul_cache], ko)
        outer, inner = s[load].split(s[load].op.axis[-1], factor=k_factor)
        s[load].vectorize(inner)

        fused = s[load].op.axis[:-1] + [outer]
        fused = s[load].fuse(*fused)

        fused, tx = s[load].split(fused, factor=cfg["tile_n"].size[2])
        fused, ty = s[load].split(fused, factor=cfg["tile_m"].size[2])
        fused, tz = s[load].split(fused, factor=cfg["tile_f"].size[2])

        s[load].bind(tz, tvm.te.thread_axis("threadIdx.z"))
        s[load].bind(ty, tvm.te.thread_axis("threadIdx.y"))
        s[load].bind(tx, tvm.te.thread_axis("threadIdx.x"))

    # max unroll
    s[batch_matmul_op].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[batch_matmul_op].pragma(kernel_scope, "unroll_explicit", False)

    return s
