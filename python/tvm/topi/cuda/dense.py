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
# pylint: disable=invalid-name, unused-argument
"""Schedule for dense operator"""
import logging
import tvm
from tvm import te
import tvm.autotvm as autotvm
from tvm.contrib import cublas
from .tensor_intrin import dp4a
from .. import tag
from .. import generic
from ..utils import traverse_inline, get_const_tuple

logger = logging.getLogger("topi")


def _matmul_cublas_common(
    cfg,
    tensor_a,
    tensor_b,
    bias=None,
    out_dtype=None,
    transpose_a=False,
    transpose_b=False,
):
    assert len(tensor_a.shape) == 2 and len(tensor_b.shape) == 2, "only support 2-dim matmul"
    if bias is not None:
        assert len(bias.shape) == 1
    if out_dtype is None:
        out_dtype = tensor_a.dtype
    if out_dtype not in [tensor_a.dtype, "int32"]:
        assert out_dtype == tensor_a.dtype, "Mixed precision other than int8 + int32 not supported."
    batch, in_dim = get_const_tuple(tensor_a.shape)
    out_dim, _ = get_const_tuple(tensor_b.shape)
    matmul = cublas.matmul(tensor_a, tensor_b, transpose_a, transpose_b, dtype=out_dtype)
    if all(isinstance(d, int) for d in [batch, in_dim, out_dim]):
        cfg.add_flop(batch * in_dim * out_dim * 2)
    if bias is not None:
        matmul = te.compute(
            (batch, out_dim), lambda i, j: matmul[i, j] + bias[j], tag=tag.BROADCAST
        )
    return matmul


@autotvm.register_topi_compute("matmul_cublas.cuda")
def matmul_cublas(
    cfg,
    tensor_a,
    tensor_b,
    bias=None,
    out_dtype=None,
    transpose_a=False,
    transpose_b=False,
):
    """Matmul operator on CUDA with CUBLAS"""
    return _matmul_cublas_common(cfg, tensor_a, tensor_b, bias, out_dtype, transpose_a, transpose_b)


@autotvm.register_topi_schedule("matmul_cublas.cuda")
def schedule_matmul_cublas(_, outs):
    """Schedule matmul operator using CUBLAS"""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("dense_cublas.cuda")
def dense_cublas(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator on CUDA with CUBLAS. This is an alias of matmul_nt operator."""
    return _matmul_cublas_common(cfg, data, weight, bias, out_dtype, False, True)


@autotvm.register_topi_schedule("dense_cublas.cuda")
def schedule_dense_cublas(_, outs):
    """Schedule dense operator using CUBLAS"""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("dense_int8.cuda")
def dense_int8(cfg, data, weight, bias=None, out_dtype=None):
    """Dense operator for int8 on CUDA"""
    if out_dtype is None:
        out_dtype = data.dtype

    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)
    k = te.reduce_axis((0, in_dim), name="k")

    matmul = te.compute(
        (batch, out_dim),
        lambda i, j: te.sum(
            data[i, k].astype(out_dtype) * weight[j, k].astype(out_dtype), axis=[k]
        ),
        tag="dense_int8",
    )

    cfg.add_flop(batch * in_dim * out_dim * 2)

    if bias is not None:
        matmul = te.compute(
            (batch, out_dim),
            lambda i, j: matmul[i, j] + bias[j].astype(out_dtype),
            tag=tag.BROADCAST,
        )
        cfg.add_flop(batch * out_dim)

    return matmul


@autotvm.register_topi_schedule("dense_int8.cuda")
def schedule_dense_int8(cfg, outs):
    """Dense schedule for int8 on CUDA"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "dense_int8" in op.tag:
            _schedule_dense_int8(cfg, s, op.output(0))

    traverse_inline(s, outs[0].op, _callback)
    return s


def _schedule_dense_int8(cfg, s, output):
    data, weight = s[output].op.input_tensors
    if len(weight.op.input_tensors) == 1 and weight.op.input_tensors[0] == data:
        s[weight].compute_inline()

    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)

    in_dim_factor = 4
    assert in_dim % in_dim_factor == 0, "Input dimension must divide {}".format(in_dim_factor)
    if in_dim % 16 == 0:
        in_dim_factor = 16

    # create tuning space
    cfg.define_split("tile_y", batch, num_outputs=4)
    cfg.define_split("tile_x", out_dim, num_outputs=4)
    cfg.define_split("tile_k", in_dim // in_dim_factor, num_outputs=2)
    cfg.define_knob("auto_unroll_max_step", [0, 512, 1500])

    # create cache stage
    AA = s.cache_read(data, "shared", [output])
    WW = s.cache_read(weight, "shared", [output])
    CC = s.cache_write(output, "local")

    # handle bias
    if output.op not in s.outputs:
        s[output].compute_inline()
        output = s.outputs[0].output(0)

    n, x = s[output].op.axis

    # this is the scope to attach global config inside this kernel
    kernel_scope, n = s[output].split(n, nparts=1)

    ko = CC.op.reduce_axis[0]
    ko, ki = s[CC].split(ko, factor=4)
    ko, kt = cfg["tile_k"].apply(s, CC, ko)
    target = tvm.target.Target.current(allow_none=False)
    do_tensorize = "+dotprod" in target.mattr or target.supports_integer_dot_product

    if do_tensorize:
        dtypes = (data.dtype, weight.dtype)
        s[CC].tensorize(ki, dp4a("shared", "shared", "local", dtypes))
    by, vy, ty, yi = cfg["tile_y"].apply(s, output, n)
    bx, vx, tx, xi = cfg["tile_x"].apply(s, output, x)

    s[output].reorder(by, bx, vy, vx, ty, tx, yi, xi)
    s[output].bind(by, te.thread_axis("blockIdx.y"))
    s[output].bind(bx, te.thread_axis("blockIdx.x"))
    s[output].bind(vy, te.thread_axis("vthread"))
    s[output].bind(vx, te.thread_axis("vthread"))
    s[output].bind(ty, te.thread_axis("threadIdx.y"))
    s[output].bind(tx, te.thread_axis("threadIdx.x"))
    n_ty = cfg["tile_y"].size[2]
    n_tx = cfg["tile_x"].size[2]

    s[CC].compute_at(s[output], tx)
    yo, xo = CC.op.axis[:2]
    s[CC].reorder(ko, kt, yo, xo, ki)

    for load in [AA, WW]:
        s[load].compute_at(s[CC], ko)

        outer, inner = s[load].split(s[load].op.axis[-1], factor=in_dim_factor)
        s[load].vectorize(inner)
        fused = s[load].op.axis[:-1] + [outer]
        fused = s[load].fuse(*fused)

        fused, tx = s[load].split(fused, factor=n_tx)
        fused, ty = s[load].split(fused, factor=n_ty)
        s[load].bind(tx, te.thread_axis("threadIdx.x"))
        s[load].bind(ty, te.thread_axis("threadIdx.y"))

    s[output].pragma(kernel_scope, "auto_unroll_max_step", cfg["auto_unroll_max_step"].val)
    s[output].pragma(kernel_scope, "unroll_explicit", False)
    return s
