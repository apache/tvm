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
"""x86 batch_matmul operators"""
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity
from tvm.contrib import cblas, mkl
from .. import generic
from ..utils import traverse_inline, get_const_tuple, get_max_power2_factor


@autotvm.register_topi_compute("batch_matmul.x86")
def batch_matmul(cfg, x, y, out_shape=None, out_dtype=None):
    """Computes batch matrix multiplication of `x` and `y` when `x` and `y` are
    data in batch. Supports broadcasting in batch dimension.

    Parameters
    ----------
    cfg : ConfigSpace
        Autotvm tuning space config file
    x : tvm.te.Tensor
        3-D with shape [batch, M, K]
    y : tvm.te.Tensor
        3-D with shape [batch, N, K]
    out_shape : tuple or None
        Shape of the outputs

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    assert len(x.shape) == 3 and len(y.shape) == 3, "only support 3-dim batch_matmul"
    XB, M, XK = get_const_tuple(x.shape)
    YB, N, YK = get_const_tuple(y.shape)
    assert (XB == YB) or (YB == 1) or (XB == 1), "batch dimension doesn't match"
    assert XK == YK, "shapes of x and y is inconsistent"
    B = te.max(XB, YB)
    K = XK
    if out_shape is not None:
        assert out_shape[0] == B, "got invalid output shape"
        assert out_shape[1] == M, "got invalid output shape"
        assert out_shape[2] == N, "got invalid output shape"
    if cfg.is_fallback:
        _default_batch_matmul_config(cfg, M, N, K)

    k = te.reduce_axis((0, K), name="k")
    if out_dtype is None or out_dtype == x.dtype:
        C = te.compute(
            (B, M, N),
            lambda b, i, j: te.sum(
                x[b if XB != 1 else 0, i, k] * y[b if YB != 1 else 0, j, k], axis=k
            ),
            tag="batch_matmul",
        )
    else:
        C = te.compute(
            (B, M, N),
            lambda b, i, j: te.sum(
                x[b if XB != 1 else 0, i, k].astype(out_dtype)
                * y[b if YB != 1 else 0, j, k].astype(out_dtype),
                axis=k,
            ),
            tag="batch_matmul",
        )
    return C


@autotvm.register_topi_schedule("batch_matmul.x86")
def schedule_batch_matmul(cfg, outs):
    """Schedule for batch_matmul

    Parameters
    ----------
    cfg : ConfigSpace
        AutoTVM tuning space config file.
    outs : Array of Tensor
        The computation graph description of batch_matmul
        in the format of an array of tensors.

    Returns
    -------
    sch: Schedule
        The computation schedule for the op.
    """
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "batch_matmul" in op.tag:
            C = op.output(0)
            A, B = op.input_tensors
            _, M, K = get_const_tuple(A.shape)
            _, _, N = get_const_tuple(C.shape)

            if op not in s.outputs:
                s[C].compute_inline()
                O = outs[0]
            else:
                O = C

            CC = s.cache_write(C, "global")

            # create tuning space
            cfg.define_split("tile_y", M, num_outputs=2)
            cfg.define_split("tile_x", N, num_outputs=2)
            cfg.define_split("tile_k", K, num_outputs=2)

            b, y, x = s[O].op.axis
            yo, yi = cfg["tile_y"].apply(s, O, y)
            xo, xi = cfg["tile_x"].apply(s, O, x)
            s[O].reorder(b, yo, xo, yi, xi)
            bxyo = s[O].fuse(b, yo, xo)
            s[O].parallel(bxyo)

            s[CC].compute_at(s[O], bxyo)
            (k,) = s[CC].op.reduce_axis
            ko, ki = cfg["tile_k"].apply(s, CC, k)

            Crf = s.rfactor(CC, ki)
            s[Crf].compute_at(s[CC], s[CC].op.axis[0])
            _, _, y, x = s[Crf].op.axis
            s[Crf].fuse(y, x)
            s[Crf].vectorize(s[Crf].op.axis[0])
            s[O].pragma(bxyo, "auto_unroll_max_step", 16)

    traverse_inline(s, outs[0].op, _callback)
    return s


def _default_batch_matmul_config(cfg, M, N, K):
    cfg["tile_k"] = SplitEntity([K // 16, 16])
    x_bn = get_max_power2_factor(N, 8)
    cfg["tile_x"] = SplitEntity([N // x_bn, x_bn])
    y_bn = get_max_power2_factor(M, 8)
    cfg["tile_y"] = SplitEntity([M // y_bn, y_bn])


def batch_matmul_blas_common(cfg, x, y, out_shape, lib):
    """Computes batch matrix multiplication of `x` and `y` when `x` and `y` are
    data in batch, using one of BLAS libraries. Supports broadcasting in batch dimension.

    Parameters
    ----------
    cfg : ConfigSpace
        Autotvm tuning space config file
    x : tvm.te.Tensor
        3-D with shape [batch, M, K]
    y : tvm.te.Tensor
        3-D with shape [batch, N, K]
    out_shape : tuple or None
        Shape of the output
    lib : A contrib module which implements batch_matmul function
        cblas and mkl are supported

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    assert len(x.shape) == 3 and len(y.shape) == 3, "only support 3-dim batch_matmul"
    XB, M, XK = get_const_tuple(x.shape)
    YB, N, YK = get_const_tuple(y.shape)
    assert (XB == YB) or (YB == 1) or (XB == 1), "batch dimension doesn't match"
    assert XK == YK, "shapes of x and y is inconsistent"
    if out_shape is not None:
        assert out_shape[0] in (XB, YB), "got invalid output shape"
        assert out_shape[1] == M, "got invalid output shape"
        assert out_shape[2] == N, "got invalid output shape"
    cfg.add_flop(XB * M * N * XK * 2)
    return lib.batch_matmul(x, y, False, True)


@autotvm.register_topi_compute("batch_matmul_cblas.x86")
def batch_matmul_cblas(cfg, x, y, out_shape=None):
    """Compute batch_matmul using cblas"""
    return batch_matmul_blas_common(cfg, x, y, out_shape, cblas)


@autotvm.register_topi_schedule("batch_matmul_cblas.x86")
def schedule_batch_matmul_cblas(_, outs):
    """Create schedule for batch_matmul_cblas"""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("batch_matmul_mkl.x86")
def batch_matmul_mkl(cfg, x, y, out_shape=None):
    """Compute batch_matmul using mkl"""
    return batch_matmul_blas_common(cfg, x, y, out_shape, mkl)


@autotvm.register_topi_schedule("batch_matmul_mkl.x86")
def schedule_batch_matmul_mkl(_, outs):
    """Create schedule for batch_matmul_mul"""
    return generic.schedule_extern(outs)
