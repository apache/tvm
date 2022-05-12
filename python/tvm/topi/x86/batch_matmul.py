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
import tvm
from tvm import te
from tvm import autotvm
from tvm.autotvm.task.space import SplitEntity
from tvm.contrib import cblas, mkl
from .. import generic, nn
from ..transform import layout_transform
from ..utils import traverse_inline, get_const_tuple, get_max_power2_factor
from .dense import dense_vnni_schedule
from .injective import schedule_injective_from_existing


@autotvm.register_topi_compute("batch_matmul_vnni.x86")
def batch_matmul_vnni_compute(cfg, x, y, *_):
    """Compute for uint8 x int8 -> int32 batch_matmul"""
    batch, m, k = x.shape
    packed_y_layout = "BNK16n4k"
    packed_y = layout_transform(y, "BNK", packed_y_layout)
    _, n_o, _, n_i, _ = packed_y.shape
    ak = te.reduce_axis((0, k), name="k")

    z = te.compute(
        (batch, m, n_o * n_i),
        lambda b, i, j: te.sum(
            x[b, i, ak].astype("int32")
            * packed_y[b, tvm.tir.indexdiv(j, 16), tvm.tir.indexdiv(ak, 4), j % 16, ak % 4].astype(
                "int32"
            ),
            axis=ak,
        ),
        tag="batch_matmul_vnni",
        attrs={"schedule_rule": "meta_schedule.batch_matmul_vnni"},
    )

    _, a_y, _ = z.op.axis
    cfg.define_split("tile_y", a_y, num_outputs=2)
    cfg.define_knob("layout_trans_compute_root", [0, 1])

    return z


def batch_matmul_vnni_schedule(cfg, s, C, O, layout_trans):
    """Schedule batch_matmul compute using VNNI vpdpbusd instruction"""
    # C: The output of batched GEMM
    # O: The output of the fused op

    # Schedule the GEMM part
    s, fused_inner = dense_vnni_schedule(cfg, s, C, O, do_parallel=False)
    # Parallelize over batch
    fused = s[O].fuse(O.op.axis[0], fused_inner)
    s[O].parallel(fused)

    if cfg["layout_trans_compute_root"].val:
        s[layout_trans].compute_root()
        schedule_injective_from_existing(s, layout_trans)
    else:
        s[layout_trans].compute_at(s[O], fused)
        _, _, _, ni, ki = s[layout_trans].op.axis
        s[layout_trans].vectorize(ki)
        s[layout_trans].unroll(ni)

    return s


@autotvm.register_topi_compute("batch_matmul.x86")
def batch_matmul(
    cfg, tensor_a, tensor_b, out_shape=None, out_dtype=None, transpose_a=False, transpose_b=True
):
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
    if cfg.is_fallback:
        if transpose_a:
            _, K, M = get_const_tuple(tensor_a.shape)
        else:
            _, M, K = get_const_tuple(tensor_a.shape)
        if transpose_b:
            _, N, _ = get_const_tuple(tensor_b.shape)
        else:
            _, _, N = get_const_tuple(tensor_b.shape)
        _default_batch_matmul_config(cfg, M, N, K)
    return nn.batch_matmul(
        tensor_a,
        tensor_b,
        out_shape,
        out_dtype,
        transpose_a,
        transpose_b,
    )


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
            if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
                s[B].compute_inline()
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


@autotvm.register_topi_schedule("batch_matmul_vnni.x86")
def schedule_batch_matmul_vnni(cfg, outs):
    """Schedule for batch_matmul_vnni"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "batch_matmul_vnni" in op.tag:
            layout_trans = op.input_tensors[1]
            batch_matmul_vnni_schedule(cfg, s, op.output(0), outs[0], layout_trans)

    traverse_inline(s, outs[0].op, _callback)
    return s


def _default_batch_matmul_config(cfg, M, N, K):
    cfg["tile_k"] = SplitEntity([K // 16, 16])
    x_bn = get_max_power2_factor(N, 8)
    cfg["tile_x"] = SplitEntity([N // x_bn, x_bn])
    y_bn = get_max_power2_factor(M, 8)
    cfg["tile_y"] = SplitEntity([M // y_bn, y_bn])


def batch_matmul_blas_common(cfg, tensor_a, tensor_b, out_shape, trans_a, trans_b, lib):
    """Computes batch matrix multiplication of `tensor_a` and `tensor_b` when `tensor_a` and
    `tensor_b` are data in batch, using one of BLAS libraries. Supports broadcasting in batch
    dimension.

    Parameters
    ----------
    cfg : ConfigSpace
        Autotvm tuning space config file

    tensor_a : tvm.te.Tensor
        3-D with shape [batch, M, K] or [batch, K, M].

    tensor_b : tvm.te.Tensor
        3-D with shape [batch, K, N] or [batch, N, K].

    out_shape : List[Optional]
        Explicit intended output shape of the computation. Can be useful in cases
        with dynamic input shapes.

    trans_a : Optional[bool] = False
        Whether the first tensor is in transposed format.

    trans_b : Optional[bool] = True
        Whether the second tensor is in transposed format.

    lib : A contrib module which implements batch_matmul function
        cblas and mkl are supported

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    assert len(tensor_a.shape) == 3 and len(tensor_b.shape) == 3, "only support 3-dim batch_matmul"
    if trans_a:
        XB, XK, M = get_const_tuple(tensor_a.shape)
    else:
        XB, M, XK = get_const_tuple(tensor_a.shape)
    if trans_b:
        YB, N, YK = get_const_tuple(tensor_b.shape)
    else:
        YB, YK, N = get_const_tuple(tensor_a.shape)
    assert (XB == YB) or (YB == 1) or (XB == 1), "batch dimension doesn't match"
    assert XK == YK, "shapes of x and y is inconsistent"
    if out_shape is not None:
        assert out_shape[0] in (XB, YB), "got invalid output shape"
        assert out_shape[1] == M, "got invalid output shape"
        assert out_shape[2] == N, "got invalid output shape"
    cfg.add_flop(XB * M * N * XK * 2)
    return lib.batch_matmul(tensor_a, tensor_b, trans_a, trans_b)


@autotvm.register_topi_compute("batch_matmul_cblas.x86")
def batch_matmul_cblas(
    cfg, tensor_a, tensor_b, out_shape=None, out_dtype=None, transpose_a=False, transpose_b=True
):
    """Compute batch_matmul using cblas"""
    del out_dtype  # Unused argument
    return batch_matmul_blas_common(
        cfg, tensor_a, tensor_b, out_shape, transpose_a, transpose_b, cblas
    )


@autotvm.register_topi_schedule("batch_matmul_cblas.x86")
def schedule_batch_matmul_cblas(_, outs):
    """Create schedule for batch_matmul_cblas"""
    return generic.schedule_extern(outs)


@autotvm.register_topi_compute("batch_matmul_mkl.x86")
def batch_matmul_mkl(
    cfg, tensor_a, tensor_b, out_shape=None, out_dtype=None, transpose_a=False, transpose_b=True
):
    """Compute batch_matmul using mkl"""
    del out_dtype  # Unused argument
    return batch_matmul_blas_common(
        cfg, tensor_a, tensor_b, out_shape, transpose_a, transpose_b, mkl
    )


@autotvm.register_topi_schedule("batch_matmul_mkl.x86")
def schedule_batch_matmul_mkl(_, outs):
    """Create schedule for batch_matmul_mul"""
    return generic.schedule_extern(outs)
