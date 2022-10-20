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
# pylint: disable=invalid-name, unused-variable, too-many-locals
# pylint: disable=unused-argument, redefined-builtin
"""GEMM Convolution schedule on ARM"""
import tvm
from tvm.target import Target
from tvm import te
from tvm.topi import nn
from tvm.autotvm.task.space import AnnotateEntity, ReorderEntity, OtherOptionEntity
from ..utils import get_const_tuple, get_const_int
from ..nn.utils import get_pad_tuple
from .tensor_intrin import (
    gemm_4x4_int8_int8_int32,
    gemm_acc_4x4_int8_int8_int32,
    gemm_acc_nx16_int8_int8_int32,
    gemm_acc_2x2_int8_int8_int32,
)


def configure_knobs(cfg, M, K, target):
    """Configure auto-tuning knobs for the interleaved strategy"""

    x, y = cfg.axis(M // 4), cfg.axis(K // 16)
    cfg.define_reorder("reorder_gemm", [x, y], policy="candidate", candidate=[[x, y], [y, x]])

    outer_loop, inner_loop = cfg.axis(4), cfg.axis(16)
    cfg.define_annotate(
        "A_interleaved_unroll_vec", [outer_loop, inner_loop], policy="try_unroll_vec"
    )

    # Fallback configuration
    if cfg.is_fallback:
        cfg["reorder_gemm"] = ReorderEntity([0, 1])
        cfg["A_interleaved_unroll_vec"] = AnnotateEntity(["unroll", "vec"])

    if not target.features.has_dotprod:
        cfg.define_knob("gemm_quantized_unroll", [True, False])
        if cfg.is_fallback:
            cfg["gemm_quantized_unroll"] = OtherOptionEntity(False)


# Compute function
def compute_conv2d_gemm_without_weight_transform(
    cfg,
    data,
    B_interleaved_t,
    strides,
    padding,
    dilation,
    out_dtype,
    kernel_size,
    output_channels,
    interleave_A,
):
    """Compute conv2d by transforming the input,
    executing GEMM and transforming the output back"""
    batches, IH, IW, IC = get_const_tuple(data.shape)

    KH, KW = get_const_tuple(kernel_size)
    OC = get_const_int(output_channels)
    kernel_area = KH * KW

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = get_const_tuple(dilation)

    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1

    pad_top, pad_left, pad_down, pad_right = get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    HSTR, WSTR = strides if isinstance(strides, (tuple, list)) else (strides, strides)

    OH = (IH + pad_top + pad_down - dilated_kernel_h) // HSTR + 1
    OW = (IW + pad_left + pad_right - dilated_kernel_w) // WSTR + 1
    if pad_top or pad_left:
        data_pad = nn.pad(
            data, [0, pad_top, pad_left, 0], [0, pad_down, pad_right, 0], name="data_pad"
        )
    else:
        data_pad = data

    # Im2col
    M = OH * OW
    K = IC * kernel_area
    N = OC

    A_shape = (batches, M, K)
    if kernel_area == 1:
        A = tvm.topi.reshape(data_pad, A_shape)
    else:
        A = te.compute(
            A_shape,
            lambda n, x, y: data_pad[
                n,
                HSTR * (x // OW) + dilation_h * ((y // IC) // KW),
                WSTR * (x % OW) + dilation_w * ((y // IC) % KW),
                y % IC,
            ],
            name="data_im2col",
        )

    #  Pad if necessary
    N_transformed = B_interleaved_t.shape[0]
    tile_rows_B = B_interleaved_t.shape[2]
    tile_cols_B = B_interleaved_t.shape[3]

    # Select the tiling strategy for A.
    # The tiling information is chosen to maximize register usage during
    # the tile computation.
    #
    # Please refer to:
    # - https://discuss.tvm.apache.org/t/rfc-improve-quantized-convolution-performance-for-armv8-architectures # pylint: disable=line-too-long
    # - https://discuss.tvm.apache.org/t/rfc-accelerate-quantized-convolution-through-dot-product
    # - https://discuss.tvm.apache.org/t/rfc-improve-quantized-convolution-through-mmla-instruction
    # - Conv2DGemmWeightTransformRel in src/relay/op/nn/convolution.h
    # In order to have more information
    #
    target = Target.current(allow_none=False)
    if target.features.has_matmul_i8:
        # If smmla/ummla is enabled, we are loading 8 rows from A. Each row
        # will contain 8 elements
        tile_rows_A = 8
        tile_cols_A = 8
    elif target.features.has_dotprod and interleave_A:
        # If dot product has been enabled, and we are interleaving A
        # tile size should be 8x4
        tile_rows_A = 8
        tile_cols_A = 4
    else:
        # If either there is no dot product or if we are using a native strategy
        # tile size should be 4x16
        tile_rows_A = 4
        tile_cols_A = 16

    pad_M = 0
    pad_K = 0

    if M % tile_rows_A != 0:
        pad_M = tile_rows_A - (M % tile_rows_A)

    if K % tile_cols_A != 0:
        pad_K = tile_cols_A - (K % tile_cols_A)

    M_padded = M + pad_M
    K_padded = K + pad_K
    N_padded = N_transformed * tile_rows_B

    pad_before = (0, 0, 0)
    pad_after = (0, pad_M, pad_K)

    if pad_M != 0 or pad_K != 0:
        A = nn.pad(A, pad_before=pad_before, pad_after=pad_after, name="A_padded")

    idxm = tvm.tir.indexmod
    k = te.reduce_axis((0, K_padded), "k")

    if interleave_A:
        # Configuration space
        configure_knobs(cfg, M_padded, K_padded, target)

        # Pack the input data
        A_interleaved = te.compute(
            (batches, M_padded // tile_rows_A, K_padded // tile_cols_A, tile_rows_A, tile_cols_A),
            lambda b, x, y, z, w: A[b, z + tile_rows_A * x, w + tile_cols_A * y],
            name="A_interleaved",
        )
        target = Target.current(allow_none=False)
        if target.features.has_matmul_i8:
            # Execute GEMM. In the case of mmla, we need to enforce the tiling
            # from the compute. This is because mmla is doing a tiled computation
            # as well. So we have a big 8x12 tile, with small 2x2 sub-tiles
            # generated by mmla. In theory we could make the tile 2x2 and
            # fuse and split during scheduling, but this would not work
            # because of possible padding
            C_interleaved = te.compute(
                (
                    batches,
                    M_padded // tile_rows_A,
                    N_transformed,
                    tile_rows_A // 2,
                    tile_rows_B // 2,
                    2,
                    2,
                ),
                lambda b, x, y, w, z, s, t: te.sum(
                    A_interleaved[b, x, k // tile_cols_A, 2 * w + s, idxm(k, tile_cols_A)].astype(
                        "int32"
                    )
                    * B_interleaved_t[y, k // tile_cols_B, 2 * z + t, idxm(k, tile_cols_B)].astype(
                        "int32"
                    ),
                    axis=k,
                ),
                name="C_interleaved",
            )
            # Unpack the result
            C = te.compute(
                (batches, M, N),
                lambda b, x, y: C_interleaved[
                    b,
                    x // tile_rows_A,
                    y // tile_rows_B,
                    idxm(x, tile_rows_A) // 2,
                    idxm(y, tile_rows_B) // 2,
                    idxm(idxm(x, tile_rows_A), 2),
                    idxm(idxm(y, tile_rows_B), 2),
                ].astype(out_dtype),
                name="C",
            )
        else:
            # Execute GEMM
            C_interleaved = te.compute(
                (batches, M_padded // tile_rows_A, N_transformed, tile_rows_A, tile_rows_B),
                lambda b, x, y, w, z: te.sum(
                    A_interleaved[b, x, k // tile_cols_A, w, idxm(k, tile_cols_A)].astype("int32")
                    * B_interleaved_t[y, k // tile_cols_B, z, idxm(k, tile_cols_B)].astype("int32"),
                    axis=k,
                ),
                name="C_interleaved",
            )
            # Unpack the result
            C = te.compute(
                (batches, M, N),
                lambda b, x, y: C_interleaved[
                    b,
                    x // tile_rows_A,
                    y // tile_rows_B,
                    idxm(x, tile_rows_A),
                    idxm(y, tile_rows_B),
                ].astype(out_dtype),
                name="C",
            )
        zero = tvm.tir.const(0)
    else:
        # No need to pack/unpack, execute GEMM directly
        C = te.compute(
            (batches, M_padded, N_padded),
            lambda b, x, y: te.sum(
                A[b, x, k].astype("int32")
                * B_interleaved_t[
                    y // tile_rows_B, k // tile_cols_B, idxm(y, tile_rows_B), idxm(k, tile_cols_B)
                ].astype("int32"),
                axis=k,
            ),
            name="C",
        )

        # We need to ensure that infer bound pass does not remove the padding
        # which is necessary for the tensorizations to work. So we need to
        # add a dummy reference to the padding area of the result
        zero = (
            tvm.tir.const(1, C.dtype) * C[0, M_padded - 1, N_padded - 1]
            - tvm.tir.const(1, C.dtype) * C[0, M_padded - 1, N_padded - 1]
        )

    # Reshape the result into a convolution output
    out_shape = (batches, OH, OW, OC)
    out = te.compute(
        out_shape,
        lambda b, x, y, z: (C(b, y + OW * x, z) + zero).astype(out_dtype),
        name="conv2d_gemm_output",
    )
    return out


def schedule_conv2d_gemm_interleaved(cfg, s, out, final_out):
    """Schedule the conv2d_gemm interleaved strategy"""
    C = out.op.input_tensors[0]
    C_interleaved = C.op.input_tensors[0]
    A_interleaved = C_interleaved.op.input_tensors[0]

    # Input transform
    A_interleaved_input = A_interleaved.op.input_tensors[0]
    if A_interleaved_input.op.name == "A_padded":
        s[A_interleaved_input].compute_at(s[A_interleaved], A_interleaved.op.axis[3])
        s[A_interleaved_input].vectorize(A_interleaved_input.op.axis[2])
        s[A_interleaved_input].compute_inline()
        data_im2col = A_interleaved_input.op.input_tensors[0]
    else:
        data_im2col = A_interleaved_input

    b, m, n = data_im2col.op.axis
    if data_im2col.op.name == "data_im2col":
        n_outer, n_inner = s[data_im2col].split(n, 16)
        s[data_im2col].unroll(n_outer)
        s[data_im2col].vectorize(n_inner)
        b_m_fused = s[data_im2col].fuse(b, m)
        s[data_im2col].parallel(b_m_fused)
    else:
        s[data_im2col].compute_inline()

    # Computation(through tensorize)
    b, xo, yo, xi, yi = C_interleaved.op.axis[0:5]
    outer_gemm, inner_gemm = cfg["reorder_gemm"].apply(s, C_interleaved, [xo, yo])

    b_outer_gemm_fused = s[C_interleaved].fuse(b, outer_gemm)
    s[C_interleaved].parallel(b_outer_gemm_fused)
    s[A_interleaved].compute_at(s[C_interleaved], b_outer_gemm_fused)
    _, _, _, outer_A_interleaved, inner_A_interleaved = A_interleaved.op.axis
    cfg["A_interleaved_unroll_vec"].apply(
        s, A_interleaved, [outer_A_interleaved, inner_A_interleaved]
    )

    in_type = A_interleaved.dtype
    out_type = C.dtype

    k = C_interleaved.op.reduce_axis[0]
    _, M, N = C.shape
    if in_type in ["int8", "uint8"]:
        target = Target.current(allow_none=False)
        if target.features.has_matmul_i8:
            gemm_acc = gemm_acc_2x2_int8_int8_int32(in_type)
            xi_inner, yi_inner = C_interleaved.op.axis[-2:]
            k_outer, k_inner = s[C_interleaved].split(k, 8)
            s[C_interleaved].reorder(
                b_outer_gemm_fused, inner_gemm, k_outer, xi, yi, xi_inner, yi_inner, k_inner
            )
            s[C_interleaved].tensorize(xi_inner, gemm_acc)
            s[C_interleaved].unroll(xi)
            s[C_interleaved].unroll(yi)
        elif target.features.has_dotprod:
            gemm_acc = gemm_acc_4x4_int8_int8_int32(in_type)
            xi_outer, yi_outer, xi_inner, yi_inner = s[C_interleaved].tile(
                xi, yi, x_factor=8, y_factor=4
            )
            k_outer, k_inner = s[C_interleaved].split(k, 4)
            xi_inner_outer, xi_inner_inner = s[C_interleaved].split(xi_inner, 4)
            s[C_interleaved].reorder(
                b_outer_gemm_fused,
                inner_gemm,
                xi_outer,
                yi_outer,
                k_outer,
                xi_inner_outer,
                xi_inner_inner,
                yi_inner,
                k_inner,
            )
            s[C_interleaved].tensorize(xi_inner_inner, gemm_acc)
            s[C_interleaved].unroll(xi_inner_outer)

        elif target.features.has_asimd:
            s[C_interleaved].reorder(yi, xi)
            K = A_interleaved_input.shape[2]
            assert in_type in ["int8", "uint8"], "Only int8 and uint8 gemm are supported"
            unroll = cfg["gemm_quantized_unroll"].val
            gemm = gemm_4x4_int8_int8_int32(M, N, K, unroll, in_type)
            s[C_interleaved].tensorize(yi, gemm)

    # Output transform
    if out != final_out:
        n, h, w, c = out.op.axis
        _, inner = s[out].split(c, 4)
        s[C].compute_at(s[out], inner)
        s[out].vectorize(inner)
    return s


def schedule_conv2d_gemm_native(cfg, s, out, final_out):
    """Schedule the conv2d_gemm hybrid strategy"""
    C = out.op.input_tensors[0]
    A = C.op.input_tensors[0]
    in_type = A.dtype

    # Computation
    b, x, y = C.op.axis
    (k,) = C.op.reduce_axis
    k_outer, k_inner = s[C].split(k, 16)
    x_outer, y_outer, x_inner, y_inner = s[C].tile(x, y, x_factor=4, y_factor=16)
    s[C].reorder(b, x_outer, y_outer, k_outer, x_inner, y_inner, k_inner)
    gemm_acc = gemm_acc_nx16_int8_int8_int32(in_type, rows=1)
    s[C].unroll(x_inner)
    s[C].tensorize(y_inner, gemm_acc)
    s[C].parallel(x_outer)

    # Input transform
    if A.op.name == "A_padded":
        padding_A = True
        data_im2col = A.op.input_tensors[0]
    else:
        padding_A = False
        data_im2col = A

    b, m, n = data_im2col.op.axis
    if data_im2col.op.name == "data_im2col":
        n_outer, n_inner = s[data_im2col].split(n, 16)
        s[data_im2col].unroll(n_outer)
        s[data_im2col].vectorize(n_inner)
        s[data_im2col].parallel(m)
    elif padding_A:
        s[data_im2col].compute_inline()
        s[A].compute_at(s[C], x_inner)
    else:
        s[data_im2col].compute_at(s[C], x_inner)

    # Output transform
    if out != final_out:
        n, h, w, c = out.op.axis
        _, inner = s[out].split(c, 4)
        s[out].vectorize(inner)
    return s
