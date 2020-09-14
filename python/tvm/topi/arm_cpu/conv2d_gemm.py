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
from tvm import te
from tvm.topi import nn
from tvm.autotvm.task.space import AnnotateEntity, ReorderEntity, OtherOptionEntity
from ..util import get_const_tuple, get_const_int
from ..nn.util import get_pad_tuple
from .tensor_intrin import gemm_quantized, gemm_quantized_impl


def is_aarch64_arm():
    """ Checks whether we are compiling for an AArch64 target. """
    target = tvm.target.Target.current(allow_none=False)
    return "aarch64" in target.attrs.get("mtriple", "")


# Compute function
def compute_conv2d_gemm_without_weight_transform(
    cfg, data, B_interleaved_t, strides, padding, dilation, out_dtype, kernel_size, output_channels
):
    """Compute conv2d by transforming the input,
    executing GEMM and transforming the output back"""
    batches, IH, IW, IC = get_const_tuple(data.shape)

    KH, KW = get_const_tuple(kernel_size)
    OC = get_const_int(output_channels)

    K_AREA = KH * KW

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

    # --- Im2col
    M = OH * OW
    K = IC * K_AREA
    N = OC

    A_shape = (batches, M, K)
    if K_AREA == 1:
        A = te.compute(
            A_shape,
            lambda n, x, y: data_pad[n, HSTR * (x // OW), WSTR * (x % OW), y],
            name="data_flatten",
        )
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
    N_transformed = B_interleaved_t.shape[0]

    # --- Pad if necessary
    idxm = tvm.tir.indexmod

    pad_m = 0
    pad_k = 0

    if M % 4 != 0:
        pad_m = 4 - (M % 4)

    if K % 16 != 0:
        pad_k = 16 - (K % 16)

    M_padded = M + pad_m
    K_padded = K + pad_k

    pad_before = (0, 0, 0)
    pad_after = (0, pad_m, pad_k)

    if pad_m != 0 or pad_k != 0:
        A = nn.pad(A, pad_before=pad_before, pad_after=pad_after, name="A_padded")

    # --- GEMM: A*B'
    k = te.reduce_axis((0, K_padded), "k")

    A_interleaved = te.compute(
        (batches, M_padded // 4, K_padded // 16, 4, 16),
        lambda b, x, y, z, w: A[b, z + 4 * x, w + 16 * y],
        name="A_interleaved",
    )

    C_interleaved = te.compute(
        (batches, M_padded // 4, N_transformed, 4, 4),
        lambda b, x, y, w, z: te.sum(
            A_interleaved[b, x, k // 16, w, idxm(k, 16)].astype(out_dtype)
            * B_interleaved_t[y, k // 16, z, idxm(k, 16)].astype(out_dtype),
            axis=k,
        ),
        name="C_interleaved",
    )

    # --- Unpack C
    C = te.compute(
        (batches, M, N),
        lambda b, x, y: C_interleaved[b, x // 4, y // 4, idxm(x, 4), idxm(y, 4)],
        name="C",
    )

    # --- Produce the conv output
    out_shape = (batches, OH, OW, OC)
    out = te.compute(out_shape, lambda b, x, y, z: C(b, y + OW * x, z), name="conv2d_gemm_output")

    # Configuration space
    x, y = cfg.axis(M_padded // 4), cfg.axis(K_padded // 16)
    cfg.define_reorder("reorder_gemm", [x, y], policy="candidate", candidate=[[x, y], [y, x]])

    outer_loop, inner_loop = cfg.axis(4), cfg.axis(16)
    cfg.define_annotate(
        "A_interleaved_unroll_vec", [outer_loop, inner_loop], policy="try_unroll_vec"
    )
    cfg.define_knob("gemm_quantized_unroll", [True, False])
    cfg.define_knob("gemm_quantized_interleave", [True, False])

    # Fallback configuration
    if cfg.is_fallback:
        cfg["reorder_gemm"] = ReorderEntity([0, 1])
        cfg["A_interleaved_unroll_vec"] = AnnotateEntity(["unroll", "vec"])
        cfg["gemm_quantized_unroll"] = OtherOptionEntity(False)
        cfg["gemm_quantized_interleave"] = OtherOptionEntity(True)
    return out


# Schedules
def schedule_conv2d_gemm(cfg, s, out, final_out):
    """Create schedule for tensors"""
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
    b, xo, yo, xi, yi = C_interleaved.op.axis
    outer_gemm, inner_gemm = cfg["reorder_gemm"].apply(s, C_interleaved, [xo, yo])
    s[C_interleaved].reorder(yi, xi)
    b_outer_gemm_fused = s[C_interleaved].fuse(b, outer_gemm)
    s[C_interleaved].parallel(b_outer_gemm_fused)
    s[A_interleaved].compute_at(s[C_interleaved], b_outer_gemm_fused)
    _, _, _, outer_A_interleaved, inner_A_interleaved = A_interleaved.op.axis
    cfg["A_interleaved_unroll_vec"].apply(
        s, A_interleaved, [outer_A_interleaved, inner_A_interleaved]
    )

    in_type = A_interleaved.dtype
    out_type = C.dtype
    if is_aarch64_arm() and out_type == "int32":
        K = A_interleaved_input.shape[2]
        _, M, N = C.shape
        assert in_type in ["int8", "uint8"], "Only int8 and uint8 gemm are supported"
        unroll = cfg["gemm_quantized_unroll"].val
        interleave = cfg["gemm_quantized_interleave"].val
        gemm = gemm_quantized(M, N, K, unroll, interleave, in_type, out_type)
        s[C_interleaved].pragma(
            b_outer_gemm_fused,
            "import_llvm",
            gemm_quantized_impl(M, N, K, unroll, interleave, in_type),
        )
        s[C_interleaved].tensorize(yi, gemm)

    # Output transform
    if out != final_out:
        n, h, w, c = out.op.axis
        _, inner = s[out].split(c, 4)
        s[C].compute_at(s[out], inner)
        s[out].vectorize(inner)
    return s
