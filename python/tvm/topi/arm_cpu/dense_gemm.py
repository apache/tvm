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
"""GEMM Convolution schedule on AArch64"""
import tvm
from tvm import te
from tvm.topi import nn
from tvm.topi.arm_cpu.arm_utils import get_tiling_A, get_tiling_B_transformed
from ..utils import get_const_tuple, traverse_inline
from .. import tag

# Compute function
def dense_gemm_compute(
    cfg, data, weight, bias=None, out_dtype=None, transpose_a=False, transpose_b=True
):
    """
    Compute dense using GeMM.

    transpose_b : Optional[bool] = True
    Whether the weight tensor is in transposed format.
    """

    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape)  # batch, in_dim
    if bool(transpose_b):  # out_dim
        (N, _) = get_const_tuple(weight.shape)
    else:
        (_, N) = get_const_tuple(weight.shape)

    in_dtype = data.dtype

    tile_M, tile_K_A = get_tiling_A(False, in_dtype)
    tile_N, tile_K_B = get_tiling_B_transformed(False, out_dtype, False)

    pad_M = 0
    pad_K = 0
    pad_N = 0

    if M % tile_M != 0:
        pad_M = tile_M - (M % tile_M)

    if K % tile_K_A != 0:
        pad_K = tile_K_A - (K % tile_K_A)

    M_padded = M + pad_M
    K_padded = K + pad_K
    k = te.reduce_axis((0, K_padded), name="k")

    pad_before = (0, 0)
    pad_after = (pad_M, pad_K)

    if pad_K != 0:
        data = nn.pad(data, pad_before=pad_before, pad_after=pad_after, name="A_padded_K")
    elif pad_M != 0:
        data = nn.pad(data, pad_before=pad_before, pad_after=pad_after, name="A_padded_M")

    if N % tile_N != 0:
        pad_N = tile_N - (N % tile_N)
    N_padded = N + pad_N

    if bool(transpose_b):
        weight = te.compute(
            (K_padded, N_padded), lambda x, y: weight[y, x], name="weight_transposed"
        )

    if pad_K != 0 or pad_N != 0:
        weight = nn.pad(weight, pad_before=(0, 0), pad_after=(pad_N, pad_K), name="weight_padded")

    C = te.compute(
        (M_padded, N_padded),
        lambda x, y: te.sum(
            data[x, k].astype(out_dtype) * weight[k, y].astype(out_dtype),
            axis=k,
        ).astype(out_dtype),
        name="C",
    )

    if bias is not None:
        C = te.compute(
            (M_padded, N_padded),
            lambda i, j: C[i, j] + bias[j].astype(out_dtype),
            tag=tag.BROADCAST,
            name="dense_biased_output",
        )

    zero = (
        tvm.tir.const(1, C.dtype) * C[0, N_padded - 1]
        - tvm.tir.const(1, C.dtype) * C[0, N_padded - 1]
    )

    out = te.compute(
        (M, N), lambda x, y: (C[x, y] + zero).astype(out_dtype), name="dense_gemm_output"
    )

    return out


def _dense_gemm_schedule_template(s, out):
    C = out.op.input_tensors[0]
    A = C.op.input_tensors[0]
    in_type = A.dtype
    y_tile_size, _ = get_tiling_B_transformed(False, in_type)
    if C.op.name == "dense_biased_output":
        s[C].compute_inline()
        C = C.op.input_tensors[0]
    x, y = s[C].op.axis
    (k,) = s[C].op.reduce_axis
    k_outer, k_inner = s[C].split(k, factor=4)
    x_outer, x_inner = s[C].split(x, factor=4)
    y_outer, y_inner = s[C].split(y, factor=y_tile_size)
    s[C].parallel(x_outer)
    s[C].reorder(
        x_outer,
        y_outer,
        k_outer,
        k_inner,
        x_inner,
        y_inner,
    )
    s[C].unroll(x_inner)
    s[C].vectorize(y_inner)

    return s


def dense_gemm_schedule(cfg, outs):
    """Schedule the dense_gemm strategy"""
    s = te.create_schedule([x.op for x in outs])
    out = outs[0]
    x, y = out.op.axis
    _, inner = s[out].split(y, 4)
    s[out].parallel(x)
    s[out].vectorize(inner)

    def _callback(op):
        if "dense_gemm_output" in op.name:
            _dense_gemm_schedule_template(s, op.output(0))

    traverse_inline(s, out.op, _callback)
    return s
