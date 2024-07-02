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
"""GeMM dense schedule on AArch64"""
import tvm
from tvm import te
from tvm.topi import nn
from tvm.topi.arm_cpu.arm_utils import get_tiling_A, get_tiling_B_transformed, pad_dim_to_multiple
from ..utils import get_const_tuple, traverse_inline
from .. import tag

# Compute function
def dense_gemm_compute(
    cfg, data, weight, bias=None, out_dtype=None, transpose_a=False, transpose_b=True
):
    """
    Compute dense using GeMM.

    Parameters
    ----------
    cfg : Autotvm tuning space config file,
        empty in this case, but it's needed as an arg.

    data : tvm.te.Tensor
        2-D with shape [M, K] or [K, M].

    weight : tvm.te.Tensor
        2-D with shape [K, N] or [N, K].

    bias : Optional[tvm.te.Tensor]
        1-D with shape [N]


    out_dtype : Optional[str]
        Specifies the output data type.

    transpose_a : Optional[bool] = False
    Whether the data tensor is in transposed format.

    transpose_b : Optional[bool] = True
    Whether the weight tensor is in transposed format.

    Returns
    -------
    out : tvm.te.Tensor
        1-D with shape [out_dim]
    """

    if out_dtype is None:
        out_dtype = data.dtype
    M, K = get_const_tuple(data.shape)  # batch, in_dim
    if bool(transpose_b):  # out_dim
        (N, _) = get_const_tuple(weight.shape)
    else:
        (_, N) = get_const_tuple(weight.shape)

    tile_M, tile_K = get_tiling_A(False, out_dtype)
    tile_N, _ = get_tiling_B_transformed(False, out_dtype, False)

    M_padded, pad_M = pad_dim_to_multiple(M, tile_M)
    K_padded, pad_K = pad_dim_to_multiple(K, tile_K)
    N_padded, pad_N = pad_dim_to_multiple(N, tile_N)
    m_pad_after = (pad_M, pad_K)
    n_pad_after = (pad_N, pad_K) if transpose_b else (pad_K, pad_N)

    if pad_M != 0 or pad_K != 0:
        data = nn.pad(data, pad_before=(0, 0), pad_after=m_pad_after, name="data_padded")

    k = te.reduce_axis((0, K_padded), name="k")

    if bool(transpose_b):
        weight = te.compute(
            (K_padded, N_padded), lambda x, y: weight[y, x], name="weight_transposed"
        )

    if pad_N != 0 or pad_K != 0:
        weight = nn.pad(weight, pad_before=(0, 0), pad_after=n_pad_after, name="weight_padded")

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

    # We need to ensure that infer bound pass does not remove the padding
    # which is necessary for the tensorizations to work. So we need to
    # add a dummy reference to the padding area of the result
    zero = (
        tvm.tir.const(1, C.dtype) * C[0, N_padded - 1]
        - tvm.tir.const(1, C.dtype) * C[0, N_padded - 1]
    )

    out = te.compute(
        (M, N), lambda x, y: (C[x, y] + zero).astype(out_dtype), name="dense_gemm_output"
    )

    return out


def _dense_gemm_schedule(s, out):
    C = out.op.input_tensors[0]
    A = C.op.input_tensors[0]
    out_type = A.dtype
    tile_M, tile_K = get_tiling_A(False, out_type)
    tile_N, _ = get_tiling_B_transformed(False, out_type, False)

    if C.op.name == "dense_biased_output":
        s[C].compute_inline()
        C = C.op.input_tensors[0]
    x, y = s[C].op.axis
    (k,) = s[C].op.reduce_axis

    k_outer, k_inner = s[C].split(k, factor=tile_K)
    x_outer, x_inner = s[C].split(x, factor=tile_M)
    y_outer, y_inner = s[C].split(y, factor=tile_N)
    y_inner_outer, y_inner_inner = s[C].split(y_inner, nparts=4)
    s[C].parallel(x_outer)
    s[C].reorder(
        x_outer,
        y_outer,
        k_outer,
        k_inner,
        y_inner_outer,
        x_inner,
        y_inner_inner,
    )
    s[C].unroll(y_inner_outer)
    s[C].unroll(x_inner)
    s[C].vectorize(y_inner_inner)

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
            _dense_gemm_schedule(s, op.output(0))

    traverse_inline(s, out.op, _callback)
    return s
