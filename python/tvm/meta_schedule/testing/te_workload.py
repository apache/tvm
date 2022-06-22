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
"""Workloads in TE"""
# pylint: disable=missing-docstring
from typing import Tuple

from tvm import te, tir, topi


def batch_matmul_nkkm(  # pylint: disable=invalid-name,missing-docstring
    B: int,
    N: int,
    M: int,
    K: int,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    x = te.placeholder((B, N, K), name="X")
    y = te.placeholder((B, K, M), name="Y")
    k = te.reduce_axis((0, K), name="k")
    z = te.compute(  # pylint: disable=invalid-name
        (B, N, M),
        lambda b, i, j: te.sum(x[b][i][k] * y[b][k][j], axis=[k]),
        name="Z",
    )
    return (x, y, z)


def conv1d_nlc(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    L: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    inputs = te.placeholder((N, L, CI), name="inputs")
    weight = te.placeholder((kernel_size, CI // groups, CO), name="weight")

    batch_size, in_len, _ = inputs.shape
    k_len, channel_per_group, out_channel = weight.shape
    out_channel_per_group = out_channel // groups
    out_len = (in_len + 2 * padding - dilation * (k_len - 1) - 1) // stride + 1
    rc = te.reduce_axis((0, channel_per_group), name="rc")
    rl = te.reduce_axis((0, k_len), name="rl")

    padded = topi.nn.pad(inputs, [0, padding, 0])
    output = te.compute(
        (batch_size, out_len, out_channel),
        lambda n, l, co: te.sum(
            (
                padded[
                    n,
                    l * stride + rl * dilation,
                    co // out_channel_per_group * channel_per_group + rc,
                ]
                * weight[rl, rc, co]
            ),
            axis=[rl, rc],
        ),
        name="conv1d_nlc",
    )
    return (inputs, weight, output)


def conv2d_nhwc(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    inputs = te.placeholder((N, H, W, CI), name="inputs")
    weight = te.placeholder((kernel_size, kernel_size, CI // groups, CO), name="weight")
    batch_size, in_h, in_w, _ = inputs.shape
    k_h, k_w, channel_per_group, out_channel = weight.shape
    out_channel_per_group = out_channel // groups

    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    rc = te.reduce_axis((0, channel_per_group), name="rc")

    padded = topi.nn.pad(inputs, [0, padding, padding, 0])
    output = te.compute(
        (batch_size, out_h, out_w, out_channel),
        lambda n, h, w, co: te.sum(
            (
                padded[
                    n,
                    h * stride + rh * dilation,
                    w * stride + rw * dilation,
                    co // out_channel_per_group * channel_per_group + rc,
                ]
                * weight[rh, rw, rc, co]
            ),
            axis=[rh, rw, rc],
        ),
        name="conv2d_nhwc",
    )
    return (inputs, weight, output)


def conv3d_ndhwc(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    D: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    inputs = te.placeholder((N, D, H, W, CI), name="inputs")
    weight = te.placeholder(
        (kernel_size, kernel_size, kernel_size, CI // groups, CO), name="weight"
    )
    batch_size, in_d, in_h, in_w, _ = inputs.shape
    k_d, k_h, k_w, channel_per_group, out_channel = weight.shape
    out_channel_per_group = out_channel // groups

    out_d = (in_d + 2 * padding - dilation * (k_d - 1) - 1) // stride + 1
    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rd = te.reduce_axis((0, k_d), name="rd")
    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    rc = te.reduce_axis((0, channel_per_group), name="rc")

    padded = topi.nn.pad(inputs, [0, padding, padding, padding, 0])
    output = te.compute(
        (batch_size, out_d, out_h, out_w, out_channel),
        lambda n, d, h, w, co: te.sum(
            (
                padded[
                    n,
                    d * stride + rd * dilation,
                    h * stride + rh * dilation,
                    w * stride + rw * dilation,
                    co // out_channel_per_group * channel_per_group + rc,
                ]
                * weight[rd, rh, rw, rc, co]
            ),
            axis=[rd, rh, rw, rc],
        ),
        name="conv3d_ndhwc",
    )
    return (inputs, weight, output)


def depthwise_conv2d_nhwc(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    H: int,
    W: int,
    C: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    factor: int = 1,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    inputs = te.placeholder((N, H, W, C))
    weight = te.placeholder((factor, kernel_size, kernel_size, C))
    batch_size, in_h, in_w, in_channel = inputs.shape
    factor, k_h, k_w, in_channel = weight.shape
    out_channel = in_channel * factor
    assert int(factor) == 1, "Not optimized for factor != 1"
    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    padded = topi.nn.pad(inputs, [0, padding, padding, 0])
    output = te.compute(
        (batch_size, out_h, out_w, out_channel),
        lambda n, h, w, c: te.sum(
            (
                padded[
                    n,
                    h * stride + rh * dilation,
                    w * stride + rw * dilation,
                    c // factor,
                ]
                * weight[c % factor, rh, rw, c // factor]
            ),
            axis=[rh, rw],
        ),
        name="depth_conv2d_nhwc",
    )
    return (inputs, weight, output)


def conv2d_transpose_nhwc(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    inputs = te.placeholder((N, H, W, CI), name="inputs")
    weight = te.placeholder((kernel_size, kernel_size, CI, CO), name="weight")

    batch, in_h, in_w, in_c = inputs.shape
    filter_h, filter_w, in_c, out_c = weight.shape
    stride_h, stride_w = (stride, stride)

    # compute padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = topi.nn.get_pad_tuple(
        padding, (filter_h, filter_w)
    )
    bpad_top = filter_h - 1 - fpad_top
    bpad_bottom = filter_h - 1 - fpad_bottom
    bpad_left = filter_w - 1 - fpad_left
    bpad_right = filter_w - 1 - fpad_right

    # padding stage
    padded = topi.nn.pad(
        inputs,
        [
            0,
            (bpad_top + stride_h - 1) // stride_h,
            (bpad_left + stride_w - 1) // stride_w,
            0,
        ],
        [
            0,
            (bpad_bottom + stride_h - 1) // stride_h,
            (bpad_right + stride_w - 1) // stride_w,
            0,
        ],
    )

    # remove extra padding introduced by dilatation
    idx_div = te.indexdiv
    idx_mod = te.indexmod
    border_h = idx_mod(stride_h - idx_mod(bpad_top, stride_h), stride_h)
    border_w = idx_mod(stride_w - idx_mod(bpad_left, stride_w), stride_w)

    # dilation stage
    strides = [1, stride_h, stride_w, 1]
    n = len(padded.shape)

    # We should embed this dilation directly into te.compute rather than creating a new te.compute.
    # Only in this way can we use unroll to eliminate the multiplication of zeros.
    def _dilate(*indices):
        not_zero = []
        index_tuple = []
        for i in range(n):
            if not strides[i] == 1:
                index_tuple.append(idx_div(indices[i], strides[i]))
                not_zero.append(idx_mod(indices[i], strides[i]).equal(0))
            else:
                index_tuple.append(indices[i])
        if not_zero:
            not_zero = te.all(*not_zero)
            return te.if_then_else(not_zero, padded(*index_tuple), tir.const(0.0, padded.dtype))
        return padded(*index_tuple)

    # convolution stage
    out_h = (in_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (in_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    rc = te.reduce_axis((0, in_c), name="rc")
    rh = te.reduce_axis((0, filter_h), name="rh")
    rw = te.reduce_axis((0, filter_w), name="rw")

    output = te.compute(
        (batch, out_h, out_w, out_c),
        lambda n, h, w, co: te.sum(
            _dilate(n, h + rh + border_h, w + rw + border_w, rc)
            * weight[filter_h - 1 - rh, filter_w - 1 - rw, rc, co],
            axis=[rh, rw, rc],
        ),
        name="conv2d_transpose_nhwc",
    )
    return (inputs, weight, output)


def conv2d_capsule_nhwijc(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    capsule_size: int = 4,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    inputs = te.placeholder((N, H, W, capsule_size, capsule_size, CI), name="inputs")
    weight = te.placeholder(
        (kernel_size, kernel_size, capsule_size, capsule_size, CI, CO), name="weight"
    )
    batch_size, in_h, in_w, _, _, in_channel = inputs.shape
    k_h, k_w, _, _, _, out_channel = weight.shape

    out_h = (in_h + 2 * padding - kernel_size) // stride + 1
    out_w = (in_w + 2 * padding - kernel_size) // stride + 1

    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    cap_k = te.reduce_axis((0, capsule_size), name="cap_k")
    rc = te.reduce_axis((0, in_channel), name="rc")

    padded = topi.nn.pad(inputs, [0, padding, padding, 0, 0, 0])
    output = te.compute(
        (batch_size, out_h, out_w, capsule_size, capsule_size, out_channel),
        lambda n, h, w, cap_i, cap_j, co: te.sum(
            (
                padded[n, h * stride + rh, w * stride + rw, cap_i, cap_k, rc]
                * weight[rh, rw, cap_k, cap_j, rc, co]
            ),
            axis=[rh, rw, cap_k, rc],
        ),
        name="conv2d_capsule_nhwijc",
    )
    return (inputs, weight, output)


def norm_bmn(  # pylint: disable=invalid-name,missing-docstring
    B: int,
    M: int,
    N: int,
) -> Tuple[te.Tensor, te.Tensor]:
    a = te.placeholder((B, M, N), name="A")
    i = te.reduce_axis((0, M), name="i")
    j = te.reduce_axis((0, N), name="j")
    c = te.compute(
        (B,),
        lambda b: te.sum(a[b][i][j] * a[b][i][j], axis=[i, j]),
        name="C",
    )
    d = te.compute((B,), lambda b: te.sqrt(c[b]), name="D")
    return (a, d)


def conv2d_nhwc_without_layout_rewrite(  # pylint: disable=invalid-name
    Input: int,
    Filter: int,
    stride: int,
    padding: int,
    dilation: int,
    out_dtype="float32",
):
    """A copy of `topi.nn.conv2d_nhwc` but without the 'layout_free` attribute.
    We use this in single op and subgraph evaluation
    because we don't want to introduce graph level optimization.
    """
    assert isinstance(stride, int) or len(stride) == 2
    assert isinstance(dilation, int) or len(dilation) == 2

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    batch, in_height, in_width, in_channel = Input.shape  # type: ignore
    kernel_h, kernel_w, _channel, num_filter = Filter.shape  # type: ignore

    # compute the output shape
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_top, pad_left, pad_down, pad_right = topi.nn.get_pad_tuple(
        padding, (dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = num_filter
    out_height = topi.utils.simplify(
        (in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1
    )
    out_width = topi.utils.simplify(
        (in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1
    )
    pad_before = [0, pad_top, pad_left, 0]
    pad_after = [0, pad_down, pad_right, 0]
    PaddedInput = topi.nn.pad(Input, pad_before, pad_after, name="PaddedInput")
    rc = te.reduce_axis((0, in_channel), name="rc")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")
    Output = te.compute(
        (batch, out_height, out_width, out_channel),
        lambda nn, yy, xx, ff: te.sum(
            PaddedInput[
                nn, yy * stride_h + ry * dilation_h, xx * stride_w + rx * dilation_w, rc
            ].astype(out_dtype)
            * Filter[ry, rx, rc, ff].astype(out_dtype),  # type: ignore
            axis=[ry, rx, rc],
        ),
        name="Conv2dOutput",
        tag="conv2d_nhwc",
    )
    return Output


def conv2d_nhwc_bn_relu(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    strides: int,
    padding: int,
    dilation: int = 1,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor, te.Tensor, te.Tensor, te.Tensor]:
    data = te.placeholder((N, H, W, CI), name="data")
    kernel = te.placeholder((kernel_size, kernel_size, CI, CO), name="kernel")
    bias = te.placeholder((CO,), name="bias")
    bn_scale = te.placeholder((CO,), name="bn_scale")
    bn_offset = te.placeholder((CO,), name="bn_offset")
    OH = (H + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1
    OW = (W + 2 * padding - (kernel_size - 1) * dilation - 1) // strides + 1
    conv = conv2d_nhwc_without_layout_rewrite(data, kernel, strides, padding, dilation)
    conv = te.compute(
        (N, OH, OW, CO), lambda i, j, k, l: conv[i, j, k, l] + bias[l], name="bias_add"
    )
    conv = te.compute(
        (N, OH, OW, CO), lambda i, j, k, l: conv[i, j, k, l] * bn_scale[l], name="bn_mul"
    )
    conv = te.compute(
        (N, OH, OW, CO), lambda i, j, k, l: conv[i, j, k, l] + bn_offset[l], name="bn_add"
    )
    out = topi.nn.relu(conv)
    return (data, kernel, bias, bn_offset, bn_scale, out)


def transpose_batch_matmul(  # pylint: disable=invalid-name,missing-docstring
    batch: int,
    seq_len: int,
    n_head: int,
    n_dim: int,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    query = te.placeholder((batch, seq_len, n_head, n_dim), name="query")
    value = te.placeholder((batch, seq_len, n_head, n_dim), name="value")
    query_T = te.compute(
        (batch, n_head, seq_len, n_dim),
        lambda b, h, l, d: query[b, l, h, d],
        name="query_T",
    )
    value_T = te.compute(
        (batch, n_head, n_dim, seq_len),
        lambda b, h, d, l: value[b, l, h, d],
        name="value_T",
    )
    k = te.reduce_axis((0, n_dim), name="k")
    out = te.compute(
        (batch, n_head, seq_len, seq_len),
        lambda b, h, i, j: te.sum(query_T[b, h, i, k] * value_T[b, h, k, j], axis=[k]),
        name="C",
    )
    return (query, value, out)


def conv2d_winograd_nhwc(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    tile_size = 4  # _infer_tile_size(data, kernel)
    inputs = te.placeholder((N, H, W, CI), name="inputs")
    N, H, W, CI = topi.utils.get_const_tuple(inputs.shape)
    if isinstance(dilation, int):
        dilation_h = dilation_w = dilation
    else:
        dilation_h, dilation_w = dilation

    assert (dilation_h, dilation_w) == (1, 1), "Does not support dilation"

    KH = KW = kernel_size
    HPAD, WPAD, _, _ = topi.nn.get_pad_tuple(padding, (KH, KW))
    HSTR, WSTR = (stride, stride) if isinstance(stride, int) else stride
    assert HSTR == 1 and WSTR == 1 and KH == KW

    data_pad = topi.nn.pad(inputs, (0, HPAD, WPAD, 0), (0, HPAD, WPAD, 0), name="data_pad")

    r = KW
    m = tile_size
    alpha = m + r - 1
    A, B, _G = topi.nn.winograd_util.winograd_transform_matrices(m, r, "float32")

    H = (H + 2 * HPAD - KH) // HSTR + 1
    W = (W + 2 * WPAD - KW) // WSTR + 1
    nH, nW = (H + m - 1) // m, (W + m - 1) // m
    P = N * nH * nW
    _rkh = te.reduce_axis((0, KH), name="r_kh")
    _rkw = te.reduce_axis((0, KW), name="r_kw")
    kshape = (alpha, alpha, CI, CO)
    kernel_pack = te.placeholder(kshape, inputs.dtype, name="weight")

    idxdiv = te.indexdiv
    idxmod = te.indexmod
    # pack input tile
    input_tile = te.compute(
        (alpha, alpha, P, CI),
        lambda eps, nu, p, ci: data_pad[idxdiv(p, (nH * nW))][idxmod(idxdiv(p, nW), nH) * m + eps][
            idxmod(p, nW) * m + nu
        ][ci],
        name="input_tile",
    )

    # transform data
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_b")
    data_pack = te.compute(
        (alpha, alpha, P, CI),
        lambda eps, nu, p, ci: te.sum(
            input_tile[r_a][r_b][p][ci] * B[r_a][eps] * B[r_b][nu], axis=[r_a, r_b]
        ),
        name="data_pack",
        attrs={"auto_scheduler_simplify_const_tensor_indices": ["eps", "nu", "r_a", "r_b"]},
    )

    # do batch gemm
    ci = te.reduce_axis((0, CI), name="ci")
    bgemm = te.compute(
        (alpha, alpha, P, CO),
        lambda eps, nu, p, co: te.sum(
            data_pack[eps][nu][p][ci] * kernel_pack[eps][nu][ci][co], axis=[ci]
        ),
        name="bgemm",
    )

    # inverse transform
    r_a = te.reduce_axis((0, alpha), "r_a")
    r_b = te.reduce_axis((0, alpha), "r_b")
    inverse = te.compute(
        (m, m, P, CO),
        lambda vh, vw, p, co: te.sum(
            bgemm[r_a][r_b][p][co] * A[r_a][vh] * A[r_b][vw], axis=[r_a, r_b]
        ),
        name="inverse",
        attrs={"auto_scheduler_simplify_const_tensor_indices": ["vh", "vw", "r_a", "r_b"]},
    )

    # output
    output = te.compute(
        (N, H, W, CO),
        lambda n, h, w, co: inverse[
            idxmod(h, m), idxmod(w, m), n * nH * nW + idxdiv(h, m) * nW + idxdiv(w, m), co
        ],
        name="conv2d_winograd",
    )

    return (inputs, kernel_pack, output)


def matmul(n: int, m: int, k: int) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    a = te.placeholder((n, k), name="A")
    b = te.placeholder((k, m), name="B")
    k = te.reduce_axis((0, k), name="k")
    c = te.compute(
        (n, m),
        lambda i, j: te.sum(a[i, k] * b[k, j], axis=[k]),
        name="C",
    )
    return (a, b, c)


def matmul_fp16(n: int, m: int, k: int) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    a = te.placeholder((n, k), name="A", dtype="float16")
    b = te.placeholder((k, m), name="B", dtype="float16")
    k = te.reduce_axis((0, k), name="k")

    def f_compute(i, j):
        v_a = tir.Cast(dtype="float32", value=a[i, k])
        v_b = tir.Cast(dtype="float32", value=b[k, j])
        return te.sum(v_a * v_b, axis=[k])

    c = te.compute((n, m), f_compute, name="C")
    return (a, b, c)


def matmul_relu(n: int, m: int, k: int) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    a = te.placeholder((n, k), name="A")
    b = te.placeholder((k, m), name="B")
    k = te.reduce_axis((0, k), name="k")
    c = te.compute(
        (n, m),
        lambda i, j: te.sum(a[i, k] * b[k, j], axis=[k]),
        name="C",
    )
    d = topi.nn.relu(c)  # pylint: disable=invalid-name
    return (a, b, d)


def matmul_relu_fp16(n: int, m: int, k: int) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    a = te.placeholder((n, k), name="A", dtype="float16")
    b = te.placeholder((k, m), name="B", dtype="float16")
    k = te.reduce_axis((0, k), name="k")

    def f_compute(i, j):
        v_a = tir.Cast(dtype="float32", value=a[i, k])
        v_b = tir.Cast(dtype="float32", value=b[k, j])
        return te.sum(v_a * v_b, axis=[k])

    c = te.compute((n, m), f_compute, name="C")
    d = topi.nn.relu(c)  # pylint: disable=invalid-name
    return (a, b, d)


def conv2d_nchw(  # pylint: disable=invalid-name
    n: int,
    h: int,
    w: int,
    ci: int,
    co: int,
    kh: int,
    kw: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    x = te.placeholder((n, ci, h, w), name="X")
    w = te.placeholder((co, ci, kh, kw), name="W")
    y = topi.nn.conv2d_nchw(Input=x, Filter=w, stride=stride, padding=padding, dilation=dilation)
    return (x, w, y)


def conv2d_nchw_bias_bn_relu(  # pylint: disable=invalid-name
    n: int,
    h: int,
    w: int,
    ci: int,
    co: int,
    kh: int,
    kw: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor, te.Tensor, te.Tensor, te.Tensor]:
    oh = (h + 2 * padding - (kh - 1) * dilation - 1) // stride + 1  # pylint: disable=invalid-name
    ow = (w + 2 * padding - (kw - 1) * dilation - 1) // stride + 1  # pylint: disable=invalid-name
    x = te.placeholder((n, ci, h, w), name="X")
    w = te.placeholder((co, ci, kh, kw), name="W")
    b = te.placeholder((co, 1, 1), name="B")
    bn_scale = te.placeholder((co, 1, 1), name="bn_scale")
    bn_offset = te.placeholder((co, 1, 1), name="bn_offset")
    y = topi.nn.conv2d_nchw(Input=x, Filter=w, stride=stride, padding=padding, dilation=dilation)
    y = te.compute((n, co, oh, ow), lambda i, j, k, l: y[i, j, k, l] + b[j, 0, 0], name="bias_add")
    y = te.compute(
        (n, co, oh, ow), lambda i, j, k, l: y[i, j, k, l] * bn_scale[j, 0, 0], name="bn_mul"
    )
    y = te.compute(
        (n, co, oh, ow), lambda i, j, k, l: y[i, j, k, l] + bn_offset[j, 0, 0], name="bn_add"
    )
    y = topi.nn.relu(y)
    return (x, w, b, bn_scale, bn_offset, y)


def max_pool2d_nchw(  # pylint: disable=invalid-name
    n: int,
    h: int,
    w: int,
    ci: int,
    padding: int,
) -> Tuple[te.Tensor, te.Tensor]:  # pylint: disable=invalid-name
    x = te.placeholder((n, ci, h, w), name="X")
    y = topi.nn.pool2d(x, [2, 2], [1, 1], [1, 1], [padding, padding, padding, padding], "max")
    return (x, y)


def softmax_mn(m, n) -> Tuple[te.Tensor, te.Tensor]:  # pylint: disable=invalid-name
    a = te.placeholder((m, n), name="A")
    b = topi.nn.softmax(a, axis=1)

    return (a, b)


def conv2d_nhwc_f16(  # pylint: disable=invalid-name,missing-docstring
    N: int,
    H: int,
    W: int,
    CI: int,
    CO: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
):
    inputs = te.placeholder((N, H, W, CI), name="inputs", dtype="float16")
    weight = te.placeholder(
        (kernel_size, kernel_size, CI // groups, CO), name="weight", dtype="float16"
    )
    batch_size, in_h, in_w, _ = inputs.shape
    k_h, k_w, channel_per_group, out_channel = weight.shape
    out_channel_per_group = out_channel // groups

    out_h = (in_h + 2 * padding - dilation * (k_h - 1) - 1) // stride + 1
    out_w = (in_w + 2 * padding - dilation * (k_w - 1) - 1) // stride + 1
    rh = te.reduce_axis((0, k_h), name="rh")
    rw = te.reduce_axis((0, k_w), name="rw")
    rc = te.reduce_axis((0, channel_per_group), name="rc")

    padded = topi.nn.pad(inputs, [0, padding, padding, 0])
    output = te.compute(
        (batch_size, out_h, out_w, out_channel),
        lambda n, h, w, co: te.sum(
            (
                tir.Cast(
                    value=padded[
                        n,
                        h * stride + rh * dilation,
                        w * stride + rw * dilation,
                        co // out_channel_per_group * channel_per_group + rc,
                    ],
                    dtype="float32",
                )
                * tir.Cast(value=weight[rh, rw, rc, co], dtype="float32")
            ),
            axis=[rh, rw, rc],
        ),
        name="conv2d_nhwc",
    )
    return (inputs, weight, output)


def batch_matmul_nkkm_f16(  # pylint: disable=invalid-name,missing-docstring
    B: int,
    N: int,
    M: int,
    K: int,
) -> Tuple[te.Tensor, te.Tensor, te.Tensor]:
    x = te.placeholder((B, N, K), name="X", dtype="float16")
    y = te.placeholder((B, K, M), name="Y", dtype="float16")
    k = te.reduce_axis((0, K), name="k")
    z = te.compute(  # pylint: disable=invalid-name
        (B, N, M),
        lambda b, i, j: te.sum(
            tir.Cast("float32", x[b][i][k]) * tir.Cast("float32", y[b][k][j]), axis=[k]
        ),
        name="Z",
    )
    return (x, y, z)


def create_te_workload(name: str, idx: int) -> tir.PrimFunc:
    workload_func, params = CONFIGS[name]
    return te.create_prim_func(workload_func(*params[idx]))  # type: ignore


CONFIGS = {
    "C1D": (
        conv1d_nlc,
        [
            # derived from conv2d_shapes
            (1, 256, 64, 128, 3, 2, 1),
            #    (1, 256, 64, 128, 1, 2, 0),
            #    (1, 256, 64, 64, 1, 1, 0),
            #    (1, 128, 128, 256, 3, 2, 1),
            (1, 128, 128, 256, 1, 2, 0),
            #    (1, 128, 128, 128, 3, 1, 1),
            #    (1, 64, 256, 512, 3, 2, 1),
            #    (1, 64, 256, 512, 1, 2, 0),
            (1, 64, 256, 256, 5, 1, 2),
            (1, 32, 512, 512, 3, 1, 1),
        ],
    ),
    "C2D": (
        conv2d_nhwc,
        [
            # all conv2d layers in resnet-18
            (1, 224, 224, 3, 64, 7, 2, 3),
            #    (1, 56, 56, 64, 128, 3, 2, 1),
            #    (1, 56, 56, 64, 128, 1, 2, 0),
            #    (1, 56, 56, 64, 64, 3, 1, 1),
            (1, 56, 56, 64, 64, 1, 1, 0),
            #    (1, 28, 28, 128, 256, 3, 2, 1),
            #    (1, 28, 28, 128, 256, 1, 2, 0),
            #    (1, 28, 28, 128, 128, 3, 1, 1),
            #    (1, 14, 14, 256, 512, 3, 2, 1),
            #    (1, 14, 14, 256, 512, 1, 2, 0),
            (1, 14, 14, 256, 256, 3, 1, 1),
            (1, 7, 7, 512, 512, 3, 1, 1),
        ],
    ),
    "C3D": (
        conv3d_ndhwc,
        [
            # Derived from conv2d_shapes. Use depth=16 for all configurations
            (1, 16, 224, 224, 3, 64, 7, 2, 3),
            #    (1, 16, 56, 56, 64, 128, 3, 2, 1),
            #    (1, 16, 56, 56, 64, 128, 1, 2, 0),
            #    (1, 16, 56, 56, 64, 64, 3, 1, 1),
            (1, 16, 56, 56, 64, 64, 1, 1, 0),
            #    (1, 16, 28, 28, 128, 256, 3, 2, 1),
            #    (1, 16, 28, 28, 128, 256, 1, 2, 0),
            #    (1, 16, 28, 28, 128, 128, 3, 1, 1),
            #    (1, 16, 14, 14, 256, 512, 3, 2, 1),
            #    (1, 16, 14, 14, 256, 512, 1, 2, 0),
            (1, 16, 14, 14, 256, 256, 3, 1, 1),
            (1, 16, 7, 7, 512, 512, 3, 1, 1),
        ],
    ),
    "GMM": (
        batch_matmul_nkkm,
        [
            (1, 128, 128, 128),
            (1, 512, 32, 512),
            (1, 512, 512, 512),
            (1, 1024, 1024, 1024),
        ],
    ),
    "GRP": (
        conv2d_nhwc,
        [
            # Derived from conv2d_shapes. Use group=4 for all configurations
            (1, 56, 56, 64, 128, 3, 2, 1, 1, 4),
            #    (1, 56, 56, 64, 128, 1, 2, 0 , 1, 4),
            #    (1, 56, 56, 64, 64, 3, 1, 1  , 1, 4),
            (1, 56, 56, 64, 64, 1, 1, 0, 1, 4),
            #    (1, 28, 28, 128, 256, 3, 2, 1, 1, 4),
            #    (1, 28, 28, 128, 256, 1, 2, 0, 1, 4),
            #    (1, 28, 28, 128, 128, 3, 1, 1, 1, 4),
            #    (1, 14, 14, 256, 512, 3, 2, 1, 1, 4),
            #    (1, 14, 14, 256, 512, 1, 2, 0, 1, 4),
            (1, 14, 14, 256, 256, 3, 1, 1, 1, 4),
            (1, 7, 7, 512, 512, 3, 1, 1, 1, 4),
        ],
    ),
    "DIL": (
        conv2d_nhwc,
        [
            # Derived from conv2d_shapes. Use dilation=2 for all configurations
            (1, 224, 224, 3, 64, 7, 2, 3, 2),
            #    (1, 56, 56, 64, 128, 3, 2, 1 , 2),
            #    (1, 56, 56, 64, 128, 1, 2, 0 , 2),
            #    (1, 56, 56, 64, 64, 3, 1, 1  , 2),
            (1, 56, 56, 64, 64, 1, 1, 0, 2),
            #    (1, 28, 28, 128, 256, 3, 2, 1, 2),
            #    (1, 28, 28, 128, 256, 1, 2, 0, 2),
            #    (1, 28, 28, 128, 128, 3, 1, 1, 2),
            #    (1, 14, 14, 256, 512, 3, 2, 1, 2),
            #    (1, 14, 14, 256, 512, 1, 2, 0, 2),
            (1, 14, 14, 256, 256, 3, 1, 1, 2),
            (1, 7, 7, 512, 512, 3, 1, 1, 2),
        ],
    ),
    "DEP": (
        depthwise_conv2d_nhwc,
        [
            # all depthwise conv2d layers in mobilenet
            (1, 112, 112, 32, 3, 1, 1),
            (1, 112, 112, 64, 3, 2, 1),
            #    (1,  56,  56, 128, 3, 1, 1),
            #    (1,  56,  56, 128, 3, 2, 1),
            #    (1,  28,  28, 256, 3, 1, 1),
            #    (1,  28,  28, 256, 3, 2, 1),
            #    (1,  14,  14, 512, 3, 1, 1),
            (1, 14, 14, 512, 3, 2, 1),
            (1, 7, 7, 1024, 3, 1, 1),
        ],
    ),
    "T2D": (
        conv2d_transpose_nhwc,
        [
            # all conv2d tranpose layers in DCGAN
            (1, 4, 4, 512, 256, 4, 2, 1),
            (1, 8, 8, 256, 128, 4, 2, 1),
            (1, 16, 16, 128, 64, 4, 2, 1),
            (1, 32, 32, 64, 3, 4, 2, 1),
        ],
    ),
    "CAP": (
        conv2d_capsule_nhwijc,
        [
            # all conv2d capsule layers in matrix capsules withemrouting (ICLR 2018)
            (1, 16, 16, 32, 32, 3, 2, 1),
            (1, 8, 8, 32, 32, 3, 1, 1),
            (1, 16, 16, 8, 16, 3, 2, 1),
            (1, 8, 8, 16, 16, 3, 1, 1),
        ],
    ),
    "NRM": (
        norm_bmn,
        [
            (1, 256, 256),
            (1, 512, 512),
            (1, 1024, 1024),
            (1, 4096, 1024),
        ],
    ),
    "SFM": (
        softmax_mn,
        [
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
        ],
    ),
    "C2d-BN-RELU": (
        conv2d_nhwc_bn_relu,
        [
            (1, 224, 224, 3, 64, 7, 2, 3),
            (1, 56, 56, 64, 128, 3, 2, 1),
            (1, 28, 28, 128, 256, 1, 2, 0),
            (1, 7, 7, 512, 512, 3, 1, 1),
        ],
    ),
    "TBG": (
        transpose_batch_matmul,
        [
            (1, 128, 12, 64),
            (1, 128, 16, 64),
            (1, 64, 12, 128),
            (1, 128, 12, 128),
        ],
    ),
}
