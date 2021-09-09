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

import sys

import tvm
from tvm import te
from tvm import topi
from tvm.topi import testing
from .infrastructure import (
    ceildiv,
    build_and_run,
    get_block_shape,
    get_conv2d_nhwc_shape,
    get_packed_filter_layout,
    get_packed_activation_layout,
)

import numpy as np
import pytest


def conv2d_logical(
    shape_nhwc,
    shape_oihw,
    kernel_size,
    stride,
    padding,
    dtype,
    storage_scope="global",
):
    """
    Conv2d TE wherein both input activation and filter tensors
    are defined with their logical NHWC/OIHW shapes, respectively.
    The packed physical layout for the activation and filter are:
      Activation: nhwc8h8w32c
      Filter: oihw8i32o4i
    """
    assert kernel_size == tuple(shape_oihw[2:])

    block_shape = get_block_shape()
    block_H, block_W, block_C = block_shape
    shape = get_packed_activation_layout(shape_nhwc, block_shape)
    logical_output_shape = get_conv2d_nhwc_shape(
        shape_nhwc, kernel_size, stride, padding, [1, 1], shape_oihw[0]
    )
    output_shape = get_packed_activation_layout(logical_output_shape, block_shape)

    N, H, W, C = shape_nhwc
    X = te.placeholder(shape_nhwc, dtype=dtype)
    # Combination of padding required by conv2d operator and padding to evenly divisible
    # number of blocks. Note that this padding should be inlined in the schedule so
    # as to avoid input copying.
    pad_h = (block_H - ((H + padding[1]) % block_H)) % block_H
    pad_w = (block_W - ((W + padding[3]) % block_W)) % block_W
    X_pad = topi.nn.pad(X, [0, padding[0], padding[2], 0], [0, pad_h, pad_w, 0], pad_value=0)
    # Calculate packed layout
    X_packed = te.compute(
        shape,
        lambda n, ho, wo, co, hi, wi, ci: X_pad[
            n, ho * block_H + hi, wo * block_W + wi, co * block_C + ci
        ],
    )

    # Filter shape using KCRS (OIHW) notation
    K, C, R, S = shape_oihw
    filter_Ki, filter_Ci, filter_Cii = 32, 32, 4
    shape_filter = get_packed_filter_layout(K, C, R, S)
    filt = te.placeholder(shape_oihw, dtype=dtype)
    # Channel padding to multiples of 32
    pad_c = (filter_Ci - (C % filter_Ci)) % filter_Ci
    pad_k = (filter_Ki - (K % filter_Ki)) % filter_Ki
    filt_pad = topi.nn.pad(
        filt, [0, 0, 0, 0], [pad_k, pad_c, R, S], pad_value=0, name="padded_filter"
    )
    filt_packed = te.compute(
        shape_filter,
        lambda ko, co, r, s, cio, ki, cii: filt_pad[
            ko * filter_Ki + ki, co * filter_Ci + cio * filter_Cii + cii, r, s
        ],
        name="packed_filter",
    )

    rh = te.reduce_axis((0, kernel_size[0]), name="rh")
    rw = te.reduce_axis((0, kernel_size[1]), name="rw")
    rc = te.reduce_axis((0, C), name="rc")

    def compute(n, ho, wo, ko, hi, wi, ki):
        # Construct blockized strided conv2d height index
        h = ho * block_H + hi
        h_contig = h * stride[0] + rh
        h_block_id = h_contig // block_H
        h_block_offset = h_contig % block_H

        # Construct blockized strided conv2d width index
        w = wo * block_W + wi
        w_contig = w * stride[1] + rw
        w_block_id = w_contig // block_W
        w_block_offset = w_contig % block_W

        # Construct blockized conv2d channel index
        c_block_id = rc // block_C
        c_block_offset = rc % block_C

        # Construct flat filter input channel indices
        rco = rc // filter_Ci
        rcio = (rc % filter_Ci) // filter_Cii
        rcii = rc % filter_Cii

        return te.sum(
            X_packed[
                n,
                h_block_id,
                w_block_id,
                c_block_id,
                h_block_offset,
                w_block_offset,
                c_block_offset,
            ]
            * filt_packed[ko, rco, rh, rw, rcio, ki, rcii],
            axis=[rh, rw, rc],
        )

    Y = te.compute(output_shape, compute)
    s = te.create_schedule(Y.op)

    # Ensure the padding and array packing is performed inline
    s[X_pad].compute_inline()
    s[X_packed].compute_inline()

    s[filt_pad].compute_inline()
    s[filt_packed].compute_inline()

    binds = {}
    if storage_scope and storage_scope != "global":
        with tvm.transform.PassContext():
            Xb = tvm.tir.decl_buffer(shape, name="Xb", dtype=dtype, scope=storage_scope)
            Yb = tvm.tir.decl_buffer(output_shape, name="Yb", dtype=dtype, scope=storage_scope)
            binds = {X: Xb, Y: Yb}

    return (s, [X, filt, Y], binds)


def conv2d_packed_filter(
    shape_nhwc,
    shape_oihw8i32o4i,
    kernel_size,
    stride,
    padding,
    dtype,
    storage_scope="global",
):
    """
    Conv2d TE wherein the input activation is defined by its
    logical NHWC shape, but the filter is provided in the
    packed layout oihw8i32o4i. The physical packed layout used
    for the activation is: nhwc8h8w32c
    """
    assert kernel_size == tuple(shape_oihw8i32o4i[2:4])

    block_shape = get_block_shape()
    block_H, block_W, block_C = block_shape
    shape = get_packed_activation_layout(shape_nhwc, block_shape)
    logical_output_shape = get_conv2d_nhwc_shape(
        shape_nhwc,
        kernel_size,
        stride,
        padding,
        [1, 1],
        shape_oihw8i32o4i[0] * shape_oihw8i32o4i[5],
    )

    output_shape = get_packed_activation_layout(logical_output_shape, block_shape)

    N, H, W, C = shape_nhwc
    X = te.placeholder(shape_nhwc, dtype=dtype)
    # Combination of padding required by conv2d operator and padding to evenly divisible
    # number of blocks. Note that this padding should be inlined in the schedule so
    # as to avoid input copying.
    pad_h = (block_H - ((H + padding[1]) % block_H)) % block_H
    pad_w = (block_W - ((W + padding[3]) % block_W)) % block_W

    X_pad = topi.nn.pad(X, [0, padding[0], padding[2], 0], [0, pad_h, pad_w, 0], pad_value=0)
    # Calculate packed layout
    packed_shape = get_packed_activation_layout(X_pad.shape, block_shape)

    X_packed = te.compute(
        packed_shape,
        lambda n, ho, wo, co, hi, wi, ci: X_pad[
            n, ho * block_H + hi, wo * block_W + wi, co * block_C + ci
        ],
    )

    # Filter shape using KCRS (OIHW) notation
    filter_Ki, filter_Ci, filter_Cii = 32, 32, 4
    assert shape_oihw8i32o4i[-1] == filter_Cii
    assert shape_oihw8i32o4i[-2] == filter_Ki
    assert shape_oihw8i32o4i[-3] == filter_Ci // filter_Cii

    filt_packed = te.placeholder(shape_oihw8i32o4i, dtype=dtype)

    rh = te.reduce_axis((0, kernel_size[0]), name="rh")
    rw = te.reduce_axis((0, kernel_size[1]), name="rw")
    rc = te.reduce_axis((0, C), name="rc")

    def compute(n, ho, wo, ko, hi, wi, ki):
        # Construct blockized strided conv2d height index
        h = ho * block_H + hi
        h_contig = h * stride[0] + rh
        h_block_id = h_contig // block_H
        h_block_offset = h_contig % block_H

        # Construct blockized strided conv2d width index
        w = wo * block_W + wi
        w_contig = w * stride[1] + rw
        w_block_id = w_contig // block_W
        w_block_offset = w_contig % block_W

        # Construct blockized conv2d channel index
        c_block_id = rc // block_C
        c_block_offset = rc % block_C

        # Construct flat filter input channel indices
        rco = rc // filter_Ci
        rcio = (rc % filter_Ci) // filter_Cii
        rcii = rc % filter_Cii

        return te.sum(
            X_packed[
                n,
                h_block_id,
                w_block_id,
                c_block_id,
                h_block_offset,
                w_block_offset,
                c_block_offset,
            ]
            * filt_packed[ko, rco, rh, rw, rcio, ki, rcii],
            axis=[rh, rw, rc],
        )

    Y = te.compute(output_shape, compute)
    s = te.create_schedule(Y.op)

    # Ensure the padding and array packing is performed inline
    s[X_pad].compute_inline()
    s[X_packed].compute_inline()

    # Perform scheduling
    n, hid, wid, cid, hoff, woff, coff = s[Y].op.axis
    slice = s[Y].fuse(wid, cid)
    Xl = s.cache_read(X_packed, storage_scope, [Y])
    Yl = s.cache_write(Y, storage_scope)

    s[Yl].compute_at(s[Y], hid)
    n, hid, slice, hoff, woff, coff = s[Yl].op.axis
    s[Xl].compute_at(s[Yl], slice)

    binds = {}
    if storage_scope and storage_scope != "global":
        with tvm.transform.PassContext():
            Xb = tvm.tir.decl_buffer(shape, name="Xb", dtype=dtype, scope=storage_scope)
            Yb = tvm.tir.decl_buffer(output_shape, name="Yb", dtype=dtype, scope=storage_scope)
            binds = {X: Xb, Y: Yb}

    return (s, [X, filt_packed, Y], binds)


def conv2d_packed_filter_nhwhwc(
    shape_nhwc,
    shape_oihw8i32o4i,
    kernel_size,
    stride,
    padding,
    dtype,
    storage_scope="global",
):
    """
    Conv2d TE wherein the input activation is defined by its
    logical NHWC shape, but the filter is provided in the
    packed layout oihw8i32o4i. The physical packed layout used
    for the activation is: nhw8h8wc

    """
    assert kernel_size == tuple(shape_oihw8i32o4i[2:4])

    block_shape = get_block_shape()
    block_H, block_W, _ = block_shape
    shape = get_packed_activation_layout(shape_nhwc, block_shape, packed_C=False)
    logical_output_shape = get_conv2d_nhwc_shape(
        shape_nhwc,
        kernel_size,
        stride,
        padding,
        [1, 1],
        shape_oihw8i32o4i[0] * shape_oihw8i32o4i[5],
    )
    output_shape = get_packed_activation_layout(logical_output_shape, block_shape, packed_C=False)

    N, H, W, C = shape_nhwc
    X = te.placeholder(shape_nhwc, dtype=dtype)
    # Combination of padding required by conv2d operator and padding to evenly divisible
    # number of blocks. Note that this padding should be inlined in the schedule so
    # as to avoid input copying.
    pad_h = (block_H - ((H + padding[1]) % block_H)) % block_H
    pad_w = (block_W - ((W + padding[3]) % block_W)) % block_W
    X_pad = topi.nn.pad(X, [0, padding[0], padding[2], 0], [0, pad_h, pad_w, 0], pad_value=0)
    # Calculate packed layout
    packed_shape = get_packed_activation_layout(X_pad.shape, block_shape, packed_C=False)
    X_packed = te.compute(
        packed_shape, lambda n, ho, wo, hi, wi, c: X_pad[n, ho * block_H + hi, wo * block_W + wi, c]
    )

    # Filter shape using KCRS (OIHW) notation
    filter_Ki, filter_Ci, filter_Cii = 32, 32, 4
    assert shape_oihw8i32o4i[-1] == filter_Cii
    assert shape_oihw8i32o4i[-2] == filter_Ki
    assert shape_oihw8i32o4i[-3] == filter_Ci // filter_Cii

    filt_packed = te.placeholder(shape_oihw8i32o4i, dtype=dtype)

    rh = te.reduce_axis((0, kernel_size[0]), name="rh")
    rw = te.reduce_axis((0, kernel_size[1]), name="rw")
    rc = te.reduce_axis((0, C), name="rc")

    def compute(n, ho, wo, hi, wi, k):
        # Construct blockized strided conv2d height index
        h = ho * block_H + hi
        h_contig = h * stride[0] + rh
        h_block_id = h_contig // block_H
        h_block_offset = h_contig % block_H

        # Construct blockized strided conv2d width index
        w = wo * block_W + wi
        w_contig = w * stride[1] + rw
        w_block_id = w_contig // block_W
        w_block_offset = w_contig % block_W

        # Construct flat filter input channel indices
        rco = rc // filter_Ci
        rcio = (rc % filter_Ci) // filter_Cii
        rcii = rc % filter_Cii

        # Construct split filter output channel index
        ko = k // filter_Ki
        ki = k % filter_Ki

        return te.sum(
            X_packed[n, h_block_id, w_block_id, h_block_offset, w_block_offset, rc]
            * filt_packed[ko, rco, rh, rw, rcio, ki, rcii],
            axis=[rh, rw, rc],
        )

    Y = te.compute(output_shape, compute)
    s = te.create_schedule(Y.op)

    # Ensure the padding and array packing is performed inline
    s[X_pad].compute_inline()
    s[X_packed].compute_inline()

    n, ho, wo, hi, wi, k = s[Y].op.axis
    rh, rw, rc = s[Y].op.reduce_axis

    rco, rci = s[Y].split(rc, factor=32)
    s[Y].reorder(n, rco, wo, ho, k, hi, wi)
    Xl = s.cache_read(X_packed, storage_scope, [Y])
    s[Xl].compute_at(s[Y], rco)

    ko, ki = s[Y].split(k, factor=32)
    s[Y].reorder(n, rco, wo, ho, ko, hi, wi, ki)
    Fl = s.cache_read(filt_packed, storage_scope, [Y])
    s[Fl].compute_at(s[Y], ko)

    binds = {}
    if storage_scope and storage_scope != "global":
        with tvm.transform.PassContext():
            Xb = tvm.tir.decl_buffer(shape, name="Xb", dtype=dtype, scope=storage_scope)
            Yb = tvm.tir.decl_buffer(output_shape, name="Yb", dtype=dtype, scope=storage_scope)
            binds = {X: Xb, Y: Yb}

    return (s, [X, filt_packed, Y], binds)


class BaseConv2d:
    batch = tvm.testing.parameter(1)
    in_size = tvm.testing.parameter(8, 56)
    in_channel = tvm.testing.parameter(64)
    out_channel = tvm.testing.parameter(64)
    kernel = tvm.testing.parameter(3)
    stride = tvm.testing.parameter(1)
    pad = tvm.testing.parameter(1)
    dtype = tvm.testing.parameter("float32")


class TestConv2dLogical(BaseConv2d):
    @tvm.testing.parametrize_targets("llvm")
    def test_conv2d(self, shape_nhwc, shape_oihw, kernel, stride, pad, dtype, target):
        inputs = [
            np.random.uniform(0, 255, size=shape_nhwc).astype(dtype),
            np.random.uniform(0, 255, size=shape_oihw).astype(dtype),
        ]
        np_filter = inputs[1].transpose(2, 3, 1, 0)
        ref_output = testing.conv2d_nhwc_python(inputs[0], np_filter, stride, pad)
        output = build_and_run(
            inputs,
            conv2d_logical,
            target,
            target,
            shape_nhwc=shape_nhwc,
            shape_oihw=shape_oihw,
            kernel_size=(kernel, kernel),
            stride=(stride, stride),
            padding=(pad, pad, pad, pad),
            dtype=dtype,
        )
        return output, ref_output


class TestConv2dPackedFilter(BaseConv2d):
    conv2d_impl = tvm.testing.parameter(conv2d_packed_filter, conv2d_packed_filter_nhwhwc)

    @tvm.testing.parametrize_targets("llvm")
    def test_conv2d(
        self,
        conv2d_impl,
        shape_nhwc,
        shape_oihw,
        shape_oihw8i32o4i,
        kernel,
        stride,
        pad,
        dtype,
        target,
    ):
        inputs = [
            np.random.uniform(0, 255, size=shape_nhwc).astype(dtype),
            np.random.uniform(0, 255, size=shape_oihw8i32o4i).astype(dtype),
        ]
        np_filter = (
            inputs[1].transpose(0, 5, 1, 4, 6, 2, 3).reshape(shape_oihw).transpose(2, 3, 1, 0)
        )
        ref_output = testing.conv2d_nhwc_python(inputs[0], np_filter, stride, pad)
        output = build_and_run(
            inputs,
            conv2d_impl,
            target,
            target,
            shape_nhwc=shape_nhwc,
            shape_oihw8i32o4i=shape_oihw8i32o4i,
            kernel_size=(kernel, kernel),
            stride=(stride, stride),
            padding=(pad, pad, pad, pad),
            dtype=dtype,
        )
        return output, ref_output


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
