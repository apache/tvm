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

import platform
import tvm
from tvm import te
from tvm import topi
from tvm.topi import testing

from .infrastructure import (
    build_and_run,
    conv2d_compute,
    conv2d_verify,
    get_block_shape,
    get_packed_filter_shape,
    get_packed_shape,
)

import numpy as np
import pytest


def conv2dconv2d(
    shape_input,
    pad1,
    stride1,
    dilation1,
    shape_filter1,
    pad2,
    stride2,
    dilation2,
    shape_filter2,
    k_split_factor,
    h_split_factor,
    dtype,
    storage_scope="global",
):
    """
    Conv2d -> Conv2d wherein the input activation is defined by its
    logical NHWC layout.  The filter is provided in its physical
    packed layout (oihw8i32o4i).  The input is padded and then packed
    into its physical packed layout (nhwc8h8w32c).  The resulting
    computation is in the same physical packed layout (nhwc8h8w32c).
    """

    # nhwc layout
    X = te.placeholder(shape_input, dtype=dtype, name="logical_input")

    # oihw8i32o4i layout
    filt_packed1 = te.placeholder(shape_filter1, dtype=dtype, name="packed_filter1")
    filt_packed2 = te.placeholder(shape_filter2, dtype=dtype, name="packed_filter2")

    block_H, block_W, block_C = get_block_shape()

    # Calculate padded input
    N, H, W, C = shape_input
    pad_h = (block_H - ((H + pad1[1]) % block_H)) % block_H
    pad_w = (block_W - ((W + pad1[3]) % block_W)) % block_W
    X_pad = topi.nn.pad(
        X, [0, pad1[0], pad1[2], 0], [0, pad_h, pad_w, 0], pad_value=0, name="padded_input"
    )

    # Calculate packed input
    packed_shape = get_packed_shape(X_pad.shape)
    X_packed = te.compute(
        packed_shape,
        lambda n, ho, wo, co, hi, wi, ci: X_pad[
            n, ho * block_H + hi, wo * block_W + wi, co * block_C + ci
        ],
        name="packed_input",
    )

    output_shape1, compute1 = conv2d_compute(X_packed, filt_packed1, pad1, stride1, dilation1)
    temp_Y = te.compute(output_shape1, compute1, name="temp_output")

    output_shape2, compute2 = conv2d_compute(temp_Y, filt_packed2, pad2, stride2, dilation2)
    Y = te.compute(output_shape2, compute2, name="packed_output")
    s = te.create_schedule(Y.op)

    # Ensure the padding and array packing is performed inline
    s[X_pad].compute_inline()
    s[X_packed].compute_inline()

    # cache reads and writes
    Xl = s.cache_read(X_packed, storage_scope, [temp_Y])
    F1l = s.cache_read(filt_packed1, storage_scope, [temp_Y])
    F2l = s.cache_read(filt_packed2, storage_scope, [Y])
    Yl = s.cache_write(Y, storage_scope)

    # conv2d #1 schedule
    n, ho, wo, ko, hi, wi, ki = s[temp_Y].op.axis
    rh, rw, rc = s[temp_Y].op.reduce_axis
    rco, rci = s[temp_Y].split(rc, factor=block_C)
    koo, koi = s[temp_Y].split(ko, factor=k_split_factor)
    hoo, hoi = s[temp_Y].split(ho, factor=h_split_factor)
    s[temp_Y].reorder(n, koo, hoo, koi, hoi, wo, rco, hi, wi, ki, rci)
    s[Xl].compute_at(s[temp_Y], hoo)
    s[F1l].compute_at(s[temp_Y], hoo)

    # cache write schedule
    n, ho, wo, ko, hi, wi, ki = s[Y].op.axis
    koo, koi = s[Y].split(ko, factor=k_split_factor)
    hoo, hoi = s[Y].split(ho, factor=h_split_factor)
    s[Y].reorder(n, koo, hoo, koi, hoi, wo, hi, wi, ki)
    s[Yl].compute_at(s[Y], hoo)

    # conv2d #2 schedule
    n, ho, wo, ko, hi, wi, ki = s[Yl].op.axis
    rh, rw, rc = s[Yl].op.reduce_axis
    rco, rci = s[Yl].split(rc, factor=block_C)
    koo, koi = s[Yl].split(ko, factor=k_split_factor)
    hoo, hoi = s[Yl].split(ho, factor=h_split_factor)
    s[Yl].reorder(n, koo, hoo, koi, hoi, wo, rco, hi, wi, ki, rci)
    s[temp_Y].compute_at(s[Yl], hoo)
    s[F2l].compute_at(s[Yl], hoo)

    binds = {}
    if storage_scope and storage_scope != "global":
        with tvm.transform.PassContext():
            Xb = tvm.tir.decl_buffer(packed_shape, name="Xb", dtype=dtype, scope=storage_scope)
            Yb = tvm.tir.decl_buffer(output_shape2, name="Yb", dtype=dtype, scope=storage_scope)
            binds = {X: Xb, Y: Yb}

    return (s, [X, filt_packed1, filt_packed2, Y], binds)


class BaseConv2dConv2d:
    # input
    batch = tvm.testing.parameter(1)
    in_size = tvm.testing.parameter(64)
    in_channel = tvm.testing.parameter(128)
    # conv2d #1
    pad1 = tvm.testing.parameter(0)
    stride1 = tvm.testing.parameter(1)
    kernel_size1 = tvm.testing.parameter(1, 3)
    out_channel1 = tvm.testing.parameter(128)
    # conv2d #2
    stride2 = tvm.testing.parameter(1)
    kernel_size2 = tvm.testing.parameter(1, 3)
    out_channel2 = tvm.testing.parameter(128)
    # schedule params
    k_split_factor = tvm.testing.parameter(1, 2)
    h_split_factor = tvm.testing.parameter(1, 2)
    dtype = tvm.testing.parameter("float32")


class TestConv2dConv2dPackedFilter(BaseConv2dConv2d):
    @tvm.testing.parametrize_targets("llvm")
    @pytest.mark.skipif(
        platform.processor() == "i386", reason="Test known to be flaky on i386 machines"
    )
    def test_conv2d(
        self,
        batch,
        in_size,
        in_channel,
        pad1,
        stride1,
        kernel_size1,
        out_channel1,
        stride2,
        kernel_size2,
        out_channel2,
        k_split_factor,
        h_split_factor,
        dtype,
        target,
    ):
        # TODO: no support for padding in conv2d #2
        pad2 = 0

        # TODO: no support for dilation
        dilation1 = 1
        dilation2 = 1

        shape_input = [batch, in_size, in_size, in_channel]
        shape_filter1_oihw = [out_channel1, in_channel, kernel_size1, kernel_size1]
        shape_filter1_oihw8i32o4i = get_packed_filter_shape(shape_filter1_oihw)

        shape_filter2_oihw = [out_channel2, out_channel1, kernel_size2, kernel_size2]
        shape_filter2_oihw8i32o4i = get_packed_filter_shape(shape_filter2_oihw)

        inputs = [
            np.random.uniform(0, 255, size=shape_input).astype(dtype),
            np.random.uniform(0, 255, size=shape_filter1_oihw8i32o4i).astype(dtype),
            np.random.uniform(0, 255, size=shape_filter2_oihw8i32o4i).astype(dtype),
        ]
        np_filter1 = (
            inputs[1]
            .transpose(0, 5, 1, 4, 6, 2, 3)
            .reshape(shape_filter1_oihw)
            .transpose(2, 3, 1, 0)
        )
        np_filter2 = (
            inputs[2]
            .transpose(0, 5, 1, 4, 6, 2, 3)
            .reshape(shape_filter2_oihw)
            .transpose(2, 3, 1, 0)
        )
        temp_output = testing.conv2d_nhwc_python(inputs[0], np_filter1, stride1, pad1)
        ref_output = testing.conv2d_nhwc_python(temp_output, np_filter2, stride2, pad2)
        output = build_and_run(
            inputs,
            conv2dconv2d,
            target,
            target,
            shape_input=shape_input,
            pad1=(pad1, pad1, pad1, pad1),
            stride1=(stride1, stride1),
            dilation1=(dilation1, dilation1),
            shape_filter1=shape_filter1_oihw8i32o4i,
            pad2=(pad2, pad2, pad1, pad1),
            stride2=(stride2, stride2),
            dilation2=(dilation2, dilation2),
            shape_filter2=shape_filter2_oihw8i32o4i,
            k_split_factor=k_split_factor,
            h_split_factor=h_split_factor,
            dtype=dtype,
        )

        conv2d_verify(output, ref_output, dtype)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
