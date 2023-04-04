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

"""Contrib tests for blocked conv2d and maxpool2d"""

import numpy as np

import tvm
import tvm.testing
from tvm import te, topi
from tvm.topi import testing

from .infrastructure import build_and_run, get_block_shape, get_packed_shape


# Blocked layout: NHWC8h8w32c :: [N, H//8, W//8, C//32, 8h, 8w, 32c]
def maxpool2d_logical(
    shape_nhwc,
    window_shape,
    stride,
    padding,
    dtype,
    storage_scope="global",
):
    """
    Maxpool2d TE wherein the input activation is defined by its
    logical NHWC shape. The packed physical layout for the
    activation is nhwc8h8w32c.
    """

    block_h, block_w, block_c = get_block_shape()
    shape = get_packed_shape(shape_nhwc)
    logical_output_shape = (
        shape_nhwc[0],
        (shape_nhwc[1] - window_shape[0] + padding[0] + padding[1]) // stride[0] + 1,
        (shape_nhwc[2] - window_shape[1] + padding[2] + padding[3]) // stride[0] + 1,
        shape_nhwc[3],
    )
    output_shape = get_packed_shape(logical_output_shape)

    _, height, width, _ = shape_nhwc
    placeholder_x = te.placeholder(shape_nhwc, dtype=dtype)

    # Combination of padding required by maxpool operator and padding to evenly divisible
    # number of blocks. Note that this padding should be inlined in the schedule so
    # as to avoid input copying.
    pad_h = (block_h - ((height + padding[1]) % block_h)) % block_h
    pad_w = (block_w - ((width + padding[3]) % block_w)) % block_w
    x_pad = topi.nn.pad(
        placeholder_x, [0, padding[0], padding[2], 0], [0, pad_h, pad_w, 0], pad_value=0
    )

    # Calculate packed layout
    x_packed = te.compute(
        shape,
        lambda n, ho, wo, co, hi, wi, ci: x_pad[
            n, ho * block_h + hi, wo * block_w + wi, co * block_c + ci
        ],
    )

    reduce_h = te.reduce_axis((0, window_shape[0]), name="rh")
    reduce_w = te.reduce_axis((0, window_shape[1]), name="rw")

    def compute(batch, h_outer, w_outer, c_outer, h_inner, w_inner, c_inner):
        # Construct blockized strided maxpool height indices
        h = h_outer * block_h + h_inner
        h_contig = h * stride[0] + reduce_h
        h_block_id = h_contig // block_h
        h_block_offset = h_contig % block_h

        # Construct blockized strided maxpool width indices
        w_idx = w_outer * block_w + w_inner
        w_contig = w_idx * stride[1] + reduce_w
        w_block_id = w_contig // block_w
        w_block_offset = w_contig % block_w

        return te.max(
            x_packed[
                batch, h_block_id, w_block_id, c_outer, h_block_offset, w_block_offset, c_inner
            ],
            axis=[reduce_h, reduce_w],
        )

    compute_y = te.compute(output_shape, compute)
    schedule = te.create_schedule(compute_y.op)

    # Ensure the padding and array packing is performed inline
    schedule[x_pad].compute_inline()
    schedule[x_packed].compute_inline()

    binds = {}
    if storage_scope and storage_scope != "global":
        with tvm.transform.PassContext():
            x_buffer = tvm.tir.decl_buffer(shape, name="Xb", dtype=dtype, scope=storage_scope)
            y_buffer = tvm.tir.decl_buffer(
                output_shape, name="Yb", dtype=dtype, scope=storage_scope
            )
            binds = {placeholder_x: x_buffer, compute_y: y_buffer}

    return (schedule, [placeholder_x, compute_y], binds)


class BaseMaxPooling:
    batch = tvm.testing.parameter(1)
    in_size = tvm.testing.parameter(8, 112)
    in_channel = tvm.testing.parameter(64)
    window_size = tvm.testing.parameter(3)
    stride = tvm.testing.parameter(2)
    pad = tvm.testing.parameter(1)
    dtype = tvm.testing.parameter("float32")


class TestMaxPooling(BaseMaxPooling):
    """Test MaxPool class"""

    @tvm.testing.parametrize_targets("llvm")
    def test_maxpool(self, shape_nhwc, window_size, stride, pad, dtype, target):
        """Test blocked maxpool"""
        inputs = [np.random.uniform(0, 255, size=shape_nhwc).astype(dtype)]
        ref_output = testing.poolnd_python(
            inputs[0],
            (window_size, window_size),
            strides=(stride, stride),
            dilation=(1, 1),
            padding_before=(pad, pad),
            padding_after=(pad, pad),
            pool_type="max",
        )
        output = build_and_run(
            inputs,
            maxpool2d_logical,
            target,
            target,
            shape_nhwc,
            window_shape=(window_size, window_size),
            stride=(stride, stride),
            padding=(pad, pad, pad, pad),
            dtype=dtype,
        )
        assert all([output is not None, ref_output is not None])


if __name__ == "__main__":
    tvm.testing.main()
