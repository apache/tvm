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

""" Hexagon testing infrastructure """

import tvm
import numpy


def ceildiv(o, d):
    return tvm.tir.floordiv(o + d - 1, d)


def get_packed_activation_layout(shape_nhwc, block_shape, packed_C=True):
    assert len(shape_nhwc) == 4
    shape = [shape_nhwc[0]]
    off_h, off_w, off_c = block_shape
    shape.append(ceildiv(shape_nhwc[1], off_h))
    shape.append(ceildiv(shape_nhwc[2], off_w))
    if packed_C:
        shape.append(ceildiv(shape_nhwc[3], off_c))
        shape.extend(block_shape)
    else:
        shape.extend([off_h, off_w, shape_nhwc[3]])
    return shape


def get_block_shape():
    return 8, 8, 32


def get_filter_block_shape():
    return 8, 32, 4


def get_packed_filter_layout(out_channel, in_channel, kernel_h, kernel_w):
    filter_Cio, filter_Ki, filter_Cii = get_filter_block_shape()
    filter_Ci = filter_Cio * filter_Cii
    return (
        int(ceildiv(out_channel, filter_Ki)),
        int(ceildiv(in_channel, filter_Ci)),
        kernel_h,
        kernel_w,
        filter_Cio,
        filter_Ki,
        filter_Cii,
    )


def build_and_run(inputs, func, target, target_host, *args, **kwargs):
    schedule, placeholders, binds = func(*args, **kwargs)

    func = tvm.build(schedule, placeholders, target=target, target_host=target_host, binds=binds)
    dev = tvm.device(target)
    tensors = []
    for tensor in inputs:
        tensors.append(tvm.nd.array(tensor, dev))
    tensors.append(
        tvm.nd.array(
            numpy.zeros([i.value for i in placeholders[-1].shape], dtype=placeholders[-1].dtype),
            dev,
        )
    )
    func(*tensors)

    return tensors[-1].asnumpy()


def get_conv2d_nhwc_shape(shape_nhwc, kernel_size, strides, padding, dilation, out_channels):
    assert len(shape_nhwc) == 4
    kernel = []
    kernel.append((kernel_size[0] - 1) * dilation[0] + 1)
    kernel.append((kernel_size[1] - 1) * dilation[1] + 1)
    return (
        shape_nhwc[0],
        (shape_nhwc[1] - kernel[0] + padding[0] + padding[1]) // strides[0] + 1,
        (shape_nhwc[2] - kernel[1] + padding[2] + padding[3]) // strides[1] + 1,
        out_channels,
    )


def verify_conv2d(output, ref_output, dtype):
    # nhwc8h8w32c
    if len(output.shape) == 7:
        # nhwc8h8w32c -> nhwc
        output = output.transpose(0, 1, 4, 2, 5, 3, 6).reshape(
            output.shape[0],
            output.shape[1] * output.shape[4],
            output.shape[2] * output.shape[5],
            output.shape[3] * output.shape[6],
        )

    # nhwhwc
    else:
        # nhwhwc -> nhwc
        output = output.transpose(0, 1, 3, 2, 4, 5).reshape(
            output.shape[0],
            output.shape[1] * output.shape[3],
            output.shape[2] * output.shape[4],
            output.shape[5],
        )

    # slice output to match ref_output shape
    # e.g. 8x8 spatial 3x3 filter = 6x6 ref output
    # but still 8x8 output given the blocked layout
    output = output[
        0 : ref_output.shape[0] : 1,
        0 : ref_output.shape[1] : 1,
        0 : ref_output.shape[2] : 1,
        0 : ref_output.shape[3] : 1,
    ]

    if "int" in dtype:
        tol = {"atol": 0, "rtol": 0}
    elif dtype == "float32":
        tol = {"rtol": 1e-4, "atol": 2e-4}
    tvm.testing.assert_allclose(output, ref_output, **tol)
