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
from tvm import te
import numpy


def ceildiv(o, d):
    assert o >= 0
    assert d >= 0
    return tvm.tir.floordiv(o + d - 1, d)


# defines inner block shape: 8h8w32c
def get_block_shape():
    return 8, 8, 32


# defines inner filter block shape: 8i32o41
def get_filter_block_shape():
    return 8, 32, 4


# input: locgical shape in nhwc layout
# output:  physical packed shape in nhw8h8w32c layout
def get_packed_shape(logical_shape_nhwc):
    assert len(logical_shape_nhwc) == 4
    physical_shape_nhwc8h8w32c = [logical_shape_nhwc[0]]
    block_shape = get_block_shape()
    off_h, off_w, off_c = block_shape
    physical_shape_nhwc8h8w32c.append(ceildiv(logical_shape_nhwc[1], off_h))
    physical_shape_nhwc8h8w32c.append(ceildiv(logical_shape_nhwc[2], off_w))
    physical_shape_nhwc8h8w32c.append(ceildiv(logical_shape_nhwc[3], off_c))
    physical_shape_nhwc8h8w32c.extend(block_shape)
    return physical_shape_nhwc8h8w32c


# input: physical packed shape in nhw8h8w32c layout
# output: logical shape in nhwc layout
def get_logical_shape(physical_shape_nhwc8h8w32c):
    assert len(physical_shape_nhwc8h8w32c) == 7
    logical_shape_nhwc = [physical_shape_nhwc8h8w32c[0]]
    logical_shape_nhwc.append(physical_shape_nhwc8h8w32c[1] * physical_shape_nhwc8h8w32c[4])
    logical_shape_nhwc.append(physical_shape_nhwc8h8w32c[2] * physical_shape_nhwc8h8w32c[5])
    logical_shape_nhwc.append(physical_shape_nhwc8h8w32c[3] * physical_shape_nhwc8h8w32c[6])
    return logical_shape_nhwc


# input: logical shape in oihw layout
# output: physical packed shape in oihw8i3204i layout
def get_packed_filter_shape(logical_shape_oihw):
    assert len(logical_shape_oihw) == 4
    filter_block_shape = get_filter_block_shape()
    filter_Cio, filter_Ki, filter_Cii = filter_block_shape
    filter_Ci = filter_Cio * filter_Cii
    physical_shape_oihw8i32o4i = []
    physical_shape_oihw8i32o4i.append(int(ceildiv(logical_shape_oihw[0], filter_Ki)))
    physical_shape_oihw8i32o4i.append(int(ceildiv(logical_shape_oihw[1], filter_Ci)))
    physical_shape_oihw8i32o4i.append(logical_shape_oihw[2])
    physical_shape_oihw8i32o4i.append(logical_shape_oihw[3])
    physical_shape_oihw8i32o4i.extend(filter_block_shape)
    return physical_shape_oihw8i32o4i


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


def conv2d_verify(output, ref_output, dtype):
    # nhwc8h8w32c -> nhwc
    logical_output_shape = get_logical_shape(output.shape)
    output = output.transpose(0, 1, 4, 2, 5, 3, 6).reshape(logical_output_shape)

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


def conv2d_compute(X, filt, pad, stride, dilation):
    block_shape = get_block_shape()
    block_H, block_W, block_C = block_shape
    filter_Cio, filter_Ki, filter_Cii = get_filter_block_shape()
    filter_Ci = filter_Cio * filter_Cii

    shape_filter = filt.shape
    kernel_size = tuple(shape_filter[2:4])
    out_channels = shape_filter[0] * shape_filter[5]

    logical_input_shape = get_logical_shape(X.shape)
    logical_output_shape = get_conv2d_nhwc_shape(
        logical_input_shape,
        kernel_size,
        stride,
        pad,
        dilation,
        out_channels,
    )

    output_shape = get_packed_shape(logical_output_shape)
    n, ho, wo, ko, hi, wi, ki = output_shape
    rh = te.reduce_axis((0, kernel_size[0]), name="rh")
    rw = te.reduce_axis((0, kernel_size[1]), name="rw")
    rc = te.reduce_axis((0, logical_input_shape[3]), name="rc")

    def compute(n, ho, wo, ko, hi, wi, ki):
        h = ho * block_H + hi
        h_contig = h * stride[0] + rh
        h_block_id = h_contig // block_H
        h_block_offset = h_contig % block_H

        w = wo * block_W + wi
        w_contig = w * stride[1] + rw
        w_block_id = w_contig // block_W
        w_block_offset = w_contig % block_W

        c_block_id = rc // block_C
        c_block_offset = rc % block_C

        rco = rc // filter_Ci
        rcio = (rc % filter_Ci) // filter_Cii
        rcii = rc % filter_Cii

        return te.sum(
            X[
                n,
                h_block_id,
                w_block_id,
                c_block_id,
                h_block_offset,
                w_block_offset,
                c_block_offset,
            ]
            * filt[ko, rco, rh, rw, rcio, ki, rcii],
            axis=[rh, rw, rc],
        )

    return output_shape, compute
