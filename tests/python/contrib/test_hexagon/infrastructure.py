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
# pylint: disable=invalid-name

""" Hexagon testing infrastructure """

import numpy
import tvm
from tvm import te


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


def get_packed_filter_shape(logical_shape_oihw):
    """return packed filter shape

    Parameters
    ----------
    logical_shape_oihw :
       logical shape in oihw layout

    Returns
    -------
    physical_shape_oihw8i32o4i :
        physical packed shape in oihw8i3204i layout
    """
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


def build_and_run(inputs, func, target: str, target_host: str, *args, **kwargs):
    """build and run the function func"""
    schedule, placeholders, binds = func(*args, **kwargs)

    func = tvm.build(
        schedule, placeholders, target=tvm.target.Target(target, host=target_host), binds=binds
    )
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
    """transpose and reshape output and compare with ref_output"""
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
    """Define conv2d compute"""
    block_shape = get_block_shape()
    block_H, block_W, block_C = block_shape
    filter_c_io, _, filter_c_ii = get_filter_block_shape()
    filter_c_i = filter_c_io * filter_c_ii

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

        rco = rc // filter_c_i
        rcio = (rc % filter_c_i) // filter_c_ii
        rcii = rc % filter_c_ii

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


def transform_numpy(arr_np, current_layout: str, new_layout: str):
    """Reshape and transpose numpy array according to the specified layout"""
    if current_layout == "nhwc":
        if new_layout == "nhwc":
            return arr_np
        if new_layout in ["nhwc-8h2w32c2w-2d", "nhwc-8h2w32c2w-1d"]:
            n, h, w, c = arr_np.shape
            return arr_np.reshape([n, h // 8, 8, w // 4, 2, 2, c // 32, 32]).transpose(
                0, 1, 3, 6, 2, 4, 7, 5
            )
        if new_layout in ["nhwc-4h2w32c2w-2d"]:
            n, h, w, c = arr_np.shape
            return arr_np.reshape([n, h // 4, 4, w // 4, 2, 2, c // 32, 32]).transpose(
                0, 1, 3, 6, 2, 4, 7, 5
            )
        if new_layout in ["n11c-1024c-2d", "n11c-1024c-1d"]:
            n, h, w, c = arr_np.shape
            assert h == 1 and w == 1, "The size of h and w must be 1"
            return arr_np.reshape([n, 1, 1, c // 1024, 1024])
        if new_layout == "nc-1024-2d":
            n, c = arr_np.shape
            return arr_np.reshape([n, c // 1024, 1024])
        if new_layout == "nhwc-1024c-2d":
            N, H, W, C = arr_np.shape
            return arr_np.reshape([N, H, W, C // 1024, 1024])
        if new_layout == "nc-2048-2d":
            N, C = arr_np.shape
            return arr_np.reshape([N, C // 2048, 2048])
        if new_layout == "nhwc-2048c-2d":
            N, H, W, C = arr_np.shape
            return arr_np.reshape([N, H, W, C // 2048, 2048])
        if new_layout == "nhwc-8h8w32c-2d":
            n, h, w, c = arr_np.shape
            return arr_np.reshape([n, h // 8, 8, w // 8, 8, c // 32, 32]).transpose(
                0, 1, 3, 5, 2, 4, 6
            )
        if new_layout == "n11c-2048c-2d":
            n, h, w, c = arr_np.shape
            assert h == 1 and w == 1, "The size of h and w must be 1"
            return arr_np.reshape([n, h, w, c // 2048, 2048])
        raise RuntimeError(f"Unexpected new_layout '{new_layout}'")

    if current_layout == "nc":
        n, c = arr_np.shape
        if new_layout in ["nc-1024c-2d"]:
            return arr_np.reshape([n, c // 1024, 1024])
        if new_layout in ["nc-512c-2d"]:
            return arr_np.reshape([n, c // 512, 512])
        if new_layout in ["nc-2048c-2d"]:
            return arr_np.reshape([n, c // 2048, 2048])
        raise RuntimeError(f"Unexpected new_layout '{new_layout}'")

    if current_layout == "nhw":
        if new_layout in ["nhw-32h16w-2d"]:
            n, h, w = arr_np.shape
            return arr_np.reshape([n, h // 32, 32, w // 16, 16]).transpose(0, 1, 3, 2, 4)

        raise RuntimeError(f"Unexpected new_layout '{new_layout}'")

    if current_layout == "ncw":
        if new_layout == "ncw":
            return arr_np
        if new_layout in ["ncw-32c64w-2d"]:
            n, c, w = arr_np.shape
            return arr_np.reshape([n, c // 32, 32, w // 64, 64]).transpose(0, 1, 3, 2, 4)

        raise RuntimeError(f"Unexpected new_layout '{new_layout}'")

    raise RuntimeError(f"Unexpected current_layout '{current_layout}'")


def quantize_np(arr_np: numpy.ndarray, dtype: str):
    """
    Returns quantized array along with scale and zero-point

    Parameters
    ----------
    arr_np: numpy.ndarray
        Input numpy array to be quantized
    dtype: str
        dtype of the quantized array: "uint8", "int8", etc

    Returns
    -------
    quant_np: numpy.ndarray
        Quantized numpy array
    scale: float
        Scale
    zero_point: int
        Value corresponding to float 0

    """
    if dtype == "uint8":
        qmax = 255
        qmin = 0
    elif dtype == "int8":
        qmax = 127
        qmin = -128
    else:
        raise RuntimeError(f"Unsupported quantized data type '{dtype}'")
    fmin = numpy.amin(arr_np)
    fmax = numpy.amax(arr_np)

    # Include floating-point zero in the range
    if fmax < 0:
        fmax = 0.0
    elif fmin > 0:
        fmin = 0.0

    scale = (fmax - fmin) / (qmax - qmin)
    zero_point = numpy.rint((fmax * qmin - fmin * qmax) / (fmax - fmin)).astype("int32")
    quant_np = (arr_np / scale + zero_point).astype(dtype)
    return quant_np, scale, zero_point


def get_hexagon_target(cpu_ver: str) -> tvm.target.Target:
    """Creates a Hexagon target"""
    target = tvm.target.hexagon(cpu_ver)
    return tvm.target.Target(target, host=target)
