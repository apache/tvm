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
"""Test code for bilinear scale """

import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.topi.testing

from tvm import te, topi
from tvm.contrib.pickle_memoize import memoize

layout_2d = tvm.testing.parameter("NCHW", "NHWC")
layout_3d = tvm.testing.parameter("NCDHW", "NDHWC")
coordinate_transformation_mode = tvm.testing.parameter("asymmetric", "align_corners", "half_pixel")
interpolation_method = tvm.testing.parameter("nearest_neighbor", "linear")

resize_2d_test_case = tvm.testing.parameter(
    dict(sizes=(4, 16, 32, 32, 50, 50), coord_trans="align_corners", method="linear"),
    dict(sizes=(6, 32, 64, 64, 20, 20), coord_trans="align_corners", method="linear"),
    dict(sizes=(4, 16, 32, 32, 50, 50), coord_trans="asymmetric", method="nearest_neighbor"),
    dict(sizes=(4, 16, 32, 32, 64, 50), coord_trans="asymmetric", method="nearest_neighbor"),
    dict(sizes=(4, 16, 32, 32, 50, 96), coord_trans="asymmetric", method="nearest_neighbor"),
    dict(sizes=(4, 16, 32, 32, 96, 96), coord_trans="asymmetric", method="nearest_neighbor"),
    dict(sizes=(4, 16, 32, 32, 50, 50), coord_trans="align_corners", method="nearest_neighbor"),
    dict(sizes=(4, 16, 32, 32, 50, 50), coord_trans="half_pixel", method="nearest_neighbor"),
    dict(sizes=(4, 16, 32, 32, 50, 50), coord_trans="asymmetric", method="linear"),
    dict(sizes=(4, 16, 32, 32, 50, 50), coord_trans="half_pixel", method="linear"),
)


def test_resize2d(
    target,
    dev,
    resize_2d_test_case,
    layout_2d,
):
    (batch, in_channel, in_height, in_width, out_height, out_width) = resize_2d_test_case["sizes"]
    coordinate_transformation_mode = resize_2d_test_case["coord_trans"]
    interpolation_method = resize_2d_test_case["method"]

    if layout_2d == "NCHW":
        A = te.placeholder((batch, in_channel, in_height, in_width), name="A", dtype="float32")
        dtype = A.dtype
        out_shape = (batch, in_channel, out_height, out_width)
        a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype(dtype)
    elif layout_2d == "NHWC":
        A = te.placeholder((batch, in_height, in_width, in_channel), name="A", dtype="float32")
        dtype = A.dtype
        out_shape = (batch, out_height, out_width, in_channel)
        a_np = np.random.uniform(size=(batch, in_height, in_width, in_channel)).astype(dtype)
    else:
        raise NotImplementedError("Layout not supported {} ".format(layout_2d))

    B = topi.image.resize2d(
        A,
        [0.0] * 4,
        (out_height, out_width),
        layout=layout_2d,
        coordinate_transformation_mode=coordinate_transformation_mode,
        method=interpolation_method,
    )
    scale_h = out_height / in_height
    scale_w = out_width / in_width
    b_np = tvm.topi.testing.resize2d_python(
        a_np, (scale_h, scale_w), layout_2d, interpolation_method, coordinate_transformation_mode
    )

    with tvm.target.Target(target):
        s = tvm.topi.testing.get_injective_schedule(target)(B)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(np.zeros(out_shape, dtype=dtype), dev)
    f = tvm.build(s, [A, B], target)
    f(a, b)

    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-3, atol=1e-3)


resize_3d_test_case = tvm.testing.parameter((3, 16, 32, 32, 32, 10, 10, 10))


def test_resize3d(
    target,
    dev,
    resize_3d_test_case,
    layout_3d,
    coordinate_transformation_mode,
    interpolation_method,
):
    (
        batch,
        in_channel,
        in_depth,
        in_height,
        in_width,
        out_depth,
        out_height,
        out_width,
    ) = resize_3d_test_case

    if layout_3d == "NCDHW":
        A = te.placeholder(
            (batch, in_channel, in_depth, in_height, in_width), name="A", dtype="float32"
        )
        dtype = A.dtype
        out_shape = (batch, in_channel, out_depth, out_height, out_width)
        a_np = np.random.uniform(size=(batch, in_channel, in_depth, in_height, in_width)).astype(
            dtype
        )
    elif layout_3d == "NDHWC":
        A = te.placeholder(
            (batch, in_depth, in_height, in_width, in_channel), name="A", dtype="float32"
        )
        dtype = A.dtype
        out_shape = (batch, out_depth, out_height, out_width, in_channel)
        a_np = np.random.uniform(size=(batch, in_depth, in_height, in_width, in_channel)).astype(
            dtype
        )
    else:
        raise NotImplementedError("Layout not supported {} ".format(layout_3d))

    B = topi.image.resize3d(
        A,
        [0.0] * 6,
        (out_depth, out_height, out_width),
        layout=layout_3d,
        coordinate_transformation_mode=coordinate_transformation_mode,
        method=interpolation_method,
    )

    scale_d = out_depth / in_depth
    scale_h = out_height / in_height
    scale_w = out_width / in_width
    b_np = tvm.topi.testing.resize3d_python(
        a_np,
        (scale_d, scale_h, scale_w),
        layout_3d,
        interpolation_method,
        coordinate_transformation_mode,
    )

    with tvm.target.Target(target):
        s = tvm.topi.testing.get_injective_schedule(target)(B)
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(np.zeros(out_shape, dtype=dtype), dev)
    f = tvm.build(s, [A, B], target)
    f(a, b)

    tvm.testing.assert_allclose(b.numpy(), b_np, rtol=1e-3, atol=1e-3)


box_set_1 = dict(
    boxes=np.array([[0.2, 0.3, 0.7, 0.9]], dtype="float32"),
    indices=np.array([0], dtype="int32"),
    crop_size=(7, 11),
)
box_set_2 = dict(
    boxes=np.array([[0.2, 0.3, 0.7, 0.9], [0, 0.1, 0.8, 1]], dtype="float32"),
    indices=np.array([1, 0], dtype="int32"),
    crop_size=(90, 60),
)
crop_and_resize_test_case = tvm.testing.parameter(
    dict(image_shape=(1, 255, 255, 3), **box_set_1),
    dict(image_shape=(1, 100, 100, 3), **box_set_1, method="nearest_neighbor"),
    dict(image_shape=(1, 3, 224, 224), **box_set_1, layout="NCHW"),
    dict(image_shape=(10, 224, 224, 5), **box_set_2, extrapolation_value=0.3),
)


@tvm.testing.uses_gpu
def test_crop_and_resize(target, dev, crop_and_resize_test_case):
    image_shape = crop_and_resize_test_case["image_shape"]
    np_boxes = crop_and_resize_test_case["boxes"]
    np_box_indices = crop_and_resize_test_case["indices"]
    np_crop_size = crop_and_resize_test_case["crop_size"]
    method = crop_and_resize_test_case.get("method", "bilinear")
    extrapolation_value = crop_and_resize_test_case.get("extrapolation_value", 0.0)
    layout = crop_and_resize_test_case.get("layout", "NHWC")

    images = te.placeholder(image_shape, name="images", dtype="float32")
    np_images = np.random.uniform(size=image_shape).astype("float32")
    boxes = te.placeholder(np_boxes.shape, name="boxes", dtype="float32")
    box_ind = te.placeholder(np_box_indices.shape, name="box_ind", dtype="int32")

    batch = len(np_box_indices)
    target_height, target_width = np_crop_size[0], np_crop_size[1]
    if layout == "NHWC":
        channel = image_shape[3]
        out_shape = (batch, target_height, target_width, channel)
    elif layout == "NCHW":
        channel = image_shape[1]
        out_shape = (batch, channel, target_height, target_width)
    else:
        raise NotImplementedError("Layout {} is not supported.".format(layout))

    out = topi.image.crop_and_resize(
        images,
        boxes,
        box_ind,
        np_crop_size,
        layout=layout,
        method=method,
        extrapolation_value=extrapolation_value,
    )

    baseline_np = tvm.topi.testing.crop_and_resize_python(
        np_images, np_boxes, np_box_indices, np_crop_size, layout, method, extrapolation_value
    )

    with tvm.target.Target(target):
        s = tvm.topi.testing.get_injective_schedule(target)(out)
    tvm_images = tvm.nd.array(np_images, dev)
    tvm_boxes = tvm.nd.array(np_boxes, dev)
    tvm_indices = tvm.nd.array(np_box_indices, dev)
    tvm_out = tvm.nd.array(np.zeros(out_shape, dtype="float32"), dev)
    f = tvm.build(s, [images, boxes, box_ind, out], target, name="crop_and_resize")
    f(tvm_images, tvm_boxes, tvm_indices, tvm_out)

    tvm.testing.assert_allclose(tvm_out.numpy(), baseline_np, rtol=1e-3, atol=1e-3)


affine_grid_test_case = tvm.testing.parameter(
    (1, (16, 32)),
    (4, (16, 32)),
)


def test_affine_grid(target, dev, affine_grid_test_case):
    num_batch, target_shape = affine_grid_test_case

    dtype = "float32"
    data_shape = (num_batch, 2, 3)
    data = te.placeholder(data_shape, dtype=dtype)
    out = topi.image.affine_grid(data, target_shape)

    @memoize("topi.tests.test_affine_grid.verify_affine_grid")
    def get_ref_data():
        data_np = np.random.uniform(size=data_shape).astype(dtype)
        out_np = tvm.topi.testing.affine_grid_python(data_np, target_shape)
        return data_np, out_np

    data_np, out_np = get_ref_data()

    with tvm.target.Target(target):
        s = tvm.topi.testing.get_injective_schedule(target)(out)
    tvm_data = tvm.nd.array(data_np, dev)
    tvm_out = tvm.nd.empty(out_np.shape, dtype, dev)
    f = tvm.build(s, [data, out], target)
    f(tvm_data, tvm_out)

    tvm.testing.assert_allclose(tvm_out.numpy(), out_np, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("method", ["nearest", "bilinear", "bicubic"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [True, False])
@pytest.mark.parametrize("dimension", ["2d", "3d"])
def test_grid_sample(target, dev, method, padding_mode, align_corners, dimension):
    if dimension == "3d" and method == "bicubic":
        pytest.skip('3D "bicubic"(tricubic) is not supported in pytorch')

    if dimension == "2d":
        data_shape = (4, 4, 8, 8)
        grid_shape = (4, 2, 16, 16)
        layout = "NCHW"
    elif dimension == "3d":
        # choosing smaller sizes to be testable on weaker GPUs
        data_shape = (4, 4, 4, 4, 4)
        grid_shape = (4, 3, 8, 8, 8)
        layout = "NCDHW"
    else:
        raise ValueError(f"Unknown dimension: {dimension}")

    dtype = "float32"
    data = te.placeholder(data_shape, dtype=dtype)
    grid = te.placeholder(grid_shape, dtype=dtype)
    out = topi.image.grid_sample(data, grid, method, layout, padding_mode, align_corners)

    @memoize("topi.tests.test_grid_sample.verify_grid_sample")
    def get_ref_data():
        data_np = np.random.uniform(size=data_shape).astype(dtype)
        # allow grid values to be out-of-bound
        grid_np = np.random.uniform(size=grid_shape, low=-1.5, high=1.5).astype(dtype)
        out_np = tvm.topi.testing.grid_sample_python(
            data_np, grid_np, method, layout, padding_mode, align_corners
        )
        return data_np, grid_np, out_np

    data_np, grid_np, out_np = get_ref_data()

    with tvm.target.Target(target):
        s = tvm.topi.testing.get_injective_schedule(target)(out)
    tvm_data = tvm.nd.array(data_np, dev)
    tvm_grid = tvm.nd.array(grid_np, dev)
    tvm_out = tvm.nd.empty(out_np.shape, dtype, dev)
    f = tvm.build(s, [data, grid, out], target)
    f(tvm_data, tvm_grid, tvm_out)

    tvm.testing.assert_allclose(tvm_out.numpy(), out_np, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    tvm.testing.main()
