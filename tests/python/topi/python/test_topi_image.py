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
import tvm
from tvm import te
from tvm import topi
import tvm.testing
import tvm.topi.testing
from tvm.contrib.pickle_memoize import memoize


def verify_resize(
    batch,
    in_channel,
    in_height,
    in_width,
    out_height,
    out_width,
    layout="NCHW",
    coord_trans="align_corners",
    method="bilinear",
):
    if layout == "NCHW":
        A = te.placeholder((batch, in_channel, in_height, in_width), name="A", dtype="float32")
        dtype = A.dtype
        out_shape = (batch, in_channel, out_height, out_width)
        a_np = np.random.uniform(size=(batch, in_channel, in_height, in_width)).astype(dtype)
    elif layout == "NHWC":
        A = te.placeholder((batch, in_height, in_width, in_channel), name="A", dtype="float32")
        dtype = A.dtype
        out_shape = (batch, out_height, out_width, in_channel)
        a_np = np.random.uniform(size=(batch, in_height, in_width, in_channel)).astype(dtype)
    else:
        raise NotImplementedError("Layout not supported {} ".format(layout))
    B = topi.image.resize(
        A,
        (out_height, out_width),
        layout=layout,
        coordinate_transformation_mode=coord_trans,
        method=method,
    )
    if method == "bilinear":
        b_np = tvm.topi.testing.bilinear_resize_python(
            a_np, (out_height, out_width), layout, coord_trans
        )
    else:
        # TODO: Nearest neighbor case doesn't do anything with coordinate transform mode, and also
        # nearest_neighbors and align_corners combination in topi doesn't match the output of this
        # function.
        scale_h = out_height / in_height
        scale_w = out_width / in_width
        b_np = tvm.topi.testing.upsampling_python(a_np, (scale_h, scale_w), layout)

    def check_device(device, ctx):
        print("Running on target: %s" % device)
        with tvm.target.Target(device):
            s = tvm.topi.testing.get_injective_schedule(device)(B)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(out_shape, dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)

        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-3, atol=1e-3)

    for device, ctx in tvm.testing.enabled_targets():
        check_device(device, ctx)


@tvm.testing.uses_gpu
def test_resize():
    # Scale NCHW
    verify_resize(4, 16, 32, 32, 50, 50, "NCHW")
    # Scale NCHW + Align Corners
    verify_resize(6, 32, 64, 64, 20, 20, "NCHW")
    # Scale NHWC
    verify_resize(4, 16, 32, 32, 50, 50, "NHWC")
    # Scale NHWC + Align Corners
    verify_resize(6, 32, 64, 64, 20, 20, "NHWC")
    for method in ["nearest_neighbor", "bilinear"]:
        for coord_trans in ["asymmetric", "half_pixel", "align_corners"]:
            for layout in ["NCHW", "NHWC"]:
                # TODO: When topi test has an option for align corners and nearest neighbor that
                # produces correct results, re-enable it.
                if coord_trans == "align_corners" and method == "nearest_neighbor":
                    continue
                verify_resize(4, 16, 32, 32, 50, 50, layout, coord_trans, method=method)


def verify_resize3d(
    batch,
    in_channel,
    in_depth,
    in_height,
    in_width,
    out_depth,
    out_height,
    out_width,
    layout="NCDHW",
    coordinate_transformation_mode="half_pixel",
    method="trilinear",
):
    if layout == "NCDHW":
        A = te.placeholder(
            (batch, in_channel, in_depth, in_height, in_width), name="A", dtype="float32"
        )
        dtype = A.dtype
        out_shape = (batch, in_channel, out_depth, out_height, out_width)
        a_np = np.random.uniform(size=(batch, in_channel, in_depth, in_height, in_width)).astype(
            dtype
        )
    elif layout == "NDHWC":
        A = te.placeholder(
            (batch, in_depth, in_height, in_width, in_channel), name="A", dtype="float32"
        )
        dtype = A.dtype
        out_shape = (batch, out_depth, out_height, out_width, in_channel)
        a_np = np.random.uniform(size=(batch, in_depth, in_height, in_width, in_channel)).astype(
            dtype
        )
    else:
        raise NotImplementedError("Layout not supported {} ".format(layout))

    B = topi.image.resize3d(
        A,
        (out_depth, out_height, out_width),
        layout=layout,
        coordinate_transformation_mode=coordinate_transformation_mode,
        method=method,
    )

    if method == "trilinear":
        b_np = tvm.topi.testing.trilinear_resize3d_python(
            a_np, (out_depth, out_height, out_width), layout, coordinate_transformation_mode
        )
    else:
        scale_d = out_depth / in_depth
        scale_h = out_height / in_height
        scale_w = out_width / in_width
        b_np = tvm.topi.testing.upsampling3d_python(a_np, (scale_d, scale_h, scale_w), layout)

    def check_device(device, ctx):
        print("Running on target: %s" % device)
        with tvm.target.Target(device):
            s = tvm.topi.testing.get_injective_schedule(device)(B)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros(out_shape, dtype=dtype), ctx)
        f = tvm.build(s, [A, B], device)
        f(a, b)

        tvm.testing.assert_allclose(b.asnumpy(), b_np, rtol=1e-3, atol=1e-3)

    for device, ctx in tvm.testing.enabled_targets():
        check_device(device, ctx)


@tvm.testing.uses_gpu
def test_resize3d():
    # Trilinear
    verify_resize3d(4, 8, 16, 16, 16, 25, 25, 25, "NCDHW")
    verify_resize3d(1, 8, 16, 16, 16, 25, 25, 25, "NDHWC")
    verify_resize3d(3, 16, 32, 32, 32, 10, 10, 10, "NCDHW", "align_corners")
    verify_resize3d(3, 16, 32, 32, 32, 10, 10, 10, "NDHWC", "align_corners")
    verify_resize3d(3, 16, 32, 32, 32, 10, 10, 10, "NCDHW", "asymmetric")
    verify_resize3d(3, 16, 32, 32, 32, 10, 10, 10, "NDHWC", "asymmetric")

    # Nearest neighbor
    verify_resize3d(4, 8, 16, 16, 16, 25, 25, 25, "NCDHW", method="nearest_neighbor")
    verify_resize3d(4, 8, 16, 16, 16, 25, 25, 25, "NDHWC", method="nearest_neighbor")


@tvm.testing.uses_gpu
def test_crop_and_resize():
    def verify_crop_and_resize(
        image_shape,
        np_boxes,
        np_box_indices,
        np_crop_size,
        layout="NHWC",
        method="bilinear",
        extrapolation_value=0.0,
    ):

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

        def check_device(device, ctx):
            print("Running on target: %s" % device)
            with tvm.target.Target(device):
                s = tvm.topi.testing.get_injective_schedule(device)(out)
            tvm_images = tvm.nd.array(np_images, ctx)
            tvm_boxes = tvm.nd.array(np_boxes, ctx)
            tvm_indices = tvm.nd.array(np_box_indices, ctx)
            tvm_out = tvm.nd.array(np.zeros(out_shape, dtype="float32"), ctx)
            f = tvm.build(s, [images, boxes, box_ind, out], device, name="crop_and_resize")
            f(tvm_images, tvm_boxes, tvm_indices, tvm_out)

            tvm.testing.assert_allclose(tvm_out.asnumpy(), baseline_np, rtol=1e-3, atol=1e-3)

        for device, ctx in tvm.testing.enabled_targets():
            check_device(device, ctx)

    boxes_1 = np.array([[0.2, 0.3, 0.7, 0.9]], dtype="float32")
    boxes_2 = np.array([[0.2, 0.3, 0.7, 0.9], [0, 0.1, 0.8, 1]], dtype="float32")
    indices_1 = np.array([0], dtype="int32")
    indices_2 = np.array([1, 0], dtype="int32")
    size_1 = (7, 11)
    size_2 = (90, 60)

    verify_crop_and_resize((1, 255, 255, 3), boxes_1, indices_1, size_1, layout="NHWC")
    verify_crop_and_resize(
        (10, 224, 224, 5), boxes_2, indices_2, size_2, extrapolation_value=0.3, layout="NHWC"
    )
    verify_crop_and_resize((1, 100, 100, 3), boxes_1, indices_1, size_1, method="nearest_neighbor")
    verify_crop_and_resize((1, 3, 224, 224), boxes_1, indices_1, size_1, layout="NCHW")


@tvm.testing.uses_gpu
def test_affine_grid():
    def verify_affine_grid(num_batch, target_shape):
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

        def check_device(device, ctx):
            print("Running on target: %s" % device)
            with tvm.target.Target(device):
                s = tvm.topi.testing.get_injective_schedule(device)(out)
            tvm_data = tvm.nd.array(data_np, ctx)
            tvm_out = tvm.nd.empty(out_np.shape, dtype, ctx)
            f = tvm.build(s, [data, out], device)
            f(tvm_data, tvm_out)

            tvm.testing.assert_allclose(tvm_out.asnumpy(), out_np, rtol=1e-5, atol=1e-5)

        for device, ctx in tvm.testing.enabled_targets():
            check_device(device, ctx)

    verify_affine_grid(1, (16, 32))
    verify_affine_grid(4, (16, 32))


@tvm.testing.uses_gpu
def test_grid_sample():
    def verify_grid_sample(data_shape, grid_shape):
        dtype = "float32"
        data = te.placeholder(data_shape, dtype=dtype)
        grid = te.placeholder(grid_shape, dtype=dtype)
        out = topi.image.grid_sample(data, grid, "bilinear")

        @memoize("topi.tests.test_grid_sample.verify_grid_sample")
        def get_ref_data():
            data_np = np.random.uniform(size=data_shape).astype(dtype)
            # allow grid values to be out-of-bound
            grid_np = np.random.uniform(size=grid_shape, low=-1.5, high=1.5).astype(dtype)
            out_np = tvm.topi.testing.grid_sample_nchw_python(data_np, grid_np, "bilinear")
            return data_np, grid_np, out_np

        data_np, grid_np, out_np = get_ref_data()

        def check_device(device, ctx):
            print("Running on target: %s" % device)
            with tvm.target.Target(device):
                s = tvm.topi.testing.get_injective_schedule(device)(out)
            tvm_data = tvm.nd.array(data_np, ctx)
            tvm_grid = tvm.nd.array(grid_np, ctx)
            tvm_out = tvm.nd.empty(out_np.shape, dtype, ctx)
            f = tvm.build(s, [data, grid, out], device)
            f(tvm_data, tvm_grid, tvm_out)

            tvm.testing.assert_allclose(tvm_out.asnumpy(), out_np, rtol=1e-5, atol=1e-5)

        for device, ctx in tvm.testing.enabled_targets():
            check_device(device, ctx)

    verify_grid_sample((4, 4, 16, 32), (4, 2, 8, 8))
    verify_grid_sample((4, 4, 16, 32), (4, 2, 32, 32))


if __name__ == "__main__":
    test_resize()
    test_resize3d()
    test_crop_and_resize()
    test_affine_grid()
    test_grid_sample()
