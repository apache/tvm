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
""" Support level5 operator test cases.
"""
import math
import platform
import sys

import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.topi.testing
from tvm import relay, te
from tvm.relay.testing import run_infer_type

executor_kind = tvm.testing.parameter("graph", "vm")


def test_resize1d_infer_type():
    n, c, w = te.size_var("n"), te.size_var("c"), te.size_var("w")
    x = relay.var("x", relay.TensorType((n, c, w), "int8"))
    tw = te.var("tw")
    z = relay.image.resize1d(x, (tw,))
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((n, c, tw), "int8")

    x = relay.var("x", relay.TensorType((n, c, w), "int8"))
    z = relay.image.resize1d(x, (200,), None, "NCW", "linear", "align_corners")
    assert "size=" in z.astext()
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((n, c, 200), "int8")


class TestResize1D:
    interpolate_method = tvm.testing.parameter("nearest_neighbor", "linear", "cubic")
    coord_trans = tvm.testing.parameter("asymmetric", "align_corners", "half_pixel")

    layout = tvm.testing.parameter("NWC", "NCW")
    dshape, scale = tvm.testing.parameters(
        ((1, 4, 4), 2),
        ((2, 8, 17), 3),
        ((2, 8, 17), 3),
        ((3, 4, 5), 5),
    )

    def test_resize(
        self, target, dev, executor_kind, dshape, scale, interpolate_method, layout, coord_trans
    ):
        target_kind = tvm.target.Target(target).kind.name
        if (
            target_kind == "vulkan"
            and dshape == (3, 4, 5)
            and scale == 5
            and interpolate_method == "nearest_neighbor"
            and coord_trans == "align_corners"
        ):
            pytest.xfail("Known failing case for these parameters")

        if layout == "NWC":
            size = (dshape[1] * scale,)
        else:
            size = (dshape[2] * scale,)

        x_data = np.random.uniform(size=dshape).astype("float32")

        ref_res = tvm.topi.testing.resize1d_python(
            x_data, (scale,), layout, interpolate_method, coord_trans
        )
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.image.resize1d(
            x, size, None, layout, interpolate_method, coordinate_transformation_mode=coord_trans
        )
        assert "size=" in z.astext()
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType(ref_res.shape, "float32")
        func = relay.Function([x], z)
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-3, atol=1e-4)


def test_resize2d_infer_type():
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "int8"))
    th, tw = te.var("th"), te.var("tw")
    z = relay.image.resize2d(x, (th, tw))
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((n, c, th, tw), "int8")

    x = relay.var("x", relay.TensorType((n, c, h, w), "int8"))
    z = relay.image.resize2d(x, (100, 200), None, "NCHW", "linear", "align_corners")
    assert "size=" in z.astext()
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((n, c, 100, 200), "int8")


class TestResize2D:
    interpolate_method = tvm.testing.parameter("nearest_neighbor", "linear", "cubic")
    coord_trans = tvm.testing.parameter("asymmetric", "align_corners", "half_pixel")

    layout = tvm.testing.parameter("NHWC", "NCHW")

    dshape, scale = tvm.testing.parameters(
        ((1, 4, 4, 4), 2),
        ((2, 8, 17, 20), 3),
        ((2, 8, 17, 20), 3),
        ((3, 4, 5, 6), 5),
    )

    def test_resize(
        self, target, dev, executor_kind, dshape, scale, interpolate_method, layout, coord_trans
    ):
        target_kind = tvm.target.Target(target).kind.name
        if (
            target_kind == "vulkan"
            and dshape == (3, 4, 5, 6)
            and scale == 5
            and interpolate_method == "nearest_neighbor"
            and coord_trans == "align_corners"
        ):
            pytest.xfail("Known failing case for these parameters")

        if layout == "NHWC":
            size = (dshape[1] * scale, dshape[2] * scale)
        else:
            size = (dshape[2] * scale, dshape[3] * scale)

        x_data = np.random.uniform(size=dshape).astype("float32")

        ref_res = tvm.topi.testing.resize2d_python(
            x_data, (scale, scale), layout, interpolate_method, coord_trans
        )
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.image.resize2d(
            x, size, None, layout, interpolate_method, coordinate_transformation_mode=coord_trans
        )
        assert "size=" in z.astext()
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType(ref_res.shape, "float32")
        func = relay.Function([x], z)
        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-3, atol=1e-4)


def test_resize3d_infer_type():
    n, c, d, h, w = (
        te.size_var("n"),
        te.size_var("c"),
        te.size_var("d"),
        te.size_var("h"),
        te.size_var("w"),
    )
    x = relay.var("x", relay.TensorType((n, c, d, h, w), "int8"))
    td, th, tw = te.var("td"), te.var("th"), te.var("tw")
    z = relay.image.resize3d(x, (td, th, tw))
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((n, c, td, th, tw), "int8")

    x = relay.var("x", relay.TensorType((n, c, d, h, w), "int8"))
    z = relay.image.resize3d(x, (10, 10, 20), None, "NCDHW", "linear", "align_corners")
    assert "size=" in z.astext()
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((n, c, 10, 10, 20), "int8")


class TestResize3D:
    interpolate_method = tvm.testing.parameter("nearest_neighbor", "linear", "cubic")
    coord_trans = tvm.testing.parameter("asymmetric", "align_corners", "half_pixel")

    layout = tvm.testing.parameter("NDHWC", "NCDHW")

    dshape, scale = tvm.testing.parameters(
        ((1, 4, 4, 4, 4), 2),
    )

    def test_resize(
        self, target, dev, executor_kind, dshape, scale, interpolate_method, layout, coord_trans
    ):
        if layout == "NDHWC":
            size = (dshape[1] * scale, dshape[2] * scale, dshape[3] * scale)
        else:
            size = (dshape[2] * scale, dshape[3] * scale, dshape[4] * scale)

        x_data = np.random.uniform(size=dshape).astype("float32")
        ref_res = tvm.topi.testing.resize3d_python(
            x_data, (scale, scale, scale), layout, interpolate_method, coord_trans
        )
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.image.resize3d(x, size, None, layout, interpolate_method, coord_trans)
        assert "size=" in z.astext()
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType(ref_res.shape, "float32")
        func = relay.Function([x], z)

        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            x_data
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-4, atol=1e-6)


class TestCropAndResize:
    interpolate_method = tvm.testing.parameter("bilinear", "nearest_neighbor")
    layout = tvm.testing.parameter("NHWC", "NCHW")

    @pytest.mark.skipif(
        platform.machine() == "aarch64",
        reason="Currently failing on AArch64 - see https://github.com/apache/tvm/issues/10673",
    )
    def test_crop_and_resize(self, target, dev, executor_kind, layout, interpolate_method):
        target_kind = tvm.target.Target(target).kind.name
        if (
            target_kind == "vulkan"
            and layout == "NHWC"
            and interpolate_method == "nearest_neighbor"
        ):
            pytest.xfail("Known failing case for these parameters")

        extrapolation_value = 0.0

        if layout == "NHWC":
            img_shape = (10, 224, 224, 3)
            boxes = np.array([[0.1, 0.2, 0.8, 0.7], [0.2, 0, 1, 0.6]]).astype("float32")
            box_indices = np.array([1, 0]).astype("int32")
            crop_size = np.array([20, 30]).astype("int32")
        elif layout == "NCHW":
            img_shape = (5, 3, 255, 255)
            boxes = np.array([[0, 0, 1, 1], [0.2, 0.1, 1, 0.9]]).astype("float32")
            box_indices = np.array([0, 1]).astype("int32")
            crop_size = np.array([30, 30]).astype("int32")
        else:
            raise ValueError(f"Unknown layout: {layout}")

        image_data = np.random.uniform(size=img_shape).astype("float32")

        ref_res = tvm.topi.testing.crop_and_resize_python(
            image_data,
            boxes,
            box_indices,
            crop_size,
            layout,
            interpolate_method,
            extrapolation_value,
        )

        img = relay.var("img", relay.TensorType(img_shape, "float32"))
        bx = relay.var("bx", relay.TensorType(boxes.shape, "float32"))
        bx_idx = relay.var("bx_idx", relay.TensorType(box_indices.shape, "int32"))

        z = relay.image.crop_and_resize(
            img, bx, bx_idx, list(crop_size), layout, interpolate_method, extrapolation_value
        )
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType(ref_res.shape, "float32")
        func = relay.Function([img, bx, bx_idx], z)

        op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            image_data, boxes, box_indices
        )
        tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-3, atol=1e-04)


@tvm.testing.uses_gpu
def test_multibox_prior(executor_kind):
    def get_ref_result(
        dshape, sizes=(1.0,), ratios=(1.0,), steps=(-1.0, -1.0), offsets=(0.5, 0.5), clip=True
    ):
        in_height = dshape[2]
        in_width = dshape[3]
        num_sizes = len(sizes)
        num_ratios = len(ratios)
        size_ratio_concat = sizes + ratios
        steps_h = steps[0] if steps[0] > 0 else 1.0 / in_height
        steps_w = steps[1] if steps[1] > 0 else 1.0 / in_width
        offset_h = offsets[0]
        offset_w = offsets[1]

        oshape = (1, in_height * in_width * (num_sizes + num_ratios - 1), 4)
        dtype = "float32"
        np_out = np.zeros(oshape).astype(dtype)

        for i in range(in_height):
            center_h = (i + offset_h) * steps_h
            for j in range(in_width):
                center_w = (j + offset_w) * steps_w
                for k in range(num_sizes + num_ratios - 1):
                    w = (
                        size_ratio_concat[k] * in_height / in_width / 2.0
                        if k < num_sizes
                        else size_ratio_concat[0]
                        * in_height
                        / in_width
                        * math.sqrt(size_ratio_concat[k + 1])
                        / 2.0
                    )
                    h = (
                        size_ratio_concat[k] / 2.0
                        if k < num_sizes
                        else size_ratio_concat[0] / math.sqrt(size_ratio_concat[k + 1]) / 2.0
                    )
                    count = (
                        i * in_width * (num_sizes + num_ratios - 1)
                        + j * (num_sizes + num_ratios - 1)
                        + k
                    )
                    np_out[0][count][0] = center_w - w
                    np_out[0][count][1] = center_h - h
                    np_out[0][count][2] = center_w + w
                    np_out[0][count][3] = center_h + h
        if clip:
            np_out = np.clip(np_out, 0, 1)

        return np_out

    def verify_multibox_prior(
        x,
        dshape,
        ref_res,
        sizes=(1.0,),
        ratios=(1.0,),
        steps=(-1.0, -1.0),
        offsets=(0.5, 0.5),
        clip=True,
        check_size=False,
        check_type_only=False,
    ):

        z = relay.vision.multibox_prior(x, sizes, ratios, steps, offsets, clip)
        zz = run_infer_type(z)
        if check_size:
            assert "sizes=" in z.astext()
        assert zz.checked_type == relay.TensorType(
            (1, dshape[2] * dshape[3] * (len(sizes) + len(ratios) - 1), 4), "float32"
        )

        if check_type_only:
            return

        data = np.random.uniform(low=-1, high=1, size=dshape).astype("float32")
        func = relay.Function([x], z)
        func = run_infer_type(func)
        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    sizes = (0.3, 1.5, 0.7)
    ratios = (1.3, 2.4)
    steps = (2.0, 1.5)
    offsets = (0.2, 0.3)
    dshape = (1, 3, 56, 56)
    ref_res = get_ref_result(dshape, sizes, ratios, steps, offsets)
    x = relay.var("x", relay.TensorType(dshape, "float32"))
    verify_multibox_prior(x, dshape, ref_res, sizes, ratios, steps, offsets, check_size=True)
    y = relay.var("y", relay.TensorType((te.size_var("n"), 3, 56, 56), "float32"))
    verify_multibox_prior(
        x, dshape, ref_res, sizes, ratios, steps, offsets, check_size=True, check_type_only=True
    )

    dshape = (1, 24, 32, 32)
    ref_res = get_ref_result(dshape, clip=False)
    x = relay.var("x", relay.TensorType(dshape, "float32"))
    verify_multibox_prior(x, dshape, ref_res, clip=False)
    y = relay.var("y", relay.TensorType((te.size_var("n"), 24, 32, 32), "float32"))
    verify_multibox_prior(x, dshape, ref_res, clip=False, check_type_only=True)


@tvm.testing.uses_gpu
def test_get_valid_counts():
    def verify_get_valid_counts(dshape, score_threshold, id_index, score_index):
        dtype = "float32"
        batch_size, num_anchor, elem_length = dshape
        np_data = np.random.uniform(low=-2, high=2, size=dshape).astype(dtype)
        np_out1 = np.zeros(shape=(batch_size,))
        np_out2 = np.zeros(shape=dshape).astype(dtype)
        np_out3 = np.zeros(shape=(batch_size, num_anchor))
        for i in range(batch_size):
            np_out1[i] = 0
            inter_idx = 0
            for j in range(num_anchor):
                score = np_data[i, j, score_index]
                if score > score_threshold and (id_index < 0 or np_data[i, j, id_index] >= 0):
                    for k in range(elem_length):
                        np_out2[i, inter_idx, k] = np_data[i, j, k]
                    np_out1[i] += 1
                    np_out3[i, inter_idx] = j
                    inter_idx += 1
                if j >= np_out1[i]:
                    for k in range(elem_length):
                        np_out2[i, j, k] = -1.0
                    np_out3[i, j] = -1

        x = relay.var("x", relay.ty.TensorType(dshape, dtype))
        z = relay.vision.get_valid_counts(x, score_threshold, id_index, score_index)
        assert "score_threshold" in z.astext()
        func = relay.Function([x], z.astuple())
        func = run_infer_type(func)
        for target, dev in tvm.testing.enabled_targets():
            out = relay.create_executor("vm", device=dev, target=target).evaluate(func)(np_data)

            tvm.testing.assert_allclose(out[0].numpy(), np_out1, rtol=1e-3, atol=1e-04)
            tvm.testing.assert_allclose(out[1].numpy(), np_out2, rtol=1e-3, atol=1e-04)
            tvm.testing.assert_allclose(out[2].numpy(), np_out3, rtol=1e-3, atol=1e-04)

    verify_get_valid_counts((1, 2500, 6), 0, 0, 1)
    verify_get_valid_counts((1, 2500, 5), -1, -1, 0)
    verify_get_valid_counts((3, 1000, 6), 0.55, 1, 0)
    verify_get_valid_counts((16, 500, 5), 0.95, -1, 0)


@tvm.testing.uses_gpu
def test_non_max_suppression(executor_kind):
    def verify_nms(
        x0_data,
        x1_data,
        x2_data,
        x3_data,
        dshape,
        ref_res,
        ref_indices_res,
        iou_threshold=0.5,
        force_suppress=False,
        top_k=-1,
        check_type_only=False,
    ):
        x0 = relay.var("x0", relay.ty.TensorType(dshape, "float32"))
        x1 = relay.var("x1", relay.ty.TensorType((dshape[0],), "int32"))
        x2 = relay.var("x2", relay.ty.TensorType((dshape[0], dshape[1]), "int32"))
        x3 = relay.var("x3", relay.ty.TensorType((), "int32"))
        z = relay.vision.non_max_suppression(
            x0,
            x1,
            x2,
            x3,
            iou_threshold=iou_threshold,
            force_suppress=force_suppress,
            top_k=top_k,
            return_indices=False,
        )
        z_indices = relay.vision.non_max_suppression(
            x0,
            x1,
            x2,
            x3,
            iou_threshold=iou_threshold,
            force_suppress=force_suppress,
            top_k=top_k,
            return_indices=True,
        )
        if isinstance(z_indices, relay.expr.TupleWrapper):
            z_indices = z_indices.astuple()
        zz = run_infer_type(z)
        zz_indices = run_infer_type(z_indices)
        assert zz.checked_type == relay.ty.TensorType(dshape, "float32")
        assert zz_indices.checked_type == relay.ty.TupleType(
            [
                relay.ty.TensorType((dshape[0], dshape[1]), "int32"),
                relay.ty.TensorType((dshape[0], 1), "int32"),
            ]
        )

        if check_type_only:
            return

        func = relay.Function([x0, x1, x2, x3], z)
        func = run_infer_type(func)
        func_indices = relay.Function([x0, x1, x2, x3], z_indices)
        func_indices = run_infer_type(func_indices)
        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                x0_data, x1_data, x2_data, x3_data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)
            op_indices_res = relay.create_executor(
                executor_kind, device=dev, target=target
            ).evaluate(func_indices)(x0_data, x1_data, x2_data, x3_data)
            tvm.testing.assert_allclose(op_indices_res[0].numpy(), ref_indices_res, rtol=1e-5)

    np_data = np.array(
        [
            [
                [0, 0.8, 1, 20, 25, 45],
                [1, 0.7, 30, 60, 50, 80],
                [0, 0.4, 4, 21, 19, 40],
                [2, 0.9, 35, 61, 52, 79],
                [1, 0.5, 100, 60, 70, 110],
            ]
        ]
    ).astype("float32")
    np_valid_count = np.array([4]).astype("int32")
    np_indices = np.array([[0, 1, 3, 4, -1]]).astype("int32")
    np_max_output_size = -1

    np_result = np.array(
        [
            [
                [2, 0.9, 35, 61, 52, 79],
                [0, 0.8, 1, 20, 25, 45],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ]
        ]
    )
    np_indices_result = np.array([[4, 0, -1, -1, -1]])
    num_anchors = 5

    dshape = (te.size_var("n"), num_anchors, 6)
    verify_nms(
        np_data,
        np_valid_count,
        np_indices,
        np_max_output_size,
        dshape,
        np_result,
        np_indices_result,
        force_suppress=True,
        top_k=2,
        check_type_only=True,
    )
    dshape = (1, num_anchors, 6)
    verify_nms(
        np_data,
        np_valid_count,
        np_indices,
        np_max_output_size,
        dshape,
        np_result,
        np_indices_result,
        force_suppress=True,
        top_k=2,
        check_type_only=False,
    )

    np_result = np.array(
        [
            [
                [2, 0.9, 35, 61, 52, 79],
                [0, 0.8, 1, 20, 25, 45],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1],
            ]
        ]
    )
    np_indices_result = np.array([[4, 0, -1, -1, -1]])
    np_max_output_size = 2
    dshape = (te.size_var("n"), num_anchors, 6)
    verify_nms(
        np_data,
        np_valid_count,
        np_indices,
        np_max_output_size,
        dshape,
        np_result,
        np_indices_result,
        check_type_only=True,
    )
    dshape = (1, num_anchors, 6)
    verify_nms(
        np_data,
        np_valid_count,
        np_indices,
        np_max_output_size,
        dshape,
        np_result,
        np_indices_result,
        top_k=2,
    )

    np_data = np.array(
        [
            [
                [0, 0.8, 1, 20, 25, 45, 1, 2, 3, 4],
                [1, 0.7, 30, 60, 50, 80, 5, 6, 7, 8],
                [0, 0.4, 4, 21, 19, 40, 9, 10, 11, 12],
                [2, 0.9, 35, 61, 52, 79, 13, 14, 15, 16],
                [1, 0.5, 100, 60, 70, 110, 17, 18, 19, 20],
            ]
        ]
    ).astype("float32")
    np_result = np.array(
        [
            [
                [2, 0.9, 35, 61, 52, 79, 13, 14, 15, 16],
                [0, 0.8, 1, 20, 25, 45, 1, 2, 3, 4],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ]
        ]
    )
    dshape = (1, 5, 10)
    verify_nms(
        np_data,
        np_valid_count,
        np_indices,
        np_max_output_size,
        dshape,
        np_result,
        np_indices_result,
        force_suppress=True,
        top_k=2,
        check_type_only=False,
    )


@tvm.testing.uses_gpu
def test_multibox_transform_loc(executor_kind):
    def test_default_value():
        num_anchors = 3
        num_classes = 3

        np_cls_prob = np.array([[[0.2, 0.5, 0.3], [0.25, 0.3, 0.45], [0.7, 0.1, 0.2]]]).astype(
            "float32"
        )
        np_loc_preds = np.array(
            [[0.1, -0.2, 0.3, 0.2, 0.2, 0.4, 0.5, -0.3, 0.7, -0.2, -0.4, -0.8]]
        ).astype("float32")
        np_anchors = np.array(
            [[[-0.1, -0.1, 0.1, 0.1], [-0.2, -0.2, 0.2, 0.2], [1.2, 1.2, 1.5, 1.5]]]
        ).astype("float32")

        expected_np_out = np.array(
            [
                [
                    [1, 0.69999999, 0, 0, 0.10818365, 0.10008108],
                    [0, 0.44999999, 1, 1, 1, 1],
                    [0, 0.30000001, 0, 0, 0.22903419, 0.20435292],
                ]
            ]
        )

        cls_prob = relay.var(
            "cls_prob", relay.ty.TensorType((1, num_anchors, num_classes), "float32")
        )
        loc_pred = relay.var("loc_pred", relay.ty.TensorType((1, num_anchors * 4), "float32"))
        anchors = relay.var("anchors", relay.ty.TensorType((1, num_anchors, 4), "float32"))

        mtl = relay.vision.multibox_transform_loc(
            cls_prob=cls_prob, loc_pred=loc_pred, anchor=anchors
        )
        ret = run_infer_type(mtl.astuple())
        ref_type = relay.ty.TupleType(
            tvm.runtime.convert(
                [
                    relay.ty.TensorType((1, num_anchors, 6), "float32"),
                    relay.ty.TensorType((1,), "int"),
                ]
            )
        )

        assert ret.checked_type == ref_type

        nms = relay.vision.non_max_suppression(mtl[0], mtl[1], mtl[0], return_indices=False)
        func = relay.Function([cls_prob, loc_pred, anchors], nms)
        func = run_infer_type(func)
        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                np_cls_prob, np_loc_preds, np_anchors
            )
            tvm.testing.assert_allclose(op_res.numpy(), expected_np_out, rtol=1e-5)

    def test_threshold():
        num_anchors = 5
        num_classes = 5
        n = te.size_var("n")
        cls_prob = relay.var(
            "cls_prob", relay.ty.TensorType((n, num_anchors, num_classes), "float32")
        )
        loc_pred = relay.var("loc_pred", relay.ty.TensorType((n, num_anchors * 4), "float32"))
        anchors = relay.var("anchors", relay.ty.TensorType((1, num_anchors, 4), "float32"))
        threshold = 0.02
        variances = (0.2, 0.2, 0.3, 0.3)

        ret = relay.vision.multibox_transform_loc(
            cls_prob=cls_prob,
            loc_pred=loc_pred,
            anchor=anchors,
            threshold=threshold,
            variances=variances,
        )
        ret = run_infer_type(ret.astuple())
        ref_type = relay.ty.TupleType(
            tvm.runtime.convert(
                [
                    relay.ty.TensorType((n, num_anchors, 6), "float32"),
                    relay.ty.TensorType((n,), "int"),
                ]
            )
        )
        assert ret.checked_type == ref_type

    test_default_value()
    test_threshold()


@tvm.testing.uses_gpu
def test_roi_align(executor_kind):
    def verify_roi_align(
        data_shape,
        rois_shape,
        channel,
        in_size,
        pooled_size,
        spatial_scale,
        sample_ratio,
        mode,
        layout,
        ref_func,
    ):
        data = relay.var("data", relay.ty.TensorType(data_shape, "float32"))
        rois = relay.var("rois", relay.ty.TensorType(rois_shape, "float32"))
        z = relay.vision.roi_align(
            data,
            rois,
            pooled_size=(pooled_size, pooled_size),
            spatial_scale=spatial_scale,
            sample_ratio=sample_ratio,
            mode=mode,
            layout=layout,
        )
        zz = run_infer_type(z)

        num_roi = rois_shape[0]

        if layout == "NCHW":
            assert zz.checked_type == relay.ty.TensorType(
                (num_roi, channel, pooled_size, pooled_size), "float32"
            )
        else:
            assert zz.checked_type == relay.ty.TensorType(
                (num_roi, pooled_size, pooled_size, channel), "float32"
            )

        func = relay.Function([data, rois], z)
        func = run_infer_type(func)
        np_data = np.random.uniform(size=data_shape).astype("float32")
        np_rois = np.random.uniform(size=rois_shape).astype("float32") * in_size
        np_rois[:, 0] = np.random.randint(low=0, high=data_shape[0], size=num_roi)
        ref_res = ref_func(
            np_data,
            np_rois,
            pooled_size=pooled_size,
            spatial_scale=spatial_scale,
            sample_ratio=sample_ratio,
            mode=mode,
        )
        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                np_data, np_rois
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, atol=1e-6, rtol=1e-3)

    def verify_roi_align_nchw(
        data_shape, rois_shape, pooled_size, spatial_scale, sample_ratio, mode
    ):
        _, channel, in_size, _ = data_shape
        return verify_roi_align(
            data_shape,
            rois_shape,
            channel,
            in_size,
            pooled_size,
            spatial_scale,
            sample_ratio,
            mode,
            "NCHW",
            tvm.topi.testing.roi_align_nchw_python,
        )

    def verify_roi_align_nhwc(
        data_shape, rois_shape, pooled_size, spatial_scale, sample_ratio, mode
    ):
        _, in_size, _, channel = data_shape
        return verify_roi_align(
            data_shape,
            rois_shape,
            channel,
            in_size,
            pooled_size,
            spatial_scale,
            sample_ratio,
            mode,
            "NHWC",
            tvm.topi.testing.roi_align_nhwc_python,
        )

    verify_roi_align_nchw(
        (1, 4, 16, 16), (32, 5), pooled_size=7, spatial_scale=1.0, sample_ratio=-1, mode="avg"
    )
    verify_roi_align_nchw(
        (4, 4, 16, 16), (32, 5), pooled_size=7, spatial_scale=0.5, sample_ratio=2, mode="avg"
    )
    verify_roi_align_nchw(
        (1, 4, 16, 16), (32, 5), pooled_size=7, spatial_scale=1.0, sample_ratio=-1, mode="max"
    )
    verify_roi_align_nchw(
        (4, 4, 16, 16), (32, 5), pooled_size=7, spatial_scale=0.5, sample_ratio=2, mode="max"
    )
    verify_roi_align_nhwc(
        (1, 16, 16, 4), (32, 5), pooled_size=7, spatial_scale=1.0, sample_ratio=-1, mode="avg"
    )
    verify_roi_align_nhwc(
        (4, 16, 16, 4), (32, 5), pooled_size=7, spatial_scale=0.5, sample_ratio=2, mode="avg"
    )
    verify_roi_align_nhwc(
        (1, 16, 16, 4), (32, 5), pooled_size=7, spatial_scale=1.0, sample_ratio=-1, mode="max"
    )
    verify_roi_align_nhwc(
        (4, 16, 16, 4), (32, 5), pooled_size=7, spatial_scale=0.5, sample_ratio=2, mode="max"
    )


@tvm.testing.uses_gpu
def test_roi_pool(executor_kind):
    def verify_roi_pool(data_shape, rois_shape, pooled_size, spatial_scale):
        data = relay.var("data", relay.ty.TensorType(data_shape, "float32"))
        rois = relay.var("rois", relay.ty.TensorType(rois_shape, "float32"))
        z = relay.vision.roi_pool(
            data,
            rois,
            pooled_size=(pooled_size, pooled_size),
            spatial_scale=spatial_scale,
            layout="NCHW",
        )
        zz = run_infer_type(z)
        batch, channel, in_size, _ = data_shape
        num_roi = rois_shape[0]
        assert zz.checked_type == relay.ty.TensorType(
            (num_roi, channel, pooled_size, pooled_size), "float32"
        )

        func = relay.Function([data, rois], z)
        func = run_infer_type(func)
        np_data = np.random.uniform(size=data_shape).astype("float32")
        np_rois = np.random.uniform(size=rois_shape).astype("float32") * in_size
        np_rois[:, 0] = np.random.randint(low=0, high=batch, size=num_roi).astype("float32")
        ref_res = tvm.topi.testing.roi_pool_nchw_python(
            np_data, np_rois, pooled_size=pooled_size, spatial_scale=spatial_scale
        )
        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                np_data, np_rois
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-4)

    verify_roi_pool((1, 4, 16, 16), (32, 5), pooled_size=7, spatial_scale=1.0)
    verify_roi_pool((4, 4, 16, 16), (32, 5), pooled_size=7, spatial_scale=0.5)


@tvm.testing.uses_gpu
def test_proposal(executor_kind):
    def verify_proposal(np_cls_prob, np_bbox_pred, np_im_info, np_out, attrs):
        cls_prob = relay.var("cls_prob", relay.ty.TensorType(np_cls_prob.shape, "float32"))
        bbox_pred = relay.var("bbox_pred", relay.ty.TensorType(np_bbox_pred.shape, "float32"))
        im_info = relay.var("im_info", relay.ty.TensorType(np_im_info.shape, "float32"))
        z = relay.vision.proposal(cls_prob, bbox_pred, im_info, **attrs)
        zz = run_infer_type(z)
        assert zz.checked_type == relay.ty.TensorType(np_out.shape, "float32")

        func = relay.Function([cls_prob, bbox_pred, im_info], z)
        func = run_infer_type(func)
        for target in ["llvm", "cuda"]:
            if not tvm.testing.device_enabled(target):
                print("Skip test because %s is not enabled." % target)
                continue
            dev = tvm.device(target, 0)
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                np_cls_prob, np_bbox_pred, np_im_info
            )
            tvm.testing.assert_allclose(op_res.numpy(), np_out, rtol=1e-4)

    attrs = {
        "scales": (0.5,),
        "ratios": (0.5,),
        "feature_stride": 16,
        "iou_loss": False,
        "rpn_min_size": 16,
        "threshold": 0.7,
        "rpn_pre_nms_top_n": 200,
        "rpn_post_nms_top_n": 4,
    }

    np_cls_prob = np.array(
        [
            [
                [[0.3, 0.6, 0.2], [0.4, 0.7, 0.5], [0.1, 0.4, 0.3]],
                [[0.7, 0.5, 0.3], [0.6, 0.4, 0.8], [0.9, 0.2, 0.5]],
            ]
        ],
        dtype="float32",
    )
    np_bbox_pred = np.array(
        [
            [
                [[0.5, 1.0, 0.6], [0.8, 1.2, 2.0], [0.9, 1.0, 0.8]],
                [[0.5, 1.0, 0.7], [0.8, 1.2, 1.6], [2.1, 1.5, 0.7]],
                [[1.0, 0.5, 0.7], [1.5, 0.9, 1.6], [1.4, 1.5, 0.8]],
                [[1.0, 0.5, 0.6], [1.5, 0.9, 2.0], [1.8, 1.0, 0.9]],
            ]
        ],
        dtype="float32",
    )
    np_im_info = np.array([[48.0, 48.0, 1.0]], dtype="float32")
    np_out = np.array(
        [
            [0.0, 0.0, 2.8451548, 28.38012, 18.154846],
            [0.0, 0.0, 15.354933, 41.96971, 41.245064],
            [0.0, 18.019852, 1.0538368, 51.98015, 25.946163],
            [0.0, 27.320923, -1.266357, 55.0, 24.666357],
        ],
        dtype="float32",
    )

    verify_proposal(np_cls_prob, np_bbox_pred, np_im_info, np_out, attrs)

    np_out = np.array(
        [
            [0.0, -5.25, -2.5, 21.75, 19.0],
            [0.0, 11.25, -2.0, 37.25, 18.5],
            [0.0, 26.849998, -2.3000002, 53.45, 18.6],
            [0.0, -4.95, 13.799999, 22.25, 35.5],
        ],
        dtype="float32",
    )
    attrs["iou_loss"] = True
    verify_proposal(np_cls_prob, np_bbox_pred, np_im_info, np_out, attrs)


def test_yolo_reorg_infer_shape():
    def verify_yolo_reorg(shape, stride, out_shape):
        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.vision.yolo_reorg(x, stride=stride)
        zz = run_infer_type(z)
        assert "stride=" in z.astext()
        assert zz.checked_type == relay.ty.TensorType(out_shape, "float32")

    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    idxd = tvm.tir.indexdiv
    verify_yolo_reorg((n, c, 20, 20), 10, (n, c * 10 * 10, 2, 2))
    verify_yolo_reorg((n, c, h, w), 2, (n, c * 2 * 2, idxd(h, 2), idxd(w, 2)))


@tvm.testing.uses_gpu
def test_yolo_reorg(executor_kind):
    def verify_yolo_reorg(shape, stride):
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        ref_res = tvm.topi.testing.reorg_python(x_data, stride)

        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.vision.yolo_reorg(x, stride=stride)
        zz = run_infer_type(z)
        assert "stride=" in z.astext()
        assert zz.checked_type == relay.ty.TensorType(ref_res.shape, "float32")

        func = relay.Function([x], z)

        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                x_data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5)

    verify_yolo_reorg((1, 100, 20, 20), 10)
    verify_yolo_reorg((1, 4, 6, 6), 2)


class TestDeformableConv2D:
    batch, in_channel, size, out_channel, deformable_groups = tvm.testing.parameters(
        (1, 4, 16, 4, 4),
        (2, 4, 16, 4, 1),
    )
    kernel_size = tvm.testing.parameter((3, 3))
    groups = tvm.testing.parameter(1, 2)
    layout = tvm.testing.parameter("NCHW", "NHWC")
    dtype = tvm.testing.parameter("float32")

    @tvm.testing.fixture
    def data_shape(self, layout, batch, in_channel, size):
        if layout == "NCHW":
            return (batch, in_channel, size, size)
        elif layout == "NHWC":
            return (batch, size, size, in_channel)

    @tvm.testing.fixture
    def kernel_shape(self, layout, in_channel, out_channel, groups, kernel_size):
        if layout == "NCHW":
            return (out_channel, in_channel // groups, kernel_size[0], kernel_size[1])
        elif layout == "NHWC":
            return (kernel_size[0], kernel_size[1], in_channel // groups, out_channel)

    @tvm.testing.fixture
    def out_shape(self, layout, batch, out_channel, size):
        if layout == "NCHW":
            return (batch, out_channel, size, size)
        elif layout == "NHWC":
            return (batch, size, size, out_channel)

    @tvm.testing.fixture
    def offset_shape(self, layout, batch, kernel_size, deformable_groups, out_shape):
        if layout == "NCHW":
            return (
                batch,
                2 * kernel_size[0] * kernel_size[1] * deformable_groups,
                out_shape[2],
                out_shape[3],
            )
        elif layout == "NHWC":
            return (
                batch,
                out_shape[1],
                out_shape[2],
                2 * kernel_size[0] * kernel_size[1] * deformable_groups,
            )

    @tvm.testing.fixture
    def kernel_layout(self, layout):
        return {"NCHW": "OIHW", "NHWC": "HWIO"}[layout]

    @tvm.testing.fixture
    def relay_setup(
        self,
        dtype,
        data_shape,
        layout,
        kernel_layout,
        kernel_size,
        deformable_groups,
        groups,
        out_channel,
    ):
        data = relay.var("data", shape=data_shape, dtype=dtype)
        offset = relay.var("offset", dtype=dtype)
        kernel = relay.var("kernel", dtype=dtype)
        expr = relay.nn.deformable_conv2d(
            data,
            offset,
            kernel,
            strides=(1, 1),
            padding=(1, 1),
            dilation=(1, 1),
            data_layout=layout,
            kernel_layout=kernel_layout,
            kernel_size=kernel_size,
            deformable_groups=deformable_groups,
            groups=groups,
            channels=out_channel,
        )
        func = relay.Function([data, offset, kernel], expr)
        return expr, func

    def test_infer_type(self, relay_setup, out_shape, offset_shape, kernel_shape):
        expr, func = relay_setup
        yy = run_infer_type(expr)
        assert yy.checked_type == relay.TensorType(out_shape), yy.checked_type
        assert yy.args[1].checked_type == relay.TensorType(offset_shape), yy.args[1].checked_type
        assert yy.args[2].checked_type == relay.TensorType(kernel_shape), yy.args[2].checked_type

    # The reference python implementation only supports groups==1.
    @pytest.mark.parametrize("groups", [1])
    def test_run(
        self,
        target,
        dev,
        dtype,
        executor_kind,
        data_shape,
        offset_shape,
        kernel_shape,
        relay_setup,
        deformable_groups,
        groups,
        layout,
    ):
        target = tvm.target.Target(target)
        if layout == "NHWC" and target.kind.name != "llvm":
            pytest.xfail("Can only run NHWC layout on llvm")

        expr, func = relay_setup
        data = np.random.uniform(size=data_shape).astype(dtype)
        offset = np.random.uniform(size=offset_shape).astype(dtype)
        kernel = np.random.uniform(size=kernel_shape).astype(dtype)
        if layout == "NCHW":
            ref_res = tvm.topi.testing.deformable_conv2d_nchw_python(
                data,
                offset,
                kernel,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                deformable_groups=deformable_groups,
                groups=groups,
            )
        else:
            ref_res = tvm.topi.testing.deformable_conv2d_nhwc_python(
                data,
                offset,
                kernel,
                stride=(1, 1),
                padding=(1, 1),
                dilation=(1, 1),
                deformable_groups=deformable_groups,
                groups=groups,
            )

        op_res1 = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
            data, offset, kernel
        )
        tvm.testing.assert_allclose(op_res1.numpy(), ref_res, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_depth_to_space(executor_kind):
    def verify_depth_to_space(dshape, block_size, layout, mode):
        if layout == "NHWC":
            out_shape = [
                dshape[0],
                dshape[1] * block_size,
                dshape[2] * block_size,
                dshape[3] / (block_size * block_size),
            ]
        else:
            out_shape = [
                dshape[0],
                dshape[1] / (block_size * block_size),
                dshape[2] * block_size,
                dshape[3] * block_size,
            ]

        x_data = np.random.uniform(size=dshape).astype("float32")
        if layout == "NHWC":
            x_data = np.transpose(x_data, axes=[0, 3, 1, 2])
        ref_res = tvm.topi.testing.depth_to_space_python(x_data, block_size, mode=mode)
        if layout == "NHWC":
            x_data = np.transpose(x_data, axes=[0, 2, 3, 1])
            ref_res = np.transpose(ref_res, axes=[0, 2, 3, 1])

        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.nn.depth_to_space(x, block_size, layout, mode)
        assert "block_size=" in z.astext()
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType(ref_res.shape, "float32")
        func = relay.Function([x], z)

        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                x_data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-4)

    for layout in ["NHWC", "NCHW"]:
        for mode in ["DCR", "CDR"]:
            verify_depth_to_space((1, 4, 4, 4), 2, layout, mode)


@tvm.testing.uses_gpu
def test_space_to_depth(executor_kind):
    def verify_space_to_depth(dshape, block_size, layout):
        if layout == "NHWC":
            out_shape = [
                dshape[0],
                dshape[1] / block_size,
                dshape[2] / block_size,
                dshape[3] * (block_size * block_size),
            ]
        else:
            out_shape = [
                dshape[0],
                dshape[1] * (block_size * block_size),
                dshape[2] / block_size,
                dshape[3] / block_size,
            ]

        x_data = np.random.uniform(size=dshape).astype("float32")
        if layout == "NHWC":
            x_data = np.transpose(x_data, axes=[0, 3, 1, 2])
        ref_res = tvm.topi.testing.space_to_depth_python(x_data, block_size)
        if layout == "NHWC":
            x_data = np.transpose(x_data, axes=[0, 2, 3, 1])
            ref_res = np.transpose(ref_res, axes=[0, 2, 3, 1])

        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.nn.space_to_depth(x, block_size, layout)
        assert "block_size=" in z.astext()
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType(ref_res.shape, "float32")
        func = relay.Function([x], z)

        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                x_data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-4)

    for layout in ["NHWC", "NCHW"]:
        verify_space_to_depth((1, 4, 4, 4), 2, layout)


def test_dilation2d_infer_type():
    # symbolic in batch dimension
    n, h, w, c = te.var("n"), 224, 224, 10
    x = relay.var("x", relay.ty.TensorType((n, c, h, w), "float32"))
    kc, kh, kw = 10, 8, 8
    w = relay.var("w", relay.ty.TensorType((kc, kw, kh), "float32"))
    y = relay.image.dilation2d(
        x,
        w,
        # kernel_size=(3, 3),
        strides=[1, 1, 1, 1],
        dilations=[1, 1, 1, 1],
        padding=[0, 0, 0, 0],
    )
    yy = run_infer_type(y)
    assert yy.checked_type == relay.TensorType((n, 10, 217, 217), "float32")


class TestDilation2DRun:
    data_layout, kernel_layout = tvm.testing.parameters(("NCHW", "IHW"), ("NHWC", "HWI"))
    dtype = tvm.testing.parameter("float32")

    config = tvm.testing.parameter(
        dict(
            image=[[[[0.1], [0.2]], [[0.3], [0.4]]]],
            kernel=[[[0.4], [0.3]], [[0.1], [0.0]]],
            out=[[[[0.5]]]],
        ),
        dict(
            image=[[[[0.1], [0.2]], [[0.3], [0.4]]]],
            kernel=[[[0.4], [0.3]], [[0.1], [0.0]]],
            out=[[[[0.5], [0.6]], [[0.7], [0.8]]]],
            padding=[0, 0, 1, 1],
        ),
        dict(
            image=[[[[0.1, 0.2, 0.0], [0.2, 0.3, 0.1]], [[0.3, 0.4, 0.2], [0.4, 0.5, 0.3]]]],
            kernel=[[[0.4, 0.5, 0.3], [0.3, 0.4, 0.2]], [[0.1, 0.2, 0.0], [0.0, 0.1, -0.1]]],
            out=[[[[0.5, 0.7, 0.3], [0.6, 0.8, 0.4]], [[0.7, 0.9, 0.5], [0.8, 1.0, 0.6]]]],
            padding=[0, 0, 1, 1],
        ),
        dict(
            image=[[[[0.1], [0.2]], [[0.3], [0.4]]], [[[0.2], [0.3]], [[0.4], [0.5]]]],
            kernel=[[[0.4], [0.3]], [[0.1], [0.0]]],
            out=[[[[0.5], [0.6]], [[0.7], [0.8]]], [[[0.6], [0.7]], [[0.8], [0.9]]]],
            padding=[0, 0, 1, 1],
        ),
        dict(
            image=[[[[0.1], [0.2]], [[0.3], [0.4]]]],
            kernel=[[[0.4], [0.3]]],
            out=[[[[0.5]], [[0.7]]]],
        ),
        dict(
            image=[[[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]], [[0.7], [0.8], [0.9]]]],
            kernel=[[[0.4], [0.3]], [[0.1], [0.2]]],
            out=[[[[0.7], [0.8], [0.6]], [[1.0], [1.1], [0.9]], [[0.8], [0.9], [0.9]]]],
            padding=[1, 1],
            dilations=[2, 2],
        ),
        dict(
            image=[
                [
                    [[0.1], [0.2], [0.3], [0.4]],
                    [[0.5], [0.6], [0.7], [0.8]],
                    [[0.9], [1.0], [1.1], [1.2]],
                ]
            ],
            kernel=[[[0.4], [0.3]], [[0.1], [0.2]]],
            out=[[[[0.8], [1.0]], [[1.2], [1.4]]]],
            strides=[1, 2],
        ),
    )

    @tvm.testing.fixture
    def test_case(self, config, data_layout, dtype):
        indata = np.array(config["image"], dtype=dtype)
        kernel = np.array(config["kernel"], dtype=dtype)
        out = np.array(config["out"], dtype=dtype)

        if data_layout == "NHWC":
            pass
        elif data_layout == "NCHW":
            indata = indata.transpose([0, 3, 1, 2])
            kernel = kernel.transpose([2, 0, 1])
            out = out.transpose([0, 3, 1, 2])
        else:
            raise ValueError(f"Unsupported layout '{data_layout}'")

        return indata, kernel, out

    @tvm.testing.parametrize_targets("llvm")
    def test_dilation2d(
        self,
        target,
        dev,
        test_case,
        dtype,
        config,
        data_layout,
        kernel_layout,
    ):
        strides = config.get("strides", [1, 1])
        padding = config.get("padding", [0, 0])
        dilations = config.get("dilations", [1, 1])

        indata, kernel, out = test_case

        x = relay.var("x", shape=indata.shape, dtype=dtype)
        w = relay.var("w", shape=kernel.shape, dtype=dtype)
        y = relay.image.dilation2d(
            x,
            w,
            strides=strides,
            dilations=dilations,
            padding=padding,
            data_layout=data_layout,
            kernel_layout=kernel_layout,
        )
        func = relay.Function([x, w], y)

        op_res = relay.create_executor("graph", device=dev, target=target).evaluate(func)(
            indata, kernel
        )
        tvm.testing.assert_allclose(op_res.numpy(), out, rtol=1e-5, atol=1e-5)


@tvm.testing.uses_gpu
def test_affine_grid(executor_kind):
    def verify_affine_grid(num_batch, target_shape):
        dtype = "float32"
        data_shape = (num_batch, 2, 3)
        data = relay.var("data", relay.ty.TensorType(data_shape, dtype))
        y = relay.image.affine_grid(data, target_shape)
        yy = run_infer_type(y)
        assert yy.checked_type == relay.ty.TensorType(
            (num_batch, len(target_shape), *target_shape), dtype
        )

        func = relay.Function([data], y)
        data_np = np.random.uniform(size=data_shape).astype(dtype)
        ref_res = tvm.topi.testing.affine_grid_python(data_np, target_shape)

        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                data_np
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5, atol=1e-5)

    verify_affine_grid(1, (16, 32))
    verify_affine_grid(4, (16, 32))


@tvm.testing.uses_gpu
def test_grid_sample(executor_kind):
    def verify_grid_sample(
        data_shape, grid_shape, method="bilinear", padding_mode="zeros", align_corners=True
    ):
        dtype = "float32"
        data = relay.var("data", relay.ty.TensorType(data_shape, dtype))
        grid = relay.var("grid", relay.ty.TensorType(grid_shape, dtype))

        if len(data_shape) == 4:
            layout = "NCHW"
            batch, channel, _, _ = data_shape
            _, _, out_height, out_width = grid_shape
            tensor_type = relay.TensorType((batch, channel, out_height, out_width), dtype)
        else:  # len(data_shape) == 5:
            layout = "NCDHW"
            batch, channel, _, _, _ = data_shape
            _, _, out_depth, out_height, out_width = grid_shape
            tensor_type = relay.TensorType(
                (batch, channel, out_depth, out_height, out_width), dtype
            )

        y = relay.image.grid_sample(
            data,
            grid,
            method=method,
            layout=layout,
            padding_mode=padding_mode,
            align_corners=align_corners,
        )
        yy = run_infer_type(y)
        assert yy.checked_type == tensor_type
        func = relay.Function([data, grid], y)

        data_np = np.random.uniform(size=data_shape).astype(dtype)
        grid_np = np.random.uniform(size=grid_shape, low=-1.5, high=1.5).astype(dtype)
        ref_res = tvm.topi.testing.grid_sample_python(
            data_np, grid_np, method, layout, padding_mode, align_corners
        )

        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                data_np, grid_np
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-5, atol=1e-5)

    methods = ["nearest", "bilinear", "bicubic"]
    padding_modes = ["zeros", "border", "reflection"]
    align_corners = [True, False]

    data_2D_shape = (4, 4, 8, 8)
    grid_2D_shape = (4, 2, 16, 16)
    # choosing smaller sizes to be testable on weaker GPUs
    data_3D_shape = (4, 4, 4, 4, 4)
    grid_3D_shape = (4, 3, 8, 8, 8)

    for _method in methods:
        for _padding in padding_modes:
            for _align in align_corners:
                verify_grid_sample(data_2D_shape, grid_2D_shape, _method, _padding, _align)

                # 3D "bicubic"(tricubic) is not supported in pytorch
                if _method != "bicubic":
                    verify_grid_sample(data_3D_shape, grid_3D_shape, _method, _padding, _align)


@tvm.testing.uses_gpu
def test_space_to_batch_nd(executor_kind):
    def verify_space_to_batch_nd(dshape, block_shape, paddings):
        x_data = np.random.uniform(size=dshape).astype("float32")
        pad_before, pad_after = map(list, zip(*paddings))
        ref_res = tvm.topi.testing.space_to_batch_nd_python(
            x_data, block_shape, pad_before, pad_after
        )

        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.nn.space_to_batch_nd(x, block_shape, paddings)
        assert "block_shape=" in z.astext()
        assert "paddings=" in z.astext()
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType(ref_res.shape, "float32")
        func = relay.Function([x], z)

        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                x_data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-4)

    verify_space_to_batch_nd([3, 3, 2, 1], [3], [[0, 0]])
    verify_space_to_batch_nd([2, 2, 4, 1], [2, 2], [[0, 0], [2, 0]])


@tvm.testing.uses_gpu
def test_batch_to_space_nd(executor_kind):
    def verify_batch_to_space_nd(dshape, block_shape, crops):
        x_data = np.random.uniform(size=dshape).astype("float32")
        crop_begin_list, crop_end_list = map(list, zip(*crops))
        ref_res = tvm.topi.testing.batch_to_space_nd_python(
            x_data, block_shape, crop_begin_list, crop_end_list
        )

        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.nn.batch_to_space_nd(x, block_shape, crops)
        assert "block_shape=" in z.astext()
        assert "crops=" in z.astext()
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType(ref_res.shape, "float32")
        func = relay.Function([x], z)

        for target, dev in tvm.testing.enabled_targets():
            op_res = relay.create_executor(executor_kind, device=dev, target=target).evaluate(func)(
                x_data
            )
            tvm.testing.assert_allclose(op_res.numpy(), ref_res, rtol=1e-4)

    verify_batch_to_space_nd([4, 1, 1, 3], [2, 2], [[0, 0], [0, 0]])
    verify_batch_to_space_nd([8, 1, 3, 1], [2, 2], [[0, 0], [2, 0]])


@tvm.testing.uses_gpu
def test_all_class_non_max_suppression(executor_kind):
    def verify_all_class_non_max_suppression(
        boxes_np,
        scores_np,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        expected_indices,
    ):
        boxes = relay.var("boxes", relay.ty.TensorType(boxes_np.shape, "float32"))
        scores = relay.var("scores", relay.ty.TensorType(scores_np.shape, "float32"))

        out = relay.vision.all_class_non_max_suppression(
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        )

        func = relay.Function([boxes, scores], out.astuple())
        func = run_infer_type(func)

        for target, dev in tvm.testing.enabled_targets():
            selected_indices, num_detections = relay.create_executor(
                executor_kind, device=dev, target=target
            ).evaluate(func)(boxes_np, scores_np)
            tvm_res = selected_indices.numpy()[: num_detections.numpy()[0]]
            np.testing.assert_equal(tvm_res, expected_indices)

    boxes = np.array(
        [
            [
                [0.0, 0.0, 0.3, 0.3],
                [0.0, 0.0, 0.4, 0.4],
                [0.0, 0.0, 0.5, 0.5],
                [0.5, 0.5, 0.9, 0.9],
                [0.5, 0.5, 1.0, 1.0],
            ],
            [
                [0.0, 0.0, 0.3, 0.3],
                [0.0, 0.0, 0.4, 0.4],
                [0.5, 0.5, 0.95, 0.95],
                [0.5, 0.5, 0.96, 0.96],
                [0.5, 0.5, 1.0, 1.0],
            ],
        ]
    ).astype("float32")

    scores = np.array(
        [
            [[0.1, 0.2, 0.6, 0.3, 0.9], [0.1, 0.2, 0.6, 0.3, 0.9]],
            [[0.1, 0.2, 0.6, 0.3, 0.9], [0.1, 0.2, 0.6, 0.3, 0.9]],
        ]
    ).astype("float32")

    max_output_boxes_per_class = 2
    iou_threshold = 0.8
    score_threshold = 0.0

    expected = np.array(
        [[0, 0, 4], [0, 0, 2], [0, 1, 4], [0, 1, 2], [1, 0, 4], [1, 0, 1], [1, 1, 4], [1, 1, 1]]
    )

    verify_all_class_non_max_suppression(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, expected
    )

    boxes = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ]
        ]
    ).astype(np.float32)
    scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
    max_output_boxes_per_class = 3
    iou_threshold = 0.5
    score_threshold = 0.4

    expected = np.array([[0, 0, 3], [0, 0, 0]])

    verify_all_class_non_max_suppression(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, expected
    )


if __name__ == "__main__":
    tvm.testing.main()
