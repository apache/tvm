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
import numpy as np
import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_infer_type
import tvm.topi.testing
import tvm.testing


def test_resize_infer_type():
    n, c, h, w = te.size_var("n"), te.size_var("c"), te.size_var("h"), te.size_var("w")
    x = relay.var("x", relay.TensorType((n, c, h, w), "int8"))
    th, tw = te.var("th"), te.var("tw")
    z = relay.image.resize(x, (th, tw))
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((n, c, th, tw), "int8")

    x = relay.var("x", relay.TensorType((n, c, h, w), "int8"))
    z = relay.image.resize(x, (100, 200), "NCHW", "bilinear", "align_corners")
    assert "size=" in z.astext()
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((n, c, 100, 200), "int8")


@tvm.testing.uses_gpu
def test_resize():
    def verify_resize(dshape, scale, method, layout, coord_trans):
        if layout == "NHWC":
            size = (dshape[1] * scale, dshape[2] * scale)
        else:
            size = (dshape[2] * scale, dshape[3] * scale)

        x_data = np.random.uniform(size=dshape).astype("float32")

        if method == "bilinear":
            ref_res = tvm.topi.testing.bilinear_resize_python(x_data, size, layout, coord_trans)
        else:
            ref_res = tvm.topi.testing.upsampling_python(x_data, (scale, scale), layout)
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.image.resize(x, size, layout, method, coordinate_transformation_mode=coord_trans)
        assert "size=" in z.astext()
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType(ref_res.shape, "float32")
        func = relay.Function([x], z)

        for target, ctx in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-3, atol=1e-4)

    for method in ["nearest_neighbor", "bilinear"]:
        for coord_trans in ["asymmetric", "half_pixel", "align_corners"]:
            for layout in ["NHWC", "NCHW"]:
                # TODO: Topi test does not have a function to produce numpy output for resize with
                # nearest_neighbors and align_corners. Enable when topi test has this option
                if coord_trans == "align_corners" and method == "nearest_neighbor":
                    continue
                verify_resize((1, 4, 4, 4), 2, method, layout, coord_trans)
                verify_resize((2, 8, 17, 20), 3, method, layout, coord_trans)
                verify_resize((2, 8, 17, 20), 3, method, layout, coord_trans)
                verify_resize((3, 4, 5, 6), 5, method, layout, coord_trans)


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
    z = relay.image.resize3d(x, (10, 10, 20), "NCDHW", "trilinear", "align_corners")
    assert "size=" in z.astext()
    zz = run_infer_type(z)
    assert zz.checked_type == relay.TensorType((n, c, 10, 10, 20), "int8")


@tvm.testing.parametrize_targets
def test_resize3d(target, ctx):
    def verify_resize(dshape, scale, method, layout):
        if layout == "NDHWC":
            size = (dshape[1] * scale, dshape[2] * scale, dshape[3] * scale)
        else:
            size = (dshape[2] * scale, dshape[3] * scale, dshape[4] * scale)

        x_data = np.random.uniform(size=dshape).astype("float32")
        if method == "trilinear":
            ref_res = tvm.topi.testing.trilinear_resize3d_python(x_data, size, layout)
        else:
            ref_res = tvm.topi.testing.upsampling3d_python(x_data, (scale, scale, scale), layout)
        x = relay.var("x", relay.TensorType(dshape, "float32"))
        z = relay.image.resize3d(x, size, layout, method, "align_corners")
        assert "size=" in z.astext()
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType(ref_res.shape, "float32")
        func = relay.Function([x], z)

        for kind in ["graph", "debug"]:
            intrp = relay.create_executor(kind, ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(x_data)
            tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-4, atol=1e-6)

    for method in ["trilinear", "nearest_neighbor"]:
        for layout in ["NDHWC", "NCDHW"]:
            verify_resize((1, 4, 4, 4, 4), 2, method, layout)


@tvm.testing.uses_gpu
def test_crop_and_resize():
    def verify_crop_and_resize(
        img_shape, boxes, box_indices, crop_size, layout, method, extrapolation_value=0.0
    ):

        image_data = np.random.uniform(size=img_shape).astype("float32")

        ref_res = tvm.topi.testing.crop_and_resize_python(
            image_data, boxes, box_indices, crop_size, layout, method, extrapolation_value
        )

        img = relay.var("img", relay.TensorType(img_shape, "float32"))
        bx = relay.var("bx", relay.TensorType(boxes.shape, "float32"))
        bx_idx = relay.var("bx_idx", relay.TensorType(box_indices.shape, "int32"))

        z = relay.image.crop_and_resize(
            img, bx, bx_idx, list(crop_size), layout, method, extrapolation_value
        )
        zz = run_infer_type(z)
        assert zz.checked_type == relay.TensorType(ref_res.shape, "float32")
        func = relay.Function([img, bx, bx_idx], z)

        for target, ctx in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(image_data, boxes, box_indices)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-3, atol=1e-04)

    boxes_nhwc = np.array([[0.1, 0.2, 0.8, 0.7], [0.2, 0, 1, 0.6]]).astype("float32")
    indices_nhwc = np.array([1, 0]).astype("int32")
    size_nhwc = np.array([20, 30]).astype("int32")
    boxes_nchw = np.array([[0, 0, 1, 1], [0.2, 0.1, 1, 0.9]]).astype("float32")
    indices_nchw = np.array([0, 1]).astype("int32")
    size_nchw = np.array([30, 30]).astype("int32")

    for method in ["bilinear", "nearest_neighbor"]:
        verify_crop_and_resize(
            (10, 224, 224, 3), boxes_nhwc, indices_nhwc, size_nhwc, "NHWC", method
        )
        verify_crop_and_resize(
            (5, 3, 255, 255), boxes_nchw, indices_nchw, size_nchw, "NCHW", method, 0.1
        )


@tvm.testing.uses_gpu
def test_multibox_prior():
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
        for target, ctx in tvm.testing.enabled_targets():
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            op_res1 = intrp1.evaluate(func)(data)
            tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5)
            intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
            op_res2 = intrp2.evaluate(func)(data)
            tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-5)

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
        for target, ctx in tvm.testing.enabled_targets():
            intrp = relay.create_executor("debug", ctx=ctx, target=target)
            out = intrp.evaluate(func)(np_data)

            tvm.testing.assert_allclose(out[0].asnumpy(), np_out1, rtol=1e-3, atol=1e-04)
            tvm.testing.assert_allclose(out[1].asnumpy(), np_out2, rtol=1e-3, atol=1e-04)
            tvm.testing.assert_allclose(out[2].asnumpy(), np_out3, rtol=1e-3, atol=1e-04)

    verify_get_valid_counts((1, 2500, 6), 0, 0, 1)
    verify_get_valid_counts((1, 2500, 5), -1, -1, 0)
    verify_get_valid_counts((3, 1000, 6), 0.55, 1, 0)
    verify_get_valid_counts((16, 500, 5), 0.95, -1, 0)


@tvm.testing.uses_gpu
def test_non_max_suppression():
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
        assert "iou_threshold" in z.astext()
        assert "iou_threshold" in z_indices.astext()
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
        for target, ctx in tvm.testing.enabled_targets():
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            op_res1 = intrp1.evaluate(func)(x0_data, x1_data, x2_data, x3_data)
            tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5)
            intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
            op_res2 = intrp2.evaluate(func)(x0_data, x1_data, x2_data, x3_data)
            tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-5)
            op_indices_res1 = intrp1.evaluate(func_indices)(x0_data, x1_data, x2_data, x3_data)
            tvm.testing.assert_allclose(op_indices_res1[0].asnumpy(), ref_indices_res, rtol=1e-5)
            op_indices_res2 = intrp2.evaluate(func_indices)(x0_data, x1_data, x2_data, x3_data)
            tvm.testing.assert_allclose(op_indices_res2[0].asnumpy(), ref_indices_res, rtol=1e-5)

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
def test_multibox_transform_loc():
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
        for target, ctx in tvm.testing.enabled_targets():
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            op_res1 = intrp1.evaluate(func)(np_cls_prob, np_loc_preds, np_anchors)
            tvm.testing.assert_allclose(op_res1.asnumpy(), expected_np_out, rtol=1e-5)
            intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
            op_res2 = intrp2.evaluate(func)(np_cls_prob, np_loc_preds, np_anchors)
            tvm.testing.assert_allclose(op_res2.asnumpy(), expected_np_out, rtol=1e-5)

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
def test_roi_align():
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
        for target, ctx in tvm.testing.enabled_targets():
            print("test on", target)
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            op_res1 = intrp1.evaluate(func)(np_data, np_rois)
            tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-4)
            intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
            op_res2 = intrp2.evaluate(func)(np_data, np_rois)
            tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-4)

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
def test_roi_pool():
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
        for target, ctx in tvm.testing.enabled_targets():
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            op_res1 = intrp1.evaluate(func)(np_data, np_rois)
            tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-4)
            intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
            op_res2 = intrp2.evaluate(func)(np_data, np_rois)
            tvm.testing.assert_allclose(op_res2.asnumpy(), ref_res, rtol=1e-4)

    verify_roi_pool((1, 4, 16, 16), (32, 5), pooled_size=7, spatial_scale=1.0)
    verify_roi_pool((4, 4, 16, 16), (32, 5), pooled_size=7, spatial_scale=0.5)


@tvm.testing.uses_gpu
def test_proposal():
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
            ctx = tvm.context(target, 0)
            intrp1 = relay.create_executor("graph", ctx=ctx, target=target)
            op_res1 = intrp1.evaluate(func)(np_cls_prob, np_bbox_pred, np_im_info)
            tvm.testing.assert_allclose(op_res1.asnumpy(), np_out, rtol=1e-4)
            intrp2 = relay.create_executor("debug", ctx=ctx, target=target)
            op_res2 = intrp2.evaluate(func)(np_cls_prob, np_bbox_pred, np_im_info)
            tvm.testing.assert_allclose(op_res2.asnumpy(), np_out, rtol=1e-4)

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
def test_yolo_reorg():
    def verify_yolo_reorg(shape, stride):
        x_data = np.random.uniform(low=-1, high=1, size=shape).astype("float32")
        ref_res = tvm.topi.testing.reorg_python(x_data, stride)

        x = relay.var("x", relay.TensorType(shape, "float32"))
        z = relay.vision.yolo_reorg(x, stride=stride)
        zz = run_infer_type(z)
        assert "stride=" in z.astext()
        assert zz.checked_type == relay.ty.TensorType(ref_res.shape, "float32")

        func = relay.Function([x], z)

        for target, ctx in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-5)

    verify_yolo_reorg((1, 100, 20, 20), 10)
    verify_yolo_reorg((1, 4, 6, 6), 2)


@tvm.testing.uses_gpu
def test_deformable_conv2d():
    def test_infer_type(batch, in_channel, size, out_channel, deformable_groups, groups, layout):
        kernel_size = (3, 3)
        if layout == "NCHW":
            kernel_layout = "OIHW"
            data_shape = (batch, in_channel, size, size)
            weight_shape = (out_channel, in_channel // groups, kernel_size[0], kernel_size[1])
            out_shape = (batch, out_channel, size, size)
            offset_shape = (
                batch,
                2 * kernel_size[0] * kernel_size[1] * deformable_groups,
                out_shape[2],
                out_shape[3],
            )
        else:
            kernel_layout = "HWIO"
            data_shape = (batch, size, size, in_channel)
            weight_shape = (kernel_size[0], kernel_size[1], in_channel // groups, out_channel)
            out_shape = (batch, size, size, out_channel)
            offset_shape = (
                batch,
                out_shape[1],
                out_shape[2],
                2 * kernel_size[0] * kernel_size[1] * deformable_groups,
            )

        data = relay.var("data", shape=data_shape)
        offset = relay.var("offset")
        kernel = relay.var("kernel")
        y = relay.nn.deformable_conv2d(
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
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType(out_shape), yy.checked_type
        assert yy.args[1].checked_type == relay.TensorType(offset_shape), yy.args[1].checked_type
        assert yy.args[2].checked_type == relay.TensorType(weight_shape), yy.args[2].checked_type

    test_infer_type(1, 4, 16, 4, 4, 1, "NCHW")
    test_infer_type(2, 4, 16, 4, 1, 2, "NCHW")
    test_infer_type(1, 4, 16, 4, 4, 1, "NHWC")
    test_infer_type(2, 4, 16, 4, 1, 2, "NHWC")

    def test_run(batch, in_channel, size, out_channel, deformable_groups, groups, layout):
        kernel_size = (3, 3)
        if layout == "NCHW":
            kernel_layout = "OIHW"
            data_shape = (batch, in_channel, size, size)
            kernel_shape = (out_channel, in_channel // groups, kernel_size[0], kernel_size[1])
            out_shape = (batch, out_channel, size, size)
            offset_shape = (
                batch,
                2 * kernel_size[0] * kernel_size[1] * deformable_groups,
                out_shape[2],
                out_shape[3],
            )
        else:
            kernel_layout = "HWIO"
            data_shape = (batch, size, size, in_channel)
            kernel_shape = (kernel_size[0], kernel_size[1], in_channel // groups, out_channel)
            out_shape = (batch, size, size, out_channel)
            offset_shape = (
                batch,
                out_shape[1],
                out_shape[2],
                2 * kernel_size[0] * kernel_size[1] * deformable_groups,
            )

        dtype = "float32"
        data = relay.var("data", shape=data_shape, dtype=dtype)
        offset = relay.var("offset")
        kernel = relay.var("kernel")
        y = relay.nn.deformable_conv2d(
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
        func = relay.Function([data, offset, kernel], y)
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
        for target, ctx in tvm.testing.enabled_targets():
            if target == "cuda" and layout == "NHWC":
                continue  # Cannot run NHWC layout on cuda target, only on llvm
            for kind in ["graph", "debug"]:
                intrp1 = relay.create_executor(kind, ctx=ctx, target=target)
                op_res1 = intrp1.evaluate(func)(data, offset, kernel)
                tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5, atol=1e-5)

    test_run(1, 4, 16, 4, 1, 1, "NCHW")
    test_run(1, 4, 16, 4, 1, 1, "NHWC")
    test_run(2, 4, 16, 4, 4, 1, "NCHW")
    test_run(2, 4, 16, 4, 4, 1, "NHWC")


@tvm.testing.uses_gpu
def test_depth_to_space():
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

        for target, ctx in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-4)

    for layout in ["NHWC", "NCHW"]:
        for mode in ["DCR", "CDR"]:
            verify_depth_to_space((1, 4, 4, 4), 2, layout, mode)


@tvm.testing.uses_gpu
def test_space_to_depth():
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

        for target, ctx in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-4)

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


@tvm.testing.uses_gpu
def test_dilation2d_run():
    def run_test_dilation2d(
        indata,
        kernel,
        out,
        dtype="float32",
        strides=[1, 1],
        padding=[0, 0],
        dilations=[1, 1],
        except_targets=["cuda"],
        **attrs,
    ):

        dshape = indata.shape
        kshape = kernel.shape

        if except_targets is None:
            except_targets = []

        x = relay.var("x", shape=dshape, dtype=dtype)
        w = relay.var("w", shape=kshape, dtype=dtype)
        y = relay.image.dilation2d(
            x, w, strides=strides, dilations=dilations, padding=padding, **attrs
        )
        func = relay.Function([x, w], y)

        for target, ctx in tvm.testing.enabled_targets():
            if target in except_targets:
                continue
            intrp = relay.create_executor("graph", ctx=ctx, target=target)
            op_res = intrp.evaluate(func)(indata, kernel)
            tvm.testing.assert_allclose(op_res.asnumpy(), out, rtol=1e-5, atol=1e-5)

    def _convert_data(indata, kernel, out, layout=None):
        indata = np.asarray(indata)
        kernel = np.asarray(kernel)
        out = np.asarray(out)
        if layout == "NCHW":
            indata = indata.transpose([0, 3, 1, 2])
            kernel = kernel.transpose([2, 0, 1])
            out = out.transpose([0, 3, 1, 2])
        return indata, kernel, out

    image = [[[[0.1], [0.2]], [[0.3], [0.4]]]]
    kernel = [[[0.4], [0.3]], [[0.1], [0.0]]]
    out = [[[[0.5]]]]
    run_test_dilation2d(*_convert_data(image, kernel, out, layout="NCHW"))
    run_test_dilation2d(*_convert_data(image, kernel, out), data_layout="NHWC", kernel_layout="HWI")

    image = [[[[0.1], [0.2]], [[0.3], [0.4]]]]
    kernel = [[[0.4], [0.3]], [[0.1], [0.0]]]
    out = [[[[0.5], [0.6]], [[0.7], [0.8]]]]
    run_test_dilation2d(*_convert_data(image, kernel, out, layout="NCHW"), padding=[0, 0, 1, 1])
    run_test_dilation2d(
        *_convert_data(image, kernel, out),
        padding=[0, 0, 1, 1],
        data_layout="NHWC",
        kernel_layout="HWI",
    )

    image = [[[[0.1, 0.2, 0.0], [0.2, 0.3, 0.1]], [[0.3, 0.4, 0.2], [0.4, 0.5, 0.3]]]]
    kernel = [[[0.4, 0.5, 0.3], [0.3, 0.4, 0.2]], [[0.1, 0.2, 0.0], [0.0, 0.1, -0.1]]]
    out = [[[[0.5, 0.7, 0.3], [0.6, 0.8, 0.4]], [[0.7, 0.9, 0.5], [0.8, 1.0, 0.6]]]]
    run_test_dilation2d(*_convert_data(image, kernel, out, layout="NCHW"), padding=[0, 0, 1, 1])
    run_test_dilation2d(
        *_convert_data(image, kernel, out),
        padding=[0, 0, 1, 1],
        data_layout="NHWC",
        kernel_layout="HWI",
    )

    image = [[[[0.1], [0.2]], [[0.3], [0.4]]], [[[0.2], [0.3]], [[0.4], [0.5]]]]
    kernel = [[[0.4], [0.3]], [[0.1], [0.0]]]
    out = [[[[0.5], [0.6]], [[0.7], [0.8]]], [[[0.6], [0.7]], [[0.8], [0.9]]]]
    run_test_dilation2d(*_convert_data(image, kernel, out, layout="NCHW"), padding=[0, 0, 1, 1])
    run_test_dilation2d(
        *_convert_data(image, kernel, out),
        padding=[0, 0, 1, 1],
        data_layout="NHWC",
        kernel_layout="HWI",
    )

    image = [[[[0.1], [0.2]], [[0.3], [0.4]]]]
    kernel = [[[0.4], [0.3]]]
    out = [[[[0.5]], [[0.7]]]]
    run_test_dilation2d(*_convert_data(image, kernel, out, layout="NCHW"))
    run_test_dilation2d(*_convert_data(image, kernel, out), data_layout="NHWC", kernel_layout="HWI")

    image = [[[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]], [[0.7], [0.8], [0.9]]]]
    kernel = [[[0.4], [0.3]], [[0.1], [0.2]]]
    out = [[[[0.7], [0.8], [0.6]], [[1.0], [1.1], [0.9]], [[0.8], [0.9], [0.9]]]]
    run_test_dilation2d(
        *_convert_data(image, kernel, out, layout="NCHW"), padding=[1, 1], dilations=[2, 2]
    )
    run_test_dilation2d(
        *_convert_data(image, kernel, out),
        padding=[1, 1],
        dilations=[2, 2],
        data_layout="NHWC",
        kernel_layout="HWI",
    )

    image = [
        [[[0.1], [0.2], [0.3], [0.4]], [[0.5], [0.6], [0.7], [0.8]], [[0.9], [1.0], [1.1], [1.2]]]
    ]
    kernel = [[[0.4], [0.3]], [[0.1], [0.2]]]
    out = [[[[0.8], [1.0]], [[1.2], [1.4]]]]
    run_test_dilation2d(*_convert_data(image, kernel, out, layout="NCHW"), strides=[1, 2])
    run_test_dilation2d(
        *_convert_data(image, kernel, out), strides=[1, 2], data_layout="NHWC", kernel_layout="HWI"
    )


@tvm.testing.uses_gpu
def test_affine_grid():
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

        for target, ctx in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp1 = relay.create_executor(kind, ctx=ctx, target=target)
                op_res1 = intrp1.evaluate(func)(data_np)
                tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5, atol=1e-5)

    verify_affine_grid(1, (16, 32))
    verify_affine_grid(4, (16, 32))


@tvm.testing.uses_gpu
def test_grid_sample():
    def verify_grid_sample(data_shape, grid_shape):
        dtype = "float32"
        batch, channel, _, _ = data_shape
        _, _, out_height, out_width = grid_shape
        data = relay.var("data", relay.ty.TensorType(data_shape, dtype))
        grid = relay.var("grid", relay.ty.TensorType(grid_shape, dtype))
        y = relay.image.grid_sample(data, grid, method="bilinear", layout="NCHW")
        yy = run_infer_type(y)
        assert yy.checked_type == relay.TensorType((batch, channel, out_height, out_width), dtype)
        func = relay.Function([data, grid], y)

        data_np = np.random.uniform(size=data_shape).astype(dtype)
        grid_np = np.random.uniform(size=grid_shape, low=-1.5, high=1.5).astype(dtype)
        ref_res = tvm.topi.testing.grid_sample_nchw_python(data_np, grid_np, method="bilinear")

        for target, ctx in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp1 = relay.create_executor(kind, ctx=ctx, target=target)
                op_res1 = intrp1.evaluate(func)(data_np, grid_np)
                tvm.testing.assert_allclose(op_res1.asnumpy(), ref_res, rtol=1e-5, atol=1e-5)

    verify_grid_sample((4, 4, 16, 32), (4, 2, 8, 8))
    verify_grid_sample((4, 4, 16, 32), (4, 2, 32, 32))


@tvm.testing.uses_gpu
def test_space_to_batch_nd():
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

        for target, ctx in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-4)

    verify_space_to_batch_nd([3, 3, 2, 1], [3], [[0, 0]])
    verify_space_to_batch_nd([2, 2, 4, 1], [2, 2], [[0, 0], [2, 0]])


@tvm.testing.uses_gpu
def test_batch_to_space_nd():
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

        for target, ctx in tvm.testing.enabled_targets():
            for kind in ["graph", "debug"]:
                intrp = relay.create_executor(kind, ctx=ctx, target=target)
                op_res = intrp.evaluate(func)(x_data)
                tvm.testing.assert_allclose(op_res.asnumpy(), ref_res, rtol=1e-4)

    verify_batch_to_space_nd([4, 1, 1, 3], [2, 2], [[0, 0], [0, 0]])
    verify_batch_to_space_nd([8, 1, 3, 1], [2, 2], [[0, 0], [2, 0]])


if __name__ == "__main__":
    test_resize_infer_type()
    test_resize()
    test_resize3d_infer_type()
    test_crop_and_resize()
    test_multibox_prior()
    test_multibox_transform_loc()
    test_get_valid_counts()
    test_roi_align()
    test_roi_pool()
    test_proposal()
    test_yolo_reorg_infer_shape()
    test_yolo_reorg()
    test_non_max_suppression()
    test_deformable_conv2d()
    test_depth_to_space()
    test_space_to_depth()
    test_dilation2d_infer_type()
    test_dilation2d_run()
    test_affine_grid()
    test_grid_sample()
    test_space_to_batch_nd()
