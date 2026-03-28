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

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import TVMError, relax, tirx
from tvm.ir import Op
from tvm.relax.transform import LegalizeOps
from tvm.script import relax as R


def _check_inference(bb: relax.BlockBuilder, call: relax.Call, expected_sinfo: relax.StructInfo):
    ret = bb.normalize(call)
    tvm.ir.assert_structural_equal(ret.struct_info, expected_sinfo)


def test_roi_align_op_correctness():
    x = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    rois = relax.Var("rois", R.Tensor((4, 5), "float32"))
    assert relax.op.vision.roi_align(x, rois, (7, 7), 1.0).op == Op.get("relax.vision.roi_align")


def test_roi_align_infer_struct_info():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 32, 32, 3), "float32"))
    rois = relax.Var("rois", R.Tensor((5, 5), "float32"))

    _check_inference(
        bb,
        relax.op.vision.roi_align(x0, rois, (7, 7), 0.25),
        relax.TensorStructInfo((5, 3, 7, 7), "float32"),
    )
    _check_inference(
        bb,
        relax.op.vision.roi_align(x1, rois, (5, 7), 1.0, layout="NHWC"),
        relax.TensorStructInfo((5, 5, 7, 3), "float32"),
    )


def test_roi_align_infer_struct_info_aligned():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    rois = relax.Var("rois", R.Tensor((5, 5), "float32"))

    _check_inference(
        bb,
        relax.op.vision.roi_align(x, rois, (7, 7), 1.0, aligned=True),
        relax.TensorStructInfo((5, 3, 7, 7), "float32"),
    )


def test_roi_align_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    c = tirx.Var("c", "int64")
    h = tirx.Var("h", "int64")
    w = tirx.Var("w", "int64")
    num_roi = tirx.Var("num_roi", "int64")

    x = relax.Var("x", R.Tensor((n, c, h, w), "float32"))
    rois = relax.Var("rois", R.Tensor((num_roi, 5), "float32"))

    _check_inference(
        bb,
        relax.op.vision.roi_align(x, rois, (7, 7), 0.5),
        relax.TensorStructInfo((num_roi, c, 7, 7), "float32"),
    )


def test_roi_align_wrong_input_ndim():
    bb = relax.BlockBuilder()
    x0 = relax.Var("x", R.Tensor((2, 3, 32), "float32"))
    x1 = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    rois0 = relax.Var("rois", R.Tensor((4,), "float32"))
    rois1 = relax.Var("rois", R.Tensor((4, 5), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.roi_align(x0, rois1, (7, 7), 1.0))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.roi_align(x1, rois0, (7, 7), 1.0))


def test_roi_align_wrong_rois_last_dim():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    rois = relax.Var("rois", R.Tensor((4, 4), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.roi_align(x, rois, (7, 7), 1.0))


def test_roi_align_wrong_layout():
    bb = relax.BlockBuilder()
    x = relax.Var("x", R.Tensor((2, 3, 32, 32), "float32"))
    rois = relax.Var("rois", R.Tensor((4, 5), "float32"))

    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.roi_align(x, rois, (7, 7), 1.0, layout="HWCN"))


def test_roi_align_legalize():
    @tvm.script.ir_module
    class ROIAlign:
        @R.function
        def main(
            x: R.Tensor((1, 2, 8, 8), "float32"),
            rois: R.Tensor((2, 5), "float32"),
        ) -> R.Tensor((2, 2, 3, 3), "float32"):
            gv: R.Tensor((2, 2, 3, 3), "float32") = R.vision.roi_align(
                x,
                rois,
                pooled_size=(3, 3),
                spatial_scale=1.0,
                sample_ratio=2,
                layout="NCHW",
                mode="avg",
            )
            return gv

    mod = LegalizeOps()(ROIAlign)
    assert "call_tir" in str(mod)
    tvm.ir.assert_structural_equal(
        mod["main"].ret_struct_info,
        relax.TensorStructInfo((2, 2, 3, 3), "float32"),
    )


def test_roi_align_legalize_aligned():
    @tvm.script.ir_module
    class ROIAlign:
        @R.function
        def main(
            x: R.Tensor((1, 1, 4, 4), "float32"),
            rois: R.Tensor((1, 5), "float32"),
        ) -> R.Tensor((1, 1, 1, 1), "float32"):
            gv: R.Tensor((1, 1, 1, 1), "float32") = R.vision.roi_align(
                x,
                rois,
                pooled_size=(1, 1),
                spatial_scale=1.0,
                sample_ratio=2,
                aligned=True,
                layout="NCHW",
                mode="avg",
            )
            return gv

    mod = LegalizeOps()(ROIAlign)
    assert "call_tir" in str(mod)
    tvm.ir.assert_structural_equal(
        mod["main"].ret_struct_info,
        relax.TensorStructInfo((1, 1, 1, 1), "float32"),
    )


def test_roi_align_legalize_sample_ratio_zero():
    @tvm.script.ir_module
    class ROIAlign:
        @R.function
        def main(
            x: R.Tensor((1, 2, 8, 8), "float32"),
            rois: R.Tensor((1, 5), "float32"),
        ) -> R.Tensor((1, 2, 2, 2), "float32"):
            gv: R.Tensor((1, 2, 2, 2), "float32") = R.vision.roi_align(
                x,
                rois,
                pooled_size=(2, 2),
                spatial_scale=1.0,
                sample_ratio=0,
                layout="NCHW",
                mode="avg",
            )
            return gv

    mod = LegalizeOps()(ROIAlign)
    assert "call_tir" in str(mod)
    tvm.ir.assert_structural_equal(
        mod["main"].ret_struct_info,
        relax.TensorStructInfo((1, 2, 2, 2), "float32"),
    )


def test_all_class_non_max_suppression_infer_struct_info():
    bb = relax.BlockBuilder()
    batch_size, num_classes, num_boxes = 10, 8, 5
    boxes = relax.Var("boxes", R.Tensor((batch_size, num_boxes, 4), "float32"))
    scores = relax.Var("scores", R.Tensor((batch_size, num_classes, num_boxes), "float32"))
    max_output_boxes_per_class = relax.const(10, "int64")
    iou_threshold = relax.const(0.5, "float32")
    score_threshold = relax.const(0.1, "float32")

    _check_inference(
        bb,
        relax.op.vision.all_class_non_max_suppression(
            boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, "onnx"
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((batch_size * num_classes * num_boxes, 3), "int64"),
                relax.TensorStructInfo((1,), "int64"),
            ]
        ),
    )


def test_all_class_non_max_suppression_wrong_input_number():
    boxes = relax.Var("boxes", R.Tensor((1, 5, 4), "float32"))
    scores = relax.Var("scores", R.Tensor((1, 3, 5), "float32"))

    with pytest.raises(TVMError):
        relax.op.vision.all_class_non_max_suppression(boxes, scores)


def test_all_class_non_max_suppression_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    batch_size = tirx.Var("batch_size", "int64")
    num_classes = tirx.Var("num_classes", "int64")
    num_boxes = tirx.Var("num_boxes", "int64")
    boxes = relax.Var("boxes", R.Tensor((batch_size, num_boxes, 4), "float32"))
    scores = relax.Var("scores", R.Tensor((batch_size, num_classes, num_boxes), "float32"))
    max_output_boxes_per_class = relax.const(10, "int64")
    iou_threshold = relax.const(0.5, "float32")
    score_threshold = relax.const(0.1, "float32")

    _check_inference(
        bb,
        relax.op.vision.all_class_non_max_suppression(
            boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, "onnx"
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((batch_size * num_classes * num_boxes, 3), "int64"),
                relax.TensorStructInfo((1,), "int64"),
            ]
        ),
    )


def test_all_class_non_max_suppression_legalize_dynamic_trim():
    @tvm.script.ir_module
    class NMSModule:
        @R.function
        def main(
            boxes: R.Tensor((1, 5, 4), "float32"),
            scores: R.Tensor((1, 2, 5), "float32"),
        ) -> R.Tuple(R.Tensor(dtype="int64", ndim=2), R.Tensor((1,), "int64")):
            max_output_boxes_per_class = R.const(3, "int64")
            iou_threshold = R.const(0.5, "float32")
            score_threshold = R.const(0.1, "float32")
            return R.vision.all_class_non_max_suppression(
                boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, "onnx"
            )

    mod = LegalizeOps()(NMSModule)

    # Check legalized function has dynamic output (uses dynamic_strided_slice)
    assert "dynamic_strided_slice" in str(mod)

    ret_sinfo = mod["main"].ret_struct_info
    tvm.ir.assert_structural_equal(
        ret_sinfo,
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(ndim=2, dtype="int64"),
                relax.TensorStructInfo((1,), "int64"),
            ]
        ),
    )


@tvm.testing.requires_llvm
def test_all_class_non_max_suppression_legalize_e2e():
    @tvm.script.ir_module
    class NMSModule:
        @R.function
        def main(
            boxes: R.Tensor((1, 5, 4), "float32"),
            scores: R.Tensor((1, 2, 5), "float32"),
        ) -> R.Tuple(R.Tensor(dtype="int64", ndim=2), R.Tensor((1,), "int64")):
            max_output_boxes_per_class = R.const(3, "int64")
            iou_threshold = R.const(0.5, "float32")
            score_threshold = R.const(0.1, "float32")
            return R.vision.all_class_non_max_suppression(
                boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, "onnx"
            )

    boxes_data = np.array(
        [
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.1, 0.1, 1.1, 1.1],
                [2.0, 2.0, 3.0, 3.0],
                [4.0, 4.0, 5.0, 5.0],
                [6.0, 6.0, 7.0, 7.0],
            ]
        ],
        dtype=np.float32,
    )
    scores_data = np.array(
        [[[0.9, 0.8, 0.7, 0.6, 0.5], [0.85, 0.75, 0.65, 0.55, 0.45]]],
        dtype=np.float32,
    )

    mod = LegalizeOps()(NMSModule)

    # Check struct info
    tvm.ir.assert_structural_equal(
        mod["main"].ret_struct_info,
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo(ndim=2, dtype="int64"),
                relax.TensorStructInfo((1,), "int64"),
            ]
        ),
    )

    # Check runtime execution
    exe = tvm.compile(mod, target="llvm")
    vm = relax.VirtualMachine(exe, tvm.cpu())
    result = vm["main"](
        tvm.runtime.tensor(boxes_data, tvm.cpu()),
        tvm.runtime.tensor(scores_data, tvm.cpu()),
    )

    selected_indices = result[0].numpy()
    num_total_detections = int(result[1].numpy()[0])
    tvm.testing.assert_allclose(selected_indices.shape, (num_total_detections, 3))


def test_multibox_transform_loc_op_correctness():
    cls = relax.Var("cls", R.Tensor((1, 5, 10), "float32"))
    loc = relax.Var("loc", R.Tensor((1, 40), "float32"))
    anc = relax.Var("anc", R.Tensor((1, 10, 4), "float32"))
    assert (
        relax.op.vision.multibox_transform_loc(
            cls, loc, anc, False, 0.0, (1.0, 1.0, 1.0, 1.0), True
        ).op
        == Op.get("relax.vision.multibox_transform_loc")
    )


def test_multibox_transform_loc_infer_struct_info():
    bb = relax.BlockBuilder()
    cls = relax.Var("cls", R.Tensor((2, 3, 5), "float32"))
    loc = relax.Var("loc", R.Tensor((2, 20), "float32"))
    anc = relax.Var("anc", R.Tensor((1, 5, 4), "float32"))
    _check_inference(
        bb,
        relax.op.vision.multibox_transform_loc(
            cls, loc, anc, False, 0.0, (0.1, 0.1, 0.2, 0.2), True
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 5, 4), "float32"),
                relax.TensorStructInfo((2, 3, 5), "float32"),
            ]
        ),
    )


def test_multibox_transform_loc_wrong_cls_ndim():
    bb = relax.BlockBuilder()
    cls = relax.Var("cls", R.Tensor((2, 3), "float32"))
    loc = relax.Var("loc", R.Tensor((2, 20), "float32"))
    anc = relax.Var("anc", R.Tensor((1, 5, 4), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.multibox_transform_loc(cls, loc, anc))


def test_multibox_transform_loc_wrong_shape_relation():
    bb = relax.BlockBuilder()
    cls = relax.Var("cls", R.Tensor((2, 3, 5), "float32"))
    anc = relax.Var("anc", R.Tensor((1, 5, 4), "float32"))
    loc_bad_div = relax.Var("loc_bad_div", R.Tensor((2, 19), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.multibox_transform_loc(cls, loc_bad_div, anc))
    # Divisible by 4 but loc_dim != 4*N (N=5 -> expect 20, not 24)
    loc_bad_n = relax.Var("loc_bad_n", R.Tensor((2, 24), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.multibox_transform_loc(cls, loc_bad_n, anc))


def test_multibox_transform_loc_wrong_anchor_shape():
    bb = relax.BlockBuilder()
    cls = relax.Var("cls", R.Tensor((2, 3, 5), "float32"))
    loc = relax.Var("loc", R.Tensor((2, 20), "float32"))
    anc_bad_batch = relax.Var("anc_bad_batch", R.Tensor((2, 5, 4), "float32"))
    anc_bad_last = relax.Var("anc_bad_last", R.Tensor((1, 5, 5), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.multibox_transform_loc(cls, loc, anc_bad_batch))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.multibox_transform_loc(cls, loc, anc_bad_last))


def test_multibox_transform_loc_wrong_dtype():
    bb = relax.BlockBuilder()
    cls = relax.Var("cls", R.Tensor((2, 3, 5), "float32"))
    loc = relax.Var("loc", R.Tensor((2, 20), "float16"))
    anc = relax.Var("anc", R.Tensor((1, 5, 4), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.multibox_transform_loc(cls, loc, anc))


def test_multibox_transform_loc_wrong_batch():
    bb = relax.BlockBuilder()
    cls = relax.Var("cls", R.Tensor((2, 3, 5), "float32"))
    loc = relax.Var("loc", R.Tensor((1, 20), "float32"))
    anc = relax.Var("anc", R.Tensor((1, 5, 4), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.multibox_transform_loc(cls, loc, anc))


def _multibox_ref_numpy(
    cls_pred, loc_pred, anchor, variances, clip=False, threshold=0.0, keep_background=True
):
    """Numpy reference aligned with ``topi.vision.multibox_transform_loc``."""

    def _softmax(x, axis):
        x_max = np.max(x, axis=axis, keepdims=True)
        exp = np.exp(x - x_max)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    B, C, N = cls_pred.shape
    loc = loc_pred.reshape(B, N, 4)
    scores = _softmax(cls_pred.astype("float64"), axis=1).astype(np.float32)
    if threshold > 0.0:
        scores = np.where(scores >= threshold, scores, 0.0).astype(np.float32)
    if not keep_background:
        scores = scores.copy()
        scores[:, 0, :] = 0.0
    vx, vy, vw, vh = variances
    boxes = np.zeros((B, N, 4), dtype=np.float32)
    for b in range(B):
        for a in range(N):
            l, t, r, br = anchor[0, a, :]
            ay = (t + br) * 0.5
            ax = (l + r) * 0.5
            ah = br - t
            aw = r - l
            ex, ey, ew, eh = loc[b, a, :]
            ycenter = ey * vy * ah + ay
            xcenter = ex * vx * aw + ax
            half_h = 0.5 * np.exp(eh * vh) * ah
            half_w = 0.5 * np.exp(ew * vw) * aw
            ymin = ycenter - half_h
            xmin = xcenter - half_w
            ymax = ycenter + half_h
            xmax = xcenter + half_w
            if clip:
                ymin = np.clip(ymin, 0.0, 1.0)
                xmin = np.clip(xmin, 0.0, 1.0)
                ymax = np.clip(ymax, 0.0, 1.0)
                xmax = np.clip(xmax, 0.0, 1.0)
            boxes[b, a, :] = (ymin, xmin, ymax, xmax)
    return boxes, scores


@tvm.testing.requires_llvm
def test_multibox_transform_loc_legalize_e2e():
    @tvm.script.ir_module
    class Mod:
        @R.function
        def main(
            cls: R.Tensor((1, 3, 5), "float32"),
            loc: R.Tensor((1, 20), "float32"),
            anc: R.Tensor((1, 5, 4), "float32"),
        ) -> R.Tuple(R.Tensor((1, 5, 4), "float32"), R.Tensor((1, 3, 5), "float32")):
            return R.vision.multibox_transform_loc(
                cls,
                loc,
                anc,
                clip=False,
                threshold=0.0,
                variances=(1.0, 1.0, 1.0, 1.0),
                keep_background=True,
            )

    cls_data = np.random.randn(1, 3, 5).astype(np.float32)
    loc_data = np.random.randn(1, 20).astype(np.float32) * 0.05
    anc_data = np.array(
        [
            [
                [0.1, 0.1, 0.5, 0.5],
                [0.2, 0.2, 0.6, 0.6],
                [0.0, 0.0, 1.0, 1.0],
                [0.3, 0.3, 0.7, 0.7],
                [0.05, 0.05, 0.45, 0.45],
            ]
        ],
        dtype=np.float32,
    )

    mod = LegalizeOps()(Mod)
    exe = tvm.compile(mod, target="llvm")
    vm = relax.VirtualMachine(exe, tvm.cpu())
    ref_b, ref_s = _multibox_ref_numpy(cls_data, loc_data, anc_data, (1.0, 1.0, 1.0, 1.0))
    out = vm["main"](
        tvm.runtime.tensor(cls_data, tvm.cpu()),
        tvm.runtime.tensor(loc_data, tvm.cpu()),
        tvm.runtime.tensor(anc_data, tvm.cpu()),
    )
    tvm.testing.assert_allclose(out[0].numpy(), ref_b, rtol=1e-4, atol=1e-5)
    tvm.testing.assert_allclose(out[1].numpy(), ref_s, rtol=1e-4, atol=1e-5)


@tvm.testing.requires_llvm
def test_multibox_transform_loc_legalize_e2e_nonunity_variances():
    @tvm.script.ir_module
    class Mod:
        @R.function
        def main(
            cls: R.Tensor((1, 3, 5), "float32"),
            loc: R.Tensor((1, 20), "float32"),
            anc: R.Tensor((1, 5, 4), "float32"),
        ) -> R.Tuple(R.Tensor((1, 5, 4), "float32"), R.Tensor((1, 3, 5), "float32")):
            return R.vision.multibox_transform_loc(
                cls,
                loc,
                anc,
                clip=False,
                threshold=0.0,
                variances=(0.1, 0.1, 0.2, 0.2),
                keep_background=True,
            )

    cls_data = np.random.randn(1, 3, 5).astype(np.float32)
    loc_data = np.random.randn(1, 20).astype(np.float32) * 0.05
    anc_data = np.array(
        [
            [
                [0.1, 0.1, 0.5, 0.5],
                [0.2, 0.2, 0.6, 0.6],
                [0.0, 0.0, 1.0, 1.0],
                [0.3, 0.3, 0.7, 0.7],
                [0.05, 0.05, 0.45, 0.45],
            ]
        ],
        dtype=np.float32,
    )

    mod = LegalizeOps()(Mod)
    exe = tvm.compile(mod, target="llvm")
    vm = relax.VirtualMachine(exe, tvm.cpu())
    ref_b, ref_s = _multibox_ref_numpy(cls_data, loc_data, anc_data, (0.1, 0.1, 0.2, 0.2))
    out = vm["main"](
        tvm.runtime.tensor(cls_data, tvm.cpu()),
        tvm.runtime.tensor(loc_data, tvm.cpu()),
        tvm.runtime.tensor(anc_data, tvm.cpu()),
    )
    tvm.testing.assert_allclose(out[0].numpy(), ref_b, rtol=1e-4, atol=1e-5)
    tvm.testing.assert_allclose(out[1].numpy(), ref_s, rtol=1e-4, atol=1e-5)


@tvm.testing.requires_llvm
def test_multibox_transform_loc_legalize_attr_branches():
    @tvm.script.ir_module
    class Mod:
        @R.function
        def main(
            cls: R.Tensor((1, 3, 4), "float32"),
            loc: R.Tensor((1, 16), "float32"),
            anc: R.Tensor((1, 4, 4), "float32"),
        ) -> R.Tuple(R.Tensor((1, 4, 4), "float32"), R.Tensor((1, 3, 4), "float32")):
            return R.vision.multibox_transform_loc(
                cls,
                loc,
                anc,
                clip=True,
                threshold=0.4,
                variances=(1.0, 1.0, 1.0, 1.0),
                keep_background=False,
            )

    cls_data = np.array(
        [[[2.0, 0.1, -0.5, 0.0], [0.2, 2.2, 0.3, -1.0], [0.1, 0.4, 2.0, 0.5]]],
        dtype=np.float32,
    )
    loc_data = np.array(
        [[0.1, -0.2, 0.0, 0.0, -0.2, 0.1, 0.3, -0.1, 0.0, 0.0, 0.8, 0.8, 0.2, 0.2, -0.6, -0.6]],
        dtype=np.float32,
    )
    anc_data = np.array(
        [[[0.1, 0.1, 0.5, 0.5], [0.2, 0.2, 0.6, 0.6], [0.0, 0.0, 1.0, 1.0], [0.4, 0.4, 1.2, 1.2]]],
        dtype=np.float32,
    )

    mod = LegalizeOps()(Mod)
    exe = tvm.compile(mod, target="llvm")
    vm = relax.VirtualMachine(exe, tvm.cpu())
    ref_b, ref_s = _multibox_ref_numpy(
        cls_data,
        loc_data,
        anc_data,
        (1.0, 1.0, 1.0, 1.0),
        clip=True,
        threshold=0.4,
        keep_background=False,
    )
    out = vm["main"](
        tvm.runtime.tensor(cls_data, tvm.cpu()),
        tvm.runtime.tensor(loc_data, tvm.cpu()),
        tvm.runtime.tensor(anc_data, tvm.cpu()),
    )
    boxes = out[0].numpy()
    scores = out[1].numpy()
    tvm.testing.assert_allclose(boxes, ref_b, rtol=1e-4, atol=1e-5)
    tvm.testing.assert_allclose(scores, ref_s, rtol=1e-4, atol=1e-5)
    assert np.all(boxes >= 0.0) and np.all(boxes <= 1.0)
    tvm.testing.assert_allclose(scores[:, 0, :], np.zeros_like(scores[:, 0, :]))


if __name__ == "__main__":
    tvm.testing.main()
