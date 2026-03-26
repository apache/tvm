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
import tvm.topi.testing
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


def test_get_valid_counts_op_correctness():
    data = relax.Var("data", R.Tensor((2, 10, 6), "float32"))
    assert relax.op.vision.get_valid_counts(data, 0.5).op == Op.get("relax.vision.get_valid_counts")


def test_get_valid_counts_infer_struct_info():
    bb = relax.BlockBuilder()
    data = relax.Var("data", R.Tensor((2, 10, 6), "float32"))
    _check_inference(
        bb,
        relax.op.vision.get_valid_counts(data, score_threshold=0.5, id_index=0, score_index=1),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2,), "int32"),
                relax.TensorStructInfo((2, 10, 6), "float32"),
                relax.TensorStructInfo((2, 10), "int32"),
            ]
        ),
    )


def test_get_valid_counts_infer_struct_info_shape_var():
    bb = relax.BlockBuilder()
    n = tirx.Var("n", "int64")
    m = tirx.Var("m", "int64")
    k = tirx.Var("k", "int64")
    data = relax.Var("data", R.Tensor((n, m, k), "float32"))
    _check_inference(
        bb,
        relax.op.vision.get_valid_counts(data, score_threshold=0.0),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((n,), "int32"),
                relax.TensorStructInfo((n, m, k), "float32"),
                relax.TensorStructInfo((n, m), "int32"),
            ]
        ),
    )


def test_get_valid_counts_wrong_ndim():
    bb = relax.BlockBuilder()
    data = relax.Var("data", R.Tensor((10, 6), "float32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.get_valid_counts(data))


def test_nms_op_correctness():
    data = relax.Var("data", R.Tensor((2, 10, 6), "float32"))
    valid_count = relax.Var("valid_count", R.Tensor((2,), "int32"))
    indices = relax.Var("indices", R.Tensor((2, 10), "int32"))
    assert relax.op.vision.non_max_suppression(
        data, valid_count, indices
    ).op == Op.get("relax.vision.non_max_suppression")


def test_nms_infer_struct_info_return_indices():
    bb = relax.BlockBuilder()
    data = relax.Var("data", R.Tensor((2, 10, 6), "float32"))
    valid_count = relax.Var("valid_count", R.Tensor((2,), "int32"))
    indices = relax.Var("indices", R.Tensor((2, 10), "int32"))
    _check_inference(
        bb,
        relax.op.vision.non_max_suppression(
            data, valid_count, indices, return_indices=True
        ),
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((2, 10), "int32"),
                relax.TensorStructInfo((2, 1), "int32"),
            ]
        ),
    )


def test_nms_infer_struct_info_return_data():
    bb = relax.BlockBuilder()
    data = relax.Var("data", R.Tensor((2, 10, 6), "float32"))
    valid_count = relax.Var("valid_count", R.Tensor((2,), "int32"))
    indices = relax.Var("indices", R.Tensor((2, 10), "int32"))
    _check_inference(
        bb,
        relax.op.vision.non_max_suppression(
            data, valid_count, indices, return_indices=False
        ),
        relax.TensorStructInfo((2, 10, 6), "float32"),
    )


def test_nms_infer_struct_info_return_data_shape_var():
    bb = relax.BlockBuilder()
    batch_size = tirx.Var("batch_size", "int64")
    num_anchors = tirx.Var("num_anchors", "int64")
    elem_length = tirx.Var("elem_length", "int64")
    data = relax.Var("data", R.Tensor((batch_size, num_anchors, elem_length), "float32"))
    valid_count = relax.Var("valid_count", R.Tensor((batch_size,), "int32"))
    indices = relax.Var("indices", R.Tensor((batch_size, num_anchors), "int32"))
    _check_inference(
        bb,
        relax.op.vision.non_max_suppression(
            data, valid_count, indices, return_indices=False
        ),
        relax.TensorStructInfo((batch_size, num_anchors, elem_length), "float32"),
    )


def test_nms_wrong_ndim():
    bb = relax.BlockBuilder()
    data = relax.Var("data", R.Tensor((10, 6), "float32"))
    valid_count = relax.Var("valid_count", R.Tensor((2,), "int32"))
    indices = relax.Var("indices", R.Tensor((2, 10), "int32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.non_max_suppression(data, valid_count, indices))


def test_nms_wrong_valid_count_ndim():
    bb = relax.BlockBuilder()
    data = relax.Var("data", R.Tensor((2, 10, 6), "float32"))
    valid_count = relax.Var("valid_count", R.Tensor((2, 1), "int32"))
    indices = relax.Var("indices", R.Tensor((2, 10), "int32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.non_max_suppression(data, valid_count, indices))


def test_nms_wrong_indices_ndim():
    bb = relax.BlockBuilder()
    data = relax.Var("data", R.Tensor((2, 10, 6), "float32"))
    valid_count = relax.Var("valid_count", R.Tensor((2,), "int32"))
    indices = relax.Var("indices", R.Tensor((20,), "int32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.non_max_suppression(data, valid_count, indices))


def test_nms_wrong_aux_input_dtype():
    bb = relax.BlockBuilder()
    data = relax.Var("data", R.Tensor((2, 10, 6), "float32"))
    valid_count_i64 = relax.Var("valid_count_i64", R.Tensor((2,), "int64"))
    valid_count_i32 = relax.Var("valid_count_i32", R.Tensor((2,), "int32"))
    indices_i64 = relax.Var("indices_i64", R.Tensor((2, 10), "int64"))
    indices_i32 = relax.Var("indices_i32", R.Tensor((2, 10), "int32"))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.non_max_suppression(data, valid_count_i64, indices_i32))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.non_max_suppression(data, valid_count_i32, indices_i64))


def test_nms_wrong_aux_input_shape():
    bb = relax.BlockBuilder()
    data = relax.Var("data", R.Tensor((2, 10, 6), "float32"))
    valid_count_bad_batch = relax.Var("valid_count_bad_batch", R.Tensor((3,), "int32"))
    valid_count = relax.Var("valid_count", R.Tensor((2,), "int32"))
    indices_bad_batch = relax.Var("indices_bad_batch", R.Tensor((3, 10), "int32"))
    indices_bad_anchors = relax.Var("indices_bad_anchors", R.Tensor((2, 9), "int32"))
    with pytest.raises(TVMError):
        bb.normalize(
            relax.op.vision.non_max_suppression(
                data, valid_count_bad_batch, indices_bad_anchors
            )
        )
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.non_max_suppression(data, valid_count, indices_bad_batch))
    with pytest.raises(TVMError):
        bb.normalize(relax.op.vision.non_max_suppression(data, valid_count, indices_bad_anchors))


def test_get_valid_counts_legalize():
    @tvm.script.ir_module
    class GVC:
        @R.function
        def main(
            data: R.Tensor((1, 5, 6), "float32"),
        ) -> R.Tuple(
            R.Tensor((1,), "int32"),
            R.Tensor((1, 5, 6), "float32"),
            R.Tensor((1, 5), "int32"),
        ):
            gv = R.vision.get_valid_counts(data, score_threshold=0.5, id_index=0, score_index=1)
            return gv

    mod = LegalizeOps()(GVC)
    assert "call_tir" in str(mod)
    tvm.ir.assert_structural_equal(
        mod["main"].ret_struct_info,
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((1,), "int32"),
                relax.TensorStructInfo((1, 5, 6), "float32"),
                relax.TensorStructInfo((1, 5), "int32"),
            ]
        ),
    )


def test_nms_legalize():
    @tvm.script.ir_module
    class NMS:
        @R.function
        def main(
            data: R.Tensor((1, 5, 6), "float32"),
            valid_count: R.Tensor((1,), "int32"),
            indices: R.Tensor((1, 5), "int32"),
        ) -> R.Tuple(R.Tensor((1, 5), "int32"), R.Tensor((1, 1), "int32")):
            gv = R.vision.non_max_suppression(
                data,
                valid_count,
                indices,
                max_output_size=-1,
                iou_threshold=0.5,
                force_suppress=False,
                top_k=-1,
                coord_start=2,
                score_index=1,
                id_index=0,
                return_indices=True,
                invalid_to_bottom=False,
            )
            return gv

    mod = LegalizeOps()(NMS)
    assert "call_tir" in str(mod)
    tvm.ir.assert_structural_equal(
        mod["main"].ret_struct_info,
        relax.TupleStructInfo(
            [
                relax.TensorStructInfo((1, 5), "int32"),
                relax.TensorStructInfo((1, 1), "int32"),
            ]
        ),
    )


def test_nms_legalize_return_data():
    @tvm.script.ir_module
    class NMS:
        @R.function
        def main(
            data: R.Tensor((1, 5, 6), "float32"),
            valid_count: R.Tensor((1,), "int32"),
            indices: R.Tensor((1, 5), "int32"),
        ) -> R.Tensor((1, 5, 6), "float32"):
            gv = R.vision.non_max_suppression(
                data,
                valid_count,
                indices,
                max_output_size=-1,
                iou_threshold=0.5,
                force_suppress=False,
                top_k=-1,
                coord_start=2,
                score_index=1,
                id_index=0,
                return_indices=False,
                invalid_to_bottom=True,
            )
            return gv

    mod = LegalizeOps()(NMS)
    assert "call_tir" in str(mod)
    tvm.ir.assert_structural_equal(
        mod["main"].ret_struct_info,
        relax.TensorStructInfo((1, 5, 6), "float32"),
    )


@tvm.testing.requires_llvm
def test_get_valid_counts_e2e():
    """Run get_valid_counts through legalization and compare with the numpy reference."""

    @tvm.script.ir_module
    class GVCModule:
        @R.function
        def main(
            data: R.Tensor((2, 5, 6), "float32"),
        ) -> R.Tuple(
            R.Tensor((2,), "int32"),
            R.Tensor((2, 5, 6), "float32"),
            R.Tensor((2, 5), "int32"),
        ):
            return R.vision.get_valid_counts(data, score_threshold=0.5, id_index=0, score_index=1)

    data_np = np.array(
        [
            [
                [0.0, 0.95, 0.0, 0.0, 1.0, 1.0],
                [1.0, 0.30, 0.0, 0.0, 1.0, 1.0],
                [-1.0, 0.90, 0.0, 0.0, 1.0, 1.0],
                [2.0, 0.75, 2.0, 2.0, 3.0, 3.0],
                [1.0, 0.10, 4.0, 4.0, 5.0, 5.0],
            ],
            [
                [0.0, 0.55, 0.0, 0.0, 1.0, 1.0],
                [1.0, 0.80, 1.0, 1.0, 2.0, 2.0],
                [2.0, 0.40, 2.0, 2.0, 3.0, 3.0],
                [3.0, 0.60, 3.0, 3.0, 4.0, 4.0],
                [-1.0, 0.95, 5.0, 5.0, 6.0, 6.0],
            ],
        ],
        dtype="float32",
    )
    ref_valid_count, ref_out_data, ref_out_indices = tvm.topi.testing.get_valid_counts_python(
        data_np, score_threshold=0.5, id_index=0, score_index=1
    )

    mod = LegalizeOps()(GVCModule)
    exe = tvm.compile(mod, target="llvm")
    vm = relax.VirtualMachine(exe, tvm.cpu())
    result = vm["main"](tvm.runtime.tensor(data_np, tvm.cpu()))

    tvm.testing.assert_allclose(result[0].numpy(), ref_valid_count)
    tvm.testing.assert_allclose(result[1].numpy(), ref_out_data)
    tvm.testing.assert_allclose(result[2].numpy(), ref_out_indices)


def _prepare_nms_inputs(raw_data: np.ndarray):
    """Prepare classic NMS inputs with the numpy get_valid_counts reference."""

    return tvm.topi.testing.get_valid_counts_python(
        raw_data, score_threshold=0.5, id_index=0, score_index=1
    )


def _run_nms_e2e(
    data_np: np.ndarray,
    valid_count_np: np.ndarray,
    indices_np: np.ndarray,
    *,
    max_output_size: int = -1,
    iou_threshold: float = 0.5,
    force_suppress: bool = False,
    top_k: int = -1,
    coord_start: int = 2,
    score_index: int = 1,
    id_index: int = 0,
    return_indices: bool = True,
    invalid_to_bottom: bool = False,
):
    """Run classic NMS through legalization and VM execution."""

    data_shape = tuple(int(dim) for dim in data_np.shape)
    valid_count_shape = tuple(int(dim) for dim in valid_count_np.shape)
    indices_shape = tuple(int(dim) for dim in indices_np.shape)
    data = relax.Var("data", relax.TensorStructInfo(data_shape, "float32"))
    valid_count = relax.Var("valid_count", relax.TensorStructInfo(valid_count_shape, "int32"))
    indices = relax.Var("indices", relax.TensorStructInfo(indices_shape, "int32"))

    bb = relax.BlockBuilder()
    with bb.function("main", (data, valid_count, indices)):
        result = bb.emit(
            relax.op.vision.non_max_suppression(
                data,
                valid_count,
                indices,
                max_output_size=max_output_size,
                iou_threshold=iou_threshold,
                force_suppress=force_suppress,
                top_k=top_k,
                coord_start=coord_start,
                score_index=score_index,
                id_index=id_index,
                return_indices=return_indices,
                invalid_to_bottom=invalid_to_bottom,
            )
        )
        bb.emit_func_output(result)

    mod = LegalizeOps()(bb.get())
    exe = tvm.compile(mod, target="llvm")
    vm = relax.VirtualMachine(exe, tvm.cpu())
    return vm["main"](
        tvm.runtime.tensor(data_np, tvm.cpu()),
        tvm.runtime.tensor(valid_count_np, tvm.cpu()),
        tvm.runtime.tensor(indices_np, tvm.cpu()),
    )


@tvm.testing.requires_llvm
def test_nms_e2e_return_indices():
    """Run classic NMS through legalization and compare with the numpy reference."""

    raw_data = np.array(
        [
            [
                [0.0, 0.95, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.90, 0.05, 0.05, 1.05, 1.05],
                [1.0, 0.85, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.60, 2.0, 2.0, 3.0, 3.0],
                [-1.0, 0.99, 0.0, 0.0, 1.0, 1.0],
            ]
        ],
        dtype="float32",
    )
    valid_count_np, filtered_data_np, filtered_indices_np = _prepare_nms_inputs(raw_data)
    ref_indices, ref_valid_box_count = tvm.topi.testing.non_max_suppression_python(
        filtered_data_np,
        valid_count_np,
        filtered_indices_np,
        max_output_size=-1,
        iou_threshold=0.5,
        force_suppress=False,
        top_k=-1,
        coord_start=2,
        score_index=1,
        id_index=0,
        return_indices=True,
        invalid_to_bottom=False,
    )
    result = _run_nms_e2e(
        filtered_data_np,
        valid_count_np,
        filtered_indices_np,
        return_indices=True,
        invalid_to_bottom=False,
    )

    tvm.testing.assert_allclose(result[0].numpy(), ref_indices)
    tvm.testing.assert_allclose(result[1].numpy(), ref_valid_box_count)


@tvm.testing.requires_llvm
def test_nms_e2e_top_k():
    """Validate that classic NMS honors top_k before suppression."""

    raw_data = np.array(
        [
            [
                [-1.0, 0.99, 9.0, 9.0, 10.0, 10.0],
                [0.0, 0.97, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.96, 2.0, 2.0, 3.0, 3.0],
                [0.0, 0.95, 4.0, 4.0, 5.0, 5.0],
                [1.0, 0.94, 6.0, 6.0, 7.0, 7.0],
                [0.0, 0.20, 8.0, 8.0, 9.0, 9.0],
            ]
        ],
        dtype="float32",
    )
    valid_count_np, filtered_data_np, filtered_indices_np = _prepare_nms_inputs(raw_data)
    ref_indices, ref_valid_box_count = tvm.topi.testing.non_max_suppression_python(
        filtered_data_np,
        valid_count_np,
        filtered_indices_np,
        max_output_size=-1,
        iou_threshold=0.5,
        force_suppress=False,
        top_k=2,
        coord_start=2,
        score_index=1,
        id_index=0,
        return_indices=True,
        invalid_to_bottom=False,
    )
    result = _run_nms_e2e(
        filtered_data_np,
        valid_count_np,
        filtered_indices_np,
        top_k=2,
        return_indices=True,
        invalid_to_bottom=False,
    )

    tvm.testing.assert_allclose(result[0].numpy(), ref_indices)
    tvm.testing.assert_allclose(result[1].numpy(), ref_valid_box_count)
    np.testing.assert_array_equal(ref_indices, np.array([[1, 2, -1, -1, -1, -1]], dtype="int32"))
    np.testing.assert_array_equal(ref_valid_box_count, np.array([[2]], dtype="int32"))


@tvm.testing.requires_llvm
def test_nms_e2e_invalid_to_bottom():
    """Validate that invalid_to_bottom compacts only boxes that remain valid after NMS."""

    raw_data = np.array(
        [
            [
                [0.0, 0.95, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.90, 0.05, 0.05, 1.05, 1.05],
                [1.0, 0.85, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.60, 2.0, 2.0, 3.0, 3.0],
                [-1.0, 0.99, 8.0, 8.0, 9.0, 9.0],
            ]
        ],
        dtype="float32",
    )
    valid_count_np, filtered_data_np, filtered_indices_np = _prepare_nms_inputs(raw_data)
    ref_out_data = tvm.topi.testing.non_max_suppression_python(
        filtered_data_np,
        valid_count_np,
        filtered_indices_np,
        max_output_size=-1,
        iou_threshold=0.5,
        force_suppress=False,
        top_k=-1,
        coord_start=2,
        score_index=1,
        id_index=0,
        return_indices=False,
        invalid_to_bottom=True,
    )
    result = _run_nms_e2e(
        filtered_data_np,
        valid_count_np,
        filtered_indices_np,
        return_indices=False,
        invalid_to_bottom=True,
    )
    expected_out_data = np.array(
        [
            [
                [0.0, 0.95, 0.0, 0.0, 1.0, 1.0],
                [1.0, 0.85, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.60, 2.0, 2.0, 3.0, 3.0],
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
                [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            ]
        ],
        dtype="float32",
    )

    tvm.testing.assert_allclose(result.numpy(), ref_out_data)
    tvm.testing.assert_allclose(result.numpy(), expected_out_data)


@tvm.testing.requires_llvm
def test_nms_e2e_index_remap():
    """Validate that returned indices remap from filtered order back to original order."""

    raw_data = np.array(
        [
            [
                [-1.0, 0.99, 9.0, 9.0, 10.0, 10.0],
                [0.0, 0.60, 4.0, 4.0, 5.0, 5.0],
                [0.0, 0.10, 8.0, 8.0, 9.0, 9.0],
                [0.0, 0.95, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.90, 0.05, 0.05, 1.05, 1.05],
                [1.0, 0.80, 2.0, 2.0, 3.0, 3.0],
            ]
        ],
        dtype="float32",
    )
    valid_count_np, filtered_data_np, filtered_indices_np = _prepare_nms_inputs(raw_data)
    ref_indices, ref_valid_box_count = tvm.topi.testing.non_max_suppression_python(
        filtered_data_np,
        valid_count_np,
        filtered_indices_np,
        max_output_size=-1,
        iou_threshold=0.5,
        force_suppress=False,
        top_k=-1,
        coord_start=2,
        score_index=1,
        id_index=0,
        return_indices=True,
        invalid_to_bottom=False,
    )
    result = _run_nms_e2e(
        filtered_data_np,
        valid_count_np,
        filtered_indices_np,
        return_indices=True,
        invalid_to_bottom=False,
    )

    tvm.testing.assert_allclose(result[0].numpy(), ref_indices)
    tvm.testing.assert_allclose(result[1].numpy(), ref_valid_box_count)
    np.testing.assert_array_equal(ref_indices, np.array([[3, 5, 1, -1, -1, -1]], dtype="int32"))
    np.testing.assert_array_equal(ref_valid_box_count, np.array([[3]], dtype="int32"))


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


if __name__ == "__main__":
    tvm.testing.main()
