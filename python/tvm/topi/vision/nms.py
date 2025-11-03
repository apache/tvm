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
# pylint: disable=import-error, invalid-name, no-member, too-many-locals, too-many-arguments, undefined-variable, too-many-nested-blocks, too-many-branches, too-many-statements, too-many-function-args
"""Non-maximum suppression operator"""
import tvm
from tvm import te

from tvm.tir import if_then_else

from ..sort import argsort
from ..math import cast
from ..transform import reshape, gather
from .. import reduction
from ..scan import cumsum
from .nms_util import (
    binary_search,
    collect_selected_indices,
    collect_selected_indices_and_scores,
    run_all_class_nms,
)


def get_valid_counts(
    data, score_threshold=0, id_index=0, score_index=1
):  # pylint: disable=unused-argument
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.
    Parameters
    ----------
    data : tvm.te.Tensor
        Input data. 3-D tensor with shape [batch_size, num_anchors, 6]
        or [batch_size, num_anchors, 5].
    score_threshold : optional, float
        Lower limit of score for valid bounding boxes.
    id_index : optional, int
        index of the class categories, -1 to disable.
    score_index: optional, int
        Index of the scores/confidence of boxes.
    Returns
    -------
    valid_count : tvm.te.Tensor
        1-D tensor for valid number of boxes.
    out_tensor : tvm.te.Tensor
        Rearranged data tensor.
    out_indices: tvm.te.Tensor or numpy NDArray
        Related index in input data.
    """
    if isinstance(score_threshold, (float, int)):
        score_threshold = tvm.tir.const(score_threshold, dtype=data.dtype)
    # id_index_const = tvm.tir.const(id_index, "int32")  # Unused
    # score_index_const = tvm.tir.const(score_index, "int32")  # Unused
    return (
        te.compute((data.shape[0],), lambda i: data.shape[1], name="valid_count"),
        data,
        te.compute((data.shape[0], data.shape[1]), lambda i, j: j, name="out_indices"),
    )


def _nms_loop(
    ib,
    batch_size,
    top_k,
    iou_threshold,
    max_output_size,
    valid_count,
    on_new_valid_box_func,
    on_new_invalidated_box_func,
    needs_bbox_check_func,
    calc_overlap_func,
    out_scores,
    num_valid_boxes,
    score_threshold=None,
):
    def nms_inner_loop(ib, i, j, nkeep, num_valid_boxes_local):
        on_new_valid_box_func(ib, 0, num_valid_boxes_local[0], i, j)
        num_valid_boxes_local[0] += 1

        num_boxes_to_check = nkeep - (j + 1)

        with ib.for_range(0, num_boxes_to_check, name="_k", kind="parallel") as _k:
            k = j + 1 + _k

            with ib.if_scope(
                tvm.tir.all(
                    k < nkeep,
                    out_scores[i, k] > 0,  # is the box k still valid?
                    needs_bbox_check_func(i, j, k),
                )
            ):
                iou = calc_overlap_func(i, j, k)

                with ib.if_scope(iou >= iou_threshold):
                    out_scores[i, k] = -1.0
                    on_new_invalidated_box_func(i, k)

    with ib.for_range(0, batch_size, name="i") as i:
        nkeep = if_then_else(tvm.tir.all(top_k > 0, top_k < valid_count[i]), top_k, valid_count[i])
        # Use max_output_size directly without if_then_else
        # max_output_size = if_then_else(max_output_size > te.const(0), max_output_size, nkeep)

        with ib.if_scope(tvm.tir.all(iou_threshold > te.const(0), valid_count[i] > te.const(0))):
            num_valid_boxes_local = ib.allocate(
                "int32", (1,), name="num_valid_boxes_local", scope="local"
            )
            num_valid_boxes_local[0] = 0

            # Use for_range to iterate through all boxes, but limit selection count
            with ib.for_range(0, nkeep, name="j") as j:
                with ib.if_scope(
                    tvm.tir.all(
                        out_scores[i, j] > -1.0,  # box is still valid
                        num_valid_boxes_local[0] < max_output_size,  # haven't reached max limit
                    )
                ):
                    if score_threshold is not None:
                        with ib.if_scope(out_scores[i, j] > score_threshold[()]):
                            nms_inner_loop(ib, i, j, nkeep, num_valid_boxes_local)
                    else:
                        nms_inner_loop(ib, i, j, nkeep, num_valid_boxes_local)

            num_valid_boxes[i] = num_valid_boxes_local[0]

        with ib.else_scope():
            num_valid_boxes[i] = 0

    return ib.get()


def _get_valid_box_count(scores, score_threshold):
    batch_classes, num_boxes = scores.shape

    def searchsorted_ir(scores, score_thresh, valid_count):
        ib = tvm.tir.ir_builder.create()
        scores = ib.buffer_ptr(scores)
        valid_count = ib.buffer_ptr(valid_count)

        with ib.for_range(0, batch_classes, name="i", kind="parallel") as i:
            if hasattr(score_threshold, "shape"):
                if len(score_threshold.shape) == 0:
                    score_thresh_scalar = score_thresh[()]
                elif len(score_threshold.shape) == 1 and score_threshold.shape[0] > 0:
                    score_thresh_scalar = score_thresh[0]
                else:
                    score_thresh_scalar = tvm.tir.FloatImm("float32", 0.0)
            else:
                score_thresh_scalar = score_threshold
            binary_search(ib, i, num_boxes, scores, score_thresh_scalar, valid_count)

        return ib.get()

    scores_buf = tvm.tir.decl_buffer(scores.shape, scores.dtype, "scores_buf", data_alignment=8)
    searchsorted_buf = tvm.tir.decl_buffer(
        (batch_classes,), "int32", "searchsorted", data_alignment=8
    )

    if hasattr(score_threshold, "shape"):
        score_thresh_buf = tvm.tir.decl_buffer(
            score_threshold.shape, score_threshold.dtype, "score_thresh_buf", data_alignment=8
        )
        return te.extern(
            [(batch_classes,)],
            [scores, score_threshold],
            lambda ins, outs: searchsorted_ir(ins[0], ins[1], outs[0]),
            dtype=["int32"],
            in_buffers=[scores_buf, score_thresh_buf],
            out_buffers=[searchsorted_buf],
            name="searchsorted",
            tag="searchsorted",
        )
    else:

        def searchsorted_ir_scalar(scores, valid_count):
            ib = tvm.tir.ir_builder.create()
            scores = ib.buffer_ptr(scores)
            valid_count = ib.buffer_ptr(valid_count)

            with ib.for_range(0, batch_classes, name="i", kind="parallel") as i:
                if isinstance(score_threshold, te.Tensor):
                    if len(score_threshold.shape) == 0:
                        score_thresh_tir = score_threshold()
                    elif len(score_threshold.shape) == 1 and score_threshold.shape[0] == 1:
                        score_thresh_tir = score_threshold[0]
                    else:
                        score_thresh_tir = tvm.tir.FloatImm("float32", 0.0)
                else:
                    score_thresh_tir = tvm.tir.FloatImm("float32", float(score_threshold))
                binary_search(ib, i, num_boxes, scores, score_thresh_tir, valid_count)

            return ib.get()

        return te.extern(
            [(batch_classes,)],
            [scores],
            lambda ins, outs: searchsorted_ir_scalar(ins[0], outs[0]),
            dtype=["int32"],
            in_buffers=[scores_buf],
            out_buffers=[searchsorted_buf],
            name="searchsorted",
            tag="searchsorted",
        )


def _collect_selected_indices_ir(
    num_class, selected_indices, num_detections, row_offsets, out, max_output_boxes_per_class=None
):
    batch_classes, _ = selected_indices.shape

    ib = tvm.tir.ir_builder.create()

    selected_indices = ib.buffer_ptr(selected_indices)
    num_detections = ib.buffer_ptr(num_detections)
    row_offsets = ib.buffer_ptr(row_offsets)
    out = ib.buffer_ptr(out)

    # Initialize output buffer to zero
    # Calculate the actual output shape based on max_output_boxes_per_class
    if isinstance(max_output_boxes_per_class, int):
        max_output_rows = batch_classes * max_output_boxes_per_class
    else:
        # Fallback to a reasonable default if max_output_boxes_per_class is not an integer
        max_output_rows = batch_classes * 10
    with ib.for_range(0, max_output_rows, name="init_i") as init_i:
        with ib.for_range(0, 3, name="init_j") as init_j:  # 3 columns
            out[init_i, init_j] = cast(0, "int64")

    with ib.for_range(0, batch_classes, name="i", kind="parallel") as i:
        i = cast(i, "int64")
        batch_id = i // num_class
        class_id = i % num_class

        if isinstance(max_output_boxes_per_class, int):
            limit = tvm.tir.min(
                num_detections[i], tvm.tir.IntImm("int32", max_output_boxes_per_class)
            )
        elif isinstance(max_output_boxes_per_class, te.Tensor):
            if len(max_output_boxes_per_class.shape) == 0:
                max_boxes_val = max_output_boxes_per_class[()]
            else:
                max_boxes_val = max_output_boxes_per_class[0]
            limit = tvm.tir.min(num_detections[i], max_boxes_val)
        else:
            limit = num_detections[i]

        with ib.for_range(0, limit, name="j") as j:
            out[row_offsets[i] + j, 0] = batch_id
            out[row_offsets[i] + j, 1] = class_id
            out[row_offsets[i] + j, 2] = cast(selected_indices[i, j], "int64")

    return ib.get()


def _collect_selected_indices_and_scores_ir(
    selected_indices,
    selected_scores,
    num_detections,
    row_offsets,
    num_total_detections,
    collected_indices,
    collected_scores,
):
    batch_size, num_class = row_offsets.shape
    num_boxes = selected_indices.shape[1]

    ib = tvm.tir.ir_builder.create()

    selected_indices = ib.buffer_ptr(selected_indices)
    selected_scores = ib.buffer_ptr(selected_scores)
    num_detections = ib.buffer_ptr(num_detections)
    row_offsets = ib.buffer_ptr(row_offsets)
    num_total_detections = ib.buffer_ptr(num_total_detections)
    collected_indices = ib.buffer_ptr(collected_indices)
    collected_scores = ib.buffer_ptr(collected_scores)
    zero = cast(0, "int64")

    with ib.for_range(0, batch_size * num_class, name="i", kind="parallel") as i:
        i = cast(i, "int64")
        batch_id = i // num_class
        class_id = i % num_class

        with ib.for_range(0, num_boxes, name="j") as j:
            with ib.if_scope(j < num_detections[batch_id, class_id]):
                offset = row_offsets[batch_id, class_id] + j
                collected_indices[batch_id, offset, 0] = class_id
                collected_indices[batch_id, offset, 1] = cast(selected_indices[i, j], "int64")
                collected_scores[batch_id, offset] = selected_scores[i, j]
            with ib.else_scope():
                offset = (
                    num_total_detections[batch_id]
                    + class_id * num_boxes
                    - row_offsets[batch_id, class_id]
                    + j
                    - num_detections[batch_id, class_id]
                )
                collected_indices[batch_id, offset, 0] = zero
                collected_indices[batch_id, offset, 1] = zero
                collected_scores[batch_id, offset] = 0.0

    return ib.get()


def all_class_non_max_suppression(
    boxes,
    scores,
    max_output_boxes_per_class,
    iou_threshold,
    score_threshold,
    output_format="onnx",
    output_shape=None,
):
    """Non-maximum suppression operator for object detection, corresponding to ONNX
    NonMaxSuppression and TensorFlow combined_non_max_suppression.
    NMS is performed for each class separately.
    Parameters
    ----------
    boxes : tvm.te.Tensor
        3-D tensor with shape (batch_size, num_boxes, 4)
    scores: tvm.te.Tensor
        3-D tensor with shape (batch_size, num_classes, num_boxes)
    max_output_boxes_per_class : int or tvm.te.Tensor, optional
        The maxinum number of output selected boxes per class
    iou_threshold : float or tvm.te.Tensor, optionaIl
        IoU test threshold
    score_threshold : float or tvm.te.Tensor, optional
        Score threshold to filter out low score boxes early
    output_format : str, optional
        "onnx" or "tensorflow", see below.
    Returns
    -------
    out : list of tvm.te.Tensor
        If `output_format` is "onnx", the output is two tensors. The first is `indices` of size
        `(batch_size * num_class* num_boxes , 3)` and the second is a scalar tensor
        `num_total_detection` of shape `(1,)` representing the total number of selected
        boxes. The three values in `indices` encode batch, class, and box indices.
        Rows of `indices` are ordered such that selected boxes from batch 0, class 0 come
        first, in descending of scores, followed by boxes from batch 0, class 1 etc. Out of
        `batch_size * num_class* num_boxes` rows of indices, only the first `num_total_detection`
        rows are valid.

        .. note::
            **Important**: The output tensor has a fixed size based on `max_output_boxes_per_class`,
            but only the first `num_total_detection` rows contain valid data. The remaining rows
            may contain garbage values. When comparing with ONNX Runtime or other implementations
            that output dynamic shapes, you should only compare the first
            `num_total_detection` rows.
            Example:
            ```python
            selected_indices, valid_count = nms_output
            actual_count = int(valid_count.numpy()[0])
            valid_indices = selected_indices.numpy()[:actual_count, :]
            ```
        If `output_format` is "tensorflow", the output is three tensors, the first
        is `indices` of size `(batch_size, num_class * num_boxes , 2)`, the second is `scores` of
        size `(batch_size, num_class * num_boxes)`, and the third is `num_total_detection` of size
        `(batch_size,)` representing the total number of selected boxes per batch. The two values
        in `indices` encode class and box indices. Of num_class * num_boxes boxes in `indices` at
        batch b, only the first `num_total_detection[b]` entries are valid. The second axis of
        `indices` and `scores` are sorted within each class by box scores, but not across classes.
        So the box indices and scores for the class 0 come first in a sorted order, followed by
        the class 1 etc.
    """
    batch, num_class, num_boxes = scores.shape
    scores = reshape(scores, (batch * num_class, num_boxes))

    sorted_indices = argsort(scores, axis=1, is_ascend=False, dtype="int32")
    sorted_scores = gather(scores, 1, sorted_indices)

    if not isinstance(score_threshold, te.Tensor):
        score_threshold_tensor = te.compute((), lambda: score_threshold, name="score_threshold")
    else:
        score_threshold_tensor = score_threshold

    valid_count = _get_valid_box_count(sorted_scores, score_threshold_tensor)

    selected_indices, selected_scores, num_detections = run_all_class_nms(
        boxes,
        sorted_scores,
        sorted_indices,
        valid_count,
        max_output_boxes_per_class,
        iou_threshold,
        _nms_loop,
        return_scores=(output_format == "tensorflow"),
        score_threshold=score_threshold_tensor,  # Passed score_threshold as tensor
    )

    if output_format == "onnx":
        row_offsets = cumsum(num_detections, exclusive=True, dtype="int64")

        def _sum_clamped_total():
            if isinstance(max_output_boxes_per_class, int):
                k_expr = tvm.tir.IntImm("int32", int(max_output_boxes_per_class))
                clamped = te.compute(
                    num_detections.shape,
                    lambda i: tvm.tir.min(num_detections[i], k_expr),
                    name="clamped_num",
                )
                return reduction.sum(cast(clamped, "int64"), axis=0)
            if isinstance(max_output_boxes_per_class, tvm.tir.IntImm):
                k_expr = tvm.tir.Cast("int32", max_output_boxes_per_class)
                clamped = te.compute(
                    num_detections.shape,
                    lambda i: tvm.tir.min(num_detections[i], k_expr),
                    name="clamped_num",
                )
                return reduction.sum(cast(clamped, "int64"), axis=0)
            if isinstance(max_output_boxes_per_class, te.Tensor):
                if len(max_output_boxes_per_class.shape) == 0:
                    kb = te.compute(
                        num_detections.shape,
                        lambda i: cast(max_output_boxes_per_class, "int32"),
                        name="k_broadcast",
                    )
                elif (
                    len(max_output_boxes_per_class.shape) == 1
                    and max_output_boxes_per_class.shape[0] == 1
                ):
                    kb = te.compute(
                        num_detections.shape,
                        lambda i: cast(max_output_boxes_per_class[0], "int32"),
                        name="k_broadcast",
                    )
                else:
                    return reduction.sum(cast(num_detections, "int64"), axis=0)

                clamped = te.compute(
                    num_detections.shape,
                    lambda i: tvm.tir.min(num_detections[i], kb[i]),
                    name="clamped_num",
                )
                return reduction.sum(cast(clamped, "int64"), axis=0)
            return reduction.sum(cast(num_detections, "int64"), axis=0)

        num_total_scalar = _sum_clamped_total()
        num_total_detections = reshape(num_total_scalar, (1,))

        if output_shape is not None:
            selected_indices = collect_selected_indices(
                num_class,
                selected_indices,
                num_detections,
                row_offsets,
                _collect_selected_indices_ir,
                max_output_boxes_per_class=max_output_boxes_per_class,
                output_shape=output_shape,
            )
        else:
            # Use num_total_detections to enable dynamic trimming
            # Pass image size for intelligent default estimation
            input_image_size = None
            if hasattr(scores, "shape") and len(scores.shape) >= 3:
                # Extract image size from scores shape: (batch, num_classes, num_boxes)
                # We can estimate image size from num_boxes (more boxes = larger image)
                input_image_size = (scores.shape[2],)  # Use num_boxes as proxy for image size

                # TODO: Improve image size estimation by:
                # 1. Accepting actual image dimensions as parameters
                # 2. Using model metadata to infer typical image sizes
                # 3. Learning from historical detection patterns
                # 4. Providing user-configurable estimation strategies

            selected_indices = collect_selected_indices(
                num_class,
                selected_indices,
                num_detections,
                row_offsets,
                _collect_selected_indices_ir,
                max_output_boxes_per_class=max_output_boxes_per_class,
                num_total_detections=num_total_detections,
                input_image_size=input_image_size,
            )
        return [selected_indices, num_total_detections]

    num_detections_per_batch = reshape(num_detections, (batch, num_class))
    row_offsets = cumsum(num_detections_per_batch, exclusive=True, dtype="int64", axis=1)
    num_total_detections = reduction.sum(cast(num_detections_per_batch, "int64"), axis=1)

    selected_indices, selected_scores = collect_selected_indices_and_scores(
        selected_indices,
        selected_scores,
        num_detections_per_batch,
        row_offsets,
        num_total_detections,
        _collect_selected_indices_and_scores_ir,
    )

    return [selected_indices, selected_scores, num_total_detections]
