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


def get_valid_counts(data, score_threshold=0, id_index=0, score_index=1):
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
    id_index_const = tvm.tir.const(id_index, "int32")
    score_index_const = tvm.tir.const(score_index, "int32")
    return hybrid_get_valid_counts(
        data,
        score_threshold,
        id_index_const,
        score_index_const,
        tvm.tir.const(1, data.dtype),
        data.shape[0],
        data.shape[1],
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
):
    def nms_inner_loop(ib, i, j, nkeep, num_valid_boxes_local):
        # The box j is valid, invalidate other boxes that overlap with j above iou_threshold
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
                    # invalidate the box k
                    out_scores[i, k] = -1.0
                    on_new_invalidated_box_func(i, k)

    with ib.for_range(0, batch_size, name="i") as i:
        nkeep = if_then_else(tvm.tir.all(top_k > 0, top_k < valid_count[i]), top_k, valid_count[i])
        max_output_size = if_then_else(max_output_size > te.const(0), max_output_size, nkeep)

        with ib.if_scope(tvm.tir.all(iou_threshold > te.const(0), valid_count[i] > te.const(0))):
            num_valid_boxes_local = ib.allocate(
                "int32", (1,), name="num_valid_boxes_local", scope="local"
            )
            box_idx = ib.allocate("int32", (1,), name="box_idx", scope="local")
            num_valid_boxes_local[0] = 0
            box_idx[0] = 0

            # Apply nms
            # No need to do more iteration if we have already reached max_output_size boxes
            with ib.while_loop(
                tvm.tir.all(box_idx[0] < nkeep, num_valid_boxes_local[0] < max_output_size)
            ):
                # Proceed to the inner loop if the box with id box_idx is still valid
                with ib.if_scope(out_scores[i, box_idx[0]] > -1.0):
                    nms_inner_loop(ib, i, box_idx[0], nkeep, num_valid_boxes_local)
                box_idx[0] += 1

            num_valid_boxes[i] = num_valid_boxes_local[0]

        with ib.else_scope():
            num_valid_boxes[i] = 0

    return ib.get()


def _get_valid_box_count(scores, score_threshold):
    batch_classes, num_boxes = scores.shape

    def searchsorted_ir(scores, valid_count):
        ib = tvm.tir.ir_builder.create()
        scores = ib.buffer_ptr(scores)
        valid_count = ib.buffer_ptr(valid_count)

        with ib.for_range(0, batch_classes, name="i", kind="parallel") as i:
            binary_search(ib, i, num_boxes, scores, score_threshold, valid_count)

        return ib.get()

    scores_buf = tvm.tir.decl_buffer(scores.shape, scores.dtype, "scores_buf", data_alignment=8)
    searchsorted_buf = tvm.tir.decl_buffer(
        (batch_classes,), "int32", "searchsorted", data_alignment=8
    )

    return te.extern(
        [(batch_classes,)],
        [scores],
        lambda ins, outs: searchsorted_ir(ins[0], outs[0]),
        dtype=["int32"],
        in_buffers=[scores_buf],
        out_buffers=[searchsorted_buf],
        name="searchsorted",
        tag="searchsorted",
    )


def _collect_selected_indices_ir(num_class, selected_indices, num_detections, row_offsets, out):
    batch_classes, _ = selected_indices.shape

    ib = tvm.tir.ir_builder.create()

    selected_indices = ib.buffer_ptr(selected_indices)
    num_detections = ib.buffer_ptr(num_detections)
    row_offsets = ib.buffer_ptr(row_offsets)
    out = ib.buffer_ptr(out)

    with ib.for_range(0, batch_classes, name="i", kind="parallel") as i:
        i = cast(i, "int64")
        batch_id = i // num_class
        class_id = i % num_class

        with ib.for_range(0, num_detections[i], name="j") as j:
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

    valid_count = _get_valid_box_count(sorted_scores, score_threshold)

    selected_indices, selected_scores, num_detections = run_all_class_nms(
        boxes,
        sorted_scores,
        sorted_indices,
        valid_count,
        max_output_boxes_per_class,
        iou_threshold,
        _nms_loop,
        return_scores=(output_format == "tensorflow"),
    )

    if output_format == "onnx":
        row_offsets = cumsum(num_detections, exclusive=True, dtype="int64")
        num_total_detections = reduction.sum(cast(num_detections, "int64"), axis=1)

        selected_indices = collect_selected_indices(
            num_class, selected_indices, num_detections, row_offsets, _collect_selected_indices_ir
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
