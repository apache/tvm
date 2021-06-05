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
"""Non-maximum suppression operations."""
from tvm.relay import expr
from . import _make


def get_valid_counts(data, score_threshold, id_index=0, score_index=1):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : relay.Expr
        Input data. 3-D tensor with shape [batch_size, num_anchors, 6].

    score_threshold : optional, float
        Lower limit of score for valid bounding boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    Returns
    -------
    valid_count : relay.Expr
        1-D tensor for valid number of boxes.

    out_tensor : relay.Expr
        Rearranged data tensor.

    out_indices: relay.Expr
        Indices in input data
    """
    if not isinstance(score_threshold, expr.Expr):
        score_threshold = expr.const(score_threshold, "float32")
    return expr.TupleWrapper(
        _make.get_valid_counts(data, score_threshold, id_index, score_index), 3
    )


def non_max_suppression(
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
):
    """Non-maximum suppression operator for object detection.

    Parameters
    ----------
    data : relay.Expr
        3-D tensor with shape [batch_size, num_anchors, 6]
        or [batch_size, num_anchors, 5].
        The last dimension should be in format of
        [class_id, score, box_left, box_top, box_right, box_bottom]
        or [score, box_left, box_top, box_right, box_bottom]. It could
        be the second output out_tensor of get_valid_counts.

    valid_count : relay.Expr
        1-D tensor for valid number of boxes. It could be the output
        valid_count of get_valid_counts.

    indices: relay.Expr
        2-D tensor with shape [batch_size, num_anchors], represents
        the index of box in original data. It could be the third
        output out_indices of get_valid_counts. The values in the
        second dimension are like the output of arange(num_anchors)
        if get_valid_counts is not used before non_max_suppression.

    max_output_size : int or relay.Expr, optional
        Max number of output valid boxes for each instance.
        Return all valid boxes if the value of max_output_size is less than 0.

    iou_threshold : float or relay.Expr, optional
        Non-maximum suppression threshold.

    force_suppress : bool, optional
        Suppress all detections regardless of class_id.

    top_k : int, optional
        Keep maximum top k detections before nms, -1 for no limit.

    coord_start : int, optional
        The starting index of the consecutive 4 coordinates.

    score_index : int, optional
        Index of the scores/confidence of boxes.

    id_index : int, optional
        index of the class categories, -1 to disable.

    return_indices : bool, optional
        Whether to return box indices in input data.

    invalid_to_bottom : bool, optional
        Whether to move all valid bounding boxes to the top.

    Returns
    -------
    out : relay.Expr or relay.Tuple
        return relay.Expr if return_indices is disabled, a 3-D tensor
        with shape [batch_size, num_anchors, 6] or [batch_size, num_anchors, 5].
        If return_indices is True, return relay.Tuple of two 2-D tensors, with
        shape [batch_size, num_anchors] and [batch_size, num_valid_anchors] respectively.
    """
    if not isinstance(max_output_size, expr.Expr):
        max_output_size = expr.const(max_output_size, "int32")
    if not isinstance(iou_threshold, expr.Expr):
        iou_threshold = expr.const(iou_threshold, "float32")
    out = _make.non_max_suppression(
        data,
        valid_count,
        indices,
        max_output_size,
        iou_threshold,
        force_suppress,
        top_k,
        coord_start,
        score_index,
        id_index,
        return_indices,
        invalid_to_bottom,
    )
    if return_indices:
        return expr.TupleWrapper(out, 2)
    return out


def all_class_non_max_suppression(
    boxes,
    scores,
    max_output_boxes_per_class=-1,
    iou_threshold=-1.0,
    score_threshold=-1.0,
    output_format="onnx",
):
    """Non-maximum suppression operator for object detection, corresponding to ONNX
    NonMaxSuppression and TensorFlow combined_non_max_suppression.
    NMS is performed for each class separately.

    Parameters
    ----------
    boxes : relay.Expr
        3-D tensor with shape (batch_size, num_boxes, 4)

    scores: relay.Expr
        3-D tensor with shape (batch_size, num_classes, num_boxes)

    max_output_boxes_per_class : int or relay.Expr, optional
        The maxinum number of output selected boxes per class

    iou_threshold : float or relay.Expr, optionaIl
        IoU test threshold

    score_threshold : float or relay.Expr, optional
        Score threshold to filter out low score boxes early

    output_format : string, optional
        "onnx" or "tensorflow". Specify by which frontends the outputs are
        intented to be consumed.

    Returns
    -------
    out : relay.Tuple
        If `output_format` is "onnx", the output is a relay.Tuple of two tensors, the first is
        `indices` of size `(batch_size * num_class* num_boxes , 3)` and the second is a scalar
        tensor `num_total_detection` of shape `(1,)` representing the total number of selected
        boxes. The three values in `indices` encode batch, class, and box indices.
        Rows of `indices` are ordered such that selected boxes from batch 0, class 0 come first,
        in descending of scores, followed by boxes from batch 0, class 1 etc. Out of
        `batch_size * num_class* num_boxes` rows of indices,  only the first `num_total_detection`
        rows are valid.

        If `output_format` is "tensorflow", the output is a relay.Tuple of three tensors, the first
        is `indices` of size `(batch_size, num_class * num_boxes , 2)`, the second is `scores` of
        size `(batch_size, num_class * num_boxes)`, and the third is `num_total_detection` of size
        `(batch_size,)` representing the total number of selected boxes per batch. The two values
        in `indices` encode class and box indices. Of num_class * num_boxes boxes in `indices` at
        batch b, only the first `num_total_detection[b]` entries are valid. The second axis of
        `indices` and `scores` are sorted within each class by box scores, but not across classes.
        So the box indices and scores for the class 0 come first in a sorted order, followed by
        the class 1 etc.
    """
    if not isinstance(max_output_boxes_per_class, expr.Expr):
        max_output_boxes_per_class = expr.const(max_output_boxes_per_class, "int32")
    if not isinstance(iou_threshold, expr.Expr):
        iou_threshold = expr.const(iou_threshold, "float32")
    if not isinstance(score_threshold, expr.Expr):
        score_threshold = expr.const(score_threshold, "float32")

    out = _make.all_class_non_max_suppression(
        boxes,
        scores,
        max_output_boxes_per_class,
        iou_threshold,
        score_threshold,
        output_format,
    )

    if output_format == "onnx":
        return expr.TupleWrapper(out, 2)

    return expr.TupleWrapper(out, 3)
