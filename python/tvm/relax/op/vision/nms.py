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
"""Non-maximum suppression operator"""

# from tvm import relax  # Unused import
from . import _ffi_api


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
    boxes : relax.Expr
        3-D tensor with shape (batch_size, num_boxes, 4)
    scores: relax.Expr
        3-D tensor with shape (batch_size, num_classes, num_boxes)
    max_output_boxes_per_class : relax.Expr
        The maxinum number of output selected boxes per class
    iou_threshold : relax.Expr
        IoU test threshold
    score_threshold : relax.Expr
        Score threshold to filter out low score boxes early
    output_format : str, optional
        "onnx" or "tensorflow", see below.

    Returns
    -------
    out : relax.Expr
        If `output_format` is "onnx", the output is two tensors. The first is `indices` of size
        `(batch_size * num_class* num_boxes , 3)` and the second is a scalar tensor
        `num_total_detection` of shape `(1,)` representing the total number of selected
        boxes. The three values in `indices` encode batch, class, and box indices.
        Rows of `indices` are ordered such that selected boxes from batch 0, class 0 come
        first, in descending of scores, followed by boxes from batch 0, class 1 etc.
        The output uses dynamic_strided_slice to trim to only valid detections,
        so the first tensor has shape (num_total_detection, 3) containing only valid rows.

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
    return _ffi_api.all_class_non_max_suppression(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, output_format
    )


def get_valid_counts(data, score_threshold=0, id_index=0, score_index=1):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : relax.Expr
        3-D tensor with shape [batch_size, num_anchors, elem_length].

    score_threshold : float, optional
        Lower limit of score for valid bounding boxes.

    id_index : int, optional
        Index of the class categories, -1 to disable.

    score_index : int, optional
        Index of the scores/confidence of boxes.

    Returns
    -------
    out : relax.Expr
        A tuple of three tensors:
        - valid_count: 1-D tensor [batch_size]
        - out_tensor: 3-D tensor [batch_size, num_anchors, elem_length]
        - out_indices: 2-D tensor [batch_size, num_anchors]
    """
    return _ffi_api.get_valid_counts(data, score_threshold, id_index, score_index)


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
    data : relax.Expr
        3-D tensor with shape [batch_size, num_anchors, elem_length].

    valid_count : relax.Expr
        1-D tensor for valid number of boxes.

    indices : relax.Expr
        2-D tensor with shape [batch_size, num_anchors].

    max_output_size : int, optional
        Max number of output valid boxes, -1 for no limit.

    iou_threshold : float, optional
        Non-maximum suppression IoU threshold.

    force_suppress : bool, optional
        Whether to suppress all detections regardless of class_id.

    top_k : int, optional
        Keep maximum top k detections before nms, -1 for no limit.

    coord_start : int, optional
        Start index of the consecutive 4 coordinates.

    score_index : int, optional
        Index of the scores/confidence of boxes.

    id_index : int, optional
        Index of the class categories, -1 to disable.

    return_indices : bool, optional
        Whether to return box indices in input data.

    invalid_to_bottom : bool, optional
        Whether to move all valid bounding boxes to the top.

    Returns
    -------
    out : relax.Expr
        If return_indices is True, returns a tuple of (box_indices, valid_box_count).
        Otherwise returns the modified data tensor.
    """
    return _ffi_api.non_max_suppression(
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
