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

    iou_threshold : float, optional
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
    if isinstance(max_output_size, int):
        max_output_size = expr.const(max_output_size, "int32")
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
