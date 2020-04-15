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


def get_valid_counts(data,
                     score_threshold,
                     id_index=0,
                     score_index=1):
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
    """
    return expr.TupleWrapper(
        _make.get_valid_counts(data, score_threshold,
                               id_index, score_index), 2)


def non_max_suppression(data,
                        valid_count,
                        max_output_size=-1,
                        iou_threshold=0.5,
                        force_suppress=False,
                        top_k=-1,
                        coord_start=2,
                        score_index=1,
                        id_index=0,
                        return_indices=True,
                        invalid_to_bottom=False):
    """Non-maximum suppression operator for object detection.

    Parameters
    ----------
    data : relay.Expr
        3-D tensor with shape [batch_size, num_anchors, 6].
        The last dimension should be in format of
        [class_id, score, box_left, box_top, box_right, box_bottom].

    valid_count : relay.Expr
        1-D tensor for valid number of boxes.

    max_output_size : int, optional
        Max number of output valid boxes for each instance.
        By default all valid boxes are returned.

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
    out : relay.Expr
        3-D tensor with shape [batch_size, num_anchors, 6].
    """
    return _make.non_max_suppression(data, valid_count, max_output_size,
                                     iou_threshold, force_suppress, top_k,
                                     coord_start, score_index, id_index,
                                     return_indices, invalid_to_bottom)
