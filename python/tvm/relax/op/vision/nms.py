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
        first, in descending of scores, followed by boxes from batch 0, class 1 etc. Out of
        `batch_size * num_class* num_boxes` rows of indices, only the first `num_total_detection`
        rows are valid.

        TODO: Implement true dynamic output shapes to match ONNX Runtime behavior exactly.
        This would eliminate the need for manual trimming and improve memory efficiency.
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
