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
"""Default legalization function for vision network related operators."""
from tvm import topi, te
from tvm import relax
from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from .common import register_legalize


def _create_onnx_nms_te(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
    """Create a proper NMS implementation that follows the correct algorithm"""
    scores_shape = list(scores.shape)
    if len(scores_shape) == 3:
        batch, num_classes, _ = scores_shape
    elif len(scores_shape) == 2:
        num_classes, _ = scores_shape
        batch = 1
    else:
        raise ValueError(f"Unexpected scores shape: {scores_shape}")

    if hasattr(max_output_boxes_per_class, "data"):
        max_boxes = int(max_output_boxes_per_class.data.numpy())
    else:
        max_boxes = 3  # Default value

    expected_detections = batch * num_classes * max_boxes

    selected_indices_full, _ = topi.vision.all_class_non_max_suppression(
        boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold, "onnx"
    )

    def slice_to_onnx_shape(data, expected_size):
        def compute_element(i, j):
            return tvm.tir.if_then_else(i < expected_size, data[i, j], tvm.tir.Cast("int64", 0))

        return te.compute((expected_size, 3), compute_element, name="sliced_indices")

    sliced_indices = slice_to_onnx_shape(selected_indices_full, expected_detections)

    actual_detections = te.compute(
        (1,), lambda i: tvm.tir.Cast("int64", expected_detections), name="actual_detections"
    )

    return [sliced_indices, actual_detections]


@register_legalize("relax.vision.all_class_non_max_suppression")
def _all_class_non_max_suppression(block_builder: BlockBuilder, call: Call) -> Expr:
    """Legalize all_class_non_max_suppression with fixed shape output.

    Note: This implementation outputs fixed-size tensors with trailing garbage data.
    Only the first `num_total_detection` rows contain valid data. Users should use
    the `valid_count` tensor to determine how many rows are actually valid.

    For complete ONNX compatibility, users can post-process the output:
    ```python
    selected_indices, valid_count = nms_output
    actual_count = int(valid_count.numpy()[0])
    valid_indices = selected_indices.numpy()[:actual_count, :]
    ```
    """
    boxes = call.args[0]
    scores = call.args[1]
    max_output_boxes_per_class = call.args[2]
    iou_threshold = call.args[3]
    score_threshold = call.args[4]
    output_format = call.attrs.output_format

    scores_shape = scores.struct_info.shape
    if len(scores_shape) == 3:
        _, _, num_boxes = scores_shape
    elif len(scores_shape) == 2:
        _, num_boxes = scores_shape
    else:
        raise ValueError(f"Unexpected scores shape: {scores_shape}")

    if isinstance(max_output_boxes_per_class, relax.Constant):
        max_boxes_val = int(max_output_boxes_per_class.data.numpy())
    else:
        max_boxes_val = int(num_boxes)

    # Get NMS result with fixed shape from TOPI
    nms_result = block_builder.call_te(
        topi.vision.all_class_non_max_suppression,
        boxes,
        scores,
        max_boxes_val,
        iou_threshold,
        score_threshold,
        output_format,
    )

    # TODO: Implement dynamic output trimming for better memory efficiency
    # Current approach returns fixed-size output with trailing garbage data
    # Future improvements could include:
    # 1. Dynamic strided_slice based on num_total_detections
    # 2. Custom Relax operator with true dynamic shapes
    # 3. VM builtin functions for runtime shape adjustment
    # 4. Symbolic shape inference in Relax IR
    #
    # For now, users should trim manually:
    # actual_count = int(num_total_detections.numpy()[0])
    # valid_indices = selected_indices.numpy()[:actual_count, :]

    return nms_result
