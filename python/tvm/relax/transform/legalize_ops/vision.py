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
import tvm
from tvm import topi, te, tir
import tvm.relax as relax
from tvm.tir import if_then_else
from tvm.relax.op.base import call_pure_packed
from tvm.relax.struct_info import ShapeStructInfo
from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from .common import register_legalize


def _create_onnx_nms_te(boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
    """Create a proper NMS implementation that follows the correct algorithm"""
    scores_shape = list(scores.shape)
    if len(scores_shape) == 3:
        batch, num_classes, num_boxes = scores_shape
    elif len(scores_shape) == 2:
        num_classes, num_boxes = scores_shape
        batch = 1
    else:
        raise ValueError(f"Unexpected scores shape: {scores_shape}")

    if hasattr(max_output_boxes_per_class, "data"):
        max_boxes = int(max_output_boxes_per_class.data.numpy())
    else:
        max_boxes = 3  # Default value

    expected_detections = batch * num_classes * max_boxes


    selected_indices_full, num_total_detections = topi.vision.all_class_non_max_suppression(
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
def _all_class_non_max_suppression(bb: BlockBuilder, call: Call) -> Expr:
    """Legalize all_class_non_max_suppression with dynamic trimming to match ONNX output shape"""
    boxes = call.args[0]
    scores = call.args[1]
    max_output_boxes_per_class = call.args[2]
    iou_threshold = call.args[3]
    score_threshold = call.args[4]
    output_format = call.attrs.output_format

    scores_shape = scores.struct_info.shape
    if len(scores_shape) == 3:
        batch, num_classes, num_boxes = scores_shape
    elif len(scores_shape) == 2:
        num_classes, num_boxes = scores_shape
        batch = 1
    else:
        raise ValueError(f"Unexpected scores shape: {scores_shape}")

    if isinstance(max_output_boxes_per_class, relax.Constant):
        max_boxes_val = int(max_output_boxes_per_class.data.numpy())
    else:
        max_boxes_val = int(num_boxes)

    # Get NMS result with fixed shape
    nms_result = bb.call_te(
        topi.vision.all_class_non_max_suppression,
        boxes,
        scores,
        max_boxes_val,
        iou_threshold,
        score_threshold,
        output_format,
    )

    selected_indices, valid_count = nms_result[0], nms_result[1]
    
    # Extract actual detection count from valid_count
    actual_count = bb.emit(
        relax.op.call_pure_packed(
            "vm.builtin.tensor_to_shape", 
            valid_count, 
            sinfo_args=[relax.ShapeStructInfo([1])]
        )
    )
    
    # Convert to shape and extract the count value
    actual_count_var = relax.Var("actual_count", relax.ShapeStructInfo([relax.PrimValue(0)]))
    bb.match_cast(actual_count, relax.ShapeStructInfo([actual_count_var]))
    
    # Use dynamic strided_slice to trim to actual size
    # This creates output shape [actual_count, 3] instead of [max_boxes, 3]
    trimmed_indices = bb.emit(
        relax.op.dynamic_strided_slice(
            selected_indices,
            begin=[relax.const(0, "int64")],
            end=[actual_count_var],
            strides=[relax.const(1, "int64")],
            axes=[0]
        )
    )
    
    return relax.Tuple([trimmed_indices, valid_count])
