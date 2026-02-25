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

from tvm import relax, te, tir, topi

from ...block_builder import BlockBuilder
from ...expr import Call, Expr, TupleGetItem
from .common import register_legalize


@register_legalize("relax.vision.all_class_non_max_suppression")
def _all_class_non_max_suppression(block_builder: BlockBuilder, call: Call) -> Expr:
    """Legalize all_class_non_max_suppression with dynamic output trimming.

    This implementation uses dynamic_strided_slice to trim the NMS output to only
    contain valid detections, improving memory efficiency and ONNX compatibility.

    Returns
    -------
    result : Tuple[Tensor, Tensor]
        A tuple of (trimmed_indices, num_total_detections) where:
        - trimmed_indices: Tensor of shape (num_total_detections, 3) containing only
          valid detection indices (batch_id, class_id, box_id)
        - num_total_detections: Tensor of shape (1,) with the count of valid detections
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

    # Dynamic output trimming using dynamic_strided_slice
    # Extract selected_indices and num_total_detections from the NMS result
    selected_indices = block_builder.emit(TupleGetItem(nms_result, 0))
    num_total_detections = block_builder.emit(TupleGetItem(nms_result, 1))

    # Build slicing parameters using TE to avoid high-level Relax ops during legalization
    def build_begin():
        return te.compute((2,), lambda i: tir.const(0, "int64"), name="begin")

    def build_strides():
        return te.compute((2,), lambda i: tir.const(1, "int64"), name="strides")

    def build_end(count_tensor):
        # end = [count_tensor[0], 3]
        def compute_end(i):
            return tir.if_then_else(
                i == 0,
                tir.Cast("int64", count_tensor[0]),
                tir.const(3, "int64"),
            )

        return te.compute((2,), compute_end, name="end")

    begin = block_builder.call_te(build_begin)
    strides = block_builder.call_te(build_strides)
    end = block_builder.call_te(build_end, num_total_detections)

    # Apply dynamic strided slice to trim to valid detections only
    trimmed_indices = block_builder.emit(
        relax.op.dynamic_strided_slice(selected_indices, begin, end, strides)
    )

    # Return trimmed indices along with num_total_detections for compatibility
    return relax.Tuple([trimmed_indices, num_total_detections])
