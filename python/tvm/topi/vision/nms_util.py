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
# pylint: disable=invalid-name
"""Common utilities used in Non-maximum suppression operators"""
import tvm
from tvm import te


def _get_boundaries(output, box_idx):
    l = tvm.te.min(
        output[box_idx],
        output[box_idx + 2],
    )
    t = tvm.te.min(
        output[box_idx + 1],
        output[box_idx + 3],
    )
    r = tvm.te.max(
        output[box_idx],
        output[box_idx + 2],
    )
    b = tvm.te.max(
        output[box_idx + 1],
        output[box_idx + 3],
    )
    return l, t, r, b


def calculate_overlap(out_tensor, box_a_idx, box_b_idx):
    """Calculate overlap of two boxes."""
    a_l, a_t, a_r, a_b = _get_boundaries(out_tensor, box_a_idx)
    b_l, b_t, b_r, b_b = _get_boundaries(out_tensor, box_b_idx)

    # Overlapping width and height
    w = tvm.te.max(0.0, tvm.te.min(a_r, b_r) - tvm.te.max(a_l, b_l))
    h = tvm.te.max(0.0, tvm.te.min(a_b, b_b) - tvm.te.max(a_t, b_t))

    # Overlapping area
    area = h * w

    # total area of the figure formed by box a and box b
    # except for overlapping area
    u = (a_r - a_l) * (a_b - a_t) + (b_r - b_l) * (b_b - b_t) - area
    return tvm.tir.Select(u <= 0.0, 0.0, area / u)


def binary_search(ib, y, num_boxes, scores, score_threshold, out):
    """Binary search for score_threshold on scores sorted in descending order"""
    lo = ib.allocate("int32", (1,), name="lo", scope="local")
    hi = ib.allocate("int32", (1,), name="hi", scope="local")

    lo[0] = 0
    hi[0] = num_boxes

    with ib.while_loop(lo[0] < hi[0]):
        mid = (hi[0] + lo[0]) >> 1
        with ib.if_scope(scores[y, mid] > score_threshold):
            lo[0] = mid + 1
        with ib.else_scope():
            hi[0] = mid

    out[y] = lo[0]


def collect_selected_indices(num_class, selected_indices, num_detections, row_offsets, ir):
    """Collect selected indices from the core NMS loop into one linear output

    Parameters
    ----------
    num_class : int

    selected_indices: tvm.te.Tensor
        2-D tensor with shape (batch_size * num_classes, num_boxes), representing the indices
        of selected boxes by the core NMS loop.

    num_detections tvm.te.Tensor
        1-D tensor with shape (batch_size * num_classes,), representing
        the number of boxes selected by the core NMS loop, per batch and class

    row_offsets tvm.te.Tensor
        1-D tensor with shape (batch_size * num_classes,), this should be the exclusive scan
        of num_detections

    ir : function
        A function to generate IR for CPU or GPU, see its usage in vision/nms.py and cuda/nms.py

    Returns
    -------
    out : tvm.te.Tensor
        The output is indices of size (batch_size * num_class* num_boxes , 3).
        Rows of indices are ordered such that selected boxes from batch 0, class 0 come
        first, in descending of scores, followed by boxes from batch 0, class 1 etc.
    """
    batch_class, num_boxes = selected_indices.shape
    return te.extern(
        [(batch_class * num_boxes, 3)],
        [selected_indices, num_detections, row_offsets],
        lambda ins, outs: ir(num_class, ins[0], ins[1], ins[2], outs[0]),
        dtype=["int64"],
        name="collect_indices",
        tag="collect_indices",
    )


def collect_selected_indices_and_scores(
    selected_indices, selected_scores, num_detections, row_offsets, num_total_detections, ir
):
    """Collect selected indices and scores from the core NMS loop into one linear output

    Parameters
    ----------
    num_class : int

    selected_indices: tvm.te.Tensor
        2-D tensor with shape (batch_size * num_classes, num_boxes), representing the indices
        of selected boxes by the core NMS loop.

    selected_indices: tvm.te.Tensor
        2-D tensor with shape (batch_size * num_classes, num_boxes), representing the scores
        of selected boxes by the core NMS loop.

    num_detections tvm.te.Tensor
        2-D tensor with shape (batch_size, num_classes), representing
        the number of boxes selected by the core NMS loop, per batch and class

    row_offsets tvm.te.Tensor
        2-D tensor with shape (batch_size, num_classes), this should be the exclusive scan
        of num_detections along axis 1

    ir : function
        A function to generate IR for CPU or GPU, see its usage in vision/nms.py and cuda/nms.py

    Returns
    -------
    out : [tvm.te.Tensor, tvm.te.Tensor]
        The output is two tensors. The first is indices of size
        (batch_size, num_class* num_boxes, 2), and the second is scores of size
        (batch_size, num_class* num_boxes).
    """
    batch_size, num_class = row_offsets.shape
    num_boxes = selected_indices.shape[1]
    return te.extern(
        [(batch_size, num_class * num_boxes, 2), (batch_size, num_class * num_boxes)],
        [selected_indices, selected_scores, num_detections, row_offsets, num_total_detections],
        lambda ins, outs: ir(ins[0], ins[1], ins[2], ins[3], ins[4], outs[0], outs[1]),
        dtype=["int64", "float32"],
        name="collect_indices_and_scores",
        tag="collect_indices_and_scores",
    )


def _all_class_nms_ir(
    boxes,
    sorted_scores,
    sorted_indices,
    valid_count,
    batch_class,
    num_class,
    num_anchors,
    iou_threshold,
    max_output_size_per_class,
    box_indices,
    selected_scores,
    num_valid_boxes,
    nms_loop,
):
    ib = tvm.tir.ir_builder.create()
    boxes = ib.buffer_ptr(boxes)
    sorted_scores = ib.buffer_ptr(sorted_scores)
    sorted_indices = ib.buffer_ptr(sorted_indices)
    valid_count = ib.buffer_ptr(valid_count)
    box_indices = ib.buffer_ptr(box_indices)
    num_valid_boxes = ib.buffer_ptr(num_valid_boxes)

    if selected_scores is not None:
        selected_scores = ib.buffer_ptr(selected_scores)

    if isinstance(iou_threshold, float):
        iou_threshold = tvm.tir.FloatImm("float32", iou_threshold)

    if isinstance(max_output_size_per_class, int):
        max_output_size_per_class = tvm.tir.const(max_output_size_per_class)

    def calc_overlap(i, j, k):
        offset_j = sorted_indices[i, j] * 4
        offset_k = sorted_indices[i, k] * 4
        batch_id = i // num_class
        base_bbox_idx = batch_id * num_anchors * 4
        return calculate_overlap(
            boxes,
            base_bbox_idx + offset_j,
            base_bbox_idx + offset_k,
        )

    def on_new_valid_box(ib, tid, num_current_valid_box, i, j):
        with ib.if_scope(tid + 0 == 0):
            box_indices[i, num_current_valid_box] = sorted_indices[i, j]

            if selected_scores is not None:
                selected_scores[i, num_current_valid_box] = sorted_scores[i, j]

    def on_new_invalidated_box(*_):
        pass

    def needs_bbox_check(*_):
        return tvm.tir.const(True)

    return nms_loop(
        ib,
        batch_class,
        tvm.tir.IntImm("int32", -1),  # top_k
        iou_threshold,
        max_output_size_per_class,
        valid_count,
        on_new_valid_box,
        on_new_invalidated_box,
        needs_bbox_check,
        calc_overlap,
        sorted_scores,
        num_valid_boxes,
    )


def run_all_class_nms(
    boxes,
    sorted_scores,
    sorted_indices,
    valid_count,
    max_output_size_per_class,
    iou_threshold,
    nms_loop,
    return_scores=False,
):
    """The core all class NMS routine

    Parameters
    ----------
    boxes : tvm.te.Tensor
        3-D tensor with shape (batch_size, num_boxes, 4)

    sorted_scores: tvm.te.Tensor
        2-D tensor with shape (batch_size * num_classes, num_boxes)
        One of the outputs from argsort

    sorted_indices: tvm.te.Tensor
        2-D tensor with shape (batch_size * num_classes, num_boxes)
        The other output from argsort

    valid_count: tvm.te.Tensor
        1-D tensor with shape (batch_size * num_classes,), representing
        the number of boxes whose score is above score_threshold, per batch and class

    max_output_boxes_per_class : int or tvm.te.Tensor, optional
        The maxinum number of output selected boxes per class

    iou_threshold : float or tvm.te.Tensor, optionaIl
        IoU test threshold

    nms_loop : function
        A core NMS loop, see its usage in vision/nms.py and cuda/nms.py

    return_scores : bool, optional
        Whether or not to return selected scores, needed by the tensorflow output format.

    Returns
    -------
    out : a list of tvm.te.Tensor
        The output is three tensors, the first and second are indices and scores of size
        (batch_size * num_class, num_boxes), and the third is a tensor
        num_selected_boxes of shape (batch_size * num_class,) representing the total number of
        selected boxes per batch and class. If return_scores is False, the second output is
        None.
    """
    batch, num_boxes, _ = boxes.shape
    batch_class = sorted_scores.shape[0]
    num_class = batch_class // batch

    if return_scores is False:
        selected_indices, num_detections = te.extern(
            [(batch_class, num_boxes), (1, batch_class)],
            [boxes, sorted_scores, sorted_indices, valid_count],
            lambda ins, outs: _all_class_nms_ir(
                ins[0],  # boxes
                ins[1],  # sorted_scores
                ins[2],  # sorted_indices
                ins[3],  # valid_count
                batch_class,
                num_class,
                num_boxes,
                iou_threshold,
                max_output_size_per_class,
                outs[0],  # box_indices
                None,  # scores
                outs[1],  # num_selected_boxes
                nms_loop,
            ),
            dtype=["int32", "int32"],
            name="all_class_nms",
            tag="all_class_nms",
        )
        return selected_indices, None, num_detections

    return te.extern(
        [(batch_class, num_boxes), (batch_class, num_boxes), (1, batch_class)],
        [boxes, sorted_scores, sorted_indices, valid_count],
        lambda ins, outs: _all_class_nms_ir(
            ins[0],  # boxes
            ins[1],  # sorted_scores
            ins[2],  # sorted_indices
            ins[3],  # valid_count
            batch_class,
            num_class,
            num_boxes,
            iou_threshold,
            max_output_size_per_class,
            outs[0],  # box_indices
            outs[1],  # selected scores
            outs[2],  # num_selected_boxes
            nms_loop,
        ),
        dtype=["int32", "float32", "int32"],
        name="all_class_nms",
        tag="all_class_nms",
    )
