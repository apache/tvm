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
    hi[0] = num_boxes.astype("int32")

    with ib.while_loop(lo[0] < hi[0]):
        mid = (hi[0] + lo[0]) >> 1
        with ib.if_scope(scores[y, mid] > score_threshold):
            lo[0] = mid + 1
        with ib.else_scope():
            hi[0] = mid

    out[y] = lo[0]


def _estimate_max_detections(batch_class, input_image_size=None):
    """Estimate maximum detections based on input image size and number of classes.

    This provides a more intelligent default for production environments.
    """
    if input_image_size is not None:
        # Estimate based on image size: larger images typically have more objects
        if len(input_image_size) >= 2:
            height, width = input_image_size[-2], input_image_size[-1]
            total_pixels = height * width

            # Base estimation per class based on image size
            if total_pixels < 300000:  # Small images (< 300k pixels)
                base_detections_per_class = min(50, max(10, total_pixels // 2000))
            elif total_pixels < 1000000:  # Medium images (< 1M pixels)
                base_detections_per_class = min(100, max(25, total_pixels // 3000))
            else:  # Large images (>= 1M pixels)
                base_detections_per_class = min(200, max(50, total_pixels // 4000))

            # Scale down for many classes (more realistic for multi-class scenarios)
            if batch_class > 20:
                # For many classes, reduce per-class detections to avoid explosion
                detections_per_class = min(base_detections_per_class, 50)
            else:
                detections_per_class = base_detections_per_class
        else:
            detections_per_class = 50  # fallback
    else:
        # Fallback to class-based estimation
        if batch_class == 1:
            detections_per_class = 100  # Single class detection
        elif batch_class <= 10:
            detections_per_class = 50  # Small multi-class
        else:
            detections_per_class = 25  # Large multi-class (COCO-like)

    return batch_class * detections_per_class


def collect_selected_indices(
    num_class,
    selected_indices,
    num_detections,
    row_offsets,
    ir,
    max_output_boxes_per_class=None,
    output_shape=None,
    num_total_detections=None,
    input_image_size=None,
):
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

    if output_shape is not None:
        return te.extern(
            [output_shape],
            [selected_indices, num_detections, row_offsets],
            lambda ins, outs: ir(
                num_class, ins[0], ins[1], ins[2], outs[0], max_output_boxes_per_class
            ),
            dtype=["int64"],
            name="collect_indices",
            tag="collect_indices",
        )

    # TODO: Implement dynamic trimming based on num_total_detections
    if num_total_detections is not None:
        if isinstance(max_output_boxes_per_class, int):
            out_rows = batch_class * max_output_boxes_per_class
        else:
            # Smart fallback based on input image size and typical production scenarios
            out_rows = _estimate_max_detections(batch_class, input_image_size)

        return te.extern(
            [(out_rows, 3)],
            [selected_indices, num_detections, row_offsets],
            lambda ins, outs: ir(
                num_class, ins[0], ins[1], ins[2], outs[0], max_output_boxes_per_class
            ),
            dtype=["int64"],
            name="collect_indices",
            tag="collect_indices",
        )

    if isinstance(max_output_boxes_per_class, int):
        out_rows = batch_class * max_output_boxes_per_class
        return te.extern(
            [(out_rows, 3)],
            [selected_indices, num_detections, row_offsets],
            lambda ins, outs: ir(
                num_class, ins[0], ins[1], ins[2], outs[0], max_output_boxes_per_class
            ),
            dtype=["int64"],
            name="collect_indices",
            tag="collect_indices",
        )

    if isinstance(max_output_boxes_per_class, te.Tensor):
        try:
            if len(max_output_boxes_per_class.shape) == 0:
                max_boxes_val = int(max_output_boxes_per_class.data.numpy())
            elif (
                len(max_output_boxes_per_class.shape) == 1
                and max_output_boxes_per_class.shape[0] == 1
            ):
                max_boxes_val = int(max_output_boxes_per_class.data.numpy()[0])
            else:
                max_boxes_val = num_boxes
        except (ValueError, IndexError, AttributeError):
            max_boxes_val = num_boxes

        out_rows = batch_class * max_boxes_val
        return te.extern(
            [(out_rows, 3)],
            [selected_indices, num_detections, row_offsets],
            lambda ins, outs: ir(
                num_class, ins[0], ins[1], ins[2], outs[0], max_output_boxes_per_class
            ),
            dtype=["int64"],
            name="collect_indices",
            tag="collect_indices",
        )

    return te.extern(
        [(batch_class * num_boxes, 3)],
        [selected_indices, num_detections, row_offsets],
        lambda ins, outs: ir(
            num_class, ins[0], ins[1], ins[2], outs[0], max_output_boxes_per_class
        ),
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
    score_threshold=None,
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
    elif isinstance(iou_threshold, te.Tensor):
        if len(iou_threshold.shape) == 0:
            iou_threshold = iou_threshold()
        elif len(iou_threshold.shape) == 1 and iou_threshold.shape[0] == 1:
            iou_threshold = iou_threshold[0]
        else:
            iou_threshold = tvm.tir.FloatImm("float32", 0.5)

    if isinstance(max_output_size_per_class, int):
        max_output_size_per_class = tvm.tir.const(max_output_size_per_class)
    elif isinstance(max_output_size_per_class, te.Tensor):
        if len(max_output_size_per_class.shape) == 0:
            max_output_size_per_class = max_output_size_per_class()
        elif len(max_output_size_per_class.shape) == 1 and max_output_size_per_class.shape[0] == 1:
            # Use tensor indexing to get the first element
            max_output_size_per_class = max_output_size_per_class[0]
        else:
            max_output_size_per_class = tvm.tir.const(1000)

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
        score_threshold,
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
    score_threshold=None,
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
        all_class_num0_buf = tvm.tir.decl_buffer(
            (batch_class, num_boxes), "int32", "all_class_nms0", data_alignment=8
        )
        all_class_num1_buf = tvm.tir.decl_buffer(
            (batch_class,), "int32", "all_class_nms1", data_alignment=8
        )
        extern_inputs = [boxes, sorted_scores, sorted_indices, valid_count]
        if score_threshold is not None:
            extern_inputs.append(score_threshold)

        selected_indices, num_detections = te.extern(
            [(batch_class, num_boxes), (batch_class,)],
            extern_inputs,
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
                ins[4] if score_threshold is not None else None,  # score_threshold
            ),
            out_buffers=[all_class_num0_buf, all_class_num1_buf],
            dtype=["int32", "int32"],
            name="all_class_nms",
            tag="all_class_nms",
        )
        return selected_indices, None, num_detections

    extern_inputs = [boxes, sorted_scores, sorted_indices, valid_count]
    if score_threshold is not None:
        extern_inputs.append(score_threshold)

    return te.extern(
        [(batch_class, num_boxes), (batch_class, num_boxes), (batch_class,)],
        extern_inputs,
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
            ins[4] if score_threshold is not None else None,  # score_threshold
        ),
        dtype=["int32", "float32", "int32"],
        name="all_class_nms",
        tag="all_class_nms",
    )
