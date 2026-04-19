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
# pylint: disable=import-error, invalid-name, no-member, too-many-locals, too-many-arguments, undefined-variable, too-many-nested-blocks, too-many-branches, too-many-statements, too-many-function-args
"""Non-maximum suppression operator"""

import tvm
from tvm import te
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T
from tvm.tirx import if_then_else

from .. import reduction
from ..math import cast
from ..scan import cumsum
from ..sort import argsort
from ..transform import gather, reshape
from .nms_util import (
    binary_search,
    collect_selected_indices,
    collect_selected_indices_and_scores,
    run_all_class_nms,
)


def _get_valid_counts_ir(
    data, score_threshold, id_index, score_index, valid_count, out_tensor, out_indices
):
    """IR for get_valid_counts. Filters boxes by score and compacts valid ones to the top."""
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    box_data_length = data.shape[2]

    with IRBuilder() as ib:
        data = T.buffer_proxy(data)
        valid_count = T.buffer_proxy(valid_count)
        out_tensor = T.buffer_proxy(out_tensor)
        out_indices = T.buffer_proxy(out_indices)

        with T.parallel(0, batch_size) as i:
            valid_count[i] = T.int32(0)

            with T.serial(0, num_anchors) as j:
                score = data[i, j, score_index]
                if id_index < 0:
                    is_valid = score > score_threshold
                else:
                    is_valid = tvm.tirx.all(score > score_threshold, data[i, j, id_index] >= 0)

                with T.If(is_valid):
                    with T.Then():
                        cur = valid_count[i]
                        with T.serial(0, box_data_length) as k:
                            out_tensor[i, cur, k] = data[i, j, k]
                        out_indices[i, cur] = j
                        valid_count[i] = cur + 1

            # Fill remaining slots with -1
            with T.serial(0, num_anchors) as j:
                with T.If(j >= valid_count[i]):
                    with T.Then():
                        with T.serial(0, box_data_length) as k:
                            out_tensor[i, j, k] = tvm.tirx.Cast(data.dtype, T.float32(-1.0))
                        out_indices[i, j] = T.int32(-1)

        return ib.get()


def get_valid_counts(data, score_threshold=0, id_index=0, score_index=1):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data. 3-D tensor with shape [batch_size, num_anchors, elem_length].

    score_threshold : optional, float
        Lower limit of score for valid bounding boxes.

    id_index : optional, int
        Index of the class categories, -1 to disable.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    Returns
    -------
    valid_count : tvm.te.Tensor
        1-D tensor for valid number of boxes, shape [batch_size].

    out_tensor : tvm.te.Tensor
        Rearranged data tensor, shape [batch_size, num_anchors, elem_length].

    out_indices: tvm.te.Tensor
        Related index in input data, shape [batch_size, num_anchors].
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    box_data_length = data.shape[2]

    is_score_threshold_tensor = isinstance(score_threshold, te.Tensor)
    if not is_score_threshold_tensor:
        score_threshold = tvm.tirx.const(score_threshold, dtype=data.dtype)

    id_index_const = tvm.tirx.const(id_index, "int32")
    score_index_const = tvm.tirx.const(score_index, "int32")

    valid_count_buf = tvm.tirx.decl_buffer((batch_size,), "int32", "valid_count")
    out_tensor_buf = tvm.tirx.decl_buffer(
        (batch_size, num_anchors, box_data_length), data.dtype, "out_tensor"
    )
    out_indices_buf = tvm.tirx.decl_buffer(
        (batch_size, num_anchors), "int32", "out_indices"
    )

    if is_score_threshold_tensor:
        score_thresh_buf = tvm.tirx.decl_buffer(
            score_threshold.shape, score_threshold.dtype, "score_threshold"
        )
        valid_count, out_tensor, out_indices = te.extern(
            [(batch_size,), (batch_size, num_anchors, box_data_length), (batch_size, num_anchors)],
            [data, score_threshold],
            lambda ins, outs: _get_valid_counts_ir(
                ins[0], ins[1], id_index_const, score_index_const,
                outs[0], outs[1], outs[2],
            ),
            dtype=["int32", data.dtype, "int32"],
            out_buffers=[valid_count_buf, out_tensor_buf, out_indices_buf],
            in_buffers=[
                tvm.tirx.decl_buffer(data.shape, data.dtype, "data"),
                score_thresh_buf,
            ],
            name="get_valid_counts",
            tag="get_valid_counts",
        )
    else:
        # score_threshold is a TIR constant, not a tensor
        def _ir_with_const_threshold(ins, outs):
            return _get_valid_counts_ir(
                ins[0], score_threshold, id_index_const, score_index_const,
                outs[0], outs[1], outs[2],
            )

        valid_count, out_tensor, out_indices = te.extern(
            [(batch_size,), (batch_size, num_anchors, box_data_length), (batch_size, num_anchors)],
            [data],
            _ir_with_const_threshold,
            dtype=["int32", data.dtype, "int32"],
            out_buffers=[valid_count_buf, out_tensor_buf, out_indices_buf],
            in_buffers=[tvm.tirx.decl_buffer(data.shape, data.dtype, "data")],
            name="get_valid_counts",
            tag="get_valid_counts",
        )

    return valid_count, out_tensor, out_indices


def _classic_nms_ir(
    data,
    sorted_index,
    valid_count,
    indices,
    batch_size,
    num_anchors,
    box_data_length,
    max_output_size,
    iou_threshold,
    force_suppress,
    top_k,
    coord_start,
    score_index,
    id_index,
    return_indices,
    out_data,
    out_box_indices,
    out_valid_box_count,
    soft_nms_sigma=0.0,
    score_threshold=0.0,
):
    """IR for classic single-class non-maximum suppression."""
    with IRBuilder() as ib:
        data = T.buffer_proxy(data)
        sorted_index = T.buffer_proxy(sorted_index)
        valid_count = T.buffer_proxy(valid_count)
        indices = T.buffer_proxy(indices)
        out_data = T.buffer_proxy(out_data)
        out_box_indices = T.buffer_proxy(out_box_indices)
        if out_valid_box_count is not None:
            out_valid_box_count = T.buffer_proxy(out_valid_box_count)

        is_soft_nms = soft_nms_sigma > 0.0
        # For hard NMS the historical threshold is 0.0; for soft NMS use score_threshold.
        thresh = tvm.tirx.Cast(data.dtype, T.float32(score_threshold if is_soft_nms else 0.0))

        with T.parallel(0, batch_size) as i:
            # Step 1: Reorder data by sorted score
            nkeep_buf = T.alloc_buffer((1,), "int32", scope="local")
            nkeep_local = T.buffer_proxy(nkeep_buf)
            nkeep_local[0] = valid_count[i]
            with T.If(tvm.tirx.all(top_k > 0, top_k < nkeep_local[0])):
                with T.Then():
                    nkeep_local[0] = top_k

            # Copy sorted boxes to output
            with T.serial(0, num_anchors) as j:
                with T.If(j < nkeep_local[0]):
                    with T.Then():
                        src_idx = sorted_index[i, j]
                        with T.serial(0, box_data_length) as k:
                            out_data[i, j, k] = data[i, src_idx, k]
                        out_box_indices[i, j] = sorted_index[i, j]
                    with T.Else():
                        with T.serial(0, box_data_length) as k:
                            out_data[i, j, k] = tvm.tirx.Cast(data.dtype, T.float32(-1.0))
                        out_box_indices[i, j] = T.int32(-1)

            # Step 2: Apply NMS - greedy suppression
            num_valid_boxes_buf = T.alloc_buffer((1,), "int32", scope="local")
            num_valid_boxes = T.buffer_proxy(num_valid_boxes_buf)
            num_valid_boxes[0] = T.int32(0)

            with T.serial(0, nkeep_local[0]) as j:
                # Check if box j is still valid (score > threshold) and within max_output_size
                with T.If(
                    tvm.tirx.all(
                        out_data[i, j, score_index] > thresh,
                        tvm.tirx.Select(
                            max_output_size > 0,
                            num_valid_boxes[0] < max_output_size,
                            tvm.tirx.const(True),
                        ),
                    )
                ):
                    with T.Then():
                        num_valid_boxes[0] = num_valid_boxes[0] + 1

                        # Suppress overlapping boxes
                        with T.serial(0, nkeep_local[0]) as k:
                            with T.If(
                                tvm.tirx.all(
                                    k > j,
                                    out_data[i, k, score_index] > thresh,
                                )
                            ):
                                with T.Then():
                                    # Check class ID match (or force_suppress)
                                    do_suppress = tvm.tirx.const(False)
                                    if force_suppress:
                                        do_suppress = tvm.tirx.const(True)
                                    elif id_index >= 0:
                                        do_suppress = (
                                            out_data[i, j, id_index] == out_data[i, k, id_index]
                                        )
                                    else:
                                        do_suppress = tvm.tirx.const(True)

                                    with T.If(do_suppress):
                                        with T.Then():
                                            # Calculate IoU
                                            a_l = tvm.te.min(
                                                out_data[i, j, coord_start],
                                                out_data[i, j, coord_start + 2],
                                            )
                                            a_t = tvm.te.min(
                                                out_data[i, j, coord_start + 1],
                                                out_data[i, j, coord_start + 3],
                                            )
                                            a_r = tvm.te.max(
                                                out_data[i, j, coord_start],
                                                out_data[i, j, coord_start + 2],
                                            )
                                            a_b = tvm.te.max(
                                                out_data[i, j, coord_start + 1],
                                                out_data[i, j, coord_start + 3],
                                            )

                                            b_l = tvm.te.min(
                                                out_data[i, k, coord_start],
                                                out_data[i, k, coord_start + 2],
                                            )
                                            b_t = tvm.te.min(
                                                out_data[i, k, coord_start + 1],
                                                out_data[i, k, coord_start + 3],
                                            )
                                            b_r = tvm.te.max(
                                                out_data[i, k, coord_start],
                                                out_data[i, k, coord_start + 2],
                                            )
                                            b_b = tvm.te.max(
                                                out_data[i, k, coord_start + 1],
                                                out_data[i, k, coord_start + 3],
                                            )

                                            w = tvm.te.max(
                                                tvm.tirx.Cast(data.dtype, T.float32(0.0)),
                                                tvm.te.min(a_r, b_r) - tvm.te.max(a_l, b_l),
                                            )
                                            h = tvm.te.max(
                                                tvm.tirx.Cast(data.dtype, T.float32(0.0)),
                                                tvm.te.min(a_b, b_b) - tvm.te.max(a_t, b_t),
                                            )
                                            area = h * w
                                            u = (
                                                (a_r - a_l) * (a_b - a_t)
                                                + (b_r - b_l) * (b_b - b_t)
                                                - area
                                            )
                                            iou = tvm.tirx.Select(
                                                u <= tvm.tirx.Cast(data.dtype, T.float32(0.0)),
                                                tvm.tirx.Cast(data.dtype, T.float32(0.0)),
                                                area / u,
                                            )

                                            with T.If(iou >= iou_threshold):
                                                with T.Then():
                                                    if is_soft_nms:
                                                        # Soft-NMS Gaussian decay
                                                        decay = tvm.tirx.exp(
                                                            -(iou * iou)
                                                            / tvm.tirx.Cast(
                                                                data.dtype,
                                                                T.float32(soft_nms_sigma),
                                                            )
                                                        )
                                                        out_data[i, k, score_index] = (
                                                            out_data[i, k, score_index] * decay
                                                        )
                                                    else:
                                                        out_data[i, k, score_index] = tvm.tirx.Cast(
                                                            data.dtype, T.float32(-1.0)
                                                        )
                                                        out_box_indices[i, k] = T.int32(-1)

                    with T.Else():
                        # Box suppressed or beyond max_output_size
                        with T.serial(0, box_data_length) as k:
                            out_data[i, j, k] = tvm.tirx.Cast(data.dtype, T.float32(-1.0))
                        out_box_indices[i, j] = T.int32(-1)

            # Step 3: If return_indices, remap to original indices
            if return_indices:
                if out_valid_box_count is not None:
                    # Count valid boxes and remap indices
                    valid_idx_buf = T.alloc_buffer((1,), "int32", scope="local")
                    valid_idx = T.buffer_proxy(valid_idx_buf)
                    valid_idx[0] = T.int32(0)

                    with T.serial(0, num_anchors) as j:
                        with T.If(out_box_indices[i, j] >= 0):
                            with T.Then():
                                orig_idx = out_box_indices[i, j]
                                out_box_indices[i, valid_idx[0]] = indices[i, orig_idx]
                                valid_idx[0] = valid_idx[0] + 1

                    out_valid_box_count[i, 0] = valid_idx[0]

                    # Fill remaining with -1
                    with T.serial(0, num_anchors) as j:
                        with T.If(j >= valid_idx[0]):
                            with T.Then():
                                out_box_indices[i, j] = T.int32(-1)

        return ib.get()


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
    soft_nms_sigma=0.0,
    score_threshold=0.0,
):
    """Non-maximum suppression operator for object detection.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D tensor with shape [batch_size, num_anchors, elem_length].

    valid_count : tvm.te.Tensor
        1-D tensor for valid number of boxes, shape [batch_size].

    indices : tvm.te.Tensor
        2-D tensor with shape [batch_size, num_anchors].

    max_output_size : optional, int
        Max number of output valid boxes for each instance.
        Return all valid boxes if the value is less than 0.

    iou_threshold : optional, float
        Non-maximum suppression IoU threshold.

    force_suppress : optional, boolean
        Whether to suppress all detections regardless of class_id. When
        ``id_index`` is ``-1``, all valid boxes are treated as belonging to the
        same class, so this flag has the same effect as ``True``.

    top_k : optional, int
        Keep maximum top k detections before nms, -1 for no limit.

    coord_start : required, int
        Start index of the consecutive 4 coordinates.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    id_index : optional, int
        Index of the class categories, -1 to disable.

    return_indices : optional, boolean
        Whether to return box indices in input data.

    invalid_to_bottom : optional, boolean
        Whether to move all valid bounding boxes to the top.

    soft_nms_sigma : optional, float
        Sigma for soft-NMS Gaussian penalty. 0.0 means standard hard NMS.

    score_threshold : optional, float
        Minimum score for a box to be eligible during soft-NMS.

    Returns
    -------
    out : tvm.te.Tensor or tuple of tvm.te.Tensor
        If return_indices is True, returns a tuple of (box_indices, valid_box_count).
        Otherwise returns the modified data tensor.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    box_data_length = data.shape[2]

    if isinstance(max_output_size, int):
        max_output_size = tvm.tirx.const(max_output_size, dtype="int32")
    if isinstance(iou_threshold, (float, int)):
        iou_threshold = tvm.tirx.const(iou_threshold, dtype=data.dtype)

    # Sort by score
    score_shape = (batch_size, num_anchors)
    score_tensor = te.compute(
        score_shape, lambda i, j: data[i, j, score_index], name="score_tensor"
    )
    sort_tensor = argsort(score_tensor, valid_count=valid_count, axis=1, is_ascend=False)

    data_buf = tvm.tirx.decl_buffer(data.shape, data.dtype, "data")
    sort_buf = tvm.tirx.decl_buffer(sort_tensor.shape, sort_tensor.dtype, "sorted_index")
    valid_count_buf = tvm.tirx.decl_buffer(valid_count.shape, valid_count.dtype, "valid_count")
    indices_buf = tvm.tirx.decl_buffer(indices.shape, indices.dtype, "indices")

    out_data_buf = tvm.tirx.decl_buffer(data.shape, data.dtype, "out_data")
    out_box_indices_buf = tvm.tirx.decl_buffer(
        (batch_size, num_anchors), "int32", "out_box_indices"
    )

    if return_indices:
        out_valid_box_count_buf = tvm.tirx.decl_buffer(
            (batch_size, 1), "int32", "out_valid_box_count"
        )

        out_data, out_box_indices, out_valid_box_count = te.extern(
            [data.shape, (batch_size, num_anchors), (batch_size, 1)],
            [data, sort_tensor, valid_count, indices],
            lambda ins, outs: _classic_nms_ir(
                ins[0], ins[1], ins[2], ins[3],
                batch_size, num_anchors, box_data_length,
                max_output_size, iou_threshold,
                force_suppress, top_k,
                coord_start, score_index, id_index,
                return_indices,
                outs[0], outs[1], outs[2],
                soft_nms_sigma, score_threshold,
            ),
            dtype=[data.dtype, "int32", "int32"],
            out_buffers=[out_data_buf, out_box_indices_buf, out_valid_box_count_buf],
            in_buffers=[data_buf, sort_buf, valid_count_buf, indices_buf],
            name="non_max_suppression",
            tag="non_max_suppression",
        )
        if soft_nms_sigma > 0.0:
            return [out_data, out_box_indices, out_valid_box_count]
        return [out_box_indices, out_valid_box_count]

    out_data, out_box_indices = te.extern(
        [data.shape, (batch_size, num_anchors)],
        [data, sort_tensor, valid_count, indices],
        lambda ins, outs: _classic_nms_ir(
            ins[0], ins[1], ins[2], ins[3],
            batch_size, num_anchors, box_data_length,
            max_output_size, iou_threshold,
            force_suppress, top_k,
            coord_start, score_index, id_index,
            return_indices,
            outs[0], outs[1], None,
            soft_nms_sigma, score_threshold,
        ),
        dtype=[data.dtype, "int32"],
        out_buffers=[out_data_buf, out_box_indices_buf],
        in_buffers=[data_buf, sort_buf, valid_count_buf, indices_buf],
        name="non_max_suppression",
        tag="non_max_suppression",
    )

    if invalid_to_bottom:
        # Rearrange to move valid boxes to top
        return _rearrange_out(out_data, batch_size, num_anchors, box_data_length, score_index)

    return out_data


def _rearrange_out(data, batch_size, num_anchors, box_data_length, score_index):
    """Move valid boxes (score >= 0) to the top of output."""
    out_buf = tvm.tirx.decl_buffer(
        (batch_size, num_anchors, box_data_length), data.dtype, "rearranged"
    )

    def _rearrange_ir(ins, outs):
        with IRBuilder() as ib:
            data = T.buffer_proxy(ins[0])
            out = T.buffer_proxy(outs[0])

            with T.parallel(0, batch_size) as i:
                valid_idx_buf = T.alloc_buffer((1,), "int32", scope="local")
                valid_idx = T.buffer_proxy(valid_idx_buf)
                valid_idx[0] = T.int32(0)

                with T.serial(0, num_anchors) as j:
                    with T.If(
                        data[i, j, score_index] >= tvm.tirx.Cast(data.dtype, T.float32(0.0))
                    ):
                        with T.Then():
                            with T.serial(0, box_data_length) as k:
                                out[i, valid_idx[0], k] = data[i, j, k]
                            valid_idx[0] = valid_idx[0] + 1

                with T.serial(0, num_anchors) as j:
                    with T.If(j >= valid_idx[0]):
                        with T.Then():
                            with T.serial(0, box_data_length) as k:
                                out[i, j, k] = tvm.tirx.Cast(data.dtype, T.float32(-1.0))

            return ib.get()

    return te.extern(
        [(batch_size, num_anchors, box_data_length)],
        [data],
        _rearrange_ir,
        dtype=[data.dtype],
        out_buffers=[out_buf],
        name="rearrange_out",
        tag="rearrange_out",
    )


def _nms_loop(
    batch_size,
    top_k,
    iou_threshold,
    max_output_size,
    valid_count,
    on_new_valid_box_func,
    on_new_invalidated_box_func,
    needs_bbox_check_func,
    calc_overlap_func,
    out_scores,
    num_valid_boxes,
    score_threshold=None,
):
    """NMS loop using modern IRBuilder. Must be called within IRBuilder context."""
    out_scores = T.buffer_proxy(out_scores)
    num_valid_boxes = T.buffer_proxy(num_valid_boxes)

    def nms_inner_loop(i, j, nkeep, num_valid_boxes_local):
        on_new_valid_box_func(0, num_valid_boxes_local[0], i, j)
        num_valid_boxes_local[0] = num_valid_boxes_local[0] + 1

        num_boxes_to_check = nkeep - (j + 1)

        with T.parallel(0, num_boxes_to_check) as _k:
            k = j + 1 + _k

            with T.If(
                tvm.tirx.all(
                    k < nkeep,
                    out_scores[i, k] > 0,  # is the box k still valid?
                    needs_bbox_check_func(i, j, k),
                )
            ):
                with T.Then():
                    iou = calc_overlap_func(i, j, k)

                    with T.If(iou >= iou_threshold):
                        with T.Then():
                            out_scores[i, k] = T.float32(-1.0)
                            on_new_invalidated_box_func(i, k)

    with T.serial(0, batch_size) as i:
        nkeep = if_then_else(tvm.tirx.all(top_k > 0, top_k < valid_count[i]), top_k, valid_count[i])

        with T.If(tvm.tirx.all(iou_threshold > te.const(0), valid_count[i] > te.const(0))):
            with T.Then():
                num_valid_boxes_local_buf = T.alloc_buffer((1,), "int32", scope="local")
                num_valid_boxes_local = T.buffer_proxy(num_valid_boxes_local_buf)
                num_valid_boxes_local[0] = T.int32(0)

                with T.serial(0, nkeep) as j:
                    with T.If(
                        tvm.tirx.all(
                            out_scores[i, j] > -1.0,  # box is still valid
                            num_valid_boxes_local[0] < max_output_size,  # haven't reached max limit
                        )
                    ):
                        with T.Then():
                            if score_threshold is not None:
                                with T.If(out_scores[i, j] > score_threshold[()]):
                                    with T.Then():
                                        nms_inner_loop(i, j, nkeep, num_valid_boxes_local)
                            else:
                                nms_inner_loop(i, j, nkeep, num_valid_boxes_local)

                num_valid_boxes[i] = num_valid_boxes_local[0]

            with T.Else():
                num_valid_boxes[i] = T.int32(0)


def _get_valid_box_count(scores, score_threshold):
    batch_classes, num_boxes = scores.shape

    def searchsorted_ir(scores_buf, score_thresh_buf, valid_count_buf):
        with IRBuilder() as ib:
            with T.parallel(0, batch_classes) as i:
                if hasattr(score_threshold, "shape"):
                    if len(score_threshold.shape) == 0:
                        score_thresh_scalar = score_thresh_buf[()]
                    elif len(score_threshold.shape) == 1 and score_threshold.shape[0] > 0:
                        score_thresh_scalar = score_thresh_buf[0]
                    else:
                        score_thresh_scalar = tvm.tirx.FloatImm("float32", 0.0)
                else:
                    score_thresh_scalar = score_threshold
                binary_search(i, num_boxes, scores_buf, score_thresh_scalar, valid_count_buf)

            return ib.get()

    scores_buf = tvm.tirx.decl_buffer(scores.shape, scores.dtype, "scores_buf", data_alignment=8)
    searchsorted_buf = tvm.tirx.decl_buffer(
        (batch_classes,), "int32", "searchsorted", data_alignment=8
    )

    if hasattr(score_threshold, "shape"):
        score_thresh_buf = tvm.tirx.decl_buffer(
            score_threshold.shape, score_threshold.dtype, "score_thresh_buf", data_alignment=8
        )
        return te.extern(
            [(batch_classes,)],
            [scores, score_threshold],
            lambda ins, outs: searchsorted_ir(ins[0], ins[1], outs[0]),
            dtype=["int32"],
            in_buffers=[scores_buf, score_thresh_buf],
            out_buffers=[searchsorted_buf],
            name="searchsorted",
            tag="searchsorted",
        )
    else:

        def searchsorted_ir_scalar(scores_buf, valid_count_buf):
            with IRBuilder() as ib:
                with T.parallel(0, batch_classes) as i:
                    if isinstance(score_threshold, te.Tensor):
                        if len(score_threshold.shape) == 0:
                            score_thresh_tir = score_threshold()
                        elif len(score_threshold.shape) == 1 and score_threshold.shape[0] == 1:
                            score_thresh_tir = score_threshold[0]
                        else:
                            score_thresh_tir = tvm.tirx.FloatImm("float32", 0.0)
                    else:
                        score_thresh_tir = tvm.tirx.FloatImm("float32", float(score_threshold))
                    binary_search(i, num_boxes, scores_buf, score_thresh_tir, valid_count_buf)

                return ib.get()

        return te.extern(
            [(batch_classes,)],
            [scores],
            lambda ins, outs: searchsorted_ir_scalar(ins[0], outs[0]),
            dtype=["int32"],
            in_buffers=[scores_buf],
            out_buffers=[searchsorted_buf],
            name="searchsorted",
            tag="searchsorted",
        )


def _collect_selected_indices_ir(
    num_class, selected_indices, num_detections, row_offsets, out, max_output_boxes_per_class=None
):
    batch_classes, _ = selected_indices.shape

    with IRBuilder() as ib:
        with T.seq_scope():
            out = T.buffer_proxy(out)

            # Initialize output buffer to zero
            # Calculate the actual output shape based on max_output_boxes_per_class
            if isinstance(max_output_boxes_per_class, int):
                max_output_rows = batch_classes * max_output_boxes_per_class
            else:
                # Fallback to a reasonable default if max_output_boxes_per_class is not an integer
                max_output_rows = batch_classes * 10
            with T.serial(0, max_output_rows) as init_i:
                with T.serial(0, 3) as init_j:  # 3 columns
                    out[init_i, init_j] = cast(0, "int64")

            with T.parallel(0, batch_classes) as i:
                i_64 = cast(i, "int64")
                batch_id = i_64 // num_class
                class_id = i_64 % num_class

                if isinstance(max_output_boxes_per_class, int):
                    limit = tvm.tirx.min(
                        num_detections[i], tvm.tirx.IntImm("int32", max_output_boxes_per_class)
                    )
                elif isinstance(max_output_boxes_per_class, te.Tensor):
                    if len(max_output_boxes_per_class.shape) == 0:
                        max_boxes_val = max_output_boxes_per_class[()]
                    else:
                        max_boxes_val = max_output_boxes_per_class[0]
                    limit = tvm.tirx.min(num_detections[i], max_boxes_val)
                else:
                    limit = num_detections[i]

                with T.serial(0, limit) as j:
                    out[row_offsets[i] + j, 0] = batch_id
                    out[row_offsets[i] + j, 1] = class_id
                    out[row_offsets[i] + j, 2] = cast(selected_indices[i, j], "int64")

        return ib.get()


def _collect_selected_indices_and_scores_ir(
    selected_indices,
    selected_scores,
    num_detections,
    row_offsets,
    num_total_detections,
    collected_indices,
    collected_scores,
):
    batch_size, num_class = row_offsets.shape
    num_boxes = selected_indices.shape[1]

    with IRBuilder() as ib:
        collected_indices = T.buffer_proxy(collected_indices)
        collected_scores = T.buffer_proxy(collected_scores)
        zero = cast(0, "int64")

        with T.parallel(0, batch_size * num_class) as i:
            i_64 = cast(i, "int64")
            batch_id = i_64 // num_class
            class_id = i_64 % num_class

            with T.serial(0, num_boxes) as j:
                with T.If(j < num_detections[batch_id, class_id]):
                    with T.Then():
                        offset = row_offsets[batch_id, class_id] + j
                        collected_indices[batch_id, offset, 0] = class_id
                        collected_indices[batch_id, offset, 1] = cast(
                            selected_indices[i, j], "int64"
                        )
                        collected_scores[batch_id, offset] = selected_scores[i, j]
                    with T.Else():
                        offset = (
                            num_total_detections[batch_id]
                            + class_id * num_boxes
                            - row_offsets[batch_id, class_id]
                            + j
                            - num_detections[batch_id, class_id]
                        )
                        collected_indices[batch_id, offset, 0] = zero
                        collected_indices[batch_id, offset, 1] = zero
                        collected_scores[batch_id, offset] = T.float32(0.0)

        return ib.get()


def all_class_non_max_suppression(
    boxes,
    scores,
    max_output_boxes_per_class,
    iou_threshold,
    score_threshold,
    output_format="onnx",
    output_shape=None,
):
    """Non-maximum suppression operator for object detection, corresponding to ONNX
    NonMaxSuppression and TensorFlow combined_non_max_suppression.
    NMS is performed for each class separately.

    Parameters
    ----------
    boxes : tvm.te.Tensor
        3-D tensor with shape (batch_size, num_boxes, 4)
    scores : tvm.te.Tensor
        3-D tensor with shape (batch_size, num_classes, num_boxes)
    max_output_boxes_per_class : int or tvm.te.Tensor, optional
        The maxinum number of output selected boxes per class
    iou_threshold : float or tvm.te.Tensor, optional
        IoU test threshold
    score_threshold : float or tvm.te.Tensor, optional
        Score threshold to filter out low score boxes early
    output_format : str, optional
        "onnx" or "tensorflow", see below.

    Returns
    -------
    out : list of tvm.te.Tensor
        If `output_format` is "onnx", the output is two tensors. The first is `indices` of size
        `(batch_size * num_class* num_boxes , 3)` and the second is a scalar tensor
        `num_total_detection` of shape `(1,)` representing the total number of selected
        boxes. The three values in `indices` encode batch, class, and box indices.
        Rows of `indices` are ordered such that selected boxes from batch 0, class 0 come
        first, in descending of scores, followed by boxes from batch 0, class 1 etc. Out of
        `batch_size * num_class* num_boxes` rows of indices, only the first `num_total_detection`
        rows are valid.

        .. note::
            **Important**: The output tensor has a fixed size based on `max_output_boxes_per_class`,
            but only the first `num_total_detection` rows contain valid data. The remaining rows
            may contain garbage values. When comparing with ONNX Runtime or other implementations
            that output dynamic shapes, you should only compare the first
            `num_total_detection` rows.
            Example::

                selected_indices, valid_count = nms_output
                actual_count = int(valid_count.numpy()[0])
                valid_indices = selected_indices.numpy()[:actual_count, :]

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
    batch, num_class, num_boxes = scores.shape
    scores = reshape(scores, (batch * num_class, num_boxes))

    sorted_indices = argsort(scores, axis=1, is_ascend=False, dtype="int32")
    sorted_scores = gather(scores, 1, sorted_indices)

    if not isinstance(score_threshold, te.Tensor):
        score_threshold_tensor = te.compute((), lambda: score_threshold, name="score_threshold")
    else:
        score_threshold_tensor = score_threshold

    valid_count = _get_valid_box_count(sorted_scores, score_threshold_tensor)

    selected_indices, selected_scores, num_detections = run_all_class_nms(
        boxes,
        sorted_scores,
        sorted_indices,
        valid_count,
        max_output_boxes_per_class,
        iou_threshold,
        _nms_loop,
        return_scores=(output_format == "tensorflow"),
        score_threshold=score_threshold_tensor,  # Passed score_threshold as tensor
    )

    if output_format == "onnx":
        row_offsets = cumsum(num_detections, exclusive=True, dtype="int64")

        def _sum_clamped_total():
            if isinstance(max_output_boxes_per_class, int):
                k_expr = tvm.tirx.IntImm("int32", int(max_output_boxes_per_class))
                clamped = te.compute(
                    num_detections.shape,
                    lambda i: tvm.tirx.min(num_detections[i], k_expr),
                    name="clamped_num",
                )
                return reduction.sum(cast(clamped, "int64"), axis=0)
            if isinstance(max_output_boxes_per_class, tvm.tirx.IntImm):
                k_expr = tvm.tirx.Cast("int32", max_output_boxes_per_class)
                clamped = te.compute(
                    num_detections.shape,
                    lambda i: tvm.tirx.min(num_detections[i], k_expr),
                    name="clamped_num",
                )
                return reduction.sum(cast(clamped, "int64"), axis=0)
            if isinstance(max_output_boxes_per_class, te.Tensor):
                if len(max_output_boxes_per_class.shape) == 0:
                    kb = te.compute(
                        num_detections.shape,
                        lambda i: cast(max_output_boxes_per_class, "int32"),
                        name="k_broadcast",
                    )
                elif (
                    len(max_output_boxes_per_class.shape) == 1
                    and max_output_boxes_per_class.shape[0] == 1
                ):
                    kb = te.compute(
                        num_detections.shape,
                        lambda i: cast(max_output_boxes_per_class[0], "int32"),
                        name="k_broadcast",
                    )
                else:
                    return reduction.sum(cast(num_detections, "int64"), axis=0)

                clamped = te.compute(
                    num_detections.shape,
                    lambda i: tvm.tirx.min(num_detections[i], kb[i]),
                    name="clamped_num",
                )
                return reduction.sum(cast(clamped, "int64"), axis=0)
            return reduction.sum(cast(num_detections, "int64"), axis=0)

        num_total_scalar = _sum_clamped_total()
        num_total_detections = reshape(num_total_scalar, (1,))

        if output_shape is not None:
            selected_indices = collect_selected_indices(
                num_class,
                selected_indices,
                num_detections,
                row_offsets,
                _collect_selected_indices_ir,
                max_output_boxes_per_class=max_output_boxes_per_class,
                output_shape=output_shape,
            )
        else:
            # Use num_total_detections to enable dynamic trimming
            # Pass image size for intelligent default estimation
            input_image_size = None
            if hasattr(scores, "shape") and len(scores.shape) >= 3:
                # Extract image size from scores shape: (batch, num_classes, num_boxes)
                # We can estimate image size from num_boxes (more boxes = larger image)
                input_image_size = (scores.shape[2],)  # Use num_boxes as proxy for image size

                # TODO: Improve image size estimation by:
                # 1. Accepting actual image dimensions as parameters
                # 2. Using model metadata to infer typical image sizes
                # 3. Learning from historical detection patterns
                # 4. Providing user-configurable estimation strategies

            selected_indices = collect_selected_indices(
                num_class,
                selected_indices,
                num_detections,
                row_offsets,
                _collect_selected_indices_ir,
                max_output_boxes_per_class=max_output_boxes_per_class,
                num_total_detections=num_total_detections,
                input_image_size=input_image_size,
            )
        return [selected_indices, num_total_detections]

    num_detections_per_batch = reshape(num_detections, (batch, num_class))
    row_offsets = cumsum(num_detections_per_batch, exclusive=True, dtype="int64", axis=1)
    num_total_detections = reduction.sum(cast(num_detections_per_batch, "int64"), axis=1)

    selected_indices, selected_scores = collect_selected_indices_and_scores(
        selected_indices,
        selected_scores,
        num_detections_per_batch,
        row_offsets,
        num_total_detections,
        _collect_selected_indices_and_scores_ir,
    )

    return [selected_indices, selected_scores, num_total_detections]
