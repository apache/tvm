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

from tvm.te import hybrid
from tvm.tir import if_then_else

from ..sort import argsort
from ..math import cast
from ..transform import reshape, gather
from .. import reduction
from ..scan import cumsum
from .nms_util import (
    binary_search,
    collect_selected_indices,
    collect_selected_indices_and_scores,
    run_all_class_nms,
)


@hybrid.script
def hybrid_rearrange_box_out(data, one, batch_size, num_anchors):
    """Hybrid routine to rearrange nms output to
    move all valid entries to top.

    Parameters
    ----------
    data : tvm.te.Tensor or numpy NDArray
        NMS output. 3-D tensor with shape
        [batch_size, num_anchors, 6].

    one: tvm.tir.const
        Constant one with the same dtype as data.

    batch_size: tvm.tir.IntImm or tvm.tir.Var
        Batch size. We need to pass it in since hybrid script doesn't support
        binding variable to symbolic dim.

    num_anchors: tvm.tir.IntImm or tvm.tir.Var
        Number of anchors.

    Returns
    -------
    output : tvm.te.Tensor or numpy NDArray
        Transformed NMS output. 3-D tensor with shape
        [batch_size, num_anchors, 6].
    """
    elem_length = data.shape[2]
    output = output_tensor((batch_size, num_anchors, elem_length), data.dtype)
    valid_indices = allocate((batch_size,), "int32")

    for i in parallel(batch_size):
        valid_indices[i] = 0
        for j in range(num_anchors):
            if data[i, j, 0] >= 0:
                for k in range(elem_length):
                    output[i, valid_indices[i], k] = data[i, j, k]
                valid_indices[i] += 1
            if j >= valid_indices[i]:
                for k in range(elem_length):
                    output[i, j, k] = -one
    return output


@hybrid.script
def hybrid_rearrange_indices_out(data, one, batch_size, num_anchors):
    """Hybrid routine to rearrange nms output to
    move all valid entries to top.

    Parameters
    ----------
    data : tvm.te.Tensor or numpy NDArray
        NMS output. 3-D tensor with shape
        [batch_size, num_anchors, 6] or
        [batch_size, num_anchors, 5], or 2-D
        tensor with shape [batch_size, num_anchors].

    one: tvm.tir.const
        Constant one with the same dtype as data.

    batch_size: tvm.tir.IntImm or tvm.tir.Var
        Batch size. We need to pass it in since hybrid script doesn't support
        binding variable to symbolic dim.

    num_anchors: tvm.tir.IntImm or tvm.tir.Var
        Number of anchors.

    Returns
    -------
    output : tvm.te.Tensor or numpy NDArray
        2-D tensor with shape [batch_size, num_anchors].

    valid_box_count : tvm.te.Tensor or numpy NDArray
        Tensor with shape [batch_size, 1], indicates
        the valid number of boxes.
    """
    valid_box_count = output_tensor((batch_size, 1), "int32")
    output = output_tensor((batch_size, num_anchors), data.dtype)
    valid_indices = allocate((batch_size,), "int32")

    for i in parallel(batch_size):
        valid_indices[i] = 0
        for j in range(num_anchors):
            if data[i, j] >= 0:
                output[i, valid_indices[i]] = data[i, j]
                valid_indices[i] += 1
            if data[i, j] > num_anchors or data[i, j] < -num_anchors:
                output[i, valid_indices[i]] = 0
                valid_indices[i] += 1
            if j >= valid_indices[i]:
                output[i, j] = -one
        valid_box_count[i, 0] = valid_indices[i]

    return output, valid_box_count


@hybrid.script
def hybrid_get_valid_counts(
    data, score_threshold, id_index, score_index, one, batch_size, num_anchors
):
    """Hybrid routine to get valid count of bounding boxes
    given a score threshold. Also moves valid boxes to the
    top of input data.

    Parameters
    ----------
    data : tvm.te.Tensor or numpy NDArray
        Input data. 3-D tensor with shape [batch_size, num_anchors, 6]
        or [batch_size, num_anchors, 5].

    score_threshold : tvm.te.Tensor
        Lower limit of score for valid bounding boxes.

    id_index : tvm.tir.const
        index of the class categories, -1 to disable.

    score_index: tvm.tir.const
        Index of the scores/confidence of boxes.

    one: tvm.tir.const
        Constant one with the same dtype as data.

    batch_size: tvm.tir.IntImm or tvm.tir.Var
        Batch size. We need to pass it in since hybrid script doesn't support
        binding variable to symbolic dim.

    num_anchors: tvm.tir.IntImm or tvm.tir.Var
        Number of anchors.

    Returns
    -------
    valid_count : tvm.te.Tensor or numpy NDArray
        1-D tensor for valid number of boxes.

    out_tensor : tvm.te.Tensor or numpy NDArray
        Rearranged data tensor.

    out_indices: tvm.te.Tensor or numpy NDArray
        Related index in input data.
    """
    box_data_length = data.shape[2]
    valid_count = output_tensor((batch_size,), "int32")
    out_tensor = output_tensor((batch_size, num_anchors, box_data_length), data.dtype)
    out_indices = output_tensor((batch_size, num_anchors), "int32")
    for i in parallel(batch_size):
        valid_count[i] = 0
        for j in range(num_anchors):
            score = data[i, j, score_index]
            if score > score_threshold and (id_index < 0 or data[i, j, id_index] >= 0):
                for k in range(box_data_length):
                    out_tensor[i, valid_count[i], k] = data[i, j, k]
                out_indices[i, valid_count[i]] = j
                valid_count[i] += 1
            if j >= valid_count[i]:
                for k in range(box_data_length):
                    out_tensor[i, j, k] = -one
                out_indices[i, j] = -1
    return valid_count, out_tensor, out_indices


def get_valid_counts(data, score_threshold=0, id_index=0, score_index=1):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data. 3-D tensor with shape [batch_size, num_anchors, 6]
        or [batch_size, num_anchors, 5].

    score_threshold : optional, float
        Lower limit of score for valid bounding boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    Returns
    -------
    valid_count : tvm.te.Tensor
        1-D tensor for valid number of boxes.

    out_tensor : tvm.te.Tensor
        Rearranged data tensor.

    out_indices: tvm.te.Tensor or numpy NDArray
        Related index in input data.
    """
    if isinstance(score_threshold, (float, int)):
        score_threshold = tvm.tir.const(score_threshold, dtype=data.dtype)
    id_index_const = tvm.tir.const(id_index, "int32")
    score_index_const = tvm.tir.const(score_index, "int32")
    return hybrid_get_valid_counts(
        data,
        score_threshold,
        id_index_const,
        score_index_const,
        tvm.tir.const(1, data.dtype),
        data.shape[0],
        data.shape[1],
    )


@hybrid.script
def hybrid_nms(
    data,
    sorted_index,
    valid_count,
    indices,
    batch_size,
    num_anchors,
    max_output_size,
    iou_threshold,
    force_suppress,
    top_k,
    coord_start,
    score_index,
    id_index,
    return_indices,
    zero,
    one,
):
    """Hybrid routing for non-maximum suppression.

    Parameters
    ----------
    data: tvm.te.Tensor or numpy NDArray
        Bounding boxes with class and score. 3-D tensor with shape
        [batch_size, num_anchors, 6]. It could be the second output
        out_tensor of get_valid_counts.

    sorted_index : tvm.te.Tensor or numpy NDArray
        Bounding box indexes sorted by score, with shape
        [batch_size, num_anchors].

    valid_count : tvm.te.Tensor or numpy NDArray
        1-D tensor for valid number of boxes. It could be the output
        valid_count of get_valid_counts.

    indices : tvm.te.Tensor or numpy.NDArray
        indices in original tensor, with shape [batch_size, num_anchors],
        represents the index of box in original data. It could be the third
        output out_indices of get_valid_counts. The values in the second
        dimension are like the output of arange(num_anchors) if get_valid_counts
        is not used before non_max_suppression.

    batch_size: tvm.tir.IntImm or tvm.tir.Var
        Batch size. We need to pass it in since hybrid script doesn't support
        binding variable to symbolic dim.

    num_anchors: tvm.tir.IntImm or tvm.tir.Var
        The number of anchors.

    max_output_size : tvm.te.Tensor
        Max number of output valid boxes for each instance.
        Return all valid boxes if max_output_size < 0.

    iou_threshold : tvm.te.Tensor
        Overlapping(IoU) threshold to suppress object with smaller score.

    force_suppress : tvm.tir.const
        Whether to suppress all detections regardless of class_id.

    top_k : tvm.tir.const
        Keep maximum top k detections before nms, -1 for no limit.

    coord_start : tvm.tir.const
        Start index of the consecutive 4 coordinates.

    score_index: tvm.tir.const
        Index of the scores/confidence of boxes.

    id_index : tvm.tir.const
        index of the class categories, -1 to disable.

    return_indices : tvm.tir.const
        Whether to return box indices in input data.

    zero: tvm.tir.const
        Constant zero with the same dtype as data.

    one: tvm.tir.const
        Constant one with the same dtype as data.

    Returns
    -------
    output : tvm.te.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6]
        or [batch_size, num_anchors, 5].

    box_indices: tvm.te.Tensor
        2-D tensor with shape [batch_size, num_anchors].
    """

    box_data_length = data.shape[2]

    # box_indices is the expected indices of boxes
    box_indices = output_tensor((batch_size, num_anchors), sorted_index.dtype)
    output = output_tensor(
        (
            batch_size,
            num_anchors,
            box_data_length,
        ),
        data.dtype,
    )

    for i in range(batch_size):
        if iou_threshold > 0:
            if valid_count[i] > 0:
                # Reorder output
                nkeep = valid_count[i]
                if 0 < top_k < nkeep:
                    nkeep = top_k
                for j in parallel(nkeep):
                    for k in range(box_data_length):
                        output[i, j, k] = data[i, sorted_index[i, j], k]
                    box_indices[i, j] = sorted_index[i, j]
                if 0 < top_k < valid_count[i]:
                    for j in parallel(valid_count[i] - nkeep):
                        for k in range(box_data_length):
                            output[i, j + nkeep, k] = -one
                        box_indices[i, j + nkeep] = -1

            # Apply nms
            box_start_idx = coord_start
            batch_idx = i
            num_valid_boxes = 0

            for j in range(valid_count[i]):
                if num_valid_boxes == max_output_size:
                    for k in range(box_data_length):
                        output[i, j, k] = -one
                    box_indices[i, j] = -1

                elif output[i, j, score_index] > 0:
                    box_a_idx = j
                    is_valid_box = 1

                    # a_l: left, a_t: top, a_r: right, a_b: bottom
                    a_l = min(
                        output[batch_idx, box_a_idx, box_start_idx],
                        output[batch_idx, box_a_idx, box_start_idx + 2],
                    )
                    a_t = min(
                        output[batch_idx, box_a_idx, box_start_idx + 1],
                        output[batch_idx, box_a_idx, box_start_idx + 3],
                    )
                    a_r = max(
                        output[batch_idx, box_a_idx, box_start_idx],
                        output[batch_idx, box_a_idx, box_start_idx + 2],
                    )
                    a_b = max(
                        output[batch_idx, box_a_idx, box_start_idx + 1],
                        output[batch_idx, box_a_idx, box_start_idx + 3],
                    )

                    # check if current box j is valid by calculating iou with
                    # all existing valid boxes
                    for k in range(j):
                        check_iou = 0
                        if (
                            is_valid_box == 1
                            and k < j
                            and output[i, k, score_index] > 0
                            and (id_index < 0 or output[i, k, id_index] >= 0)
                        ):
                            if force_suppress:
                                check_iou = 1
                            elif id_index < 0 or output[i, j, id_index] == output[i, k, id_index]:
                                check_iou = 1

                        if check_iou > 0:
                            box_b_idx = k

                            # b_l: left, b_t: top, b_r: right, b_b: bottom
                            b_l = min(
                                output[batch_idx, box_b_idx, box_start_idx],
                                output[batch_idx, box_b_idx, box_start_idx + 2],
                            )
                            b_t = min(
                                output[batch_idx, box_b_idx, box_start_idx + 1],
                                output[batch_idx, box_b_idx, box_start_idx + 3],
                            )
                            b_r = max(
                                output[batch_idx, box_b_idx, box_start_idx],
                                output[batch_idx, box_b_idx, box_start_idx + 2],
                            )
                            b_b = max(
                                output[batch_idx, box_b_idx, box_start_idx + 1],
                                output[batch_idx, box_b_idx, box_start_idx + 3],
                            )

                            # Overlapping width and height
                            w = max(zero, min(a_r, b_r) - max(a_l, b_l))
                            h = max(zero, min(a_b, b_b) - max(a_t, b_t))

                            # Overlapping area
                            area = h * w

                            # total area of the figure formed by box a and box b
                            # except for overlapping area
                            u = (a_r - a_l) * (a_b - a_t) + (b_r - b_l) * (b_b - b_t) - area

                            # get the iou
                            iou = zero if u <= zero else area / u

                            if iou >= iou_threshold:
                                is_valid_box = 0

                    if is_valid_box == 0:
                        for k in range(box_data_length):
                            output[i, j, k] = -one
                        box_indices[i, j] = -1
                    else:
                        num_valid_boxes += 1

        else:
            for j in parallel(valid_count[i]):
                for k in range(box_data_length):
                    output[i, j, k] = data[i, j, k]
                box_indices[i, j] = j

        # Set invalid entry to be -1
        for j in parallel(num_anchors - valid_count[i]):
            for k in range(box_data_length):
                output[i, j + valid_count[i], k] = -one
            box_indices[i, j + valid_count[i]] = -1

        if return_indices:
            for j in range(valid_count[i]):
                idx = box_indices[i, j]
                if box_indices[i, j] >= 0:
                    box_indices[i, j] = indices[i, idx]

    return output, box_indices


@tvm.target.generic_func
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
    data : tvm.te.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6] or [batch_size, num_anchors, 5].

    valid_count : tvm.te.Tensor
        1-D tensor for valid number of boxes.

    indices : tvm.te.Tensor
        2-D tensor with shape [batch_size, num_anchors].

    max_output_size : optional, int or tvm.te.Tensor
        Max number of output valid boxes for each instance.
        Return all valid boxes if the value of max_output_size is less than 0.

    iou_threshold : optional, float or tvm.te.Tensor
        Non-maximum suppression threshold.

    force_suppress : optional, boolean
        Whether to suppress all detections regardless of class_id.

    top_k : optional, int
        Keep maximum top k detections before nms, -1 for no limit.

    coord_start : required, int
        Start index of the consecutive 4 coordinates.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    return_indices : optional, boolean
        Whether to return box indices in input data.

    invalid_to_bottom : optional, boolean
        Whether to move all valid bounding boxes to the top.

    Returns
    -------
    out : tvm.te.Tensor or tuple of tvm.te.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6]
        or [batch_size, num_anchors, 5]. Out is a tuple of tvm.te.Tensor
        if return_indices is True, the Tensor in the tuple is 2-D tensor
        with shape [batch_size, num_anchors] and shape
        [batch_size, num_valid_anchors] respectively.

    Example
    --------
    .. code-block:: python

        # An example to use non_max_suppression
        dshape = (1, 5, 6)
        data = te.placeholder(dshape, name="data")
        valid_count = te.placeholder((dshape[0],), dtype="int32", name="valid_count")
        iou_threshold = 0.7
        force_suppress = True
        top_k = -1
        out = non_max_suppression(data, valid_count, indices, iou_threshold=iou_threshold,
                                  force_suppress=force_suppress, top_k=top_k)
        np_data = np.random.uniform(dshape)
        np_valid_count = np.array([4])
        s = topi.generic.schedule_nms(out)
        f = tvm.build(s, [data, valid_count, out], "llvm")
        dev = tvm.cpu()
        tvm_data = tvm.nd.array(np_data, dev)
        tvm_valid_count = tvm.nd.array(np_valid_count, dev)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data.dtype), dev)
        f(tvm_data, tvm_valid_count, tvm_out)
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    if isinstance(max_output_size, int):
        max_output_size = tvm.tir.const(max_output_size, dtype="int32")
    if isinstance(iou_threshold, float):
        iou_threshold = tvm.tir.const(iou_threshold, dtype=data.dtype)
    score_axis = score_index
    score_shape = (batch_size, num_anchors)
    score_tensor = te.compute(score_shape, lambda i, j: data[i, j, score_axis])
    sort_tensor = argsort(score_tensor, valid_count=valid_count, axis=1, is_ascend=False)

    out, box_indices = hybrid_nms(
        data,
        sort_tensor,
        valid_count,
        indices,
        batch_size,
        num_anchors,
        max_output_size,
        iou_threshold,
        tvm.tir.const(force_suppress, dtype="bool"),
        tvm.tir.const(top_k, dtype="int32"),
        tvm.tir.const(coord_start, dtype="int32"),
        tvm.tir.const(score_index, dtype="int32"),
        tvm.tir.const(id_index, dtype="int32"),
        tvm.tir.const(return_indices, dtype="bool"),
        zero=tvm.tir.const(0, dtype=data.dtype),
        one=tvm.tir.const(1, dtype=data.dtype),
    )

    if return_indices:
        return hybrid_rearrange_indices_out(
            box_indices,
            one=tvm.tir.const(1, dtype="int32"),
            batch_size=batch_size,
            num_anchors=num_anchors,
        )

    if invalid_to_bottom:
        out = hybrid_rearrange_box_out(
            out,
            one=tvm.tir.const(1, dtype=data.dtype),
            batch_size=batch_size,
            num_anchors=num_anchors,
        )
    return out


def _nms_loop(
    ib,
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
):
    def nms_inner_loop(ib, i, j, nkeep, num_valid_boxes_local):
        # The box j is valid, invalidate other boxes that overlap with j above iou_threshold
        on_new_valid_box_func(ib, 0, num_valid_boxes_local[0], i, j)
        num_valid_boxes_local[0] += 1

        num_boxes_to_check = nkeep - (j + 1)

        with ib.for_range(0, num_boxes_to_check, name="_k", kind="parallel") as _k:
            k = j + 1 + _k

            with ib.if_scope(
                tvm.tir.all(
                    k < nkeep,
                    out_scores[i, k] > 0,  # is the box k still valid?
                    needs_bbox_check_func(i, j, k),
                )
            ):
                iou = calc_overlap_func(i, j, k)

                with ib.if_scope(iou >= iou_threshold):
                    # invalidate the box k
                    out_scores[i, k] = -1.0
                    on_new_invalidated_box_func(i, k)

    with ib.for_range(0, batch_size, name="i") as i:
        nkeep = if_then_else(tvm.tir.all(top_k > 0, top_k < valid_count[i]), top_k, valid_count[i])
        max_output_size = if_then_else(max_output_size > 0, max_output_size, nkeep)

        with ib.if_scope(tvm.tir.all(iou_threshold > 0, valid_count[i] > 0)):
            num_valid_boxes_local = ib.allocate(
                "int32", (1,), name="num_valid_boxes_local", scope="local"
            )
            box_idx = ib.allocate("int32", (1,), name="box_idx", scope="local")
            num_valid_boxes_local[0] = 0
            box_idx[0] = 0

            # Apply nms
            # No need to do more iteration if we have already reached max_output_size boxes
            with ib.while_loop(
                tvm.tir.all(box_idx[0] < nkeep, num_valid_boxes_local[0] < max_output_size)
            ):
                # Proceed to the inner loop if the box with id box_idx is still valid
                with ib.if_scope(out_scores[i, box_idx[0]] > -1.0):
                    nms_inner_loop(ib, i, box_idx[0], nkeep, num_valid_boxes_local)
                box_idx[0] += 1

            num_valid_boxes[i] = num_valid_boxes_local[0]

        with ib.else_scope():
            num_valid_boxes[i] = 0

    return ib.get()


def _get_valid_box_count(scores, score_threshold):
    batch_classes, num_boxes = scores.shape

    def searchsorted_ir(scores, valid_count):
        ib = tvm.tir.ir_builder.create()
        scores = ib.buffer_ptr(scores)
        valid_count = ib.buffer_ptr(valid_count)

        with ib.for_range(0, batch_classes, name="i", kind="parallel") as i:
            binary_search(ib, i, num_boxes, scores, score_threshold, valid_count)

        return ib.get()

    scores_buf = tvm.tir.decl_buffer(scores.shape, scores.dtype, "scores_buf", data_alignment=8)

    return te.extern(
        [(batch_classes,)],
        [scores],
        lambda ins, outs: searchsorted_ir(ins[0], outs[0]),
        dtype=["int32"],
        in_buffers=[scores_buf],
        name="searchsorted",
        tag="searchsorted",
    )


def _collect_selected_indices_ir(num_class, selected_indices, num_detections, row_offsets, out):
    batch_classes, _ = selected_indices.shape

    ib = tvm.tir.ir_builder.create()

    selected_indices = ib.buffer_ptr(selected_indices)
    num_detections = ib.buffer_ptr(num_detections)
    row_offsets = ib.buffer_ptr(row_offsets)
    out = ib.buffer_ptr(out)

    with ib.for_range(0, batch_classes, name="i", kind="parallel") as i:
        i = cast(i, "int64")
        batch_id = i // num_class
        class_id = i % num_class

        with ib.for_range(0, num_detections[i], name="j") as j:
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

    ib = tvm.tir.ir_builder.create()

    selected_indices = ib.buffer_ptr(selected_indices)
    selected_scores = ib.buffer_ptr(selected_scores)
    num_detections = ib.buffer_ptr(num_detections)
    row_offsets = ib.buffer_ptr(row_offsets)
    num_total_detections = ib.buffer_ptr(num_total_detections)
    collected_indices = ib.buffer_ptr(collected_indices)
    collected_scores = ib.buffer_ptr(collected_scores)
    zero = cast(0, "int64")

    with ib.for_range(0, batch_size * num_class, name="i", kind="parallel") as i:
        i = cast(i, "int64")
        batch_id = i // num_class
        class_id = i % num_class

        with ib.for_range(0, num_boxes, name="j") as j:
            with ib.if_scope(j < num_detections[batch_id, class_id]):
                offset = row_offsets[batch_id, class_id] + j
                collected_indices[batch_id, offset, 0] = class_id
                collected_indices[batch_id, offset, 1] = cast(selected_indices[i, j], "int64")
                collected_scores[batch_id, offset] = selected_scores[i, j]
            with ib.else_scope():
                offset = (
                    num_total_detections[batch_id]
                    + class_id * num_boxes
                    - row_offsets[batch_id, class_id]
                    + j
                    - num_detections[batch_id, class_id]
                )
                collected_indices[batch_id, offset, 0] = zero
                collected_indices[batch_id, offset, 1] = zero
                collected_scores[batch_id, offset] = 0.0

    return ib.get()


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
    boxes : tvm.te.Tensor
        3-D tensor with shape (batch_size, num_boxes, 4)

    scores: tvm.te.Tensor
        3-D tensor with shape (batch_size, num_classes, num_boxes)

    max_output_boxes_per_class : int or tvm.te.Tensor, optional
        The maxinum number of output selected boxes per class

    iou_threshold : float or tvm.te.Tensor, optionaIl
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

    valid_count = _get_valid_box_count(sorted_scores, score_threshold)

    selected_indices, selected_scores, num_detections = run_all_class_nms(
        boxes,
        sorted_scores,
        sorted_indices,
        valid_count,
        max_output_boxes_per_class,
        iou_threshold,
        _nms_loop,
        return_scores=(output_format == "tensorflow"),
    )

    if output_format == "onnx":
        row_offsets = cumsum(num_detections, exclusive=True, dtype="int64")
        num_total_detections = reduction.sum(cast(num_detections, "int64"), axis=1)

        selected_indices = collect_selected_indices(
            num_class, selected_indices, num_detections, row_offsets, _collect_selected_indices_ir
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


@hybrid.script
def hybrid_regular_nms(
    boxes,
    scores,
    max_detections_per_class,
    max_detections,
    batch_size,
    num_boxes,
    num_classes,
    num_classes_with_background,
    iou_threshold,
    score_threshold,
):
    """Hybrid routing for regular non-maximum suppression.

    Parameters
    ----------
    boxes : tvm.te.Tensor
        3-D tensor with shape (batch_size, num_boxes, 4)

    scores: tvm.te.Tensor
        3-D tensor with shape (batch_size, num_boxes, num_classes_with_background)

    max_detections_per_class : tvm.tir.const
        The maxinum number of output selected boxes per class

    max_detections : tvm.tir.const
        The maxinum number of output selected boxes

    batch_size : tvm.tir.IntImm or tvm.tir.Var
        The number of batches

    num_boxes : tvm.tir.IntImm or tvm.tir.Var
        The number of bounding boxes

    num_classes : tvm.tir.const
        The number of classes without background

    num_classes_with_background : tvm.tir.IntImm or tvm.tir.Var
        The number of classes including background ones

    iou_threshold : tvm.tir.const
        IoU test threshold

    score_threshold : tvm.tir.const
        Score threshold to filter out low score boxes early

    Returns
    -------
    detection_boxes : tvm.te.Tensor
        3-D tensor with shape [batch_size, max_detections, 4].

    detection_classes : tvm.te.Tensor
        2-D tensor with shape [batch_size, max_detections].

    detection_scores : tvm.te.Tensor
        2-D tensor with shape [batch_size, max_detections].

    num_detections : tvm.te.Tensor
        1-D tensor with shape [batch_size].
    """
    # output tensors
    detection_boxes = output_tensor((batch_size, max_detections, 4), boxes.dtype)
    detection_classes = output_tensor((batch_size, max_detections), "int32")
    detection_scores = output_tensor((batch_size, max_detections), scores.dtype)
    num_detections = output_tensor((batch_size,), "int32")

    # scratch buffers
    class_scores = allocate((num_boxes,), scores.dtype)
    keep_indices = allocate((num_boxes,), "int32")
    keep_scores = allocate((num_boxes,), scores.dtype)
    sorted_indices = allocate((max_detections + num_boxes,), "int32")
    sorted_scores = allocate((max_detections + num_boxes,), scores.dtype)
    active_box_candidate = allocate((num_boxes,), "int32")
    selected = allocate((num_boxes,), "int32")
    box_indices_after_regular_nms = allocate((max_detections + num_boxes,), "int32")
    scores_after_regular_nms = allocate((max_detections + num_boxes,), scores.dtype)

    label_offset = num_classes_with_background - num_classes
    tmp_idx = 0

    for batch_idx in range(batch_size):
        size_of_sorted_indices = 0

        for class_id in range(num_classes):
            for box_id in range(num_boxes):
                # get scores of boxes corresponding to all anchors for single class
                class_scores[box_id] = scores[batch_idx, box_id, class_id + label_offset]

            # perform non-maximal suppression on single class

            # select detections above score threshold
            num_scores_kept = 0
            for i in range(num_boxes):
                if class_scores[i] >= score_threshold:
                    keep_scores[num_scores_kept] = class_scores[i]
                    keep_indices[num_scores_kept] = i
                    num_scores_kept += 1

            # iota
            for i in range(num_scores_kept):
                sorted_indices[i] = i
            # decreasing sort of scores
            for i in range(num_scores_kept):
                for j in range(num_scores_kept - i - 1):
                    if keep_scores[sorted_indices[j]] < keep_scores[sorted_indices[j + 1]]:
                        tmp_idx = sorted_indices[j]
                        sorted_indices[j] = sorted_indices[j + 1]
                        sorted_indices[j + 1] = tmp_idx

            selected_size = 0

            for i in range(num_scores_kept):
                active_box_candidate[i] = 1

            num_active_candidate = num_scores_kept
            for i in range(num_scores_kept):
                if (
                    num_active_candidate != 0
                    and selected_size < min(num_scores_kept, max_detections_per_class)
                    and active_box_candidate[i] == 1
                ):
                    selected[selected_size] = keep_indices[sorted_indices[i]]
                    selected_size += 1

                    active_box_candidate[i] = 0
                    num_active_candidate -= 1

                    for j in range(i + 1, num_scores_kept):
                        if active_box_candidate[j] == 1:
                            # compute IOU
                            i_ymin = boxes[batch_idx, keep_indices[sorted_indices[i]], 0]
                            i_xmin = boxes[batch_idx, keep_indices[sorted_indices[i]], 1]
                            i_ymax = boxes[batch_idx, keep_indices[sorted_indices[i]], 2]
                            i_xmax = boxes[batch_idx, keep_indices[sorted_indices[i]], 3]

                            j_ymin = boxes[batch_idx, keep_indices[sorted_indices[j]], 0]
                            j_xmin = boxes[batch_idx, keep_indices[sorted_indices[j]], 1]
                            j_ymax = boxes[batch_idx, keep_indices[sorted_indices[j]], 2]
                            j_xmax = boxes[batch_idx, keep_indices[sorted_indices[j]], 3]

                            area_i = (i_ymax - i_ymin) * (i_xmax - i_xmin)
                            area_j = (j_ymax - j_ymin) * (j_xmax - j_xmin)

                            iou = 0.0
                            if area_i > 0 and area_j > 0:
                                intersection_ymin = max(i_ymin, j_ymin)
                                intersection_xmin = max(i_xmin, j_xmin)
                                intersection_ymax = min(i_ymax, j_ymax)
                                intersection_xmax = min(i_xmax, j_xmax)
                                intersection_area = max(
                                    intersection_ymax - intersection_ymin, 0.0
                                ) * max(intersection_xmax - intersection_xmin, 0.0)
                                iou = intersection_area / (area_i + area_j - intersection_area)

                            if iou > iou_threshold:
                                active_box_candidate[j] = 0
                                num_active_candidate -= 1

            # end of non-maximal suppression on single class

            # add selected indices from non-max suppression of boxes in this class
            output_index = size_of_sorted_indices
            for i in range(selected_size):
                selected_index = selected[i]

                box_indices_after_regular_nms[output_index] = (
                    selected_index * num_classes_with_background + class_id + label_offset
                )
                scores_after_regular_nms[output_index] = class_scores[selected_index]

                output_index += 1

            # sort the max scores among the selected indices
            # get the indices for top scores
            num_indices_to_sort = min(output_index, max_detections)

            # iota
            for i in range(output_index):
                sorted_indices[i] = i
            # deacreasing sort of scores
            for i in range(output_index):
                for j in range(output_index - i - 1):
                    if (
                        scores_after_regular_nms[sorted_indices[j]]
                        < scores_after_regular_nms[sorted_indices[j + 1]]
                    ):
                        tmp_idx = sorted_indices[j]
                        sorted_indices[j] = sorted_indices[j + 1]
                        sorted_indices[j + 1] = tmp_idx

            # copy values to temporary vectors
            for i in range(num_indices_to_sort):
                sorted_scores[i] = scores_after_regular_nms[sorted_indices[i]]
                sorted_indices[i] = box_indices_after_regular_nms[sorted_indices[i]]

            # copy scores and indices from temporary vectors
            for i in range(num_indices_to_sort):
                box_indices_after_regular_nms[i] = sorted_indices[i]
                scores_after_regular_nms[i] = sorted_scores[i]

            size_of_sorted_indices = num_indices_to_sort

        # fill output tensors
        for output_box_index in range(max_detections):
            box_ymin = 0.0
            box_xmin = 0.0
            box_ymax = 0.0
            box_xmax = 0.0
            class_idx = 0
            selected_score = 0.0

            if output_box_index < size_of_sorted_indices:
                anchor_idx = (
                    box_indices_after_regular_nms[output_box_index] // num_classes_with_background
                )

                box_ymin = boxes[batch_idx, anchor_idx, 0]
                box_xmin = boxes[batch_idx, anchor_idx, 1]
                box_ymax = boxes[batch_idx, anchor_idx, 2]
                box_xmax = boxes[batch_idx, anchor_idx, 3]
                class_idx = (
                    box_indices_after_regular_nms[output_box_index]
                    - anchor_idx * num_classes_with_background
                    - label_offset
                )
                selected_score = scores_after_regular_nms[output_box_index]

            detection_boxes[batch_idx, output_box_index, 0] = box_ymin
            detection_boxes[batch_idx, output_box_index, 1] = box_xmin
            detection_boxes[batch_idx, output_box_index, 2] = box_ymax
            detection_boxes[batch_idx, output_box_index, 3] = box_xmax
            detection_classes[batch_idx, output_box_index] = class_idx
            detection_scores[batch_idx, output_box_index] = selected_score

        num_detections[batch_idx] = size_of_sorted_indices

    return detection_boxes, detection_classes, detection_scores, num_detections


def regular_non_max_suppression(
    boxes,
    scores,
    max_detections_per_class,
    max_detections,
    num_classes,
    iou_threshold,
    score_threshold,
):
    """Regular non-maximum suppression operator for object detection, corresponding to TFLite's
    regular NMS. NMS is performed for each class separately.

    Parameters
    ----------
    boxes : tvm.te.Tensor
        3-D tensor with shape (batch_size, num_boxes, 4). The four values in boxes
        encode (ymin, xmin, ymax, xmax) coordinates of a box

    scores: tvm.te.Tensor
        3-D tensor with shape (batch_size, num_boxes, num_classes_with_background)

    max_detections_per_class : int
        The maxinum number of output selected boxes per class

    max_detections : int
        The maxinum number of output selected boxes

    num_classes : int
        The number of classes without background

    iou_threshold : float
        IoU test threshold

    score_threshold : float
        Score threshold to filter out low score boxes early

    Returns
    -------
    out : list of tvm.te.Tensor
        The output is a list of four tensors. The first is `detection_boxes` of size
        `(batch_size, max_detections , 4)`, the second is `detection_classes` of size
        `(batch_size, max_detections)`, the third is `detection_scores` of size
        `(batch_size, max_detections)`, and the fourth is `num_detections` of size `(batch_size,)`
        representing the total number of selected boxes per batch.
    """
    batch_size, num_boxes, num_classes_with_background = scores.shape

    detection_boxes, detection_classes, detection_scores, num_detections = hybrid_regular_nms(
        boxes=boxes,
        scores=scores,
        max_detections_per_class=tvm.tir.const(max_detections_per_class, dtype="int32"),
        max_detections=tvm.tir.const(max_detections, dtype="int32"),
        batch_size=batch_size,
        num_boxes=num_boxes,
        num_classes=tvm.tir.const(num_classes, dtype="int32"),
        num_classes_with_background=num_classes_with_background,
        iou_threshold=tvm.tir.const(iou_threshold, dtype="float32"),
        score_threshold=tvm.tir.const(score_threshold, dtype="float32"),
    )

    return [
        detection_boxes,
        cast(detection_classes, dtype="float32"),
        detection_scores,
        num_detections,
    ]
