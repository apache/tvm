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

from tvm import hybrid
from ..sort import argsort

@hybrid.script
def hybrid_rearrange_out(data, one):
    """Hybrid routine to rearrange nms output to
    move all valid entries to top.

    Parameters
    ----------
    data : tvm.Tensor or numpy NDArray
        NMS output. 3-D tensor with shape
        [batch_size, num_anchors, 6].

    one: tvm.const
        Constant one with the same dtype as data.

    Returns
    -------
    output : tvm.Tensor or numpy NDArray
        Transformed NMS output. 3-D tensor with shape
        [batch_size, num_anchors, 6].
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    elem_length = data.shape[2]
    output = output_tensor((batch_size,
                            num_anchors,
                            elem_length),
                           data.dtype)

    for i in parallel(batch_size):
        valid_idx = 0
        for j in range(num_anchors):
            if data[i, j, 0] >= 0:
                for k in range(elem_length):
                    output[i, valid_idx, k] = data[i, j, k]
                valid_idx += 1
            if j >= valid_idx:
                for k in range(elem_length):
                    output[i, j, k] = -one
    return output


@hybrid.script
def hybrid_get_valid_counts(data, score_threshold, id_index, score_index, one):
    """Hybrid routine to get valid count of bounding boxes
    given a score threshold. Also moves valid boxes to the
    top of input data.

    Parameters
    ----------
    data : tvm.Tensor or numpy NDArray
        Input data. 3-D tensor with shape [batch_size, num_anchors, 6]
        or [batch_size, num_anchors, 5].

    score_threshold : tvm.const
        Lower limit of score for valid bounding boxes.

    id_index : tvm.const
        index of the class categories, -1 to disable.

    score_index: tvm.const
        Index of the scores/confidence of boxes.

    one: tvm.const
        Constant one with the same dtype as data.

    Returns
    -------
    out_tensor : tvm.Tensor or numpy NDArray
        Rearranged data tensor.

    valid_count : tvm.Tensor or numpy NDArray
        1-D tensor for valid number of boxes.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    box_data_length = data.shape[2]
    valid_count = output_tensor((batch_size,), "int32")
    out_tensor = output_tensor((batch_size,
                                num_anchors,
                                box_data_length),
                               data.dtype)
    for i in parallel(batch_size):
        valid_count[i] = 0
        for j in range(num_anchors):
            score = data[i, j, score_index]
            if score > score_threshold and \
                    (id_index < 0 or data[i, j, id_index] >= 0):
                for k in range(box_data_length):
                    out_tensor[i, valid_count[i], k] = data[i, j, k]
                valid_count[i] += 1
            if j >= valid_count[i]:
                for k in range(box_data_length):
                    out_tensor[i, j, k] = -one
    return valid_count, out_tensor

@tvm.target.generic_func
def get_valid_counts(data, score_threshold=0, id_index=0, score_index=1):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : tvm.Tensor
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
    out_tensor : tvm.Tensor
        Rearranged data tensor.

    valid_count : tvm.Tensor
        1-D tensor for valid number of boxes.
    """
    score_threshold_const = tvm.const(score_threshold, data.dtype)
    id_index_const = tvm.const(id_index, "int32")
    score_index_const = tvm.const(score_index, "int32")
    return hybrid_get_valid_counts(data, score_threshold_const,
                                   id_index_const, score_index_const,
                                   tvm.const(1, data.dtype))


@hybrid.script
def hybrid_nms(data, sorted_index, valid_count,
               max_output_size, iou_threshold, force_suppress,
               top_k, coord_start, id_index, score_index, zero, one):
    """Hybrid routing for non-maximum suppression.

    Parameters
    ----------
    data: tvm.Tensor or numpy NDArray
        Bounding boxes with class and score. 3-D tensor with shape
        [batch_size, num_anchors, 6].

    sorted_index : tvm.Tensor or numpy NDArray
        Bounding box indexes sorted by score, with shape
        [batch_size, num_anchors].

    valid_count : tvm.Tensor or numpy NDArray
        1-D tensor for valid number of boxes.

    max_output_size : tvm.const
        Max number of output valid boxes for each instance.
        By default all valid boxes are returned.

    iou_threshold : tvm.const
        Overlapping(IoU) threshold to suppress object with smaller score.

    force_suppress : tvm.const
        Whether to suppress all detections regardless of class_id.

    top_k : tvm.const
        Keep maximum top k detections before nms, -1 for no limit.

    coord_start : tvm.const
        Start index of the consecutive 4 coordinates.

    id_index : tvm.const
        index of the class categories, -1 to disable.

    score_index: tvm.const
        Index of the scores/confidence of boxes.

    zero: tvm.const
        Constant zero with the same dtype as data.

    one: tvm.const
        Constant one with the same dtype as data.

    Returns
    -------
    output : tvm.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6].

    box_indices: tvm.Tensor
        2-D tensor with shape [batch_size, num_anchors].
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    box_data_length = data.shape[2]
    box_indices = output_tensor((batch_size, num_anchors), "int32")
    output = output_tensor((batch_size,
                            num_anchors,
                            box_data_length,), data.dtype)

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
            for j in range(valid_count[i]):
                if output[i, j, score_index] > 0 and (id_index < 0 or output[i, j, id_index] >= 0):
                    box_a_idx = j
                    for k in parallel(valid_count[i]):
                        check_iou = 0
                        if k > j and output[i, k, score_index] > 0 \
                                and (id_index < 0 or output[i, k, id_index] >= 0):
                            if force_suppress:
                                check_iou = 1
                            elif id_index < 0 or output[i, j, id_index] == output[i, k, id_index]:
                                check_iou = 1
                        if check_iou > 0:
                            a_l = output[batch_idx, box_a_idx, box_start_idx]
                            a_t = output[batch_idx, box_a_idx, box_start_idx + 1]
                            a_r = output[batch_idx, box_a_idx, box_start_idx + 2]
                            a_b = output[batch_idx, box_a_idx, box_start_idx + 3]
                            box_b_idx = k
                            b_t = output[batch_idx, box_b_idx, box_start_idx + 1]
                            b_b = output[batch_idx, box_b_idx, box_start_idx + 3]
                            b_l = output[batch_idx, box_b_idx, box_start_idx]
                            b_r = output[batch_idx, box_b_idx, box_start_idx + 2]
                            w = max(zero, min(a_r, b_r) - max(a_l, b_l))
                            h = max(zero, min(a_b, b_b) - max(a_t, b_t))
                            area = h * w
                            u = (a_r - a_l) * (a_b - a_t) + (b_r - b_l) * (b_b - b_t) - area
                            iou = zero if u <= zero else area / u
                            if iou >= iou_threshold:
                                output[i, k, score_index] = -one
                                if id_index >= 0:
                                    output[i, k, id_index] = -one
                                box_indices[i, k] = -1
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
        # Only return max_output_size valid boxes
        num_valid_boxes = 0
        if max_output_size > 0:
            for j in parallel(valid_count[i]):
                if output[i, j, 0] >= zero:
                    if num_valid_boxes == max_output_size:
                        for k in range(box_data_length):
                            output[i, j, k] = -one
                        box_indices[i, j] = -1
                    else:
                        num_valid_boxes += 1
    return output, box_indices


@tvm.target.generic_func
def non_max_suppression(data, valid_count, max_output_size=-1,
                        iou_threshold=0.5, force_suppress=False, top_k=-1,
                        coord_start=2, score_index=1, id_index=0,
                        return_indices=True, invalid_to_bottom=False):
    """Non-maximum suppression operator for object detection.

    Parameters
    ----------
    data : tvm.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6] or [batch_size, num_anchors, 5].

    valid_count : tvm.Tensor
        1-D tensor for valid number of boxes.

    max_output_size : optional, int
        Max number of output valid boxes for each instance.
        By default all valid boxes are returned.

    iou_threshold : optional, float
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
    out : tvm.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6].

    Example
    --------
    .. code-block:: python

        # An example to use non_max_suppression
        dshape = (1, 5, 6)
        data = tvm.placeholder(dshape, name="data")
        valid_count = tvm.placeholder((dshape[0],), dtype="int32", name="valid_count")
        iou_threshold = 0.7
        force_suppress = True
        top_k = -1
        out = non_max_suppression(data, valid_count, iou_threshold=iou_threshold,
                                  force_suppress=force_suppress, top_k=top_k)
        np_data = np.random.uniform(dshape)
        np_valid_count = np.array([4])
        s = topi.generic.schedule_nms(out)
        f = tvm.build(s, [data, valid_count, out], "llvm")
        ctx = tvm.cpu()
        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_valid_count = tvm.nd.array(np_valid_count, ctx)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data.dtype), ctx)
        f(tvm_data, tvm_valid_count, tvm_out)
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    score_axis = score_index
    score_shape = (batch_size, num_anchors)
    score_tensor = tvm.compute(score_shape, lambda i, j: data[i, j, score_axis])
    sort_tensor = argsort(score_tensor, valid_count=valid_count, axis=1, is_ascend=False)
    out, box_indices = hybrid_nms(data, sort_tensor, valid_count,
                                  tvm.const(max_output_size, dtype="int32"),
                                  tvm.const(iou_threshold, dtype=data.dtype),
                                  tvm.const(force_suppress, dtype="bool"),
                                  tvm.const(top_k, dtype="int32"),
                                  tvm.const(coord_start, dtype="int32"),
                                  tvm.const(id_index, dtype="int32"),
                                  tvm.const(score_index, dtype="int32"),
                                  zero=tvm.const(0, dtype=data.dtype),
                                  one=tvm.const(1, dtype=data.dtype))
    if not return_indices and invalid_to_bottom:
        out = hybrid_rearrange_out(out, one=tvm.const(1, dtype=data.dtype))

    return box_indices if return_indices else out
