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
# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments, undefined-variable
"""SSD multibox operators"""
import tvm

from tvm.te import hybrid
from tvm.tir import exp, sqrt

from tvm import topi

from ..nms import non_max_suppression


@hybrid.script
def hybrid_multibox_prior(data, sizes, ratios, steps, offsets):
    """Hybrid routing for multibox_prior operator.

    Parameters
    ----------
    data : tvm.te.Tensor or numpy NDArray
        4-D tensor with shape [batch, channel, height, width]]

    sizes : tvm ConsExpr
        Sizes for anchor boxes.

    ratios : tvm ConsExpr
        Ratios for anchor boxes.

    steps : tvm ConsExpr
        Priorbox step across y and x, -1 for auto calculation.

    offsets : tvm ConsExpr
        Priorbox center offsets, y and x respectively.

    Returns
    -------
    output : tvm.te.Tensor or numpy NDArray
        3-D tensor with shape [1, h_in * w_in * (num_sizes + num_ratios - 1), 4]
    """
    in_height = data.shape[2]
    in_width = data.shape[3]
    num_sizes = len(sizes)
    num_ratios = len(ratios)
    num_boxes = in_height * in_width * (num_sizes + num_ratios - 1)
    output = output_tensor((1, num_boxes, 4), "float32")
    steps_h = steps[0] * 1.0 if steps[0] > 0 else 1.0 / in_height
    steps_w = steps[1] * 1.0 if steps[1] > 0 else 1.0 / in_width
    offset_h = offsets[0]
    offset_w = offsets[1]

    # Need to define var out of const_range + if
    w = 0.0
    h = 0.0

    for i in parallel(in_height):
        center_h = (i + offset_h) * steps_h
        for j in range(in_width):
            center_w = (j + offset_w) * steps_w
            for k in const_range(num_sizes + num_ratios - 1):
                if k < num_sizes:
                    w = float32(sizes[k] * in_height) / in_width / 2.0
                    h = sizes[k] / 2.0
                else:
                    w = (
                        float32(sizes[0] * in_height)
                        / in_width
                        * sqrt(ratios[k - num_sizes + 1] * 1.0)
                        / 2.0
                    )
                    h = sizes[0] / sqrt(ratios[k - num_sizes + 1] * 1.0) / 2.0
                count = (
                    i * in_width * (num_sizes + num_ratios - 1)
                    + j * (num_sizes + num_ratios - 1)
                    + k
                )
                output[0, count, 0] = center_w - w
                output[0, count, 1] = center_h - h
                output[0, count, 2] = center_w + w
                output[0, count, 3] = center_h + h

    return output


def multibox_prior(data, sizes=(1,), ratios=(1,), steps=(-1, -1), offsets=(0.5, 0.5), clip=False):
    """Generate prior(anchor) boxes from data, sizes and ratios.

    Parameters
    ----------
    data : tvm.te.Tensor
        4-D with shape [batch, c_in, h_in, w_in]]

    sizes : tuple of float
        Tuple of sizes for anchor boxes.

    ratios : tuple of float
        Tuple of ratios for anchor boxes.

    steps : Tuple of float
        Priorbox step across y and x, -1 for auto calculation.

    offsets : tuple of int
        Priorbox center offsets, y and x respectively.

    clip : boolean
        Whether to clip out-of-boundary boxes.

    Returns
    -------
    out : tvm.te.Tensor
        3-D tensor with shape [1, h_in * w_in * (num_sizes + num_ratios - 1), 4]
    """
    out = hybrid_multibox_prior(
        data,
        tvm.runtime.convert(sizes),
        tvm.runtime.convert(ratios),
        tvm.runtime.convert(steps),
        tvm.runtime.convert(offsets),
    )
    if clip:
        out = topi.clip(out, 0, 1)
    return out


@hybrid.script
def _hybrid_transform_loc(anchor, pred_loc, variance, clip, batch_idx, anchor_idx):
    """Transform prior anchor box to output box through location predictions."""
    al = anchor[0, anchor_idx, 0]
    at = anchor[0, anchor_idx, 1]
    ar = anchor[0, anchor_idx, 2]
    ab = anchor[0, anchor_idx, 3]

    px = pred_loc[batch_idx, 0]
    py = pred_loc[batch_idx, 1]
    pw = pred_loc[batch_idx, 2]
    ph = pred_loc[batch_idx, 3]

    vx = variance[0]
    vy = variance[1]
    vw = variance[2]
    vh = variance[3]

    output = output_tensor((4,), pred_loc.dtype)

    aw = ar - al
    ah = ab - at
    ax = (al + ar) / 2.0
    ay = (at + ab) / 2.0
    ox = px * vx * aw + ax
    oy = py * vy * ah + ay
    ow = exp(pw * vw) * aw / 2.0
    oh = exp(ph * vh) * ah / 2.0
    output[0] = max(0.0, min(1.0, ox - ow)) if clip else ox - ow
    output[1] = max(0.0, min(1.0, oy - oh)) if clip else oy - oh
    output[2] = max(0.0, min(1.0, ox + ow)) if clip else ox + ow
    output[3] = max(0.0, min(1.0, oy + oh)) if clip else oy + oh
    return output


@hybrid.script
def hybrid_multibox_transform_loc(
    cls_prob,
    loc_pred,
    anchor,
    clip,
    threshold,
    variances,
    keep_background,
):
    """Hybrid routing for transform location in multibox_detection operator.

    Parameters
    ----------
    cls_prob : tvm.te.Tensor or numpy NDArray
        3-D tensor of class probabilities.

    loc_pred : tvm.te.Tensor or numpy NDArray
        2-D tensor of location regression predictions.

    anchor : tvm.te.Tensor or numpy NDArray
        3-D tensor of prior anchor boxes.

    clip : tvm.tir.const
        Whether to clip out-of-boundary boxes.

    threshold : tvm.tir.const
        Threshold to be a positive prediction.

    variances : tvm.nd.NDArray
        Variances to be decoded from box regression output.

    keep_background : tvm.tir.const
        Whether to keep boxes detected as background or not.

    Returns
    -------
    out_loc : tvm.te.Tensor or numpy NDArray
        3-D tensor of transformed location.

    valid_count : tvm.te.Tensor or numpy NDArray
        1_d tensor of valid counts for boxes.
    """
    batch_size = cls_prob.shape[0]
    num_classes = cls_prob.shape[1]
    num_anchors = cls_prob.shape[2]
    pred_coord = allocate(
        (
            batch_size,
            4,
        ),
        loc_pred.dtype,
    )
    out_loc = output_tensor((batch_size, num_anchors, 6), loc_pred.dtype)
    valid_count = output_tensor((batch_size,), "int32")

    start_cls_idx = 0 if keep_background else 1

    for i in parallel(batch_size):
        valid_count[i] = 0
        for j in range(num_anchors):
            # Find the predicted class id and probability
            score = -1.0
            cls_id = 0
            for k in range(start_cls_idx, num_classes):
                temp = cls_prob[i, k, j]
                cls_id = k if temp > score else cls_id
                score = max(temp, score)
            if cls_id > 0 and score < threshold:
                cls_id = 0
            # [id, prob, xmin, ymin, xmax, ymax]
            # Remove background if 'keep_background=False', restore original id
            if keep_background or cls_id > 0:
                out_loc[i, valid_count[i], 0] = cls_id - 0.0 if keep_background else cls_id - 1.0
                out_loc[i, valid_count[i], 1] = score
                for l in range(4):
                    pred_coord[i, l] = loc_pred[i, j * 4 + l]
                out_coord = _hybrid_transform_loc(anchor, pred_coord, variances, clip, i, j)
                out_loc[i, valid_count[i], 2] = out_coord[0]
                out_loc[i, valid_count[i], 3] = out_coord[1]
                out_loc[i, valid_count[i], 4] = out_coord[2]
                out_loc[i, valid_count[i], 5] = out_coord[3]
                valid_count[i] += 1

    return out_loc, valid_count


def multibox_transform_loc(
    cls_prob,
    loc_pred,
    anchor,
    clip=True,
    threshold=0.01,
    variances=(0.1, 0.1, 0.2, 0.2),
    keep_background=False,
):
    """Location transformation for multibox detection

    Parameters
    ----------
    cls_prob : tvm.te.Tensor
        Class probabilities.

    loc_pred : tvm.te.Tensor
        Location regression predictions.

    anchor : tvm.te.Tensor
        Prior anchor boxes.

    clip : boolean
        Whether to clip out-of-boundary boxes.

    threshold : float
        Threshold to be a positive prediction.

    variances : tuple of float
        Variances to be decoded from box regression output.

    keep_background : boolean
        Whether to keep boxes detected as background or not.

    Returns
    -------
    ret : tuple of tvm.te.Tensor
    """

    return hybrid_multibox_transform_loc(
        cls_prob,
        loc_pred,
        anchor,
        tvm.tir.const(clip, "bool"),
        tvm.tir.const(threshold, "float32"),
        tvm.runtime.convert(variances),
        tvm.tir.const(keep_background, "bool"),
    )


def multibox_detection(
    cls_prob,
    loc_pred,
    anchor,
    clip=True,
    threshold=0.01,
    nms_threshold=0.5,
    force_suppress=False,
    variances=(0.1, 0.1, 0.2, 0.2),
    nms_topk=-1,
):
    """Convert multibox detection predictions.

    Parameters
    ----------
    cls_prob : tvm.te.Tensor
        Class probabilities.

    loc_pred : tvm.te.Tensor
        Location regression predictions.

    anchor : tvm.te.Tensor
        Prior anchor boxes.

    clip : boolean
        Whether to clip out-of-boundary boxes.

    nms_threshold : float
        Non-maximum suppression threshold.

    force_suppress : boolean
        Whether to suppress all detections regardless of class_id.

    threshold : float
        Threshold to be a positive prediction.

    variances : tuple of float
        Variances to be decoded from box regression output.

    nms_topk : int
        Keep maximum top k detections before nms, -1 for no limit.

    Returns
    -------
    out : tvm.te.Tensor
        3-D tensor with shape (batch_size, num_anchors, 6)
    """
    inter_out = multibox_transform_loc(cls_prob, loc_pred, anchor, clip, threshold, variances)
    out = non_max_suppression(
        inter_out[0],
        inter_out[1],
        inter_out[1],
        max_output_size=-1,
        iou_threshold=nms_threshold,
        force_suppress=force_suppress,
        top_k=nms_topk,
        return_indices=False,
    )
    return out
