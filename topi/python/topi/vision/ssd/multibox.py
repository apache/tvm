# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments, undefined-variable
"""SSD multibox operators"""
from __future__ import absolute_import as _abs
import tvm

from tvm import hybrid
from tvm.intrin import exp, sqrt

import topi

from ..nms import nms

@hybrid.script
def hybrid_multibox_prior(data, sizes, ratios, steps, offsets):
    """Hybrid routing for multibox_prior operator.

    Parameters
    ----------
    data : tvm.Tensor or numpy NDArray
        4-D tensor with shape [batch, channel, height, width]]

    sizes : tvm.ndarray
        1-D tensor of sizes for anchor boxes.

    ratios : tvm.ndarray
        1-D tensor of ratios for anchor boxes.

    steps : tvm.ndarray
        1-D tensor of priorbox step across y and x, -1 for auto calculation.

    offsets : tvm.ndarray
        1-D tensor priorbox center offsets, y and x respectively.

    Returns
    -------
    output : tvm.Tensor or numpy NDArray
        3-D tensor with shape [1, h_in * w_in * (num_sizes + num_ratios - 1), 4]
    """
    in_height, in_width = data.shape[2], data.shape[3]
    num_sizes, num_ratios = sizes.shape[0], ratios.shape[0]
    num_boxes = in_height * in_width * (num_sizes + num_ratios - 1)
    output = output_tensor((1, num_boxes, 4), data.dtype)
    steps_h = steps[0] if steps[0] > 0 else 1.0 / in_height
    steps_w = steps[1] if steps[1] > 0 else 1.0 / in_width
    offset_h = offsets[0]
    offset_w = offsets[1]

    for i in parallel(in_height):
        center_h = (i + offset_h) * steps_h
        for j in range(in_width):
            center_w = (j + offset_w) * steps_w
            for k in range(num_sizes + num_ratios - 1):
                if k < num_sizes:
                    w = sizes[k] * in_height / in_width / 2.0
                    h = sizes[k] / 2.0
                else:
                    w = sizes[0] * in_height / in_width \
                        * sqrt(ratios[k - num_sizes + 1]) / 2.0
                    h = sizes[0] * sqrt(ratios[k - num_sizes + 1]) / 2.0
                count = i * in_width * (num_sizes + num_ratios - 1) \
                        + j * (num_sizes + num_ratios - 1) + k
                output[0, count, 0] = center_w - w
                output[0, count, 1] = center_h - h
                output[0, count, 2] = center_w + w
                output[0, count, 3] = center_h + h

    return output


@tvm.target.generic_func
def multibox_prior(data, sizes=(1,), ratios=(1,), steps=(-1, -1), offsets=(0.5, 0.5), clip=False):
    """Generate prior(anchor) boxes from data, sizes and ratios.

    Parameters
    ----------
    data : tvm.Tensor
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
    out : tvm.Tensor
        3-D tensor with shape [1, h_in * w_in * (num_sizes + num_ratios - 1), 4]
    """
    out = hybrid_multibox_prior(data, sizes, ratios, steps, offsets)
    if clip:
        out = topi.clip(out, 0, 1)
    return out

@hybrid.script
def _hybridy_transform_loc(box, pred_loc, variance, clip):
    """Transform prior anchor box to output box through location predictions.
    """
    al, at, ar, ab = box[0], box[1], box[2], box[3]
    px, py, pw, ph = pred_loc[0], pred_loc[1], \
                     pred_loc[2], pred_loc[3]
    vx, vy, vw, vh = variance[0], variance[1], \
                     variance[2], variance[3]
    aw = ar - al
    ah = ab - at
    ax = (al + ar) / 2.0
    ay = (at + ab) / 2.0
    ox = px * vx * aw + ax
    oy = py * vy * ah + ay
    ow = exp(pw * vw) * aw / 2.0
    oh = exp(ph * vh) * ah / 2.0
    out_l = max(0, min(1, ox - ow)) if clip else ox - ow
    out_t = max(0, min(1, oy - oh)) if clip else oy - oh
    out_r = max(0, min(1, ox + ow)) if clip else ox + ow
    out_b = max(0, min(1, oy + oh)) if clip else oy + oh
    return out_l, out_t, out_r, out_b

@hybrid.script
def hybrid_multibox_transform_loc(cls_prob, loc_pred, anchor,
                                  clip, threshold, variances):
    """Hybrid routing for transform location in multibox_detection operator.

    Parameters
    ----------
    cls_prob : tvm.Tensor or numpy NDArray
        3-D tensor of class probabilities.

    loc_pred : tvm.Tensor or numpy NDArray
        3-D tensor of location regression predictions.

    anchor : tvm.Tensor or numpy NDArray
        3-D tensor of prior anchor boxes.

    clip : tvm.const
        Whether to clip out-of-boundary boxes.

    threshold : tvm.const
        Threshold to be a positive prediction.

    variances : tvm.ndarray
        Variances to be decoded from box regression output.

    Returns
    -------
<<<<<<< HEAD
    stmt : Stmt
        The result IR statement.
    """
    def transform_loc(loc, loc_base_idx, anchor, anchor_base_idx, clip, vx, vy, vw, vh):
        """Transform prior anchor box to output box through location predictions.
        """
        al = anchor[anchor_base_idx]
        at = anchor[anchor_base_idx + 1]
        ar = anchor[anchor_base_idx + 2]
        ab = anchor[anchor_base_idx + 3]
        aw = ar - al
        ah = ab - at
        ax = (al + ar) / 2.0
        ay = (at + ab) / 2.0
        px = loc[loc_base_idx]
        py = loc[loc_base_idx + 1]
        pw = loc[loc_base_idx + 2]
        ph = loc[loc_base_idx + 3]
        ox = px * vx * aw + ax
        oy = py * vy * ah + ay
        ow = tvm.exp(pw * vw) * aw / 2.0
        oh = tvm.exp(ph * vh) * ah / 2.0
        return tvm.if_then_else(clip, tvm.max(0, tvm.min(1, ox - ow)), ox - ow), \
               tvm.if_then_else(clip, tvm.max(0, tvm.min(1, oy - oh)), oy - oh), \
               tvm.if_then_else(clip, tvm.max(0, tvm.min(1, ox + ow)), ox + ow), \
               tvm.if_then_else(clip, tvm.max(0, tvm.min(1, oy + oh)), oy + oh)
=======
    out_loc : tvm.Tensor or numpy NDArray
        3-D tensor of transformed location.
>>>>>>> Modify SSD tutorial

    valid_count : tvm.Tensor or numpy NDArray
        1_d tensor of valid counts for boxes.
    """
    batch_size = cls_prob.shape[0]
    num_classes = cls_prob.shape[1]
    num_anchors = cls_prob.shape[2]
    out_loc = output_tensor((batch_size, num_anchors, 6),
                            loc_pred.dtype)
    valid_count = output_tensor((batch_size,), "int32")

    for i in parallel(batch_size):
        valid_count[i] = 0
        for j in range(num_anchors):
            # Find the predicted class id and probability
<<<<<<< HEAD
            score = ib.allocate('float32', (1,), name="score", scope="local")
            cls_id = ib.allocate('int32', (1,), name="id", scope="local")
            score[0] = -1.0
            cls_id[0] = 0
            with ib.for_range(0, num_classes, name="j") as j:
                with ib.if_scope(j > 0):
                    temp = p_cls_prob[n * num_anchors * num_classes + j * num_anchors + i]
                    cls_id[0] = tvm.if_then_else(temp > score[0], j, cls_id[0])
                    score[0] = tvm.max(temp, score[0])
            with ib.if_scope(tvm.all(cls_id[0] > 0, score[0] < threshold)):
                cls_id[0] = 0
=======
            score = -1.0
            cls_id = 0
            for k in range(num_classes):
                if k > 0:
                    temp = cls_prob[i, k, j]
                    cls_id = j if temp > score else cls_id
                    score = max(temp, score)
            if cls_id > 0 and score < threshold:
                cls_id = 0
>>>>>>> Modify SSD tutorial
            # [id, prob, xmin, ymin, xmax, ymax]
            # Remove background, restore original id
            if cls_id > 0:
                out_loc[i, valid_count[i], 0] = cls_id - 1.0
                out_loc[i, valid_count[i], 1] = score
                out_coord = _hybridy_transform_loc(anchor[j], loc_pred[i, j],
                                                   variances, clip)
                out_loc[i, valid_count[i], 2] = out_coord[0]
                out_loc[i, valid_count[i], 3] = out_coord[1]
                out_loc[i, valid_count[i], 4] = out_coord[2]
                out_loc[i, valid_count[i], 5] = out_coord[3]

    return out_loc, valid_count

@tvm.target.generic_func
def multibox_transform_loc(cls_prob, loc_pred, anchor, clip=True, threshold=0.01,
                           variances=(0.1, 0.1, 0.2, 0.2)):
    """Location transformation for multibox detection

    Parameters
    ----------
    cls_prob : tvm.Tensor
        Class probabilities.

    loc_pred : tvm.Tensor
        Location regression predictions.

    anchor : tvm.Tensor
        Prior anchor boxes.

    clip : boolean
        Whether to clip out-of-boundary boxes.

    threshold : float
        Threshold to be a positive prediction.

    variances : tuple of float
        Variances to be decoded from box regression output.

    Returns
    -------
    ret : tuple of tvm.Tensor
    """
    out, valid_count = hybrid_multibox_transform_loc(cls_prob, loc_pred, anchor,
                                                     tvm.const(clip, "bool"),
                                                     tvm.const(threshold, "float32"),
                                                     variances)
    return out, valid_count

@tvm.target.generic_func
def multibox_detection(cls_prob, loc_pred, anchor, clip=True, threshold=0.01, nms_threshold=0.5,
                       force_suppress=False, variances=(0.1, 0.1, 0.2, 0.2), nms_topk=-1):
    """Convert multibox detection predictions.

    Parameters
    ----------
    cls_prob : tvm.Tensor
        Class probabilities.

    loc_pred : tvm.Tensor
        Location regression predictions.

    anchor : tvm.Tensor
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
    out : tvm.Tensor
        3-D tensor with shape (batch_size, num_anchors, 6)
    """
    inter_out = multibox_transform_loc(cls_prob, loc_pred, anchor,
                                       clip, threshold, variances)
    out = nms(inter_out[0], inter_out[1], nms_threshold, force_suppress, nms_topk)
    return out
