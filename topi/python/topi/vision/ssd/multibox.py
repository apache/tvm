# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments
"""SSD multibox operators"""
from __future__ import absolute_import as _abs
import math
import tvm

from tvm import api

import topi

from ..nms import nms

def multibox_prior_ir(data, out, sizes, ratios, steps, offsets):
    """Low level IR routing for multibox_prior operator.

    Parameters
    ----------
    data : Buffer
        Input data buffer.

    out : Buffer
        Output buffer.

    sizes : tuple of float
        Tuple of sizes for anchor boxes.

    ratios : tuple of float
        Tuple of ratios for anchor boxes.

    steps : Tuple of float
        Priorbox step across y and x, -1 for auto calculation.

    offsets : tuple of int
        Priorbox center offsets, y and x respectively.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    ib = tvm.ir_builder.create()
    p_out = ib.buffer_ptr(out)
    in_height = data.shape[2]
    in_width = data.shape[3]
    num_sizes = len(sizes)
    num_ratios = len(ratios)
    size_ratio_concat = sizes + ratios
    steps_h = steps[0] if steps[0] > 0 else 1.0 / in_height
    steps_w = steps[1] if steps[1] > 0 else 1.0 / in_width
    offset_h = offsets[0]
    offset_w = offsets[1]

    with ib.for_range(0, in_height, for_type="parallel", name="i") as i:
        center_h = (i + offset_h) * steps_h
        with ib.for_range(0, in_width, name="j") as j:
            center_w = (j + offset_w) * steps_w
            for k in range(num_sizes + num_ratios - 1):
                w = tvm.select(k < num_sizes,
                               size_ratio_concat[k] * in_height / in_width / 2.0,
                               size_ratio_concat[0] * in_height / in_width *
                               math.sqrt(size_ratio_concat[k + 1]) / 2.0)
                h = tvm.select(k < num_sizes, size_ratio_concat[k] / 2.0,
                               size_ratio_concat[0] / math.sqrt(size_ratio_concat[k + 1]) / 2.0)
                count = (i * in_width * (num_sizes + num_ratios - 1) +
                         j * (num_sizes + num_ratios - 1) + k) * 4
                p_out[count] = center_w - w
                p_out[count + 1] = center_h - h
                p_out[count + 2] = center_w + w
                p_out[count + 3] = center_h + h

    return ib.get()


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
    num_sizes = len(sizes)
    num_ratios = len(ratios)
    oshape = (1, data.shape[2] * data.shape[3] * (num_sizes + num_ratios - 1), 4)
    out = tvm.extern(oshape, [data], lambda ins, outs:
                     multibox_prior_ir(ins[0], outs[0], sizes, ratios, steps, offsets),
                     tag="multibox_prior")
    if clip:
        out = topi.clip(out, 0, 1)
    return out


def transform_loc_ir(cls_prob, loc_pred, anchor, valid_count, out, clip, threshold, variances):
    """Low level IR routing for transform location in multibox_detection operator.

    Parameters
    ----------
    cls_prob : Buffer
        Buffer of class probabilities.

    loc_pred : Buffer
        Buffer of location regression predictions.

    anchor : Buffer
        Buffer of prior anchor boxes.

    valid_count : Buffer
        Buffer of number of valid output boxes.

    out : Buffer
        Output buffer.

    clip : boolean
        Whether to clip out-of-boundary boxes.

    threshold : float
        Threshold to be a positive prediction.

    variances : tuple of float
        Variances to be decoded from box regression output.

    Returns
    -------
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
        return tvm.select(clip, tvm.max(0, tvm.min(1, ox - ow)), ox - ow), \
               tvm.select(clip, tvm.max(0, tvm.min(1, oy - oh)), oy - oh), \
               tvm.select(clip, tvm.max(0, tvm.min(1, ox + ow)), ox + ow), \
               tvm.select(clip, tvm.max(0, tvm.min(1, oy + oh)), oy + oh)

    batch_size = cls_prob.shape[0]
    num_classes = cls_prob.shape[1]
    num_anchors = cls_prob.shape[2]

    ib = tvm.ir_builder.create()
    p_cls_prob = ib.buffer_ptr(cls_prob)
    p_loc_pred = ib.buffer_ptr(loc_pred)
    p_anchor = ib.buffer_ptr(anchor)
    p_valid_count = ib.buffer_ptr(valid_count)
    p_out = ib.buffer_ptr(out)
    with ib.for_range(0, batch_size, for_type="parallel", name="n") as n:
        p_valid_count[n] = 0
        with ib.for_range(0, num_anchors, name="i") as i:
            # Find the predicted class id and probability
            score = ib.allocate('float32', (1,), name="score", scope="local")
            cls_id = ib.allocate('int32', (1,), name="id", scope="local")
            score[0] = -1.0
            cls_id[0] = 0
            with ib.for_range(0, num_classes, name="j") as j:
                with ib.if_scope(j > 0):
                    temp = p_cls_prob[n * num_anchors * num_classes + j * num_anchors + i]
                    cls_id[0] = tvm.select(temp > score[0], j, cls_id[0])
                    score[0] = tvm.max(temp, score[0])
            with ib.if_scope(tvm.all(cls_id[0] > 0, score[0] < threshold)):
                cls_id[0] = 0
            # [id, prob, xmin, ymin, xmax, ymax]
            # Remove background, restore original id
            with ib.if_scope(cls_id[0] > 0):
                out_base_idx = n * num_anchors * 6 + p_valid_count[n] * 6
                p_out[out_base_idx] = cls_id[0] - 1.0
                p_out[out_base_idx + 1] = score[0]
                offset = i * 4
                p_out[out_base_idx + 2], p_out[out_base_idx + 3], p_out[out_base_idx + 4], \
                p_out[out_base_idx + 5] = transform_loc(p_loc_pred, n * num_anchors * 4 + offset,
                                                        p_anchor, offset, clip, variances[0],
                                                        variances[1], variances[2], variances[3])
                p_valid_count[n] += 1

    return ib.get()


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
    batch_size = cls_prob.shape[0]
    num_anchors = anchor.shape[1]
    oshape = (batch_size, num_anchors, 6)
    # Define data alignment for intermediate buffer
    valid_count_dtype = "int32"
    valid_count_buf = api.decl_buffer((batch_size,), valid_count_dtype,
                                      "valid_count_buf", data_alignment=4)
    out_buf = api.decl_buffer(oshape, cls_prob.dtype, "out_buf", data_alignment=8)
    valid_count, out = \
        tvm.extern([(batch_size,), oshape],
                   [cls_prob, loc_pred, anchor],
                   lambda ins, outs: transform_loc_ir(
                       ins[0], ins[1], ins[2], outs[0], outs[1], clip, threshold, variances),
                   dtype=[valid_count_dtype, cls_prob.dtype],
                   out_buffers=[valid_count_buf, out_buf],
                   tag="multibox_transform_loc")
    return [out, valid_count]


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
