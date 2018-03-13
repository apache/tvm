"""SSD multibox operators"""
from __future__ import absolute_import as _abs
import tvm
import topi
import math

def multibox_prior_IR(data, out, sizes, ratios, steps, offsets):
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
    num_sizes = len(sizes)
    num_ratios = len(ratios)
    oshape = (1, data.shape[2] * data.shape[3] * (num_sizes + num_ratios - 1), 4)
    out = tvm.extern(oshape, [data], lambda ins, outs:
                     multibox_prior_IR(ins[0], outs[0],sizes, ratios, steps, offsets),
                     tag="multibox_prior")
    if clip:
        out = topi.clip(out, 0, 1)
    return out



def multibox_detection_IR(cls_prob, loc_pred, anchor, out, clip, threshold,
                          nms_threshold, force_suppress, variances, nms_topk):
    def transform_loc(loc_pred, loc_base_idx, anchor, anchor_base_idx, clip, vx, vy, vw, vh):
        al = anchor[anchor_base_idx]
        at = anchor[anchor_base_idx + 1]
        ar = anchor[anchor_base_idx + 2]
        ab = anchor[anchor_base_idx + 3]
        aw = ar - al
        ah = ab - at
        ax = (al + at) / 2.0
        ay = (at + ab) / 2.0
        px = loc_pred[loc_base_idx]
        py = loc_pred[loc_base_idx + 1]
        pw = loc_pred[loc_base_idx + 2]
        ph = loc_pred[loc_base_idx + 3]
        ox = px * vx * aw + ax
        oy = py * vy * ah + ay
        ow= math.exp(pw * vw) * aw / 2
        oh = math.exp(ph * vh) * ah / 2
        return tvm.select(clip, tvm.make.Max(0, tvm.make.Min(1, ox - ow)), ox - ow), \
               tvm.select(clip, tvm.make.Max(0, tvm.make.Min(1, oy - oh)), oy - oh), \
               tvm.select(clip, tvm.make.Max(0, tvm.make.Min(1, ox + ow)), ox + ow), \
               tvm.select(clip, tvm.make.Max(0, tvm.make.Min(1, oy + oh)), oy + oh)

    def calculate_overlap(box_a, box_b):
        w = tvm.make.Max(0, tvm.make.Min(box_a[2], box_b[2])) - tvm.make.Max(box_a[0], box_b[0])
        h = tvm.make.Max(0, tvm.make.Min(box_a[3], box_b[3])) - tvm.make.Max(box_a[1], box_b[1])
        i = w * h
        u = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]) + \
            (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - i
        return tvm.select(u <= 0, 0.0, i / u)

    batch_size = cls_prob.shape[0]
    num_classes = cls_prob.shape[1]
    num_anchors = cls_prob.shape[2]

    ib = tvm.ir_builder.create()
    p_out = ib.buffer_ptr(out)
    temp_space = ib.allocate('float32', (num_anchors, 6), name="temp_space", scope="local")
    inner_loop_type = "parallel" if batch_size == 1 else "serial"
    with ib.for_range(0, batch_size, for_type="parallel", name="n") as n:
        valid_count = 0
        with ib.for_range(0, num_anchors, name="i") as i:
            # Find the predicted class id and probability
            score = -1
            id = 0
            with ib.for_range(0, num_classes, name="j") as j:
                temp = cls_prob[n * num_classes * num_classes + j * num_classes + i]
                id = tvm.select(temp > score, id, j)
                score = tvm.make.Max(temp, score)
            with ib.if_scope(id > 0 and score < threshold):
                id = 0
            # [id, prob, xmin, ymin, xmax, ymax]
            # Remove background, restore original id
            with ib.if_scope(id > 0):
                out_base_idx = n * num_anchors * 6 + valid_count * 6
                p_out[out_base_idx] = tvm.select(id > 0, id - 1, p_out[out_base_idx])
                p_out[out_base_idx + 1] = tvm.select(id > 0, score, p_out[out_base_idx + 1])
                offsets = i * 4
                p_out[out_base_idx + 2], p_out[out_base_idx + 3], p_out[out_base_idx + 4], \
                p_out[out_base_idx + 5] = transform_loc(loc_pred, n * num_anchors * 4 + offsets,
                                                      anchor, offsets, clip, variances[0],
                                                      variances[1], variances[2], variances[3])
                valid_count += 1

        with ib.if_scope(valid_count > 0 and nms_threshold > 0 and nms_threshold <= 1):
            # Sort and apply NMS
            with ib.for_range(0, num_anchors, name="i") as i:
                with ib.for_range(0, 6, name="j") as j:
                    temp_space[i * 6 + j] = p_out[n * num_anchors * 6 + i * 6 + j]
            # sort confidence in descend order
            sorter = temp_space
            nkeep = sorter.shape[0]
            with ib.if_scope( 0 < nms_topk < nkeep and nms_topk < nkeep):











