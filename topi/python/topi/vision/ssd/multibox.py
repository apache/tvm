"""SSD multibox operators"""
from __future__ import absolute_import as _abs
import tvm
import topi
import math

def multibox_prior_ir(data, out, sizes, ratios, steps, offsets):
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
                     multibox_prior_ir(ins[0], outs[0],sizes, ratios, steps, offsets),
                     tag="multibox_prior")
    if clip:
        out = topi.clip(out, 0, 1)
    return out


def multibox_detection_ir(cls_prob, loc_pred, anchor, out, clip, threshold,
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

    def calculate_overlap(out_tensor, box_a_idx, box_b_idx):
        w = tvm.make.Max(0, tvm.make.Min(out_tensor[box_a_idx + 2], out_tensor[box_b_idx + 2])) \
            - tvm.make.Max(out_tensor[box_a_idx], out_tensor[box_b_idx])
        h = tvm.make.Max(0, tvm.make.Min(out_tensor[box_a_idx + 3], out_tensor[box_b_idx + 3])) \
            - tvm.make.Max(out_tensor[box_a_idx + 1], out_tensor[box_b_idx + 1])
        i = w * h
        u = (out_tensor[box_a_idx + 2], out_tensor[box_a_idx]) * \
            (out_tensor[box_a_idx + 3], out_tensor[box_a_idx + 1]) + \
            (out_tensor[box_b_idx + 2], out_tensor[box_b_idx]) * \
            (out_tensor[box_b_idx + 3], out_tensor[box_b_idx] + 1) - i
        return tvm.select(u <= 0, 0.0, i / u)

    batch_size = cls_prob.shape[0]
    num_classes = cls_prob.shape[1]
    num_anchors = cls_prob.shape[2]

    ib = tvm.ir_builder.create()
    p_out = ib.buffer_ptr(out)
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

        with ib.if_scope(valid_count > 0 and 0 < nms_threshold <= 1):
            # Sort and apply NMS
            temp_space = ib.allocate('float32', (valid_count, 6), name="temp_space", scope="local")
            with ib.for_range(0, valid_count, name="i") as i:
                with ib.for_range(0, 6, name="j") as j:
                    temp_space[i * 6 + j] = p_out[n * num_anchors * 6 + i * 6 + j]
            # Sort confidence in descend order
            sort_result = tvm.extern((valid_count,), [temp_space], lambda ins, outs: \
                tvm.intrin.call_packed("tvm.contrib.generic.utils.stable_sort",
                                       ins[0], outs[0], 1, True), dtype='int64', name="C")
            # Redorder output
            nkeep = sort_result.shape[0]
            with ib.if_scope( 0 < nms_topk < nkeep):
                nkeep = nms_topk
            with ib.for_range(0, nkeep, name="i") as i:
                with ib.for_range(0, 6, name="j") as j:
                    p_out[n * num_anchors * 6 + i * 6 + j] = temp_space[sort_result[i] * 6 + j]
            # Apply nms
            with ib.for_range(0, valid_count, name="i") as i:
                offset_i = i * 6
                with ib.if_scope(p_out[n * num_anchors * 6 + offset_i] >= 0):
                    with ib.for_range(i + 1, valid_count, name="j") as j:
                        offset_j = j * 6
                        with ib.if_scope(p_out[n * num_anchors * 6 + offset_j] >= 0):
                            with ib.if_scope(force_suppress or p_out[n * num_anchors * 6 + offset_i]
                                             == p_out[n * num_anchors * 6 + offset_j]):
                                # When force_suppress == True or class_id equals
                                iou = calculate_overlap(p_out, n * num_anchors * 6 + offset_i + 2,
                                                        n * num_anchors * 6 + offset_j + 2)
                                with ib.if_scope(iou >= nms_threshold):
                                    p_out[n * num_anchors * 6 + offset_j] = -1


@tvm.target.generic_func
def multibox_detection(cls_prob, loc_pred, anchor, clip=True, threshold=0.01, nms_threshold=0.5,
                       force_suppress=False, variances=(0.1, 0.1, 0.2, 0.2), nms_topk=-1):
    batch_size = cls_prob.shape[0]
    num_anchor = anchor.shape[1]
    oshape = (batch_size, num_anchor, 6)
    out = tvm.extern(oshape, [cls_prob, loc_pred, anchor], lambda ins, outs:
                     multibox_detection_ir(ins[0], ins[1], ins[2], outs[0], clip, threshold, nms_threshold,
                                           force_suppress, variances, nms_topk), tag="multibox_detection")
    return out