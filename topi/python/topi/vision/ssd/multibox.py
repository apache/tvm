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


def transform_loc_ir(cls_prob, loc_pred, anchor, valid_count, out, clip, threshold, variances):
    def transform_loc(loc, loc_base_idx, anchor, anchor_base_idx, clip, vx, vy, vw, vh):
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
        ow= tvm.exp(pw * vw) * aw / 2.0
        oh = tvm.exp(ph * vh) * ah / 2.0
        return tvm.select(clip, tvm.make.Max(0, tvm.make.Min(1, ox - ow)), ox - ow), \
               tvm.select(clip, tvm.make.Max(0, tvm.make.Min(1, oy - oh)), oy - oh), \
               tvm.select(clip, tvm.make.Max(0, tvm.make.Min(1, ox + ow)), ox + ow), \
               tvm.select(clip, tvm.make.Max(0, tvm.make.Min(1, oy + oh)), oy + oh)

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
            id = ib.allocate('int32', (1,), name="id", scope="local")
            score[0] = -1.0
            id[0] = 0
            with ib.for_range(0, num_classes, name="j") as j:
                with ib.if_scope(j > 0):
                    temp = p_cls_prob[n * num_anchors * num_classes + j * num_anchors + i]
                    id[0] = tvm.select(temp > score[0], j, id[0])
                    score[0] = tvm.make.Max(temp, score[0])
            with ib.if_scope(tvm.all(id[0] > 0, score[0] < threshold)):
                id[0] = 0
            # [id, prob, xmin, ymin, xmax, ymax]
            # Remove background, restore original id
            with ib.if_scope(id[0] > 0):
                out_base_idx = n * num_anchors * 6 + p_valid_count[n] * 6
                p_out[out_base_idx] = id[0] - 1.0
                p_out[out_base_idx + 1] = score[0]
                offset = i * 4
                p_out[out_base_idx + 2], p_out[out_base_idx + 3], p_out[out_base_idx + 4], \
                p_out[out_base_idx + 5] = transform_loc(p_loc_pred, n * num_anchors * 4 + offset,
                                                        p_anchor, offset, clip, variances[0],
                                                        variances[1], variances[2], variances[3])
                p_valid_count[n] += 1

    return ib.get()


def nms_ir(inter_out, sort_result, valid_count, out, nms_threshold, force_suppress, nms_topk):
    def calculate_overlap(out_tensor, box_a_idx, box_b_idx):
        w = tvm.make.Max(0.0, tvm.make.Min(out_tensor[box_a_idx + 2], out_tensor[box_b_idx + 2])
                         - tvm.make.Max(out_tensor[box_a_idx], out_tensor[box_b_idx]))
        h = tvm.make.Max(0.0, tvm.make.Min(out_tensor[box_a_idx + 3], out_tensor[box_b_idx + 3])
                         - tvm.make.Max(out_tensor[box_a_idx + 1], out_tensor[box_b_idx + 1]))
        i = w * h
        u = (out_tensor[box_a_idx + 2] - out_tensor[box_a_idx]) * \
            (out_tensor[box_a_idx + 3] - out_tensor[box_a_idx + 1]) + \
            (out_tensor[box_b_idx + 2] - out_tensor[box_b_idx]) * \
            (out_tensor[box_b_idx + 3] - out_tensor[box_b_idx + 1]) - i
        return tvm.select(u <= 0.0, 0.0, i / u)

    ib = tvm.ir_builder.create()
    p_inter_out = ib.buffer_ptr(inter_out)
    p_sort_result = ib.buffer_ptr(sort_result)
    p_valid_count = ib.buffer_ptr(valid_count)
    p_out = ib.buffer_ptr(out)
    batch_size = out.shape[0]
    num_anchors = out.shape[1]

    nms_threshold_node = tvm.make.node("FloatImm", dtype="float32", value=nms_threshold)
    nms_topk_node = tvm.make.node("IntImm", dtype="int32", value=nms_topk)
    force_suppress_node = tvm.make.node("IntImm", dtype="int32", value=1 if force_suppress else 0)
    with ib.for_range(0, batch_size, for_type="parallel", name="n") as n:
        with ib.if_scope(tvm.all(nms_threshold_node > 0, nms_threshold_node < 1, p_valid_count[0] > 0)):
            # Reorder output
            nkeep = ib.allocate('int32', (1,), name="nkeep", scope="local")
            nkeep[0] = p_valid_count[n]
            with ib.if_scope(tvm.all(nms_topk_node > 0, nms_topk_node < p_valid_count[n])):
                nkeep[0] = nms_topk
            with ib.for_range(0, nkeep[0], name="l") as l:
                with ib.for_range(0, 6, name="m") as m:
                    p_out[n * num_anchors * 6 + l * 6 + m] \
                        = p_inter_out[n * num_anchors * 6 + p_sort_result[n * num_anchors + l] * 6 + m]
            with ib.if_scope(nkeep[0] < p_valid_count[n]):
                with ib.for_range(0, p_valid_count[n] - nkeep[0], name="l") as l:
                    with ib.for_range(0, 6, name="m") as m:
                        p_out[n * num_anchors * 6 + (l + nkeep[0]) * 6 + m] \
                            = p_inter_out[n * num_anchors * 6 + (l + nkeep[0]) * 6 + m]
            # Apply nms
            with ib.for_range(0, p_valid_count[n], name="l") as l:
                offset_l = l * 6
                with ib.if_scope(p_out[n * num_anchors * 6 + offset_l] >= 0):
                    with ib.for_range(0, p_valid_count[n], name="m") as m:
                        offset_m = m * 6
                        with ib.if_scope(tvm.all(m > l, p_out[n * num_anchors * 6 + offset_m] >= 0)):
                            with ib.if_scope(tvm.any(force_suppress_node > 0,
                                                     p_out[n * num_anchors * 6 + offset_l]
                                                     == p_out[n * num_anchors * 6 + offset_m]
                                                     )):
                                # When force_suppress == True or class_id equals
                                iou = calculate_overlap(p_out, n * num_anchors * 6 + offset_l + 2,
                                                        n * num_anchors * 6 + offset_m + 2)
                                with ib.if_scope(iou >= nms_threshold):
                                    p_out[n * num_anchors * 6 + offset_m] = -1.0
        with ib.else_scope():
            with ib.for_range(0, p_valid_count[n], name="l") as l:
                with ib.for_range(0, 6, name="m") as m:
                    p_out[n * num_anchors * 6 + l * 6 + m] = p_inter_out[n * num_anchors * 6 + l * 6 + m]
        # Set invalid entry to be -1
        with ib.for_range(0, num_anchors - p_valid_count[n], name="l") as l:
            with ib.for_range(0, 6, name="m") as m:
                p_out[n * num_anchors * 6 + (l + p_valid_count[n]) * 6 + m] = -1.0
    return ib.get()


@tvm.target.generic_func
def multibox_detection(cls_prob, loc_pred, anchor, clip=True, threshold=0.01, nms_threshold=0.5,
                       force_suppress=False, variances=(0.1, 0.1, 0.2, 0.2), nms_topk=-1):
    batch_size = cls_prob.shape[0]
    num_anchors = anchor.shape[1]
    oshape = (batch_size, num_anchors, 6)
    valid_count, inter_out = tvm.extern([(batch_size,), oshape], [cls_prob, loc_pred, anchor], lambda ins, outs:
    transform_loc_ir(ins[0], ins[1], ins[2], outs[0], outs[1],
                     clip, threshold, variances),
                                        dtype=["int32", "float32"],
                                        tag="multibox_detection_transform_loc")
    sort_tensor = tvm.extern((batch_size, num_anchors), [inter_out, valid_count], lambda ins, outs:
    tvm.call_packed("tvm.contrib.generic.utils.stable_sort",ins[0], ins[1],
                    outs[0], batch_size, 1, True),
                             dtype='int32', name="multibox_detection_sort")
    out = tvm.extern(oshape, [inter_out, sort_tensor, valid_count], lambda ins, outs:
    nms_ir(ins[0], ins[1], ins[2], outs[0], nms_threshold, force_suppress, nms_topk),
                     dtype="float32", tag="multibox_detection_nms")
    return out