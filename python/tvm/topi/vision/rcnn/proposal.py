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
# pylint: disable=invalid-name, singleton-comparison, bad-continuation
"""Proposal operator"""
import math
import tvm
from tvm import te
from ...utils import get_const_tuple, get_const_int
from ...sort import argsort


def generate_anchor(ratio, scale, base_size):
    """Generate anchor"""
    w = h = float(base_size)
    x_ctr = 0.5 * (w - 1.0)
    y_ctr = 0.5 * (h - 1.0)
    size = w * h
    size_ratios = math.floor(size / ratio)
    new_w = math.floor(math.sqrt(size_ratios) + 0.5) * scale
    new_h = math.floor((new_w / scale * ratio) + 0.5) * scale
    return (
        x_ctr - 0.5 * (new_w - 1.0),
        y_ctr - 0.5 * (new_h - 1.0),
        x_ctr + 0.5 * (new_w - 1.0),
        y_ctr + 0.5 * (new_h - 1.0),
    )


def reg_bbox(x1, y1, x2, y2, dx, dy, dw, dh):
    """Bounding box regression function"""
    bbox_w = x2 - x1 + 1.0
    bbox_h = y2 - y1 + 1.0
    ctr_x = x1 + 0.5 * (bbox_w - 1.0)
    ctr_y = y1 + 0.5 * (bbox_h - 1.0)

    pred_ctr_x = dx * bbox_w + ctr_x
    pred_ctr_y = dy * bbox_h + ctr_y
    pred_w = te.exp(dw) * bbox_w
    pred_h = te.exp(dh) * bbox_h

    pred_x1 = pred_ctr_x - 0.5 * (pred_w - 1.0)
    pred_y1 = pred_ctr_y - 0.5 * (pred_h - 1.0)
    pred_x2 = pred_ctr_x + 0.5 * (pred_w - 1.0)
    pred_y2 = pred_ctr_y + 0.5 * (pred_h - 1.0)
    return pred_x1, pred_y1, pred_x2, pred_y2


def reg_iou(x1, y1, x2, y2, dx1, dy1, dx2, dy2):
    """Bounding box regression function"""
    pred_x1 = x1 + dx1
    pred_y1 = y1 + dy1
    pred_x2 = x2 + dx2
    pred_y2 = y2 + dy2
    return pred_x1, pred_y1, pred_x2, pred_y2


def predict_bbox_ir(
    cls_prob_buf,
    bbox_pred_buf,
    im_info_buf,
    out_buf,
    scales,
    ratios,
    feature_stride,
    rpn_min_size,
    iou_loss,
):
    """Predict bounding boxes based on anchors, scores and deltas.

    Parameters
    ----------
    cls_prob_buf : tvm.te.schedule.Buffer
        4-D with shape [batch, 2 * num_anchors, height, width]

    bbox_pred_buf : tvm.te.schedule.Buffer
        4-D with shape [batch, 4 * num_anchors, height, width]

    im_info_buf : tvm.te.schedule.Buffer
        2-D with shape [batch, 3]

    out_buf : tvm.te.schedule.Buffer
        3-D with shape [batch, num_bbox, 5]
        The last dimension is in format of [w_start, h_start, w_end, h_end, score]

    scales : list/tuple of float
        Scales of anchor windows.

    ratios : list/tuple of float
        Ratios of anchor windows.

    feature_stride : int
        The size of the receptive field each unit in the convolution layer of the rpn, for example
        the product of all stride's prior to this layer.

    rpn_min_size : int
        Minimum height or width in proposal.

    iou_loss : bool
        Usage of IoU loss.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    batch, num_anchors, height, width = get_const_tuple(cls_prob_buf.shape)
    num_anchors //= 2
    ib = tvm.tir.ir_builder.create()

    p_score = ib.buffer_ptr(cls_prob_buf)
    p_delta = ib.buffer_ptr(bbox_pred_buf)
    p_im_info = ib.buffer_ptr(im_info_buf)
    p_out = ib.buffer_ptr(out_buf)

    idxm = tvm.tir.indexmod
    idxd = tvm.tir.indexdiv

    with ib.for_range(0, batch * height * width) as tid:
        w = idxm(tid, width)
        h = idxm(idxd(tid, width), height)
        b = idxd(idxd(tid, width), height)

        for k in range(num_anchors):
            out_index = tid * num_anchors + k
            ratio = ratios[k // len(scales)]
            scale = scales[k % len(scales)]
            anchor = generate_anchor(ratio, scale, feature_stride)
            im_height = p_im_info[b * 3]
            im_width = p_im_info[b * 3 + 1]
            x1 = anchor[0] + w * feature_stride
            y1 = anchor[1] + h * feature_stride
            x2 = anchor[2] + w * feature_stride
            y2 = anchor[3] + h * feature_stride

            delta = [
                p_delta[((((b * num_anchors + k) * 4 + i) * height + h) * width + w)]
                for i in range(4)
            ]
            regression_func = reg_iou if iou_loss else reg_bbox
            pred_x1, pred_y1, pred_x2, pred_y2 = regression_func(x1, y1, x2, y2, *delta)

            pred_x1 = tvm.te.max(tvm.te.min(pred_x1, im_width - 1.0), 0.0)
            pred_y1 = tvm.te.max(tvm.te.min(pred_y1, im_height - 1.0), 0.0)
            pred_x2 = tvm.te.max(tvm.te.min(pred_x2, im_width - 1.0), 0.0)
            pred_y2 = tvm.te.max(tvm.te.min(pred_y2, im_height - 1.0), 0.0)

            real_height = (im_height / feature_stride).astype("int32")
            real_width = (im_width / feature_stride).astype("int32")

            bbox_w = pred_x2 - pred_x1 + 1.0
            bbox_h = pred_y2 - pred_y1 + 1.0
            min_size = p_im_info[b * 3 + 2] * rpn_min_size

            pred_score = p_score[((b * num_anchors * 2 + num_anchors + k) * height + h) * width + w]
            pred_score = tvm.tir.Select(
                tvm.tir.any(h >= real_height, w >= real_width), -1.0, pred_score
            )
            p_out[out_index * 5 + 0] = pred_x1
            p_out[out_index * 5 + 1] = pred_y1
            p_out[out_index * 5 + 2] = pred_x2
            p_out[out_index * 5 + 3] = pred_y2
            p_out[out_index * 5 + 4] = pred_score

            with ib.if_scope(tvm.tir.any(bbox_w < min_size, bbox_h < min_size)):
                p_out[out_index * 5 + 0] -= min_size / 2.0
                p_out[out_index * 5 + 1] -= min_size / 2.0
                p_out[out_index * 5 + 2] += min_size / 2.0
                p_out[out_index * 5 + 3] += min_size / 2.0
                p_out[out_index * 5 + 4] = -1.0

    return ib.get()


def argsort_ir(data_buf, out_index_buf):
    """Batched odd-even transposition sort.

    Parameters
    ----------
    data_buf : tvm.te.schedule.Buffer
        2-D with shape [batch, num_bbox]

    out_index_buf : tvm.te.schedule.Buffer
        2-D with shape [batch, num_bbox]. Indices of data in sorted order.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    batch, num_bbox = get_const_tuple(data_buf.shape)
    ib = tvm.tir.ir_builder.create()
    p_data = ib.buffer_ptr(data_buf)
    index_out = ib.buffer_ptr(out_index_buf)
    temp_data = ib.allocate("float32", (1,), name="temp_data", scope="local")
    temp_index = ib.allocate("int32", (1,), name="temp_index", scope="local")
    idxm = tvm.tir.indexmod
    with ib.for_range(0, batch, kind="unroll") as b:
        start = b * num_bbox
        for i in range(2):
            with ib.for_range(0, (num_bbox + 1) // 2) as tid:
                bbox_id = tid * 2 + i
                with ib.if_scope(bbox_id < num_bbox):
                    index_out[start + bbox_id] = bbox_id
        with ib.for_range(0, num_bbox) as k:
            with ib.for_range(0, (num_bbox + 1) // 2) as tid:
                offset = start + 2 * tid + idxm(k, 2)
                with ib.if_scope(
                    tvm.tir.all(offset + 1 < num_bbox, p_data[offset] < p_data[offset + 1])
                ):
                    temp_data[0] = p_data[offset]
                    p_data[offset] = p_data[offset + 1]
                    p_data[offset + 1] = temp_data[0]
                    temp_index[0] = index_out[offset]
                    index_out[offset] = index_out[offset + 1]
                    index_out[offset + 1] = temp_index[0]
    return ib.get()


def nms_ir(sorted_bbox_buf, out_buf, nms_threshold):
    """Non-maximum suppression.

    Parameters
    ----------
    sorted_bbox_buf : tvm.te.schedule.Buffer
        3-D with shape [batch, num_bbox, 5]. The last dimension is in format of
        [w_start, h_start, w_end, h_end, score].

    out_buf : tvm.te.schedule.Buffer
        2-D with shape [batch, num_bbox]. Boolean mask of whether a bounding box should be removed.

    nms_threshold : float
        Non-maximum suppression threshold.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """

    def calculate_overlap(out_tensor, box_a_idx, box_b_idx):
        """Calculate overlap of two boxes."""
        w = tvm.te.max(
            0.0,
            tvm.te.min(out_tensor[box_a_idx + 2], out_tensor[box_b_idx + 2])
            - tvm.te.max(out_tensor[box_a_idx], out_tensor[box_b_idx])
            + 1.0,
        )
        h = tvm.te.max(
            0.0,
            tvm.te.min(out_tensor[box_a_idx + 3], out_tensor[box_b_idx + 3])
            - tvm.te.max(out_tensor[box_a_idx + 1], out_tensor[box_b_idx + 1])
            + 1.0,
        )
        i = w * h
        u = (
            (out_tensor[box_a_idx + 2] - out_tensor[box_a_idx] + 1.0)
            * (out_tensor[box_a_idx + 3] - out_tensor[box_a_idx + 1] + 1.0)
            + (out_tensor[box_b_idx + 2] - out_tensor[box_b_idx] + 1.0)
            * (out_tensor[box_b_idx + 3] - out_tensor[box_b_idx + 1] + 1.0)
            - i
        )
        return i / u

    batch, num_bbox = get_const_tuple(out_buf.shape)
    ib = tvm.tir.ir_builder.create()
    p_data = ib.buffer_ptr(sorted_bbox_buf)
    p_out = ib.buffer_ptr(out_buf)
    with ib.for_range(0, batch, kind="unroll", name="n") as b:
        base_idx = b * num_bbox
        for i in range(num_bbox):
            p_out[base_idx + i] = False
        with ib.for_range(0, num_bbox - 1) as l:
            with ib.for_range(0, num_bbox) as i:
                with ib.if_scope(tvm.tir.all(i < num_bbox, i > l, p_out[base_idx + l] == False)):
                    iou = calculate_overlap(p_data, (base_idx + l) * 5, (base_idx + i) * 5)
                    with ib.if_scope(iou > nms_threshold):
                        p_out[base_idx + i] = True
    return ib.get()


def prepare_output_ir(sorted_bbox_buf, remove_mask_buf, out_buf):
    """Copy output after applying nms to continuous memory.

    Parameters
    ----------
    sorted_bbox_buf : tvm.te.schedule.Buffer
        3-D with shape [batch, num_bbox, 5]. The last dimension is in format of
        [w_start, h_start, w_end, h_end, score].

    remove_mask_buf : tvm.te.schedule.Buffer
        2-D with shape [batch, num_bbox]. Boolean mask of whether a bounding box should be removed.

    out_buf : tvm.te.schedule.Buffer
        2-D with shape [batch * rpn_post_nms_top_n, 5]. The last dimension is in format of
        [batch_index, w_start, h_start, w_end, h_end].

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    batch, num_bbox, _ = get_const_tuple(sorted_bbox_buf.shape)
    rpn_post_nms_top_n = get_const_int(out_buf.shape[0]) // batch
    ib = tvm.tir.ir_builder.create()
    i = ib.allocate("int32", (batch,), "i", scope="local")
    p_sorted_bbox = ib.buffer_ptr(sorted_bbox_buf)
    p_remove = ib.buffer_ptr(remove_mask_buf)
    p_out = ib.buffer_ptr(out_buf)

    nkeep = ib.allocate("int32", (batch,), "nkeep", scope="local")

    with ib.for_range(0, batch) as b:
        nkeep[b] = 0
        i[b] = 0

    with ib.for_range(0, num_bbox) as j:
        with ib.for_range(0, batch) as b:
            with ib.if_scope(p_remove[b * num_bbox + j] == False):
                nkeep[b] += 1
    with ib.for_range(0, batch) as b:
        with ib.if_scope(nkeep[b] > 0):
            with ib.for_range(
                0, te.ceil(tvm.tir.const(rpn_post_nms_top_n, "float32") / nkeep[b]).astype("int32")
            ):
                with ib.for_range(0, num_bbox) as j:
                    offset_j = (b * num_bbox + j) * 5
                    offset_i = (b * rpn_post_nms_top_n + i[b]) * 5
                    with ib.if_scope(
                        tvm.tir.all(
                            i[b] < rpn_post_nms_top_n, p_remove[(b * num_bbox + j)] == False
                        )
                    ):
                        p_out[offset_i] = tvm.tir.Cast("float32", b)
                        with ib.for_range(0, 4, kind="unroll") as k:
                            p_out[offset_i + k + 1] = p_sorted_bbox[offset_j + k]
                        i[b] = i[b] + 1

    body = ib.get()
    return body


def proposal(
    cls_prob,
    bbox_pred,
    im_info,
    scales,
    ratios,
    feature_stride,
    threshold,
    rpn_pre_nms_top_n,
    rpn_post_nms_top_n,
    rpn_min_size,
    iou_loss,
):
    """Proposal operator.

    Parameters
    ----------
    cls_prob : tvm.te.Tensor
        4-D with shape [batch, 2 * num_anchors, height, width]

    bbox_pred : tvm.te.Tensor
        4-D with shape [batch, 4 * num_anchors, height, width]

    im_info : tvm.te.Tensor
        2-D with shape [batch, 3]

    scales : list/tuple of float
        Scales of anchor windows.

    ratios : list/tuple of float
        Ratios of anchor windows.

    feature_stride : int
        The size of the receptive field each unit in the convolution layer of the rpn, for example
        the product of all stride's prior to this layer.

    threshold : float
        Non-maximum suppression threshold.

    rpn_pre_nms_top_n : int
        Number of top scoring boxes to apply NMS. -1 to use all boxes.

    rpn_post_nms_top_n : int
        Number of top scoring boxes to keep after applying NMS to RPN proposals.

    rpn_min_size : int
        Minimum height or width in proposal.

    iou_loss : bool
        Usage of IoU loss.

    Returns
    -------
    out : tvm.te.Tensor
        2-D tensor with shape [batch * rpn_post_nms_top_n, 5]. The last dimension is in format of
        [batch_index, w_start, h_start, w_end, h_end].
    """
    # pylint: disable=unused-argument
    batch, _, height, width = get_const_tuple(cls_prob.shape)
    num_anchors = len(scales) * len(ratios)
    num_bbox = height * width * num_anchors
    rpn_pre_nms_top_n = min(rpn_pre_nms_top_n, num_bbox) if rpn_pre_nms_top_n > 0 else num_bbox

    bbox = te.extern(
        (batch, num_bbox, 5),
        [cls_prob, bbox_pred, im_info],
        lambda ins, outs: predict_bbox_ir(
            ins[0], ins[1], ins[2], outs[0], scales, ratios, feature_stride, rpn_min_size, iou_loss
        ),
        dtype=bbox_pred.dtype,
    )
    score = te.compute((batch, num_bbox), lambda b, i: bbox[b, i, 4], tag="bbox_score")
    valid_count_shape = (1,)
    valid_count = te.compute(valid_count_shape, lambda i: num_bbox)
    sorted_index = argsort(score, valid_count=valid_count, axis=1, is_ascend=False)
    sorted_bbox = te.compute(
        (batch, rpn_pre_nms_top_n, 5),
        lambda b, i, j: bbox[b, sorted_index[b, i], j],
        tag="sorted_bbox",
    )
    nms_remove_mask = te.extern(
        (batch, rpn_pre_nms_top_n),
        [sorted_bbox],
        lambda ins, outs: nms_ir(ins[0], outs[0], threshold),
        dtype="bool",
    )
    nms_out = te.extern(
        (batch * rpn_post_nms_top_n, 5),
        [sorted_bbox, nms_remove_mask],
        lambda ins, outs: prepare_output_ir(ins[0], ins[1], outs[0]),
        dtype=sorted_bbox.dtype,
    )
    return nms_out
