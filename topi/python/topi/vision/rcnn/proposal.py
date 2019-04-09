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
# pylint: disable=invalid-name
"""Proposal operator"""
import math
import tvm


def generate_anchor(ratio, scale, base_size):
    """Generate anchor"""
    w = h = float(base_size)
    x_ctr = 0.5 * (w - 1.)
    y_ctr = 0.5 * (h - 1.)
    size = w * h
    size_ratios = math.floor(size / ratio)
    new_w = math.floor(math.sqrt(size_ratios) + 0.5) * scale
    new_h = math.floor((new_w / scale * ratio) + 0.5) * scale
    return (x_ctr - 0.5 * (new_w - 1.0), y_ctr - 0.5 * (new_h - 1.0),
            x_ctr + 0.5 * (new_w - 1.0), y_ctr + 0.5 * (new_h - 1.0))


def reg_bbox(x1, y1, x2, y2, dx, dy, dw, dh):
    """Bounding box regression function"""
    bbox_w = x2 - x1 + 1.0
    bbox_h = y2 - y1 + 1.0
    ctr_x = x1 + 0.5 * (bbox_w - 1.0)
    ctr_y = y1 + 0.5 * (bbox_h - 1.0)

    pred_ctr_x = dx * bbox_w + ctr_x
    pred_ctr_y = dy * bbox_h + ctr_y
    pred_w = tvm.exp(dw) * bbox_w
    pred_h = tvm.exp(dh) * bbox_h

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


@tvm.target.generic_func
def proposal(cls_prob, bbox_pred, im_info, scales, ratios, feature_stride, threshold,
             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_min_size, iou_loss):
    """Proposal operator.

    Parameters
    ----------
    cls_prob : tvm.Tensor
        4-D with shape [batch, 2 * num_anchors, height, width]

    bbox_pred : tvm.Tensor
        4-D with shape [batch, 4 * num_anchors, height, width]

    im_info : tvm.Tensor
        2-D with shape [batch, 3]

    scales : list/tuple of float
        Scales of anchor windoes.

    ratios : list/tuple of float
        Ratios of anchor windoes.

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
    out : tvm.Tensor
        2-D tensor with shape [batch * rpn_post_nms_top_n, 5]. The last dimension is in format of
        [batch_index, w_start, h_start, w_end, h_end].
    """
    # pylint: disable=unused-argument
    raise ValueError("missing register for topi.vision.rcnn.proposal")
