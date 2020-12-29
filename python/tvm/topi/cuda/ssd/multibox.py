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
# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments, too-many-statements, too-many-function-args
"""SSD multibox operators"""
import math
import tvm
from tvm import te
from tvm.tir import if_then_else, exp

from tvm import topi

from ..nms import non_max_suppression


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
    max_threads = int(math.sqrt(tvm.target.Target.current(allow_none=False).max_num_threads))
    tx = te.thread_axis("threadIdx.x")
    ty = te.thread_axis("threadIdx.y")
    bx = te.thread_axis("blockIdx.x")
    by = te.thread_axis("blockIdx.y")
    ib = tvm.tir.ir_builder.create()
    p_out = ib.buffer_ptr(out)
    in_height = data.shape[2]
    in_width = data.shape[3]
    nthread_tx = max_threads
    nthread_bx = in_height // max_threads + 1
    nthread_ty = max_threads
    nthread_by = in_width // max_threads + 1
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(ty, "thread_extent", nthread_ty)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    ib.scope_attr(by, "thread_extent", nthread_by)

    num_sizes = len(sizes)
    num_ratios = len(ratios)
    size_ratio_concat = sizes + ratios
    steps_h = steps[0] if steps[0] > 0 else 1.0 / in_height
    steps_w = steps[1] if steps[1] > 0 else 1.0 / in_width
    offset_h = offsets[0]
    offset_w = offsets[1]

    i = bx * max_threads + tx
    j = by * max_threads + ty
    with ib.if_scope((i < in_height)):
        with ib.if_scope((j < in_width)):
            center_h = (i + offset_h) * steps_h
            center_w = (j + offset_w) * steps_w

            for k in range(num_sizes + num_ratios - 1):
                w = if_then_else(
                    k < num_sizes,
                    float(size_ratio_concat[k]) * in_height / in_width / 2.0,
                    float(size_ratio_concat[0])
                    * in_height
                    / in_width
                    * math.sqrt(size_ratio_concat[k + 1])
                    / 2.0,
                )
                h = if_then_else(
                    k < num_sizes,
                    size_ratio_concat[k] / 2.0,
                    size_ratio_concat[0] / math.sqrt(size_ratio_concat[k + 1]) / 2.0,
                )
                count = (
                    i * in_width * (num_sizes + num_ratios - 1)
                    + j * (num_sizes + num_ratios - 1)
                    + k
                ) * 4
                p_out[count] = center_w - w
                p_out[count + 1] = center_h - h
                p_out[count + 2] = center_w + w
                p_out[count + 3] = center_h + h

    body = ib.get()
    return body


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
    num_sizes = len(sizes)
    num_ratios = len(ratios)
    oshape = (1, data.shape[2] * data.shape[3] * (num_sizes + num_ratios - 1), 4)
    out = te.extern(
        oshape,
        [data],
        lambda ins, outs: multibox_prior_ir(ins[0], outs[0], sizes, ratios, steps, offsets),
        tag="multibox_prior",
    )
    if clip:
        out = topi.clip(out, 0, 1)
    return out


def transform_loc_pre(cls_prob, valid_count, temp_valid_count, temp_cls_id, temp_score, threshold):
    """Low level IR routing for transform location data preparation.

    Parameters
    ----------
    cls_prob : Buffer
        Buffer of class probabilities.

    valid_count : Buffer
        Buffer of number of valid output boxes.

    temp_valid_count : Buffer
        Output intermediate result buffer

    temp_cls_id : Buffer
        Output intermediate result buffer

    temp_score : Buffer
        Output buffer

    threshold : float
        Threshold to be a positive prediction.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    batch_size = cls_prob.shape[0]
    num_classes = cls_prob.shape[1]
    num_anchors = cls_prob.shape[2]

    ib = tvm.tir.ir_builder.create()

    cls_prob = ib.buffer_ptr(cls_prob)
    cls_id = ib.buffer_ptr(temp_cls_id)
    valid_count = ib.buffer_ptr(valid_count)
    temp_valid_count = ib.buffer_ptr(temp_valid_count)
    score = ib.buffer_ptr(temp_score)

    threshold = tvm.tir.FloatImm("float32", threshold)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_tx = max_threads
    nthread_bx = (batch_size * num_anchors) // max_threads + 1
    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    tid = bx * max_threads + tx
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    with ib.if_scope(tid < batch_size * num_anchors):
        i = idxd(tid, num_anchors)
        j = idxm(tid, num_anchors)
        valid_count[i] = 0
        score[tid] = -1.0
        cls_id[tid] = 0
        with ib.for_range(0, num_classes - 1) as k:
            temp = cls_prob[i * num_classes * num_anchors + (k + 1) * num_anchors + j]
            cls_id[tid] = if_then_else(temp > score[tid], k + 1, cls_id[tid])
            score[tid] = tvm.te.max(temp, score[tid])
        with ib.if_scope(tvm.tir.all(cls_id[tid] > 0, score[tid] < threshold)):
            cls_id[tid] = 0
        with ib.if_scope(cls_id[tid] > 0):
            temp_valid_count[tid] = 1
        with ib.else_scope():
            temp_valid_count[tid] = 0

        with ib.if_scope(tid < batch_size):
            with ib.for_range(0, num_anchors) as k:
                with ib.if_scope(k > 0):
                    temp_valid_count[tid * num_anchors + k] += temp_valid_count[
                        tid * num_anchors + k - 1
                    ]
            valid_count[i] = temp_valid_count[tid * num_anchors + num_anchors - 1]

    return ib.get()


def transform_loc_ir(
    loc_pred,
    anchor,
    temp_valid_count,
    temp_cls_id,
    temp_score,
    out,
    clip,
    variances,
    batch_size,
    num_anchors,
):
    """Low level IR routing for transform location in multibox_detection operator.

    Parameters
    ----------
    loc_pred : Buffer
        Buffer of location regression predictions.

    anchor : Buffer
        Buffer of prior anchor boxes.

    temp_valid_count : Buffer
        Intermediate result buffer.

    temp_cls_id : Buffer
        Intermediate result buffer.

    temp_score : Buffer
        Input buffer which stores intermediate results.

    out : Buffer
        Output buffer.

    clip : boolean
        Whether to clip out-of-boundary boxes.

    variances : tuple of float
        Variances to be decoded from box regression output.

    batch_size : int
        Batch size

    num_anchors : int
        Number of anchors

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """

    def transform_loc(loc, loc_base_idx, anchor, anchor_base_idx, clip, vx, vy, vw, vh):
        """Transform prior anchor box to output box through location predictions."""
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
        ow = exp(pw * vw) * aw / 2.0
        oh = exp(ph * vh) * ah / 2.0
        return (
            tvm.tir.if_then_else(clip, tvm.te.max(0.0, tvm.te.min(1.0, ox - ow)), ox - ow),
            tvm.tir.if_then_else(clip, tvm.te.max(0.0, tvm.te.min(1.0, oy - oh)), oy - oh),
            tvm.tir.if_then_else(clip, tvm.te.max(0.0, tvm.te.min(1.0, ox + ow)), ox + ow),
            tvm.tir.if_then_else(clip, tvm.te.max(0.0, tvm.te.min(1.0, oy + oh)), oy + oh),
        )

    ib = tvm.tir.ir_builder.create()

    loc_pred = ib.buffer_ptr(loc_pred)
    anchor = ib.buffer_ptr(anchor)
    temp_valid_count = ib.buffer_ptr(temp_valid_count)
    cls_id = ib.buffer_ptr(temp_cls_id)
    score = ib.buffer_ptr(temp_score)
    out_loc = ib.buffer_ptr(out)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_tx = max_threads
    nthread_bx = (batch_size * num_anchors) // max_threads + 1
    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    tid = bx * max_threads + tx

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    with ib.if_scope(tid < batch_size * num_anchors):
        i = idxd(tid, num_anchors)
        j = idxm(tid, num_anchors)

        with ib.if_scope(cls_id[tid] > 0):
            with ib.if_scope(tid == 0):
                out_base_idx = i * num_anchors * 6
                out_loc[out_base_idx] = cls_id[tid] - 1.0
                out_loc[out_base_idx + 1] = score[tid]
                (
                    out_loc[out_base_idx + 2],
                    out_loc[out_base_idx + 3],
                    out_loc[out_base_idx + 4],
                    out_loc[out_base_idx + 5],
                ) = transform_loc(
                    loc_pred,
                    tid * 4,
                    anchor,
                    j * 4,
                    clip,
                    variances[0],
                    variances[1],
                    variances[2],
                    variances[3],
                )
            with ib.else_scope():
                out_base_idx = i * num_anchors * 6 + temp_valid_count[tid - 1] * 6
                out_loc[out_base_idx] = cls_id[tid] - 1.0
                out_loc[out_base_idx + 1] = score[tid]
                (
                    out_loc[out_base_idx + 2],
                    out_loc[out_base_idx + 3],
                    out_loc[out_base_idx + 4],
                    out_loc[out_base_idx + 5],
                ) = transform_loc(
                    loc_pred,
                    tid * 4,
                    anchor,
                    j * 4,
                    clip,
                    variances[0],
                    variances[1],
                    variances[2],
                    variances[3],
                )

    return ib.get()


def multibox_transform_loc(
    cls_prob, loc_pred, anchor, clip=True, threshold=0.01, variances=(0.1, 0.1, 0.2, 0.2)
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

    Returns
    -------
    ret : tuple of tvm.te.Tensor composed of

    out : tvm.te.Tensor
        3-D tensor with shape (batch_size, num_anchors, 6)

    valid_count : tvm.te.Tensor
        1-D tensor with shape (batch_size,), number of valid anchor boxes.
    """
    batch_size = cls_prob.shape[0]
    num_anchors = cls_prob.shape[2]
    oshape = (batch_size, num_anchors, 6)
    # Define data alignment for intermediate buffer
    valid_count_dtype = "int32"
    out_loc_dtype = loc_pred.dtype

    valid_count_buf = tvm.tir.decl_buffer(
        (batch_size,), valid_count_dtype, "valid_count_buf", data_alignment=4
    )
    loc_pred_buf = tvm.tir.decl_buffer(
        loc_pred.shape, loc_pred.dtype, "loc_pred_buf", data_alignment=8
    )
    anchor_buf = tvm.tir.decl_buffer(anchor.shape, anchor.dtype, "anchor_buf", data_alignment=8)

    temp_valid_count_buf = tvm.tir.decl_buffer(
        (
            batch_size,
            num_anchors,
        ),
        valid_count_dtype,
        "temp_valid_count",
        data_alignment=8,
    )
    temp_cls_id_buf = tvm.tir.decl_buffer(
        (
            batch_size,
            num_anchors,
        ),
        valid_count_dtype,
        "temp_cls_id",
        data_alignment=8,
    )
    temp_score_buf = tvm.tir.decl_buffer(
        (
            batch_size,
            num_anchors,
        ),
        cls_prob.dtype,
        "temp_score",
        data_alignment=8,
    )

    valid_count, temp_valid_count, temp_cls_id, temp_score = te.extern(
        [
            (batch_size,),
            (
                batch_size,
                num_anchors,
            ),
            (
                batch_size,
                num_anchors,
            ),
            (
                batch_size,
                num_anchors,
            ),
        ],
        [cls_prob],
        lambda ins, outs: transform_loc_pre(ins[0], outs[0], outs[1], outs[2], outs[3], threshold),
        dtype=[valid_count_dtype, valid_count_dtype, valid_count_dtype, cls_prob.dtype],
        out_buffers=[valid_count_buf, temp_valid_count_buf, temp_cls_id_buf, temp_score_buf],
        tag="multibox_transform_loc_phase_one",
    )

    out_loc = te.extern(
        [oshape],
        [loc_pred, anchor, temp_valid_count, temp_cls_id, temp_score],
        lambda ins, outs: transform_loc_ir(
            ins[0],
            ins[1],
            ins[2],
            ins[3],
            ins[4],
            outs[0],
            clip,
            variances,
            batch_size,
            num_anchors,
        ),
        in_buffers=[
            loc_pred_buf,
            anchor_buf,
            temp_valid_count_buf,
            temp_cls_id_buf,
            temp_score_buf,
        ],
        dtype=[out_loc_dtype],
        tag="multibox_transform_loc",
    )

    return [out_loc, valid_count]


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
