# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments
"""Non-maximum suppression operator"""
import tvm

from tvm import api, hybrid

@hybrid.script
def rearrange_out(input, output):
    """Rearrange nms output to move all valid entries to top.

    Parameters
    ----------
    input : Tensor or Var or numpy NDArray
        NMS output. 3-D tensor with shape
        [batch_size, num_anchors, 6].

    output : Tensor or Var or numpy NDArray
        Transformed NMS output. 3-D tensor with shape
        [batch_size, num_anchors, 6].
    """
    batch_size = input.shape[0]
    num_anchors = input.shape[1]
    elem_length = input.shape[2]
    for i in range(batch_size):
        for j in range(num_anchors):
            for k in range(elem_length):
                output[i, j, k] = -1.0

    for i in range(batch_size):
        valid_idx = 0
        for j in range(num_anchors):
            if input[i, j, 0] >= 0:
                for k in range(elem_length):
                    output[i, valid_idx, k] = input[i, j, k]
                valid_idx = valid_idx + 1


@hybrid.script
def get_valid_counts(data, inter_data, valid_count, score_threshold):
    """Get valid count of bounding boxes given a score threshlod.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : Tensor or Var or numpy NDArray
        Input data. 3-D tensor with shape [batch_size, num_anchors, 6].

    inter_data : Tensor or Var or numpy NDArray
        Intermediate output. 3-D tensor with shape
        [batch_size, num_anchors, 6].

    valid_count : Tensor or Var or numpy NDArray
        1-D tensor for valid number of boxes.

    score_threshold : float
        Lower limit of score for valid bounding boxes.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    for i in range(batch_size):
        valid_count[i] = 0
        inter_idx = 0
        for j in range(num_anchors):
            score = data[i, j, 1]
            if score >= score_threshold:
                valid_count[i] += 1
                inter_data[i, inter_idx] = data[i, j]
                inter_idx = inter_idx + 1


def nms_ir(data, sort_result, valid_count, out, nms_threshold, force_suppress, nms_topk):
    """Low level IR routing for transform location in multibox_detection operator.

    Parameters
    ----------
    data: Buffer
        Buffer of output boxes with class and score.

    sort_result : Buffer
        Buffer of output box indexes sorted by score.

    valid_count : Buffer
        Buffer of number of valid output boxes.

    out : Buffer
        Output buffer.

    nms_threshold : float
        Non-maximum suppression threshold.

    force_suppress : boolean
        Whether to suppress all detections regardless of class_id.

    nms_topk : int
        Keep maximum top k detections before nms, -1 for no limit.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    def calculate_overlap(out_tensor, box_a_idx, box_b_idx):
        """Calculate overlap of two boxes.
        """
        w = tvm.make.Max(0.0, tvm.make.Min(out_tensor[box_a_idx + 2], out_tensor[box_b_idx + 2])
                         - tvm.make.Max(out_tensor[box_a_idx], out_tensor[box_b_idx]))
        h = tvm.make.Max(0.0, tvm.make.Min(out_tensor[box_a_idx + 3], out_tensor[box_b_idx + 3])
                         - tvm.make.Max(out_tensor[box_a_idx + 1], out_tensor[box_b_idx + 1]))
        i = w * h
        u = (out_tensor[box_a_idx + 2] - out_tensor[box_a_idx]) * \
            (out_tensor[box_a_idx + 3] - out_tensor[box_a_idx + 1]) + \
            (out_tensor[box_b_idx + 2] - out_tensor[box_b_idx]) * \
            (out_tensor[box_b_idx + 3] - out_tensor[box_b_idx + 1]) - i
        return tvm.expr.Select(u <= 0.0, 0.0, i / u)

    ib = tvm.ir_builder.create()
    p_data = ib.buffer_ptr(data)
    p_sort_result = ib.buffer_ptr(sort_result)
    p_valid_count = ib.buffer_ptr(valid_count)
    p_out = ib.buffer_ptr(out)
    batch_size = out.shape[0]
    num_anchors = out.shape[1]

    nms_threshold_node = tvm.make.node("FloatImm", dtype="float32", value=nms_threshold)
    nms_topk_node = tvm.make.node("IntImm", dtype="int32", value=nms_topk)
    force_suppress_node = tvm.make.node("IntImm", dtype="int32", value=1 if force_suppress else 0)
    with ib.for_range(0, batch_size, for_type="parallel", name="n") as n:
        with ib.if_scope(tvm.all(nms_threshold_node > 0, nms_threshold_node < 1,
                                 p_valid_count[0] > 0)):
            # Reorder output
            nkeep = tvm.if_then_else(
                tvm.all(nms_topk_node > 0, nms_topk < p_valid_count[n]),
                nms_topk, p_valid_count[n])
            with ib.for_range(0, nkeep, name="l") as l:
                with ib.for_range(0, 6, name="m") as m:
                    p_out[(n * num_anchors * 6
                           + l * 6 + m)] = p_data[(n * num_anchors * 6
                                                   + p_sort_result[n * num_anchors + l] * 6 + m)]
            with ib.if_scope(tvm.all(nms_topk_node > 0, nms_topk < p_valid_count[n])):
                with ib.for_range(0, p_valid_count[n] - nkeep, name="l") as l:
                    with ib.for_range(0, 6, name="m") as m:
                        p_out[(n * num_anchors * 6
                               + (l + nkeep) * 6 + m)] = p_data[(n * num_anchors * 6
                                                                 + (l + nkeep) * 6 + m)]
            # Apply nms
            with ib.for_range(0, p_valid_count[n], name="l") as l:
                offset_l = l * 6
                with ib.if_scope(p_out[n * num_anchors * 6 + offset_l] >= 0):
                    with ib.for_range(0, p_valid_count[n], name="m") as m:
                        offset_m = m * 6
                        with ib.if_scope(tvm.all(m > l, p_out[n * num_anchors * 6
                                                              + offset_m] >= 0)):
                            with ib.if_scope(tvm.any(force_suppress_node > 0,
                                                     p_out[n * num_anchors * 6 + offset_l] ==
                                                     p_out[n * num_anchors * 6 + offset_m])):
                                # When force_suppress == True or class_id equals
                                iou = calculate_overlap(p_out, n * num_anchors * 6 + offset_l + 2,
                                                        n * num_anchors * 6 + offset_m + 2)
                                with ib.if_scope(iou >= nms_threshold):
                                    p_out[n * num_anchors * 6 + offset_m] = -1.0
        with ib.else_scope():
            with ib.for_range(0, p_valid_count[n], name="l") as l:
                with ib.for_range(0, 6, name="m") as m:
                    p_out[(n * num_anchors * 6
                           + l * 6 + m)] = p_data[n * num_anchors * 6 + l * 6 + m]
        # Set invalid entry to be -1
        with ib.for_range(0, num_anchors - p_valid_count[n], name="l") as l:
            with ib.for_range(0, 6, name="m") as m:
                p_out[n * num_anchors * 6 + (l + p_valid_count[n]) * 6 + m] = -1.0
    return ib.get()

@hybrid.script
def calculate_iou(inter_data, batch_idx, box_a_idx, box_b_idx, box_start_idx):
    a_t = inter_data[batch_idx, box_a_idx, box_start_idx + 1]
    a_b = inter_data[batch_idx, box_a_idx, box_start_idx + 3]
    a_l = inter_data[batch_idx, box_a_idx, box_start_idx]
    a_r = inter_data[batch_idx, box_a_idx, box_start_idx + 2]
    b_t = inter_data[batch_idx, box_b_idx, box_start_idx + 1]
    b_b = inter_data[batch_idx, box_b_idx, box_start_idx + 3]
    b_l = inter_data[batch_idx, box_b_idx, box_start_idx]
    b_r = inter_data[batch_idx, box_b_idx, box_start_idx + 2]
    w = max(0.0, min(a_r, b_r) - max(a_l, b_l))
    h = max(0.0, min(a_b, b_b) - max(a_t, b_t))
    i = h * w
    u = (a_r - a_l) * (a_b - a_t) + (b_r - b_l) * (b_b - b_t) - i
    return 0.0 if u <= 0 else i / u

@hybrid.script
def hybrid_nms(data, sorted_index, valid_count, output, iou_threshold, force_suppress, nms_topk):
    """Hybrid routing for non-maximum suppression.

    Parameters
    ----------
    data: Tensor or Var or numpy NDArray
        Bounding boxes with class and score. 3-D tensor with shape
        [batch_size, num_anchors, 6].

    sorted_index : Tensor or Var or numpy NDArray
        Bounding box indexes sorted by score, with shape
        [batch_size, num_anchors].

    valid_count : Tensor or Var or numpy NDArray
        1-D tensor for valid number of boxes.

    output : Tensor or Var or numpy NDArray
        NMS output tensor.

    iou_threshold : float
        Overlapping(IoU) threshold to suppress object with smaller score.

    force_suppress : boolean
        Whether to suppress all detections regardless of class_id.

    nms_topk : int
        Keep maximum top k detections before nms, -1 for no limit.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    box_data_length = data.shape[2]
    for i in parallel(batch_size):
        if iou_threshold > 0  and valid_count[i] > 0:
            # Reorder output
            nkeep = nms_topk if 0 < nms_topk < valid_count[i] else valid_count[i]
            for j in range(nkeep):
                for k in range(box_data_length):
                    output[i, j, k] = data[i, sorted_index[i, j], k]
            if 0 < nms_topk < valid_count[i]:
                for j in range(valid_count[i] - nkeep):
                    for k in range(box_data_length):
                        output[i, j + nkeep, k] = data[i, j + nkeep, k]
            # Apply nms
            for j in range(valid_count[i]):
                if output[i, j, 0] >= 0:
                    for k in range(valid_count[i]):
                        if k > j and output[i, k, 0] >= 0 and (force_suppress
                                                               or output[i, j, 0]
                                                               == output[i, k, 0]):
                            #iou = calculate_iou(output, i, j, k, 2)
                            inter_data = output
                            batch_idx = i
                            box_a_idx, box_b_idx = j, k
                            box_start_idx = 2
                            a_t = inter_data[batch_idx, box_a_idx, box_start_idx + 1]
                            a_b = inter_data[batch_idx, box_a_idx, box_start_idx + 3]
                            a_l = inter_data[batch_idx, box_a_idx, box_start_idx]
                            a_r = inter_data[batch_idx, box_a_idx, box_start_idx + 2]
                            b_t = inter_data[batch_idx, box_b_idx, box_start_idx + 1]
                            b_b = inter_data[batch_idx, box_b_idx, box_start_idx + 3]
                            b_l = inter_data[batch_idx, box_b_idx, box_start_idx]
                            b_r = inter_data[batch_idx, box_b_idx, box_start_idx + 2]
                            w = max(0.0, min(a_r, b_r) - max(a_l, b_l))
                            h = max(0.0, min(a_b, b_b) - max(a_t, b_t))
                            i = h * w
                            u = (a_r - a_l) * (a_b - a_t) + (b_r - b_l) * (b_b - b_t) - i
                            iou = 0.0 if u <= 0 else i / u
                            if iou >= iou_threshold:
                                output[i, k, 0] = -1.0
        else:
            for j in range(valid_count[i]):
                for k in range(box_data_length):
                    output[i, j, k] = data[i, j, k]
        # Set invalid entry to be -1
        for j in range(num_anchors - valid_count[i]):
            for k in range(box_data_length):
                output[i, j + valid_count[i], k] = -1.0


@tvm.target.generic_func
def nms(data, valid_count, nms_threshold=0.5, force_suppress=False, nms_topk=-1,
        do_rearrange=False):
    """Non-maximum suppression operator for object detection.

    Parameters
    ----------
    data : tvm.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6].
        The last dimension should be in format of
        [class_id, score, box_left, box_top, box_right, box_bottom].

    valid_count : tvm.Tensor
        1-D tensor for valid number of boxes.

    nms_threshold : optional, float
        Non-maximum suppression threshold.

    force_suppress : optional, boolean
        Whether to suppress all detections regardless of class_id.

    nms_topk : optional, int
        Keep maximum top k detections before nms, -1 for no limit.

    do_rearrange : optional, boolean
        Whether to move all valid bounding boxes to the top.

    Returns
    -------
    out : tvm.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6].

    Example
    --------
    .. code-block:: python

        # An example to use nms
        dshape = (1, 5, 6)
        data = tvm.placeholder(dshape, name="data")
        valid_count = tvm.placeholder((dshape[0],), dtype="int32", name="valid_count")
        nms_threshold = 0.7
        force_suppress = True
        nms_topk = -1
        out = nms(data, valid_count, nms_threshold, force_suppress, nms_topk)
        np_data = np.random.uniform(size=dshape).astype("float32")
        np_valid_count = np.array([4]).astype("int32")
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
    valid_count_dtype = "int32"
    valid_count_buf = api.decl_buffer(valid_count.shape, valid_count_dtype,
                                      "valid_count_buf", data_alignment=4)
    data_buf = api.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    score_axis = 1
    score_shape = (batch_size, num_anchors)
    score_tensor = tvm.compute(score_shape, lambda i, j: data[i, j, score_axis])
    score_tensor_buf = api.decl_buffer(score_tensor.shape, data.dtype,
                                       "score_tensor_buf", data_alignment=8)
    sort_tensor_dtype = "int32"
    sort_tensor_buf = api.decl_buffer(score_shape, sort_tensor_dtype,
                                      "sort_tensor_buf", data_alignment=8)
    sort_tensor = \
        tvm.extern(score_shape,
                   [score_tensor, valid_count],
                   lambda ins, outs: tvm.call_packed(
                       "tvm.contrib.sort.argsort", ins[0], ins[1],
                       outs[0], score_axis, True),
                   dtype=sort_tensor_dtype,
                   in_buffers=[score_tensor_buf, valid_count_buf],
                   out_buffers=sort_tensor_buf,
                   name="nms_sort")
    d_plc = tvm.placeholder(data.shape, name="nms_out", dtype="float32")
    out = tvm.placeholder(data.shape, name="nms_out", dtype="float32")
    inter_func = hybrid.parse(hybrid_nms, [data, sort_tensor, valid_count, out, nms_threshold, force_suppress, nms_topk])
    print(inter_func)
    out = \
        tvm.extern(data.shape,
                   [data, sort_tensor, valid_count],
                   lambda ins, outs: hybrid.parse(
                       hybrid_nms,
                       [ins[0], ins[1], ins[2], outs[0], nms_threshold,
                        force_suppress, nms_topk]),
                   dtype="float32",
                   in_buffers=[data_buf, sort_tensor_buf, valid_count_buf],
                   tag="nms")
    if do_rearrange:
        out = tvm.extern(out.shape, [out],
                         lambda ins, outs: hybrid.parse(
                             rearrange_out, [ins[0], outs[0]]),
                         dtype="float32",)
    return out
