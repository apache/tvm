# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments
"""Non-maximum suppression operator"""
import tvm

from tvm import api, hybrid

@hybrid.script
def rearrange_out(input):
    """Rearrange nms output to move all valid entries to top.

    Parameters
    ----------
    input : tvm.Tensor or numpy NDArray
        NMS output. 3-D tensor with shape
        [batch_size, num_anchors, 6].

    Returns
    -------
    output : tvm.Tensor or numpy NDArray
        Transformed NMS output. 3-D tensor with shape
        [batch_size, num_anchors, 6].
    """
    output = output_tensor((input.shape[0],
                            input.shape[1],
                            input.shape[2],),
                           input.dtype)
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
    return output


@hybrid.script
def get_valid_counts(data, score_threshold):
    """Get valid count of bounding boxes given a score threshlod.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : tvm.Tensor or numpy NDArray
        Input data. 3-D tensor with shape [batch_size, num_anchors, 6].

    score_threshold : tvm.const
        Lower limit of score for valid bounding boxes.

    Returns
    -------
    out_tensor : tvm.Tensor or numpy NDArray
        Rearranged data tensor.

    valid_count : tvm.Tensor or numpy NDArray
        1-D tensor for valid number of boxes.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    box_data_length = data.shape[2]
    valid_count = output_tensor((batch_size,), "int32")
    out_tensor = output_tensor((batch_size,
                                num_anchors,
                                box_data_length),
                               data.dtype)
    for i in range(batch_size):
        valid_count[i] = 0
        inter_idx = 0
        for j in range(num_anchors):
            score = data[i, j, 1]
            if score >= score_threshold:
                for k in range(box_data_length):
                    out_tensor[i, inter_idx, k] = data[i, j, k]
                valid_count[i] += 1
                inter_idx = inter_idx + 1

    return valid_count, out_tensor


@hybrid.script
def hybrid_nms(data, sorted_index, valid_count,
               iou_threshold, force_suppress, topk):
    """Hybrid routing for non-maximum suppression.

    Parameters
    ----------
    data: tvm.Tensor or numpy NDArray
        Bounding boxes with class and score. 3-D tensor with shape
        [batch_size, num_anchors, 6].

    sorted_index : tvm.Tensor or numpy NDArray
        Bounding box indexes sorted by score, with shape
        [batch_size, num_anchors].

    valid_count : tvm.Tensor or numpy NDArray
        1-D tensor for valid number of boxes.

    iou_threshold : tvm.const
        Overlapping(IoU) threshold to suppress object with smaller score.

    force_suppress : tvm.const
        Whether to suppress all detections regardless of class_id.

    topk : tvm.const
        Keep maximum top k detections before nms, -1 for no limit.

    Returns
    -------
    valid_count : tvm.Tensor or numpy NDArray
        1-D tensor for valid number of boxes.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    box_data_length = data.shape[2]
    output = output_tensor((batch_size,
                            num_anchors,
                            box_data_length,),
                           data.dtype)
    for i in parallel(batch_size):
        if iou_threshold > 0:
            if valid_count[i] > 0:
                # Reorder output
                nkeep = valid_count[i]
                if topk > 0:
                    if topk < valid_count[i]:
                        nkeep = topk
                for j in range(nkeep):
                    for k in range(box_data_length):
                        output[i, j, k] = data[i, sorted_index[i, j], k]
                if topk > 0:
                    if topk < valid_count[i]:
                        for j in range(valid_count[i] - nkeep):
                            for k in range(box_data_length):
                                output[i, j + nkeep, k] = data[i, j + nkeep, k]
            # Apply nms
            for j in range(valid_count[i]):
                if output[i, j, 0] >= 0:
                    for k in range(valid_count[i]):
                        check_iou = 0
                        if k > j:
                            if output[i, k, 0] >= 0:
                                if force_suppress:
                                    check_iou = 1
                                elif output[i, j, 0] == output[i, k, 0]:
                                    check_iou = 1
                        if check_iou:
                            batch_idx = i
                            box_a_idx = j
                            box_b_idx = k
                            box_start_idx = 2
                            a_t = output[batch_idx, box_a_idx, box_start_idx + 1]
                            a_b = output[batch_idx, box_a_idx, box_start_idx + 3]
                            a_l = output[batch_idx, box_a_idx, box_start_idx]
                            a_r = output[batch_idx, box_a_idx, box_start_idx + 2]
                            b_t = output[batch_idx, box_b_idx, box_start_idx + 1]
                            b_b = output[batch_idx, box_b_idx, box_start_idx + 3]
                            b_l = output[batch_idx, box_b_idx, box_start_idx]
                            b_r = output[batch_idx, box_b_idx, box_start_idx + 2]
                            w = max(0.0, min(a_r, b_r) - max(a_l, b_l))
                            h = max(0.0, min(a_b, b_b) - max(a_t, b_t))
                            area = h * w
                            u = (a_r - a_l) * (a_b - a_t) + (b_r - b_l) * (b_b - b_t) - area
                            iou = 0.0 if u <= 0.0 else area / u
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
    return output


@tvm.target.generic_func
def nms(data, valid_count, iou_threshold=0.5, force_suppress=False,
        topk=-1, do_rearrange=False):
    """Non-maximum suppression operator for object detection.

    Parameters
    ----------
    data : tvm.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6].
        The last dimension should be in format of
        [class_id, score, box_left, box_top, box_right, box_bottom].

    valid_count : tvm.Tensor
        1-D tensor for valid number of boxes.

    iou_threshold : optional, float
        Non-maximum suppression threshold.

    force_suppress : optional, boolean
        Whether to suppress all detections regardless of class_id.

    topk : optional, int
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
        iou_threshold = 0.7
        force_suppress = True
        topk = -1
        out = nms(data, valid_count, iou_threshold, force_suppress, topk)
        np_data = np.random.uniform(dshape)
        np_valid_count = np.array([4])
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
    out = hybrid_nms(data, sort_tensor, valid_count,
                     tvm.const(iou_threshold, dtype="float32"),
                     tvm.const(force_suppress, dtype="bool"),
                     tvm.const(topk, dtype="int32"))
    if do_rearrange:
        out = rearrange_out(out)

    return out

@tvm.target.generic_func
def box_nms(data, iou_threshold=0.5, score_threshold=0,
            force_suppress=True, topk=-1):
    """Apply non-maximum suppression to input.
    Comparing to nms, this function takes score_threshold
    as argument and automatically filters valid anchor boxes.

    Parameters
    ----------
    data : tvm.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6].
        The last dimension should be in format of
        [class_id, score, box_left, box_top, box_right, box_bottom].

    iou_threshold : optional, float
        Non-maximum suppression threshold.

    score_threshold : optional, float
        Lower limit of score for valid bounding boxes.

    force_suppress : optional, boolean
        Whether to suppress all detections regardless of class_id.

    topk : optional, int
        Keep maximum top k detections before nms, -1 for no limit.

    Returns
    -------
    out : tvm.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6].
    """
    score_threshold_const = tvm.const(score_threshold,
                                      dtype="float32")
    valid_count, out = get_valid_counts(data, score_threshold_const)
    return nms(out, valid_count, iou_threshold,
               force_suppress, topk, True)


if __name__ == '__main__':
    import tvm
    import topi
    import numpy as np

    score_threshold = 0.13
    overlap_thresh = 0.5

    # This works.
    # Here we first call get_valid_counts with np data,
    # then build nms function and feed data into it.
    np_data = np.random.uniform(size=(1, 5000, 6)).astype("float32")
    np_valid_count, np_inter_out = topi.vision.get_valid_counts(np_data, score_threshold)
    data = tvm.placeholder((1, 5000, 6), name="data", dtype="float32")
    valid_count = tvm.placeholder((1,), name="valid_count", dtype="int32")
    result = topi.vision.nms(data, valid_count, iou_threshold=overlap_thresh, force_suppress=True, do_rearrange=True)
    st = tvm.create_schedule(result.op)
    f = tvm.build(st, [data, valid_count, result], "llvm")
    ctx = tvm.cpu(0)
    np_out = np.zeros(np_inter_out.shape)
    aa = tvm.nd.array(np_inter_out.astype(data.dtype), ctx)
    bb = tvm.nd.array(np_valid_count.astype(valid_count.dtype), ctx)
    cc = tvm.nd.array(np_out.astype(result.dtype), ctx)
    f(aa, bb, cc)


    # This will fail
    # We combine get_valid_counts and nms into box_nms
    data = tvm.placeholder((1, 5000, 6), name="data", dtype="float32")
    result = topi.vision.box_nms(data, iou_threshold=overlap_thresh, force_suppress=True, score_threshold=score_threshold)
    st = tvm.create_schedule(result.op)
    f = tvm.build(st, [data, result], "llvm")
