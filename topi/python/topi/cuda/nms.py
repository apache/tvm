# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments, too-many-statements, singleton-comparison
"""Non-maximum suppression operator"""
import math
import tvm

from tvm import api
from topi.vision import nms


def sort_ir(data, index, output, axis, is_descend):
    """Low level IR to do sorting on the GPU, same usage as tvm.contrib.sort.argsort on the CPU.

    Parameters
    ----------
    data: Buffer
        2D Buffer of input boxes' score with shape [batch_size, num_anchors].

    index : Buffer
        Buffer of number of valid number of boxes.

    output : Buffer
        Output buffer of indicies of sorted tensor.

    axis : int
        The axis used for sorting.

    is_descend : bool
        If the sorted data is in descending order.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """

    max_threads = int(
        tvm.target.current_target(allow_none=False).max_num_threads)
    tx = tvm.thread_axis("threadIdx.x")
    bx = tvm.thread_axis("blockIdx.x")
    ib = tvm.ir_builder.create()
    p_data = ib.buffer_ptr(data)
    p_index = ib.buffer_ptr(index)
    p_out = ib.buffer_ptr(output)
    ndim = len(data.shape)
    assert data.dtype == "float32", "Currently only supports input dtype to be float32"
    assert axis < ndim, "Axis out of boundary for input ndim %d" % ndim

    axis_mul_before = 1
    axis_mul_after = 1
    if axis < 0:
        axis = ndim + axis
    for i in range(0, ndim):
        if i < axis:
            axis_mul_before *= data.shape[i]
        elif i > axis:
            axis_mul_after *= data.shape[i]

    dshape = 0
    for i in range(0, len(index.shape)):
        dshape += index.shape[i]
    dshape = tvm.select(dshape > axis_mul_before*axis_mul_after, dshape,
                        axis_mul_before*axis_mul_after)

    sizes_temp = ib.allocate(
        "int32", dshape, name="sizes_temp", scope="global")
    sizes = ib.allocate("int32", dshape, name="sizes", scope="global")
    temp_index = ib.allocate("int32", dshape, name="temp_index", scope="local")
    temp_data = ib.allocate("float32", dshape, name="temp_data", scope="local")
    data_new = ib.allocate("float32", dshape, name="data_new", scope="global")
    index_new = ib.allocate("int32", dshape, name="index_new", scope="global")
    nthread_tx = max_threads
    nthread_bx = dshape // max_threads + 1
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    tid = bx * max_threads + tx

    with ib.if_scope(tid < axis_mul_before * axis_mul_after):
        sizes[tid] = p_index[tid]
        sizes_temp[tid] = p_index[tid]

    with ib.if_scope(tid < axis_mul_before * axis_mul_after):
        with ib.for_range(0, tvm.floor(tvm.sqrt((axis_mul_before * axis_mul_after) \
             .astype("float32"))) + 1, name="k") as k:
            with ib.if_scope(tid - (tvm.const(1, "int32") << k) >= 0):
                with ib.if_scope(k % 2 == 0):
                    sizes[tid] += sizes_temp[tid - (
                        tvm.const(1, "int32") << k)]
                    sizes_temp[tid] = sizes[tid]
                with ib.else_scope():
                    sizes_temp[tid] += sizes[tid - (
                        tvm.const(1, "int32") << k)]
                    sizes[tid] = sizes_temp[tid]

    with ib.if_scope(tid < axis_mul_before * axis_mul_after):
        i = tid / axis_mul_after
        j = tid % axis_mul_after
        current_sort_num = p_index[tid]
        base_idx = i * data.shape[axis] * axis_mul_after + j
        with ib.for_range(0, current_sort_num, name="k") as k:
            full_idx = base_idx + k * axis_mul_after
            with ib.if_scope(tid == 0):
                start = 0
            with ib.else_scope():
                start = sizes[tid-1]
            index_new[start + k] = k
            data_new[start + k] = p_data[full_idx]

    with ib.if_scope(tid < axis_mul_before * axis_mul_after):
        with ib.if_scope(tid == 0):
            start = 0
        with ib.else_scope():
            start = sizes[tid-1]
        # OddEvenTransposeSort
        with ib.for_range(0, p_index[tid], name="k") as k:
            with ib.for_range(0, p_index[tid] - 1, name="i") as i:
                with ib.if_scope(i % 2 == (k & 1)):
                    with ib.if_scope(((data_new[i+start] < data_new[i+start+1]) ^
                                      is_descend) == False):
                        temp_data[tid] = data_new[i+start]
                        data_new[i+start] = data_new[i+start+1]
                        data_new[i+start+1] = temp_data[tid]
                        temp_index[tid] = index_new[i+start]
                        index_new[i+start] = index_new[i+start+1]
                        index_new[i+start+1] = temp_index[tid]

    with ib.if_scope(tid < axis_mul_before * axis_mul_after):
        i = tid / axis_mul_after
        j = tid % axis_mul_after
        current_sort_num = p_index[tid]
        base_idx = i * data.shape[axis] * axis_mul_after + j
        with ib.for_range(0, data.shape[axis], name="k") as k:
            with ib.if_scope(tid == 0):
                start = 0
            with ib.else_scope():
                start = sizes[tid-1]
            p_out[base_idx + k * axis_mul_after] = tvm.select(
                k < current_sort_num,
                index_new[k+start], k)
    body = ib.get()
    return body


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
        return tvm.select(u <= 0.0, 0.0, i / u)

    max_threads = int(math.sqrt(
        tvm.target.current_target(allow_none=False).max_num_threads))
    tx = tvm.thread_axis("threadIdx.x")
    ty = tvm.thread_axis("threadIdx.y")
    bx = tvm.thread_axis("blockIdx.x")
    by = tvm.thread_axis("blockIdx.y")
    ib = tvm.ir_builder.create()
    p_data = ib.buffer_ptr(data)
    p_sort_result = ib.buffer_ptr(sort_result)
    p_valid_count = ib.buffer_ptr(valid_count)
    p_out = ib.buffer_ptr(out)
    batch_size = out.shape[0]
    num_anchors = out.shape[1]
    nthread_tx = max_threads
    nthread_bx = num_anchors // max_threads + 1
    nthread_ty = max_threads
    nthread_by = 6 // max_threads + 1
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(ty, "thread_extent", nthread_ty)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    ib.scope_attr(by, "thread_extent", nthread_by)
    i = bx * max_threads + tx
    j = by * max_threads + ty

    nms_threshold_node = tvm.make.node(
        "FloatImm", dtype="float32", value=nms_threshold)
    nms_topk_node = tvm.make.node("IntImm", dtype="int32", value=nms_topk)
    force_suppress_node = tvm.make.node(
        "IntImm", dtype="int32", value=1 if force_suppress else 0)
    with ib.for_range(0, batch_size, for_type="unroll", name="n") as n:
        with ib.if_scope(
            tvm.all(nms_threshold_node > 0, nms_threshold_node < 1,
                    p_valid_count[0] > 0)):
            # Reorder output
            nkeep = tvm.select(
                tvm.all(nms_topk_node > 0, nms_topk < p_valid_count[n]),
                nms_topk, p_valid_count[n])
            with ib.if_scope(i < nkeep):
                with ib.if_scope(j < 6):
                    p_out[(n * num_anchors * 6
                           + i * 6 + j)] = p_data[(n * num_anchors * 6
                                                   + p_sort_result[n * num_anchors + i] * 6 + j)]
            with ib.if_scope(tvm.all(nms_topk_node > 0, nms_topk < p_valid_count[n])):
                with ib.if_scope(i < p_valid_count[n] - nkeep):
                    with ib.if_scope(j < 6):
                        p_out[(n * num_anchors * 6
                               + (i + nkeep) * 6 + j)] = p_data[(n * num_anchors * 6
                                                                 + (i + nkeep) * 6 + j)]
            # Apply nms
            with ib.if_scope(i < p_valid_count[n]):
                offset_i = i * 6
                with ib.if_scope(p_out[n * num_anchors * 6 + offset_i] >= 0):
                    with ib.if_scope(j < p_valid_count[n]):
                        offset_j = j * 6
                        with ib.if_scope(tvm.all(j > i, p_out[n * num_anchors * 6
                                                              + offset_j] >= 0)):
                            with ib.if_scope(tvm.any(force_suppress_node > 0,
                                                     p_out[n * num_anchors * 6 + offset_i] ==
                                                     p_out[n * num_anchors * 6 + offset_j])):
                                # When force_suppress == True or class_id equals
                                iou = calculate_overlap(
                                    p_out, n * num_anchors * 6 + offset_i + 2,
                                    n * num_anchors * 6 + offset_j + 2)
                                with ib.if_scope(iou >= nms_threshold):
                                    p_out[
                                        n * num_anchors * 6 + offset_j] = -1.0
        with ib.else_scope():
            with ib.if_scope(i < p_valid_count[n]):
                with ib.if_scope(j < 6):
                    p_out[(n * num_anchors * 6
                           + i * 6 + j)] = p_data[n * num_anchors * 6 + i * 6 + j]
        # Set invalid entry to be -1
        with ib.if_scope(i < num_anchors - p_valid_count[n]):
            with ib.if_scope(j < 6):
                p_out[n * num_anchors * 6 + (i +
                                             p_valid_count[n]) * 6 + j] = -1.0
    body = ib.get()
    return body


@nms.register(["cuda", "gpu"])
def nms_gpu(data, valid_count, nms_threshold=0.5, force_suppress=False, nms_topk=-1):
    """Non-maximum suppression operator for object detection.

    Parameters
    ----------
    data: tvm.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6].
        The last dimension should be in format of
        [class_id, score, box_left, box_top, box_right, box_bottom].

    valid_count : tvm.Tensor
        1-D tensor for valid number of boxes.

    nms_threshold : float
        Non-maximum suppression threshold.

    force_suppress : boolean
        Whether to suppress all detections regardless of class_id.

    nms_topk : int
        Keep maximum top k detections before nms, -1 for no limit.

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
        valid_count = tvm.placeholder(
            (dshape[0],), dtype="int32", name="valid_count")
        nms_threshold = 0.7
        force_suppress = True
        nms_topk = -1
        out = nms(data, valid_count, nms_threshold, force_suppress, nms_topk)
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
    data_buf = api.decl_buffer(
        data.shape, data.dtype, "data_buf", data_alignment=8)
    score_axis = 1
    score_shape = (batch_size, num_anchors)
    score_tensor = tvm.compute(
        score_shape, lambda i, j: data[i, j, score_axis], name="score_tensor")
    score_tensor_buf = api.decl_buffer(score_tensor.shape, data.dtype,
                                       "score_tensor_buf", data_alignment=8)
    sort_tensor_dtype = "int32"
    sort_tensor_buf = api.decl_buffer(score_shape, sort_tensor_dtype,
                                      "sort_tensor_buf", data_alignment=8)

    sort_tensor = \
        tvm.extern(score_shape,
                   [score_tensor, valid_count],
                   lambda ins, outs: sort_ir(
                       ins[0], ins[1], outs[0], score_axis, True),
                   dtype=sort_tensor_dtype,
                   in_buffers=[score_tensor_buf, valid_count_buf],
                   out_buffers=sort_tensor_buf,
                   name="nms_sort")
    out = \
        tvm.extern(data.shape,
                   [data, sort_tensor, valid_count],
                   lambda ins, outs: nms_ir(
                       ins[0], ins[1], ins[2], outs[0], nms_threshold,
                       force_suppress, nms_topk),
                   dtype="float32",
                   in_buffers=[data_buf, sort_tensor_buf, valid_count_buf],
                   tag="nms")
    return out
