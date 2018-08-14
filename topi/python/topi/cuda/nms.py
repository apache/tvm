# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments, too-many-statements, singleton-comparison
"""Non-maximum suppression operator"""
import math
import tvm

from tvm import api
from topi.vision import nms


def sort_pre_ir(index, sizes_out, axis_mul_before, axis_mul_after):
    """Low level IR routing subfunction 1/4 for computing segments' staring locatons.

    Parameters
    ----------
    index : Buffer
        Buffer of number of valid output boxes.

    sizes_out : Buffer
        Output buffer of start locations of each sorting segment.

    axis_mul_before : int
        The multiplication result of axis dimensions before axis.

    axis_mul_after : int
        The multiplication result of axis dimensions after axis.

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
    p_index = ib.buffer_ptr(index)
    dshape = sizes_out.shape
    sizes = ib.buffer_ptr(sizes_out)
    nthread_tx = max_threads
    nthread_bx = dshape[0] // max_threads + 1
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    tid = bx * max_threads + tx

    with ib.if_scope(tid < axis_mul_before * axis_mul_after):
        sizes[tid] = p_index[tid]

    # scan
    with ib.if_scope(tid < 1):
        with ib.for_range(0, axis_mul_before * axis_mul_after - 1, name="k") as k:
            sizes[k + 1] += sizes[k]
    body = ib.get()
    return body


def sort_pre_ir_data(data, index, sizes_in, data_out, index_out, \
                     axis, axis_mul_before, axis_mul_after):
    """Low level IR routing subfunction 2/4 for flattening data and indices into segmented format.

    Parameters
    ----------
    data: Buffer
        Buffer of output boxes with class and score.

    index : Buffer
        Buffer of number of valid output boxes.

    sizes_in : Buffer
        Buffer of start locations of each sorting segment.

    data_out : Buffer
        Buffer of flattened segmented data.

    index_out : Buffer
        Buffer of flattened segmented indices.

    axis : int
        The axis used for sorting.

    axis_mul_before : int
        The multiplication result of axis dimensions before axis.

    axis_mul_after : int
        The multiplication result of axis dimensions after axis.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    ib = tvm.ir_builder.create()
    sizes = ib.buffer_ptr(sizes_in)
    p_index = ib.buffer_ptr(index)
    p_data = ib.buffer_ptr(data)
    data_new = ib.buffer_ptr(data_out)
    index_new = ib.buffer_ptr(index_out)
    max_threads = int(
        tvm.target.current_target(allow_none=False).max_num_threads)
    tx = tvm.thread_axis("threadIdx.x")
    bx = tvm.thread_axis("blockIdx.x")
    dshape = tvm.max(sizes_in.shape[0], p_index[0])
    nthread_tx = max_threads
    nthread_bx = dshape // max_threads + 1
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    tid = bx * max_threads + tx
    with ib.if_scope(axis_mul_before * axis_mul_after > 1):
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
    with ib.else_scope():
        with ib.if_scope(tid == 0):
            with ib.for_range(0, p_index[0], name="k") as k:
                index_new[k] = k

    body = ib.get()
    return body

def sort_oet_ir(data, index, new_data, new_index, loc, out_index, axis_mul_before, \
                axis_mul_after, axis, is_descend):
    """Low level IR routing subfunction 3/4 for Odd-Even-Transposition sorting.

    Parameters
    ----------
    data: Buffer
        Buffer of output boxes with class and score.

    index : Buffer
        Buffer of number of valid output boxes.

    new_data : Buffer
        Buffer of flattened segmented data.

    new_index : Buffer
        Buffer of flattened segmented indices.

    loc : Buffer
        Buffer of start locations of each sorting segment.

    out_index : Buffer
        Output buffer of output box indexes sorted by score in a flattened segmented format.

    axis_mul_before : int
        The multiplication result of axis dimensions before axis.

    axis_mul_after : int
        The multiplication result of axis dimensions after axis.

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
    dshape = loc.shape
    fshape = data.shape[axis] * dshape[0]
    temp_data = ib.allocate(
        "float32", dshape, name="temp_data", scope="local")
    p_data = ib.buffer_ptr(data)
    p_index = ib.buffer_ptr(index)
    data_new = ib.buffer_ptr(new_data)
    index_new = ib.buffer_ptr(new_index)
    index_out = ib.buffer_ptr(out_index)
    sizes = ib.buffer_ptr(loc)
    nthread_tx = max_threads
    nthread_bx = fshape // max_threads + 1
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    tid = bx * max_threads + tx

    with ib.if_scope(axis_mul_before * axis_mul_after > 1):
        with ib.if_scope(tid < axis_mul_before * axis_mul_after):
            with ib.if_scope(tid == 0):
                start = 0
            with ib.else_scope():
                start = sizes[tid-1]
            # OddEvenTransposeSort
            with ib.for_range(0, p_index[tid], name="k") as k:
                with ib.for_range(0, p_index[tid] - 1, name="i") as i:
                    with ib.if_scope(i % 2 == k % 2):
                        with ib.if_scope(((data_new[i+start] < data_new[i+start+1]) == is_descend)):
                            temp_data[tid] = data_new[i+start]
                            data_new[i+start] = data_new[i+start+1]
                            data_new[i+start+1] = temp_data[tid]
                            index_out[tid] = index_new[i+start]
                            index_new[i+start] = index_new[i+start+1]
                            index_new[i+start+1] = index_out[tid]
        with ib.if_scope(tid < 1):
            with ib.for_range(0, sizes[dshape[0] - 1], name="i") as i:
                index_out[i] = index_new[i]
    with ib.else_scope():
        with ib.for_range(0, fshape, name="k", for_type="unroll") as k:
            with ib.if_scope(tvm.all(k % 2 == tid % 2, tid < fshape)):
                with ib.if_scope(k % 2 == 0):
                    with ib.if_scope(tvm.all(tid + 1 < fshape, (p_data[tid] < p_data[tid+1]) \
                                             == is_descend)):
                        data_new[tid] = p_data[tid+1]
                        index_out[tid] = index_new[tid+1]
                    with ib.else_scope():
                        data_new[tid] = p_data[tid]
                        index_out[tid] = index_new[tid]
                with ib.else_scope():
                    with ib.if_scope(tvm.all(tid + 1 < fshape, (data_new[tid] < data_new[tid+1]) \
                                             == is_descend)):
                        p_data[tid] = data_new[tid+1]
                        index_new[tid] = index_out[tid+1]
                    with ib.else_scope():
                        p_data[tid] = data_new[tid]
                        index_new[tid] = index_out[tid]
            with ib.if_scope(tvm.all(k % 2 != tid % 2, tid < fshape)):
                with ib.if_scope(k % 2 == 0):
                    with ib.if_scope(tvm.all(tid > 0, (p_data[tid-1] < p_data[tid]) == is_descend)):
                        data_new[tid] = p_data[tid-1]
                        index_out[tid] = index_new[tid-1]
                    with ib.else_scope():
                        data_new[tid] = p_data[tid]
                        index_out[tid] = index_new[tid]
                with ib.else_scope():
                    with ib.if_scope(tvm.all(tid > 0, (data_new[tid-1] < data_new[tid]) \
                                             == is_descend)):
                        p_data[tid] = data_new[tid-1]
                        index_new[tid] = index_out[tid-1]
                    with ib.else_scope():
                        p_data[tid] = data_new[tid]
                        index_new[tid] = index_out[tid]
        with ib.if_scope(fshape % 2 == 1):
            with ib.if_scope(tid < 1):
                with ib.for_range(0, fshape, name="k") as k:
                    index_out[tid] = index_new[tid]
    body = ib.get()
    return body


def sort_ir_out(data, index, new_index, loc, output, axis_mul_before, axis_mul_after, axis):
    """Low level IR routing subfunction 4/4 for writing sorted indices to output format.

    Parameters
    ----------
    data: Buffer
        Buffer of output boxes with class and score.

    index : Buffer
        Buffer of number of valid output boxes.

    new_index : Buffer
        Buffer of sorted indices in a flatten format.

    loc : Buffer
        Buffer of start locations of each sorting segment.

    output : Buffer
        Output buffer of output box indexes sorted by score.

    axis_mul_before : int
        The multiplication result of axis dimensions before axis.

    axis_mul_after : int
        The multiplication result of axis dimensions after axis.

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
    dshape = tvm.max(loc.shape[0], data.shape[axis])
    p_index = ib.buffer_ptr(index)
    index_new = ib.buffer_ptr(new_index)
    sizes = ib.buffer_ptr(loc)
    p_out = ib.buffer_ptr(output)
    nthread_tx = max_threads
    nthread_bx = dshape // max_threads + 1
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    tid = bx * max_threads + tx

    with ib.if_scope(axis_mul_before * axis_mul_after > 1):
        with ib.if_scope(tid < axis_mul_before * axis_mul_after):
            i = tid / axis_mul_after
            j = tid % axis_mul_after
            base_idx = i * data.shape[axis] * axis_mul_after + j
            with ib.for_range(0, data.shape[axis], name="k") as k:
                with ib.if_scope(tid == 0):
                    start = 0
                with ib.else_scope():
                    start = sizes[tid-1]
                p_out[base_idx + k * axis_mul_after] = tvm.select(
                    k < p_index[tid], index_new[k+start], k)
    with ib.else_scope():
        with ib.if_scope(tid < data.shape[axis]):
            p_out[tid] = tvm.select(tid < p_index[0], index_new[tid], tid)

    body = ib.get()
    return body


def sort_gpu(data, data_buf, index, index_buf, output_buf, axis, is_descend):
    """Function to generate low level IR to do sorting on the GPU, use it by calling sort_gpu.

    Parameters
    ----------
    data: tvm.Tensor
        3-D tensor with shape [batch_size, num_anchors, 6].
        The last dimension should be in format of
        [class_id, score, box_left, box_top, box_right, box_bottom].

    data_buf: Buffer
        2D Buffer of input boxes' score with shape [batch_size, num_anchors].

    index : tvm.Tensor
        1-D tensor for valid number of boxes.

    index_buf : Buffer
        Buffer of number of valid number of boxes.

    output_buf : Buffer
        Output buffer of indicies of sorted tensor.

    axis : int
        The axis used for sorting.

    is_descend : bool
        If the sorted data is in descending order.

    Returns
    -------
    out : tvm.Tensor
        3-D tensor with shape [batch_size, num_anchors].
    """

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

    dshape = axis_mul_before*axis_mul_after
    fshape = data.shape[axis] * dshape

    loc_buf = api.decl_buffer(dshape, index.dtype, "sizes", data_alignment=8)
    new_index_buf = api.decl_buffer(
        fshape, index.dtype, "index_new", data_alignment=8)
    out_index_buf = api.decl_buffer(
        fshape, index.dtype, "index_out", data_alignment=8)
    new_data_buf = api.decl_buffer(
        dshape, data.dtype, "data_new", data_alignment=8)

    loc = \
        tvm.extern([(dshape,)],
                   [index],
                   lambda ins, outs: sort_pre_ir(
                       ins[0], outs[0], axis_mul_before, axis_mul_after),
                   dtype=[index.dtype],
                   in_buffers=index_buf,
                   out_buffers=[loc_buf],
                   tag="sorting_prepare")

    data_new, index_new = \
        tvm.extern([(dshape,), (fshape,)],
                   [data, index, loc],
                   lambda ins, outs: sort_pre_ir_data(
                       ins[0], ins[1], ins[2], outs[0], outs[1], axis,
                       axis_mul_before, axis_mul_after),
                   dtype=[data.dtype, index.dtype],
                   in_buffers=[data_buf, index_buf, loc_buf],
                   out_buffers=[new_data_buf, new_index_buf],
                   tag="sorting_data")

    index_out = \
        tvm.extern([(fshape,)],
                   [data, index, data_new, index_new, loc],
                   lambda ins, outs: sort_oet_ir(
                       ins[0], ins[1], ins[2], ins[3], ins[4], outs[0],
                       axis_mul_before, axis_mul_after, axis, is_descend),
                   dtype=[index.dtype],
                   in_buffers=[data_buf, index_buf,
                               new_data_buf, new_index_buf, loc_buf],
                   out_buffers=[out_index_buf],
                   tag="sorting_oet")
    out = \
        tvm.extern([data.shape],
                   [data, index, index_out, loc],
                   lambda ins, outs: sort_ir_out(
                       ins[0], ins[1], ins[2], ins[3], outs[0],
                       axis_mul_before, axis_mul_after, axis),
                   dtype=[index.dtype],
                   in_buffers=[data_buf, index_buf, out_index_buf, loc_buf],
                   out_buffers=output_buf,
                   tag="sorting_output")
    return out


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

    sort_tensor = sort_gpu(score_tensor, score_tensor_buf, valid_count,
                           valid_count_buf, sort_tensor_buf, score_axis, True)
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
