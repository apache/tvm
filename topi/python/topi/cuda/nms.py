# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments
"""Non-maximum suppression operator"""
import tvm

from tvm import api
from topi.vision import nms
import math
import numpy as np

def sort_ir(data, index, output, axis, is_descend):
    def swap(a, b):
        a, b = b, a
    max_threads = int(math.sqrt(tvm.target.current_target(allow_none=False).max_num_threads))
    tx = tvm.thread_axis("threadIdx.x")
    ty = tvm.thread_axis("threadIdx.y")
    bx = tvm.thread_axis("blockIdx.x")
    by = tvm.thread_axis("blockIdx.y")
    ib = tvm.ir_builder.create()
    p_data = ib.buffer_ptr(data)
    p_index = ib.buffer_ptr(index)
    p_out = ib.buffer_ptr(output)
    dshape = p_index[index.shape[0]-1]
    data_new = ib.allocate("float32", dshape, name="data_new", scope="global")
    index_new = ib.allocate("int32", dshape, name="index_new", scope="global")
#    p_index_new = ib.buffer_ptr(index_new)
#    p_data_new = ib.buffer_ptr(data_new)
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

    nthread_tx = max_threads
    nthread_bx = axis_mul_before // max_threads + 1
    nthread_ty = max_threads
    nthread_by = axis_mul_after // max_threads + 1
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(ty, "thread_extent", nthread_ty)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    ib.scope_attr(by, "thread_extent", nthread_by)
    i = bx * max_threads + tx
    j = by * max_threads + ty
    tid = i * nthread_tx * nthread_bx + j

    with ib.if_scope(i < axis_mul_before):
        with ib.if_scope(j < axis_mul_after):
            current_sort_num = p_index[i * axis_mul_after + j]
            base_idx = i * data.shape[axis] * axis_mul_after + j
            with ib.for_range(0, current_sort_num, name = "k") as k:
                full_idx = base_idx + k * axis_mul_after
                index_new[k] = k
                data_new[k] = p_data[full_idx]
            # sync_threads
            # sorting
            size = current_sort_num
            with ib.if_scope(tid < size - 1):
                with ib.for_range(0, size - 1, name = "level") as level:
                    with ib.if_scope(tid % 2 == (level & 1)):
                        with ib.if_scope(~((data_new[tid] < data_new[tid + 1]) ^ is_descend)):
                            swap(data_new[tid], data_new[tid+1])
                            swap(index_new[tid], index_new[tid+1])
            #convert back
            with ib.for_range(0, data.shape[axis], for_type = "unroll", name = "l") as l:
                p_out[base_idx + l * axis_mul_after] = tvm.select(k < size, index_new[l], l)

    body = ib.get()
    input(body)
    return body


def OddEvenTransposeSort_ir(data, index, is_descend):
    """ Low level IR routing for sorting operation on GPUs.

    Parameters
    ----------
    ----------
    """
    def swap(a, b):
        a, b = b, a

    max_threads = int(math.sqrt(tvm.target.current_target(allow_none=False).max_num_threads))
    tx = tvm.thread_axis("threadIdx.x")
    bx = tvm.thread_axis("blockIdx.x")
    ib = tvm.ir_builder.create()
    p_data = ib.buffer_ptr(data)
    p_index = ib.buffer_ptr(index)
    in_size = data.shape
    nthread_tx = max_threads
    nthread_bx = in_size // max_threads + 1
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    tid = bx * max_threads + tx
    with ib.if_scope(tid < in_size - 1):
        with ib.for_range(0, in_size - 1, name = "level", for_type = "unroll") as level:
            with ib.if_scope(tid % 2 == (1 & level)):
                with ib.if_scope(~(is_descend ^ (p_data[tid] < p_data[tid+1]))): # xnor comp
                    # doing swap on global mem which is non-efficient
                    swap(p_data[tid], p_data[tid+1])
                    swap(p_index[tid], p_index[tid+1])
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

    max_threads = int(math.sqrt(tvm.target.current_target(allow_none=False).max_num_threads))
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

    nms_threshold_node = tvm.make.node("FloatImm", dtype="float32", value=nms_threshold)
    nms_topk_node = tvm.make.node("IntImm", dtype="int32", value=nms_topk)
    force_suppress_node = tvm.make.node("IntImm", dtype="int32", value=1 if force_suppress else 0)
    with ib.for_range(0, batch_size, for_type="unroll", name="n") as n:
        with ib.if_scope(tvm.all(nms_threshold_node > 0, nms_threshold_node < 1,
                                 p_valid_count[0] > 0)):
            # Reorder output
            nkeep = tvm.select(tvm.all(nms_topk_node > 0, nms_topk < p_valid_count[n]),
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
                                iou = calculate_overlap(p_out, n * num_anchors * 6 + offset_i + 2,
                                                        n * num_anchors * 6 + offset_j + 2)
                                with ib.if_scope(iou >= nms_threshold):
                                    p_out[n * num_anchors * 6 + offset_j] = -1.0
        with ib.else_scope():
            with ib.if_scope(i < p_valid_count[n]):
                with ib.if_scope(j < 6):
                    p_out[(n * num_anchors * 6
                           + i * 6 + j)] = p_data[n * num_anchors * 6 + i * 6 + j]
        # Set invalid entry to be -1
        with ib.if_scope(i < num_anchors - p_valid_count[n]):
            with ib.if_scope(j < 6):
                p_out[n * num_anchors * 6 + (i + p_valid_count[n]) * 6 + j] = -1.0
    body = ib.get()
    input(body)
    return body

#@sort.register("cuda")
def sort(data, index, is_descend):
    oshape = data.shape
    out = tvm.extern(oshape, [data, index], lambda ins, outs:
                     OddEvenTransposeSort_ir(ins[0], ins[1], is_descend),
                     tag="sort_gpu")
    return out

@nms.register("cuda")
def nms(data, valid_count, nms_threshold=0.5, force_suppress=False, nms_topk=-1):
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
        valid_count = tvm.placeholder((dshape[0],), dtype="int32", name="valid_count")
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
    data_buf = api.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    score_axis = 1
    score_shape = (batch_size, num_anchors)
    score_tensor = tvm.compute(score_shape, lambda i, j: data[i, j, score_axis])
    score_tensor_buf = api.decl_buffer(score_tensor.shape, data.dtype,
                                       "score_tensor_buf", data_alignment=8)
    sort_tensor_dtype = "int32"
    sort_tensor_buf = api.decl_buffer(score_shape, sort_tensor_dtype,
                                      "sort_tensor_buf", data_alignment=8)

#    dshape = (valid_count[valid_count.shape[0]-1],)
#    sorter = tvm.placeholder(dshape, name = "sorter", dtype = "float32")
#    index = tvm.placeholder(dshape, name = "index", dtype = "int32")
#    sorter_buf = api.decl_buffer(sorter.shape, sorter.dtype,
#                                 "sorter_data_buf", data_alignment=8)
#    index_buf = api.decl_buffer(index.shape, index.dtype,
#                                "sorter_index_buf", data_alignment=8)

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
