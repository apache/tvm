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
# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments, too-many-statements, singleton-comparison
# pylint: disable=bad-continuation, unused-argument
"""Non-maximum suppression operator"""
import tvm
from tvm import te

from tvm.tir import if_then_else
from .sort import argsort, argsort_thrust
from .. import tag


def cuda_atomic_add_rule(op):
    if op.dtype == "float32":
        return tvm.tir.call_pure_extern("float32", "atomicAdd", op.args[0], op.args[1])
    if op.dtype == "float64":
        return tvm.tir.call_pure_extern("float64", "atomicAdd", op.args[0], op.args[1])
    if op.dtype == "int32":
        return tvm.tir.call_pure_extern("int32", "atomicAdd", op.args[0], op.args[1])
    raise RuntimeError("only support int32, float32 and float64")


tvm.target.intrin.register_intrin_rule(
    "cuda", "atomic_add", cuda_atomic_add_rule, override=True)


def atomic_add(x, y):
    return tvm.tir.call_pure_intrin(y.dtype, "atomic_add", x, y)


def get_valid_counts_ir(data, valid_count, out, score_threshold, id_index, score_index):
    """Low level IR to get valid count of bounding boxes
    given a score threshold. Also prepares to move valid boxes to the
    top of input data.

    Parameters
    ----------
    data : Buffer
        Input data. 3-D Buffer with shape [batch_size, num_anchors, elem_length].

    valid_count : Buffer
        1D buffer for valid number of boxes with shape [batch_size, ].

    flag : Buffer
        2D Buffer of flag indicating valid data with shape [batch_size, num_anchors].

    score_threshold : float32
        Lower limit of score for valid bounding boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    elem_length = data.shape[2]

    ib = tvm.tir.ir_builder.create()

    data = ib.buffer_ptr(data)

    valid_count = ib.buffer_ptr(valid_count)
    out = ib.buffer_ptr(out)
    atomic_add_return = ib.allocate(
        valid_count.dtype, (1,), name='atomic_add_return', scope='local')
    one_count = tvm.tir.const(1, dtype=valid_count.dtype)
    one = tvm.tir.const(1, dtype=out.dtype)
    score_threshold = tvm.ir.make_node(
        "FloatImm", dtype="float32", value=score_threshold)
    id_index = tvm.ir.make_node("IntImm", dtype="int32", value=id_index)
    score_index = tvm.ir.make_node("IntImm", dtype="int32", value=score_index)

    max_threads = int(tvm.target.Target.current(
        allow_none=False).max_num_threads)
    nthread_tx = max_threads
    nthread_bx = batch_size * num_anchors // max_threads + 1
    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    tid = bx * max_threads + tx
    idxd = tvm.tir.indexdiv

    # initialize valid_count
    with ib.if_scope(tid < batch_size):
        valid_count[tid] = 0
    with ib.if_scope(tid < batch_size * num_anchors):
        i = idxd(tid, num_anchors)
        with ib.if_scope(
                tvm.tir.all(data[tid * elem_length + score_index] > score_threshold,
                            tvm.tir.any(id_index < 0, data[tid * elem_length + id_index] >= 0))):
            atomic_add_return[0] = atomic_add(tvm.tir.call_pure_intrin("handle", "tvm_address_of",
                                                                       valid_count[i]), one_count)
            with ib.for_range(0, elem_length) as k:
                out[tid * elem_length + k] = data[tid * elem_length + k]
        with ib.else_scope():
            with ib.for_range(0, elem_length) as k:
                out[tid * elem_length + k] = -one

    return ib.get()


def get_valid_counts(data, score_threshold=0, id_index=0, score_index=1):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data. 3-D tensor with shape [batch_size, num_anchors, elem_length].

    score_threshold : optional, float
        Lower limit of score for valid bounding boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    Returns
    -------
    valid_count : tvm.te.Tensor
        1-D tensor for valid number of boxes.

    out_tensor : tvm.te.Tensor
        Rearranged data tensor.
    """
    batch_size = data.shape[0]
    data_buf = tvm.tir.decl_buffer(
        data.shape, data.dtype, "data_buf", data_alignment=8)
    valid_count_buf = tvm.tir.decl_buffer(
        (batch_size,), "int32", "valid_count_buf", data_alignment=8)
    out_buf = tvm.tir.decl_buffer(
        data.shape, data.dtype, "out_buf", data_alignment=8)

    valid_count, out = \
        te.extern([(batch_size,), data.shape], [data],
                  lambda ins, outs: get_valid_counts_ir(
            ins[0], outs[0], outs[1], score_threshold, id_index, score_index),
            dtype=["int32", data.dtype],
            in_buffers=[data_buf],
            out_buffers=[valid_count_buf, out_buf],
            name="get_valid_counts",
            tag="get_valid_counts_gpu")

    return [valid_count, out]


def nms_ir(data, sorted_index, valid_count, out, box_indices,
           max_output_size, iou_threshold, force_suppress,
           top_k, coord_start, id_index, score_index):
    """Low level IR routing for transform location in multibox_detection operator.

    Parameters
    ----------
    data : Buffer
        Buffer of output boxes with class and score.

    sort_index : Buffer
        Buffer of output box indexes sorted by score.

    valid_count : Buffer
        Buffer of number of valid output boxes.

    out : Buffer
        Output buffer.

    max_output_size : int
        Max number of output valid boxes for each instance.
        By default all valid boxes are returned.

    iou_threshold : float
        Overlapping(IoU) threshold to suppress object with smaller score.

    force_suppress : boolean
        Whether to suppress all detections regardless of class_id.

    top_k : int
        Keep maximum top k detections before nms, -1 for no limit.

    coord_start : int
        Start index of the consecutive 4 coordinates.

    id_index : int
        index of the class categories, -1 to disable.

    score_index : optional, int
        Index of the scores/confidence of boxes.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    def calculate_overlap(out_tensor, box_a_idx, box_b_idx):
        """Calculate overlap of two boxes.
        """
        w = tvm.te.max(0.0, tvm.te.min(out_tensor[box_a_idx + 2], out_tensor[box_b_idx + 2])
                       - tvm.te.max(out_tensor[box_a_idx], out_tensor[box_b_idx]))
        h = tvm.te.max(0.0, tvm.te.min(out_tensor[box_a_idx + 3], out_tensor[box_b_idx + 3])
                       - tvm.te.max(out_tensor[box_a_idx + 1], out_tensor[box_b_idx + 1]))
        i = w * h
        u = (out_tensor[box_a_idx + 2] - out_tensor[box_a_idx]) * \
            (out_tensor[box_a_idx + 3] - out_tensor[box_a_idx + 1]) + \
            (out_tensor[box_b_idx + 2] - out_tensor[box_b_idx]) * \
            (out_tensor[box_b_idx + 3] - out_tensor[box_b_idx + 1]) - i
        return tvm.tir.Select(u <= 0.0, 0.0, i / u)

    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    box_data_length = data.shape[2]

    ib = tvm.tir.ir_builder.create()

    data = ib.buffer_ptr(data)
    sorted_index = ib.buffer_ptr(sorted_index)
    valid_count = ib.buffer_ptr(valid_count)
    out = ib.buffer_ptr(out)
    box_indices = ib.buffer_ptr(box_indices)
    num_valid_boxes = ib.allocate(
        "int32", (1,), name="num_valid_boxes", scope="local")

    max_threads = int(
        tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_tx = max_threads
    nthread_bx = num_anchors // max_threads + 1
    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)
    j = bx * max_threads + tx

    iou_threshold = tvm.ir.make_node(
        "FloatImm", dtype="float32", value=iou_threshold)
    top_k = tvm.ir.make_node("IntImm", dtype="int32", value=top_k)
    coord_start = tvm.ir.make_node("IntImm", dtype="int32", value=coord_start)
    id_index = tvm.ir.make_node("IntImm", dtype="int32", value=id_index)
    score_index = tvm.ir.make_node("IntImm", dtype="int32", value=score_index)
    force_suppress = tvm.ir.make_node(
        "IntImm", dtype="int32", value=1 if force_suppress else 0)

    with ib.for_range(0, batch_size, for_type="unroll") as i:
        base_idx = i * num_anchors * box_data_length
        with ib.if_scope(tvm.tir.all(iou_threshold > 0, valid_count[i] > 0)):
            # Reorder output
            nkeep = if_then_else(
                tvm.tir.all(top_k > 0, top_k < valid_count[i]),
                top_k, valid_count[i])
            with ib.if_scope(j < nkeep):
                with ib.for_range(0, box_data_length) as k:
                    out[(base_idx + j * box_data_length + k)] = \
                        data[(base_idx + sorted_index[i * num_anchors + j]
                              * box_data_length + k)]
                box_indices[i * num_anchors +
                            j] = sorted_index[i * num_anchors + j]
            with ib.if_scope(tvm.tir.all(top_k > 0, top_k < valid_count[i])):
                with ib.if_scope(j < valid_count[i] - nkeep):
                    with ib.for_range(0, box_data_length) as k:
                        out[(base_idx + (j + nkeep) * box_data_length + k)] = -1.0
                    box_indices[i * num_anchors + (j + nkeep)] = -1
            # Apply nms
            with ib.for_range(0, valid_count[i]) as k:
                offset_k = k * box_data_length
                with ib.if_scope(
                        tvm.tir.all(out[base_idx + offset_k + score_index] > 0,
                                    tvm.tir.any(id_index < 0, out[base_idx +
                                                                  offset_k + id_index] >= 0))):
                    with ib.if_scope(j < valid_count[i]):
                        offset_j = j * box_data_length
                        with ib.if_scope(
                                tvm.tir.all(j > k,
                                            out[base_idx + offset_j +
                                                score_index] > 0,
                                            tvm.tir.any(id_index < 0,
                                                        out[base_idx + offset_j + id_index] >= 0),
                                            tvm.tir.any(force_suppress > 0, id_index < 0,
                                                    out[base_idx + offset_k + id_index] ==
                                                        out[base_idx + offset_j + id_index]))):
                            iou = calculate_overlap(out, base_idx + offset_j + coord_start,
                                                    base_idx + offset_k + coord_start)
                            with ib.if_scope(iou >= iou_threshold):
                                out[base_idx + offset_j + score_index] = -1.0
                                with ib.if_scope(id_index >= 0):
                                    out[base_idx + offset_j + id_index] = -1.0
                                box_indices[i * num_anchors + j] = -1
        with ib.else_scope():
            with ib.if_scope(j < valid_count[i]):
                offset_j = j * box_data_length
                with ib.for_range(0, box_data_length) as k:
                    out[(base_idx + offset_j + k)
                        ] = data[base_idx + offset_j + k]
                box_indices[i * num_anchors + j] = j
        # Set invalid entry to be -1
        with ib.if_scope(j < num_anchors - valid_count[i]):
            with ib.for_range(0, box_data_length) as k:
                out[base_idx + (j + valid_count[i]) *
                    box_data_length + k] = -1.0
            box_indices[i * num_anchors + j + valid_count[i]] = -1
        # Only return max_output_size number of valid boxes
        num_valid_boxes[0] = 0
        with ib.if_scope(max_output_size > 0):
            with ib.if_scope(j < valid_count[i]):
                offset_j = j * box_data_length
                with ib.if_scope(out[base_idx + offset_j] >= 0):
                    with ib.if_scope(num_valid_boxes[0] == max_output_size):
                        with ib.for_range(0, box_data_length) as k:
                            out[base_idx + offset_j + k] = -1.0
                        box_indices[i * num_anchors + j] = -1
                    with ib.else_scope():
                        num_valid_boxes[0] += 1

    return ib.get()


def non_max_suppression(data, valid_count, max_output_size=-1,
                        iou_threshold=0.5, force_suppress=False, top_k=-1,
                        coord_start=2, score_index=1, id_index=0,
                        return_indices=True, invalid_to_bottom=False):
    """Non-maximum suppression operator for object detection.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D tensor with shape [batch_size, num_anchors, elem_length].
        The last dimension should be in format of
        [class_id, score, box_left, box_top, box_right, box_bottom].

    valid_count : tvm.te.Tensor
        1-D tensor for valid number of boxes.

    max_output_size : optional, int
        Max number of output valid boxes for each instance.
        By default all valid boxes are returned.

    iou_threshold : optional, float
        Non-maximum suppression threshold.

    force_suppress : optional, boolean
        Whether to suppress all detections regardless of class_id.

    top_k : optional, int
        Keep maximum top k detections before nms, -1 for no limit.

    coord_start : required, int
        Start index of the consecutive 4 coordinates.

    score_index : optional, int
        Index of the scores/confidence of boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    return_indices : boolean
        Whether to return box indices in input data.

    invalid_to_bottom : optional, boolean
        Whether to move all valid bounding boxes to the top.

    Returns
    -------
    out : tvm.te.Tensor
        3-D tensor with shape [batch_size, num_anchors, elem_length].

    Example
    --------
    .. code-block:: python

        # An example to use nms
        dshape = (1, 5, 6)
        data = te.placeholder(dshape, name="data")
        valid_count = te.placeholder((dshape[0],), dtype="int32", name="valid_count")
        iou_threshold = 0.7
        force_suppress = True
        top_k = -1
        out = non_max_suppression(data=data, valid_count=valid_count, iou_threshold=iou_threshold,
                                 force_suppress=force_supress, top_k=top_k, return_indices=False)
        np_data = np.random.uniform(dshape)
        np_valid_count = np.array([4])
        s = topi.generic.schedule_nms(out)
        f = tvm.build(s, [data, valid_count, out], "cuda")
        ctx = tvm.gpu(0)
        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_valid_count = tvm.nd.array(np_valid_count, ctx)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data.dtype), ctx)
        f(tvm_data, tvm_valid_count, tvm_out)
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]

    valid_count_dtype = "int32"
    valid_count_buf = tvm.tir.decl_buffer(valid_count.shape, valid_count_dtype,
                                          "valid_count_buf", data_alignment=4)
    score_axis = score_index
    score_shape = (batch_size, num_anchors)
    score_tensor = te.compute(
        score_shape, lambda i, j: data[i, j, score_axis], tag=tag.ELEMWISE)
    if tvm.get_global_func("tvm.contrib.thrust.sort_nms", allow_missing=True):
        sort_tensor = argsort_thrust(
            score_tensor, valid_count=None, axis=1, is_ascend=False, dtype=valid_count_dtype)
    else:
        sort_tensor = argsort(
            score_tensor, valid_count=None, axis=1, is_ascend=False, dtype=valid_count_dtype)

    sort_tensor_buf = tvm.tir.decl_buffer(sort_tensor.shape, sort_tensor.dtype,
                                          "sort_tensor_buf", data_alignment=8)

    data_buf = tvm.tir.decl_buffer(
        data.shape, data.dtype, "data_buf", data_alignment=8)

    out, box_indices = \
        te.extern([data.shape, score_shape],
                  [data, sort_tensor, valid_count],
                  lambda ins, outs: nms_ir(
            ins[0], ins[1], ins[2], outs[0], outs[1],
            max_output_size, iou_threshold, force_suppress,
            top_k, coord_start, id_index, score_index),
            dtype=[data.dtype, "int32"],
            in_buffers=[data_buf, sort_tensor_buf, valid_count_buf],
            name="nms",
            tag="nms")

    if return_indices:
        return box_indices

    return out
