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


def cuda_atomic_add_rule(op):
    if op.dtype == "float32":
        return tvm.tir.call_pure_extern("float32", "atomicAdd", op.args[0], op.args[1])
    if op.dtype == "float64":
        return tvm.tir.call_pure_extern("float64", "atomicAdd", op.args[0], op.args[1])
    if op.dtype == "int32":
        return tvm.tir.call_pure_extern("int32", "atomicAdd", op.args[0], op.args[1])
    raise RuntimeError("only support int32, float32 and float64")


def opencl_atomic_add_rule(op):
    if op.dtype == "int32":
        return tvm.tir.call_pure_extern("int32", "atomic_add", op.args[0], op.args[1])
    raise RuntimeError("only support int32")


tvm.target.intrin.register_intrin_rule("cuda", "atomic_add", cuda_atomic_add_rule, override=True)

tvm.target.intrin.register_intrin_rule(
    "opencl", "atomic_add", opencl_atomic_add_rule, override=True
)


def atomic_add(x, y):
    return tvm.tir.call_intrin(y.dtype, "tir.atomic_add", x, y)


def rearrange_indices_out_ir(data, output, valid_box_count):
    """Hybrid routine to rearrange nms output to
    move all valid entries to top.

    Parameters
    ----------
    data : tvm.te.Tensor or numpy NDArray
        NMS output. 3-D tensor with shape
        [batch_size, num_anchors, 6] or
        [batch_size, num_anchors, 5], or 2-D
        tensor with shape [batch_size, num_anchors].

    one: tvm.tir.const
        Constant one with the same dtype as data.

    batch_size: tvm.tir.IntImm or tvm.tir.Var
        Batch size. We need to pass it in since hybrid script doesn't support
        binding variable to symbolic dim.

    num_anchors: tvm.tir.IntImm or tvm.tir.Var
        Number of anchors.

    Returns
    -------
    output : tvm.te.Tensor or numpy NDArray
        2-D tensor with shape [batch_size, num_anchors].

    valid_box_count : tvm.te.Tensor or numpy NDArray
        Tensor with shape [batch_size, 1], indicates
        the valid number of boxes.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]

    ib = tvm.tir.ir_builder.create()

    data = ib.buffer_ptr(data)
    valid_box_count = ib.buffer_ptr(valid_box_count)
    output = ib.buffer_ptr(output)

    with ib.new_scope():
        i = te.thread_axis("blockIdx.x")
        ib.scope_attr(i, "thread_extent", batch_size)
        valid_idx = ib.allocate("int32", (1,), name="valid_idx", scope="local")
        valid_idx[0] = 0
        with ib.for_range(0, num_anchors, name="j") as j:
            with ib.if_scope(data[i, j] >= 0):
                with ib.if_scope(data[i, j] > num_anchors):
                    output[i, valid_idx[0]] = 0
                    valid_idx[0] = valid_idx[0] + 1
                with ib.else_scope():
                    output[i, valid_idx[0]] = data[i, j]
                    valid_idx[0] = valid_idx[0] + 1
            with ib.else_scope():
                with ib.if_scope(data[i, j] < -num_anchors):
                    output[i, valid_idx[0]] = 0
                    valid_idx[0] = valid_idx[0] + 1
            with ib.if_scope(j >= valid_idx[0]):
                output[i, j] = -1
        valid_box_count[i, 0] = valid_idx[0]

    return ib.get()


def get_valid_boxes_ir(data, valid_boxes, score_threshold, id_index, score_index):
    """Low level IR to identify bounding boxes given a score threshold.

    Parameters
    ----------
    data : Buffer
        Input data. 3-D Buffer with shape [batch_size, num_anchors, elem_length].

    score_threshold : Buffer or float32
        Lower limit of score for valid bounding boxes.

    id_index : optional, int
        index of the class categories, -1 to disable.

    score_index: optional, int
        Index of the scores/confidence of boxes.

    Returns
    -------
    valid_boxes: Buffer
        2D Buffer  indicating valid boxes with shape [batch_size, num_anchors].

    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    elem_length = data.shape[2]

    ib = tvm.tir.ir_builder.create()

    data = ib.buffer_ptr(data)

    valid_boxes = ib.buffer_ptr(valid_boxes)
    if isinstance(score_threshold, float):
        score_threshold = tvm.tir.FloatImm("float32", score_threshold)
    id_index = tvm.tir.IntImm("int32", id_index)
    score_index = tvm.tir.IntImm("int32", score_index)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = num_anchors // max_threads + 1
        nthread_by = batch_size
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        by = te.thread_axis("blockIdx.y")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        ib.scope_attr(by, "thread_extent", nthread_by)
        tid = bx * max_threads + tx

        with ib.if_scope(tid < num_anchors):
            i = by
            j = tid
            score = data[(i * num_anchors + j) * elem_length + score_index]
            with ib.if_scope(
                tvm.tir.all(
                    score > score_threshold,
                    tvm.tir.any(
                        id_index < 0, data[(i * num_anchors + j) * elem_length + id_index] >= 0
                    ),
                )
            ):
                valid_boxes[i * num_anchors + j] = 1
            with ib.else_scope():
                valid_boxes[i * num_anchors + j] = 0
    return ib.get()


def get_valid_indices_ir(valid_boxes, valid_count, valid_indices):
    """Low level IR to get the ouput indices of valid boxes
    and the count of valid boxes

    Parameters
    ----------
    valid_boxes: Buffer
        2D Buffer  indicating valid boxes with shape [batch_size, num_anchors].

    Returns
    -------
    valid_count: Buffer
        1D Buffer of number of valid boxes per batch [batch_size].

    valid_indices: Buffer
        2D Buffer indicating output sorted indcies of valid boxes [batch_size, num_anchors].
    """
    batch_size = valid_boxes.shape[0]
    num_anchors = valid_boxes.shape[1]

    ib = tvm.tir.ir_builder.create()

    valid_boxes = ib.buffer_ptr(valid_boxes)

    valid_count = ib.buffer_ptr(valid_count)
    valid_indices = ib.buffer_ptr(valid_indices)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = batch_size // max_threads + 1
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        # TODO(mbrookhart): Parallelize the sum and cumsum here
        current_index = ib.allocate("int32", (1,), name="current_index", scope="local")
        with ib.if_scope(tid < batch_size):
            current_index[0] = 0
            valid_count[tid] = 0
            with ib.for_range(0, num_anchors) as j:
                idx = tid * num_anchors + j
                valid_count[tid] = valid_count[tid] + valid_boxes[idx]
                with ib.if_scope(valid_boxes[idx] == 1):
                    valid_indices[idx] = current_index[0]
                    current_index[0] = current_index[0] + 1
                with ib.else_scope():
                    valid_indices[idx] = -1
    return ib.get()


def get_valid_counts_ir(data, valid_indices, out, out_indices):
    """Low level IR to get valid count of bounding boxes
    given a score threshold. Also prepares to move valid boxes to the
    top of input data.

    Parameters
    ----------
    data : Buffer
        Input data. 3-D Buffer with shape [batch_size, num_anchors, elem_length].

    valid_indices: Buffer
        2D Buffer of flag indicating valid data with shape [batch_size, num_anchors].

    Returns
    -------
    out : Buffer
        Sorted valid boxes

    out_indices : Buffer
        Incidices of valid boxes in original data
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    elem_length = data.shape[2]

    ib = tvm.tir.ir_builder.create()

    data = ib.buffer_ptr(data)

    valid_indices = ib.buffer_ptr(valid_indices)
    out = ib.buffer_ptr(out)
    out_indices = ib.buffer_ptr(out_indices)
    one = tvm.tir.const(1, dtype=out.dtype)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_tx = max_threads
    nthread_bx = num_anchors // max_threads + 1
    nthread_by = batch_size
    nthread_bz = elem_length
    with ib.new_scope():
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        by = te.thread_axis("blockIdx.y")
        bz = te.thread_axis("blockIdx.z")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        ib.scope_attr(by, "thread_extent", nthread_by)
        ib.scope_attr(bz, "thread_extent", nthread_bz)
        tid = bx * max_threads + tx
        with ib.if_scope(tid < num_anchors):
            i = by
            j = tid
            k = bz
            out[(i * num_anchors + j) * elem_length + k] = -one
            out_indices[i * num_anchors + j] = -1
    with ib.new_scope():
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        by = te.thread_axis("blockIdx.y")
        bz = te.thread_axis("blockIdx.z")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        ib.scope_attr(by, "thread_extent", nthread_by)
        ib.scope_attr(bz, "thread_extent", nthread_bz)
        tid = bx * max_threads + tx
        with ib.if_scope(tid < num_anchors):
            i = by
            j = tid
            k = bz
            with ib.if_scope(valid_indices[i, tid] >= 0):
                out[(i * num_anchors + valid_indices[i, tid]) * elem_length + k] = data[
                    (i * num_anchors + j) * elem_length + k
                ]
                out_indices[i * num_anchors + valid_indices[i, tid]] = j
    return ib.get()


def get_valid_counts(data, score_threshold=0, id_index=0, score_index=1):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : tvm.te.Tensor
        Input data. 3-D tensor with shape [batch_size, num_anchors, elem_length].

    score_threshold : optional, tvm.te.Tensor or float
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
    num_anchors = data.shape[1]
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    valid_boxes_buf = tvm.tir.decl_buffer(
        (batch_size, num_anchors), "int32", "valid_boxes_buf", data_alignment=8
    )
    valid_boxes = te.extern(
        [(batch_size, num_anchors)],
        [data],
        lambda ins, outs: get_valid_boxes_ir(
            ins[0], outs[0], score_threshold, id_index, score_index
        ),
        dtype=["int32"],
        in_buffers=[data_buf],
        out_buffers=[valid_boxes_buf],
        name="get_valid_boxes",
        tag="get_valid_boxes_gpu",
    )

    valid_indices_buf = tvm.tir.decl_buffer(
        (batch_size, num_anchors), "int32", "valid_indices_buf", data_alignment=8
    )
    valid_count_buf = tvm.tir.decl_buffer(
        (batch_size,), "int32", "valid_count_buf", data_alignment=8
    )
    valid_count, valid_indices = te.extern(
        [(batch_size,), (batch_size, num_anchors)],
        [valid_boxes],
        lambda ins, outs: get_valid_indices_ir(ins[0], outs[0], outs[1]),
        dtype=["int32"],
        in_buffers=[valid_boxes_buf],
        out_buffers=[valid_count_buf, valid_indices_buf],
        name="get_valid_indices",
        tag="get_valid_indices_gpu",
    )

    out_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "out_buf", data_alignment=8)
    out_indices_buf = tvm.tir.decl_buffer(
        (batch_size, num_anchors), "int32", "out_buf", data_alignment=8
    )

    out, out_indices = te.extern(
        [data.shape, (batch_size, num_anchors)],
        [data, valid_indices],
        lambda ins, outs: get_valid_counts_ir(ins[0], ins[1], outs[0], outs[1]),
        dtype=["int32", data.dtype],
        in_buffers=[data_buf, valid_indices_buf],
        out_buffers=[out_buf, out_indices_buf],
        name="get_valid_counts",
        tag="get_valid_counts_gpu",
    )

    return [valid_count, out, out_indices]


def nms_ir(
    data,
    sorted_index,
    valid_count,
    indices,
    out,
    box_indices,
    max_output_size,
    iou_threshold,
    force_suppress,
    top_k,
    coord_start,
    id_index,
    score_index,
    return_indices,
):
    """Low level IR routing for transform location in multibox_detection operator.

    Parameters
    ----------
    data : Buffer
        Buffer of output boxes with class and score.

    sorted_index : Buffer
        Buffer of output box indexes sorted by score.

    valid_count : Buffer
        Buffer of number of valid output boxes.

    indices : Buffer
        indices in original tensor, with shape [batch_size, num_anchors],
        represents the index of box in original data. It could be the third
        output out_indices of get_valid_counts. The values in the second
        dimension are like the output of arange(num_anchors) if get_valid_counts
        is not used before non_max_suppression.

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

    return_indices : boolean
        Whether to return box indices in input data.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """

    def get_boundaries(output, box_idx):
        l = tvm.te.min(
            output[box_idx],
            output[box_idx + 2],
        )
        t = tvm.te.min(
            output[box_idx + 1],
            output[box_idx + 3],
        )
        r = tvm.te.max(
            output[box_idx],
            output[box_idx + 2],
        )
        b = tvm.te.max(
            output[box_idx + 1],
            output[box_idx + 3],
        )
        return l, t, r, b

    def calculate_overlap(out_tensor, box_a_idx, box_b_idx):
        """Calculate overlap of two boxes."""
        a_l, a_t, a_r, a_b = get_boundaries(out_tensor, box_a_idx)
        b_l, b_t, b_r, b_b = get_boundaries(out_tensor, box_b_idx)

        # Overlapping width and height
        w = tvm.te.max(0.0, tvm.te.min(a_r, b_r) - tvm.te.max(a_l, b_l))
        h = tvm.te.max(0.0, tvm.te.min(a_b, b_b) - tvm.te.max(a_t, b_t))

        # Overlapping area
        area = h * w

        # total area of the figure formed by box a and box b
        # except for overlapping area
        u = (a_r - a_l) * (a_b - a_t) + (b_r - b_l) * (b_b - b_t) - area
        return tvm.tir.Select(u <= 0.0, 0.0, area / u)

    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    box_data_length = data.shape[2]

    ib = tvm.tir.ir_builder.create()

    data = ib.buffer_ptr(data)
    sorted_index = ib.buffer_ptr(sorted_index)
    valid_count = ib.buffer_ptr(valid_count)
    indices = ib.buffer_ptr(indices)
    out = ib.buffer_ptr(out)
    box_indices = ib.buffer_ptr(box_indices)

    if isinstance(iou_threshold, float):
        iou_threshold = tvm.tir.FloatImm("float32", iou_threshold)
    top_k = tvm.tir.IntImm("int32", top_k)
    coord_start = tvm.tir.IntImm("int32", coord_start)
    id_index = tvm.tir.IntImm("int32", id_index)
    score_index = tvm.tir.IntImm("int32", score_index)
    force_suppress = tvm.tir.IntImm("int32", 1 if force_suppress else 0)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)

    with ib.new_scope():
        nthread_by = batch_size
        by = te.thread_axis("blockIdx.y")
        ib.scope_attr(by, "thread_extent", nthread_by)
        i = by
        base_idx = i * num_anchors * box_data_length
        with ib.if_scope(tvm.tir.all(iou_threshold > 0, valid_count[i] > 0)):
            # Reorder output
            nkeep = if_then_else(
                tvm.tir.all(top_k > 0, top_k < valid_count[i]), top_k, valid_count[i]
            )
            with ib.for_range(0, nkeep) as j:
                with ib.for_range(0, box_data_length) as k:
                    out[(base_idx + j * box_data_length + k)] = data[
                        (base_idx + sorted_index[i * num_anchors + j] * box_data_length + k)
                    ]
                box_indices[i * num_anchors + j] = sorted_index[i * num_anchors + j]
            with ib.if_scope(tvm.tir.all(top_k > 0, top_k < valid_count[i])):
                with ib.for_range(0, valid_count[i] - nkeep) as j:
                    with ib.for_range(0, box_data_length) as k:
                        out[(base_idx + (j + nkeep) * box_data_length + k)] = -1.0
                    box_indices[i * num_anchors + (j + nkeep)] = -1
    with ib.new_scope():
        nthread_by = batch_size
        by = te.thread_axis("blockIdx.y")
        ib.scope_attr(by, "thread_extent", nthread_by)
        i = by
        base_idx = i * num_anchors * box_data_length
        with ib.if_scope(tvm.tir.all(iou_threshold > 0, valid_count[i] > 0)):
            # Apply nms
            with ib.for_range(0, valid_count[i]) as j:
                with ib.for_range(0, j) as k:
                    offset_k = k * box_data_length
                    with ib.if_scope(
                        tvm.tir.all(
                            out[base_idx + offset_k + score_index] > 0,
                            tvm.tir.any(id_index < 0, out[base_idx + offset_k + id_index] >= 0),
                        )
                    ):
                        offset_j = j * box_data_length
                        with ib.if_scope(
                            tvm.tir.all(
                                j > k,
                                out[base_idx + offset_k + score_index] > 0,
                                tvm.tir.any(id_index < 0, out[base_idx + offset_j + id_index] >= 0),
                                tvm.tir.any(
                                    force_suppress > 0,
                                    id_index < 0,
                                    out[base_idx + offset_k + id_index]
                                    == out[base_idx + offset_j + id_index],
                                ),
                            )
                        ):
                            iou = calculate_overlap(
                                out,
                                base_idx + offset_j + coord_start,
                                base_idx + offset_k + coord_start,
                            )
                            with ib.if_scope(iou >= iou_threshold):
                                out[base_idx + offset_j + score_index] = -1.0
                                with ib.if_scope(id_index >= 0):
                                    out[base_idx + offset_j + id_index] = -1.0
                                box_indices[i * num_anchors + j] = -1
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = num_anchors // max_threads + 1
        nthread_by = batch_size
        nthread_bz = box_data_length
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        by = te.thread_axis("blockIdx.y")
        bz = te.thread_axis("blockIdx.z")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        ib.scope_attr(by, "thread_extent", nthread_by)
        ib.scope_attr(bz, "thread_extent", nthread_bz)
        tid = bx * max_threads + tx
        i = by
        j = tid
        k = bz
        base_idx = i * num_anchors * box_data_length
        with ib.if_scope(tvm.tir.all(iou_threshold > 0, valid_count[i] > 0)):
            pass
        with ib.else_scope():
            with ib.if_scope(j < valid_count[i]):
                offset_j = j * box_data_length
                out[(base_idx + offset_j + k)] = data[base_idx + offset_j + k]
                box_indices[i * num_anchors + j] = j

    with ib.new_scope():
        num_valid_boxes = ib.allocate("int32", (1,), name="num_valid_boxes", scope="local")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(bx, "thread_extent", batch_size)
        i = bx
        base_idx = i * num_anchors * box_data_length
        # Set invalid entry to be -1
        with ib.for_range(0, num_anchors - valid_count[i]) as j:
            with ib.for_range(0, box_data_length) as k:
                out[base_idx + (j + valid_count[i]) * box_data_length + k] = -1.0
            box_indices[i * num_anchors + j + valid_count[i]] = -1
        # Only return max_output_size number of valid boxes
        num_valid_boxes[0] = 0
        with ib.if_scope(max_output_size > 0):
            with ib.for_range(0, valid_count[i]) as j:
                offset_j = j * box_data_length
                with ib.if_scope(out[base_idx + offset_j] >= 0):
                    with ib.if_scope(num_valid_boxes[0] == max_output_size):
                        with ib.for_range(0, box_data_length) as k:
                            out[base_idx + offset_j + k] = -1.0
                        box_indices[i * num_anchors + j] = -1
                    with ib.else_scope():
                        num_valid_boxes[0] += 1

    if return_indices:
        with ib.new_scope():
            nthread_tx = max_threads
            nthread_bx = batch_size // max_threads + 1
            tx = te.thread_axis("threadIdx.x")
            bx = te.thread_axis("blockIdx.x")
            ib.scope_attr(tx, "thread_extent", nthread_tx)
            ib.scope_attr(bx, "thread_extent", nthread_bx)
            i = bx * max_threads + tx
            with ib.if_scope(i < batch_size):
                with ib.for_range(0, valid_count[i]) as j:
                    idx = box_indices[i * num_anchors + j]
                    with ib.if_scope(idx >= 0):
                        box_indices[i * num_anchors + j] = indices[i * num_anchors + idx]

    return ib.get()


def _fetch_score_ir(data, score, axis):
    """
    Fetch score from data.
    This routine is required for dynamic shape nms.
    """
    batch_size = data.shape[0]
    num_anchors = data.shape[1]
    elem_length = data.shape[2]

    ib = tvm.tir.ir_builder.create()

    data = ib.buffer_ptr(data)
    score = ib.buffer_ptr(score)
    with ib.if_scope(num_anchors > 0):
        max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
        nthread_tx = max_threads
        nthread_bx = batch_size * num_anchors // max_threads + 1
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)

        tid = bx * max_threads + tx
        with ib.if_scope(tid < batch_size * num_anchors):
            score[tid] = data[tid * elem_length + axis]

    return ib.get()


def non_max_suppression(
    data,
    valid_count,
    indices,
    max_output_size=-1,
    iou_threshold=0.5,
    force_suppress=False,
    top_k=-1,
    coord_start=2,
    score_index=1,
    id_index=0,
    return_indices=True,
    invalid_to_bottom=False,
):
    """Non-maximum suppression operator for object detection.

    Parameters
    ----------
    data : tvm.te.Tensor
        3-D tensor with shape [batch_size, num_anchors, elem_length].
        The last dimension should be in format of
        [class_id, score, box_left, box_top, box_right, box_bottom].
        It could be the second output out_tensor of get_valid_counts.

    valid_count : tvm.te.Tensor
        1-D tensor for valid number of boxes. It could be the output
        valid_count of get_valid_counts.

    indices : tvm.te.Tensor
        2-D tensor with shape [batch_size, num_anchors], represents
        the index of box in original data. It could be the third
        output out_indices of get_valid_counts. The values in the
        second dimension are like the output of arange(num_anchors)
        if get_valid_counts is not used before non_max_suppression.

    max_output_size : optional, tvm.te.Tensor or int
        Max number of output valid boxes for each instance.
        By default all valid boxes are returned.

    iou_threshold : optional, tvm.te.Tensor or float
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
    valid_count_buf = tvm.tir.decl_buffer(
        valid_count.shape, valid_count_dtype, "valid_count_buf", data_alignment=4
    )
    score_axis = score_index
    score_shape = (batch_size, num_anchors)
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    score_buf = tvm.tir.decl_buffer(score_shape, data.dtype, "score_buf", data_alignment=8)
    score_tensor = te.extern(
        [score_shape],
        [data],
        lambda ins, outs: _fetch_score_ir(
            ins[0],
            outs[0],
            score_axis,
        ),
        dtype=[data.dtype],
        in_buffers=[data_buf],
        out_buffers=[score_buf],
        name="fetch_score",
        tag="fetch_score",
    )
    target = tvm.target.Target.current()
    if (
        target
        and target.kind.name == "cuda"
        and tvm.get_global_func("tvm.contrib.thrust.sort_nms", allow_missing=True)
    ):
        sort_tensor = argsort_thrust(
            score_tensor, valid_count=None, axis=1, is_ascend=False, dtype=valid_count_dtype
        )
    else:
        sort_tensor = argsort(
            score_tensor, valid_count=None, axis=1, is_ascend=False, dtype=valid_count_dtype
        )

    sort_tensor_buf = tvm.tir.decl_buffer(
        sort_tensor.shape, sort_tensor.dtype, "sort_tensor_buf", data_alignment=8
    )

    indices_buf = tvm.tir.decl_buffer(indices.shape, indices.dtype, "indices_buf", data_alignment=8)

    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    indices_buf = tvm.tir.decl_buffer(indices.shape, indices.dtype, "indices_buf", data_alignment=8)

    out, box_indices = te.extern(
        [data.shape, score_shape],
        [data, sort_tensor, valid_count, indices],
        lambda ins, outs: nms_ir(
            ins[0],
            ins[1],
            ins[2],
            ins[3],
            outs[0],
            outs[1],
            max_output_size,
            iou_threshold,
            force_suppress,
            top_k,
            coord_start,
            id_index,
            score_index,
            return_indices,
        ),
        dtype=[data.dtype, "int32"],
        in_buffers=[data_buf, sort_tensor_buf, valid_count_buf, indices_buf],
        name="nms",
        tag="nms",
    )
    if return_indices:
        out_shape = box_indices.shape
        valid_box_count_shape = [box_indices.shape[0], 1]
        valid_box_count = tvm.tir.decl_buffer(valid_box_count_shape, "int32", "valid_box_count")
        output = tvm.tir.decl_buffer(box_indices.shape, "int32", "output")
        return te.extern(
            [out_shape, valid_box_count_shape],
            [box_indices],
            lambda ins, outs: rearrange_indices_out_ir(ins[0], outs[0], outs[1]),
            dtype="int32",
            out_buffers=[output, valid_box_count],
            name="rearrange_indices_out_gpu",
            tag="rearrange_indices_out_gpu",
        )

    return out
