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
# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments, too-many-statements, singleton-comparison, unused-argument
"""Argsort operator """
import tvm
from tvm import te

from .injective import schedule_injective_from_existing
from ..math import identity
from ..transform import strided_slice, transpose
from .. import tag


def swap(arr, axis):
    """ swap arr[axis] and arr[-1] """
    return arr[:axis] + [arr[-1]] + arr[axis + 1 : -1] + [arr[axis]]


def _schedule_sort(outs):
    """Schedule for argsort operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of argsort
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        if tag.is_injective(op.tag):
            schedule_injective_from_existing(s, op.output(0))
        for tensor in op.input_tensors:
            if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                traverse(tensor.op)
        scheduled_ops.append(op)

    for out in outs:
        traverse(out.op)
    return s


def sort_ir(data, values_out, axis, is_ascend, indices_out=None):
    """Low level IR to do nms sorting on the GPU, same usage as tvm.contrib.sort.argsort on the CPU.

    Parameters
    ----------
    data: Buffer
        Buffer of input data. Data will be sorted in place.

    output : Buffer
        Output buffer of indicies of sorted tensor with same shape as data.

    axis : Int
        Axis long which to sort the input tensor.

    is_ascend : Boolean
        Whether to sort in ascending or descending order.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    axis_mul_before = 1
    axis_mul_after = 1
    shape = data.shape
    if axis < 0:
        axis = len(shape) + axis
    for i, value in enumerate(shape, 0):
        if i < axis:
            axis_mul_before *= value
        elif i > axis:
            axis_mul_after *= value
    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    ib = tvm.tir.ir_builder.create()
    data = ib.buffer_ptr(data)
    values_out = ib.buffer_ptr(values_out)
    if indices_out is not None:
        indices_out = ib.buffer_ptr(indices_out)
    nthread_tx = max_threads
    nthread_bx = shape[axis] // max_threads + 1

    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("vthread")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "virtual_thread", nthread_bx)
    tid = bx * nthread_tx + tx
    temp_data = ib.allocate(values_out.dtype, (1,), name="temp_data", scope="local")
    if indices_out is not None:
        temp_index = ib.allocate(indices_out.dtype, (1,), name="temp_index", scope="local")

    with ib.for_range(0, axis_mul_before) as i:
        with ib.for_range(0, axis_mul_after) as j:
            base_idx = i * shape[axis] * axis_mul_after + j
            with ib.if_scope(tid < shape[axis]):
                values_out[base_idx + tid * axis_mul_after] = data[base_idx + tid * axis_mul_after]
                if indices_out is not None:
                    indices_out[base_idx + tid * axis_mul_after] = tvm.tir.generic.cast(
                        tid, indices_out.dtype
                    )
    ib.emit(tvm.tir.Call(None, "tir.tvm_storage_sync", tvm.runtime.convert(["shared"])))
    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    with ib.for_range(0, axis_mul_before) as i:
        with ib.for_range(0, axis_mul_after) as j:
            current_sort_num = shape[axis]
            base_idx = i * shape[axis] * axis_mul_after + j
            # OddEvenTransposeSort
            with ib.for_range(0, current_sort_num) as k:
                with ib.if_scope(tid < idxd(current_sort_num + 1, 2)):
                    offset = base_idx + (2 * tid + idxm(k, 2)) * axis_mul_after
                    if is_ascend:
                        cond = tvm.tir.all(
                            2 * tid + idxm(k, 2) + 1 < current_sort_num,
                            values_out[offset] > values_out[offset + axis_mul_after],
                        )
                    else:
                        cond = tvm.tir.all(
                            2 * tid + idxm(k, 2) + 1 < current_sort_num,
                            values_out[offset] < values_out[offset + axis_mul_after],
                        )
                    with ib.if_scope(cond):
                        temp_data[0] = values_out[offset]
                        values_out[offset] = values_out[offset + axis_mul_after]
                        values_out[offset + axis_mul_after] = temp_data[0]
                        if indices_out is not None:
                            temp_index[0] = indices_out[offset]
                            indices_out[offset] = indices_out[offset + axis_mul_after]
                            indices_out[offset + axis_mul_after] = temp_index[0]
                ib.emit(tvm.tir.Call(None, "tir.tvm_storage_sync", tvm.runtime.convert(["shared"])))

    return ib.get()


def sort_nms_ir(data, valid_count, output, axis, is_ascend):
    """Low level IR to do nms sorting on the GPU, same usage as tvm.contrib.sort.argsort on the CPU.

    Parameters
    ----------
    data: Buffer
        Buffer of input data.

    valid_count : Buffer
        1D Buffer of number of valid number of boxes.

    output : Buffer
        Output buffer of indicies of sorted tensor with same shape as data.

    axis : Int
        Axis long which to sort the input tensor.

    is_ascend : Boolean
        Whether to sort in ascending or descending order.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """

    size = 1
    axis_mul_before = 1
    axis_mul_after = 1
    shape = data.shape
    if axis < 0:
        axis = len(shape) + axis
    for i, value in enumerate(shape, 0):
        size *= value
        if i < axis:
            axis_mul_before *= value
        elif i > axis:
            axis_mul_after *= value
    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    ib = tvm.tir.ir_builder.create()
    data = ib.buffer_ptr(data)
    valid_count = ib.buffer_ptr(valid_count)
    output = ib.buffer_ptr(output)
    nthread_tx = max_threads
    nthread_bx = size // max_threads + 1
    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("vthread")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "virtual_thread", nthread_bx)
    tid = bx * nthread_tx + tx
    temp_data = ib.allocate("float32", (1,), name="temp_data", scope="local")
    temp_index = ib.allocate("int32", (1,), name="temp_index", scope="local")
    is_ascend = tvm.ir.make_node("IntImm", dtype="int32", value=is_ascend)

    idxd = tvm.tir.indexdiv
    idxm = tvm.tir.indexmod

    with ib.for_range(0, axis_mul_before) as i:
        with ib.for_range(0, axis_mul_after) as j:
            current_sort_num = valid_count[i * axis_mul_after + j]
            base_idx = i * shape[axis] * axis_mul_after + j
            with ib.if_scope(tid < shape[axis]):
                output[base_idx + tid * axis_mul_after] = tid
            # OddEvenTransposeSort
            with ib.for_range(0, current_sort_num) as k:
                with ib.if_scope(tid < idxd(current_sort_num + 1, 2)):
                    offset = base_idx + (2 * tid + idxm(k, 2)) * axis_mul_after
                    with ib.if_scope(
                        tvm.tir.all(
                            is_ascend == 1,
                            2 * tid + idxm(k, 2) + 1 < current_sort_num,
                            data[offset] > data[offset + axis_mul_after],
                        )
                    ):
                        temp_data[0] = data[offset]
                        data[offset] = data[offset + axis_mul_after]
                        data[offset + axis_mul_after] = temp_data[0]
                        temp_index[0] = output[offset]
                        output[offset] = output[offset + axis_mul_after]
                        output[offset + axis_mul_after] = temp_index[0]
                    with ib.if_scope(
                        tvm.tir.all(
                            is_ascend == 0,
                            2 * tid + idxm(k, 2) + 1 < current_sort_num,
                            data[offset] < data[offset + axis_mul_after],
                        )
                    ):
                        temp_data[0] = data[offset]
                        data[offset] = data[offset + axis_mul_after]
                        data[offset + axis_mul_after] = temp_data[0]
                        temp_index[0] = output[offset]
                        output[offset] = output[offset + axis_mul_after]
                        output[offset + axis_mul_after] = temp_index[0]
                ib.emit(tvm.tir.Call(None, "tir.tvm_storage_sync", tvm.runtime.convert(["shared"])))

    return ib.get()


def argsort_nms_thrust(data, valid_count, axis=-1, is_ascend=1, dtype="float32"):
    """Performs sorting along the given axis and returns an array of indicies
    having same shape as an input array that index data in sorted order.

    Parameters
    ----------
    data: tvm.te.Tensor
        The input array.

    valid_count : tvm.te.Tensor, optional
        The number of valid elements to be sorted.

    axis : int, optional
        Axis long which to sort the input tensor.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        DType of the output indices.

    Returns
    -------
    out : tvm.te.Tensor
        The output of this function.
    """
    ndim = len(data.shape)
    if axis < 0:
        axis = ndim + axis
    if axis != ndim - 1:
        # Prepare for sorting along axis -1.
        axes = swap(list(range(ndim)), axis)
        data = transpose(data, axes)

    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    valid_count_buf = tvm.tir.decl_buffer(
        valid_count.shape, valid_count.dtype, "valid_count_buf", data_alignment=4
    )
    out_bufs = [
        tvm.tir.decl_buffer(data.shape, data.dtype, "value_buf", data_alignment=8),
        tvm.tir.decl_buffer(data.shape, "int32", "indices_buf", data_alignment=8),
    ]
    out = te.extern(
        [data.shape, data.shape],
        [data, valid_count],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.thrust.sort_nms", ins[0], ins[1], outs[0], outs[1], is_ascend
        ),
        in_buffers=[data_buf, valid_count_buf],
        out_buffers=out_bufs,
        dtype=[data.dtype, "int32"],
        name="nms_argsort_gpu",
        tag="nms_argsort_gpu",
    )

    if axis != ndim - 1:
        axes = swap(list(range(ndim)), axis)
        out = [transpose(o, axes) for o in out]

    return out[1]


def argsort(data, valid_count=None, axis=-1, is_ascend=1, dtype="float32"):
    """Performs sorting along the given axis and returns an array of indicies
    having same shape as an input array that index data in sorted order.

    Parameters
    ----------
    data: tvm.te.Tensor
        The input array.

    valid_count : tvm.te.Tensor, optional
        The number of valid elements to be sorted.

    axis : int, optional
        Axis long which to sort the input tensor.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        DType of the output indices.

    Returns
    -------
    out : tvm.te.Tensor
        The output of this function.
    """
    if valid_count is not None:
        sorted_data = identity(data)
        sorted_data_buf = tvm.tir.decl_buffer(
            data.shape, data.dtype, "sorted_data_buf", data_alignment=8
        )
        valid_count_buf = tvm.tir.decl_buffer(
            valid_count.shape, valid_count.dtype, "valid_count_buf", data_alignment=4
        )
        out_buf = tvm.tir.decl_buffer(data.shape, "int32", "out_buf", data_alignment=4)
        out = te.extern(
            [data.shape],
            [sorted_data, valid_count],
            lambda ins, outs: sort_nms_ir(ins[0], ins[1], outs[0], axis, is_ascend),
            dtype="int32",
            in_buffers=[sorted_data_buf, valid_count_buf],
            out_buffers=[out_buf],
            name="argsort_nms_gpu",
            tag="argsort_nms_gpu",
        )
    else:
        value_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "value_buf", data_alignment=8)
        indices_buf = tvm.tir.decl_buffer(data.shape, dtype, "out_buf", data_alignment=8)
        out = te.extern(
            [data.shape, data.shape],
            [data],
            lambda ins, outs: sort_ir(ins[0], outs[0], axis, is_ascend, indices_out=outs[1]),
            out_buffers=[value_buf, indices_buf],
            name="argsort_gpu",
            tag="argsort_gpu",
        )[1]
    return out


def argsort_thrust(data, valid_count=None, axis=-1, is_ascend=1, dtype="float32"):
    """Performs sorting along the given axis and returns an array of indicies
    having same shape as an input array that index data in sorted order.

    Parameters
    ----------
    data: tvm.te.Tensor
        The input array.

    valid_count : tvm.te.Tensor, optional
        The number of valid elements to be sorted.

    axis : int, optional
        Axis long which to sort the input tensor.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        DType of the output indices.

    Returns
    -------
    out : tvm.te.Tensor
        The output of this function.
    """
    if valid_count is not None:
        out = argsort_nms_thrust(data, valid_count, axis, is_ascend, dtype)
    else:
        out = topk_thrust(data, 0, axis, "indices", is_ascend, dtype)
    return out


def schedule_argsort(outs):
    """Schedule for argsort operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of argsort
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    return _schedule_sort(outs)


def topk(data, k=1, axis=-1, ret_type="both", is_ascend=False, dtype="int64"):
    """Get the top k elements in an input tensor along the given axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    k : int, optional
        Number of top elements to select. Return all elements if k < 1.

    axis : int, optional
        Axis long which to sort the input tensor.

    ret_type: str, optional
        The return type [both, values, indices].
        "both": return both top k data and indices.
        "values": return top k data only.
        "indices": return top k indices only.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        The data type of the indices output.

    Returns
    -------
    out : tvm.te.Tensor or List[tvm.te.Tensor]
        The computed result.
    """
    assert ret_type in ["both", "values", "indices"]
    ndim = len(data.shape)
    axis = axis + ndim if axis < 0 else axis
    assert 0 <= axis < ndim
    values_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "values_buf", data_alignment=8)
    indices_buf = tvm.tir.decl_buffer(data.shape, dtype, "indices_buf", data_alignment=8)
    if ret_type == "values":
        output = te.extern(
            [data.shape],
            [data],
            lambda ins, outs: sort_ir(ins[0], outs[0], axis, is_ascend),
            out_buffers=[values_buf],
            name="topk_gpu",
            tag="topk_gpu",
        )
    else:
        output = te.extern(
            [data.shape, data.shape],
            [data],
            lambda ins, outs: sort_ir(ins[0], outs[0], axis, is_ascend, indices_out=outs[1]),
            out_buffers=[values_buf, indices_buf],
            name="topk_gpu",
            tag="topk_gpu",
        )
    if k < 1:
        if ret_type == "indices":
            return output[1]
        return output
    beg = [0] * ndim
    end = []
    for i in range(ndim):
        if i == axis:
            end.append(k)
        else:
            end.append(data.shape[i])
    if ret_type == "both":
        values_out, indices_out = output
        values_out = strided_slice(values_out, beg, end)
        indices_out = strided_slice(indices_out, beg, end)
        output = [values_out, indices_out]
    elif ret_type == "values":
        output = [strided_slice(output, beg, end)]
    else:  # ret_type == "indices"
        indices_out = output[1]
        output = [strided_slice(indices_out, beg, end)]
    return output


def topk_thrust(data, k=1, axis=-1, ret_type="both", is_ascend=False, dtype="int64"):
    """Get the top k elements in an input tensor along the given axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    k : int, optional
        Number of top elements to select. Return all elements if k < 1.

    axis : int, optional
        Axis long which to sort the input tensor.

    ret_type: str, optional
        The return type [both, values, indices].
        "both": return both top k data and indices.
        "values": return top k data only.
        "indices": return top k indices only.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        The data type of the indices output.

    Returns
    -------
    out : tvm.te.Tensor or List[tvm.te.Tensor]
        The computed result.
    """
    assert ret_type in ["both", "values", "indices"]
    ndim = len(data.shape)
    axis = ndim + axis if axis < 0 else axis

    if axis != ndim - 1:
        # Prepare for sorting along axis -1.
        axes = swap(list(range(ndim)), axis)
        data = transpose(data, axes)

    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    out_bufs = [
        tvm.tir.decl_buffer(data.shape, data.dtype, "value_buf", data_alignment=8),
        tvm.tir.decl_buffer(data.shape, dtype, "indices_buf", data_alignment=8),
    ]

    out = te.extern(
        [data.shape, data.shape],
        [data],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.thrust.sort", ins[0], outs[0], outs[1], is_ascend
        ),
        in_buffers=[data_buf],
        out_buffers=out_bufs,
        name="topk_gpu",
        tag="topk_gpu",
    )

    if k > 0:
        beg = [0] * ndim
        end = data.shape[:-1] + [k]
        out = [strided_slice(o, beg, end) for o in out]

    if axis != ndim - 1:
        axes = swap(list(range(ndim)), axis)
        out = [transpose(o, axes) for o in out]

    if ret_type == "values":
        out = out[0]
    elif ret_type == "indices":
        out = out[1]

    return out


def schedule_topk(outs):
    """Schedule for argsort operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of argsort
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    return _schedule_sort(outs)
