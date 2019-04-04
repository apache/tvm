# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments, too-many-statements, singleton-comparison, unused-argument
"""Argsort operator """
import tvm

from tvm import api
from topi.vision.sort import argsort

def sort_ir(data, output, axis, is_ascend):
    """Low level IR to do nms sorting on the GPU, same usage as tvm.contrib.sort.argsort on the CPU.

    Parameters
    ----------
    data: Buffer
        Buffer of input data.

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
    max_threads = int(tvm.target.current_target(allow_none=False).max_num_threads)
    ib = tvm.ir_builder.create()
    data = ib.buffer_ptr(data)
    output = ib.buffer_ptr(output)
    nthread_tx = max_threads
    nthread_bx = size // max_threads + 1
    tx = tvm.thread_axis("threadIdx.x")
    bx = tvm.thread_axis("vthread")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "virtual_thread", nthread_bx)
    tid = bx * nthread_tx + tx
    temp_data = ib.allocate("float32", (1,), name="temp_data", scope="local")
    temp_index = ib.allocate("float32", (1,), name="temp_index", scope="local")
    is_ascend = tvm.make.node("IntImm", dtype="int32", value=is_ascend)

    with ib.for_range(0, axis_mul_before) as i:
        with ib.for_range(0, axis_mul_after) as j:
            current_sort_num = shape[axis]
            base_idx = i * shape[axis] * axis_mul_after + j
            with ib.if_scope(tid < shape[axis]):
                output[base_idx + tid * axis_mul_after] = tid.astype("float32")
            # OddEvenTransposeSort
            with ib.for_range(0, current_sort_num) as k:
                with ib.if_scope(tid < (current_sort_num + 1) // 2):
                    offset = base_idx + (2 * tid + (k % 2)) * axis_mul_after
                    with ib.if_scope(tvm.all(is_ascend == 1, \
                                             2 * tid + (k % 2) + 1 < current_sort_num, \
                                             data[offset] > data[offset + axis_mul_after])):
                        temp_data[0] = data[offset]
                        data[offset] = data[offset + axis_mul_after]
                        data[offset + axis_mul_after] = temp_data[0]
                        temp_index[0] = output[offset]
                        output[offset] = output[offset + axis_mul_after]
                        output[offset + axis_mul_after] = temp_index[0]
                    with ib.if_scope(tvm.all(is_ascend == 0, \
                                             2 * tid + (k % 2) + 1 < current_sort_num, \
                                             data[offset] < data[offset + axis_mul_after])):
                        temp_data[0] = data[offset]
                        data[offset] = data[offset + axis_mul_after]
                        data[offset + axis_mul_after] = temp_data[0]
                        temp_index[0] = output[offset]
                        output[offset] = output[offset + axis_mul_after]
                        output[offset + axis_mul_after] = temp_index[0]
                ib.emit(tvm.make.Call(None, 'tvm_storage_sync',
                                      tvm.convert(['shared']),
                                      tvm.expr.Call.Intrinsic, None, 0))

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
    max_threads = int(tvm.target.current_target(allow_none=False).max_num_threads)
    ib = tvm.ir_builder.create()
    data = ib.buffer_ptr(data)
    valid_count = ib.buffer_ptr(valid_count)
    output = ib.buffer_ptr(output)
    nthread_tx = max_threads
    nthread_bx = size // max_threads + 1
    tx = tvm.thread_axis("threadIdx.x")
    bx = tvm.thread_axis("vthread")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "virtual_thread", nthread_bx)
    tid = bx * nthread_tx + tx
    temp_data = ib.allocate("float32", (1,), name="temp_data", scope="local")
    temp_index = ib.allocate("int32", (1,), name="temp_index", scope="local")
    is_ascend = tvm.make.node("IntImm", dtype="int32", value=is_ascend)

    with ib.for_range(0, axis_mul_before) as i:
        with ib.for_range(0, axis_mul_after) as j:
            current_sort_num = valid_count[i * axis_mul_after + j]
            base_idx = i * shape[axis] * axis_mul_after + j
            with ib.if_scope(tid < shape[axis]):
                output[base_idx + tid * axis_mul_after] = tid
            # OddEvenTransposeSort
            with ib.for_range(0, current_sort_num) as k:
                with ib.if_scope(tid < (current_sort_num + 1) // 2):
                    offset = base_idx + (2 * tid + (k % 2)) * axis_mul_after
                    with ib.if_scope(tvm.all(is_ascend == 1, \
                                             2 * tid + (k % 2) + 1 < current_sort_num, \
                                             data[offset] > data[offset + axis_mul_after])):
                        temp_data[0] = data[offset]
                        data[offset] = data[offset + axis_mul_after]
                        data[offset + axis_mul_after] = temp_data[0]
                        temp_index[0] = output[offset]
                        output[offset] = output[offset + axis_mul_after]
                        output[offset + axis_mul_after] = temp_index[0]
                    with ib.if_scope(tvm.all(is_ascend == 0, \
                                             2 * tid + (k % 2) + 1 < current_sort_num, \
                                             data[offset] < data[offset + axis_mul_after])):
                        temp_data[0] = data[offset]
                        data[offset] = data[offset + axis_mul_after]
                        data[offset + axis_mul_after] = temp_data[0]
                        temp_index[0] = output[offset]
                        output[offset] = output[offset + axis_mul_after]
                        output[offset + axis_mul_after] = temp_index[0]
                ib.emit(tvm.make.Call(None, 'tvm_storage_sync',
                                      tvm.convert(['shared']),
                                      tvm.expr.Call.Intrinsic, None, 0))

    return ib.get()

@argsort.register(["cuda", "gpu"])
def argsort_gpu(data, valid_count, axis=-1, is_ascend=1, dtype="float32", flag=0):
    """Performs sorting along the given axis and returns an array of indicies
    having same shape as an input array that index data in sorted order.

    Parameters
    ----------
    data: tvm.Tensor
        The input array.

    valid_count : tvm.Tensor
        The number of valid elements to be sorted.

    axis : int
        Axis long which to sort the input tensor.

    is_ascend : boolean
        Whether to sort in ascending or descending order.

    Returns
    -------
    out : tvm.Tensor
        The output of this function.
    """
    data_buf = api.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    if flag:
        valid_count_buf = api.decl_buffer(valid_count.shape, valid_count.dtype,
                                          "valid_count_buf", data_alignment=4)
        out_buf = api.decl_buffer(data.shape, "int32", "out_buf", data_alignment=4)
        out = tvm.extern([data.shape],
                         [data, valid_count],
                         lambda ins, outs: sort_nms_ir(
                             ins[0], ins[1], outs[0], axis, is_ascend),
                         dtype="int32",
                         in_buffers=[data_buf, valid_count_buf],
                         out_buffers=[out_buf],
                         name="argsort_nms_gpu",
                         tag="argsort_nms_gpu")
    else:
        out_buf = api.decl_buffer(data.shape, dtype, "out_buf", data_alignment=8)
        out = tvm.extern([data.shape],
                         [data],
                         lambda ins, outs: sort_ir(
                             ins[0], outs[0], axis, is_ascend),
                         dtype=dtype,
                         in_buffers=[data_buf],
                         out_buffers=[out_buf],
                         name="argsort_gpu",
                         tag="argsort_gpu")
    return out
