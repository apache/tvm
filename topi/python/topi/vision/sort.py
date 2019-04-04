"""Argsort operator"""
import tvm
from tvm import api

@tvm.target.generic_func
def argsort(data, valid_count, axis=-1, is_ascend=1, dtype="float32", flag=0):
    """Performs sorting along the given axis and returns an array
    of indices having the same shape as an input array that index
    data in sorted order.

    Parameters
    ----------
    data : tvm.Tensor
        The input tensor.

    valid_count : tvm.Tensor
        1-D tensor for valid number of boxes only for ssd.

    axis : optional, int
	Axis along which to sort the input tensor.
        By default the flattened array is used.

    is_ascend : optional, boolean
        Whether to sort in ascending or descending order.

    dtype : optional, string
        DType of the output indices.

    flag : optional, boolean
        Whether valid_count is valid.

    Returns
    -------
    out : tvm.Tensor
        Sorted index tensor.

    Example
    --------
    .. code-block:: python

        # An example to use argsort
        dshape = (1, 5, 6)
        data = tvm.placeholder(dshape, name="data")
        valid_count = tvm.placeholder((dshape[0],), dtype="int32", name="valid_count")
        axis = 0
        is_ascend = False
        flag = False
        out = argsort(data, valid_count, axis, is_ascend, flag)
        np_data = np.random.uniform(dshape)
        np_valid_count = np.array([4])
        s = topi.generic.schedule_argsort(out)
        f = tvm.build(s, [data, valid_count, out], "llvm")
        ctx = tvm.cpu()
        tvm_data = tvm.nd.array(np_data, ctx)
        tvm_valid_count = tvm.nd.array(np_valid_count, ctx)
        tvm_out = tvm.nd.array(np.zeros(dshape, dtype=data.dtype), ctx)
        f(tvm_data, tvm_valid_count, tvm_out)
    """
    data_buf = api.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    if flag:
        valid_count_buf = api.decl_buffer(valid_count.shape, valid_count.dtype,
                                          "valid_count_buf", data_alignment=4)
        out_buf = api.decl_buffer(data.shape, "int32", "out_buf", data_alignment=8)
        out = \
            tvm.extern(data.shape,
                       [data, valid_count],
                       lambda ins, outs: tvm.call_packed(
                           "tvm.contrib.sort.argsort_nms", ins[0], ins[1],
                           outs[0], axis, is_ascend),
                       dtype="int32",
                       in_buffers=[data_buf, valid_count_buf],
                       out_buffers=out_buf,
                       name="argsort_nms_cpu",
                       tag="argsort_nms_cpu")
    else:
        out_buf = api.decl_buffer(data.shape, dtype, "out_buf", data_alignment=8)
        out = \
            tvm.extern(data.shape,
                       [data],
                       lambda ins, outs: tvm.call_packed(
                           "tvm.contrib.sort.argsort", ins[0],
                           outs[0], axis, is_ascend, dtype),
                       dtype=dtype,
                       in_buffers=[data_buf],
                       out_buffers=out_buf,
                       name="argsort_cpu",
                       tag="argsort_cpu")
    return out
