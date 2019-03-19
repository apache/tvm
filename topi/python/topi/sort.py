import tvm
from tvm import api

@tvm.target.generic_func
def argsort(data, valid_count, axis=-1, is_ascend=1, flag=0):
    data_buf = api.decl_buffer(data.shape, data.dtype,
                               "sort_data_buf", data_alignment=8)
    valid_count_buf = api.decl_buffer(valid_count.shape, valid_count.dtype,
                                      "valid_count_buf", data_alignment=4)
    out_buf = api.decl_buffer(data.shape, "int32",
                              "sort_out_buf", data_alignment=8)
    out = \
        tvm.extern(data.shape,
                   [data, valid_count],
                   lambda ins, outs: tvm.call_packed(
                       "tvm.contrib.sort.argsort", ins[0], ins[1],
                       outs[0], axis, is_ascend, flag),
                   dtype="int32",
                   in_buffers=[data_buf, valid_count_buf],
                   out_buffers=out_buf,
                   name="argsort_cpu")
    return out
