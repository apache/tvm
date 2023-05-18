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
"concatenate related operators"
from typing import Optional
import numpy as np
import tvm
from tvm import te
from ..utils import get_const_int


def concatenate(data: tvm.te.Tensor, axis: Optional[int] = 0):
    """Join a sequence of arrays along an existing axis.
    Optimized for CPU execution.

    Parameters
    ----------
    data : tuple of tvm.te.Tensor
        The arrays to concatenate

    axis : int, optional
        The axis along which the arrays will be joined. Default is 0.

    Returns
    -------
    ret : tvm.te.Tensor
    """

    in_outers = [int(np.prod(i.shape[axis:])) for i in data]
    in_outers_cumsum = [0, *np.cumsum(in_outers, dtype="int64")[0:-1]]

    def gen_ir_1d(data_bufs, out_buf):
        """Custom concatenation execution."""
        i_b = tvm.tir.ir_builder.create()
        data_bufs1 = [i_b.buffer_ptr(data_buf) for data_buf in data_bufs]
        out_buf = i_b.buffer_ptr(out_buf)

        for i in range(len(data)):
            with i_b.for_range(0, in_outers[i], name="j") as j:
                out_buf[in_outers_cumsum[i] + j] = data_bufs1[i][j]
        return i_b.get()

    def gen_ir(data_bufs, out_buf, inner, outer):
        """Common case of concatenation execution."""
        i_b = tvm.tir.ir_builder.create()
        data_bufs1 = [i_b.buffer_ptr(data_buf) for data_buf in data_bufs]
        out_buf = i_b.buffer_ptr(out_buf)
        if inner > 1:
            with i_b.for_range(0, inner, name="inn", kind="parallel") as inn:
                pos = inn * outer
                for i in range(len(data)):
                    offset = inn * in_outers[i]
                    with i_b.for_range(0, in_outers[i], name="j") as j:
                        out_buf[pos + in_outers_cumsum[i] + j] = data_bufs1[i][offset + j]
        else:
            for i in range(len(data)):
                with i_b.for_range(0, in_outers[i], name="j", kind="parallel") as j:
                    out_buf[in_outers_cumsum[i] + j] = data_bufs1[i][j]
        return i_b.get()

    if axis < 0:
        axis += len(data[0].shape)
    concat_axis_sizes = [int(t.shape[axis]) for t in data]
    join_size = int(np.sum(concat_axis_sizes))

    dtype = data[0].dtype
    out_shape = data[0].shape[:axis] + [join_size] + data[0].shape[axis + 1 :]
    right_val = np.prod(out_shape[axis:])
    left_val = np.prod(out_shape[:axis])

    if (
        len(data[0].shape) == 1
        or (left_val == 1 and axis == len(data[0].shape) - 1)
        or (left_val == 1 and right_val == 1)
    ):
        # badly parallelized case
        return te.extern(
            [out_shape],
            list(data),
            lambda ins, outs: gen_ir_1d(ins, outs[0]),
            dtype=dtype,
            name="concatenate_ext",
        )

    inner = get_const_int(int(left_val))
    outer = get_const_int(int(right_val))
    return te.extern(
        [out_shape],
        list(data),
        lambda ins, outs: gen_ir(ins, outs[0], inner, outer),
        dtype=dtype,
        name="concatenate_ext",
    )
