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
# pylint: disable=invalid-name,unused-variable,unused-argument,invalid-name
"concatenate related operators"
from typing import Optional
import tvm
from tvm import te
from ..utils import get_const_int, const_vector
import numpy as np

def _concat(a_tuple, axis=0):
    """Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    a_tuple : tuple of tvm.te.Tensor
        The arrays to concatenate

    axis : int, optional
        The axis along which the arrays will be joined. Default is 0.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    def gen_ir_1D(data_bufs, in_outers_tensor, out_buf):
        ib = tvm.tir.ir_builder.create()
        data_bufs1 = [ib.buffer_ptr(data_buf) for data_buf in data_bufs]
        out_buf    = ib.buffer_ptr(out_buf)
        outers     = ib.buffer_ptr(in_outers_tensor)
        pos = 0
        for i in range(len(a_tuple)):
            with ib.for_range(0, outers[i], name="j") as j:
                out_buf[pos + j] = data_bufs1[i][j]
            pos += outers[i]
        return ib.get()

    def gen_ir(data_bufs, in_outers_tensor, out_buf, inner, outer):
        ib = tvm.tir.ir_builder.create()
        data_bufs1 = [ib.buffer_ptr(data_buf) for data_buf in data_bufs]
        out_buf    = ib.buffer_ptr(out_buf)
        outers     = ib.buffer_ptr(in_outers_tensor)
        if inner > 1:
            with ib.for_range(0, inner, name="inn", kind="parallel") as inn:
                pos = inn * outer
                for i in range(len(a_tuple)):
                    offset = inn * outers[i]
                    with ib.for_range(0, outers[i], name="j") as j:
                        out_buf[pos + j] = data_bufs1[i][offset + j]
                    pos += outers[i]
        else:
            pos = 0
            for i in range(len(a_tuple)):
                with ib.for_range(0, outers[i], name="j", kind="parallel") as j:
                    out_buf[pos + j] = data_bufs1[i][j]
                pos += outers[i]
        return ib.get()

    if axis < 0:
        axis += len(a_tuple[0].shape)
    concat_axis_sizes = [int(t.shape[axis]) for t in a_tuple]
    join_size = int(np.sum(concat_axis_sizes))
    in_outers = [int(np.prod(i.shape[axis:])) for i in a_tuple]
    dtype = a_tuple[0].dtype
    out_shape = a_tuple[0].shape[:axis] + [join_size] + a_tuple[0].shape[axis+1:]
    in_outers_tensor = const_vector(in_outers)
    # check if dimensions tail is (... , axis, 1, ... , 1)
    if len(out_shape[axis + 1:]) == 0:
        rightVal = out_shape[axis]
    else:
        rightVal = np.prod(out_shape[axis + 1:])
    # check if dimensions tail is (1 , ... , 1, axis, ...)
    if len(out_shape[:axis]) == 0:
        leftVal = out_shape[axis]
    else:
        leftVal = np.prod(out_shape[:axis])

    if len(a_tuple[0].shape) == 1 or \
       rightVal == 1 or \
       (leftVal == 1 and axis == len(a_tuple[0].shape) - 1) or \
       (leftVal == 1 and rightVal == 1):
        # badly parallelized case
        return te.extern(
            [out_shape],
            list(a_tuple) + [in_outers_tensor],
            lambda ins, outs: gen_ir_1D(ins, ins[-1], outs[0]),
            dtype=dtype,
            name="concatenate_ext",
        )
    inner = get_const_int(int(np.prod(out_shape[:axis])))
    outer = get_const_int(int(np.prod(out_shape[axis:])))
    return te.extern(
        [out_shape],
        list(a_tuple) + [in_outers_tensor],
        lambda ins, outs: gen_ir(ins, ins[-1], outs[0] , inner, outer),
        dtype=dtype,
        name="concatenate_ext",
    )

def concatenate(
    data: tvm.te.Tensor,
    axis: Optional[int] = 0
):
    """Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    a_tuple : tuple of tvm.te.Tensor
        The arrays to concatenate

    axis : int, optional
        The axis along which the arrays will be joined. Default is 0.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    return _concat(data, axis = axis)
