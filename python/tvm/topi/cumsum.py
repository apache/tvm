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
"""Cumsum operator"""
from ..tir import decl_buffer, ir_builder
from ..te import extern
from .utils import prod
from .math import cast


def cumsum(data, axis=None, dtype=None):
    if dtype is None:
        dtype = data.dtype

    def maybe_cast(x):
        if dtype != data.dtype:
            return cast(x, dtype)
        return x

    axis_mul_before = 1
    axis_mul_after = 1

    if axis is None and axis != 0:
        axis = 0
        cumsum_axis_len = prod(data.shape)
        shape = (cumsum_axis_len,)
    else:
        shape = data.shape
        cumsum_axis_len = shape[axis]

        if axis < 0:
            axis = len(shape) + axis

        for i, value in enumerate(shape, 0):
            if i < axis:
                axis_mul_before *= value
            elif i > axis:
                axis_mul_after *= value

    def gen_ir(data_buf, out_buf):
        ib = ir_builder.create()
        data_buf = ib.buffer_ptr(data_buf)
        out_buf = ib.buffer_ptr(out_buf)

        with ib.for_range(0, axis_mul_before) as i:
            with ib.for_range(0, axis_mul_after) as j:
                base_idx = i * cumsum_axis_len * axis_mul_after + j
                out_buf[base_idx] = maybe_cast(data_buf[base_idx])
                with ib.for_range(0, cumsum_axis_len - 1) as _k:
                    k = _k + 1
                    cur_idx = base_idx + k * axis_mul_after
                    prev_idx = base_idx + (k - 1) * axis_mul_after
                    out_buf[cur_idx] = out_buf[prev_idx] + maybe_cast(data_buf[cur_idx])

        return ib.get()

    out_buf = decl_buffer(shape, dtype, "out_buf")

    return extern(
        [shape],
        [data],
        lambda ins, outs: gen_ir(ins[0], outs[0]),
        dtype=dtype,
        out_buffers=[out_buf],
        name="cumsum_generic",
        tag="cumsum_generic",
    )
