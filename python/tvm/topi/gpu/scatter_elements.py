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
# pylint: disable=invalid-name
"""scatter_elements related operators"""

import tvm
from tvm import te, tirx
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T

from .. import utils
from ..math import cast
from ..utils import ceil_div


def scatter_elements(data, indices, updates, axis=0, reduction="update"):
    """GPU implementation of scatter_elements with explicit thread bindings"""
    if not isinstance(axis, int):
        axis = utils.get_const_int(axis)

    # Prepare ranges and strides
    shape = data.shape
    if axis < 0:
        axis = len(shape) + axis
    axis_range = cast(shape[axis], indices.dtype)

    full_range = 1
    after_axis_range = 1
    for i, value in enumerate(shape, 0):
        full_range *= value
        if i > axis:
            after_axis_range *= value
    before_axis_stride = axis_range * after_axis_range

    ind_shape = indices.shape
    ind_axis_range = ind_shape[axis]

    ind_before_axis_range = 1
    ind_after_axis_range = 1
    for i, value in enumerate(ind_shape, 0):
        if i < axis:
            ind_before_axis_range *= value
        elif i > axis:
            ind_after_axis_range *= value
    ind_before_axis_stride = ind_axis_range * ind_after_axis_range
    ind_full_range_excl_axis = ind_before_axis_range * ind_after_axis_range

    def gen_ir(data_ptr, indices_ptr, updates_ptr, out_ptr, reduce_func):
        # pylint: disable=invalid-name
        data = T.buffer_proxy(data_ptr)
        indices = T.buffer_proxy(indices_ptr)
        updates = T.buffer_proxy(updates_ptr)
        out = T.buffer_proxy(out_ptr)

        max_threads = int(tvm.target.Target.current(allow_none=False).attrs["max_num_threads"])

        with IRBuilder() as ib:
            with T.seq_scope():
                # Init
                nthread_bx_init = cast(ceil_div(full_range, max_threads), "int32")
                tx_init = te.thread_axis("threadIdx.x")
                bx_init = te.thread_axis("blockIdx.x")
                with T.frame_scope(
                    [
                        T.attr(bx_init, "thread_extent", nthread_bx_init),
                        T.attr(tx_init, "thread_extent", max_threads),
                    ]
                ):
                    tid = bx_init * max_threads + tx_init
                    with T.If(tid < full_range):
                        with T.Then():
                            out[tid] = data[tid]

                # Scatter
                nthread_bx_scat = cast(ceil_div(ind_full_range_excl_axis, max_threads), "int32")
                tx_scat = te.thread_axis("threadIdx.x")
                bx_scat = te.thread_axis("blockIdx.x")
                with T.frame_scope(
                    [
                        T.attr(bx_scat, "thread_extent", nthread_bx_scat),
                        T.attr(tx_scat, "thread_extent", max_threads),
                    ]
                ):
                    fused = bx_scat * max_threads + tx_scat
                    with T.If(fused < ind_full_range_excl_axis):
                        with T.Then():
                            i = fused // ind_after_axis_range
                            j = fused % ind_after_axis_range
                            pre_index1 = i * ind_before_axis_stride + j
                            pre_index2 = i * before_axis_stride + j
                            with T.serial(0, ind_axis_range) as k:
                                # Offset along indices or updates
                                index1 = pre_index1 + k * ind_after_axis_range
                                # Get index and shift to positive side if need
                                k_new = indices[index1]
                                shifted_index = k_new + (k_new < 0) * axis_range
                                # Offset along data
                                index2 = pre_index2 + shifted_index * after_axis_range
                                reduce_func(out, index2, updates[index1])

            return ib.get()

    def update_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] = update

    def add_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] += update

    def mul_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] *= update

    def mean_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] = (dst_ptr[dst_index] + update) / 2

    def min_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] = tirx.min(dst_ptr[dst_index], update)

    def max_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] = tirx.max(dst_ptr[dst_index], update)

    reduce_func = None
    if reduction == "update":
        reduce_func = update_func
    elif reduction == "add":
        reduce_func = add_func
    elif reduction == "mul":
        reduce_func = mul_func
    elif reduction == "mean":
        reduce_func = mean_func
    elif reduction == "min":
        reduce_func = min_func
    elif reduction == "max":
        reduce_func = max_func
    else:
        raise NotImplementedError(
            "scatter_elements reduction not in [update, add, mul, mean, min, max]:", reduction
        )

    out_buf = tirx.decl_buffer(data.shape, data.dtype, "out_buf", layout=None)
    return te.extern(
        [data.shape],
        [data, indices, updates],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0], reduce_func),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="scatter_elements.gpu",
        tag="scatter_elements.gpu",
    )
