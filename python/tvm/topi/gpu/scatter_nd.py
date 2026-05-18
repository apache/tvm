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
# ruff: noqa: E741
"""scatter_nd related operators"""

import tvm
from tvm import te, tirx  # hide redefinition of min and max
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tirx as T

from ..math import cast
from ..scatter import _verify_scatter_nd_inputs
from ..utils import ceil_div


def scatter_nd(data, indices, updates, mode):
    """GPU implementation of scatter_nd with explicit thread bindings."""
    _verify_scatter_nd_inputs(data, indices, updates)

    def gen_ir(data_ptr, indices_ptr, updates_ptr, out_ptr):
        # pylint: disable=invalid-name
        data = T.buffer_proxy(data_ptr)
        indices = T.buffer_proxy(indices_ptr)
        updates = T.buffer_proxy(updates_ptr)
        out = T.buffer_proxy(out_ptr)

        # We combine all the indices dimensions but the first one into a single
        # dimension so we can iterate it in single loop instead of an arbitrary
        # number of loops. We do the same thing for all the update dimensions.
        fused_indices_dimension = 1
        for i in indices_ptr.shape[1:]:
            fused_indices_dimension *= i

        fused_updates_dimension = 1
        for i in updates_ptr.shape[len(indices_ptr.shape) - 1 :]:
            fused_updates_dimension *= i

        fused_shape = 1
        for i in data_ptr.shape:
            fused_shape *= i

        max_threads = int(tvm.target.Target.current(allow_none=False).attrs["max_num_threads"])

        with IRBuilder() as ib:
            with T.seq_scope():
                # Init
                nthread_bx_init = cast(ceil_div(fused_shape, max_threads), "int32")
                tx_init = te.thread_axis("threadIdx.x")
                bx_init = te.thread_axis("blockIdx.x")
                with T.frame_scope(
                    [
                        T.attr(bx_init, "thread_extent", nthread_bx_init),
                        T.attr(tx_init, "thread_extent", max_threads),
                    ]
                ):
                    tid = bx_init * max_threads + tx_init
                    with T.If(tid < fused_shape):
                        with T.Then():
                            out[tid] = data[tid]

                # Scatter
                nthread_bx_scat = cast(ceil_div(fused_updates_dimension, max_threads), "int32")
                tx_scat = te.thread_axis("threadIdx.x")
                bx_scat = te.thread_axis("blockIdx.x")
                with T.frame_scope(
                    [
                        T.attr(bx_scat, "thread_extent", nthread_bx_scat),
                        T.attr(tx_scat, "thread_extent", max_threads),
                    ]
                ):
                    j = bx_scat * max_threads + tx_scat
                    with T.If(j < fused_updates_dimension):
                        with T.Then():
                            with T.serial(0, fused_indices_dimension) as i:
                                offset = fused_updates_dimension
                                index = j  # x_M, .. x_{N-1} part of the index into out.
                                # Build up the indices[0, y_0, ..], ..,
                                # indices[M-1, y_0, ..] part of the index into out.
                                for l in reversed(range(indices_ptr.shape[0].value)):
                                    # indices[l, y_0, ... y_{k-1}]
                                    index += offset * indices[i + l * fused_indices_dimension]
                                    offset *= data_ptr.shape[l]
                                if mode == "update":
                                    out[index] = updates[i * fused_updates_dimension + j]
                                elif mode == "add":
                                    out[index] += updates[i * fused_updates_dimension + j]
                                elif mode == "mul":
                                    out[index] *= updates[i * fused_updates_dimension + j]
                                elif mode == "min":
                                    out[index] = tirx.min(
                                        out[index], updates[i * fused_updates_dimension + j]
                                    )
                                elif mode == "max":
                                    out[index] = tirx.max(
                                        out[index], updates[i * fused_updates_dimension + j]
                                    )
                                else:
                                    raise NotImplementedError(
                                        "scatter_nd mode not in [update, add, mul, min, max]:",
                                        mode,
                                    )

            return ib.get()

    out_buf = tirx.decl_buffer(data.shape, data.dtype, "out_buf", layout=None)
    return te.extern(
        [data.shape],
        [data, indices, updates],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0]),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="scatter_nd.gpu",
        tag="scatter_nd.gpu",
    )
