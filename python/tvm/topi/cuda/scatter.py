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
"""Scatter operator """
import tvm
from tvm import te


def gen_ir_1d(data, indices, updates, axis, out):
    """Generate scatter ir for 1d inputs

    Parameters
    ----------
    data : tir.Tensor
        The input data to the operator.

    indices : tir.Tensor
        The index locations to update.

    updates : tir.Tensor
        The values to update.

    axis : int
        The axis to scatter on

    out : tir.Tensor
        The output tensor.

    Returns
    -------
    ret : tir
        The computational ir.
    """
    assert axis == 0
    n = data.shape[0]

    ib = tvm.tir.ir_builder.create()

    out_ptr = ib.buffer_ptr(out)
    data_ptr = ib.buffer_ptr(data)

    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(bx, "thread_extent", 1)

    with ib.for_range(0, n, name="i") as i:
        out_ptr[i] = data_ptr[i]

    indices_ptr = ib.buffer_ptr(indices)
    updates_ptr = ib.buffer_ptr(updates)
    ni = indices.shape[0]

    with ib.for_range(0, ni, name="i") as i:
        index = indices_ptr[i]
        with ib.if_scope(index < 0):
            out_ptr[index + n] = updates_ptr[i]
        with ib.else_scope():
            out_ptr[index] = updates_ptr[i]

    return ib.get()


def gen_ir_2d(data, indices, updates, axis, out):
    """Generate scatter ir for 2d inputs

    Parameters
    ----------
    data : tir.Tensor
        The input data to the operator.

    indices : tir.Tensor
        The index locations to update.

    updates : tir.Tensor
        The values to update.

    axis : int
        The axis to scatter on

    out : tir.Tensor
        The output tensor.

    Returns
    -------
    ret : tir
        The computational ir.
    """
    n = data.shape[0]
    c = data.shape[1]

    ib = tvm.tir.ir_builder.create()
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(bx, "thread_extent", 1)

    out_ptr = ib.buffer_ptr(out)
    data_ptr = ib.buffer_ptr(data)
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, c, name="j") as j:
            out_ptr[i * c + j] = data_ptr[i * c + j]

    indices_ptr = ib.buffer_ptr(indices)
    updates_ptr = ib.buffer_ptr(updates)
    ni = indices.shape[0]
    ci = indices.shape[1]

    if axis == 0:
        with ib.for_range(0, ni, name="i") as i:
            with ib.for_range(0, ci, name="j") as j:
                index = indices_ptr[i * ci + j]
                with ib.if_scope(index < 0):
                    out_ptr[(index + n) * c + j] = updates_ptr[i * ci + j]
                with ib.else_scope():
                    out_ptr[index * c + j] = updates_ptr[i * ci + j]
    else:
        with ib.for_range(0, ni, name="i") as i:
            with ib.for_range(0, ci, name="j") as j:
                index = indices_ptr[i * ci + j]
                with ib.if_scope(index < 0):
                    out_ptr[i * c + (index + c)] = updates_ptr[i * ci + j]
                with ib.else_scope():
                    out_ptr[i * c + index] = updates_ptr[i * ci + j]

    return ib.get()


def gen_ir_3d(data, indices, updates, axis, out):
    """Generate scatter ir for 3d inputs

    Parameters
    ----------
    data : tir.Tensor
        The input data to the operator.

    indices : tir.Tensor
        The index locations to update.

    updates : tir.Tensor
        The values to update.

    axis : int
        The axis to scatter on

    out : tir.Tensor
        The output tensor.

    Returns
    -------
    ret : tir
        The computational ir.
    """
    n = data.shape[0]
    c = data.shape[1]
    h = data.shape[2]

    ib = tvm.tir.ir_builder.create()
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(bx, "thread_extent", 1)

    out_ptr = ib.buffer_ptr(out)
    data_ptr = ib.buffer_ptr(data)
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, c, name="j") as j:
            with ib.for_range(0, h, name="k") as k:
                out_ptr[(i * c + j) * h + k] = data_ptr[(i * c + j) * h + k]

    indices_ptr = ib.buffer_ptr(indices)
    updates_ptr = ib.buffer_ptr(updates)
    ni = indices.shape[0]
    ci = indices.shape[1]
    hi = indices.shape[2]

    if axis == 0:
        with ib.for_range(0, ni, name="i") as i:
            with ib.for_range(0, ci, name="j") as j:
                with ib.for_range(0, hi, name="k") as k:
                    index = indices_ptr[(i * ci + j) * hi + k]
                    with ib.if_scope(index < 0):
                        out_ptr[((index + n) * c + j) * h + k] = updates_ptr[(i * ci + j) * hi + k]
                    with ib.else_scope():
                        out_ptr[(index * c + j) * h + k] = updates_ptr[(i * ci + j) * hi + k]
    elif axis == 1:
        with ib.for_range(0, ni, name="i") as i:
            with ib.for_range(0, ci, name="j") as j:
                with ib.for_range(0, hi, name="k") as k:
                    index = indices_ptr[(i * ci + j) * hi + k]
                    with ib.if_scope(index < 0):
                        out_ptr[(i * c + (index + c)) * h + k] = updates_ptr[(i * ci + j) * hi + k]
                    with ib.else_scope():
                        out_ptr[(i * c + index) * h + k] = updates_ptr[(i * ci + j) * hi + k]
    else:
        with ib.for_range(0, ni, name="i") as i:
            with ib.for_range(0, ci, name="j") as j:
                with ib.for_range(0, hi, name="k") as k:
                    index = indices_ptr[(i * ci + j) * hi + k]
                    with ib.if_scope(index < 0):
                        out_ptr[(i * c + j) * h + (index + h)] = updates_ptr[(i * ci + j) * hi + k]
                    with ib.else_scope():
                        out_ptr[(i * c + j) * h + index] = updates_ptr[(i * ci + j) * hi + k]

    return ib.get()


def gen_ir_4d(data, indices, updates, axis, out):
    """Generate scatter ir for 4d inputs

    Parameters
    ----------
    data : tir.Tensor
        The input data to the operator.

    indices : tir.Tensor
        The index locations to update.

    updates : tir.Tensor
        The values to update.

    axis : int
        The axis to scatter on

    out : tir.Tensor
        The output tensor.

    Returns
    -------
    ret : tir
        The computational ir.
    """
    n = data.shape[0]
    c = data.shape[1]
    h = data.shape[2]
    w = data.shape[3]

    ib = tvm.tir.ir_builder.create()
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(bx, "thread_extent", 1)

    out_ptr = ib.buffer_ptr(out)
    data_ptr = ib.buffer_ptr(data)
    with ib.for_range(0, n, name="i") as i:
        with ib.for_range(0, c, name="j") as j:
            with ib.for_range(0, h, name="k") as k:
                with ib.for_range(0, w, name="l") as l:
                    out_ptr[((i * c + j) * h + k) * w + l] = data_ptr[((i * c + j) * h + k) * w + l]

    indices_ptr = ib.buffer_ptr(indices)
    updates_ptr = ib.buffer_ptr(updates)
    ni = indices.shape[0]
    ci = indices.shape[1]
    hi = indices.shape[2]
    wi = indices.shape[3]

    if axis == 0:
        with ib.for_range(0, ni, name="i") as i:
            with ib.for_range(0, ci, name="j") as j:
                with ib.for_range(0, hi, name="k") as k:
                    with ib.for_range(0, wi, name="l") as l:
                        index = indices_ptr[((i * ci + j) * hi + k) * wi + l]
                        with ib.if_scope(index < 0):
                            out_ptr[(((index + n) * c + j) * h + k) * w + l] = updates_ptr[
                                ((i * ci + j) * hi + k) * wi + l
                            ]
                        with ib.else_scope():
                            out_ptr[((index * c + j) * h + k) * w + l] = updates_ptr[
                                ((i * ci + j) * hi + k) * wi + l
                            ]
    elif axis == 1:
        with ib.for_range(0, ni, name="i") as i:
            with ib.for_range(0, ci, name="j") as j:
                with ib.for_range(0, hi, name="k") as k:
                    with ib.for_range(0, wi, name="l") as l:
                        index = indices_ptr[((i * ci + j) * hi + k) * wi + l]
                        with ib.if_scope(index < 0):
                            out_ptr[((i * c + (index + c)) * h + k) * w + l] = updates_ptr[
                                ((i * ci + j) * hi + k) * wi + l
                            ]
                        with ib.else_scope():
                            out_ptr[((i * c + index) * h + k) * w + l] = updates_ptr[
                                ((i * ci + j) * hi + k) * wi + l
                            ]
    elif axis == 2:
        with ib.for_range(0, ni, name="i") as i:
            with ib.for_range(0, ci, name="j") as j:
                with ib.for_range(0, hi, name="k") as k:
                    with ib.for_range(0, wi, name="l") as l:
                        index = indices_ptr[((i * ci + j) * hi + k) * wi + l]
                        with ib.if_scope(index < 0):
                            out_ptr[((i * c + j) * h + (index + h)) * w + l] = updates_ptr[
                                ((i * ci + j) * hi + k) * wi + l
                            ]
                        with ib.else_scope():
                            out_ptr[((i * c + j) * h + index) * w + l] = updates_ptr[
                                ((i * ci + j) * hi + k) * wi + l
                            ]
    else:
        with ib.for_range(0, ni, name="i") as i:
            with ib.for_range(0, ci, name="j") as j:
                with ib.for_range(0, hi, name="k") as k:
                    with ib.for_range(0, wi, name="l") as l:
                        index = indices_ptr[((i * ci + j) * hi + k) * wi + l]
                        with ib.if_scope(index < 0):
                            out_ptr[((i * c + j) * h + k) * w + (index + w)] = updates_ptr[
                                ((i * ci + j) * hi + k) * wi + l
                            ]
                        with ib.else_scope():
                            out_ptr[((i * c + j) * h + k) * w + index] = updates_ptr[
                                ((i * ci + j) * hi + k) * wi + l
                            ]

    return ib.get()


def scatter(data, indices, updates, axis=0):
    """Update data at positions defined by indices with values in updates

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    indices : relay.Expr
        The index locations to update.

    updates : relay.Expr
        The values to update.

    axis : int
        The axis to scatter on

    Returns
    -------
    ret : relay.Expr
        The computed result.
    """
    if axis < 0:
        axis += len(data.shape)
    assert axis >= 0
    assert axis < len(data.shape)

    rank = len(data.shape)
    assert 1 <= rank <= 4, "scatter only supports 1-4 dimensions"

    ir_funcs = {
        1: gen_ir_1d,
        2: gen_ir_2d,
        3: gen_ir_3d,
        4: gen_ir_4d,
    }

    out_shape = data.shape
    out_buf = tvm.tir.decl_buffer(out_shape, data.dtype, "out_buf")
    out = te.extern(
        [out_shape],
        [data, indices, updates],
        lambda ins, outs: ir_funcs[rank](ins[0], ins[1], ins[2], axis, outs[0]),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="scatter_gpu",
        tag="scatter_gpu",
    )

    return out
