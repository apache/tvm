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
from ..scatter import _verify_scatter_nd_inputs


def ceil_div(a, b):
    return (a + b - 1) // b


def gen_ir_1d(data, indices, updates, axis, out, update_func):
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

    update_func: function
        The function to be applied to a destination and the corresponding update.

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

    with ib.new_scope():
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(bx, "thread_extent", n)
        out_ptr[bx] = data_ptr[bx]

    indices_ptr = ib.buffer_ptr(indices)
    updates_ptr = ib.buffer_ptr(updates)

    ni = indices.shape[0]

    with ib.new_scope():
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(bx, "thread_extent", 1)
        with ib.for_range(0, ni, name="i") as i:
            index = indices_ptr[i]
            with ib.if_scope(index < 0):
                update_func(out_ptr, index + n, updates_ptr[i])
            with ib.else_scope():
                update_func(out_ptr, index, updates_ptr[i])

    return ib.get()


def gen_ir_2d(data, indices, updates, axis, out, update_func):
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

    update_func: function
        The function to be applied to a destination and the corresponding update

    Returns
    -------
    ret : tir
        The computational ir.
    """
    warp_size = tvm.target.Target.current(False).thread_warp_size

    n = data.shape[0]
    c = data.shape[1]

    ib = tvm.tir.ir_builder.create()

    out_ptr = ib.buffer_ptr(out)
    data_ptr = ib.buffer_ptr(data)

    with ib.new_scope():
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(bx, "thread_extent", n)
        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(tx, "thread_extent", warp_size)
        with ib.for_range(0, ceil_div(c, warp_size), name="j") as j_:
            j = j_ * warp_size + tx
            with ib.if_scope(j < c):
                idx = bx * c + j
                out_ptr[idx] = data_ptr[idx]

    indices_ptr = ib.buffer_ptr(indices)
    updates_ptr = ib.buffer_ptr(updates)

    ni = indices.shape[0]
    ci = indices.shape[1]

    if axis == 0:
        with ib.new_scope():
            j = te.thread_axis("blockIdx.x")
            ib.scope_attr(j, "thread_extent", ci)
            with ib.for_range(0, ni, name="i") as i:
                idx = i * ci + j
                index = indices_ptr[idx]
                with ib.if_scope(index < 0):
                    update_func(out_ptr, (index + n) * c + j, updates_ptr[idx])
                with ib.else_scope():
                    update_func(out_ptr, index * c + j, updates_ptr[idx])
    else:
        with ib.new_scope():
            i = te.thread_axis("blockIdx.x")
            ib.scope_attr(i, "thread_extent", ni)
            with ib.for_range(0, ci, name="j") as j:
                idx = i * ci + j
                index = indices_ptr[idx]
                with ib.if_scope(index < 0):
                    update_func(out_ptr, i * c + (index + c), updates_ptr[idx])
                with ib.else_scope():
                    update_func(out_ptr, i * c + index, updates_ptr[idx])
    return ib.get()


def gen_ir_3d(data, indices, updates, axis, out, update_func):
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

    update_func: function
        The function to be applied to a destination and the corresponding update

    Returns
    -------
    ret : tir
        The computational ir.
    """
    warp_size = tvm.target.Target.current(False).thread_warp_size

    n = data.shape[0]
    c = data.shape[1]
    h = data.shape[2]

    ib = tvm.tir.ir_builder.create()

    out_ptr = ib.buffer_ptr(out)
    data_ptr = ib.buffer_ptr(data)

    with ib.new_scope():
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(bx, "thread_extent", n)
        by = te.thread_axis("blockIdx.y")
        ib.scope_attr(by, "thread_extent", c)
        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(tx, "thread_extent", warp_size)
        with ib.for_range(0, ceil_div(h, warp_size), name="k") as k_:
            k = k_ * warp_size + tx
            with ib.if_scope(k < h):
                idx = (bx * c + by) * h + k
                out_ptr[idx] = data_ptr[idx]

    indices_ptr = ib.buffer_ptr(indices)
    updates_ptr = ib.buffer_ptr(updates)
    ni = indices.shape[0]
    ci = indices.shape[1]
    hi = indices.shape[2]

    if axis == 0:
        with ib.new_scope():
            j = te.thread_axis("blockIdx.x")
            ib.scope_attr(j, "thread_extent", ci)
            tx = te.thread_axis("threadIdx.x")
            ib.scope_attr(tx, "thread_extent", warp_size)
            with ib.for_range(0, ni, name="i") as i:
                with ib.for_range(0, ceil_div(hi, warp_size), name="k") as k_:
                    k = k_ * warp_size + tx
                    with ib.if_scope(k < hi):
                        idx = (i * ci + j) * hi + k
                        index = indices_ptr[idx]
                        with ib.if_scope(index < 0):
                            update_func(out_ptr, ((index + n) * c + j) * h + k, updates_ptr[idx])
                        with ib.else_scope():
                            update_func(out_ptr, (index * c + j) * h + k, updates_ptr[idx])
    elif axis == 1:
        with ib.new_scope():
            i = te.thread_axis("blockIdx.x")
            ib.scope_attr(i, "thread_extent", ni)
            tx = te.thread_axis("threadIdx.x")
            ib.scope_attr(tx, "thread_extent", warp_size)
            with ib.for_range(0, ci, name="j") as j:
                with ib.for_range(0, ceil_div(hi, warp_size), name="k") as k_:
                    k = k_ * warp_size + tx
                    with ib.if_scope(k < hi):
                        idx = (i * ci + j) * hi + k
                        index = indices_ptr[idx]
                        with ib.if_scope(index < 0):
                            update_func(out_ptr, (i * c + (index + c)) * h + k, updates_ptr[idx])
                        with ib.else_scope():
                            update_func(out_ptr, (i * c + index) * h + k, updates_ptr[idx])
    else:
        with ib.new_scope():
            i = te.thread_axis("blockIdx.x")
            ib.scope_attr(i, "thread_extent", ni)
            j = te.thread_axis("blockIdx.y")
            ib.scope_attr(j, "thread_extent", ci)
            with ib.for_range(0, hi, name="k") as k:
                idx = (i * ci + j) * hi + k
                index = indices_ptr[idx]
                with ib.if_scope(index < 0):
                    update_func(out_ptr, (i * c + j) * h + (index + h), updates_ptr[idx])
                with ib.else_scope():
                    update_func(out_ptr, (i * c + j) * h + index, updates_ptr[idx])
    return ib.get()


def gen_ir_4d(data, indices, updates, axis, out, update_func):
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

    update_func: function
        The function to be applied to a destination and the corresponding update

    Returns
    -------
    ret : tir
        The computational ir.
    """
    warp_size = tvm.target.Target.current(False).thread_warp_size

    n = data.shape[0]
    c = data.shape[1]
    h = data.shape[2]
    w = data.shape[3]

    ib = tvm.tir.ir_builder.create()

    out_ptr = ib.buffer_ptr(out)
    data_ptr = ib.buffer_ptr(data)
    with ib.new_scope():
        i = te.thread_axis("blockIdx.x")
        ib.scope_attr(i, "thread_extent", n)
        j = te.thread_axis("blockIdx.y")
        ib.scope_attr(j, "thread_extent", c)
        k = te.thread_axis("blockIdx.z")
        ib.scope_attr(k, "thread_extent", h)
        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(tx, "thread_extent", warp_size)
        with ib.for_range(0, ceil_div(w, warp_size), name="l") as l_:
            l = l_ * warp_size + tx
            with ib.if_scope(l < w):
                idx = ((i * c + j) * h + k) * w + l
                out_ptr[idx] = data_ptr[idx]

    indices_ptr = ib.buffer_ptr(indices)
    updates_ptr = ib.buffer_ptr(updates)
    ni = indices.shape[0]
    ci = indices.shape[1]
    hi = indices.shape[2]
    wi = indices.shape[3]

    if axis == 0:
        with ib.new_scope():
            j = te.thread_axis("blockIdx.y")
            ib.scope_attr(j, "thread_extent", ci)
            k = te.thread_axis("blockIdx.z")
            ib.scope_attr(k, "thread_extent", hi)
            tx = te.thread_axis("threadIdx.x")
            ib.scope_attr(tx, "thread_extent", warp_size)
            with ib.for_range(0, ni, name="i") as i:
                with ib.for_range(0, ceil_div(wi, warp_size), name="l") as l_:
                    l = l_ * warp_size + tx
                    with ib.if_scope(l < wi):
                        idx = ((i * ci + j) * hi + k) * wi + l
                        index = indices_ptr[idx]
                        with ib.if_scope(index < 0):
                            update_func(
                                out_ptr, (((index + n) * c + j) * h + k) * w + l, updates_ptr[idx]
                            )
                        with ib.else_scope():
                            update_func(
                                out_ptr, ((index * c + j) * h + k) * w + l, updates_ptr[idx]
                            )
    elif axis == 1:
        with ib.new_scope():
            i = te.thread_axis("blockIdx.x")
            ib.scope_attr(i, "thread_extent", ni)
            k = te.thread_axis("blockIdx.z")
            ib.scope_attr(k, "thread_extent", hi)
            tx = te.thread_axis("threadIdx.x")
            ib.scope_attr(tx, "thread_extent", warp_size)
            with ib.for_range(0, ci, name="j") as j:
                with ib.for_range(0, ceil_div(wi, warp_size), name="l") as l_:
                    l = l_ * warp_size + tx
                    with ib.if_scope(l < wi):
                        idx = ((i * ci + j) * hi + k) * wi + l
                        index = indices_ptr[idx]
                        with ib.if_scope(index < 0):
                            update_func(
                                out_ptr, ((i * c + (index + c)) * h + k) * w + l, updates_ptr[idx]
                            )
                        with ib.else_scope():
                            update_func(
                                out_ptr, ((i * c + index) * h + k) * w + l, updates_ptr[idx]
                            )
    elif axis == 2:
        with ib.new_scope():
            i = te.thread_axis("blockIdx.x")
            ib.scope_attr(i, "thread_extent", ni)
            j = te.thread_axis("blockIdx.y")
            ib.scope_attr(j, "thread_extent", ci)
            tx = te.thread_axis("threadIdx.x")
            ib.scope_attr(tx, "thread_extent", warp_size)
            with ib.for_range(0, hi, name="k") as k:
                with ib.for_range(0, ceil_div(wi, warp_size), name="l") as l_:
                    l = l_ * warp_size + tx
                    with ib.if_scope(l < wi):
                        idx = ((i * ci + j) * hi + k) * wi + l
                        index = indices_ptr[idx]
                        with ib.if_scope(index < 0):
                            update_func(
                                out_ptr, ((i * c + j) * h + (index + h)) * w + l, updates_ptr[idx]
                            )
                        with ib.else_scope():
                            update_func(
                                out_ptr, ((i * c + j) * h + index) * w + l, updates_ptr[idx]
                            )
    else:
        with ib.new_scope():
            i = te.thread_axis("blockIdx.x")
            ib.scope_attr(i, "thread_extent", ni)
            j = te.thread_axis("blockIdx.y")
            ib.scope_attr(j, "thread_extent", ci)
            k = te.thread_axis("blockIdx.z")
            ib.scope_attr(k, "thread_extent", hi)
            with ib.for_range(0, wi, name="l") as l:
                idx = ((i * ci + j) * hi + k) * wi + l
                index = indices_ptr[idx]
                with ib.if_scope(index < 0):
                    update_func(out_ptr, ((i * c + j) * h + k) * w + (index + w), updates_ptr[idx])
                with ib.else_scope():
                    update_func(out_ptr, ((i * c + j) * h + k) * w + index, updates_ptr[idx])
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

    def update_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] = update

    out_shape = data.shape
    out_buf = tvm.tir.decl_buffer(out_shape, data.dtype, "out_buf")
    out = te.extern(
        [out_shape],
        [data, indices, updates],
        lambda ins, outs: ir_funcs[rank](ins[0], ins[1], ins[2], axis, outs[0], update_func),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="scatter_gpu",
        tag="scatter_gpu",
    )

    return out


def scatter_add(data, indices, updates, axis=0):
    """Update data by adding values in updates at positions defined by indices

    Parameters
    ----------
    data : relay.Expr
        The input data to the operator.

    indices : relay.Expr
        The index locations to update.

    updates : relay.Expr
        The values to be added.

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
    assert 1 <= rank <= 4, "scatter_add only supports 1-4 dimensions"

    ir_funcs = {
        1: gen_ir_1d,
        2: gen_ir_2d,
        3: gen_ir_3d,
        4: gen_ir_4d,
    }

    def update_func(dst_ptr, dst_index, update):
        dst_ptr[dst_index] += update

    out_shape = data.shape
    out_buf = tvm.tir.decl_buffer(out_shape, data.dtype, "out_buf")
    out = te.extern(
        [out_shape],
        [data, indices, updates],
        lambda ins, outs: ir_funcs[rank](ins[0], ins[1], ins[2], axis, outs[0], update_func),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="scatter_add_gpu",
        tag="scatter_add_gpu",
    )

    return out


def scatter_nd(data, indices, shape):
    """Scatter elements from a n-dimension array.

    Given data with shape (Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1}), indices with shape
    (M, Y_0, ..., Y_{K-1}), and output with shape (X_0, X_1, ..., X_{N-1}), scatter_nd computes

    .. code-block::

        output[indices[0, y_0, ..., y_{K-1}],
               ...,
               indices[M-1, y_0, ..., y_{K-1}],
               x_M,
               ...,
               x_{N-1}
              ] = data[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}]

    all other entries in the output are 0. Repeated indices are summed.

    Parameters
    ----------
    data : tvm.te.Tensor
        The source array.

    indices : tvm.te.Tensor
        The indices of the values to extract.

    shape : Sequence[int]
        The output shape. This must be specified because it cannot be inferred.

    Returns
    -------
    ret : tvm.te.Tensor
    """
    _verify_scatter_nd_inputs(data, indices, shape)

    def gen_ir(data_ptr, indices_ptr, out_ptr):
        ib = tvm.tir.ir_builder.create()

        data = ib.buffer_ptr(data_ptr)
        indices = ib.buffer_ptr(indices_ptr)
        out = ib.buffer_ptr(out_ptr)

        # We combine all the indices dimensions but the first one into a single
        # dimension so we can iterate it in single loop instead of an arbitrary
        # number of loops. We do the same thing for all the data dimensions.
        fused_indices_dimension = 1
        for i in indices_ptr.shape[1:]:
            fused_indices_dimension *= i

        fused_data_dimension = 1
        for i in data_ptr.shape[len(indices_ptr.shape) - 1 :]:
            fused_data_dimension *= i

        fused_shape = 1
        for i in shape:
            fused_shape *= i

        # For now we avoid parallizing over dimensions indexed by `indices` as
        # there may be repeated indices and hadling parallel accumulation can
        # be hard. So we parallelize over X_M .. X_{N-1} instead. This will
        # work well when these dimensions are large enough to saturate memory
        # bandwidth, but performance will be bad when these dimensions are
        # small.
        bx = te.thread_axis("blockIdx.x")
        tx = te.thread_axis("threadIdx.x")
        max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
        tdim = min(max_threads, fused_data_dimension)
        ib.scope_attr(tx, "thread_extent", tdim)
        bdim = ceil_div(fused_data_dimension, tdim)
        ib.scope_attr(bx, "thread_extent", bdim)

        # zero data
        # TODO(tkonolige): could we use topi.full to zero it instead?
        with ib.for_range(0, ceil_div(fused_shape, bdim)) as i:
            index = i * fused_data_dimension + bx * tdim + tx
            with ib.if_scope(index < fused_shape):
                out[index] = tvm.tir.Cast(data_ptr.dtype, 0)

        with ib.for_range(0, fused_indices_dimension) as i:
            j = bx * tdim + tx
            with ib.if_scope(j < fused_data_dimension):
                offset = fused_data_dimension
                index = j  # This is x_M, .. x_{N-1} part of the index into out.
                # Build up the indices[0, y_0, .. y_{K-1}], .. indices[M-1, y_0, .. y_{K-1}] part
                # of the index into out.
                for l in reversed(range(indices_ptr.shape[0].value)):
                    # indices[i * l * fused_indices_dimension] = indices[l, y_0, ... y_{k-1}]
                    index += offset * indices[i + l * fused_indices_dimension]
                    offset *= shape[l]
                out[index] += data[i * fused_data_dimension + j]

        return ib.get()

    out_buf = tvm.tir.decl_buffer(shape, data.dtype, "out_buf")
    return te.extern(
        [shape],
        [data, indices],
        lambda ins, outs: gen_ir(ins[0], ins[1], outs[0]),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="scatter_nd_cuda",
        tag="scatter_nd_cuda",
    )
