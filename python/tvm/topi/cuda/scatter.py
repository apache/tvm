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
from tvm import te, autotvm
from ..scatter import _verify_scatter_nd_inputs
from ..generic import schedule_extern
from .nms import atomic_add
from .sort import stable_sort_by_key_thrust
from ..utils import prod, ceil_div


def _memcpy_ir(ib, out_ptr, data_ptr, shape):
    fused = prod(shape)
    with ib.new_scope():
        num_thread = int(tvm.target.Target.current(allow_none=False).max_num_threads)
        num_blocks = ceil_div(fused, num_thread)
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(bx, "thread_extent", num_blocks)
        tx = te.thread_axis("threadIdx.x")
        ib.scope_attr(tx, "thread_extent", num_thread)
        tid = bx * num_thread + tx

        with ib.if_scope(tid < fused):
            out_ptr[tid] = data_ptr[tid]


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

    _memcpy_ir(ib, out_ptr, data_ptr, data.shape)

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
    n = data.shape[0]
    c = data.shape[1]

    ib = tvm.tir.ir_builder.create()

    out_ptr = ib.buffer_ptr(out)
    data_ptr = ib.buffer_ptr(data)

    _memcpy_ir(ib, out_ptr, data_ptr, data.shape)

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

    _memcpy_ir(ib, out_ptr, data_ptr, data.shape)

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
    _memcpy_ir(ib, out_ptr, data_ptr, data.shape)

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


@autotvm.register_topi_compute("scatter.cuda")
def scatter(cfg, data, indices, updates, axis=0):
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

    cfg.add_flop(1)  # A dummy value to satisfy AutoTVM

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


@autotvm.register_topi_schedule("scatter.cuda")
def schedule_scatter(_, outs):
    return schedule_extern(outs)


def gen_scatter_1d_thrust(data, indices_sorted, updates_sorted, out):
    """Generate scatter ir for 1d inputs, using a sorting based approach.
    By sorting indices and comparing neighboring two indices, we can tell which
    of elements in the indices tensor can scatter its update value into the output.
    Sorting of indices, and sorting of updates with respect to indices, can be done
    at the same time by thrust's sort_by_key function. It is important that sorting
    be done in a "stable" way via stable_sort, to guarantee deterministic output.
    Negative indices are assumed to have been converted to corresponding positive
    indices.

    Parameters
    ----------
    data : tir.Tensor
        The input data to the operator.

    indices_sorted : tir.Tensor
        The sorted index locations to update.

    updates : tir.Tensor
        The values to update, sorted by indices.

    out : tir.Tensor
        The output tensor.

    Returns
    -------
    ret : tir
        The computational ir.
    """
    n = data.shape[0]

    ib = tvm.tir.ir_builder.create()

    out_ptr = ib.buffer_ptr(out)
    data_ptr = ib.buffer_ptr(data)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_tx = max_threads

    with ib.new_scope():
        nthread_bx = ceil_div(n, nthread_tx)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * nthread_tx + tx
        with ib.if_scope(tid < n):
            out_ptr[tid] = data_ptr[tid]

    indices_ptr = ib.buffer_ptr(indices_sorted)
    updates_ptr = ib.buffer_ptr(updates_sorted)

    ni = indices_sorted.shape[0]

    with ib.new_scope():
        nthread_bx = ceil_div(ni, nthread_tx)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * nthread_tx + tx

        with ib.if_scope(tid == ni - 1):
            # The last element can always update.
            index = indices_ptr[tid]
            update = updates_ptr[tid]
            out_ptr[index] = update

        with ib.else_scope():
            with ib.if_scope(tid < ni - 1):
                index = indices_ptr[tid]
                index_next = indices_ptr[tid + 1]

                # If the next neighbor in the sorted list of indices has a different index,
                # that means thread tid is the last one to have this index.
                # This thread can update the output.
                with ib.if_scope(index != index_next):
                    update = updates_ptr[tid]
                    out_ptr[index] = update

    return ib.get()


@autotvm.register_topi_compute("scatter_via_sort.cuda")
def scatter_via_sort(cfg, data, indices, updates, axis=0):
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
    assert axis == 0 and len(data.shape) == 1, "sorting based scatter only supported for 1d input"

    cfg.add_flop(1)  # A dummy value to satisfy AutoTVM

    out_shape = data.shape
    out_buf = tvm.tir.decl_buffer(out_shape, data.dtype, "out_buf")

    indices_sorted, updates_sorted = stable_sort_by_key_thrust(indices, updates, for_scatter=True)

    out = te.extern(
        [out_shape],
        [data, indices_sorted, updates_sorted],
        lambda ins, outs: gen_scatter_1d_thrust(ins[0], ins[1], ins[2], outs[0]),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="scatter_via_sort_gpu",
        tag="scatter_via_sort_gpu",
    )

    return out


@autotvm.register_topi_schedule("scatter_via_sort.cuda")
def schedule_scatter_via_sort(_, outs):
    return schedule_extern(outs)


def gen_scatter_add_1d_atomic(data, indices, updates, axis, out, _):
    """Generate scatter add ir for 1d inputs, using atomic_add instruction

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

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_tx = max_threads

    with ib.new_scope():
        nthread_bx = ceil_div(n, nthread_tx)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * nthread_tx + tx
        with ib.if_scope(tid < n):
            out_ptr[tid] = data_ptr[tid]

    indices_ptr = ib.buffer_ptr(indices)
    updates_ptr = ib.buffer_ptr(updates)

    ni = indices.shape[0]

    atomic_add_return = ib.allocate(updates.dtype, (1,), name="atomic_add_return", scope="local")

    with ib.new_scope():
        nthread_bx = ceil_div(ni, nthread_tx)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * nthread_tx + tx

        with ib.if_scope(tid < ni):
            index = indices_ptr[tid]
            with ib.if_scope(index < 0):
                atomic_add_return[0] = atomic_add(
                    tvm.tir.call_intrin("handle", "tir.address_of", out_ptr[index + n]),
                    updates_ptr[tid],
                )
            with ib.else_scope():
                atomic_add_return[0] = atomic_add(
                    tvm.tir.call_intrin("handle", "tir.address_of", out_ptr[index]),
                    updates_ptr[tid],
                )

    return ib.get()


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
        1: gen_scatter_add_1d_atomic,
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


def scatter_nd(data, indices, updates, mode):
    """Scatter elements from a n-dimension array.

    Given updates with shape (Y_0, ..., Y_{K-1}, X_M, ..., X_{N-1}), indices with shape
    (M, Y_0, ..., Y_{K-1}), and output copied from data with shape (X_0, X_1, ..., X_{N-1}),
    scatter_nd computes

    .. code-block::

        output[indices[0, y_0, ..., y_{K-1}],
               ...,
               indices[M-1, y_0, ..., y_{K-1}],
               x_M,
               ...,
               x_{N-1}
              ] = f(output[...], updates[y_0, ..., y_{K-1}, x_M, ..., x_{N-1}])

    where the update function f is determinted by the mode.

    Parameters
    ----------
    data : tvm.te.Tensor
        The source array.

    indices : tvm.te.Tensor
        The indices of the values to extract.

    updates : tvm.te.Tensor
        The updates to apply at the Indices

    mode : string
        The update mode for the algorithm, either "update" or "add"
        If update, the update values will replace the input data
        If add, the update values will be added to the input data

    Returns
    -------
    ret : tvm.te.Tensor
    """
    _verify_scatter_nd_inputs(data, indices, updates)

    def gen_ir(data_ptr, indices_ptr, updates_ptr, out_ptr):
        ib = tvm.tir.ir_builder.create()

        data = ib.buffer_ptr(data_ptr)
        indices = ib.buffer_ptr(indices_ptr)
        updates = ib.buffer_ptr(updates_ptr)
        out = ib.buffer_ptr(out_ptr)

        atomic_add_return = ib.allocate(
            updates.dtype, (1,), name="atomic_add_return", scope="local"
        )

        fused_indices_dimension = 1
        for i in indices_ptr.shape[1:]:
            fused_indices_dimension *= i

        fused_updates_dimension = 1
        for i in updates_ptr.shape[len(indices_ptr.shape) - 1 :]:
            fused_updates_dimension *= i

        fused_shape = 1
        for i in data_ptr.shape:
            fused_shape *= i

        max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
        tdim = min(max_threads, fused_updates_dimension)

        with ib.new_scope():
            bdim = ceil_div(fused_shape, tdim)
            bx = te.thread_axis("blockIdx.x")
            tx = te.thread_axis("threadIdx.x")
            ib.scope_attr(bx, "thread_extent", bdim)
            ib.scope_attr(tx, "thread_extent", tdim)

            index = bx * tdim + tx
            with ib.if_scope(index < fused_shape):
                out[index] = data[index]

        # For better performance, we introduce blockIdx.y to implement for-loops
        # within one thread.
        # The code is parallel over the scattered indices, so we use atomic_add
        # to guarantee correctness when mode=="add"

        # For now, atomic is not supported by target "vulkan", "metal", or "cuda" with "int64"
        # So we fallback to normal algorithm, using "+=" rather than atomic_add

        # TODO (CaptainDuke):
        # Since multiple threads compete for the same write index, which leads to
        # non-determinstic output for update mode. We could add a new attribute,
        # "allow_non_deterministic", which can be conditionally set to True by
        # each frontend when non-determinsm is allowed.
        cur_target_kind = str(tvm.target.Target.current(allow_none=False).kind)
        with ib.new_scope():
            if (
                mode == "add"
                and cur_target_kind not in ["vulkan", "metal"]
                and updates.dtype in ["int32", "float32"]
            ):
                bdim_x = fused_indices_dimension
                bdim_y = ceil_div(fused_updates_dimension, tdim)
                # In case of large input sizes, fused_indices_dimension might be too large.
                # So we use blockIdx.x because holds larger scales.
                bx = te.thread_axis("blockIdx.x")
                by = te.thread_axis("blockIdx.y")
                tx = te.thread_axis("threadIdx.x")
                ib.scope_attr(bx, "thread_extent", bdim_x)
                ib.scope_attr(by, "thread_extent", bdim_y)
                ib.scope_attr(tx, "thread_extent", tdim)

                j = by * tdim + tx
                with ib.if_scope(j < fused_updates_dimension):
                    offset = fused_updates_dimension
                    index = j  # This is x_M, .. x_{N-1} part of the index into out.
                    # Build up the indices[0, y_0, .. y_{K-1}], .. indices[M-1, y_0, .. y_{K-1}]
                    # part of the index into out.
                    up_index = bx * fused_updates_dimension + j
                    for l in reversed(range(indices_ptr.shape[0].value)):
                        # indices[bx * l * fused_indices_dimension] = indices[l, y_0, ... y_{k-1}]
                        index += offset * indices[bx + l * fused_indices_dimension]
                        offset *= data_ptr.shape[l]
                    atomic_add_return[0] = atomic_add(
                        tvm.tir.call_intrin("handle", "tir.address_of", out[index]),
                        updates[up_index],
                    )
            else:
                bdim_x = ceil_div(fused_updates_dimension, tdim)
                bx = te.thread_axis("blockIdx.x")
                tx = te.thread_axis("threadIdx.x")
                ib.scope_attr(bx, "thread_extent", bdim_x)
                ib.scope_attr(tx, "thread_extent", tdim)
                with ib.for_range(0, fused_indices_dimension) as i:
                    j = bx * tdim + tx
                    with ib.if_scope(j < fused_updates_dimension):
                        offset = fused_updates_dimension
                        index = j  # This is x_M, .. x_{N-1} part of the index into out.
                        # Build up the
                        # indices[0, y_0, .. y_{K-1}], ... indices[M-1, y_0, .. y_{K-1}]
                        # part of the index into out.
                        for l in reversed(range(indices_ptr.shape[0].value)):
                            # indices[i * l * fused_indices_dimension] = indices[l, y_0,
                            #                                                   ... y_{k-1}]
                            index += offset * indices[i + l * fused_indices_dimension]
                            offset *= data_ptr.shape[l]
                        if mode == "update":
                            out[index] = updates[i * fused_updates_dimension + j]
                        elif mode == "add":
                            out[index] += updates[i * fused_updates_dimension + j]
                        else:
                            raise NotImplementedError("scatter_nd mode not in [update, add]:", mode)

        return ib.get()

    out_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "out_buf")
    return te.extern(
        [data.shape],
        [data, indices, updates],
        lambda ins, outs: gen_ir(ins[0], ins[1], ins[2], outs[0]),
        dtype=data.dtype,
        out_buffers=[out_buf],
        name="scatter_nd_cuda",
        tag="scatter_nd_cuda",
    )
