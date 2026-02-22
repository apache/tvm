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
# pylint: disable=invalid-name, no-member, too-many-locals, too-many-arguments, too-many-statements, singleton-comparison, unused-argument, no-else-return
"""Sort related operators"""

import tvm
from tvm import te
from tvm.script.ir_builder import IRBuilder
from tvm.script.ir_builder import tir as T

from ..math import cast, ceil_log2
from ..searchsorted import binary_search
from ..transform import strided_slice, transpose
from ..utils import ceil_div, prod, swap


def _get_threads(nthread_tx, nthread_bx, nthread_by):
    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    by = te.thread_axis("blockIdx.y")
    return tx, bx, by, nthread_tx, nthread_bx, nthread_by


def _sort_init(shape, axis, keys_in, keys_out, values_out=None, value_init_func=None):
    """Initialize the output buffers by copying from inputs"""
    axis_mul_before = 1
    axis_mul_after = 1
    if axis < 0:
        axis = len(shape) + axis
    for i, value in enumerate(shape, 0):
        if i < axis:
            axis_mul_before *= value
        elif i > axis:
            axis_mul_after *= value

    # Set up threading
    max_threads = int(tvm.target.Target.current(allow_none=False).attrs["max_num_threads"])
    nthread_tx = max_threads
    nthread_bx = ceil_div(shape[axis], max_threads)
    nthread_by = axis_mul_before * axis_mul_after

    # Copy the keys_in to initial output
    tx, bx, by, ntx, nbx, nby = _get_threads(nthread_tx, nthread_bx, nthread_by)
    with T.frame_scope(
        [
            T.attr(tx, "thread_extent", ntx),
            T.attr(bx, "thread_extent", nbx),
            T.attr(by, "thread_extent", nby),
        ]
    ):
        tid = bx * nthread_tx + tx
        by_val = by % axis_mul_before
        bz = by // axis_mul_before
        idx = (by_val * shape[axis] + tid) * axis_mul_after + bz
        with T.If(tid < shape[axis]):
            with T.Then():
                keys_out[idx] = keys_in[idx]
                if values_out is not None:
                    values_out[idx] = value_init_func(idx, tid)

    return axis_mul_before, axis_mul_after


## TODO(mbrookhart): These are effective optimziation hyperparametrs
## Perhaps we can autotune?
block_size = 128
thread_work = 4


def _odd_even_sort(
    size,
    axis_mul_before,
    axis_mul_after,
    is_ascend,
    keys,
    keys_swap,
    values=None,
    values_swap=None,
):
    nthread_tx = block_size // 2
    nthread_bx = ceil_div(size, block_size)
    nthread_by = axis_mul_before * axis_mul_after

    tx, bx, by, ntx, nbx, nby = _get_threads(nthread_tx, nthread_bx, nthread_by)
    with T.frame_scope(
        [
            T.attr(tvm.tir.const(0), "hand_threaded", 0),
            T.attr(tx, "thread_extent", ntx),
            T.attr(bx, "thread_extent", nbx),
            T.attr(by, "thread_extent", nby),
        ]
    ):
        by_val = by % axis_mul_before
        bz = by // axis_mul_before
        tid = 2 * tx
        start = bx * block_size

        # Build list of allocations
        alloc_frames = [
            T.allocate([block_size], keys_swap.dtype, scope="shared"),  # tmp_keys_swap
            T.allocate([1], keys_swap.dtype, scope="local"),  # temp_keys
            T.allocate([1], keys_swap.dtype, scope="local"),  # temp_cond1
            T.allocate([1], keys_swap.dtype, scope="local"),  # temp_cond2
        ]
        if values_swap is not None:
            alloc_frames.append(
                T.allocate([block_size], values_swap.dtype, scope="shared")
            )  # tmp_values_swap
            alloc_frames.append(T.allocate([1], values_swap.dtype, scope="local"))  # temp_values

        with T.frame_scope(alloc_frames) as allocs:
            if values_swap is not None:
                (
                    tmp_keys_swap_ptr,
                    temp_keys_ptr,
                    temp_cond1_ptr,
                    temp_cond2_ptr,
                    tmp_values_swap_ptr,
                    temp_values_ptr,
                ) = allocs
            else:
                (
                    tmp_keys_swap_ptr,
                    temp_keys_ptr,
                    temp_cond1_ptr,
                    temp_cond2_ptr,
                ) = allocs
                tmp_values_swap_ptr = None
                temp_values_ptr = None

            # Create buffer views
            tmp_keys_swap = tvm.tir.decl_buffer(
                [block_size],
                keys_swap.dtype,
                "tmp_keys_swap",
                data=tmp_keys_swap_ptr,
                scope="shared",
            )
            temp_keys = tvm.tir.decl_buffer(
                [1], keys_swap.dtype, "temp_keys", data=temp_keys_ptr, scope="local"
            )
            temp_cond1 = tvm.tir.decl_buffer(
                [1], keys_swap.dtype, "temp_cond1", data=temp_cond1_ptr, scope="local"
            )
            temp_cond2 = tvm.tir.decl_buffer(
                [1], keys_swap.dtype, "temp_cond2", data=temp_cond2_ptr, scope="local"
            )
            if values_swap is not None:
                tmp_values_swap = tvm.tir.decl_buffer(
                    [block_size],
                    values_swap.dtype,
                    "tmp_values_swap",
                    data=tmp_values_swap_ptr,
                    scope="shared",
                )
                temp_values = tvm.tir.decl_buffer(
                    [1],
                    values_swap.dtype,
                    "temp_values",
                    data=temp_values_ptr,
                    scope="local",
                )

            # Copy data to scratch space
            base_idx = by_val * size * axis_mul_after + bz
            with T.serial(0, 2) as n:
                with T.If((tid + n + start) < size):
                    with T.Then():
                        T.buffer_store(
                            tmp_keys_swap,
                            keys[base_idx + (tid + n + start) * axis_mul_after],
                            [tid + n],
                        )
                        if values_swap is not None:
                            T.buffer_store(
                                tmp_values_swap,
                                values[base_idx + (tid + n + start) * axis_mul_after],
                                [tid + n],
                            )

            T.evaluate(tvm.tir.Call(None, "tir.tvm_storage_sync", tvm.runtime.convert(["shared"])))

            idxm = tvm.tir.indexmod
            # OddEvenTransposeSort
            current_sort_num = tvm.tir.min(block_size, size - start)
            with T.serial(0, current_sort_num) as k:
                n = idxm(tid + k, 2)
                with T.If(tid + n < current_sort_num - 1):
                    with T.Then():
                        T.buffer_store(temp_cond1, tmp_keys_swap[tid + n], [0])
                        T.buffer_store(temp_cond2, tmp_keys_swap[tid + n + 1], [0])
                        if is_ascend:
                            cond = temp_cond1[0] > temp_cond2[0]
                        else:
                            cond = temp_cond1[0] < temp_cond2[0]
                        with T.If(cond):
                            with T.Then():
                                T.buffer_store(temp_keys, tmp_keys_swap[tid + n], [0])
                                T.buffer_store(tmp_keys_swap, tmp_keys_swap[tid + n + 1], [tid + n])
                                T.buffer_store(tmp_keys_swap, temp_keys[0], [tid + n + 1])
                                if values_swap is not None:
                                    T.buffer_store(temp_values, tmp_values_swap[tid + n], [0])
                                    T.buffer_store(
                                        tmp_values_swap,
                                        tmp_values_swap[tid + n + 1],
                                        [tid + n],
                                    )
                                    T.buffer_store(tmp_values_swap, temp_values[0], [tid + n + 1])
                T.evaluate(
                    tvm.tir.Call(None, "tir.tvm_storage_sync", tvm.runtime.convert(["shared"]))
                )

            ## Copy sorted data to output
            with T.serial(0, 2) as n:
                with T.If(tid + n + start < size):
                    with T.Then():
                        out_idx = base_idx + (tid + n + start) * axis_mul_after
                        keys[out_idx] = tmp_keys_swap[tid + n]
                        keys_swap[out_idx] = tmp_keys_swap[tid + n]
                        if values_swap is not None:
                            values[out_idx] = tmp_values_swap[tid + n]
                            values_swap[out_idx] = tmp_values_swap[tid + n]


def _sort_common(
    size,
    axis_mul_before,
    axis_mul_after,
    is_ascend,
    keys,
    keys_swap,
    values=None,
    values_swap=None,
):
    """Either sort only values or sort values by keys."""

    ## This function performs a multi-level mergesort
    ## For blocks of length <= block_size, it does odd-even transpose sort
    ##    in GPU shared memory
    ## For intermediate block sizes (>block_size, < max_threads * thread_work)
    ##    it uses the mergpath algorthim https://arxiv.org/abs/1406.2628
    ##    to merge blocks in parallel
    ## At some point, the size of the blocks to be merged is too big for max_threads
    ##    and we switch to using a dual-level mergepath where the outer mergepath
    ##    finds the start/end locations of the inner mergepath so that we can split
    ##    the merge into more blocks

    target = tvm.target.Target.current(allow_none=False)
    max_threads = int(target.attrs["max_num_threads"])
    is_webgpu = "webgpu" in str(target)
    target_dtype = "int32" if is_webgpu else "int64"
    nthread_by = axis_mul_before * axis_mul_after
    nthread_tx = max_threads
    nthread_bx = ceil_div(size, nthread_tx)

    def compare(a, b):
        """Compare a and b in proper ascending or descending order"""
        if is_ascend:
            out = a <= b
        else:
            out = b <= a
        return out

    # Sort the lower levels of the merge using odd-even sort, it's fast for small inputs
    lower_lim = ceil_log2(block_size)

    _odd_even_sort(
        size,
        axis_mul_before * axis_mul_after,
        1,
        is_ascend,
        keys,
        keys_swap,
        values,
        values_swap,
    )

    upper_lim = ceil_log2(size)

    def get_merge_begin(source, base_idx, aCount, bCount, aStart, bStart, diag, first, last):
        max_val = tvm.te.max(0, diag - bCount)
        min_val = tvm.te.min(diag, aCount)
        if is_webgpu:
            first[0] = cast(max_val, target_dtype)
            last[0] = cast(min_val, target_dtype)
        else:
            first[0] = max_val
            last[0] = min_val
        with T.While(first[0] < last[0]):
            mid = (first[0] + last[0]) >> 1
            a = source[base_idx + (aStart + mid)]
            b = source[base_idx + (bStart + diag - 1 - mid)]
            with T.If(compare(a, b)):
                with T.Then():
                    first[0] = mid + 1
                with T.Else():
                    last[0] = mid

    def serial_merge(
        source,
        dest,
        source_idx,
        dest_idx,
        base_idx,
        aCount,
        bCount,
        aStart,
        bStart,
        kStart,
        diag,
        step_count,
        first,
        last,
        i_buf,
        j_buf,
    ):
        i_val = aStart + first[0]
        j_val = bStart + diag - last[0]
        if is_webgpu:
            i_buf[0] = cast(i_val, target_dtype)
            j_buf[0] = cast(j_val, target_dtype)
        else:
            i_buf[0] = i_val
            j_buf[0] = j_val

        with T.serial(0, tvm.te.min(aCount + bCount - diag, step_count)) as count:
            i_idx = base_idx + i_buf[0]
            j_idx = base_idx + j_buf[0]
            k_idx = base_idx + (kStart + diag + count)

            with T.If(tvm.tir.all(i_buf[0] < aStart + aCount, j_buf[0] < bStart + bCount)):
                with T.Then():
                    with T.If(compare(source[i_idx], source[j_idx])):
                        with T.Then():
                            dest[k_idx] = source[i_idx]
                            if values is not None:
                                dest_idx[k_idx] = source_idx[i_idx]
                            i_buf[0] = i_buf[0] + 1
                        with T.Else():
                            dest[k_idx] = source[j_idx]
                            if values is not None:
                                dest_idx[k_idx] = source_idx[j_idx]
                            j_buf[0] = j_buf[0] + 1
                with T.Else():
                    with T.If(i_buf[0] < aStart + aCount):
                        with T.Then():
                            dest[k_idx] = source[i_idx]
                            if values is not None:
                                dest_idx[k_idx] = source_idx[i_idx]
                            i_buf[0] = i_buf[0] + 1
                        with T.Else():
                            dest[k_idx] = source[j_idx]
                            if values is not None:
                                dest_idx[k_idx] = source_idx[j_idx]
                            j_buf[0] = j_buf[0] + 1

    def mergepath(
        source,
        dest,
        source_idx,
        dest_idx,
        base_idx,
        aCount,
        bCount,
        aStart,
        bStart,
        kStart,
        tx,
        step_count,
        even,
    ):
        with T.frame_scope(
            [
                T.allocate([1], target_dtype, scope="local"),  # first
                T.allocate([1], target_dtype, scope="local"),  # last
                T.allocate([1], target_dtype, scope="local"),  # i_buf
                T.allocate([1], target_dtype, scope="local"),  # j_buf
            ]
        ) as (first_ptr, last_ptr, i_ptr, j_ptr):
            first = T.buffer_proxy(
                tvm.tir.decl_buffer([1], target_dtype, "first", data=first_ptr, scope="local")
            )
            last = T.buffer_proxy(
                tvm.tir.decl_buffer([1], target_dtype, "last", data=last_ptr, scope="local")
            )
            i_buf = T.buffer_proxy(
                tvm.tir.decl_buffer([1], target_dtype, "i", data=i_ptr, scope="local")
            )
            j_buf = T.buffer_proxy(
                tvm.tir.decl_buffer([1], target_dtype, "j", data=j_ptr, scope="local")
            )

            diag = tx * step_count
            with T.If(even):
                with T.Then():
                    get_merge_begin(
                        source, base_idx, aCount, bCount, aStart, bStart, diag, first, last
                    )
                    serial_merge(
                        source,
                        dest,
                        source_idx,
                        dest_idx,
                        base_idx,
                        aCount,
                        bCount,
                        aStart,
                        bStart,
                        kStart,
                        diag,
                        step_count,
                        first,
                        last,
                        i_buf,
                        j_buf,
                    )
                with T.Else():
                    get_merge_begin(
                        dest, base_idx, aCount, bCount, aStart, bStart, diag, first, last
                    )
                    # Intentionally swap source/dest for reverse direction merge
                    serial_merge(  # pylint: disable=arguments-out-of-order
                        dest,
                        source,
                        dest_idx,
                        source_idx,
                        base_idx,
                        aCount,
                        bCount,
                        aStart,
                        bStart,
                        kStart,
                        diag,
                        step_count,
                        first,
                        last,
                        i_buf,
                        j_buf,
                    )

    def dual_mergepath(
        source,
        dest,
        source_idx,
        dest_idx,
        base_idx,
        start_pos,
        middle,
        end,
        bx,
        tx,
        step_count,
        even,
    ):
        with T.frame_scope(
            [
                T.allocate([1], target_dtype, scope="local"),  # outer_first
                T.allocate([1], target_dtype, scope="local"),  # outer_last
                T.allocate([1], target_dtype, scope="local"),  # first
                T.allocate([1], target_dtype, scope="local"),  # last
                T.allocate([1], target_dtype, scope="local"),  # i_buf
                T.allocate([1], target_dtype, scope="local"),  # j_buf
            ]
        ) as (outer_first_ptr, outer_last_ptr, first_ptr, last_ptr, i_ptr, j_ptr):
            outer_first = T.buffer_proxy(
                tvm.tir.decl_buffer(
                    [1], target_dtype, "outer_first", data=outer_first_ptr, scope="local"
                )
            )
            outer_last = T.buffer_proxy(
                tvm.tir.decl_buffer(
                    [1], target_dtype, "outer_last", data=outer_last_ptr, scope="local"
                )
            )
            first = T.buffer_proxy(
                tvm.tir.decl_buffer([1], target_dtype, "first", data=first_ptr, scope="local")
            )
            last = T.buffer_proxy(
                tvm.tir.decl_buffer([1], target_dtype, "last", data=last_ptr, scope="local")
            )
            i_buf = T.buffer_proxy(
                tvm.tir.decl_buffer([1], target_dtype, "i", data=i_ptr, scope="local")
            )
            j_buf = T.buffer_proxy(
                tvm.tir.decl_buffer([1], target_dtype, "j", data=j_ptr, scope="local")
            )

            diag = bx * step_count
            with T.If(even):
                with T.Then():
                    get_merge_begin(
                        source,
                        base_idx,
                        middle - start_pos,
                        end - middle,
                        start_pos,
                        middle,
                        diag,
                        outer_first,
                        outer_last,
                    )
                    aStart = start_pos + outer_first[0]
                    bStart = middle + diag - outer_last[0]
                    aCount = tvm.te.min(middle - aStart, step_count)
                    bCount = tvm.te.min(end - bStart, step_count)
                    inner_diag = tx * thread_work
                    get_merge_begin(
                        source, base_idx, aCount, bCount, aStart, bStart, inner_diag, first, last
                    )
                    serial_merge(
                        source,
                        dest,
                        source_idx,
                        dest_idx,
                        base_idx,
                        aCount,
                        bCount,
                        aStart,
                        bStart,
                        start_pos + diag,
                        inner_diag,
                        thread_work,
                        first,
                        last,
                        i_buf,
                        j_buf,
                    )
                with T.Else():
                    get_merge_begin(
                        dest,
                        base_idx,
                        middle - start_pos,
                        end - middle,
                        start_pos,
                        middle,
                        diag,
                        outer_first,
                        outer_last,
                    )
                    aStart = start_pos + outer_first[0]
                    bStart = middle + diag - outer_last[0]
                    aCount = tvm.te.min(middle - aStart, step_count)
                    bCount = tvm.te.min(end - bStart, step_count)
                    inner_diag = tx * thread_work
                    get_merge_begin(
                        dest, base_idx, aCount, bCount, aStart, bStart, inner_diag, first, last
                    )
                    serial_merge(
                        dest,
                        source,
                        dest_idx,
                        source_idx,
                        base_idx,
                        aCount,
                        bCount,
                        aStart,
                        bStart,
                        start_pos + diag,
                        inner_diag,
                        thread_work,
                        first,
                        last,
                        i_buf,
                        j_buf,
                    )

    with T.serial(0, cast(upper_lim - lower_lim, target_dtype)) as l2_width:
        width = 2 << (l2_width + lower_lim)
        # Define and launch the CUDA kernel
        target = tvm.target.Target.current()
        if "vulkan" in str(target):
            ntx = max_threads
            nbx = tvm.tir.generic.cast(ceil_div(width, max_threads * thread_work), "int32")
            nbz = tvm.tir.generic.cast(ceil_div(size, width), "int32")
        else:
            ntx = tvm.tir.generic.cast(tvm.te.min(max_threads, width), "int32")
            nbx = tvm.tir.generic.cast(ceil_div(width, max_threads * thread_work), "int32")
            nbz = tvm.tir.generic.cast(ceil_div(size, width), "int32")

        tx, bx, by, _, _, _ = _get_threads(ntx, nbx, nthread_by * nbz)
        with T.frame_scope(
            [
                T.attr(tx, "thread_extent", ntx),
                T.attr(bx, "thread_extent", nbx),
                T.attr(by, "thread_extent", nthread_by * nbz),
            ]
        ):
            by_val = by % nthread_by
            bz = by // nthread_by
            base_idx = by_val * size

            # calculate the start, mid, and end points of this section
            start_pos = width * bz
            middle = cast(tvm.te.min(start_pos + tvm.tir.indexdiv(width, 2), size), target_dtype)
            end = cast(tvm.te.min(start_pos + width, size), target_dtype)

            with T.If(start_pos < size):
                with T.Then():
                    even = tvm.tir.indexmod(l2_width, 2) == 0
                    with T.If(nbx == 1):
                        with T.Then():
                            ## merge the start->middle and middle->end arrays
                            aCount = middle - start_pos
                            bCount = end - middle
                            mergepath(
                                keys,
                                keys_swap,
                                values,
                                values_swap,
                                base_idx,
                                aCount,
                                bCount,
                                start_pos,
                                middle,
                                start_pos,
                                tx,
                                ceil_div(width, ntx),
                                even,
                            )
                        with T.Else():
                            dual_mergepath(
                                keys,
                                keys_swap,
                                values,
                                values_swap,
                                base_idx,
                                start_pos,
                                middle,
                                end,
                                bx,
                                tx,
                                max_threads * thread_work,
                                even,
                            )

    ## if the final sorted data ended up in the swap, copy it to the real output
    nthread_bx = ceil_div(size, nthread_tx)
    with T.If(tvm.tir.all(upper_lim > lower_lim, tvm.tir.indexmod(upper_lim - lower_lim, 2) == 1)):
        with T.Then():
            tx2, bx2, by2, _, _, _ = _get_threads(nthread_tx, nthread_bx, nthread_by)
            with T.frame_scope(
                [
                    T.attr(tx2, "thread_extent", nthread_tx),
                    T.attr(bx2, "thread_extent", nthread_bx),
                    T.attr(by2, "thread_extent", nthread_by),
                ]
            ):
                tid = bx2 * nthread_tx + tx2
                idx = by2 * size + tid
                with T.If(tid < size):
                    with T.Then():
                        keys[idx] = keys_swap[idx]
                        if values is not None:
                            values[idx] = values_swap[idx]


def sort_ir(
    data, values_out, values_out_swap, axis, is_ascend, indices_out=None, indices_out_swap=None
):
    """Low level IR to do sorting on the GPU, same usage as tvm.contrib.sort.argsort on the CPU.

    Parameters
    ----------
    data: Buffer
        Buffer of input data. Data will be sorted in place.

    values_out : Buffer
        Output buffer of values of sorted tensor with same shape as data.

    values_out_swap : Buffer
        Output buffer of values with same shape as data to use as swap.

    axis : Int
        Axis long which to sort the input tensor.

    is_ascend : Boolean
        Whether to sort in ascending or descending order.

    indicess_out : Buffer
        Output buffer of indices of sorted tensor with same shape as data.

    indices_out_swap : Buffer
        Output buffer of indices with same shape as data to use as swap.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    with IRBuilder() as ib:
        shape = data.shape

        data = T.buffer_proxy(data)
        values_out = T.buffer_proxy(values_out)
        values_out_swap = T.buffer_proxy(values_out_swap)
        if indices_out is not None:
            indices_out_orig = indices_out
            indices_out = T.buffer_proxy(indices_out)
            assert indices_out_swap is not None
            indices_out_swap = T.buffer_proxy(indices_out_swap)

        with T.If(shape[axis] > 0):
            with T.Then():
                axis_mul_before, axis_mul_after = _sort_init(
                    shape,
                    axis,
                    data,
                    values_out,
                    indices_out,
                    value_init_func=(
                        lambda _, tid: (
                            tvm.tir.generic.cast(tid, indices_out_orig.dtype)
                            if indices_out is not None
                            else None
                        )
                    ),
                )

                _sort_common(
                    shape[axis],
                    axis_mul_before,
                    axis_mul_after,
                    is_ascend,
                    values_out,
                    values_out_swap,
                    values=indices_out,
                    values_swap=indices_out_swap,
                )

        return ib.get()


def sort(data, axis=-1, is_ascend=1):
    """Performs sorting along the given axis and returns an array of
    sorted values with the same shape as the input data.

    Parameters
    ----------
    data: tvm.te.Tensor
        The input array.

    axis : int, optional
        Axis long which to sort the input tensor.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    Returns
    -------
    out : tvm.te.Tensor
        The output of this function.
    """
    ndim = len(data.shape)
    axis = ndim + axis if axis < 0 else axis
    if axis != ndim - 1:
        # Prepare for sorting along axis -1.
        axes = swap(list(range(ndim)), axis)
        data = transpose(data, axes)

    value_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "value_buf", data_alignment=8)
    value_buf_swap = tvm.tir.decl_buffer(data.shape, data.dtype, "value_buf_swap", data_alignment=8)

    out = te.extern(
        [data.shape, data.shape],
        [data],
        lambda ins, outs: sort_ir(ins[0], outs[0], outs[1], -1, is_ascend),
        out_buffers=[value_buf, value_buf_swap],
        name="sort_gpu",
        tag="sort_gpu",
    )[0]

    if axis != ndim - 1:
        axes = swap(list(range(ndim)), axis)
        out = transpose(out, axes)

    return out


def sort_thrust(data, axis=-1, is_ascend=1, workspace=None):
    """Performs sorting along the given axis and returns an array of
    sorted values with the same shape as the input data.

    Parameters
    ----------
    data: tvm.te.Tensor
        The input array.

    axis : int, optional
        Axis long which to sort the input tensor.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    workspace: Optional[tvm.te.Tensor]
        A buffer to store intermediate results. The size of the workspace should be sufficiently
        large, this can be obtained by overestimation or memory usage profiling. If None, it will
        fallback to use thrust internal memory allocation.


    Returns
    -------
    out : tvm.te.Tensor
        The output of this function.
    """
    dtype = "float32"
    ndim = len(data.shape)
    axis = ndim + axis if axis < 0 else axis

    if axis != ndim - 1:
        # Prepare for sorting along axis -1.
        axes = swap(list(range(ndim)), axis)
        data = transpose(data, axes)

    value_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "value_buf", data_alignment=8)
    indices_buf = tvm.tir.decl_buffer(data.shape, dtype, "out_buf", data_alignment=8)

    def f_compute(ins, outs):
        args = ["tvm.contrib.thrust.sort", ins[0], outs[0], outs[1], is_ascend]
        if workspace is not None:
            args.append(ins[1])
        return tvm.tir.call_packed(*args)

    out = te.extern(
        [data.shape, data.shape],
        [data] if workspace is None else [data, workspace],
        ## TODO(mbrookhart): This thrust function is actually doing argsort, not sort
        ## For performance, we should probably rename the contrib function and add
        ## a pure sort
        f_compute,
        out_buffers=[value_buf, indices_buf],
        name="sort_gpu",
        tag="sort_gpu",
    )[0]

    if axis != ndim - 1:
        axes = swap(list(range(ndim)), axis)
        out = transpose(out, axes)
    return out


def argsort(data, axis=-1, is_ascend=1, dtype="float32", ret_type="indices"):
    """Performs sorting along the given axis and returns an array of indices
    having same shape as an input array that index data in sorted order.

    Parameters
    ----------
    data: tvm.te.Tensor
        The input array.

    axis : int, optional
        Axis long which to sort the input tensor.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        DType of the output indices.

    ret_type : string, optional
        The return type [both, indices].
        "both": return both sorted data and indices.
        "indices": return sorted indices only.

    Returns
    -------
    out : tvm.te.Tensor
        The output of this function.
    """
    ndim = len(data.shape)
    axis = ndim + axis if axis < 0 else axis
    if axis != ndim - 1:
        # Prepare for sorting along axis -1.
        axes = swap(list(range(ndim)), axis)
        data = transpose(data, axes)

    value_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "value_buf", data_alignment=8)
    value_swap_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "value_swap_buf", data_alignment=8)
    indices_buf = tvm.tir.decl_buffer(data.shape, dtype, "out_buf", data_alignment=8)
    indices_swap_buf = tvm.tir.decl_buffer(data.shape, dtype, "out_swap_buf", data_alignment=8)

    outs = te.extern(
        [data.shape, data.shape, data.shape, data.shape],
        [data],
        lambda ins, outs: sort_ir(
            ins[0],
            outs[0],
            outs[2],
            -1,
            is_ascend,
            indices_out=outs[1],
            indices_out_swap=outs[3],
        ),
        out_buffers=[value_buf, indices_buf, value_swap_buf, indices_swap_buf],
        name="argsort_gpu",
        tag="argsort_gpu",
    )

    if axis != ndim - 1:
        axes = swap(list(range(ndim)), axis)
        outs = [transpose(out, axes) for out in outs]

    if ret_type == "indices":
        return outs[1]

    return outs[0], outs[1]


def argsort_thrust(data, axis=-1, is_ascend=1, dtype="float32", ret_type="indices", workspace=None):
    """Performs sorting along the given axis and returns an array of indices
    having same shape as an input array that index data in sorted order.

    Parameters
    ----------
    data: tvm.te.Tensor
        The input array.

    axis : int, optional
        Axis long which to sort the input tensor.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        DType of the output indices.

    ret_type : string, optional
        The return type [both, indices].
        "both": return both sorted data and indices.
        "indices": return sorted indices only.

    workspace : Optional[tvm.te.Tensor]
        A buffer to store intermediate results. The size of the workspace should be sufficiently
        large, this can be obtained by overestimation or memory usage profiling. If None, it will
        fallback to use thrust internal memory allocation.

    Returns
    -------
    out : tvm.te.Tensor
        The output of this function.
    """
    return topk_thrust(data, 0, axis, ret_type, is_ascend, dtype, workspace)


def topk(data, k=1, axis=-1, ret_type="both", is_ascend=False, dtype="int64"):
    """Get the top k elements in an input tensor along the given axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    k : int, optional
        Number of top elements to select. Return all elements if k < 1.

    axis : int, optional
        Axis long which to sort the input tensor.

    ret_type: str, optional
        The return type [both, values, indices].
        "both": return both top k data and indices.
        "values": return top k data only.
        "indices": return top k indices only.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        The data type of the indices output.

    Returns
    -------
    out : tvm.te.Tensor or List[tvm.te.Tensor]
        The computed result.
    """
    assert ret_type in ["both", "values", "indices"]
    ndim = len(data.shape)
    axis = axis + ndim if axis < 0 else axis
    assert 0 <= axis < ndim
    dshape = data.shape
    if axis != ndim - 1:
        axes = swap(list(range(ndim)), axis)
        data = transpose(data, axes)

    values_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "values_buf", data_alignment=8)
    values_swap_buf = tvm.tir.decl_buffer(
        data.shape, data.dtype, "values_swap_buf", data_alignment=8
    )
    indices_buf = tvm.tir.decl_buffer(data.shape, dtype, "indices_buf", data_alignment=8)
    indices_swap_buf = tvm.tir.decl_buffer(data.shape, dtype, "indies_swap_buf", data_alignment=8)

    if ret_type == "values":
        output = te.extern(
            [data.shape, data.shape],
            [data],
            lambda ins, outs: sort_ir(ins[0], outs[0], outs[1], -1, is_ascend),
            out_buffers=[values_buf, values_swap_buf],
            name="topk_gpu",
            tag="topk_gpu",
        )[0]
        if axis != ndim - 1:
            axes = swap(list(range(ndim)), axis)
            output = transpose(output, axes)
    else:
        output = te.extern(
            [data.shape, data.shape, data.shape, data.shape],
            [data],
            lambda ins, outs: sort_ir(ins[0], outs[0], outs[2], -1, is_ascend, outs[1], outs[3]),
            out_buffers=[values_buf, indices_buf, values_swap_buf, indices_swap_buf],
            name="topk_gpu",
            tag="topk_gpu",
        )[0:2]
        if axis != ndim - 1:
            axes = swap(list(range(ndim)), axis)
            output[0] = transpose(output[0], axes)
            output[1] = transpose(output[1], axes)

    if isinstance(k, int) and k < 1:
        if ret_type == "indices":
            return output[1]
        return output
    beg = [0] * ndim
    end = []
    strides = [1] * ndim
    for i in range(ndim):
        if i == axis:
            end.append(k if isinstance(k, int) else tvm.te.size_var("dim"))
        else:
            end.append(dshape[i])
    if ret_type == "both":
        values_out, indices_out = output
        values_out = strided_slice(values_out, beg, end, strides)
        indices_out = strided_slice(indices_out, beg, end, strides)
        output = [values_out, indices_out]
    elif ret_type == "values":
        output = [strided_slice(output, beg, end, strides)]
    else:  # ret_type == "indices"
        indices_out = output[1]
        output = [strided_slice(indices_out, beg, end, strides)]
    return output


def topk_thrust(
    data, k=1, axis=-1, ret_type="both", is_ascend=False, dtype="int64", workspace=None
):
    """Get the top k elements in an input tensor along the given axis.

    Parameters
    ----------
    data : tvm.te.Tensor
        The input tensor.

    k : int, optional
        Number of top elements to select. Return all elements if k < 1.

    axis : int, optional
        Axis long which to sort the input tensor.

    ret_type: str, optional
        The return type [both, values, indices].
        "both": return both top k data and indices.
        "values": return top k data only.
        "indices": return top k indices only.

    is_ascend : boolean, optional
        Whether to sort in ascending or descending order.

    dtype : string, optional
        The data type of the indices output.

    workspace : Optional[tvm.te.Tensor]
        A buffer to store intermediate results. The size of the workspace should be sufficiently
        large, this can be obtained by overestimation or memory usage profiling. If None, it will
        fallback to use thrust internal memory allocation.

    Returns
    -------
    out : tvm.te.Tensor or List[tvm.te.Tensor]
        The computed result.
    """
    assert ret_type in ["both", "values", "indices"]
    ndim = len(data.shape)
    axis = ndim + axis if axis < 0 else axis

    if axis != ndim - 1:
        # Prepare for sorting along axis -1.
        axes = swap(list(range(ndim)), axis)
        data = transpose(data, axes)

    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    if workspace is not None:
        workspace_buf = tvm.tir.decl_buffer(
            workspace.shape, workspace.dtype, "workspace_buf", data_alignment=8
        )
    else:
        workspace_buf = None
    out_bufs = [
        tvm.tir.decl_buffer(data.shape, data.dtype, "value_buf", data_alignment=8),
        tvm.tir.decl_buffer(data.shape, dtype, "indices_buf", data_alignment=8),
    ]

    def f_compute(ins, outs):
        args = ["tvm.contrib.thrust.sort", ins[0], outs[0], outs[1], is_ascend]
        if workspace is not None:
            args.append(ins[1])
        return tvm.tir.call_packed(*args)

    is_ascend = 1 if is_ascend else 0

    out = te.extern(
        [data.shape, data.shape],
        [data] if workspace is None else [data, workspace],
        f_compute,
        in_buffers=[data_buf] if workspace is None else [data_buf, workspace_buf],
        out_buffers=out_bufs,
        name="topk_gpu",
        tag="topk_gpu",
    )

    if isinstance(k, tvm.tir.IntImm):
        k = k.value

    if not isinstance(k, int) or k > 0:
        beg = [0] * ndim
        end = data.shape[:-1] + [k if isinstance(k, int) else tvm.te.size_var("dim")]
        strides = [1] * ndim
        out = [strided_slice(o, beg, end, strides) for o in out]

    if axis != ndim - 1:
        axes = swap(list(range(ndim)), axis)
        out = [transpose(o, axes) for o in out]

    if ret_type == "values":
        out = out[0]
    elif ret_type == "indices":
        out = out[1]

    return out


def searchsorted(sorted_sequence, values, right=False, out_dtype="int64"):
    """Find indices where elements should be inserted to maintain order.
       If `sorted_sequence` is N-dimensional, the innermost dimension of
       `values` are searched in the corresponding dimension of `sorted_sequence`.

       This implementation is optimized for GPU execution.

    Parameters
    ----------
    sorted_sequence : te.Tensor
        N-D or 1-D Tensor, containing monotonically increasing sequence
        on the innermost dimension.

    values : te.Tensor
        N-D Tensor containing the search values. When `sorted_sequence` is 1-D,
        the shape of `values` can be arbitrary. Otherwise, ranks of `sorted_sequence`
        and `values` must be the same, and outer N-1 axes must have the same size.

    right : bool, optional
        Controls which index is returned if a value lands exactly on one of sorted values. If
        False (side='left'), the index of the first suitable location found is given. If true
        (side='right'), return the last such index.

    out_dtype : string, optional
        The data type of the output indices.

    Returns
    -------
    indices : te.Tensor
        Tensor with same shape as values, representing the indices of
        elements of `values` if they are inserted in `sorted_sequence`.
    """
    if len(sorted_sequence.shape) > 1:
        for i in range(len(values.shape) - 1):
            assert values.shape[i] == sorted_sequence.shape[i], (
                "Outer dimensions of sorted_sequence and values must match for N-D searchsorted"
            )

    def ir(sorted_sequence_buf, values_buf, indices_buf):
        with IRBuilder() as ib:
            sorted_sequence_shape = sorted_sequence_buf.shape
            values_shape = values_buf.shape
            num_search = prod(values_shape)
            search_range = sorted_sequence_shape[-1]

            sorted_sequence_ptr = T.buffer_proxy(sorted_sequence_buf)
            values_ptr = T.buffer_proxy(values_buf)
            indices_ptr = T.buffer_proxy(indices_buf)

            max_threads = int(tvm.target.Target.current(allow_none=False).attrs["max_num_threads"])
            nthread_tx = max_threads
            nthread_bx = ceil_div(num_search, nthread_tx)
            tx = te.thread_axis("threadIdx.x")
            bx = te.thread_axis("blockIdx.x")
            with T.frame_scope(
                [
                    T.attr(tx, "thread_extent", nthread_tx),
                    T.attr(bx, "thread_extent", nthread_bx),
                ]
            ):
                tid = bx * nthread_tx + tx

                with T.If(tid < num_search):
                    with T.Then():
                        if len(sorted_sequence_shape) == 1:
                            sequence_offset = 0
                        else:
                            sequence_id = tid // values_shape[-1]
                            sequence_offset = sequence_id * search_range

                        indices_ptr[tid] = binary_search(
                            sequence_offset,
                            search_range,
                            sorted_sequence_ptr,
                            values_ptr[tid],
                            right,
                            out_dtype,
                        )

            return ib.get()

    return te.extern(
        values.shape,
        [sorted_sequence, values],
        lambda ins, outs: ir(ins[0], ins[1], outs[0]),
        name="searchsorted_gpu",
        dtype=out_dtype,
    )
