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
# pylint: disable=invalid-name, too-many-locals, too-many-statements
"Scan related operators"
import tvm
from tvm import te
from tvm._ffi import get_global_func
from ..transform import expand_dims, squeeze
from ..utils import ceil_div
from ..math import cast
from .. import tag
from .injective import schedule_injective_from_existing


def exclusive_sum_scan2d_ir(data, output, reduction=None):
    """
    TODO
    """
    batch_size = data.shape[0]
    scan_axis_size = data.shape[1]

    ib = tvm.tir.ir_builder.create()

    data = ib.buffer_ptr(data)
    output = ib.buffer_ptr(output)

    out_dtype = output.dtype

    if reduction is not None:
        reduction = ib.buffer_ptr(reduction)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)

    with ib.if_scope(scan_axis_size > 0):
        with ib.new_scope():
            nthread_tx = max_threads
            nthread_bx = ceil_div(scan_axis_size, max_threads)
            nthread_by = batch_size
            tx = te.thread_axis("threadIdx.x")
            bx = te.thread_axis("blockIdx.x")
            by = te.thread_axis("blockIdx.y")
            ib.scope_attr(tx, "thread_extent", nthread_tx)
            ib.scope_attr(bx, "thread_extent", nthread_bx)
            ib.scope_attr(by, "thread_extent", nthread_by)
            tid = bx * nthread_tx + tx
            with ib.if_scope(tid < scan_axis_size):
                output[by, tid] = data[by, tid]

        nthread_tx = max_threads
        nthread_bx = ceil_div(scan_axis_size, max_threads)
        nthread_by = batch_size

        # The following algorithm performs parallel exclusive scan
        # Up Sweep of exclusive scan
        lim = tvm.tir.generic.cast(
            tvm.tir.ceil(tvm.tir.log2(tvm.tir.generic.cast(scan_axis_size, "float64"))), "int64"
        )
        with ib.for_range(0, lim, dtype="int64") as l2_width:
            width = 2 << l2_width

            with ib.new_scope():
                tx = te.thread_axis("threadIdx.x")
                bx = te.thread_axis("blockIdx.x")
                ib.scope_attr(tx, "thread_extent", nthread_tx)
                ib.scope_attr(
                    bx,
                    "thread_extent",
                    tvm.tir.generic.cast(ceil_div(scan_axis_size, max_threads * width), "int32"),
                )
                tid = bx * nthread_tx + tx

                by = te.thread_axis("blockIdx.y")
                ib.scope_attr(by, "thread_extent", nthread_by)
                start = ib.allocate("int64", (1,), name="start", scope="local")
                middle = ib.allocate("int64", (1,), name="middle", scope="local")
                end = ib.allocate("int64", (1,), name="end", scope="local")
                start[0] = width * tid
                with ib.if_scope(start[0] < scan_axis_size):
                    middle[0] = start[0] + tvm.tir.indexdiv(width, 2)
                    end[0] = tvm.te.min(start[0] + width, scan_axis_size)
                    with ib.if_scope(middle[0] < scan_axis_size):
                        output[by * scan_axis_size + end[0] - 1] += output[
                            by * scan_axis_size + middle[0] - 1
                        ]

        # Down Sweep of exclusive scan
        with ib.new_scope():
            bx = te.thread_axis("blockIdx.x")
            ib.scope_attr(bx, "thread_extent", batch_size)
            with ib.if_scope(bx < batch_size):
                if reduction is not None:
                    reduction[bx] = output[(bx + 1) * scan_axis_size - 1]
                output[(bx + 1) * scan_axis_size - 1] = cast(0, out_dtype)

        with ib.for_range(0, lim, dtype="int64") as l2_width:
            width = 2 << (lim - l2_width - 1)

            with ib.new_scope():
                tx = te.thread_axis("threadIdx.x")
                bx = te.thread_axis("blockIdx.x")
                ib.scope_attr(tx, "thread_extent", nthread_tx)
                ib.scope_attr(
                    bx,
                    "thread_extent",
                    tvm.tir.generic.cast(ceil_div(scan_axis_size, max_threads * width), "int32"),
                )
                tid = bx * nthread_tx + tx

                by = te.thread_axis("blockIdx.y")
                ib.scope_attr(by, "thread_extent", nthread_by)
                start = ib.allocate("int64", (1,), name="start", scope="local")
                middle = ib.allocate("int64", (1,), name="middle", scope="local")
                end = ib.allocate("int64", (1,), name="end", scope="local")
                tmp = ib.allocate(out_dtype, (1,), name="end", scope="local")
                start[0] = width * tid
                with ib.if_scope(tvm.tir.all(start[0] < scan_axis_size)):
                    middle[0] = start[0] + tvm.tir.indexdiv(width, 2)
                    end[0] = tvm.tir.min(start[0] + width, scan_axis_size)
                    with ib.if_scope(middle[0] < scan_axis_size):
                        tmp[0] = output[by * scan_axis_size + middle[0] - 1]
                        output[by * scan_axis_size + middle[0] - 1] = output[
                            by * scan_axis_size + end[0] - 1
                        ]
                        output[by * scan_axis_size + end[0] - 1] += tmp[0]

    with ib.else_scope():
        with ib.new_scope():
            bx = te.thread_axis("blockIdx.x")
            ib.scope_attr(bx, "thread_extent", batch_size)
            with ib.if_scope(bx < batch_size):
                if reduction is not None:
                    reduction[bx] = 0
    return ib.get()


def get_reduction_from_exclusive_scan_ir(data, data_ex_scan, reduction):
    """TODO"""
    batch_size = data.shape[0]
    num_anchors = data.shape[1]

    ib = tvm.tir.ir_builder.create()

    data = ib.buffer_ptr(data)
    data_ex_scan = ib.buffer_ptr(data_ex_scan)
    reduction = ib.buffer_ptr(reduction)

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    with ib.new_scope():
        nthread_tx = max_threads
        nthread_bx = ceil_div(batch_size, max_threads)
        tx = te.thread_axis("threadIdx.x")
        bx = te.thread_axis("blockIdx.x")
        ib.scope_attr(tx, "thread_extent", nthread_tx)
        ib.scope_attr(bx, "thread_extent", nthread_bx)
        tid = bx * max_threads + tx
        with ib.if_scope(tid < batch_size):
            reduction[tid] = data_ex_scan[tid, num_anchors - 1] + data[tid, num_anchors - 1]

    return ib.get()


def get_reduction_from_exclusive_scan(data, ex_scan_output):
    """TODO"""
    assert len(data.shape) == 2, "Only 2D input supported for now"
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "valid_indices_buf", data_alignment=8)
    ex_scan_output_buf = tvm.tir.decl_buffer(
        ex_scan_output.shape, ex_scan_output.dtype, "ex_scan_output_buf", data_alignment=8
    )

    return te.extern(
        [(data.shape[0],)],
        [data, ex_scan_output],
        lambda ins, outs: get_reduction_from_exclusive_scan_ir(ins[0], ins[1], outs[0]),
        dtype=[ex_scan_output.dtype],
        in_buffers=[data_buf, ex_scan_output_buf],
        name="ex_scan_reduction",
        tag="ex_scan_reduction_gpu",
    )


def is_thrust_available():
    """
    Test if thrust based scan ops are available.
    """
    return get_global_func("tvm.contrib.thrust.sum_scan", allow_missing=True) is not None


def scan_thrust(data, output_dtype, exclusive=True, return_reduction=False):
    """TODO"""
    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    output_buf = tvm.tir.decl_buffer(data.shape, output_dtype, "output_buf", data_alignment=8)
    output = te.extern(
        [data.shape],
        [data],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.thrust.sum_scan", ins[0], outs[0], exclusive
        ),
        dtype=[output_dtype],
        in_buffers=[data_buf],
        out_buffers=[output_buf],
        name="exclusive_sum_scan2d",
        tag="exclusive_sum_scan2d_gpu",
    )

    if return_reduction:
        assert exclusive, "return_reduction should be False for inclusive scan"
        ndim = len(data.shape)
        if ndim == 1:
            output = expand_dims(output, axis=0)
            reduction = get_reduction_from_exclusive_scan(data, output)
            reduction = squeeze(reduction, 0)
        else:
            reduction = get_reduction_from_exclusive_scan(data, output)
        return output, reduction

    return output


def exclusive_scan(data, axis=-1, return_reduction=False, output_dtype=None):
    """TODO"""
    ndim = len(data.shape)
    if axis < 0:
        axis += ndim
    assert axis == ndim - 1, "Only support scan on the inner most axis."

    if output_dtype is None:
        output_dtype = data.dtype

    target = tvm.target.Target.current()
    if target and target.kind.name == "cuda" and is_thrust_available():
        return scan_thrust(data, output_dtype, exclusive=True, return_reduction=return_reduction)

    if ndim == 1:
        data = expand_dims(data, axis=0)

    data_buf = tvm.tir.decl_buffer(data.shape, data.dtype, "data_buf", data_alignment=8)
    output_buf = tvm.tir.decl_buffer(data.shape, output_dtype, "output_buf", data_alignment=8)

    if len(data.shape) == 2:
        if return_reduction:
            output, reduction = te.extern(
                [data.shape, (data.shape[0],)],
                [data],
                lambda ins, outs: exclusive_sum_scan2d_ir(ins[0], outs[0], outs[1]),
                dtype=[data.dtype, output_dtype],
                in_buffers=[data_buf],
                name="exclusive_scan",
                tag="exclusive_scan_gpu",
            )
        else:
            output = te.extern(
                [data.shape],
                [data],
                lambda ins, outs: exclusive_sum_scan2d_ir(ins[0], outs[0]),
                dtype=[output_dtype],
                in_buffers=[data_buf],
                out_buffers=[output_buf],
                name="exclusive_scan",
                tag="exclusive_scan_gpu",
            )
            reduction = None
    else:
        assert False, "Unsupported dimension {}".format(ndim)

    if ndim == 1:
        output = squeeze(output, 0)
        if return_reduction:
            reduction = squeeze(reduction, 0)

    if return_reduction:
        return output, reduction

    return output


def schedule_scan(outs):
    """Schedule for scan operator.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of scan
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
      The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    scheduled_ops = []

    def traverse(op):
        if tag.is_injective(op.tag):
            schedule_injective_from_existing(s, op.output(0))
        for tensor in op.input_tensors:
            if tensor.op.input_tensors and tensor.op not in scheduled_ops:
                traverse(tensor.op)
        scheduled_ops.append(op)

    for out in outs:
        traverse(out.op)
    return s
