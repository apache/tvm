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
# pylint: disable=too-many-arguments, invalid-name
"""Argwhere operator"""

import logging

import tvm
from tvm import te
from tvm._ffi import get_global_func
from .injective import schedule_injective_from_existing
from .nms import atomic_add
from .sort import topk, topk_thrust, argsort, argsort_thrust
from .. import tag
from ..transform import strided_slice, adv_index, squeeze

logger = logging.getLogger("topi")


def _get_sort_func(mode=0):
    """Get sort function for argwhere. mode 0 for topk and others for argsort."""
    if get_global_func("tvm.contrib.thrust.sort", allow_missing=True):
        ret = topk_thrust if mode == 0 else argsort_thrust
    else:
        logger.warning(
            "It's highly recommended to enable thrust library with set(USE_THRUST ON)"
            " when compiling argwhere for cuda target. Otherwise, it can result in"
            " significant performance degradation or incorrect result"
        )
        ret = topk if mode == 0 else argsort

    return ret


def argwhere_1d_ir(condition, out):
    """Low level IR for argwhere 1D

    Parameters
    ----------
    condition : Buffer
        The condition buffer.

    out : Buffer
        The output buffer.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    ib = tvm.tir.ir_builder.create()
    a0 = condition.shape[0]

    condition = ib.buffer_ptr(condition)
    out = ib.buffer_ptr(out)

    valid_index = ib.allocate("int32", (1,), name="valid_index", scope="global")
    tmp = ib.allocate("int32", (1,), name="tmp", scope="local")
    one_count = tvm.tir.const(1, dtype="int32")

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_tx = max_threads
    # Limit threads to a single block to make sure atomic_add works normally.
    tx = te.thread_axis("threadIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    len_inner_for = a0 // nthread_tx + 1
    valid_index[0] = 0

    with ib.for_range(0, len_inner_for, name="i") as i:
        idx = tx * len_inner_for + i
        with ib.if_scope(idx < a0):
            with ib.if_scope(condition[idx] != 0):
                tmp[0] = atomic_add(
                    tvm.tir.call_intrin("handle", "tir.address_of", valid_index[0]),
                    one_count,
                )
                out[tmp[0]] = idx

    return ib.get()


def argwhere_1d(output_shape, condition):
    """Compute for argwhere 1D

    Parameters
    ----------
    condition : list of int or tvm.tir.Any
        The output shape

    out : tvm.te.Tensor
        Tensor with boolean values.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    condition_buf = tvm.tir.decl_buffer(
        condition.shape, condition.dtype, "data_buf", data_alignment=8
    )
    out_buf = tvm.tir.decl_buffer(output_shape, "int32", "out_buf", data_alignment=8)

    out = te.extern(
        [output_shape],
        [condition],
        lambda ins, outs: argwhere_1d_ir(ins[0], outs[0]),
        dtype=["int32"],
        in_buffers=[condition_buf],
        out_buffers=[out_buf],
        name="argwhere_1d",
        tag="argwhere1d_gpu",
    )

    if isinstance(out.shape[0], (int, tvm.tir.expr.IntImm)) and int(out.shape[0]) <= 1:
        return out

    sorted_out = _get_sort_func()(
        out, k=0, axis=0, ret_type="values", is_ascend="True", dtype="int32"
    )

    return sorted_out


def argwhere_2d_ir(condition, out):
    """Low level IR for argwhere 2D

    Parameters
    ----------
    condition : Buffer
        The condition buffer.

    out : Buffer
        The output buffer.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    ib = tvm.tir.ir_builder.create()
    a0 = condition.shape[0]
    a1 = condition.shape[1]

    condition = ib.buffer_ptr(condition)
    out = ib.buffer_ptr(out)

    valid_index = ib.allocate("int32", (1,), name="valid_index", scope="local")
    tmp = ib.allocate("int32", (1,), name="tmp", scope="local")
    one_count = tvm.tir.const(1, dtype="int32")

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_tx = max_threads

    # Limit threads to a single block to make sure atomic_add works normally.
    tx = te.thread_axis("threadIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    len_inner_for = (a0 * a1) // nthread_tx + 1

    valid_index[0] = 0

    with ib.for_range(0, len_inner_for, name="i") as i:
        idx = tx * len_inner_for + i
        with ib.if_scope(idx < (a0 * a1)):
            with ib.if_scope(condition[idx] != 0):
                tmp[0] = atomic_add(
                    tvm.tir.call_intrin("handle", "tir.address_of", valid_index[0]),
                    one_count,
                )
                out[tmp[0] * 2] = tvm.tir.floordiv(idx, a1)
                out[tmp[0] * 2 + 1] = tvm.tir.floormod(idx, a1)

    return ib.get()


def argwhere_2d(output_shape, condition):
    """Compute for argwhere 2D

    Parameters
    ----------
    condition : list of int or tvm.tir.Any
        The output shape

    out : tvm.te.Tensor
        Tensor with boolean values.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    condition_buf = tvm.tir.decl_buffer(
        condition.shape, condition.dtype, "data_buf", data_alignment=8
    )
    out_buf = tvm.tir.decl_buffer(output_shape, "int32", "out_buf", data_alignment=8)

    out = te.extern(
        [output_shape],
        [condition],
        lambda ins, outs: argwhere_2d_ir(ins[0], outs[0]),
        dtype=["int32"],
        in_buffers=[condition_buf],
        out_buffers=[out_buf],
        name="argwhere_2d",
        tag="argwhere2d_gpu",
    )

    if isinstance(out.shape[0], (int, tvm.tir.expr.IntImm)) and int(out.shape[0]) <= 1:
        return out

    sort_func = _get_sort_func(1)

    # sort the output from the least significant to the most significant
    # column.
    if isinstance(out.shape[0], (int, tvm.tir.expr.IntImm)):
        out1 = strided_slice(out, [0, 1], [out.shape[0], 2])
        out2 = sort_func(out1, axis=0, dtype="int32")
        out3 = squeeze(out2)
        out = adv_index(out, [out3])

        out1 = strided_slice(out, [0, 0], [out.shape[0], 1])
        out2 = sort_func(out1, axis=0, dtype="int32")
        out3 = squeeze(out2)

        out = adv_index(out, [out3])
    else:
        out1 = strided_slice(out, [0, 1], [out.shape[0], 2], [1, 1])
        out2 = sort_func(out1, axis=0, dtype="int32")
        out3 = squeeze(out2)
        out = adv_index(out, [out3])

        out1 = strided_slice(out, [0, 0], [out.shape[0], 1], [1, 1])
        out2 = sort_func(out1, axis=0, dtype="int32")
        out3 = squeeze(out2)
        out = adv_index(out, [out3])
    return out


def argwhere_3d_ir(condition, out):
    """Low level IR for argwhere 3D

    Parameters
    ----------
    condition : Buffer
        The condition buffer.

    out : Buffer
        The output buffer.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    ib = tvm.tir.ir_builder.create()
    a0 = condition.shape[0]
    a1 = condition.shape[1]
    a2 = condition.shape[2]
    s1 = a1 * a2
    s0 = a0 * s1

    condition = ib.buffer_ptr(condition)
    out = ib.buffer_ptr(out)

    valid_index = ib.allocate("int32", (1,), name="valid_index", scope="local")
    tmp = ib.allocate("int32", (1,), name="tmp", scope="local")
    one_count = tvm.tir.const(1, dtype="int32")

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_tx = max_threads

    # Limit threads to a single block to make sure atomic_add works normally.
    tx = te.thread_axis("threadIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    len_inner_for = s0 // nthread_tx + 1

    fdiv = tvm.tir.floordiv
    fmod = tvm.tir.floormod

    valid_index[0] = 0

    with ib.for_range(0, len_inner_for, name="i") as i:
        idx = tx * len_inner_for + i
        with ib.if_scope(idx < s0):
            with ib.if_scope(condition[idx] != 0):
                tmp[0] = atomic_add(
                    tvm.tir.call_intrin("handle", "tir.address_of", valid_index[0]),
                    one_count,
                )
                out[tmp[0] * 3] = fdiv(idx, s1)
                out[tmp[0] * 3 + 1] = fdiv(fmod(idx, s1), a2)
                out[tmp[0] * 3 + 2] = fmod(idx, a2)

    return ib.get()


def argwhere_3d(output_shape, condition):
    """Compute for argwhere 3D

    Parameters
    ----------
    condition : list of int or tvm.tir.Any
        The output shape

    out : tvm.te.Tensor
        Tensor with boolean values.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    condition_buf = tvm.tir.decl_buffer(
        condition.shape, condition.dtype, "data_buf", data_alignment=8
    )
    out_buf = tvm.tir.decl_buffer(output_shape, "int32", "out_buf", data_alignment=8)

    out = te.extern(
        [output_shape],
        [condition],
        lambda ins, outs: argwhere_3d_ir(ins[0], outs[0]),
        dtype=["int32"],
        in_buffers=[condition_buf],
        out_buffers=[out_buf],
        name="argwhere_3d",
        tag="argwhere3d_gpu",
    )

    if isinstance(out.shape[0], (int, tvm.tir.expr.IntImm)) and int(out.shape[0]) <= 1:
        return out

    # sort the output from the least significant to the most significant
    # column.
    sort_func = _get_sort_func(1)

    if isinstance(out.shape[0], (int, tvm.tir.expr.IntImm)):
        for i in reversed(range(3)):
            out1 = strided_slice(out, [0, i], [out.shape[0], i + 1])
            out2 = sort_func(out1, axis=0, dtype="int32")
            out3 = squeeze(out2)
            out = adv_index(out, [out3])
    else:
        for i in reversed(range(3)):
            out1 = strided_slice(out, [0, i], [out.shape[0], i + 1], [1, 1])
            out2 = sort_func(out1, axis=0, dtype="int32")
            out3 = squeeze(out2)
            out = adv_index(out, [out3])
    return out


def argwhere_4d_ir(condition, out):
    """Low level IR for argwhere 4D

    Parameters
    ----------
    condition : Buffer
        The condition buffer.

    out : Buffer
        The output buffer.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    ib = tvm.tir.ir_builder.create()
    a0 = condition.shape[0]
    a1 = condition.shape[1]
    a2 = condition.shape[2]
    a3 = condition.shape[3]
    s1 = a2 * a3
    s2 = a1 * s1
    s0 = a0 * s2

    condition = ib.buffer_ptr(condition)
    out = ib.buffer_ptr(out)

    valid_index = ib.allocate("int32", (1,), name="valid_index", scope="local")
    tmp = ib.allocate("int32", (1,), name="tmp", scope="local")
    one_count = tvm.tir.const(1, dtype="int32")

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_tx = max_threads

    # Limit threads to a single block to make sure atomic_add works normally.
    tx = te.thread_axis("threadIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    len_inner_for = s0 // nthread_tx + 1

    fdiv = tvm.tir.floordiv
    fmod = tvm.tir.floormod

    valid_index[0] = 0

    with ib.for_range(0, len_inner_for, name="i") as i:
        idx = tx * len_inner_for + i
        with ib.if_scope(idx < s0):
            with ib.if_scope(condition[idx] != 0):
                tmp[0] = atomic_add(
                    tvm.tir.call_intrin("handle", "tir.address_of", valid_index[0]),
                    one_count,
                )
                out[tmp[0] * 4] = fdiv(idx, s2)
                out[tmp[0] * 4 + 1] = fdiv(fmod(idx, s2), s1)
                out[tmp[0] * 4 + 2] = fdiv(fmod(idx, s1), a3)
                out[tmp[0] * 4 + 3] = fmod(idx, a3)

    return ib.get()


def argwhere_4d(output_shape, condition):
    """Compute for argwhere 4D

    Parameters
    ----------
    condition : list of int or tvm.tir.Any
        The output shape

    out : tvm.te.Tensor
        Tensor with boolean values.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    condition_buf = tvm.tir.decl_buffer(
        condition.shape, condition.dtype, "data_buf", data_alignment=8
    )
    out_buf = tvm.tir.decl_buffer(output_shape, "int32", "out_buf", data_alignment=8)

    out = te.extern(
        [output_shape],
        [condition],
        lambda ins, outs: argwhere_4d_ir(ins[0], outs[0]),
        dtype=["int32"],
        in_buffers=[condition_buf],
        out_buffers=[out_buf],
        name="argwhere_4d",
        tag="argwhere4d_gpu",
    )

    if isinstance(out.shape[0], (int, tvm.tir.expr.IntImm)) and int(out.shape[0]) <= 1:
        return out

    # sort the output from the least significant to the most significant
    # column.
    sort_func = _get_sort_func(1)
    if isinstance(out.shape[0], (int, tvm.tir.expr.IntImm)):
        for i in reversed(range(4)):
            out1 = strided_slice(out, [0, i], [out.shape[0], i + 1])
            out2 = sort_func(out1, axis=0, dtype="int32")
            out3 = squeeze(out2)
            out = adv_index(out, [out3])
    else:
        for i in reversed(range(4)):
            out1 = strided_slice(out, [0, i], [out.shape[0], i + 1], [1, 1])
            out2 = sort_func(out1, axis=0, dtype="int32")
            out3 = squeeze(out2)
            out = adv_index(out, [out3])

    return out


def argwhere_5d_ir(condition, out):
    """Low level IR for argwhere 5D

    Parameters
    ----------
    condition : Buffer
        The condition buffer.

    out : Buffer
        The output buffer.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    ib = tvm.tir.ir_builder.create()
    a0 = condition.shape[0]
    a1 = condition.shape[1]
    a2 = condition.shape[2]
    a3 = condition.shape[3]
    a4 = condition.shape[4]
    s1 = a3 * a4
    s2 = a2 * s1
    s3 = a1 * s2
    s0 = a0 * s3

    condition = ib.buffer_ptr(condition)
    out = ib.buffer_ptr(out)

    valid_index = ib.allocate("int32", (1,), name="valid_index", scope="local")
    tmp = ib.allocate("int32", (1,), name="tmp", scope="local")
    one_count = tvm.tir.const(1, dtype="int32")

    max_threads = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_tx = max_threads

    # Limit threads to a single block to make sure atomic_add works normally.
    tx = te.thread_axis("threadIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    len_inner_for = s0 // nthread_tx + 1

    fdiv = tvm.tir.floordiv
    fmod = tvm.tir.floormod

    valid_index[0] = 0

    with ib.for_range(0, len_inner_for, name="i") as i:
        idx = tx * len_inner_for + i
        with ib.if_scope(idx < s0):
            with ib.if_scope(condition[idx] != 0):
                tmp[0] = atomic_add(
                    tvm.tir.call_intrin("handle", "tir.address_of", valid_index[0]),
                    one_count,
                )
                out[tmp[0] * 5] = fdiv(idx, s3)
                out[tmp[0] * 5 + 1] = fdiv(fmod(idx, s3), s2)
                out[tmp[0] * 5 + 2] = fdiv(fmod(idx, s2), s1)
                out[tmp[0] * 5 + 3] = fdiv(fmod(idx, s1), a4)
                out[tmp[0] * 5 + 4] = fmod(idx, a4)

    return ib.get()


def argwhere_5d(output_shape, condition):
    """Compute for argwhere 5D

    Parameters
    ----------
    condition : list of int or tvm.tir.Any
        The output shape

    out : tvm.te.Tensor
        Tensor with boolean values.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """
    condition_buf = tvm.tir.decl_buffer(
        condition.shape, condition.dtype, "data_buf", data_alignment=8
    )
    out_buf = tvm.tir.decl_buffer(output_shape, "int32", "out_buf", data_alignment=8)

    out = te.extern(
        [output_shape],
        [condition],
        lambda ins, outs: argwhere_5d_ir(ins[0], outs[0]),
        dtype=["int32"],
        in_buffers=[condition_buf],
        out_buffers=[out_buf],
        name="argwhere_5d",
        tag="argwhere5d_gpu",
    )

    if isinstance(out.shape[0], (int, tvm.tir.expr.IntImm)) and int(out.shape[0]) <= 1:
        return out

    # sort the output from the least significant to the most significant
    # column.
    sort_func = _get_sort_func(1)
    if isinstance(out.shape[0], (int, tvm.tir.expr.IntImm)):
        for i in reversed(range(5)):
            out1 = strided_slice(out, [0, i], [out.shape[0], i + 1])
            out2 = sort_func(out1, axis=0, dtype="int32")
            out3 = squeeze(out2)
            out = adv_index(out, [out3])
    else:
        for i in reversed(range(5)):
            out1 = strided_slice(out, [0, i], [out.shape[0], i + 1], [1, 1])
            out2 = sort_func(out1, axis=0, dtype="int32")
            out3 = squeeze(out2)
            out = adv_index(out, [out3])

    return out


def argwhere(output_shape, condition):
    """Find the indices of elements of a tensor that are non-zero.

    Parameters
    ----------
    output_shape : tvm.te.Tensor
        Tensor with output shape info.

    condition : tvm.te.Tensor
        Tensor with boolean values.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """
    if len(condition.shape) == 1:
        return argwhere_1d(output_shape.shape, condition)
    if len(condition.shape) == 2:
        return argwhere_2d(output_shape.shape, condition)
    if len(condition.shape) == 3:
        return argwhere_3d(output_shape.shape, condition)
    if len(condition.shape) == 4:
        return argwhere_4d(output_shape.shape, condition)
    if len(condition.shape) == 5:
        return argwhere_5d(output_shape.shape, condition)
    raise ValueError("Argwhere does not support rank higher than 5")


def schedule_argwhere(outs):
    """Schedule for argwhere on cuda.

    Parameters
    ----------
    outs: Array of Tensor
        The computation graph description of argwhere
        in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for argwhere
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
