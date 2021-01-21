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
from .injective import schedule_injective_from_existing
from .scan import exclusive_scan
from .. import tag
from ..utils import ceil_div, prod
from ..transform import reshape
from ..broadcast import not_equal
from ..math import cast


logger = logging.getLogger("topi")

fdiv = tvm.tir.floordiv
fmod = tvm.tir.floormod


def compact_nonzero_indices_ir(condition, write_indices, out, do_write_func):
    """Copy nonzero indices to the corresponding write locations.

    Parameters
    ----------
    condition : Buffer
        The input condition.

    write_indices : Buffer
        The result of exclusive scan on a boolean array, where True indicates that
        the condition is non zero at that position.

    out : Buffer
        The output buffer to copy indices to.

    do_write_func : a function
        A callback that accepts an output buffer, a dst index to write to, and a src index.

    Returns
    -------
    stmt : Stmt
        The result IR statement.
    """

    ib = tvm.tir.ir_builder.create()
    size_1d = prod(condition.shape)

    condition = ib.buffer_ptr(condition)
    write_indices = ib.buffer_ptr(write_indices)
    out = ib.buffer_ptr(out)

    nthread_tx = int(tvm.target.Target.current(allow_none=False).max_num_threads)
    nthread_bx = ceil_div(size_1d, nthread_tx)
    tx = te.thread_axis("threadIdx.x")
    bx = te.thread_axis("blockIdx.x")
    ib.scope_attr(tx, "thread_extent", nthread_tx)
    ib.scope_attr(bx, "thread_extent", nthread_bx)

    with ib.new_scope():
        idx = bx * nthread_tx + tx
        with ib.if_scope(idx < size_1d):
            with ib.if_scope(condition[idx] != 0):
                do_write_func(out, write_indices[idx], idx)

    return ib.get()


def argwhere_common(output_shape, condition, do_write_func):
    """A common compute used by argwhere of various ranks.

    Parameters
    ----------
    output_shape : list of int or tvm.tir.Any
        Tensor with output shape info.

    condition : tvm.te.Tensor
        The input condition.

    do_write_func : a function
        A callback that accepts an output buffer, a dst index to write to, and a src index.

    Returns
    -------
    out : tvm.te.Tensor
        Indices of non-zero elements.
    """

    flags = not_equal(condition, tvm.tir.const(0))
    flags_1d = reshape(flags, (prod(flags.shape),))
    write_indices = exclusive_scan(cast(flags_1d, dtype="int32"))

    condition_buf = tvm.tir.decl_buffer(
        condition.shape, condition.dtype, "data_buf", data_alignment=8
    )
    write_indices_buf = tvm.tir.decl_buffer(
        write_indices.shape, write_indices.dtype, "write_indices_buf", data_alignment=8
    )
    out_buf = tvm.tir.decl_buffer(output_shape, "int32", "out_buf", data_alignment=8)

    out = te.extern(
        [output_shape],
        [condition, write_indices],
        lambda ins, outs: compact_nonzero_indices_ir(ins[0], ins[1], outs[0], do_write_func),
        dtype=["int32"],
        in_buffers=[condition_buf, write_indices_buf],
        out_buffers=[out_buf],
        name="argwhere",
        tag="argwhere_gpu",
    )

    return out


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

    def do_write(out, write_index, idx):
        out[write_index] = idx

    return argwhere_common(output_shape, condition, do_write)


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

    def do_write(out, write_index, idx):
        a1 = condition.shape[1]
        out[write_index * 2] = tvm.tir.floordiv(idx, a1)
        out[write_index * 2 + 1] = tvm.tir.floormod(idx, a1)

    return argwhere_common(output_shape, condition, do_write)


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

    def do_write(out, write_index, idx):
        _, a1, a2 = condition.shape
        s1 = a1 * a2
        out[write_index * 3] = fdiv(idx, s1)
        out[write_index * 3 + 1] = fdiv(fmod(idx, s1), a2)
        out[write_index * 3 + 2] = fmod(idx, a2)

    return argwhere_common(output_shape, condition, do_write)


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

    def do_write(out, write_index, idx):
        _, a1, a2, a3 = condition.shape
        s1 = a2 * a3
        s2 = a1 * s1
        out[write_index * 4] = fdiv(idx, s2)
        out[write_index * 4 + 1] = fdiv(fmod(idx, s2), s1)
        out[write_index * 4 + 2] = fdiv(fmod(idx, s1), a3)
        out[write_index * 4 + 3] = fmod(idx, a3)

    return argwhere_common(output_shape, condition, do_write)


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

    def do_write(out, write_index, idx):
        _, a1, a2, a3, a4 = condition.shape
        s1 = a3 * a4
        s2 = a2 * s1
        s3 = a1 * s2
        out[write_index * 5] = fdiv(idx, s3)
        out[write_index * 5 + 1] = fdiv(fmod(idx, s3), s2)
        out[write_index * 5 + 2] = fdiv(fmod(idx, s2), s1)
        out[write_index * 5 + 3] = fdiv(fmod(idx, s1), a4)
        out[write_index * 5 + 4] = fmod(idx, a4)

    return argwhere_common(output_shape, condition, do_write)


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
