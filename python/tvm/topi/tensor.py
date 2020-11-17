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
# pylint: disable=invalid-name,consider-using-enumerate,unused-argument,len-as-condition
"""Elementwise operators"""
from __future__ import absolute_import as _abs
from . import cpp
from .. import te, tir
from .utils import get_const_tuple


def elemwise_sum(xs):
    """Perform element-wise sum on inputs

    Parameters
    ----------
    xs : list of tvm.te.Tensor
        Input arguments.

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.elemwise_sum(xs)


def full(shape, dtype, fill_value):
    """Fill tensor with fill_value

    Parameters
    ----------
    shape : tuple
        Input tensor shape.
    dtype : str
        Data type
    fill_value : float
        Value to be filled

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.full(shape, dtype, fill_value)


def full_like(x, fill_value):
    """Construct a tensor with same shape as input tensor,
       then fill tensor with fill_value.

    Parameters
    ----------
    x : tvm.te.Tensor
        Input argument.
    fill_value : float
        Value to be filled

    Returns
    -------
    y : tvm.te.Tensor
        The result.
    """
    return cpp.full_like(x, fill_value)


def segment_max(data, segment_ids, num_out):
    """segment_max operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        input data

    segment_ids : tvm.te.Tensor
        input segment ids

    num_out : int
        number of output

    Returns
    -------
    out : tvm.te.Tensor
        Tensor with shape determined by the segment ids.
    """

    def _segment_max(data, segment_ids, out_buf):

        ib = tir.ir_builder.create()
        input_data = ib.buffer_ptr(data)
        seg_ids = ib.buffer_ptr(segment_ids)
        out = ib.buffer_ptr(out_buf)

        shape = get_const_tuple(data.shape)
        num_segment = get_const_tuple(out_buf.shape)[0]
        inner_size = 1
        for s in range(1, len(shape)):
            inner_size = inner_size * shape[s]

        with ib.for_range(0, num_segment) as n:
            with ib.for_range(0, inner_size) as j:
                out_index = n * inner_size + j
                out[out_index] = -3.4028235e38

            with ib.for_range(0, shape[0]) as k:
                with ib.if_scope(seg_ids[k] == n):
                    with ib.for_range(0, inner_size) as l:
                        out_index = n * inner_size + l
                        in_index = k * inner_size + l
                        out[out_index] = te.max(input_data[in_index], out[out_index])

        return ib.get()

    assert len(segment_ids.shape) == 1

    out_shape = list(get_const_tuple(data.shape))
    out_shape[0] = num_out

    out = te.extern(
        out_shape,
        [data, segment_ids],
        lambda ins, outs: _segment_max(ins[0], ins[1], outs[0]),
        dtype=data.dtype,
    )

    return out


def segment_min(data, segment_ids, num_out):
    """segment_min operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        input data

    segment_ids : tvm.te.Tensor
        input segment ids

    num_out : int
        number of output

    Returns
    -------
    out : tvm.te.Tensor
        Tensor with shape determined by the segment ids.
    """

    def _segment_min(data, segment_ids, out_buf):

        ib = tir.ir_builder.create()
        input_data = ib.buffer_ptr(data)
        seg_ids = ib.buffer_ptr(segment_ids)
        out = ib.buffer_ptr(out_buf)

        shape = get_const_tuple(data.shape)
        num_segment = get_const_tuple(out_buf.shape)[0]
        inner_size = 1
        for s in range(1, len(shape)):
            inner_size = inner_size * shape[s]

        with ib.for_range(0, num_segment) as n:
            with ib.for_range(0, inner_size) as j:
                out_index = n * inner_size + j
                out[out_index] = 3.4028235e38

            with ib.for_range(0, shape[0]) as k:
                with ib.if_scope(seg_ids[k] == n):
                    with ib.for_range(0, inner_size) as l:
                        out_index = n * inner_size + l
                        in_index = k * inner_size + l
                        out[out_index] = te.min(input_data[in_index], out[out_index])

        return ib.get()

    assert len(segment_ids.shape) == 1

    out_shape = list(get_const_tuple(data.shape))
    out_shape[0] = num_out

    out = te.extern(
        out_shape,
        [data, segment_ids],
        lambda ins, outs: _segment_min(ins[0], ins[1], outs[0]),
        dtype=data.dtype,
    )

    return out


def segment_mean(data, segment_ids, num_out):
    """segment_mean operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        input data

    segment_ids : tvm.te.Tensor
        input segment ids

    num_out : int
        number of output

    Returns
    -------
    out : tvm.te.Tensor
        Tensor with shape determined by the segment ids.
    """

    def _segment_mean(data, segment_ids, out_buf):

        ib = tir.ir_builder.create()
        input_data = ib.buffer_ptr(data)
        seg_ids = ib.buffer_ptr(segment_ids)
        out = ib.buffer_ptr(out_buf)

        temp_index = ib.allocate("int32", (data.shape[0]), name="temp_index", scope="local")
        num = ib.allocate("int32", (1), name="num", scope="local")

        shape = get_const_tuple(data.shape)
        num_segment = get_const_tuple(out_buf.shape)[0]
        inner_size = 1
        for s in range(1, len(shape)):
            inner_size = inner_size * shape[s]

        with ib.for_range(0, num_segment) as n:
            with ib.for_range(0, inner_size) as j:
                out_index = n * inner_size + j
                out[out_index] = 0.0

            num[0] = 0
            with ib.for_range(0, shape[0]) as k:
                with ib.if_scope(seg_ids[k] == n):
                    temp_index[num[0]] = k
                    num[0] += 1

            with ib.if_scope(num[0] > 0):
                with ib.for_range(0, inner_size) as l:
                    out_index = n * inner_size + l
                    with ib.for_range(0, num[0]) as k:
                        in_index = temp_index[k] * inner_size + l
                        out[out_index] += input_data[in_index]
                    out[out_index] = out[out_index] / num[0]

        return ib.get()

    assert len(segment_ids.shape) == 1

    out_shape = list(get_const_tuple(data.shape))
    out_shape[0] = num_out

    out = te.extern(
        out_shape,
        [data, segment_ids],
        lambda ins, outs: _segment_mean(ins[0], ins[1], outs[0]),
        dtype=data.dtype,
    )

    return out


def segment_sum(data, segment_ids, num_out):
    """segment_sum operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        input data

    segment_ids : tvm.te.Tensor
        input segment ids

    num_out : int
        number of output

    Returns
    -------
    out : tvm.te.Tensor
        Tensor with shape determined by the segment ids.
    """

    def _segment_sum(data, segment_ids, out_buf):

        ib = tir.ir_builder.create()
        input_data = ib.buffer_ptr(data)
        seg_ids = ib.buffer_ptr(segment_ids)
        out = ib.buffer_ptr(out_buf)

        temp_index = ib.allocate("int32", (data.shape[0]), name="temp_index", scope="local")
        num = ib.allocate("int32", (1), name="num", scope="local")

        shape = get_const_tuple(data.shape)
        num_segment = get_const_tuple(out_buf.shape)[0]
        inner_size = 1
        for s in range(1, len(shape)):
            inner_size = inner_size * shape[s]

        with ib.for_range(0, num_segment) as n:
            with ib.for_range(0, inner_size) as j:
                out_index = n * inner_size + j
                out[out_index] = 0.0

            num[0] = 0
            with ib.for_range(0, shape[0]) as k:
                with ib.if_scope(seg_ids[k] == n):
                    temp_index[num[0]] = k
                    num[0] += 1

            with ib.if_scope(num[0] > 0):
                with ib.for_range(0, inner_size) as l:
                    out_index = n * inner_size + l
                    with ib.for_range(0, num[0]) as k:
                        in_index = temp_index[k] * inner_size + l
                        out[out_index] += input_data[in_index]

        return ib.get()

    assert len(segment_ids.shape) == 1

    out_shape = list(get_const_tuple(data.shape))
    out_shape[0] = num_out

    out = te.extern(
        out_shape,
        [data, segment_ids],
        lambda ins, outs: _segment_sum(ins[0], ins[1], outs[0]),
        dtype=data.dtype,
    )

    return out


def segment_prod(data, segment_ids, num_out):
    """segment_prod operator.

    Parameters
    ----------
    data : tvm.te.Tensor
        input data

    segment_ids : tvm.te.Tensor
        input segment ids

    num_out : int
        number of output

    Returns
    -------
    out : tvm.te.Tensor
        Tensor with shape determined by the segment ids.
    """

    def _segment_prod(data, segment_ids, out_buf):

        ib = tir.ir_builder.create()
        input_data = ib.buffer_ptr(data)
        seg_ids = ib.buffer_ptr(segment_ids)
        out = ib.buffer_ptr(out_buf)

        temp_index = ib.allocate("int32", (data.shape[0]), name="temp_index", scope="local")
        num = ib.allocate("int32", (1), name="num", scope="local")

        shape = get_const_tuple(data.shape)
        num_segment = get_const_tuple(out_buf.shape)[0]
        inner_size = 1
        for s in range(1, len(shape)):
            inner_size = inner_size * shape[s]

        with ib.for_range(0, num_segment) as n:
            with ib.for_range(0, inner_size) as j:
                out_index = n * inner_size + j
                out[out_index] = 1.0

            num[0] = 0
            with ib.for_range(0, shape[0]) as k:
                with ib.if_scope(seg_ids[k] == n):
                    temp_index[num[0]] = k
                    num[0] += 1

            with ib.if_scope(num[0] > 0):
                with ib.for_range(0, inner_size) as l:
                    out_index = n * inner_size + l
                    with ib.for_range(0, num[0]) as k:
                        in_index = temp_index[k] * inner_size + l
                        out[out_index] *= input_data[in_index]

        return ib.get()

    assert len(segment_ids.shape) == 1

    out_shape = list(get_const_tuple(data.shape))
    out_shape[0] = num_out

    out = te.extern(
        out_shape,
        [data, segment_ids],
        lambda ins, outs: _segment_prod(ins[0], ins[1], outs[0]),
        dtype=data.dtype,
    )

    return out
