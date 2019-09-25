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
"""Backend compiler related feature registration"""
# pylint: disable=invalid-name,unused-argument, len-as-condition
from __future__ import absolute_import
from topi.util import get_const_int, get_const_tuple
from . import op as _reg
from ._reduce import _schedule_reduce
from .op import OpPattern
from ...hybrid import script
from ...api import convert

schedule_injective = _reg.schedule_injective
schedule_broadcast = _reg.schedule_injective
schedule_concatenate = _reg.schedule_concatenate


_reg.register_schedule("collapse_sum_like", _schedule_reduce)
_reg.register_schedule("broadcast_to", schedule_broadcast)
_reg.register_schedule("broadcast_to_like", schedule_broadcast)
_reg.register_schedule("expand_dims", schedule_broadcast)
_reg.register_schedule("squeeze", schedule_injective)
_reg.register_schedule("reshape", schedule_injective)
_reg.register_schedule("reshape_like", schedule_injective)
_reg.register_schedule("full", schedule_injective)
_reg.register_schedule("full_like", schedule_injective)
_reg.register_schedule("arange", schedule_injective)
_reg.register_schedule("reverse", schedule_injective)
_reg.register_schedule("repeat", schedule_broadcast)
_reg.register_schedule("tile", schedule_broadcast)
_reg.register_schedule("cast", schedule_injective)
_reg.register_schedule("cast_like", schedule_injective)
_reg.register_schedule("reinterpret", schedule_injective)
_reg.register_schedule("strided_slice", schedule_injective)
_reg.register_schedule("slice_like", schedule_injective)
_reg.register_schedule("split", schedule_injective)
_reg.register_schedule("take", schedule_injective)
_reg.register_schedule("transpose", schedule_injective)
_reg.register_schedule("where", schedule_broadcast)
_reg.register_schedule("stack", schedule_injective)
_reg.register_schedule("concatenate", schedule_concatenate)
_reg.register_schedule("_contrib_reverse_reshape", schedule_injective)
_reg.register_schedule("gather_nd", schedule_injective)
_reg.register_schedule("sequence_mask", schedule_injective)
_reg.register_schedule("one_hot", schedule_injective)


# layout_transform
_reg.register_schedule("layout_transform", schedule_injective)
_reg.register_pattern("layout_transform", OpPattern.INJECTIVE)

# shape func
@script
def _arange_shape_func(start, stop, step):
    out = output_tensor((1,), "int64")
    out[0] = int64(ceil_div((float32(stop[0]) - float32(start[0])), float32(step[0])))
    return out

@_reg.register_shape_func("arange", True)
def arange_shape_func(attrs, inputs, _):
    return [_arange_shape_func(*inputs)]

@script
def _concatenate_shape_func(inputs, axis):
    ndim = inputs[0].shape[0]
    out = output_tensor((ndim,), "int64")
    for i in const_range(ndim):
        if i != axis:
            out[i] = inputs[0][i]
            for j in const_range(1, len(inputs)):
                assert out[i] == inputs[j][i], \
                    "Dims mismatch in the inputs of concatenate."
        else:
            out[i] = int64(0)
            for j in const_range(len(inputs)):
                out[i] += inputs[j][i]
    return out

@_reg.register_shape_func("concatenate", False)
def concatenate_shape_func(attrs, inputs, _):
    axis = get_const_int(attrs.axis)
    return [_concatenate_shape_func(inputs, convert(axis))]

@script
def _reshape_shape_func(data_shape, newshape, ndim):
    out = output_tensor((ndim,), "int64")
    src_idx = 0
    dst_idx = 0
    infer_idx = -1
    copy = False
    skip = 0
    for i in const_range(len(newshape)):
        if skip > 0:
            skip -= 1
        elif newshape[i] > 0:
            out[dst_idx] = int64(newshape[i])
            src_idx += 1
            dst_idx += 1
        elif newshape[i] == 0:
            out[dst_idx] = data_shape[src_idx]
            src_idx += 1
            dst_idx += 1
        elif newshape[i] == -1:
            assert infer_idx < 0, "One and only one dim can be inferred"
            out[dst_idx] = int64(1)
            infer_idx = i
            dst_idx += 1
        elif newshape[i] == -2:
            copy = True
        elif newshape[i] == -3:
            assert data_shape.shape[0] - src_idx > 1, \
                "Not enough dims in input shape for -3"
            out[dst_idx] = data_shape[src_idx] * data_shape[src_idx+1]
            src_idx += 2
            dst_idx += 1
        elif newshape[i] == -4:
            assert len(newshape) - i > 2, "Not enough dims in new shape for -4"
            if newshape[i+1] == -1:
                assert newshape[i+2] != -1, "Split dims cannot both be -1."
                out[dst_idx] = data_shape[src_idx] / int64(newshape[i+2])
                out[dst_idx+1] = int64(newshape[i+2])
            else:
                out[dst_idx] = int64(newshape[i+1])
                if newshape[i+2] == -1:
                    out[dst_idx+1] = data_shape[src_idx] / int64(newshape[i+1])
                else:
                    out[dst_idx+1] = int64(newshape[i+2])
            assert data_shape[src_idx] == out[dst_idx] * out[dst_idx+1],\
                "Product of split dims doesn't match to input dim"
            src_idx += 1
            dst_idx += 2
            skip = 2
        else:
            assert False, "Invalid special values in new shape"
    if len(data_shape.shape) > 0:
        # if data is not constant, we can then handle -1 and -2
        if copy:
            for i in range(src_idx, data_shape.shape[0]):
                out[dst_idx] = data_shape[i]
                dst_idx += 1
        if infer_idx >= 0:
            old_size = int64(1)
            for i in const_range(data_shape.shape[0]):
                old_size *= data_shape[i]
            new_size = int64(1)
            for i in const_range(out.shape[0]):
                new_size *= out[i]
            out[infer_idx] = old_size / new_size
    return out

@_reg.register_shape_func("reshape", False)
def reshape_shape_func(attrs, inputs, out_ndims):
    newshape = get_const_tuple(attrs.newshape)
    return [_reshape_shape_func(inputs[0], convert(newshape), out_ndims[0])]

@script
def _take_no_axis_shape_func(indices_shape, out_ndim):
    out = output_tensor((out_ndim,), "int64")
    for i in const_range(out_ndim):
        out[i] = indices_shape[i]
    return out

@script
def _take_with_axis_shape_func(data_shape, indices_shape, axis, out_ndim):
    out = output_tensor((out_ndim,), "int64")
    for i in const_range(axis):
        out[i] = data_shape[i]
    if len(indices_shape.shape) == 0:
        # indices is constant
        for i in const_range(axis+1, len(data_shape)):
            out[i-1] = data_shape[i]
    else:
        for i in const_range(len(indices_shape)):
            out[axis+i] = indices_shape[i]
        for i in const_range(axis+1, len(data_shape)):
            out[len(indices_shape)+i-1] = data_shape[i]
    return out

@_reg.register_shape_func("take", False)
def take_shape_func(attrs, inputs, out_ndims):
    """
    Shape function for take op.
    """
    if attrs.axis is None:
        return [_take_no_axis_shape_func(inputs[1], out_ndims[0])]
    else:
        axis = get_const_int(attrs.axis)
        data_ndim = int(inputs[0].shape[0])
        if axis < 0:
            axis += data_ndim
        assert 0 <= axis < data_ndim
        return [_take_with_axis_shape_func(*inputs, convert(axis), out_ndims[0])]
