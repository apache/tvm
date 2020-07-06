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
# pylint: disable=invalid-name,unused-argument, len-as-condition, too-many-nested-blocks, too-many-local-variables, too-many-arguments
from __future__ import absolute_import
import tvm
from tvm import te
from tvm.te.hybrid import script
from tvm.runtime import convert
import topi
from topi.util import get_const_int, get_const_tuple
from . import op as _reg
from . import strategy
from .op import OpPattern
from ._tensor import elemwise_shape_func

_reg.register_broadcast_schedule("broadcast_to")
_reg.register_broadcast_schedule("broadcast_to_like")
_reg.register_broadcast_schedule("expand_dims")
_reg.register_broadcast_schedule("repeat")
_reg.register_broadcast_schedule("tile")
_reg.register_broadcast_schedule("where")
_reg.register_injective_schedule("squeeze")
_reg.register_injective_schedule("reshape")
_reg.register_injective_schedule("reshape_like")
_reg.register_injective_schedule("full")
_reg.register_injective_schedule("full_like")
_reg.register_injective_schedule("arange")
_reg.register_injective_schedule("meshgrid")
_reg.register_injective_schedule("reverse")
_reg.register_injective_schedule("reverse_sequence")
_reg.register_injective_schedule("cast")
_reg.register_injective_schedule("cast_like")
_reg.register_injective_schedule("reinterpret")
_reg.register_injective_schedule("strided_slice")
_reg.register_injective_schedule("slice_like")
_reg.register_injective_schedule("split")
_reg.register_injective_schedule("take")
_reg.register_injective_schedule("transpose")
_reg.register_injective_schedule("stack")
_reg.register_injective_schedule("_contrib_reverse_reshape")
_reg.register_injective_schedule("gather")
_reg.register_injective_schedule("gather_nd")
_reg.register_injective_schedule("sequence_mask")
_reg.register_injective_schedule("one_hot")
_reg.register_reduce_schedule("collapse_sum_like")
_reg.register_reduce_schedule("collapse_sum_to")
_reg.register_injective_schedule("unravel_index")
_reg.register_injective_schedule("sparse_to_dense")

# concatenate
_reg.register_schedule("concatenate", strategy.schedule_concatenate)

# strided_set
@_reg.register_compute("strided_set")
def compute_strided_set(attrs, inputs, output_type):
    """Compute definition of strided_set"""
    return [topi.strided_set(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])]

_reg.register_injective_schedule("strided_set")

# layout_transform
_reg.register_injective_schedule("layout_transform")
_reg.register_pattern("layout_transform", OpPattern.INJECTIVE)

# argwhere
@_reg.register_compute("argwhere")
def compute_argwhere(attrs, inputs, output_type):
    """Compute definition of argwhere"""
    output_shape = []
    for s in output_type.shape:
        if hasattr(s, "value"):
            output_shape.append(s)
        else:
            # see Any, replace it with a var
            output_shape.append(te.var("any_dim", "int32"))
    new_output_type = tvm.relay.ty.TensorType(output_shape, "int32")
    return [topi.argwhere(new_output_type, inputs[0])]

_reg.register_schedule("argwhere", strategy.schedule_argwhere)

# scatter
@_reg.register_compute("scatter")
def compute_scatter(attrs, inputs, output_type):
    """Compute definition of scatter"""
    return [topi.scatter(inputs[0], inputs[1], inputs[2], attrs.axis)]

_reg.register_schedule("scatter", strategy.schedule_scatter)

#####################
#  Shape functions  #
#####################

@script
def _arange_shape_func(start, stop, step):
    out = output_tensor((1,), "int64")
    out[0] = int64(ceil_div((int64(stop[0]) - int64(start[0])), int64(step[0])))
    return out

@_reg.register_shape_func("arange", True)
def arange_shape_func(attrs, inputs, _):
    """
    Shape func for arange
    """
    return [_arange_shape_func(*inputs)]

@script
def _strided_slice_shape_func_input_data(data, begin, end, strides,
                                         slice_mode):
    ndim = len(data.shape)
    out = output_tensor((ndim,), "int64")
    for i in const_range(ndim):
        cbegin = 0
        cend = data.shape[i]
        cstride = 1
        if strides.shape[0] > i:
            cstride = strides[i]
        if begin.shape[0] > i:
            cbegin = begin[i]
        if end.shape[0] <= i:
            cend = data.shape[i]
        elif slice_mode != 0:
            cstride = 1
            if end[i] < 0:
                cend = data.shape[i]
            else:
                cend = cbegin + end[i]
        else:
            cend = end[i]
        assert cstride != 0, "Strides can't be zero."
        out[i] = int64(ceil_div((int64(cend) - int64(cbegin)), int64(cstride)))
    return out

@script
def _strided_slice_shape_func_input_shape(data_shape, begin, end, strides, slice_mode):
    ndim = data_shape.shape[0]
    out = output_tensor((ndim,), "int64")
    for i in const_range(ndim):
        cbegin = int64(0)
        cend = int64(data_shape[i])
        cstride = int64(1)
        if len(strides) > i:
            cstride = int64(strides[i])
        if len(begin) > i:
            cbegin = int64(begin[i])
        if len(end) <= i:
            cend = int64(data_shape[i])
        elif slice_mode != 0:
            cstride = int64(1)
            if end[i] < 0:
                cend = int64(data_shape[i])
            else:
                cend = cbegin + int64(end[i])
        else:
            cend = int64(end[i])
        assert cstride != 0, "Strides can't be zero."
        out[i] = int64(ceil_div((int64(cend) - int64(cbegin)), int64(cstride)))
    return out


@_reg.register_shape_func("strided_slice", True)
def strided_slice_shape_func(attrs, inputs, _):
    """
    Shape func for strided_slice
    """
    slice_mode = convert(0 if attrs.slice_mode == "end" else 1)
    # data independent if begin, end and strides exist
    if attrs.begin and attrs.end and attrs.strides:
        return [_strided_slice_shape_func_input_shape(inputs[0], attrs.begin, attrs.end,
                                                      attrs.strides, slice_mode)]
    return [_strided_slice_shape_func_input_data(*inputs, slice_mode)]

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
    if axis < 0:
        axis += inputs[0].shape[0]
    return [_concatenate_shape_func(inputs, convert(axis))]

@script
def _reshape_shape_func_input_shape(data_shape, newshape, ndim):
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
                out[dst_idx] = data_shape[src_idx] // int64(newshape[i+2])
                out[dst_idx+1] = int64(newshape[i+2])
            else:
                out[dst_idx] = int64(newshape[i+1])
                if newshape[i+2] == -1:
                    out[dst_idx+1] = data_shape[src_idx] // int64(newshape[i+1])
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
            out[infer_idx] = old_size // new_size
    return out

@_reg.register_shape_func("reshape", False)
def reshape_shape_func(attrs, inputs, out_ndims):
    newshape = get_const_tuple(attrs.newshape)
    return [_reshape_shape_func_input_shape(inputs[0],
                                            convert(newshape),
                                            out_ndims[0])]

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
    axis = get_const_int(attrs.axis)
    data_ndim = int(inputs[0].shape[0])
    if axis < 0:
        axis += data_ndim
    assert 0 <= axis < data_ndim
    return [_take_with_axis_shape_func(*inputs, convert(axis), out_ndims[0])]

@script
def _argwhere_shape_func_1d(condition):
    out = output_tensor((2, ), "int64")
    out[0] = int64(0)
    out[1] = int64(1)
    for i1 in range(condition.shape[0]):
        if condition[i1] != 0:
            out[0] += int64(1)
    return out

@script
def _argwhere_shape_func_2d(condition):
    out = output_tensor((2, ), "int64")
    out[0] = int64(0)
    out[1] = int64(2)
    for i1 in range(condition.shape[0]):
        for i2 in range(condition.shape[1]):
            if condition[i1, i2] != 0:
                out[0] += int64(1)
    return out

@script
def _argwhere_shape_func_3d(condition):
    out = output_tensor((2, ), "int64")
    out[0] = int64(0)
    out[1] = int64(3)
    for i1 in range(condition.shape[0]):
        for i2 in range(condition.shape[1]):
            for i3 in range(condition.shape[2]):
                if condition[i1, i2, i3] != 0:
                    out[0] += int64(1)
    return out

@script
def _argwhere_shape_func_4d(condition):
    out = output_tensor((2, ), "int64")
    out[0] = int64(0)
    out[1] = int64(4)
    for i1 in range(condition.shape[0]):
        for i2 in range(condition.shape[1]):
            for i3 in range(condition.shape[2]):
                for i4 in range(condition.shape[3]):
                    if condition[i1, i2, i3, i4] != 0:
                        out[0] += int64(1)
    return out

@script
def _argwhere_shape_func_5d(condition):
    out = output_tensor((2, ), "int64")
    out[0] = int64(0)
    out[1] = int64(5)
    for i1 in range(condition.shape[0]):
        for i2 in range(condition.shape[1]):
            for i3 in range(condition.shape[2]):
                for i4 in range(condition.shape[3]):
                    for i5 in range(condition.shape[4]):
                        if condition[i1, i2, i3, i4, i5] != 0:
                            out[0] += int64(1)
    return out

@_reg.register_shape_func("argwhere", True)
def argwhere_shape_func(attrs, inputs, out_ndims):
    """
    Shape function for argwhere.
    """
    if len(inputs[0].shape) == 1:
        return [_argwhere_shape_func_1d(inputs[0])]
    if len(inputs[0].shape) == 2:
        return [_argwhere_shape_func_2d(inputs[0])]
    if len(inputs[0].shape) == 3:
        return [_argwhere_shape_func_3d(inputs[0])]
    if len(inputs[0].shape) == 4:
        return [_argwhere_shape_func_4d(inputs[0])]
    if len(inputs[0].shape) == 5:
        return [_argwhere_shape_func_5d(inputs[0])]
    return ValueError("Does not support rank higher than 5 in argwhere")

_reg.register_shape_func("scatter", False, elemwise_shape_func)

@script
def _layout_transform_shape_func(data_shape,
                                 out_layout_len,
                                 dst_equal_list,
                                 dst_mul_list,
                                 dst_div_list,
                                 dst_mix_list):
    out = output_tensor((out_layout_len,), "int64")
    for i in const_range(len(dst_equal_list)):
        out[dst_equal_list[i][0]] = data_shape[dst_equal_list[i][1]]
    for i in const_range(len(dst_mul_list)):
        out[dst_mul_list[i][0]] = data_shape[dst_mul_list[i][1]] * \
                                  data_shape[dst_mul_list[i][2]]
    for i in const_range(len(dst_div_list)):
        out[dst_div_list[i][0]] = data_shape[dst_div_list[i][1]] \
                                  // dst_div_list[i][3]
        out[dst_div_list[i][2]] = int64(dst_div_list[i][3])
    for i in const_range(len(dst_mix_list)):
        out[dst_mix_list[i][0]] = data_shape[dst_mix_list[i][1]] * \
                                  dst_mix_list[i][2] // dst_mix_list[i][4]
        out[dst_mix_list[i][3]] = int64(dst_mix_list[i][4])

    return out

@_reg.register_shape_func("layout_transform", False)
def layout_transform_shape_func(attrs, inputs, _):
    """
    Shape function for layout_transform op.
    """
    def _fetch_axis(layout):
        major_axes = []
        minor_axes = {}
        num_start = -1
        for i, item in enumerate(layout):
            if "A" <= item <= "Z":
                major_axes.append(item)
            elif "a" <= item <= "z":
                last_num = int(layout[num_start:i])
                minor_axes[item] = last_num
                num_start = -1
            elif num_start < 0:
                num_start = i
        return major_axes, minor_axes

    _, src_minor_axes = _fetch_axis(attrs.src_layout)
    dst_major_axes, dst_minor_axes = _fetch_axis(attrs.dst_layout)
    src_letter_list = []
    dst_letter_list = []
    for item in attrs.src_layout:
        if "A" <= item <= "Z" or "a" <= item <= "z":
            src_letter_list.append(item)
    for item in attrs.dst_layout:
        if "A" <= item <= "Z" or "a" <= item <= "z":
            dst_letter_list.append(item)
    out_layout_len = len(dst_major_axes) + len(dst_minor_axes)
    dst_equal_list = []
    dst_mul_list = []
    dst_div_list = []
    dst_mix_list = []

    for key in dst_major_axes:
        if key.lower() not in dst_minor_axes:
            if key.lower() not in src_minor_axes:
                dst_equal_list.append((dst_letter_list.index(key),
                                       src_letter_list.index(key)))
            else:
                dst_mul_list.append((dst_letter_list.index(key),
                                     src_letter_list.index(key),
                                     src_letter_list.index(key.lower())))
        else:
            if key.lower() not in src_minor_axes:
                dst_div_list.append((dst_letter_list.index(key),
                                     src_letter_list.index(key),
                                     dst_letter_list.index(key.lower()),
                                     dst_minor_axes[key.lower()]))
            else:
                dst_mix_list.append((dst_letter_list.index(key),
                                     src_letter_list.index(key),
                                     src_minor_axes[key.lower()],
                                     dst_letter_list.index(key.lower()),
                                     dst_minor_axes[key.lower()]))

    return [_layout_transform_shape_func(inputs[0],
                                         convert(out_layout_len),
                                         convert(dst_equal_list),
                                         convert(dst_mul_list),
                                         convert(dst_div_list),
                                         convert(dst_mix_list))]

@script
def _expand_dim_shape_func(data_shape, ndim, axis, num_newaxis):
    out = output_tensor((ndim + num_newaxis,), "int64")
    for i in const_range(out.shape[0]):
        if i < axis:
            out[i] = data_shape[i]
        elif i < axis + num_newaxis:
            out[i] = int64(1)
        else:
            out[i] = data_shape[i - num_newaxis]

    return out

@_reg.register_shape_func("expand_dims", False)
def expand_dim_shape_func(attrs, inputs, _):
    """
    Shape function for expand_dim op.
    """
    axis = get_const_int(attrs.axis)
    num_newaxis = get_const_int(attrs.num_newaxis)
    if axis < 0:
        axis = inputs[0].shape[0] + axis + 1
    ndim = inputs[0].shape[0] if inputs[0].shape else 0
    return [_expand_dim_shape_func(inputs[0],
                                   convert(ndim),
                                   convert(axis),
                                   convert(num_newaxis))]

@script
def _transpose_shape_func(data_shape, axes):
    out = output_tensor((data_shape.shape[0],), "int64")
    for i in const_range(len(axes)):
        out[i] = data_shape[axes[i]]

    return out

@_reg.register_shape_func("transpose", False)
def transpose_shape_func(attrs, inputs, _):
    """
    Shape function for transpose op.
    """
    axes = attrs.axes if attrs.axes is None else get_const_tuple(attrs.axes)
    if axes is None:
        axes = list(range(inputs[0].shape[0].value))
        axes.reverse()
    for i, axis in enumerate(axes):
        if axis < 0:
            axes[i] = inputs[0].shape[0] - axis
    return [_transpose_shape_func(inputs[0], convert(axes))]

@script
def _squeeze_shape_func(data_shape, keep_axes):
    out = output_tensor((len(keep_axes),), "int64")
    for i in const_range(len(keep_axes)):
        out[i] = data_shape[keep_axes[i]]

    return out

@_reg.register_shape_func("squeeze", False)
def squeeze_shape_func(attrs, inputs, _):
    """
    Shape function for squeeze op.
    """
    axis = attrs.axis if attrs.axis is None else get_const_tuple(attrs.axis)
    keep_axes = []
    if axis is not None:
        for i in range(inputs[0].shape[0].value):
            if i not in axis:
                keep_axes.append(i)

    # Due to current relay type system, it is possible even
    # a static kernel function needs shape function. To handle
    # this case, we allow axis to be None in squeeze shape func
    # for now.
    # TODO(kevinthesun): Enhance relay type system to avoid this.
    if keep_axes:
        out = _squeeze_shape_func(inputs[0], convert(keep_axes))
    else:
        out = te.compute((), lambda *indices: 0)
    return [out]

@script
def _reshape_like_shape_func(target_shape):
    out = output_tensor((target_shape.shape[0],), "int64")
    for i in const_range(target_shape.shape[0]):
        out[i] = target_shape[i]

    return out

@_reg.register_shape_func("reshape_like", False)
def reshape_like_shape_func(attrs, inputs, _):
    """
    Shape function for reshape_like op.
    """
    return [_reshape_like_shape_func(inputs[1])]

@script
def _tile_shape_func(data, reps, ndim, tndim, rndim):
    out = output_tensor((tndim,), "int64")

    if ndim == rndim:
        for i in const_range(tndim):
            out[i] = data[i] * int64(reps[i])
    elif ndim > rndim:
        ngap = ndim - rndim
        for i in const_range(ndim):
            if i < ngap:
                out[i] = data[i]
            else:
                out[i] = data[i] * int64(reps[i - ngap])
    else:
        rgap = rndim - ndim
        for i in const_range(rndim):
            if i < rgap:
                out[i] = int64(reps[i])
            else:
                out[i] = int64(reps[i]) * data[i - rgap]
    return out

@_reg.register_shape_func("tile", False)
def tile_shape_func(attrs, inputs, _):
    """
    Shape function for tile op.
    """
    reps = get_const_tuple(attrs.reps)
    ndim = inputs[0].shape[0].value
    rndim = len(reps)
    tndim = ndim if ndim > rndim else rndim
    return [_tile_shape_func(inputs[0], convert(reps), convert(ndim),
                             convert(tndim), convert(rndim))]

@script
def _split_shape_func(data_shape, index, indices_or_sections, axis):
    out = output_tensor((data_shape.shape[0],), "int64")
    if len(indices_or_sections) == 1:
        for i in const_range(data_shape.shape[0]):
            if i == axis:
                out[i] = ceil_div(data_shape[axis], indices_or_sections[0])
            else:
                out[i] = data_shape[i]
    else:
        start = int64(0)
        if index > 0:
            start = int64(indices_or_sections[index - 1])
        end = data_shape[axis]
        if index < len(indices_or_sections):
            end = int64(indices_or_sections[index])
        for i in const_range(data_shape.shape[0]):
            if i == axis:
                out[i] = end - start
            else:
                out[i] = data_shape[i]
    return out

@_reg.register_shape_func("split", False)
def split_shape_func(attrs, inputs, _):
    """
    Shape function for split op.
    """
    if isinstance(attrs.indices_or_sections, (int, tvm.tir.IntImm)):
        indices_or_sections = get_const_int(attrs.indices_or_sections)
    else:
        indices_or_sections = get_const_tuple(attrs.indices_or_sections)

    axis = get_const_int(attrs.axis)

    num_out = indices_or_sections if isinstance(indices_or_sections, int) \
        else len(indices_or_sections) + 1
    if isinstance(indices_or_sections, int):
        indices_or_sections = [indices_or_sections]
    return [_split_shape_func(inputs[0],
                              convert(i),
                              convert(indices_or_sections),
                              convert(axis)) for i in range(num_out)]
