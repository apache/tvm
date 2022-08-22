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
# pylint: disable=invalid-name,unused-argument, len-as-condition, too-many-nested-blocks,
# pylint: disable=too-many-local-variables, too-many-arguments, no-else-return

from __future__ import absolute_import

import tvm
from tvm import te, topi
from tvm.runtime import convert
from tvm.te.hybrid import script
from tvm.topi.utils import get_const_int, get_const_tuple

from . import op as _reg
from . import strategy
from ._tensor import elemwise_shape_func
from .op import OpPattern

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
_reg.register_injective_schedule("stack")
_reg.register_injective_schedule("contrib_reverse_reshape")
_reg.register_injective_schedule("gather")
_reg.register_injective_schedule("gather_nd")
_reg.register_injective_schedule("sequence_mask")
_reg.register_injective_schedule("one_hot")
_reg.register_reduce_schedule("collapse_sum_like")
_reg.register_reduce_schedule("collapse_sum_to")
_reg.register_injective_schedule("unravel_index")
_reg.register_injective_schedule("sparse_to_dense")
_reg.register_injective_schedule("matrix_set_diag")
_reg.register_injective_schedule("adv_index")


# concatenate
@_reg.register_compute("concatenate")
def compute_concat(attrs, inputs, output_type):
    return [topi.concatenate(inputs, attrs.axis)]


_reg.register_strategy("concatenate", strategy.concatenate_strategy)

# sliding_window
@_reg.register_compute("sliding_window")
def compute_sliding_window(attrs, inputs, output_type):
    """Compute definition of sliding_window"""
    return [topi.sliding_window(inputs[0], attrs.axis, attrs.window_shape, attrs.strides)]


_reg.register_strategy("sliding_window", strategy.sliding_window_strategy)

# strided_set
@_reg.register_compute("strided_set")
def compute_strided_set(attrs, inputs, output_type):
    """Compute definition of strided_set"""
    return [topi.strided_set(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])]


_reg.register_injective_schedule("strided_set")

# layout_transform
_reg.register_injective_schedule("layout_transform")
_reg.register_pattern("layout_transform", OpPattern.INJECTIVE)
_reg.register_injective_schedule("auto_scheduler_layout_transform")
_reg.register_pattern("auto_scheduler_layout_transform", OpPattern.INJECTIVE)
_reg.register_injective_schedule("meta_schedule_layout_transform")
_reg.register_pattern("meta_schedule_layout_transform", OpPattern.INJECTIVE)

# argwhere
_reg.register_strategy("argwhere", strategy.argwhere_strategy)

# scatter
@_reg.register_compute("scatter")
def compute_scatter(attrs, inputs, output_type):
    """Compute definition of scatter"""
    return [topi.scatter(inputs[0], inputs[1], inputs[2], attrs.axis)]


_reg.register_strategy("scatter", strategy.scatter_strategy)

# sparse_fill_empty_rows
@_reg.register_compute("sparse_fill_empty_rows")
def compute_sparse_fill_empty_rows(attrs, inputs, output_type):
    """Compute definition of sparse_fill_empty_rows"""

    return topi.sparse_fill_empty_rows(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        output_type.fields[0].shape,
        output_type.fields[1].shape,
        output_type.fields[2].shape,
    )


_reg.register_strategy("sparse_fill_empty_rows", strategy.sparse_fill_empty_rows_strategy)

# sparse_reshape
@_reg.register_compute("sparse_reshape")
def compute_reshape(attrs, inputs, output_type):
    """Compute definition of sparse_reshape"""

    return topi.sparse_reshape(
        inputs[0],
        inputs[1],
        inputs[2],
        output_type.fields[0].shape,
        output_type.fields[1].shape,
    )


_reg.register_strategy("sparse_reshape", strategy.sparse_reshape_strategy)

# stft
@_reg.register_compute("stft")
def compute_stft(attrs, inputs, output_type):
    """Compute definition of stft"""
    return topi.stft(
        inputs[0],
        attrs.n_fft,
        attrs.hop_length,
        attrs.win_length,
        attrs.window,
        attrs.normalized,
        attrs.onesided,
        output_type.shape,
    )


_reg.register_strategy("stft", strategy.stft_strategy)


@script
def _stft_shape_func(data, n_fft, hop_length, onesided):
    output_shape = output_tensor((4,), "int64")
    output_shape[0] = int64(data.shape[0])
    if onesided:
        output_shape[1] = int64(int64(n_fft) // int64(2)) + int64(1)
    else:
        output_shape[1] = int64(n_fft)
    output_shape[2] = int64(int64(data.shape[1] - n_fft) // int64(hop_length)) + int64(1)
    output_shape[3] = int64(2)
    return output_shape


@_reg.register_shape_func("stft", True)
def stft_shape_func(attrs, inputs, _):
    """
    Shape func for stft.
    """
    return [
        _stft_shape_func(
            inputs[0], convert(attrs.n_fft), convert(attrs.hop_length), convert(attrs.onesided)
        )
    ]


# trilu
_reg.register_strategy("trilu", strategy.trilu_strategy)


# scatter_add
@_reg.register_compute("scatter_add")
def compute_scatter_add(attrs, inputs, output_type):
    """Compute definition of scatter_add"""
    return [topi.scatter_add(inputs[0], inputs[1], inputs[2], attrs.axis)]


_reg.register_strategy("scatter_add", strategy.scatter_add_strategy)

# scatter_nd
@_reg.register_compute("scatter_nd")
def compute_scatter_nd(attrs, inputs, output_type):
    """Compute definition of scatter_nd"""
    return [topi.scatter_nd(inputs[0], inputs[1], inputs[2], attrs.mode)]


_reg.register_strategy("scatter_nd", strategy.scatter_nd_strategy)

# cumsum
@_reg.register_compute("cumsum")
def compute_cumsum(attrs, inputs, output_type):
    """Compute definition of cumsum"""
    return [topi.cumsum(inputs[0], attrs.axis, attrs.dtype, attrs.exclusive)]


_reg.register_strategy("cumsum", strategy.cumsum_strategy)
_reg.register_shape_func("cumsum", False, elemwise_shape_func)

# cumprod
@_reg.register_compute("cumprod")
def compute_cumprod(attrs, inputs, output_type):
    """Compute definition of cumprod"""
    return [topi.cumprod(inputs[0], attrs.axis, attrs.dtype, attrs.exclusive)]


_reg.register_strategy("cumprod", strategy.cumprod_strategy)
_reg.register_shape_func("cumprod", False, elemwise_shape_func)


@_reg.register_compute("unique")
def compute_unique(attrs, inputs, output_type):
    """Compute definition of unique"""
    return topi.unique(inputs[0], attrs.sorted, attrs.return_counts)


_reg.register_strategy("unique", strategy.unique_strategy)

# invert_permutation
_reg.register_strategy("invert_permutation", strategy.invert_permutation_strategy)
_reg.register_shape_func("invert_permutation", False, elemwise_shape_func)


#####################
#  Shape functions  #
#####################


@script
def _arange_shape_func(start, stop, step):
    out = output_tensor((1,), "int64")
    if step[()] < 0:
        out[0] = int64(ceil_div((int64(start[()]) - int64(stop[()])), int64(-step[()])))
    else:
        out[0] = int64(ceil_div((int64(stop[()]) - int64(start[()])), int64(step[()])))
    return out


@_reg.register_shape_func("arange", True)
def arange_shape_func(attrs, inputs, _):
    """
    Shape func for arange
    """
    return [_arange_shape_func(*inputs)]


@script
def _strided_slice_shape_func_input_shape(data_shape, begin, end, strides, slice_mode):
    ndim = len(data_shape)
    out = output_tensor((ndim,), "int64")
    for i in const_range(ndim):
        dim_size = int64(data_shape[i])
        cbegin = int64(0)
        cend = dim_size
        cstride = int64(1)

        if len(strides) > i:
            cstride = int64(strides[i])

        if len(begin) > i:
            cbegin = int64(begin[i])
        elif cstride < 0:
            cbegin = dim_size

        if len(end) <= i:
            if cstride < 0:
                cend = int64(0)
        elif slice_mode != 0:
            cstride = int64(1)
            if end[i] < 0:
                cend = dim_size
            else:
                cend = cbegin + int64(end[i])
        else:
            if end[i] > data_shape[i]:
                cend = dim_size
            else:
                cend = int64(end[i])

        assert cstride != 0, "Strides can't be zero."

        if cbegin < 0:
            cbegin += dim_size
        if cend < 0:
            cend += dim_size

        if cstride < 0:
            if cend < 0:
                cend = int64(-1)
            if cbegin > dim_size - 1:
                cbegin = dim_size - 1
            slice_range = cbegin - cend
            step = -cstride
        else:
            slice_range = cend - cbegin
            step = cstride
        out[i] = int64(ceil_div(slice_range, step))
    return out


@script
def _strided_slice_shape_func_with_axes(data_shape, begin, end, strides, slice_mode, axes):
    ndim = data_shape.shape[0]
    out = output_tensor((ndim,), "int64")
    for i in const_range(ndim):
        out[i] = data_shape[i]

    for i in const_range(len(axes)):
        dim_size = int64(data_shape[axes[i]])
        cbegin = int64(0)
        cend = dim_size
        cstride = int64(1)

        if len(strides) > i:
            cstride = int64(strides[i])

        if len(begin) > i:
            cbegin = int64(begin[i])
        elif cstride < 0:
            cbegin = dim_size

        if len(end) <= i:
            cend = dim_size
        elif slice_mode != 0:
            cstride = int64(1)
            if end[i] < 0:
                cend = dim_size
            else:
                cend = cbegin + int64(end[i])
        else:
            if end[i] > data_shape[axes[i]]:
                cend = dim_size
            else:
                cend = int64(end[i])

        assert cstride != 0, "Strides can't be zero."

        if cbegin < 0:
            cbegin += dim_size
        if cend < 0:
            cend += dim_size

        if cstride < 0:
            if cend < 0:
                cend = int64(-1)
            if cbegin > dim_size - 1:
                cbegin = dim_size - 1
            slice_range = cbegin - cend
            step = -cstride
        else:
            slice_range = cend - cbegin
            step = cstride

        out[axes[i]] = int64(ceil_div(slice_range, step))
    return out


@_reg.register_shape_func("strided_slice", False)
def strided_slice_shape_func(attrs, inputs, _):
    """
    Shape func for strided_slice
    """
    slice_mode = convert(0 if attrs.slice_mode == "end" else 1)
    if attrs.axes is None:
        return [
            _strided_slice_shape_func_input_shape(
                inputs[0], attrs.begin, attrs.end, attrs.strides, slice_mode
            )
        ]
    return [
        _strided_slice_shape_func_with_axes(
            inputs[0], attrs.begin, attrs.end, attrs.strides, slice_mode, attrs.axes
        )
    ]


@script
def _one_hot_shape_func(indices_shape, depth, axis):
    in_ndim = indices_shape.shape[0]
    out_ndim = in_ndim + 1
    true_axis = in_ndim if axis == -1 else axis
    indices_i = 0
    out = output_tensor((out_ndim,), "int64")
    for i in range(out_ndim):
        if i == true_axis:
            out[i] = int64(depth)
        else:
            out[i] = int64(indices_shape[indices_i])
            indices_i += 1
    return out


@_reg.register_shape_func("one_hot", False)
def one_hot_shape_func(attrs, inputs, _):
    """
    Shape func for one_hot
    """
    shape_func = [_one_hot_shape_func(inputs[0], convert(attrs.depth), convert(attrs.axis))]
    return shape_func


@script
def _concatenate_shape_func(inputs, axis):
    ndim = inputs[0].shape[0]
    out = output_tensor((ndim,), "int64")
    for i in const_range(ndim):
        if i != axis:
            out[i] = inputs[0][i]
            for j in const_range(1, len(inputs)):
                assert out[i] == inputs[j][i], "Dims mismatch in the inputs of concatenate."
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
def _reshape_shape_func_input_shape(data_shape, newshape, ndim, allowzero):
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
            if allowzero:
                out[dst_idx] = int64(newshape[i])
            else:
                out[dst_idx] = data_shape[src_idx]
            src_idx += 1
            dst_idx += 1
        elif newshape[i] == -1:
            assert infer_idx < 0, "One and only one dim can be inferred"
            out[dst_idx] = int64(1)
            infer_idx = i
            src_idx += 1
            dst_idx += 1
        elif newshape[i] == -2:
            copy = True
        elif newshape[i] == -3:
            assert data_shape.shape[0] - src_idx > 1, "Not enough dims in input shape for -3"
            out[dst_idx] = data_shape[src_idx] * data_shape[src_idx + 1]
            src_idx += 2
            dst_idx += 1
        elif newshape[i] == -4:
            assert len(newshape) - i > 2, "Not enough dims in new shape for -4"
            if newshape[i + 1] == -1:
                assert newshape[i + 2] != -1, "Split dims cannot both be -1."
                out[dst_idx] = data_shape[src_idx] // int64(newshape[i + 2])
                out[dst_idx + 1] = int64(newshape[i + 2])
            else:
                out[dst_idx] = int64(newshape[i + 1])
                if newshape[i + 2] == -1:
                    out[dst_idx + 1] = data_shape[src_idx] // int64(newshape[i + 1])
                else:
                    out[dst_idx + 1] = int64(newshape[i + 2])
            assert (
                data_shape[src_idx] == out[dst_idx] * out[dst_idx + 1]
            ), "Product of split dims doesn't match to input dim"
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
    allowzero = attrs.allowzero
    return [
        _reshape_shape_func_input_shape(
            inputs[0], convert(newshape), out_ndims[0], convert(allowzero)
        )
    ]


@script
def _take_no_axis_shape_func(indices_shape, out_ndim):
    out = output_tensor((out_ndim,), "int64")
    for i in const_range(out_ndim):
        out[i] = indices_shape[i]
    return out


@script
def _take_with_axis_shape_func(data_shape, indices_shape, axis, batch_dims, out_ndim):
    out = output_tensor((out_ndim,), "int64")
    for i in const_range(axis):
        out[i] = data_shape[i]
    if len(indices_shape.shape) == 0:
        # indices is constant
        for i in const_range(axis + 1, len(data_shape)):
            out[i - 1] = data_shape[i]
    else:
        for i in const_range(len(indices_shape) - batch_dims):
            out[axis + i] = indices_shape[i + batch_dims]
        for i in const_range(axis + 1, len(data_shape)):
            out[len(indices_shape) + i - 1 - batch_dims] = data_shape[i]
    return out


@_reg.register_shape_func("take", False)
def take_shape_func(attrs, inputs, out_ndims):
    """
    Shape function for take op.
    """
    if attrs.axis is None:
        return [_take_no_axis_shape_func(inputs[1], out_ndims[0])]
    axis = get_const_int(attrs.axis)
    batch_dims = get_const_int(attrs.batch_dims)
    data_ndim = int(inputs[0].shape[0])
    if inputs[1].shape:
        indices_ndim = int(inputs[1].shape[0])
    if axis < 0:
        axis += data_ndim
    assert 0 <= axis < data_ndim
    if batch_dims < 0:
        batch_dims += indices_ndim
    return [_take_with_axis_shape_func(*inputs, convert(axis), convert(batch_dims), out_ndims[0])]


@_reg.register_legalize("take")
def legalize_dyn_topk(attrs, inputs, types):
    """Legalize take op.
    Parameters
    ----------
    attrs : tvm.ir.Attrs
        Attributes of current op
    inputs : list of tvm.relay.Expr
        The args of the Relay expr to be legalized
    types : list of types
        List of input and output types
    Returns
    -------
    result : tvm.relay.Expr
        The legalized expr
    """
    return topi.take_legalize(attrs, inputs, types)


@script
def _argwhere_shape_func_1d(condition):
    out = output_tensor((2,), "int64")
    out[0] = int64(0)
    out[1] = int64(1)
    for i1 in range(condition.shape[0]):
        if condition[i1] != 0:
            out[0] += int64(1)
    return out


@script
def _argwhere_shape_func_2d(condition):
    out = output_tensor((2,), "int64")
    out[0] = int64(0)
    out[1] = int64(2)
    for i1 in range(condition.shape[0]):
        for i2 in range(condition.shape[1]):
            if condition[i1, i2] != 0:
                out[0] += int64(1)
    return out


@script
def _argwhere_shape_func_3d(condition):
    out = output_tensor((2,), "int64")
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
    out = output_tensor((2,), "int64")
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
    out = output_tensor((2,), "int64")
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
_reg.register_shape_func("scatter_add", False, elemwise_shape_func)
_reg.register_shape_func("scatter_nd", False, elemwise_shape_func)


@script
def _sparse_fill_empty_rows_shape_func(sparse_indices, dense_shape):

    new_sparse_indices_shape = output_tensor((2,), "int64")
    new_sparse_values_shape = output_tensor((1,), "int64")
    empty_row_indicator_shape = output_tensor((1,), "int64")
    num_dense_rows = int64(dense_shape[0])

    if int64(sparse_indices.shape[0]) == int64(0):  # Handle Empty Case
        #  Total rows will equal dense_shape[0]
        new_sparse_indices_shape[0] = num_dense_rows
        new_sparse_indices_shape[1] = int64(sparse_indices.shape[1])
        new_sparse_values_shape[0] = num_dense_rows
        empty_row_indicator_shape[0] = num_dense_rows
        return (new_sparse_indices_shape, new_sparse_values_shape, empty_row_indicator_shape)

    else:
        count = int64(sparse_indices.shape[0])  # Add count of all rows already in sparse_indices
        for i in range(1, int64(sparse_indices.shape[0])):
            index = int64(sparse_indices[i, 0])
            prev_index = int64(sparse_indices[i - 1, 0] + 1)

            if index > prev_index:
                count += index - prev_index  # Add count of all rows between two consecutive indices

        count += int64(sparse_indices[0, 0])  # Add count from 0 to first row id in sparse_indices
        count += int64(
            num_dense_rows - 1 - sparse_indices[sparse_indices.shape[0] - 1, 0]
        )  # Add count from last row id to dense_shape - 1
        new_sparse_indices_shape[0] = int64(count)
        new_sparse_indices_shape[1] = int64(sparse_indices.shape[1])
        new_sparse_values_shape[0] = int64(count)
        empty_row_indicator_shape[0] = num_dense_rows
        return (new_sparse_indices_shape, new_sparse_values_shape, empty_row_indicator_shape)


@_reg.register_shape_func("sparse_fill_empty_rows", True)
def sparse_fill_empty_rows_func(attrs, inputs, _):
    return _sparse_fill_empty_rows_shape_func(inputs[0], inputs[2])


@script
def _sparse_reshape_shape_func(sparse_indices_shape, prev_shape_shape, new_shape_shape):
    indices_shape = output_tensor((2,), "int64")
    indices_shape[0] = int64(sparse_indices_shape[0])
    indices_shape[1] = int64(new_shape_shape[0])
    shape_tensor = output_tensor((1,), "int64")
    shape_tensor[0] = int64(new_shape_shape[0])
    return (indices_shape, shape_tensor)


@_reg.register_shape_func("sparse_reshape", False)
def sparse_reshape_shape_func(attrs, inputs, _):
    """
    Shape func for sparse_reshape.
    """
    return _sparse_reshape_shape_func(inputs[0], inputs[1], inputs[2])


@script
def _layout_transform_shape_func(
    data_shape, out_layout_len, dst_equal_list, dst_mul_list, dst_div_list, dst_mix_list
):
    out = output_tensor((out_layout_len,), "int64")
    for i in const_range(len(dst_equal_list)):
        out[dst_equal_list[i][0]] = data_shape[dst_equal_list[i][1]]
    for i in const_range(len(dst_mul_list)):
        out[dst_mul_list[i][0]] = data_shape[dst_mul_list[i][1]] * data_shape[dst_mul_list[i][2]]
    for i in const_range(len(dst_div_list)):
        out[dst_div_list[i][0]] = data_shape[dst_div_list[i][1]] // dst_div_list[i][3]
        out[dst_div_list[i][2]] = int64(dst_div_list[i][3])
    for i in const_range(len(dst_mix_list)):
        out[dst_mix_list[i][0]] = (
            data_shape[dst_mix_list[i][1]] * dst_mix_list[i][2] // dst_mix_list[i][4]
        )
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
                dst_equal_list.append((dst_letter_list.index(key), src_letter_list.index(key)))
            else:
                dst_mul_list.append(
                    (
                        dst_letter_list.index(key),
                        src_letter_list.index(key),
                        src_letter_list.index(key.lower()),
                    )
                )
        else:
            if key.lower() not in src_minor_axes:
                dst_div_list.append(
                    (
                        dst_letter_list.index(key),
                        src_letter_list.index(key),
                        dst_letter_list.index(key.lower()),
                        dst_minor_axes[key.lower()],
                    )
                )
            else:
                dst_mix_list.append(
                    (
                        dst_letter_list.index(key),
                        src_letter_list.index(key),
                        src_minor_axes[key.lower()],
                        dst_letter_list.index(key.lower()),
                        dst_minor_axes[key.lower()],
                    )
                )

    return [
        _layout_transform_shape_func(
            inputs[0],
            convert(out_layout_len),
            convert(dst_equal_list),
            convert(dst_mul_list),
            convert(dst_div_list),
            convert(dst_mix_list),
        )
    ]


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
    return [_expand_dim_shape_func(inputs[0], convert(ndim), convert(axis), convert(num_newaxis))]


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
    axes = list(axes)
    for i, axis in enumerate(axes):
        if axis < 0:
            axes[i] = inputs[0].shape[0] + axis
    return [_transpose_shape_func(inputs[0], convert(axes))]


_reg.register_schedule("transpose", strategy.schedule_transpose)


@script
def _squeeze_shape_func(data_shape, keep_axes, remove_axes):
    out = output_tensor((len(keep_axes),), "int64")
    for i in const_range(len(keep_axes)):
        out[i] = data_shape[keep_axes[i]]

    for i in const_range(len(remove_axes)):
        assert data_shape[remove_axes[i]] == 1, "Removed dimension must have size 1"

    return out


@_reg.register_shape_func("squeeze", False)
def squeeze_shape_func(attrs, inputs, _):
    """
    Shape function for squeeze op.
    """
    axis = attrs.axis if attrs.axis is None else get_const_tuple(attrs.axis)
    keep_axes = []
    remove_axes = []
    if axis is not None:
        for i in range(inputs[0].shape[0].value):
            if i not in axis:
                keep_axes.append(i)
            else:
                remove_axes.append(i)

    # Due to current relay type system, it is possible even
    # a static kernel function needs shape function. To handle
    # this case, we allow axis to be None in squeeze shape func
    # for now.
    # TODO(kevinthesun): Enhance relay type system to avoid this.
    if keep_axes:
        out = _squeeze_shape_func(inputs[0], convert(keep_axes), convert(remove_axes))
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
    return [
        _tile_shape_func(inputs[0], convert(reps), convert(ndim), convert(tndim), convert(rndim))
    ]


@script
def _split_shape_func(data_shape, index, indices_or_sections, param_is_indices, axis):
    out = output_tensor((data_shape.shape[0],), "int64")
    if param_is_indices:
        for i in const_range(data_shape.shape[0]):
            if i == axis:
                assert (
                    data_shape[axis] % indices_or_sections[0] == 0
                ), "num_sections must be an integer factor of the size of axis"
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
        assert indices_or_sections > 0, "Slice count must be > 0"
    else:
        indices_or_sections = list(get_const_tuple(attrs.indices_or_sections))
        assert sorted(indices_or_sections)[0] > 0 and indices_or_sections == sorted(
            indices_or_sections
        ), "split_indices must be sorted"

    axis = get_const_int(attrs.axis)

    if axis < 0:
        axis += get_const_int(inputs[0].shape[0])

    num_out = (
        indices_or_sections
        if isinstance(indices_or_sections, int)
        else len(indices_or_sections) + 1
    )

    param_is_indices = isinstance(indices_or_sections, int)
    if param_is_indices:
        indices_or_sections = [indices_or_sections]
    return [
        _split_shape_func(
            inputs[0],
            convert(i),
            convert(indices_or_sections),
            convert(param_is_indices),
            convert(axis),
        )
        for i in range(num_out)
    ]


@script
def _repeat_shape_func(data_shape, repeats, axis):
    out = output_tensor((data_shape.shape[0],), "int64")

    for i in const_range(data_shape.shape[0]):
        if i == axis:
            out[i] = int64(data_shape[i] * repeats)
        else:
            out[i] = data_shape[i]

    return out


@_reg.register_shape_func("repeat", False)
def repeat_shape_func(attrs, inputs, _):
    """
    Shape func for repeat.
    """
    axis = get_const_int(attrs.axis)
    if axis < 0:
        axis = inputs[0].shape[0] + axis
    return [_repeat_shape_func(inputs[0], attrs.repeats, convert(axis))]


@_reg.register_shape_func("broadcast_to_like", False)
def broadcast_to_like_shape_func(attrs, inputs, _):
    """
    Shape func for broadcast_to_like.
    """
    return [topi.math.identity(inputs[1])]


@script
def _stack_shape_func(data_shape, axis, num_inputs):
    out = output_tensor((data_shape.shape[0] + 1,), "int64")

    for i in const_range(data_shape.shape[0] + 1):
        if i == axis:
            out[i] = int64(num_inputs)
        elif i < axis:
            out[i] = data_shape[i]
        else:
            out[i] = data_shape[i - 1]

    return out


@_reg.register_shape_func("stack", False)
def stack_shape_func(attrs, inputs, _):
    """
    Shape func for stack.
    """
    axis = get_const_int(attrs.axis)
    if axis < 0:
        axis += inputs[0].shape[0] + 1
    return [_stack_shape_func(inputs[0], convert(axis), convert(len(inputs)))]


@script
def _broadcast_shape_tensors(shape_tensor1, shape_tensor2):
    rank1 = shape_tensor1.shape[0]
    rank2 = shape_tensor2.shape[0]
    out_rank = max(rank1, rank2)
    bcast_shape_tensor = output_tensor((out_rank,), "int64")

    for index in const_range(out_rank):
        dim1 = int64(1)
        dim2 = int64(1)

        if rank1 == out_rank:
            dim1 = shape_tensor1[index]
        elif rank1 - (out_rank - index) >= 0:
            dim1 = shape_tensor1[rank1 - (out_rank - index)]

        if rank2 == out_rank:
            dim2 = shape_tensor2[index]
        elif rank2 - (out_rank - index) >= 0:
            dim2 = shape_tensor2[rank2 - (out_rank - index)]

        assert dim1 == dim2 or dim1 == 1 or dim2 == 1, "Invalid broadcast shapes"
        bcast_shape_tensor[index] = max(dim1, dim2)

    return bcast_shape_tensor


@_reg.register_shape_func("where", False)
def where_shape_func(attrs, inputs, _):
    """
    Shape func for where.
    """

    def ensure_tensor(tensor):
        if len(tensor.shape) == 0:
            return topi.full((1,), "int64", 1)
        return tensor

    cond_shape = ensure_tensor(inputs[0])
    x_shape = ensure_tensor(inputs[1])
    y_shape = ensure_tensor(inputs[2])

    bcast_shape = _broadcast_shape_tensors(x_shape, y_shape)
    out_shape = _broadcast_shape_tensors(bcast_shape, cond_shape)

    return [out_shape]


@script
def _adv_index_post_process(data_shape, bcast_shape, num_indices):
    data_rank = data_shape.shape[0]
    bcast_rank = bcast_shape.shape[0]
    out = output_tensor((data_rank + bcast_rank - num_indices,), "int64")

    for i in const_range(bcast_rank):
        out[i] = bcast_shape[i]
    for i in const_range(data_rank - num_indices):
        out[i + bcast_rank] = data_shape[i + num_indices]
    return out


@_reg.register_shape_func("adv_index", False)
def adv_index_shape_func(attrs, inputs, _):
    """
    Shape func for adv_index.
    """
    bcast_shape = inputs[1]
    for i in inputs[2:]:
        bcast_shape = _broadcast_shape_tensors(bcast_shape, i)
    return [_adv_index_post_process(inputs[0], bcast_shape, convert(len(inputs) - 1))]


@script
def _unique_shape(data_shape):
    unique_shape = output_tensor((1,), "int64")
    indices_shape = output_tensor((1,), "int64")
    inverse_indices_shape = output_tensor((1,), "int64")
    num_unique_shape = output_tensor((1,), "int64")
    unique_shape[0] = data_shape[0]
    indices_shape[0] = data_shape[0]
    inverse_indices_shape[0] = data_shape[0]
    num_unique_shape[0] = int64(1)
    return (unique_shape, indices_shape, inverse_indices_shape, num_unique_shape)


@script
def _unique_with_counts_shape(data_shape):
    unique_shape = output_tensor((1,), "int64")
    indices_shape = output_tensor((1,), "int64")
    inverse_indices_shape = output_tensor((1,), "int64")
    num_unique_shape = output_tensor((1,), "int64")
    counts_shape = output_tensor((1,), "int64")
    unique_shape[0] = data_shape[0]
    indices_shape[0] = data_shape[0]
    inverse_indices_shape[0] = data_shape[0]
    num_unique_shape[0] = int64(1)
    counts_shape[0] = data_shape[0]
    return (unique_shape, indices_shape, inverse_indices_shape, num_unique_shape, counts_shape)


@_reg.register_shape_func("unique", False)
def unique_shape_func(attrs, inputs, _):
    """
    Shape func for unique operator.
    """
    if attrs.return_counts:
        return _unique_with_counts_shape(inputs[0])
    else:
        return _unique_shape(inputs[0])


@script
def _gather_nd_shape(data_shape, indices_shape, batch_dims, index_rank):
    ndim = data_shape.shape[0]
    # using mdim = indices_shape[0] wouldn't work because a rank cannot
    # depend on a runtime shape dimension of indices tensor, even if the
    # dimension is always a known, fixed value. As a workaround, we assume that
    # the fixed gather dimension (the size of an indexing tuple) is recorded
    # in gather_nd op attributes.
    mdim = index_rank
    kdim = indices_shape.shape[0] - 1
    out_shape = output_tensor((kdim + ndim - (mdim + batch_dims),), "int64")
    for i in range(1, kdim + 1):
        out_shape[i - 1] = indices_shape[i]
    for i in range(mdim + batch_dims, ndim):
        out_shape[kdim + i - (mdim + batch_dims)] = data_shape[i]
    return out_shape


@_reg.register_shape_func("gather_nd", False)
def gather_nd_shape_func(attrs, inputs, _):
    """
    Shape func for gather_nd operator.
    """
    batch_dims = get_const_int(attrs.batch_dims)
    index_rank = get_const_int(attrs.index_rank)

    assert index_rank > 0, "index_rank needs to be specified for dynamic gather_nd"

    return [_gather_nd_shape(inputs[0], inputs[1], convert(batch_dims), convert(index_rank))]


@script
def _gather_shape(data_shape, indices_shape, axis):
    out_shape = output_tensor((data_shape.shape[0],), "int64")
    for i in range(data_shape.shape[0]):
        if i != axis:
            assert (
                data_shape[i] == indices_shape[i]
            ), "data and indices size at non-gather axes must be the same"
        out_shape[i] = indices_shape[i]
    return out_shape


@_reg.register_shape_func("gather", False)
def gather_shape_func(attrs, inputs, _):
    """
    Shape func for gather operator.
    """
    return [_gather_shape(inputs[0], inputs[1], attrs.axis)]
