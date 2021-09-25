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

from tvm.runtime import convert
from tvm.te.hybrid import script

from .. import op as _reg

_reg.register_broadcast_schedule("dyn.broadcast_to")
_reg.register_injective_schedule("dyn.reshape")
_reg.register_injective_schedule("dyn.expand_dims")
_reg.register_injective_schedule("dyn.squeeze")
_reg.register_broadcast_schedule("dyn.tile")
_reg.register_injective_schedule("dyn.one_hot")
_reg.register_injective_schedule("dyn.full")
_reg.register_injective_schedule("dyn.strided_slice")
_reg.register_injective_schedule("dyn.sparse_to_dense")


@script
def _reshape_shape_func_input_data(data_shape, newshape, ndim):
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
            src_idx += 1
            dst_idx += 1
        elif newshape[i] == -2:
            assert False, "Value -2 is not valid in newshape argument of dynamic reshape"
        elif newshape[i] == -3:
            assert data_shape.shape[0] - src_idx > 1, "Not enough dims in input shape for -3"
            out[dst_idx] = data_shape[src_idx] * data_shape[src_idx + 1]
            src_idx += 2
            dst_idx += 1
        elif newshape[i] == -4:
            assert False, "Value -4 is not valid in newshape argument of dynamic reshape"
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


@_reg.register_shape_func("dyn.reshape", [False, True])
def dynamic_reshape_shape_func(attrs, inputs, out_ndims):
    return [_reshape_shape_func_input_data(*inputs, out_ndims[0])]


@script
def _expand_dims_shape_func_input_data(data, axis, ndims, num_newaxis):
    out = output_tensor((ndims,), "int64")

    for i in const_range(ndims):
        if i < axis:
            # We multiply by a check (i < len(data.shape)) to avoid
            # a constant folding mechanism leading to an overflow
            out[i] = int64(data.shape[i * (i < len(data.shape))])
        elif i - num_newaxis < axis:
            out[i] = int64(1)
        else:
            out[i] = int64(
                # We can't use axis in indices as it is not constant but we can
                # use negative indices (kind of, have to manually do it)
                data.shape[
                    (i - num_newaxis) * (i - num_newaxis >= 0)
                    + (i - num_newaxis + len(data.shape)) * (i - num_newaxis < 0)
                ]
            )

    return out


@_reg.register_shape_func("dyn.expand_dims", [True, True])
def dynamic_expand_dims_shape_func(attrs, inputs, out_ndims):
    return [
        _expand_dims_shape_func_input_data(
            inputs[0],
            inputs[1],
            out_ndims[0],
            convert(attrs.num_newaxis),
        )
    ]


@script
def _tile_shape_func(data, reps, ndim, tndim, rndim):
    out = output_tensor((tndim,), "int64")

    if ndim == rndim:
        for i in const_range(tndim):
            out[i] = int64(data.shape[i] * reps[i])
    elif ndim > rndim:
        ngap = ndim - rndim
        for i in const_range(ndim):
            if i < ngap:
                out[i] = int64(data.shape[i])
            else:
                out[i] = int64(data.shape[i] * reps[i - ngap])
    else:
        rgap = rndim - ndim
        for i in const_range(rndim):
            if i < rgap:
                out[i] = int64(reps[i])
            else:
                out[i] = int64(reps[i] * data.shape[i - rgap])
    return out


@_reg.register_shape_func("dyn.tile", True)
def tile_shape_func(attrs, inputs, _):
    """
    Shape function for dyn.tile op.
    """
    reps = inputs[1]
    ndim = len(inputs[0].shape)
    rndim = inputs[1].shape[0].value
    tndim = ndim if ndim > rndim else rndim
    return [_tile_shape_func(inputs[0], reps, convert(ndim), convert(tndim), convert(rndim))]


@script
def _onehot_shape_func(dshape, k, axis):
    ndim = len(dshape) + 1
    out = output_tensor((ndim,), "int64")
    for i in const_range(axis):
        out[i] = int64(dshape[i])
    out[axis] = int64(k[0])
    for j in const_range(axis + 1, ndim):
        out[j] = int64(dshape[j - 1])
    return out


@_reg.register_shape_func("dyn.one_hot", True)
def one_hot_shape_func(attrs, inputs, _):
    """
    Shape function for dyn.one_hot op.
    """
    axis = len(inputs[0].shape) if attrs.axis == -1 else attrs.axis
    return [_onehot_shape_func(inputs[0].shape, inputs[3], convert(axis))]


@script
def _strided_slice_shape_func_input_data(data_shape, begin, end, strides, slice_mode):
    ndim = len(data_shape)
    out = output_tensor((ndim,), "int64")
    for i in const_range(ndim):
        dim_size = int64(data_shape[i])
        cbegin = int64(0)
        cend = dim_size
        cstride = int64(1)

        if strides.shape[0] > i:
            cstride = int64(strides[i])

        if begin.shape[0] > i:
            cbegin = int64(begin[i])
        elif cstride < 0:
            cbegin = dim_size

        if end.shape[0] <= i:
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


@_reg.register_shape_func("dyn.strided_slice", [False, True, True, True])
def strided_slice_shape_func(attrs, inputs, _):
    """
    Shape func for strided_slice
    """
    slice_mode = convert(0 if attrs.slice_mode == "end" else 1)
    return [_strided_slice_shape_func_input_data(*inputs, slice_mode)]


@script
def _sparse_to_dense_shape_func(output_shape, ndim):
    out = output_tensor((ndim,), "int64")
    for i in const_range(ndim):
        out[i] = int64(output_shape[i])
    return out


@_reg.register_shape_func("dyn.sparse_to_dense", True)
def sparse_to_dense_shape_func(attrs, inputs, out_ndims):
    return [_sparse_to_dense_shape_func(inputs[3], out_ndims[0])]


@script
def _squeeze_shape_func_input_data(data, axis, ndims):
    out = output_tensor((ndims,), "int64")
    out_i = 0
    for i in const_range(data.shape[0]):
        not_in_axis = True
        for j in const_range(axis.shape[0]):
            if i == axis[j]:
                not_in_axis = False
        if not_in_axis:
            out[out_i] = int64(data[i])
            out_i += 1

    return out


@_reg.register_shape_func("dyn.squeeze", [False, True])
def dynamic_squeeze_shape_func(attrs, inputs, out_ndims):
    return [_squeeze_shape_func_input_data(inputs[0], inputs[1], out_ndims[0])]
