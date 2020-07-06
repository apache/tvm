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

_reg.register_injective_schedule("dyn.reshape")
_reg.register_broadcast_schedule("dyn.tile")

@script
def _reshape_shape_func_input_data(data, newshape, ndim):
    out = output_tensor((ndim,), "int64")
    data_shape = allocate((len(data.shape),), "int64")
    for x in const_range(len(data.shape)):
        data_shape[x] = int64(data.shape[x])
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
            assert data_shape.shape[0] - src_idx > 1, \
                "Not enough dims in input shape for -3"
            out[dst_idx] = data_shape[src_idx] * data_shape[src_idx+1]
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

@_reg.register_shape_func("dyn.reshape", True)
def dynamic_reshape_shape_func(attrs, inputs, out_ndims):
    return [_reshape_shape_func_input_data(*inputs, out_ndims[0])]


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
    return [_tile_shape_func(inputs[0], reps, convert(ndim),
                             convert(tndim), convert(rndim))]
