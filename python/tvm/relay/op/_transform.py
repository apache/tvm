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
# pylint: disable=invalid-name,unused-argument
from __future__ import absolute_import
from topi.util import get_const_int
from . import op as _reg
from ._reduce import _schedule_reduce
from .op import OpPattern
from ... import ir_builder as _ir_builder
from ... import intrin as _intrin
from ... import generic as _generic
from ...api import extern as _extern

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


# layout_transform
_reg.register_schedule("layout_transform", schedule_injective)
_reg.register_pattern("layout_transform", OpPattern.INJECTIVE)

# shape func
def _arange_shape_func(attrs, inputs, outputs):
    ib = _ir_builder.create()
    start = _generic.cast(ib.buffer_ptr(inputs[0])[0], "float32")
    stop = _generic.cast(ib.buffer_ptr(inputs[1]), "float32")
    step = _generic.cast(ib.buffer_ptr(inputs[2]), "float32")
    out_buf = ib.buffer_ptr(outputs[0])
    out_buf[0] = _generic.cast(_intrin.ceil((stop-start) / step), "int64")
    body = ib.get()
    return body

@_reg.register_shape_func("arange")
def arange_shape_func(attrs, inputs, out_shapes):
    out = _extern(out_shapes, inputs,
                  lambda ins, outs: _arange_shape_func(attrs, ins, outs),
                  dtype="int64")
    return [out]

def _concatenate_shape_func(attrs, inputs, outputs):
    axis = get_const_int(attrs.axis)
    ndim = len(inputs[0].shape)
    if axis < 0:
        axis += ndim
    ib = _ir_builder.create()
    out_buf = ib.buffer_ptr(outputs[0])
    for i in range(ndim):
        if i != axis:
            out_buf[i] = _generic.cast(inputs[0].shape[i], "int64")
        else:
            out_buf[i] = _generic.cast(
                sum([inputs[j].shape[i] for j in range(len(inputs))]),
                "int64")
    body = ib.get()
    return body

@_reg.register_shape_func("concatenate")
def concatenate_shape_func(attrs, inputs, out_shapes):
    out = _extern(out_shapes, inputs,
                  lambda ins, outs: _concatenate_shape_func(attrs, ins, outs),
                  dtype="int64")
    return [out]
