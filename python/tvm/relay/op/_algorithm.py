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
"Definition of classic algorithms"
# pylint: disable=invalid-name,unused-argument, too-many-nested-blocks
from __future__ import absolute_import
import tvm
from tvm.relay.ty import TensorType

import topi
from topi.util import get_const_int
from . import op as _reg
from ...hybrid import script


@_reg.register_schedule("argsort")
def schedule_argsort(_, outs, target):
    """Schedule definition of argsort"""
    with target:
        return topi.generic.schedule_argsort(outs)


@_reg.register_compute("argsort")
def compute_argsort(attrs, inputs, _, target):
    """Compute definition of argsort"""
    axis = get_const_int(attrs.axis)
    is_ascend = bool(get_const_int(attrs.is_ascend))
    dtype = attrs.dtype
    return [topi.argsort(inputs[0], axis=axis, is_ascend=is_ascend, dtype=dtype)]


_reg.register_pattern("argsort", _reg.OpPattern.OPAQUE)

# argwhere
@_reg.register_schedule("argwhere")
def schedule_argwhere(_, outs, target):
    """Schedule definition of argwhere"""
    with target:
        return topi.generic.schedule_argwhere(outs)


@_reg.register_compute("argwhere")
def compute_argwhere(attrs, inputs, output_type, _):
    """Compute definition of argwhere"""
    output_shape = []
    for s in output_type.shape:
        if hasattr(s, "value"):
            output_shape.append(s)
        else:
            # see Any, replace it with a var
            output_shape.append(tvm.var("any_dim", "int32"))
    new_output_type = TensorType(output_shape, "int32")
    return [topi.argwhere(new_output_type, inputs[0])]

@_reg.register_schedule("topk")
def schedule_topk(_, outs, target):
    """Schedule definition of argsort"""
    with target:
        return topi.generic.schedule_topk(outs)


@_reg.register_compute("topk")
def compute_topk(attrs, inputs, _, target):
    """Compute definition of argsort"""
    k = get_const_int(attrs.k)
    axis = get_const_int(attrs.axis)
    ret_type = attrs.ret_type
    is_ascend = bool(get_const_int(attrs.is_ascend))
    dtype = attrs.dtype
    out = topi.topk(inputs[0], k, axis, ret_type, is_ascend, dtype)
    out = out if isinstance(out, list) else [out]
    return out


_reg.register_pattern("topk", _reg.OpPattern.OPAQUE)

@script
def _argwhere_shape_func_2d(condition):
    out = output_tensor((2, ), "int64")
    out[0] = int64(0)
    out[1] = int64(2)
    for i1 in range(condition.shape[0]):
        for i2 in range(condition.shape[1]):
            if condition[i1, i2]:
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
                if condition[i1, i2, i3]:
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
                    if condition[i1, i2, i3, i4]:
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
                        if condition[i1, i2, i3, i4, i5]:
                            out[0] += int64(1)
    return out

@_reg.register_shape_func("argwhere", True)
def argwhere_shape_func(attrs, inputs, out_ndims):
    """
    Shape function for argwhere.
    """
    if len(inputs[0].shape) == 2:
        return [_argwhere_shape_func_2d(inputs[0])]
    elif len(inputs[0].shape) == 3:
        return [_argwhere_shape_func_3d(inputs[0])]
    elif len(inputs[0].shape) == 4:
        return [_argwhere_shape_func_4d(inputs[0])]
    elif len(inputs[0].shape) == 5:
        return [_argwhere_shape_func_5d(inputs[0])]
    return []
