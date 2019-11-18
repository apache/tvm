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
#pylint: disable=invalid-name, unused-argument, len-as-condition
"""Backend compiler related feature registration"""
from __future__ import absolute_import
import topi
from .op import register_compute, register_schedule, register_pattern, register_shape_func
from .op import schedule_injective, OpPattern
from ...hybrid import script

schedule_broadcast = schedule_injective
schedule_elemwise = schedule_injective

register_schedule("log", schedule_broadcast)
register_schedule("log1p", schedule_broadcast)
register_schedule("cos", schedule_broadcast)
register_schedule("sin", schedule_broadcast)
register_schedule("atan", schedule_broadcast)
register_schedule("exp", schedule_broadcast)
register_schedule("erf", schedule_broadcast)
register_schedule("sqrt", schedule_broadcast)
register_schedule("rsqrt", schedule_broadcast)
register_schedule("sigmoid", schedule_broadcast)
register_schedule("floor", schedule_broadcast)
register_schedule("ceil", schedule_broadcast)
register_schedule("trunc", schedule_broadcast)
register_schedule("round", schedule_broadcast)
register_schedule("sign", schedule_broadcast)
register_schedule("abs", schedule_broadcast)
register_schedule("tanh", schedule_broadcast)
register_schedule("logical_not", schedule_broadcast)
register_schedule("negative", schedule_broadcast)
register_schedule("copy", schedule_broadcast)

register_schedule("add", schedule_broadcast)
register_schedule("subtract", schedule_broadcast)
register_schedule("multiply", schedule_broadcast)
register_schedule("divide", schedule_broadcast)
register_schedule("floor_divide", schedule_broadcast)
register_schedule("power", schedule_injective)
register_schedule("mod", schedule_broadcast)
register_schedule("floor_mod", schedule_broadcast)
register_schedule("logical_and", schedule_broadcast)
register_schedule("logical_or", schedule_broadcast)
register_schedule("equal", schedule_broadcast)
register_schedule("not_equal", schedule_broadcast)
register_schedule("less", schedule_broadcast)
register_schedule("less_equal", schedule_broadcast)
register_schedule("greater", schedule_broadcast)
register_schedule("greater_equal", schedule_broadcast)
register_schedule("maximum", schedule_injective)
register_schedule("minimum", schedule_injective)
register_schedule("right_shift", schedule_injective)
register_schedule("left_shift", schedule_injective)
register_schedule("shape_of", schedule_injective)

# zeros
@register_compute("zeros")
def zeros_compute(attrs, inputs, output_type, target):
    assert not inputs
    return [topi.full(output_type.shape, output_type.dtype, 0.0)]

register_schedule("zeros", schedule_broadcast)
register_pattern("zeros", OpPattern.ELEMWISE)

# zeros_like
@register_compute("zeros_like")
def zeros_like_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.full_like(inputs[0], 0.0)]

register_schedule("zeros_like", schedule_broadcast)

# ones
@register_compute("ones")
def ones_compute(attrs, inputs, output_type, target):
    assert not inputs
    return [topi.full(output_type.shape, output_type.dtype, 1.0)]

register_schedule("ones", schedule_broadcast)
register_pattern("ones", OpPattern.ELEMWISE)

# ones_like
@register_compute("ones_like")
def ones_like(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.full_like(inputs[0], 1.0)]

register_schedule("ones_like", schedule_broadcast)

# clip
@register_compute("clip")
def clip_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.clip(inputs[0], attrs.a_min, attrs.a_max)]

register_schedule("clip", schedule_elemwise)

@script
def _cast_shape_function(x):
    out_ndim = len(x)
    out = output_tensor((out_ndim,), "int64")
    for i in const_range(out_ndim):
        out[i] = x[i]
    return out

def cast_shape_func(attrs, inputs, out_ndims):
    return [_cast_shape_function(*inputs)]

# shape func
@script
def _broadcast_shape_func(x, y, ndim):
    out = output_tensor((ndim,), "int64")
    if len(x.shape) == 0:
        for i in const_range(ndim):
            out[i] = y[i]
    elif len(y.shape) == 0:
        for i in const_range(ndim):
            out[i] = x[i]
    else:
        ndim1 = x.shape[0]
        ndim2 = y.shape[0]
        for i in const_range(1, min(ndim1, ndim2)+1):
            if x[ndim1-i] == y[ndim2-i]:
                out[ndim-i] = x[ndim1-i]
            elif x[ndim1-i] == 1:
                out[ndim-i] = y[ndim2-i]
            else:
                assert y[ndim2 - i] == 1, "Incompatible broadcast type %s and %s" % (
                    x[ndim1-i], y[ndim2-i])
                out[ndim-i] = x[ndim1-i]
        for i in const_range(min(ndim1, ndim2)+1, ndim+1):
            if ndim1 >= ndim2:
                out[ndim-i] = x[ndim1-i]
            else:
                out[ndim-i] = y[ndim2-i]
    return out

def broadcast_shape_func(attrs, inputs, out_ndims):
    """
    Shape function for broadcast op.
    """
    return [_broadcast_shape_func(*inputs, out_ndims[0])]

def elemwise_shape_func(attrs, inputs, _):
    """
    Shape function for elemwise op.
    """
    return [topi.math.identity(inputs[0])]

register_shape_func("cast", False, cast_shape_func)

register_shape_func("add", False, broadcast_shape_func)
register_shape_func("subtract", False, broadcast_shape_func)
register_shape_func("multiply", False, broadcast_shape_func)
register_shape_func("divide", False, broadcast_shape_func)
register_shape_func("floor_divide", False, broadcast_shape_func)
register_shape_func("mod", False, broadcast_shape_func)
register_shape_func("floor_mod", False, broadcast_shape_func)
register_shape_func("logical_and", False, broadcast_shape_func)
register_shape_func("logical_or", False, broadcast_shape_func)
register_shape_func("equal", False, broadcast_shape_func)
register_shape_func("not_equal", False, broadcast_shape_func)
register_shape_func("less", False, broadcast_shape_func)
register_shape_func("less_equal", False, broadcast_shape_func)
register_shape_func("greater", False, broadcast_shape_func)
register_shape_func("greater_equal", False, broadcast_shape_func)

register_shape_func("sqrt", False, elemwise_shape_func)
register_shape_func("negative", False, elemwise_shape_func)
