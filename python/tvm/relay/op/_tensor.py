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
import topi

from tvm.runtime import convert
from tvm.te.hybrid import script
from topi.util import get_const_tuple
from .op import register_compute, register_shape_func
from .op import register_broadcast_schedule, register_injective_schedule
from .op import register_pattern, OpPattern


register_broadcast_schedule("log")
register_broadcast_schedule("tan")
register_broadcast_schedule("cos")
register_broadcast_schedule("sin")
register_broadcast_schedule("atan")
register_broadcast_schedule("exp")
register_broadcast_schedule("erf")
register_broadcast_schedule("sqrt")
register_broadcast_schedule("rsqrt")
register_broadcast_schedule("sigmoid")
register_broadcast_schedule("floor")
register_broadcast_schedule("ceil")
register_broadcast_schedule("trunc")
register_broadcast_schedule("round")
register_broadcast_schedule("sign")
register_broadcast_schedule("abs")
register_broadcast_schedule("tanh")
register_broadcast_schedule("add")
register_broadcast_schedule("subtract")
register_broadcast_schedule("multiply")
register_broadcast_schedule("divide")
register_broadcast_schedule("floor_divide")
register_broadcast_schedule("power")
register_broadcast_schedule("copy")
register_broadcast_schedule("logical_not")
register_broadcast_schedule("logical_and")
register_broadcast_schedule("logical_or")
register_broadcast_schedule("logical_xor")
register_broadcast_schedule("bitwise_not")
register_broadcast_schedule("bitwise_and")
register_broadcast_schedule("bitwise_or")
register_broadcast_schedule("bitwise_xor")
register_broadcast_schedule("negative")
register_broadcast_schedule("mod")
register_broadcast_schedule("floor_mod")
register_broadcast_schedule("equal")
register_broadcast_schedule("not_equal")
register_broadcast_schedule("less")
register_broadcast_schedule("less_equal")
register_broadcast_schedule("greater")
register_broadcast_schedule("greater_equal")
register_broadcast_schedule("isnan")
register_broadcast_schedule("isfinite")
register_broadcast_schedule("isinf")
register_injective_schedule("maximum")
register_injective_schedule("minimum")
register_injective_schedule("right_shift")
register_injective_schedule("left_shift")
register_injective_schedule("shape_of")
register_injective_schedule("ndarray_size")
register_broadcast_schedule("fast_exp")
register_broadcast_schedule("fast_tanh")
register_broadcast_schedule("fast_erf")


# zeros
@register_compute("zeros")
def zeros_compute(attrs, inputs, output_type):
    assert not inputs
    return [topi.full(output_type.shape, output_type.dtype, 0.0)]

register_broadcast_schedule("zeros")
register_pattern("zeros", OpPattern.ELEMWISE)

# zeros_like
@register_compute("zeros_like")
def zeros_like_compute(attrs, inputs, output_type):
    assert len(inputs) == 1
    return [topi.full_like(inputs[0], 0.0)]

register_broadcast_schedule("zeros_like")

# ones
@register_compute("ones")
def ones_compute(attrs, inputs, output_type):
    assert not inputs
    return [topi.full(output_type.shape, output_type.dtype, 1.0)]

register_broadcast_schedule("ones")
register_pattern("ones", OpPattern.ELEMWISE)

# ones_like
@register_compute("ones_like")
def ones_like_compute(attrs, inputs, output_type):
    assert len(inputs) == 1
    return [topi.full_like(inputs[0], 1.0)]

register_broadcast_schedule("ones_like")

# clip
@register_compute("clip")
def clip_compute(attrs, inputs, output_type):
    assert len(inputs) == 1
    return [topi.clip(inputs[0], attrs.a_min, attrs.a_max)]

register_injective_schedule("clip")

@script
def _cast_shape_function(x):
    out_ndim = len(x)
    out = output_tensor((out_ndim,), "int64")
    for i in const_range(out_ndim):
        out[i] = x[i]
    return out

def cast_shape_func(attrs, inputs, out_ndims):
    return [_cast_shape_function(*inputs)]

@script
def _full_shape_func(shape):
    out_ndim = len(shape)
    out = output_tensor((out_ndim,), "int64")
    for i in const_range(out_ndim):
        out[i] = int64(shape[i])
    return out

def full_shape_func(attrs, inputs, out_ndims):
    """
    Shape func for zeros, zeros_like, ones, ones_like.
    """
    shape = get_const_tuple(attrs.shape)
    return [_full_shape_func(convert(shape))]

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
register_shape_func("zeros", False, full_shape_func)
register_shape_func("zeros_like", False, elemwise_shape_func)
register_shape_func("ones", False, full_shape_func)
register_shape_func("ones_like", False, elemwise_shape_func)
register_shape_func("full", False, full_shape_func)
register_shape_func("full_like", False, elemwise_shape_func)

register_shape_func("add", False, broadcast_shape_func)
register_shape_func("subtract", False, broadcast_shape_func)
register_shape_func("multiply", False, broadcast_shape_func)
register_shape_func("divide", False, broadcast_shape_func)
register_shape_func("floor_divide", False, broadcast_shape_func)
register_shape_func("mod", False, broadcast_shape_func)
register_shape_func("floor_mod", False, broadcast_shape_func)
register_shape_func("logical_and", False, broadcast_shape_func)
register_shape_func("logical_or", False, broadcast_shape_func)
register_shape_func("logical_xor", False, broadcast_shape_func)
register_shape_func("bitwise_not", False, broadcast_shape_func)
register_shape_func("bitwise_and", False, broadcast_shape_func)
register_shape_func("bitwise_or", False, broadcast_shape_func)
register_shape_func("bitwise_xor", False, broadcast_shape_func)
register_shape_func("equal", False, broadcast_shape_func)
register_shape_func("not_equal", False, broadcast_shape_func)
register_shape_func("less", False, broadcast_shape_func)
register_shape_func("less_equal", False, broadcast_shape_func)
register_shape_func("greater", False, broadcast_shape_func)
register_shape_func("greater_equal", False, broadcast_shape_func)
register_shape_func("maximum", False, broadcast_shape_func)
register_shape_func("minimum", False, broadcast_shape_func)

register_shape_func("sqrt", False, elemwise_shape_func)
register_shape_func("negative", False, elemwise_shape_func)
register_shape_func("exp", False, elemwise_shape_func)
register_shape_func("tan", False, elemwise_shape_func)
register_shape_func("fast_exp", False, elemwise_shape_func)
register_shape_func("fast_tanh", False, elemwise_shape_func)
register_shape_func("fast_erf", False, elemwise_shape_func)
