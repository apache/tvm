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
#pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import

from topi.nn.util import get_pad_tuple
from topi.util import get_const_tuple

from ..expr import Tuple, TupleGetItem, const
from . import nn as _nn
from .op import register_gradient
from .reduce import sum as _sum
from .tensor import (
    cos,
    cosh,
    exp,
    less,
    negative,
    ones_like,
    power,
    sin,
    sinh,
    zeros_like,
    equal,
    shape_of,
    log)
from .transform import (
    broadcast_to_like,
    collapse_sum_like,
    cast_like,
    reshape,
    reshape_like,
    strided_slice,
    take,
    tile,
    transpose,
    where,
    repeat,
    expand_dims,
    full_like
)


@register_gradient("log")
def log_grad(orig, grad):
    """Returns [grad * (1 / x)]"""
    x = orig.args[0]
    return [grad * ones_like(x) / x]


@register_gradient("log2")
def log2_grad(orig, grad):
    """Returns [grad * 1 / (log(2) * x)]"""
    x = orig.args[0]
    ones = ones_like(x)
    two = const(2.0)
    return [grad * ones / (log(two) * x)]


@register_gradient("log10")
def log10_grad(orig, grad):
    """Returns [grad * 1 / (log(10) * x)]"""
    x = orig.args[0]
    ones = ones_like(x)
    ten = const(10.0)
    return [grad * ones / (log(ten) * x)]


@register_gradient("tan")
def tan_grad(orig, grad):
    """Returns [grad / (cos^2(x))]"""
    x = orig.args[0]
    return [grad / (cos(x) * cos(x))]


@register_gradient("cos")
def cos_grad(orig, grad):
    """Returns [grad * (-sin(x))]"""
    x = orig.args[0]
    ones = ones_like(x)
    return [grad * (-ones * sin(x))]


@register_gradient("cosh")
def cosh_grad(orig, grad):
    """Returns [grad * (-sinh(x))]"""
    x = orig.args[0]
    ones = ones_like(x)
    return [grad * (-ones * sinh(x))]


@register_gradient("sin")
def sin_grad(orig, grad):
    """Returns [grad * cos(x)]"""
    x = orig.args[0]
    return [grad * cos(x)]

@register_gradient("sinh")
def sinh_grad(orig, grad):
    """Returns [grad * cosh(x)]"""
    x = orig.args[0]
    return [grad * cosh(x)]

@register_gradient("atan")
def atan_grad(orig, grad):
    """Returns [grad * 1 / (1 + x ^ 2)]"""
    x = orig.args[0]
    a = const(2.0)
    return [grad * ones_like(x) / (ones_like(x) + power(x, a))]

@register_gradient("exp")
def exp_grad(orig, grad):
    """Returns [grad * exp(x)]"""
    return [grad * exp(orig.args[0])]


@register_gradient("sqrt")
def sqrt_grad(orig, grad):
    """Returns [grad * 0.5 * (x ^ -0.5)]"""
    a = const(0.5)  # (TODO) type?
    return [grad * a * power(orig.args[0], negative(a))]


@register_gradient("sigmoid")
def sigmoid_grad(orig, grad):
    """Returns [grad * sigmoid(x) * (1 - sigmoid(x))]."""
    return [grad * orig * (ones_like(orig) - orig)]


@register_gradient("tanh")
def tanh_grad(orig, grad):
    """Returns grad * (1 - tanh(x) * tanh(x))."""
    return [grad * ones_like(orig) - orig * orig]


@register_gradient("nn.relu")
def relu_grad(orig, grad):
    """Returns grad * (select(x < 0, 0, 1))."""
    x = orig.args[0]
    zeros = zeros_like(x)
    ones = ones_like(x)
    return [where(less(x, zeros), zeros, ones * grad)]


@register_gradient("add")
def add_grad(orig, grad):
    """Returns [grad, grad]"""
    return [collapse_sum_like(grad, orig.args[0]),
            collapse_sum_like(grad, orig.args[1])]


@register_gradient("subtract")
def subtract_grad(orig, grad):
    """Returns [grad, -grad]"""
    return [collapse_sum_like(grad, orig.args[0]),
            collapse_sum_like(negative(grad), orig.args[1])]


@register_gradient("multiply")
def multiply_grad(orig, grad):
    """Returns [grad * y, grad * x]"""
    x, y = orig.args
    return [collapse_sum_like(grad * y, x),
            collapse_sum_like(grad * x, y)]


@register_gradient("divide")
def divide_grad(orig, grad):
    """Returns [grad / y,  - grad * (x / y) / y]"""
    x, y = orig.args
    return [collapse_sum_like(grad / y, x),
            collapse_sum_like(- (grad * orig / y), y)]


@register_gradient("zeros")
def zeros_grad(orig, grad):
    """Returns []"""
    return []


@register_gradient("ones")
def ones_grad(orig, grad):
    """Returns []"""
    return []


@register_gradient("zeros_like")
def zeros_like_grad(orig, grad):
    """Returns [0]"""
    return [orig]


@register_gradient("ones_like")
def ones_like_grad(orig, grad):
    """Returns [0]"""
    return [zeros_like(orig.args[0])]


@register_gradient("collapse_sum_like")
def collapse_sum_like_grad(orig, grad):
    """Returns [broadcast_to_like(grad, x), 0]"""
    x, y = orig.args
    return [broadcast_to_like(grad, x), zeros_like(y)]


@register_gradient("abs")
def abs_grad(orig, grad):
    """Returns grad * (select(x < 0, -1, 1))."""
    x = orig.args[0]
    zeros = zeros_like(x)
    ones = ones_like(x)
    return [where(less(x, zeros), -ones * grad, ones * grad)]


@register_gradient("clip")
def clip_grad(orig, grad):
    """Returns grad * (select(x < min || max < x , 0, 1))."""
    x = orig.args[0]
    a_min = orig.attrs.get_int("a_min")
    a_max = orig.attrs.get_int("a_max")
    a_mins = broadcast_to_like(const(a_min), x)
    a_maxs = broadcast_to_like(const(a_max), x)
    zeros = zeros_like(x)
    ones = ones_like(x)
    return [where(less(x, a_mins), zeros, where(less(a_maxs, x), zeros, ones * grad))]


@register_gradient("nn.max_pool2d")
def max_pool2d_grad(orig, grad):
    """Returns the gradient of max_pool2d."""
    attrs = orig.attrs
    pool_grad = _nn.max_pool2d_grad(grad, orig.args[0], pool_size=attrs.pool_size,
                                    strides=attrs.strides, padding=attrs.padding,
                                    layout=attrs.layout, ceil_mode=attrs.ceil_mode)
    return [pool_grad]


@register_gradient("nn.avg_pool2d")
def avg_pool2d_grad(orig, grad):
    """Returns the gradient of avg_pool2d."""
    attrs = orig.attrs
    pool_grad = _nn.avg_pool2d_grad(grad, orig.args[0], pool_size=attrs.pool_size,
                                    strides=attrs.strides, padding=attrs.padding,
                                    layout=attrs.layout, ceil_mode=attrs.ceil_mode,
                                    count_include_pad=attrs.count_include_pad)
    return [pool_grad]


@register_gradient("nn.global_avg_pool2d")
def global_avg_pool2d_grad(orig, grad):
    """Returns the gradient of global_avg_pool2d."""
    data = orig.args[0]
    shape = data.checked_type.shape
    layout = orig.attrs.layout

    # we assume NCHW or NHWC layout for now, but easy to add more
    assert layout in ["NCHW", "NHWC"]
    if layout == "NCHW":
        pool_size = shape[2], shape[3]
    elif layout == "NHWC":
        pool_size = shape[1], shape[2]

    pool_grad = _nn.avg_pool2d_grad(grad, data, pool_size=pool_size,
                                    strides=(1, 1), padding=(0, 0),
                                    layout=layout)
    return [pool_grad]


# not implemented, this is only for testing.
@register_gradient("concatenate")
def concatenate_grad(orig, grad):
    assert len(orig.args) == 1
    t = orig.args[0]
    x = TupleGetItem(t, 0)
    y = TupleGetItem(t, 1)
    # Assume only two element in tuple rn.
    # In the real implementation, concatenate_grad probably need to be implemented by an operator.
    return [Tuple([zeros_like(x), zeros_like(y)])]


@register_gradient("nn.conv2d")
def conv2d_grad(orig, grad):
    """Gradient of conv2d"""
    attrs = orig.attrs
    data, weight = orig.args
    data_shape = get_const_tuple(data.checked_type.shape)
    weight_shape = get_const_tuple(weight.checked_type.shape)
    _, _, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    batch, in_channel, in_h, in_w = data_shape
    out_channel, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(get_const_tuple(attrs.padding),
                                                                 (filter_h, filter_w))
    stride_h, stride_w = get_const_tuple(attrs.strides)
    dilation_h, dilation_w = get_const_tuple(attrs.dilation)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == 'NCHW', 'only support NCHW data layout'
    assert attrs.kernel_layout == 'OIHW', 'only support OIHW kernel layout'
    assert attrs.out_layout in ['', 'NCHW'], 'only support NCHW output layout'


    backward_data = _nn.conv2d_transpose(grad, weight,
                                         strides=attrs.strides,
                                         padding=attrs.padding,
                                         dilation=attrs.dilation,
                                         groups=attrs.groups,
                                         output_padding=output_padding)
    grad = tile(grad, [1, in_channel // attrs.groups, 1, 1])
    grad = reshape(grad, [-1, 1, 0, 0])  # batch * oc * ic // groups, 1, oh, ow
    data = reshape(data, [1, -1, 0, 0])  # 1, batch * ic, ih, iw

    backward_weight = _nn.conv2d(data, grad,
                                 strides=attrs.dilation,
                                 padding=attrs.padding,
                                 dilation=attrs.strides,
                                 groups=in_channel * batch)
    # infer shape of backward_weight
    padded_weight_grad_h = (in_h - (grad_h - 1) * stride_h - 1 + fpad_top + fpad_bottom) \
                           // dilation_h + 1
    padded_weight_grad_w = (in_w - (grad_w - 1) * stride_w - 1 + fpad_left + fpad_right) \
                           // dilation_w + 1
    backward_weight = reshape(backward_weight,
                              [batch, in_channel // attrs.groups, out_channel,
                               padded_weight_grad_h, padded_weight_grad_w])
    backward_weight = _sum(backward_weight, axis=0)
    backward_weight = transpose(backward_weight, [1, 0, 2, 3])

    assert padded_weight_grad_h >= filter_h
    assert padded_weight_grad_w >= filter_w
    if padded_weight_grad_h > filter_h or padded_weight_grad_w > filter_w:
        backward_weight = strided_slice(backward_weight, begin=[0, 0, 0, 0],
                                        end=[None, None, filter_h, filter_w])

    return [backward_data, backward_weight]


def _get_reduce_axis(call):
    """Helper function that returns the reduce axis of the call as plain python ints."""
    x, axis = call.args[0], call.attrs.axis
    shape = x.checked_type.concrete_shape

    # should never exclude when axis is None
    assert not (axis is None and call.attrs.exclude)

    if axis is None:
        return None

    # convert to nonnegative integers and sort
    axis = sorted([ax if ax >= 0 else len(shape) + ax for ax in map(int, axis)])
    if call.attrs.exclude:
        axis = [ax for ax in range(len(shape)) if ax not in axis]
    return axis


def _unreduce_expand(x, axis):
    """Helper function that returns x expanded on the reduced dimensions in axis."""
    # assume axis is sorted nonnegative ints
    for ax in axis:
        x = expand_dims(x, ax)
    return x


@register_gradient("max")
def max_grad(orig, grad):
    """Returns the gradient of max"""
    x, axis = orig.args[0], _get_reduce_axis(orig)
    shape = x.checked_type.concrete_shape

    repeated = orig
    if axis is None:
        repeated = full_like(x, repeated)
    else:
        # expand dims (if necessary) and repeat along each axis
        if not orig.attrs.keepdims:
            repeated = _unreduce_expand(repeated, axis)
            grad = _unreduce_expand(grad, axis)
        for ax in axis:
            repeated = repeat(repeated, shape[ax], ax)

    indicators = cast_like(equal(repeated, x), grad)
    num_selected = _sum(indicators, axis, keepdims=True)
    # spread error across all max weights
    return [indicators * grad / num_selected]


@register_gradient("nn.softmax")
def softmax_grad(orig, grad):
    """Gradient of softmax"""
    return [(grad - _sum(grad * orig, orig.attrs.axis, True)) * orig]


@register_gradient("nn.log_softmax")
def log_softmax_grad(orig, grad):
    """Gradient of log_softmax"""
    x = orig.args[0]
    sm = _nn.softmax(x, axis=orig.attrs.axis)
    grad = grad / sm
    return softmax_grad(sm, grad)


@register_gradient("nn.bias_add")
def bias_add_grad(orig, grad):
    """Returns gradient of bias_add"""
    data = orig.args[0]
    return [collapse_sum_like(grad, data),
            _sum(grad, orig.attrs.axis, keepdims=False, exclude=True)]


@register_gradient("nn.dense")
def dense_grad(orig, grad):
    """Returns [grad' @ weight, data @ grad']"""
    data, weight = orig.args
    return [collapse_sum_like(transpose(grad) * weight, data),
            collapse_sum_like(data * transpose(grad), weight)]


@register_gradient("reshape")
def reshape_grad(orig, grad):
    """Gradient of reshape"""
    return [reshape_like(grad, orig.args[0])]


@register_gradient("cast")
def cast_grad(orig, grad):
    x = orig.args[0]
    return [cast_like(grad, x)]


@register_gradient("nn.batch_flatten")
def batch_flatten_grad(orig, grad):
    """Returns grad reshaped to data dims"""
    data = orig.args[0]
    return [reshape_like(grad, data)]


@register_gradient("transpose")
def transpose_grad(orig, grad):
    """Returns grad transposed over the complement of original transpose axes"""
    orig_axes = orig.attrs.axes
    if orig_axes:
        dims = len(orig_axes)
        new_axes = [0] * dims
        for i in range(dims):
            new_axes[int(orig_axes[i])] = i
    else:
        new_axes = None
    return [transpose(grad, axes=new_axes)]


@register_gradient("negative")
def negative_grad(orig, grad):
    """Returns -grad"""
    return [-grad]


@register_gradient("sum")
def sum_grad(orig, grad):
    """Returns grad broadcasted to data dims"""
    data, axis = orig.args[0], _get_reduce_axis(orig)
    if not orig.attrs.keepdims:
        if axis is None:
            axis = list(range(len(data.checked_type.concrete_shape)))
        grad = _unreduce_expand(grad, axis)
    return [broadcast_to_like(grad, data)]


@register_gradient("nn.cross_entropy")
def cross_entropy_grad(orig, grad):
    x, y = orig.args
    shape = shape_of(x)
    batch_size = take(shape, const(0, dtype='int32'), axis=0)
    grad = grad / batch_size.astype('float32')
    return [-grad * y / x, -grad * log(x)]


@register_gradient("nn.cross_entropy_with_logits")
def cross_entropy_with_logits_grad(orig, grad):
    x, y = orig.args
    shape = shape_of(x)
    batch_size = take(shape, const(0, dtype='int32'), axis=0)
    grad = grad / batch_size.astype('float32')
    return [-grad * y, -grad * x]
