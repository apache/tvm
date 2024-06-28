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
# pylint: disable=invalid-name, unused-argument
"""Gradient definitions for Relay operators"""
import tvm
from tvm.topi.nn.utils import get_pad_tuple
from tvm.topi.utils import get_const_tuple
from tvm.error import OpError


from ..expr import Tuple, TupleGetItem, const, Var
from ..ty import TensorType
from ..loops import while_loop
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
    sqrt,
    zeros_like,
    equal,
    shape_of,
    log,
    concatenate,
)
from .transform import (
    broadcast_to_like,
    collapse_sum_like,
    cast_like,
    reshape,
    reshape_like,
    strided_slice,
    take,
    transpose,
    where,
    repeat,
    expand_dims,
    full_like,
    split,
    squeeze,
    strided_set,
    arange,
    scatter_nd,
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
    two = const(2.0, dtype=x.checked_type.dtype)
    return [grad * ones / (log(two) * x)]


@register_gradient("log10")
def log10_grad(orig, grad):
    """Returns [grad * 1 / (log(10) * x)]"""
    x = orig.args[0]
    ones = ones_like(x)
    ten = const(10.0, dtype=x.checked_type.dtype)
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
    """Returns [grad * sinh(x)]"""
    x = orig.args[0]
    return [grad * sinh(x)]


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


@register_gradient("acos")
def acos_grad(orig, grad):
    """Returns [grad * -1/((1 - (x ^ 2)) ^ 1/2)]"""
    x = orig.args[0]
    ones = ones_like(x)
    return [grad * (-ones / sqrt(ones - (x * x)))]


@register_gradient("acosh")
def acosh_grad(orig, grad):
    """Returns [grad * 1/((x - 1) ^ 1/2 * (x + 1) ^ 1/2)]"""
    x = orig.args[0]
    ones = ones_like(x)
    return [grad * ones / sqrt((x * x) - ones)]


@register_gradient("asin")
def asin_grad(orig, grad):
    """Returns [grad * 1/((1 - (x ^ 2)) ^ (1/2))]"""
    x = orig.args[0]
    ones = ones_like(x)
    return [grad * ones / sqrt(ones - (x * x))]


@register_gradient("asinh")
def asinh_grad(orig, grad):
    """Returns [grad * 1/((1 + (x ^ 2)) ^ (1/2))]"""
    x = orig.args[0]
    ones = ones_like(x)
    return [grad * ones / sqrt(ones + (x * x))]


@register_gradient("atan")
def atan_grad(orig, grad):
    """Returns [grad * 1 / (1 + x ^ 2)]"""
    x = orig.args[0]
    ones = ones_like(x)
    return [grad * ones / (ones + (x * x))]


@register_gradient("atanh")
def atanh_grad(orig, grad):
    """Returns [grad * 1 / (1 - x ^ 2)]"""
    x = orig.args[0]
    ones = ones_like(x)
    return [grad * ones / (ones - (x * x))]


@register_gradient("exp")
def exp_grad(orig, grad):
    """Returns [grad * exp(x)]"""
    return [grad * exp(orig.args[0])]


@register_gradient("sqrt")
def sqrt_grad(orig, grad):
    """Returns [grad * 0.5 * (x ^ -0.5)]"""
    x = orig.args[0]
    a = const(0.5, dtype=x.checked_type.dtype)
    return [grad * a * power(x, negative(a))]


@register_gradient("sigmoid")
def sigmoid_grad(orig, grad):
    """Returns [grad * sigmoid(x) * (1 - sigmoid(x))]."""
    return [grad * orig * (ones_like(orig) - orig)]


@register_gradient("tanh")
def tanh_grad(orig, grad):
    """Returns grad * (1 - tanh(x) * tanh(x))."""
    return [grad * (ones_like(orig) - orig * orig)]


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
    return [collapse_sum_like(grad, orig.args[0]), collapse_sum_like(grad, orig.args[1])]


@register_gradient("subtract")
def subtract_grad(orig, grad):
    """Returns [grad, -grad]"""
    return [collapse_sum_like(grad, orig.args[0]), collapse_sum_like(negative(grad), orig.args[1])]


@register_gradient("multiply")
def multiply_grad(orig, grad):
    """Returns [grad * y, grad * x]"""
    x, y = orig.args
    return [collapse_sum_like(grad * y, x), collapse_sum_like(grad * x, y)]


@register_gradient("divide")
def divide_grad(orig, grad):
    """Returns [grad / y,  - grad * (x / y) / y]"""
    x, y = orig.args
    return [collapse_sum_like(grad / y, x), collapse_sum_like(-(grad * orig / y), y)]


@register_gradient("zeros")
def zeros_grad(orig, grad):
    """Returns []"""
    return []


@register_gradient("dyn.zeros")
def dyn_zeros_grad(orig, grad):
    """Returns the gradient of dyn.zeros which is just zero."""
    assert len(orig.args) == 1
    return [zeros_like(orig.args[0])]


@register_gradient("ones")
def ones_grad(orig, grad):
    """Returns []"""
    return []


@register_gradient("dyn.ones")
def dyn_ones_grad(orig, grad):
    """Returns the gradient of dyn.ones which is just zero."""
    assert len(orig.args) == 1
    return [zeros_like(orig.args[0])]


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


@register_gradient("collapse_sum_to")
def collapse_sum_to_grad(orig, grad):
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


@register_gradient("erf")
def erf_grad(orig, grad):
    # c_2_div_sqrt_pi = 2.0 / math.sqrt(math.pi)
    (inp,) = orig.args
    c_2_div_sqrt_pi = const(1.1283791670955126, dtype=inp.checked_type.dtype)
    return [c_2_div_sqrt_pi * exp(-inp * inp) * grad]


@register_gradient("clip")
def clip_grad(orig, grad):
    """Returns grad * (select(x < min || max < x , 0, 1))."""
    x = orig.args[0]
    a_min = orig.attrs.get_int("a_min")
    a_max = orig.attrs.get_int("a_max")
    a_mins = broadcast_to_like(const(a_min, dtype=x.checked_type.dtype), x)
    a_maxs = broadcast_to_like(const(a_max, dtype=x.checked_type.dtype), x)
    zeros = zeros_like(x)
    ones = ones_like(x)
    return [where(less(x, a_mins), zeros, where(less(a_maxs, x), zeros, ones * grad))]


@register_gradient("nn.max_pool2d")
def max_pool2d_grad(orig, grad):
    """Returns the gradient of max_pool2d."""
    attrs = orig.attrs
    pool_grad = _nn.max_pool2d_grad(
        grad,
        orig.args[0],
        pool_size=attrs.pool_size,
        strides=attrs.strides,
        padding=attrs.padding,
        layout=attrs.layout,
        ceil_mode=attrs.ceil_mode,
    )
    return [pool_grad]


@register_gradient("nn.avg_pool2d")
def avg_pool2d_grad(orig, grad):
    """Returns the gradient of avg_pool2d."""
    attrs = orig.attrs
    pool_grad = _nn.avg_pool2d_grad(
        grad,
        orig.args[0],
        pool_size=attrs.pool_size,
        strides=attrs.strides,
        padding=attrs.padding,
        layout=attrs.layout,
        ceil_mode=attrs.ceil_mode,
        count_include_pad=attrs.count_include_pad,
    )
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

    pool_grad = _nn.avg_pool2d_grad(
        grad, data, pool_size=pool_size, strides=(1, 1), padding=(0, 0), layout=layout
    )
    return [pool_grad]


@register_gradient("concatenate")
def concatenate_grad(orig, grad):
    """
    Returns the gradient of concatenate, which is just the downstream gradient
    split across the inputs.
    """
    assert len(orig.args) == 1
    t = orig.args[0]

    # calculate split indices. TODO(@altanh): support Any?
    axis_dims = [ty.shape[orig.attrs.axis] for ty in t.checked_type.fields]
    splits, cumsum = [], 0
    for dim in axis_dims[:-1]:
        if isinstance(dim, tvm.tir.IntImm):
            dim = dim.value
        cumsum += dim
        splits.append(cumsum)

    grads = split(grad, tuple(splits), axis=orig.attrs.axis).tuple_value
    return [grads]


@register_gradient("nn.conv2d")
def conv2d_grad(orig, grad):
    """Gradient of conv2d"""
    attrs = orig.attrs
    data, weight = orig.args
    data_shape = get_const_tuple(data.checked_type.shape)
    weight_shape = get_const_tuple(weight.checked_type.shape)
    _, _, grad_h, grad_w = get_const_tuple(orig.checked_type.shape)
    _, _, in_h, in_w = data_shape
    _, _, filter_h, filter_w = weight_shape

    # infer output_padding
    fpad_top, fpad_left, fpad_bottom, fpad_right = get_pad_tuple(
        get_const_tuple(attrs.padding), (filter_h, filter_w)
    )
    stride_h, stride_w = get_const_tuple(attrs.strides)
    out_h = (grad_h - 1) * stride_h - fpad_top - fpad_bottom + filter_h
    out_w = (grad_w - 1) * stride_w - fpad_left - fpad_right + filter_w
    output_padding = (in_h - out_h, in_w - out_w)

    assert attrs.data_layout == "NCHW", "only support NCHW data layout"
    assert attrs.kernel_layout == "OIHW", "only support OIHW kernel layout"
    assert attrs.out_layout in ["", "NCHW"], "only support NCHW output layout"

    if attrs.out_dtype in ["", None]:
        assert data.checked_type, "Call InferType first."
        out_dtype = data.checked_type.dtype
    else:
        out_dtype = attrs.out_dtype

    backward_data = _nn.conv2d_transpose(
        grad,
        weight,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        output_padding=output_padding,
        out_dtype=out_dtype,
    )

    backward_weight = _nn.conv2d_backward_weight(
        grad,
        data,
        strides=attrs.strides,
        padding=attrs.padding,
        dilation=attrs.dilation,
        groups=attrs.groups,
        channels=attrs.channels,
        kernel_size=(filter_h, filter_w),
        grad_layout=attrs.out_layout if attrs.out_layout else attrs.data_layout,
        data_layout=attrs.data_layout,
        kernel_layout=attrs.kernel_layout,
        out_dtype=out_dtype,
    )

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
    return [grad - _sum(grad, axis=orig.attrs.axis, keepdims=True) * exp(orig)]


@register_gradient("nn.bias_add")
def bias_add_grad(orig, grad):
    """Returns gradient of bias_add"""
    data = orig.args[0]
    return [
        collapse_sum_like(grad, data),
        _sum(grad, orig.attrs.axis, keepdims=False, exclude=True),
    ]


@register_gradient("nn.dense")
def dense_grad(orig, grad):
    """Returns [grad' @ weight, data @ grad']"""
    data, weight = orig.args
    return [
        collapse_sum_like(
            _nn.dense(grad, transpose(weight), units=weight.checked_type.shape[1]), data
        ),
        collapse_sum_like(
            _nn.dense(transpose(grad), transpose(data), units=data.checked_type.shape[1]), weight
        ),
    ]


@register_gradient("nn.matmul")
def matmul_grad(orig, grad):
    """Returns [grad' @ tensor_b, tensor_a @ grad']"""
    tensor_a, tensor_b = orig.args
    if (orig.attrs["transpose_a"], orig.attrs["transpose_b"]) == (True, True):
        return [
            collapse_sum_like(
                _nn.matmul(tensor_b, grad, transpose_a=True, transpose_b=True), tensor_a
            ),
            collapse_sum_like(
                _nn.matmul(grad, tensor_a, transpose_a=True, transpose_b=True), tensor_b
            ),
        ]
    if (orig.attrs["transpose_a"], orig.attrs["transpose_b"]) == (True, False):
        return [
            collapse_sum_like(_nn.matmul(tensor_b, grad, transpose_b=True), tensor_a),
            collapse_sum_like(_nn.matmul(tensor_a, grad), tensor_b),
        ]
    if (orig.attrs["transpose_a"], orig.attrs["transpose_b"]) == (False, True):
        # Keep using Dense op here for not involving extra ops
        # TODO(jcf94): Merge all to nn.matmul when it is finally ready
        return dense_grad(orig, grad)
    # (orig.attrs["transpose_a"], orig.attrs["transpose_b"]) == (False, False)
    return [
        collapse_sum_like(_nn.matmul(grad, tensor_b, transpose_b=True), tensor_a),
        collapse_sum_like(_nn.matmul(tensor_a, grad, transpose_a=True), tensor_b),
    ]


@register_gradient("nn.batch_matmul")
def batch_matmul_grad(orig, grad):
    """gradient for nn.batch_matmul: in einsum LHS_bik,RHS_bjk->RES_bij
    grads: GRAD_OUT_bij,RHS_bjk->GRAD_IN_LHS_bik
           GRAD_OUT_bij,LHS_bik->GRAD_IN_RHS_bjk
    """
    lhs, rhs = orig.args
    if (orig.attrs["transpose_a"], orig.attrs["transpose_b"]) == (True, True):
        # ki,   jk  ->  ij
        # jk,   ij  ->  ki
        # ij,   ki  ->  jk
        return [
            collapse_sum_like(_nn.batch_matmul(rhs, grad, transpose_a=True, transpose_b=True), lhs),
            collapse_sum_like(_nn.batch_matmul(grad, lhs, transpose_a=True, transpose_b=True), rhs),
        ]
    if (orig.attrs["transpose_a"], orig.attrs["transpose_b"]) == (True, False):
        # ki,   kj  ->  ij
        # kj,   ij  ->  ki
        # ki,   ij  ->  kj
        return [
            collapse_sum_like(
                _nn.batch_matmul(rhs, grad, transpose_a=False, transpose_b=True), lhs
            ),
            collapse_sum_like(
                _nn.batch_matmul(lhs, grad, transpose_a=False, transpose_b=False), rhs
            ),
        ]
    if (orig.attrs["transpose_a"], orig.attrs["transpose_b"]) == (False, True):
        # ik,   jk  ->  ij
        # ij,   jk  ->  ik
        # ij,   ik  ->  jk
        # Keep using NT format batch_matmul here for not involving extra ops
        # TODO(jcf94): Merge all to normal batch_matmul when it is finally ready
        return [
            collapse_sum_like(
                _nn.batch_matmul(
                    grad,
                    transpose(rhs, [0, 2, 1]),
                    transpose_a=False,
                    transpose_b=True,
                ),
                lhs,
            ),
            collapse_sum_like(
                _nn.batch_matmul(
                    transpose(grad, [0, 2, 1]),
                    transpose(lhs, [0, 2, 1]),
                    transpose_a=False,
                    transpose_b=True,
                ),
                rhs,
            ),
        ]
    # (orig.attrs["transpose_a"], orig.attrs["transpose_b"]) == (False, False)
    # ik,   kj  ->  ij
    # ij,   kj  ->  ik
    # ik,   ij  ->  kj
    return [
        collapse_sum_like(_nn.batch_matmul(grad, rhs, transpose_a=False, transpose_b=True), lhs),
        collapse_sum_like(_nn.batch_matmul(lhs, grad, transpose_a=True, transpose_b=False), rhs),
    ]


@register_gradient("reshape")
def reshape_grad(orig, grad):
    """Gradient of reshape"""
    return [reshape_like(grad, orig.args[0])]


@register_gradient("dyn.reshape")
def dyn_reshape_grad(orig, grad):
    """Gradient of dyn_reshape"""
    return [reshape_like(grad, orig.args[0]), zeros_like(orig.args[1])]


@register_gradient("shape_of")
def shape_of_grad(orig, grad):
    """Gradient of shape_of"""
    return [zeros_like(orig.args[0])]


@register_gradient("cast")
def cast_grad(orig, grad):
    x = orig.args[0]
    return [cast_like(grad, x)]


@register_gradient("cast_like")
def cast_like_grad(orig, grad):
    x, like = orig.args
    return [cast_like(grad, x), zeros_like(like)]


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


@register_gradient("mean")
def mean_grad(orig, grad):
    """Returns grad broadcasted to data dims"""
    data, axis = orig.args[0], _get_reduce_axis(orig)
    shape = data.checked_type.concrete_shape
    if axis is None:
        axis = list(range(len(data.checked_type.concrete_shape)))
    if not orig.attrs.keepdims:
        grad = _unreduce_expand(grad, axis)
    mult = 1.0
    for a in axis:
        mult /= shape[a]
    return [broadcast_to_like(grad * const(mult, dtype=data.checked_type.dtype), data)]


@register_gradient("variance")
def variance_grad(orig, grad):
    """Note that we take mean as an argument in the variance node"""
    data, data_mean, axis = orig.args[0], orig.args[1], _get_reduce_axis(orig)
    unbiased = orig.attrs.unbiased
    shape = data.checked_type.concrete_shape
    if axis is None:
        axis = list(range(len(data.checked_type.concrete_shape)))
    if not orig.attrs.keepdims:
        grad = _unreduce_expand(grad, axis)
    mult1 = 2.0
    mult2 = -2.0
    count = 1
    for a in axis:
        count *= shape[a]
    if unbiased:
        mult2 = mult2 * count / (count - 1)
        count -= 1
    mult1 /= count
    return [
        (grad * const(mult1, dtype=data.checked_type.dtype)) * data,
        const(mult2, dtype=data.checked_type.dtype) * grad * data_mean,
    ]


@register_gradient("copy")
def copy_grad(orig, grad):
    return [grad]


@register_gradient("nn.cross_entropy")
def cross_entropy_grad(orig, grad):
    x, y = orig.args
    shape = shape_of(x)
    batch_size = take(shape, const(0, dtype="int32"), axis=0)
    grad = grad / batch_size.astype(x.checked_type.dtype)
    return [-grad * y / x, -grad * log(x)]


@register_gradient("nn.cross_entropy_with_logits")
def cross_entropy_with_logits_grad(orig, grad):
    x, y = orig.args
    shape = shape_of(x)
    batch_size = take(shape, const(0, dtype="int32"), axis=0)
    grad = grad / batch_size.astype(x.checked_type.dtype)
    return [-grad * y, -grad * x]


@register_gradient("take")
def take_grad(orig, grad):
    """
    Returns the gradient of take.
    """

    def make_scalar_tensor(v):
        if isinstance(v, int):
            v = const(v, dtype="int32")
        return reshape(v, (1,))

    # TODO(@altanh): we currently assume indices are in range
    data, indices = orig.args
    axis = orig.attrs.axis
    batch_dims = orig.attrs.batch_dims
    zero, one = map(make_scalar_tensor, [0, 1])
    data_grad = zeros_like(data)
    try:
        data_shape = data.checked_type.concrete_shape
    except TypeError as ty_err:
        raise OpError("currently take_grad only supports data with concrete shape") from ty_err
    if axis is None:
        axis = 0
        data_grad = reshape(data_grad, (-1,))
        data_shape = 1
        for dim in data.checked_type.concrete_shape:
            data_shape *= dim
        data_shape = (data_shape,)
    else:
        axis = int(axis)
    if batch_dims is None:
        batch_dims = 0
    else:
        batch_dims = int(batch_dims)
    if batch_dims != 0:
        raise OpError("take_grad only supports batch_dims equales to 0")
    strides = [1] * len(data_shape)

    if len(indices.checked_type.shape) == 0:
        # axis on grad has been squeezed in this case
        num_indices = one
        indices = reshape(indices, (1,))
        grad = expand_dims(grad, int(axis))
    elif len(indices.checked_type.shape) == 1:
        num_indices = take(shape_of(indices), zero, axis=0)
    else:
        raise OpError("take_grad only supports scalar or 1D indices")

    def loop_cond(data_grad, i):
        return squeeze(less(i, num_indices))

    def loop_body(data_grad, i):
        index = take(indices, i, axis=0)
        grad_slice = take(grad, i, axis=axis)
        begin, end = [], []
        for ax, size in enumerate(data_shape):
            size = make_scalar_tensor(size)
            begin.append(zero if ax != axis else index)
            end.append(size if ax != axis else index + one)
        begin, end = concatenate(begin, axis=0), concatenate(end, axis=0)
        # data_grad[:,...,index at axis,...,:] += grad_slice
        update = strided_slice(data_grad, begin, end, strides=strides)
        update = update + grad_slice  # no need to expand grad_slice since i has shape (1,)
        next_data_grad = strided_set(data_grad, update, begin, end, strides=strides)
        return (next_data_grad, i + one)

    loop_vars = [
        Var("data_grad", type_annotation=TensorType(data_shape, data.checked_type.dtype)),
        Var("i", type_annotation=TensorType((1,), "int32")),
    ]

    loop = while_loop(loop_cond, loop_vars, loop_body)
    result = loop(data_grad, zero)
    data_grad = TupleGetItem(result, 0)

    if orig.attrs.axis is None:
        data_grad = reshape_like(data_grad, data)

    return [data_grad, zeros_like(orig.args[1])]


@register_gradient("contrib_reverse_reshape")
def reverse_reshape_grad(orig, grad):
    """
    Returns the gradient of reverse_reshape (same as reshape).
    """
    return [reshape_like(grad, orig.args[0])]


@register_gradient("stack")
def stack_grad(orig, grad):
    """
    Returns grad split across stacked inputs.
    """
    stack_axis = int(orig.attrs.axis)
    sections = len(orig.args[0].checked_type.fields)
    splits = split(grad, sections, stack_axis)
    splits = Tuple([squeeze(x, axis=[stack_axis]) for x in splits])
    return [splits]


@register_gradient("squeeze")
def squeeze_grad(orig, grad):
    """
    Returns grad expanded to input size.
    """
    # this should work, can't use expand_dims since we lose
    # squeeze information when axis=None
    return [reshape_like(grad, orig.args[0])]


@register_gradient("expand_dims")
def expand_dims_grad(orig, grad):
    """
    Returns grad squeezed on expanded dims.
    """
    axis = int(orig.attrs.axis)
    for _ in range(orig.attrs.num_newaxis):
        grad = squeeze(grad, axis=[axis])
    return [grad]


@register_gradient("arange")
def arange_grad(orig, grad):
    """
    Returns the gradient of arange.
    """
    start, stop, step = orig.args
    length = take(shape_of(orig), const(0, dtype="int32"), axis=0)

    grad_start = cast_like(_sum(grad), start)
    grad_stop = zeros_like(stop)
    grad_step = cast_like(arange(length, dtype="int32"), grad) * grad
    grad_step = cast_like(_sum(grad_step), step)

    return [grad_start, grad_stop, grad_step]


@register_gradient("gather_nd")
def gather_nd_grad(orig, grad):
    """
    Returns the gradient of gather_nd, which is simply scatter_nd.
    """
    data, indices = orig.args
    return [scatter_nd(zeros_like(data), indices, grad, mode="add"), zeros_like(indices)]


@register_gradient("reshape_like")
def reshape_like_grad(orig, grad):
    """
    Returns the gradient of reshape_like.
    """
    data, shape_like = orig.args
    return [reshape_like(grad, data), zeros_like(shape_like)]


@register_gradient("where")
def where_grad(orig, grad):
    """
    Returns the gradient of where.
    """
    cond, x, y = orig.args
    g_zeros = zeros_like(grad)

    grad_x = collapse_sum_like(where(cond, grad, g_zeros), x)
    grad_y = collapse_sum_like(where(cond, g_zeros, grad), y)

    return [zeros_like(cond), grad_x, grad_y]


@register_gradient("less_equal")
def less_equal_grad(orig, grad):
    """
    Returns the gradient of less_equal.
    """
    return [zeros_like(orig.args[0]), zeros_like(orig.args[1])]


@register_gradient("not_equal")
def not_equal_grad(orig, grad):
    """
    Returns the gradient of not_equal (just zeros).
    """
    return [zeros_like(orig.args[0]), zeros_like(orig.args[1])]


@register_gradient("strided_slice")
def strided_slice_grad(orig, grad):
    """
    Returns the gradient of strided_slice, which is equal to grad where the
    input was sliced and zero elsewhere.
    """
    assert orig.attrs.axes is None, "grad for strided_slice with axes is not yet supported"
    x = orig.args[0]
    begin = get_const_tuple(orig.attrs.begin)
    end = get_const_tuple(orig.attrs.end)
    strides = get_const_tuple(orig.attrs.strides)
    if orig.attrs.slice_mode == "size":
        # convert sizes to ending indices and ignore strides
        end = list(end)
        for i, (start, size) in enumerate(zip(begin, end)):
            if size == -1:
                end[i] = int(x.checked_type.shape[i])
            else:
                end[i] = start + size
        strides = None
    else:
        assert orig.attrs.slice_mode == "end"
    return [strided_set(zeros_like(x), grad, begin, end, strides)]


@register_gradient("one_hot")
def one_hot_grad(orig, grad):
    """
    Returns the gradient of one_hot, which is the sum of grad at on and off
    indices for on_value and off_value respectively.
    """
    indices, on_value, off_value = orig.args

    g_zeros = zeros_like(grad)
    on_mask = equal(orig, on_value)
    grad_on = _sum(where(on_mask, grad, g_zeros))
    grad_off = _sum(where(on_mask, g_zeros, grad))

    return [zeros_like(indices), cast_like(grad_on, on_value), cast_like(grad_off, off_value)]
