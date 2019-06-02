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
from ..expr import const
from ..ir_pass import infer_type
from .op import register_gradient
from .transform import collapse_sum_like, broadcast_to_like, where, cast
from .tensor import exp, negative, power, less, equal, divide
from .tensor import zeros_like, ones_like
from .reduce import sum
from .nn import dense

@register_gradient("log")
def log_grad(orig, grad):
    """Returns [grad * (1 / x)]"""
    x = orig.args[0]
    return [grad * ones_like(x) / x]


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

@register_gradient("sum")
def sum_grad(orig, grad):
    """Returns [broadcast_to_like(grad, x)]"""
    return [broadcast_to_like(grad, orig.args[0])]

@register_gradient("max")
def max_grad(orig, grad):
    """Returns the gradient of max"""
    x, axis = orig.args[0], orig.attrs.axis
    orig = broadcast_to_like(orig, x)
    grad = broadcast_to_like(grad, x)
    indicators = cast(equal(orig, x), 'float32')
    count = broadcast_to_like(sum(indicators, axis, True), x)
    return [divide(indicators, count) * grad]

@register_gradient("nn.softmax")
def softmax_grad(orig, grad):
    """Returns [(grad - sum(grad * orig, orig.attrs.axis, True)) * orig]"""
    return [(grad - sum(grad * orig, orig.attrs.axis, True)) * orig]

@register_gradient("negative")
def negative_grad(orig, grad):
    return [-broadcast_to_like(grad, orig.args[0])]

# UNTESTED BELOW

@register_gradient("nn.dense")
def dense_grad(orig, grad):
    x, y = orig.args
    return [collapse_sum_like(dense(grad, y), x),
            collapse_sum_like(dense(grad, x), y)]

@register_gradient("cast")
def cast_grad(orig, grad):
    return [grad.astype(orig.attrs.dtype)]

@register_gradient("reshape")
def reshape_grad(orig, grad):
    return [broadcast_to_like(grad, orig.args[0])]
