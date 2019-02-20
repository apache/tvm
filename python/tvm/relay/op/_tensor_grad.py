#pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import
from .. import expr as _expr
from ..expr import const
from .op import register_gradient
from .transform import collapse_sum_like
from .tensor import exp, negative, sigmoid, ones_like, multiply, power


@register_gradient("log")
def log_grad(orig, grad):
    """Returns [grad * (1 / x)]"""
    x = orig.args[0]
    return [multiply(grad, ones_like(x) / x)]


@register_gradient("exp")
def exp_grad(orig, grad):
    """Returns [grad * exp(x)]"""
    return [multiply(grad, exp(orig.args[0]))]


@register_gradient("sqrt")
def sqrt_grad(orig, grad):
    """Returns [grad * 0.5 * (x ^ -0.5)]"""
    a = const(0.5)  # (TODO) type?
    return [multiply(grad, a * power(orig.args[0], negative(a)))]


@register_gradient("sigmoid")
def sigmoid_grad(orig, grad):
    """Returns [grad * sigmoid(x) * (1 - sigmoid(x))]."""
    return [multiply(grad, orig * (ones_like(orig) - orig))]


@register_gradient("tanh")
def tanh_grad(orig, grad):
    """Returns grad * (1 - tanh(x) * tanh(x))."""
    return [multiply(grad, ones_like(orig) - orig * orig)]


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
    return [collapse_sum_like(multiply(grad, orig.args[1]), orig.args[0]),
            collapse_sum_like(multiply(grad, orig.args[0]), orig.args[1])]
