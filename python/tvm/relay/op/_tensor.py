#pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import
import topi
from .op import register_compute, register_schedule, register_pattern
from .op import schedule_injective, OpPattern

schedule_broadcast = schedule_injective
schedule_elemwise = schedule_injective

# log
@register_compute("log")
def log_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.log(inputs[0])]

register_schedule("log", schedule_broadcast)

# exp
@register_compute("exp")
def exp_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.exp(inputs[0])]

register_schedule("exp", schedule_broadcast)

# sqrt
@register_compute("sqrt")
def sqrt_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.sqrt(inputs[0])]

register_schedule("sqrt", schedule_broadcast)

# sigmoid
@register_compute("sigmoid")
def sigmoid_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.sigmoid(inputs[0])]

register_schedule("sigmoid", schedule_broadcast)

# floor
@register_compute("floor")
def floor_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.floor(inputs[0])]

register_schedule("floor", schedule_broadcast)

# ceil
@register_compute("ceil")
def ceil_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.ceil(inputs[0])]

register_schedule("ceil", schedule_broadcast)

# trunc
@register_compute("trunc")
def trunc_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.trunc(inputs[0])]

register_schedule("trunc", schedule_broadcast)

# round
@register_compute("round")
def round_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.round(inputs[0])]

register_schedule("round", schedule_broadcast)

# abs
@register_compute("abs")
def abs_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.abs(inputs[0])]

register_schedule("abs", schedule_broadcast)

# tanh
@register_compute("tanh")
def tanh_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.tanh(inputs[0])]

register_schedule("tanh", schedule_broadcast)

# negative
@register_compute("negative")
def negative_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.negative(inputs[0])]

register_schedule("negative", schedule_broadcast)

# add
@register_compute("add")
def add_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.add(inputs[0], inputs[1])]

register_schedule("add", schedule_injective)

# subtract
@register_compute("subtract")
def subtract_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.subtract(inputs[0], inputs[1])]

register_schedule("subtract", schedule_broadcast)

# multiply
@register_compute("multiply")
def multiply_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.multiply(inputs[0], inputs[1])]

register_schedule("multiply", schedule_broadcast)

# divide
@register_compute("divide")
def divide_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.divide(inputs[0], inputs[1])]

register_schedule("divide", schedule_broadcast)

# power
@register_compute("power")
def power_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.power(inputs[0], inputs[1])]

register_schedule("power", schedule_injective)

# mod
@register_compute("mod")
def mod_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.mod(inputs[0], inputs[1])]

register_schedule("mod", schedule_broadcast)

# equal
@register_compute("equal")
def equal_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.equal(inputs[0], inputs[1])]

register_schedule("equal", schedule_broadcast)

# not_equal
@register_compute("not_equal")
def not_equal_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.not_equal(inputs[0], inputs[1])]

register_schedule("not_equal", schedule_broadcast)

# less
@register_compute("less")
def less_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.less(inputs[0], inputs[1])]

register_schedule("less", schedule_broadcast)

# less equal
@register_compute("less_equal")
def less_equal_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.less_equal(inputs[0], inputs[1])]

register_schedule("less_equal", schedule_broadcast)

# greater
@register_compute("greater")
def greater_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.greater(inputs[0], inputs[1])]

register_schedule("greater", schedule_broadcast)

# greater equal
@register_compute("greater_equal")
def greater_equal_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.greater_equal(inputs[0], inputs[1])]

register_schedule("greater_equal", schedule_broadcast)

# maximum
@register_compute("maximum")
def maximum_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.maximum(inputs[0], inputs[1])]

register_schedule("maximum_compute", schedule_injective)

# minimum
@register_compute("minimum")
def minimum_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.minimum(inputs[0], inputs[1])]

register_schedule("minimum", schedule_injective)

# right shift
@register_compute("right_shift")
def right_shift_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.right_shift(inputs[0], inputs[1])]

register_schedule("right_shift", schedule_injective)

# left shift
@register_compute("left_shift")
def left_shift_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.left_shift(inputs[0], inputs[1])]

register_schedule("left_shift", schedule_injective)

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
register_pattern("clip", OpPattern.ELEMWISE)

# concatenate
@register_compute("concatenate")
def concatenate_compute(attrs, inputs, output_type, target):
    return [topi.concatenate(inputs, axis=attrs.axis)]

register_schedule("concatenate", schedule_injective)
register_pattern("concatenate", OpPattern.INJECTIVE)
