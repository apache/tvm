#pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import
import tvm
import topi
import topi.cuda
from . import register_schedule, register_compute

def schedule_injective(outputs, target):
    """Generic schedule for binary broadcast."""
    with tvm.target.create(target):
        return topi.generic.schedule_injective(outputs)

schedule_broadcast = schedule_injective
schedule_elemwise = schedule_injective

# log
def log_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.log(inputs[0])]

register_compute("log", log_compute)
register_schedule("log", schedule_broadcast)

# exp
def exp_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.exp(inputs[0])]

register_compute("exp", exp_compute)
register_schedule("exp", schedule_broadcast)

# sqrt
def sqrt_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.sqrt(inputs[0])]

register_compute("sqrt", sqrt_compute)
register_schedule("sqrt", schedule_broadcast)

# sigmoid
def sigmoid_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.sigmoid(inputs[0])]

register_compute("sigmoid", sigmoid_compute)
register_schedule("sigmoid", schedule_broadcast)

# floor
def floor_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.floor(inputs[0])]

register_compute("floor", floor_compute)
register_schedule("floor", schedule_broadcast)

# ceil
def ceil_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.ceil(inputs[0])]

register_compute("ceil", ceil_compute)
register_schedule("ceil", schedule_broadcast)

# trunc
def trunc_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.trunc(inputs[0])]

register_compute("trunc", trunc_compute)
register_schedule("trunc", schedule_broadcast)

# round
def round_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.round(inputs[0])]

register_compute("round", round_compute)
register_schedule("round", schedule_broadcast)

# abs
def abs_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.abs(inputs[0])]

register_compute("abs", abs_compute)
register_schedule("abs", schedule_broadcast)

# tanh
def tanh_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.tanh(inputs[0])]

register_compute("tanh", tanh_compute)
register_schedule("tanh", schedule_broadcast)

# negative
def negative_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.negative(inputs[0])]

register_compute("negative", negative_compute)
register_schedule("negative", schedule_broadcast)

# add
def add_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.add(inputs[0], inputs[1])]

register_compute("add", add_compute)
register_schedule("add", schedule_injective)

# subtract
def subtract_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.subtract(inputs[0], inputs[1])]

register_compute("subtract", subtract_compute)
register_schedule("subtract", schedule_broadcast)

# multiply
def multiply_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.multiply(inputs[0], inputs[1])]

register_compute("multiply", multiply_compute)
register_schedule("multiply", schedule_broadcast)

# divide
def divide_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.divide(inputs[0], inputs[1])]

register_compute("divide", divide_compute)
register_schedule("divide", schedule_broadcast)

# pow
def pow_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.power(inputs[0], inputs[1])]

register_compute("pow", pow_compute)
register_schedule("pow", schedule_injective)

# mod
def mod_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.mod(inputs[0], inputs[1])]

register_compute("mod", mod_compute)
register_schedule("mod", schedule_broadcast)

# equal
def equal_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.equal(inputs[0], inputs[1])]

register_compute("equal", equal_compute)
register_schedule("equal", schedule_broadcast)

# not_equal
def not_equal_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.not_equal(inputs[0], inputs[1])]

register_compute("not_equal", not_equal_compute)
register_schedule("not_equal", schedule_broadcast)

# less
def less_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.less(inputs[0], inputs[1])]

register_compute("less", less_compute)
register_schedule("less", schedule_broadcast)

# less equal
def less_equal_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.less_equal(inputs[0], inputs[1])]

register_compute("less_equal", less_equal_compute)
register_schedule("less_equal", schedule_broadcast)

# greater
def greater_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.greater(inputs[0], inputs[1])]

register_compute("greater", greater_compute)
register_schedule("greater", schedule_broadcast)

# greater equal
def greater_equal_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.greater_equal(inputs[0], inputs[1])]

register_compute("greater_equal", greater_equal_compute)
register_schedule("greater_equal", schedule_broadcast)

# maximum
def maximum_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.maximum(inputs[0], inputs[1])]

register_compute("maximum_compute", maximum_compute)
register_schedule("maximum_compute", schedule_injective)

# minimum
def minimum_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.minimum(inputs[0], inputs[1])]

register_compute("minimum", minimum_compute)
register_schedule("minimum", schedule_injective)

# right shift
def right_shift_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.right_shift(inputs[0], inputs[1])]

register_compute("right_shift", right_shift_compute)
register_schedule("right_shift", schedule_injective)

# lift shift
def left_shift_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.left_shift(inputs[0], inputs[1])]

register_compute("left_shift", left_shift_compute)
register_schedule("left_shift", schedule_injective)

# zeros
def zeros_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.full(inputs[0], inputs[1], 0.0)]

register_compute("zeros_compute", zeros_compute)
register_schedule("zeros_compute", schedule_injective)

# zeros_like
def zeros_like_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.full_like(inputs[0], 0.0)]

register_compute("zeros_like", zeros_like_compute)
register_schedule("zeros_like", schedule_injective)

# ones
def ones_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.full(inputs[0], inputs[1], 1.0)]

register_compute("ones", ones_compute)
register_schedule("ones", schedule_injective)

# ones_like
def ones_like(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.full_like(inputs[0], 1.0)]

register_compute("ones_like", ones_like)
register_schedule("ones_like", schedule_injective)

# clip
def clip_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.clip(inputs[0], inputs[1], inputs[2])]


register_compute("clip", clip_compute)
register_schedule("clip", schedule_injective)

# concatenate
def concatenate_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 1
    return [topi.concatenate(*inputs[0], attrs.axis)]

register_compute("concatenate", concatenate_compute)
register_schedule("concatenate", schedule_injective)

# # copy
# TODO(@jroesch): How to implement copy.
# def copy_compute(attrs, inputs, output_type, target):
#     assert len(inputs) == 1
#     return [topi.copy(inputs[0])]

# register_compute("copy", copy_compute)
# register_schedule("copy", schedule_injective)
