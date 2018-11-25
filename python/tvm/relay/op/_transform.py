#pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import
import topi
import topi.cuda
from tvm import container
from . import op as _reg
from .op import (schedule_injective, register_compute, register_schedule,
                 register_pattern, OpPattern)

schedule_broadcast = schedule_injective

# squeeze
@register_compute("squeeze")
def squeeze_compiler(attrs, inputs, output_type, target):
    """Compiler for squeeze dims."""
    assert len(inputs) == 1

    if attrs.axis is None:
        axis = None
    elif isinstance(attrs.axis, container.Array):
        axis = tuple(attrs.axis)
    else:
        axis = int(attrs.axis)

    return [topi.squeeze(inputs[0], axis)]

register_pattern("squeeze", OpPattern.INJECTIVE)
register_schedule("squeeze", schedule_injective)

# expand_dims
@register_compute("expand_dims")
def expand_dims_compiler(attrs, inputs, output_type, target):
    """Compiler for expand_dims."""
    assert len(inputs) == 1

    new_axis = int(attrs.num_newaxis)
    assert new_axis >= 0

    # axis should be in range [-data.ndim - 1, data.ndim]
    axis = int(attrs.axis)
    assert axis >= -len(inputs[0].shape) - 1
    assert axis <= len(inputs[0].shape)

    return [topi.expand_dims(inputs[0], axis, new_axis)]

_reg.register_schedule("expand_dims", schedule_broadcast)
_reg.register_pattern("expand_dims", OpPattern.BROADCAST)

# strided_slice
_reg.register_schedule("strided_slice", schedule_injective)

# slice_like
_reg.register_schedule("slice_like", schedule_injective)
_reg.register_pattern("slice_like", OpPattern.INJECTIVE)
