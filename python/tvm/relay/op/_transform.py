#pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import
from . import op as _reg
from .op import schedule_injective, OpPattern

# strided_slice
_reg.register_schedule("strided_slice", schedule_injective)

# slice_like
_reg.register_schedule("slice_like", schedule_injective)
_reg.register_pattern("slice_like", OpPattern.INJECTIVE)
