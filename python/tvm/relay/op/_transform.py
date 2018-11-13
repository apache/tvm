#pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
from __future__ import absolute_import
from . import op as _reg
from .op import schedule_injective

# strided_slice
_reg.register_schedule("strided_slice", schedule_injective)
