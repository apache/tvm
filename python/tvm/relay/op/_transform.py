"""Backend compiler related feature registration"""
# pylint: disable=invalid-name
from __future__ import absolute_import
from . import op as _reg

schedule_injective = _reg.schedule_injective
schedule_broadcast = _reg.schedule_injective


_reg.register_schedule("squeeze", schedule_injective)
_reg.register_schedule("expand_dims", schedule_broadcast)
_reg.register_schedule("reshape", schedule_injective)
_reg.register_schedule("reshape_like", schedule_injective)
_reg.register_schedule("full", schedule_injective)
_reg.register_schedule("full_like", schedule_injective)
_reg.register_schedule("cast", schedule_broadcast)
_reg.register_schedule("strided_slice", schedule_injective)
_reg.register_schedule("slice_like", schedule_injective)
_reg.register_schedule("split", schedule_injective)
_reg.register_schedule("take", schedule_injective)
_reg.register_schedule("transpose", schedule_injective)
_reg.register_schedule("where", schedule_broadcast)
