"""Backend compiler related feature registration"""
# pylint: disable=invalid-name,unused-argument
from __future__ import absolute_import
from . import op as _reg
from ._reduce import _schedule_reduce
from .op import schedule_injective, OpPattern

schedule_injective = _reg.schedule_injective
schedule_broadcast = _reg.schedule_injective


_reg.register_schedule("collapse_sum_like", _schedule_reduce)
_reg.register_schedule("broadcast_to", schedule_broadcast)
_reg.register_schedule("broadcast_to_like", schedule_broadcast)
_reg.register_schedule("expand_dims", schedule_broadcast)
_reg.register_schedule("squeeze", schedule_injective)
_reg.register_schedule("reshape", schedule_injective)
_reg.register_schedule("reshape_like", schedule_injective)
_reg.register_schedule("full", schedule_injective)
_reg.register_schedule("full_like", schedule_injective)
_reg.register_schedule("arange", schedule_injective)
_reg.register_schedule("reverse", schedule_injective)
_reg.register_schedule("repeat", schedule_broadcast)
_reg.register_schedule("tile", schedule_broadcast)
_reg.register_schedule("cast", schedule_injective)
_reg.register_schedule("strided_slice", schedule_injective)
_reg.register_schedule("slice_like", schedule_injective)
_reg.register_schedule("split", schedule_injective)
_reg.register_schedule("take", schedule_injective)
_reg.register_schedule("transpose", schedule_injective)
_reg.register_schedule("where", schedule_broadcast)
_reg.register_schedule("stack", schedule_injective)
_reg.register_schedule("concatenate", schedule_injective)
_reg.register_schedule("_contrib_reverse_reshape", schedule_injective)
_reg.register_schedule("gather_nd", schedule_injective)

# layout_transform
_reg.register_schedule("layout_transform", schedule_injective)
_reg.register_pattern("layout_transform", OpPattern.INJECTIVE)
