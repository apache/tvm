# pylint: disable=invalid-name, unused-argument
"""Tensor transformation ops"""
from __future__ import absolute_import

from .tensor import _fschedule_broadcast, _fschedule_injective
from . import registry as reg
from .registry import OpPattern

# expand_dims
reg.register_pattern("expand_dims", OpPattern.BROADCAST)
reg.register_schedule("expand_dims", _fschedule_broadcast)

# transpose
reg.register_pattern("transpose", OpPattern.INJECTIVE)
reg.register_schedule("transpose", _fschedule_injective)

# reshape
reg.register_pattern("reshape", OpPattern.INJECTIVE)
reg.register_schedule("reshape", _fschedule_injective)

# squeeze
reg.register_pattern("squeeze", OpPattern.INJECTIVE)
reg.register_schedule("squeeze", _fschedule_injective)

# concatenate
reg.register_pattern("concatenate", OpPattern.INJECTIVE)
reg.register_schedule("concatenate", _fschedule_injective)

# split
reg.register_pattern("split", OpPattern.INJECTIVE)
reg.register_schedule("split", _fschedule_injective)
