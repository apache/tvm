# pylint: disable=invalid-name, unused-argument
"""Tensor transformation ops"""
from __future__ import absolute_import

import topi
from .tensor import _fschedule_broadcast, _fschedule_injective
from . import registry as reg
from .registry import OpPattern

# expand_dims
@reg.register_compute("expand_dims")
def compute_expand_dims(attrs, inputs, out_info):
    """Compute definition of expand_dims"""
    return topi.expand_dims(
        inputs[0], attrs.get_int("axis"),
        num_newaxis=attrs.get_int("num_newaxis"))
reg.register_pattern("expand_dims", OpPattern.BROADCAST)
reg.register_schedule("expand_dims", _fschedule_broadcast)

# transpose
@reg.register_compute("transpose")
def compute_transpose(attrs, inputs, out_info):
    """Compute definition of transpose"""
    axes = attrs.get_int_tuple("axes")
    axes = tuple(axes) if axes else None
    return topi.transpose(inputs[0], axes)
reg.register_pattern("transpose", OpPattern.INJECTIVE)
reg.register_schedule("transpose", _fschedule_injective)

# reshape
@reg.register_compute("reshape")
def compute_reshape(attrs, inputs, out_info):
    """Compute definition of reshape"""
    oshape = out_info[0].shape
    return topi.reshape(inputs[0], oshape)
reg.register_pattern("reshape", OpPattern.INJECTIVE)
reg.register_schedule("reshape", _fschedule_injective)

# reshape
@reg.register_compute("squeeze")
def compute_squeeze(attrs, inputs, out_info):
    """Compute definition of reshape"""
    axis = attrs.get_int_tuple("axis")
    axis = tuple(axis) if axis else None
    return topi.squeeze(inputs[0], axis)
reg.register_pattern("squeeze", OpPattern.INJECTIVE)
reg.register_schedule("squeeze", _fschedule_injective)

# concatenate
@reg.register_compute("concatenate")
def compute_concatenate(attrs, inputs, out_info):
    """Compute definition of concatenate"""
    axis = attrs.get_int("axis")
    return topi.concatenate([x for x in inputs], axis=axis)

reg.register_pattern("concatenate", OpPattern.INJECTIVE)
reg.register_schedule("concatenate", _fschedule_injective)

# split
@reg.register_compute("split")
def compute_split(attrs, inputs, out_info):
    """Compute definition of split"""
    x = attrs["indices_or_sections"]
    if x.startswith("(") or x.startswith("["):
        indices_or_sections = attrs.get_int_tuple("indices_or_sections")
    else:
        indices_or_sections = attrs.get_int("indices_or_sections")
    return topi.split(inputs[0], indices_or_sections, axis=attrs.get_int("axis"))


reg.register_pattern("split", OpPattern.INJECTIVE)
reg.register_schedule("split", _fschedule_injective)
