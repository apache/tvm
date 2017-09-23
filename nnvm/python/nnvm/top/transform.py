# pylint: disable=invalid-name, unused-argument
"""Tensor transformation ops"""
from __future__ import absolute_import

import tvm
import topi
from .tensor import _fschedule_broadcast
from ..compiler import registry as reg
from ..compiler import OpPattern

# Need add reshape, transpose
@reg.register_compute("expand_dims")
def compute_expand_dims(attrs, inputs, out_info):
    """Compute definition of expand_dims"""
    return topi.expand_dims(
        inputs[0], attrs.get_int("axis"),
        num_newaxis=attrs.get_int("num_newaxis"))
reg.register_pattern("expand_dims", OpPattern.BROADCAST)
reg.register_schedule("expand_dims", _fschedule_broadcast)


def _flatten_index(indices, shape):
    """flatten the index to 1D"""
    idx = 0
    for i, value in enumerate(shape):
        if i != 0:
            idx *= value
        idx = idx + indices[i]
    return idx

# reshape
@reg.register_compute("reshape")
def compute_reshape(attrs, inputs, out_info):
    """Compute definition of softmax"""
    # TODO(sxj) add support for general reshape
    assert len(inputs[0].shape) == 1, "Only support 1d input for now"
    oshape = out_info[0].shape
    x = inputs[0]
    return tvm.compute(oshape, lambda *i: x(_flatten_index(i, oshape)))
reg.register_pattern("reshape", OpPattern.INJECTIVE)
reg.register_schedule("reshape", _fschedule_broadcast)
