
# pylint: disable=invalid-name, unused-argument
"""Definition of nn ops"""
from __future__ import absolute_import

import topi
import tvm
from . import registry as reg
from .registry import OpPattern

@reg.register_compute("yolo2_reorg")
def compute_reorg(attrs, inputs, _):
    """Compute definition of reorg"""
    return topi.vision.reorg(inputs[0], attrs.get_int("stride"))

@reg.register_schedule("yolo2_reorg")
def schedule_reorg(attrs, outs, target):
    """Schedule definition of reorg"""
    with tvm.target.create(target):
        return topi.generic.schedule_injective(outs)

reg.register_pattern("yolo2_reorg", OpPattern.INJECTIVE)

@reg.register_compute("yolo2_region")
def compute_region(attrs, inputs, _):
    """Compute definition of region"""
    n = attrs.get_int("n")
    classes = attrs.get_int("classes")
    coords = attrs.get_int("coords")
    background = attrs.get_int("background")
    softmax = attrs.get_int("softmax")
    return topi.vision.yolo2.region(inputs[0], n, classes, coords, background, softmax)

@reg.register_schedule("yolo2_region")
def schedule_region(attrs, outs, target):
    """Schedule definition of region"""
    with tvm.target.create(target):
        return topi.generic.vision.schedule_region(outs)

reg.register_pattern("yolo2_region", OpPattern.OPAQUE)
