# pylint: disable=invalid-name, unused-argument
"""Definition of nn ops"""
from __future__ import absolute_import

import tvm
import topi
from . import registry as reg
from .registry import OpPattern

@reg.register_compute("yolo_reorg")
def compute_reorg(attrs, inputs, _):
    """Compute definition of reorg"""
    return topi.vision.reorg(inputs[0], attrs.get_int("stride"))

@reg.register_schedule("yolo_reorg")
def schedule_reorg(attrs, outs, target):
    """Schedule definition of reorg"""
    with tvm.target.create(target):
        return topi.generic.schedule_injective(outs)

reg.register_pattern("yolo_reorg", OpPattern.INJECTIVE)

# multibox_prior
@reg.register_schedule("multibox_prior")
def schedule_multibox_prior(_, outs, target):
    """Schedule definition of multibox_prior"""
    with tvm.target.create(target):
        return topi.generic.schedule_multibox_prior(outs)

@reg.register_compute("multibox_prior")
def compute_multibox_prior(attrs, inputs, _):
    """Compute definition of multibox_prior"""
    sizes = attrs.get_float_tuple('sizes')
    ratios = attrs.get_float_tuple('ratios')
    steps = attrs.get_float_tuple('steps')
    offsets = attrs.get_float_tuple('offsets')
    clip = attrs.get_bool('clip')

    return topi.vision.ssd.multibox_prior(inputs[0], sizes, ratios,
                                          steps, offsets, clip)

reg.register_pattern("multibox_prior", OpPattern.OPAQUE)

# multibox_transform_loc
@reg.register_schedule("multibox_transform_loc")
def schedule_multibox_transform_loc(_, outs, target):
    """Schedule definition of multibox_detection"""
    with tvm.target.create(target):
        return topi.generic.schedule_multibox_transform_loc(outs)

@reg.register_compute("multibox_transform_loc")
def compute_multibox_transform_loc(attrs, inputs, _):
    """Compute definition of multibox_detection"""
    clip = attrs.get_bool('clip')
    threshold = attrs.get_float('threshold')
    variance = attrs.get_float_tuple('variances')

    return topi.vision.ssd.multibox_transform_loc(inputs[0], inputs[1], inputs[2],
                                                  clip, threshold, variance)

reg.register_pattern("multibox_detection", OpPattern.OPAQUE)

# non-maximum suppression
@reg.register_schedule("nms")
def schedule_nms(_, outs, target):
    """Schedule definition of nms"""
    with tvm.target.create(target):
        return topi.generic.schedule_nms(outs)

@reg.register_compute("nms")
def compute_nms(attrs, inputs, _):
    """Compute definition of nms"""
    nms_threshold = attrs.get_float('nms_threshold')
    force_suppress = attrs.get_bool('force_suppress')
    nms_topk = attrs.get_int('nms_topk')

    return topi.vision.nms(inputs[0], inputs[1], nms_threshold,
                           force_suppress, nms_topk)

reg.register_pattern("nms", OpPattern.OPAQUE)
