# pylint: disable=invalid-name, unused-argument
"""Definition of vision ops"""
from __future__ import absolute_import

import topi
from topi.util import get_const_int, get_const_float, get_float_tuple
from .. import op as reg
from ..op import OpPattern


@reg.register_schedule("vision.multibox_prior")
def schedule_multibox_prior(_, outs, target):
    """Schedule definition of multibox_prior"""
    with target:
        return topi.generic.schedule_multibox_prior(outs)


@reg.register_compute("vision.multibox_prior")
def compute_multibox_prior(attrs, inputs, _, target):
    """Compute definition of multibox_prior"""
    sizes = get_float_tuple(attrs.sizes)
    ratios = get_float_tuple(attrs.ratios)
    steps = get_float_tuple(attrs.steps)
    offsets = get_float_tuple(attrs.offsets)
    clip = bool(get_const_int(attrs.clip))
    return [
        topi.vision.ssd.multibox_prior(inputs[0], sizes, ratios, steps,
                                       offsets, clip)
    ]


reg.register_pattern("vision.multibox_prior", OpPattern.OPAQUE)


# multibox_transform_loc
@reg.register_schedule("vision.multibox_transform_loc")
def schedule_multibox_transform_loc(_, outs, target):
    """Schedule definition of multibox_detection"""
    with target:
        return topi.generic.schedule_multibox_transform_loc(outs)


@reg.register_compute("vision.multibox_transform_loc")
def compute_multibox_transform_loc(attrs, inputs, _, target):
    """Compute definition of multibox_detection"""
    clip = bool(get_const_int(attrs.clip))
    threshold = get_const_float(attrs.threshold)
    variances = get_float_tuple(attrs.variances)
    return topi.vision.ssd.multibox_transform_loc(
        inputs[0], inputs[1], inputs[2], clip, threshold, variances)


reg.register_pattern("vision.multibox_transform_loc", OpPattern.OPAQUE)
reg.register_pattern("vision.multibox_detection", OpPattern.OPAQUE)


# Get counts of valid boxes
@reg.register_schedule("vision.get_valid_counts")
def schedule_get_valid_counts(_, outs, target):
    """Schedule definition of get_valid_counts"""
    with target:
        return topi.generic.schedule_get_valid_counts(outs)


@reg.register_compute("vision.get_valid_counts")
def compute_get_valid_counts(attrs, inputs, _, target):
    """Compute definition of get_valid_counts"""
    score_threshold = get_const_float(attrs.score_threshold)
    return topi.vision.get_valid_counts(inputs[0], score_threshold)

reg.register_pattern("vision.get_valid_counts", OpPattern.OPAQUE)


# non-maximum suppression
@reg.register_schedule("vision.non_max_suppression")
def schedule_nms(_, outs, target):
    """Schedule definition of nms"""
    with target:
        return topi.generic.schedule_nms(outs)


@reg.register_compute("vision.non_max_suppression")
def compute_nms(attrs, inputs, _, target):
    """Compute definition of nms"""
    return_indices = bool(get_const_int(attrs.return_indices))
    iou_threshold = get_const_float(attrs.iou_threshold)
    force_suppress = bool(get_const_int(attrs.force_suppress))
    topk = get_const_int(attrs.topk)
    id_index = get_const_int(attrs.id_index)
    invalid_to_bottom = bool(get_const_int(attrs.invalid_to_bottom))
    return [
        topi.vision.non_max_suppression(inputs[0], inputs[1], return_indices,
                                        iou_threshold, force_suppress, topk,
                                        id_index, invalid_to_bottom)
    ]


reg.register_pattern("vision.non_max_suppression", OpPattern.OPAQUE)
