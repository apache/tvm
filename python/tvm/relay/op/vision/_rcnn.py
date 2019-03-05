# pylint: disable=invalid-name, unused-argument
"""Faster R-CNN and Mask R-CNN operations."""
import topi
from topi.util import get_const_tuple, get_float_tuple, get_const_int
from .. import op as reg
from ..op import OpPattern


@reg.register_compute("vision.roi_align")
def compute_roi_align(attrs, inputs, _, target):
    """Compute definition of roi_align"""
    assert attrs.layout == "NCHW"
    return [topi.vision.rcnn.roi_align_nchw(
        inputs[0], inputs[1], pooled_size=get_const_tuple(attrs.pooled_size),
        spatial_scale=attrs.spatial_scale, sample_ratio=attrs.sample_ratio)]

@reg.register_schedule("vision.roi_align")
def schedule_roi_align(_, outs, target):
    """Schedule definition of roi_align"""
    with target:
        return topi.generic.vision.schedule_roi_align(outs)

reg.register_pattern("vision.roi_align", OpPattern.OUT_ELEMWISE_FUSABLE)

@reg.register_compute("vision.proposal")
def compute_proposal(attrs, inputs, _, target):
    """Compute definition of proposal"""
    scales = get_float_tuple(attrs.scales)
    ratios = get_float_tuple(attrs.ratios)
    feature_stride = attrs.feature_stride
    threshold = attrs.threshold
    rpn_pre_nms_top_n = attrs.rpn_pre_nms_top_n
    rpn_post_nms_top_n = attrs.rpn_post_nms_top_n
    rpn_min_size = attrs.rpn_min_size
    iou_loss = bool(get_const_int(attrs.iou_loss))
    with target:
        return [
            topi.vision.rcnn.proposal(inputs[0], inputs[1], inputs[2], scales, ratios,
                                      feature_stride, threshold, rpn_pre_nms_top_n,
                                      rpn_post_nms_top_n, rpn_min_size, iou_loss)
        ]

@reg.register_schedule("vision.proposal")
def schedule_proposal(_, outs, target):
    """Schedule definition of proposal"""
    with target:
        return topi.generic.schedule_proposal(outs)

reg.register_pattern("vision.proposal", OpPattern.OPAQUE)
