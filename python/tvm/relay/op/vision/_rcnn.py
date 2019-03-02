# pylint: disable=invalid-name, unused-argument
"""Faster R-CNN and Mask R-CNN operations."""
import topi
from topi.util import get_const_tuple
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
