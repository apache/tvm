"""Faster R-CNN and Mask R-CNN operations."""
from . import _make


def roi_align(data, rois, pooled_size, spatial_scale, sample_ratio=-1, layout='NCHW'):
    """ROI align operator.

    Parameters
    ----------
    data : relay.Expr
        4-D tensor with shape [batch, channel, height, width]

    rois : relay.Expr
        2-D tensor with shape [num_roi, 5]. The last dimension should be in format of
        [batch_index, w_start, h_start, w_end, h_end]

    pooled_size : list/tuple of two ints
        output size

    spatial_scale : float
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal
        of total stride in convolutional layers, which should be in range (0.0, 1.0]

    sample_ratio : int
        Optional sampling ratio of ROI align, using adaptive size by default.

    Returns
    -------
    output : relay.Expr
        4-D tensor with shape [num_roi, channel, pooled_size, pooled_size]
    """
    return _make.roi_align(data, rois, pooled_size, spatial_scale, sample_ratio, layout)
