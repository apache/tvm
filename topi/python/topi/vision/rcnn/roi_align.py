# pylint: disable=invalid-name
"""Roi align operator"""
import tvm
from ...util import get_const_tuple
from ...cpp.image import bilinear_sample_nchw


@tvm.target.generic_func
def roi_align_nchw(data, rois, pooled_size, spatial_scale, sample_ratio=-1):
    """ROI align operator in NCHW layout.

    Parameters
    ----------
    data : tvm.Tensor
        4-D with shape [batch, channel, height, width]

    rois : tvm.Tensor
        2-D with shape [num_roi, 5]. The last dimension should be in format of
        [batch_index, w_start, h_start, w_end, h_end]

    pooled_size : int or list/tuple of two ints
        output size, or [out_height, out_width]

    spatial_scale : float
        Ratio of input feature map height (or w) to raw image height (or w). Equals the reciprocal
        of total stride in convolutional layers, which should be in range (0.0, 1.0]

    sample_ratio : int
        Optional sampling ratio of ROI align, using adaptive size by default.

    Returns
    -------
    output : tvm.Tensor
        4-D with shape [num_roi, channel, pooled_size, pooled_size]
    """
    dtype = rois.dtype
    _, channel, height, width = get_const_tuple(data.shape)
    num_roi, _ = get_const_tuple(rois.shape)

    if isinstance(pooled_size, int):
        pooled_size_h = pooled_size_w = pooled_size
    else:
        pooled_size_h, pooled_size_w = pooled_size

    def _bilinear(i, c, y, x):
        outside = tvm.any(y < -1.0, x < -1.0, y > height, x > width)
        y = tvm.max(y, 0.0)
        x = tvm.max(x, 0.0)
        val = bilinear_sample_nchw(data, (i, c, y, x), height - 1, width - 1)
        return tvm.if_then_else(outside, 0.0, val)

    def _sample(i, c, ph, pw):
        roi = rois[i]
        batch_index = roi[0].astype('int32')
        roi_start_w, roi_start_h, roi_end_w, roi_end_h = roi[1], roi[2], roi[3], roi[4]
        roi_start_h *= spatial_scale
        roi_end_h *= spatial_scale
        roi_start_w *= spatial_scale
        roi_end_w *= spatial_scale

        # force malformed ROIs to be 1x1
        roi_h = tvm.max(roi_end_h - roi_start_h, tvm.const(1.0, dtype))
        roi_w = tvm.max(roi_end_w - roi_start_w, tvm.const(1.0, dtype))

        bin_h = roi_h / pooled_size_h
        bin_w = roi_w / pooled_size_w

        if sample_ratio > 0:
            roi_bin_grid_h = roi_bin_grid_w = tvm.const(sample_ratio, 'int32')
        else:
            roi_bin_grid_h = tvm.ceil(roi_h / pooled_size).astype('int32')
            roi_bin_grid_w = tvm.ceil(roi_w / pooled_size).astype('int32')

        count = roi_bin_grid_h * roi_bin_grid_w
        rh = tvm.reduce_axis((0, roi_bin_grid_h))
        rw = tvm.reduce_axis((0, roi_bin_grid_w))
        roi_start_h += ph * bin_h
        roi_start_w += pw * bin_w
        return tvm.sum(_bilinear(batch_index, c,
                                 roi_start_h + (rh + 0.5) * bin_h / roi_bin_grid_h,
                                 roi_start_w + (rw + 0.5) * bin_w / roi_bin_grid_w) / count,
                       axis=[rh, rw])

    return tvm.compute((num_roi, channel, pooled_size_h, pooled_size_w), _sample,
                       tag='pool,roi_align_nchw')
