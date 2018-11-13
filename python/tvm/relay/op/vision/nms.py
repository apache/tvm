"""Non-maximum suppression operations."""
from __future__ import absolute_import as _abs
from . import _make

def nms(data,
        valid_count,
        overlap_threshold=0.5,
        force_suppress=False,
        topk=-1):
    """Non-maximum suppression operator for object detection.

    Parameters
    ----------
    data : relay.Expr
        3-D tensor with shape [batch_size, num_anchors, 6].
        The last dimension should be in format of
        [class_id, score, box_left, box_top, box_right, box_bottom].

    valid_count : relay.Expr
        1-D tensor for valid number of boxes.

    overlap_threshold : float, optional
        Non-maximum suppression threshold.

    force_suppress : bool, optional
        Suppress all detections regardless of class_id.

    topk : int, optional
        Keep maximum top k detections before nms, -1 for no limit.

    Returns
    -------
    out : relay.Expr
        3-D tensor with shape [batch_size, num_anchors, 6].
    """
    return _make.nms(data, valid_count, overlap_threshold, force_suppress, topk)
