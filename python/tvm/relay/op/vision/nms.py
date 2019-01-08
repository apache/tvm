"""Non-maximum suppression operations."""
from __future__ import absolute_import as _abs
from . import _make

def get_valid_counts(data,
                     score_threshold):
    """Get valid count of bounding boxes given a score threshold.
    Also moves valid boxes to the top of input data.

    Parameters
    ----------
    data : relay.Expr
        Input data. 3-D tensor with shape [batch_size, num_anchors, 6].

    score_threshold : optional, float
        Lower limit of score for valid bounding boxes.

    Returns
    -------
    out_tensor : relay.Expr
        Rearranged data tensor.

    valid_count : relay.Expr
        1-D tensor for valid number of boxes.
    """
    return _make.get_valid_counts(data, score_threshold)


def nms(data,
        valid_count,
        iou_threshold=0.5,
        force_suppress=False,
        topk=-1,
        id_index=0,
        do_rearrange=False):
    """Non-maximum suppression operator for object detection.

    Parameters
    ----------
    data : relay.Expr
        3-D tensor with shape [batch_size, num_anchors, 6].
        The last dimension should be in format of
        [class_id, score, box_left, box_top, box_right, box_bottom].

    valid_count : relay.Expr
        1-D tensor for valid number of boxes.

    iou_threshold : float, optional
        Non-maximum suppression threshold.

    force_suppress : bool, optional
        Suppress all detections regardless of class_id.

    topk : int, optional
        Keep maximum top k detections before nms, -1 for no limit.

    id_index : optional, int
        index of the class categories, -1 to disable.

    do_rearrange : optional, boolean
        Whether to move all valid bounding boxes to the top.

    Returns
    -------
    out : relay.Expr
        3-D tensor with shape [batch_size, num_anchors, 6].
    """
    return _make.nms(data, valid_count, iou_threshold,
                     force_suppress, topk, id_index, do_rearrange)
