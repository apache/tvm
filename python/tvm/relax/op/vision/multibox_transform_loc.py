# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Multibox location transform for object detection."""

from . import _ffi_api


def multibox_transform_loc(
    cls_pred,
    loc_pred,
    anchor,
    clip=False,
    threshold=0.0,
    variances=(1.0, 1.0, 1.0, 1.0),
    keep_background=True,
):
    """SSD / TFLite-style decode: priors + offsets → boxes; logits → softmax scores.

    Box decode follows TFLite ``DecodeCenterSizeBoxes``; expected tensor layout matches
    ``tflite_frontend.convert_detection_postprocess`` (loc reorder yxhw→xywh, anchor ltrb).

    Parameters
    ----------
    cls_pred : relax.Expr
        ``[B, C, N]`` class logits (pre-softmax).
    loc_pred : relax.Expr
        ``[B, 4*N]`` per-anchor encodings as ``(x,y,w,h)`` after reorder (see above).
    anchor : relax.Expr
        ``[1, N, 4]`` priors: ``(left, top, right, bottom)``.
    clip : bool
        If True, clip ``ymin,xmin,ymax,xmax`` to ``[0, 1]``.
    threshold : float
        After softmax, multiply scores by mask ``(score >= threshold)``.
    variances : tuple of 4 floats
        ``(x,y,w,h)`` = TFLite ``1/x_scale, 1/y_scale, 1/w_scale, 1/h_scale``.
    keep_background : bool
        If False, set output scores at class index 0 to zero.

    Returns
    -------
    result : relax.Expr
        Tuple ``(boxes, scores)``: ``boxes`` is ``[B, N, 4]`` as ``(ymin,xmin,ymax,xmax)``;
        ``scores`` is ``[B, C, N]`` softmax, post-processed like the implementation.

    Notes
    -----
    **Shape/dtype (checked in ``FInferStructInfo`` when static):**

    - ``cls_pred``: 3-D; ``loc_pred``: 2-D; ``anchor``: 3-D.
    - ``cls_pred``, ``loc_pred``, ``anchor`` dtypes must match.
    - ``N = cls_pred.shape[2]``; ``loc_pred.shape[1] == 4*N``; ``anchor.shape == [1,N,4]``.
    - ``loc_pred.shape[1]`` must be divisible by 4.
    - ``cls_pred.shape[0]`` must equal ``loc_pred.shape[0]`` (batch).
    """
    return _ffi_api.multibox_transform_loc(
        cls_pred,
        loc_pred,
        anchor,
        clip,
        threshold,
        variances,
        keep_background,
    )
