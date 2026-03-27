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
# pylint: disable=invalid-name
"""Numpy reference for multibox_transform_loc."""

import numpy as np


def _softmax(x, axis):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp = np.exp(x - x_max)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def multibox_transform_loc_python(
    cls_pred,
    loc_pred,
    anchor,
    variances,
    clip=False,
    threshold=0.0,
    keep_background=True,
):
    """Reference implementation aligned with ``topi.vision.multibox_transform_loc``."""
    B, C, N = cls_pred.shape
    loc = loc_pred.reshape(B, N, 4)
    scores = _softmax(cls_pred.astype("float64"), axis=1).astype(np.float32)
    if threshold > 0.0:
        scores = np.where(scores >= threshold, scores, 0.0).astype(np.float32)
    if not keep_background:
        scores = scores.copy()
        scores[:, 0, :] = 0.0

    vx, vy, vw, vh = variances
    boxes = np.zeros((B, N, 4), dtype=np.float32)
    for b in range(B):
        for a in range(N):
            l, t, r, br = anchor[0, a, :]
            ay = (t + br) * 0.5
            ax = (l + r) * 0.5
            ah = br - t
            aw = r - l
            ex, ey, ew, eh = loc[b, a, :]
            ycenter = ey * vy * ah + ay
            xcenter = ex * vx * aw + ax
            half_h = 0.5 * np.exp(eh * vh) * ah
            half_w = 0.5 * np.exp(ew * vw) * aw
            ymin = ycenter - half_h
            xmin = xcenter - half_w
            ymax = ycenter + half_h
            xmax = xcenter + half_w
            if clip:
                ymin = float(np.clip(ymin, 0.0, 1.0))
                xmin = float(np.clip(xmin, 0.0, 1.0))
                ymax = float(np.clip(ymax, 0.0, 1.0))
                xmax = float(np.clip(xmax, 0.0, 1.0))
            boxes[b, a, :] = (ymin, xmin, ymax, xmax)
    return boxes, scores
