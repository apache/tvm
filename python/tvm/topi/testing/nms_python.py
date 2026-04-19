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
"""Numpy reference implementation for classic non_max_suppression."""
import numpy as np


def _iou(box_a, box_b, coord_start):
    """Compute IoU between two boxes."""
    a = box_a[coord_start : coord_start + 4]
    b = box_b[coord_start : coord_start + 4]

    a_l, a_t, a_r, a_b = min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])
    b_l, b_t, b_r, b_b = min(b[0], b[2]), min(b[1], b[3]), max(b[0], b[2]), max(b[1], b[3])

    w = max(0.0, min(a_r, b_r) - max(a_l, b_l))
    h = max(0.0, min(a_b, b_b) - max(a_t, b_t))
    area = w * h
    u = (a_r - a_l) * (a_b - a_t) + (b_r - b_l) * (b_b - b_t) - area
    return 0.0 if u <= 0 else area / u


def non_max_suppression_python(
    data,
    valid_count,
    indices,
    max_output_size=-1,
    iou_threshold=0.5,
    force_suppress=False,
    top_k=-1,
    coord_start=2,
    score_index=1,
    id_index=0,
    return_indices=True,
    invalid_to_bottom=False,
    soft_nms_sigma=0.0,
    score_threshold=0.0,
):
    """Numpy reference for classic non_max_suppression.

    Parameters
    ----------
    data : numpy.ndarray
        3-D array, shape [batch_size, num_anchors, elem_length].

    valid_count : numpy.ndarray
        1-D array, shape [batch_size].

    indices : numpy.ndarray
        2-D array, shape [batch_size, num_anchors].

    Returns
    -------
    If return_indices is True: (box_indices, valid_box_count)
    Otherwise: modified data tensor
    """
    batch_size, num_anchors, _ = data.shape
    out_data = np.full_like(data, -1.0)
    out_box_indices = np.full((batch_size, num_anchors), -1, dtype="int32")
    compacted = np.full((batch_size, num_anchors), -1, dtype="int32")
    valid_box_count = np.zeros((batch_size, 1), dtype="int32")

    is_soft_nms = soft_nms_sigma > 0.0
    thresh = score_threshold if is_soft_nms else 0.0

    for i in range(batch_size):
        nkeep = int(valid_count[i])
        if 0 < top_k < nkeep:
            nkeep = top_k

        # Sort by score descending
        scores = data[i, :nkeep, score_index].copy()
        sorted_idx = np.argsort(-scores)

        # Copy sorted boxes
        for j in range(nkeep):
            src = sorted_idx[j]
            out_data[i, j, :] = data[i, src, :]
            out_box_indices[i, j] = src

        # Greedy NMS
        num_valid = 0
        for j in range(nkeep):
            if out_data[i, j, score_index] <= thresh:
                if not is_soft_nms:
                    out_data[i, j, :] = -1.0
                    out_box_indices[i, j] = -1
                continue
            if 0 < max_output_size <= num_valid:
                out_data[i, j, :] = -1.0
                out_box_indices[i, j] = -1
                continue

            num_valid += 1

            # Suppress overlapping boxes
            for k in range(j + 1, nkeep):
                if out_data[i, k, score_index] <= thresh:
                    continue

                do_suppress = False
                if force_suppress:
                    do_suppress = True
                elif id_index >= 0:
                    do_suppress = out_data[i, j, id_index] == out_data[i, k, id_index]
                else:
                    do_suppress = True

                if do_suppress:
                    iou = _iou(out_data[i, j], out_data[i, k], coord_start)
                    if iou >= iou_threshold:
                        if is_soft_nms:
                            decay = np.exp(-(iou ** 2) / soft_nms_sigma)
                            out_data[i, k, score_index] *= decay
                        else:
                            out_data[i, k, score_index] = -1.0
                            out_box_indices[i, k] = -1

        if return_indices:
            # Compact valid indices to top and remap to original
            cnt = 0
            for j in range(num_anchors):
                if out_box_indices[i, j] >= 0:
                    orig_idx = out_box_indices[i, j]
                    compacted[i, cnt] = int(indices[i, orig_idx])
                    cnt += 1
            valid_box_count[i, 0] = cnt

    if return_indices:
        if is_soft_nms:
            return [out_data, compacted, valid_box_count]
        return [compacted, valid_box_count]

    if invalid_to_bottom:
        # Rearrange valid boxes to top
        result = np.full_like(data, -1.0)
        for i in range(batch_size):
            cnt = 0
            for j in range(num_anchors):
                if out_data[i, j, score_index] >= 0:
                    result[i, cnt, :] = out_data[i, j, :]
                    cnt += 1
        return result

    return out_data
