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
"""Numpy reference implementation for get_valid_counts."""
import numpy as np


def get_valid_counts_python(data, score_threshold=0, id_index=0, score_index=1):
    """Numpy reference for get_valid_counts.

    Parameters
    ----------
    data : numpy.ndarray
        3-D array with shape [batch_size, num_anchors, elem_length].

    score_threshold : float
        Lower limit of score for valid bounding boxes.

    id_index : int
        Index of the class categories, -1 to disable.

    score_index : int
        Index of the scores/confidence of boxes.

    Returns
    -------
    valid_count : numpy.ndarray
        1-D array, shape [batch_size].

    out_tensor : numpy.ndarray
        Rearranged data, shape [batch_size, num_anchors, elem_length].

    out_indices : numpy.ndarray
        Indices mapping, shape [batch_size, num_anchors].
    """
    batch_size, num_anchors, box_data_length = data.shape
    valid_count = np.zeros(batch_size, dtype="int32")
    out_tensor = np.full_like(data, -1.0)
    out_indices = np.full((batch_size, num_anchors), -1, dtype="int32")

    for i in range(batch_size):
        cnt = 0
        for j in range(num_anchors):
            score = data[i, j, score_index]
            if id_index < 0:
                is_valid = score > score_threshold
            else:
                is_valid = score > score_threshold and data[i, j, id_index] >= 0
            if is_valid:
                out_tensor[i, cnt, :] = data[i, j, :]
                out_indices[i, cnt] = j
                cnt += 1
        valid_count[i] = cnt

    return valid_count, out_tensor, out_indices
