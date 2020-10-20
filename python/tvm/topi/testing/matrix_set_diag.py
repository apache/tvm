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
"""MatrixSetDiag in Python"""
import numpy as np


def matrix_set_diag(input_np, diagonal, k=0, align="RIGHT_LEFT"):
    """matrix_set_diag operator implemented in numpy.

    Returns a numpy array with the diagonals of input array
    replaced with the provided diagonal values.

    Parameters
    ----------
    input_np : numpy.ndarray
        Input Array.
        Shape = [D1, D2, D3, ... , Dn-1 , Dn]

    diagonal : numpy.ndarray
        Values to be filled in the diagonal.

    k : int or tuple of int
        Diagonal Offsets.

    align : string
        Some diagonals are shorter than max_diag_len and need to be padded.
        Possible Vales:
        ["RIGHT_LEFT" (default), "LEFT_RIGHT", "LEFT_LEFT", "RIGHT_RIGHT"]

    Returns
    -------
    result : numpy.ndarray
        New Array with given diagonal values.
        Shape = [D1, D2, D3, ... , Dn-1 , Dn]
    """
    out = np.array(input_np, copy=True)

    cols = input_np.shape[-1]
    rows = input_np.shape[-2]

    onlyOneDiagonal = True
    if isinstance(k, (tuple, list)):
        if len(k) < 2 or k[0] == k[1]:
            k = k[0]
        else:
            onlyOneDiagonal = False

    if onlyOneDiagonal:
        for i in range(diagonal.shape[-1]):
            if k >= 0:
                out[..., i, i + k] = diagonal[..., i]
            else:
                out[..., i - k, i] = diagonal[..., i]
    else:
        for ki in range(k[0], k[1] + 1):
            diag_len = min(cols - max(ki, 0), rows + min(ki, 0))
            offset = 0
            if ki >= 0:
                if align[:5] == "RIGHT":
                    offset = diagonal.shape[-1] - diag_len
            else:
                if align[-5:] == "RIGHT":
                    offset = diagonal.shape[-1] - diag_len
            for i in range(diag_len):
                if ki >= 0:
                    out[..., i, i + ki] = diagonal[..., k[1] - ki, i + offset]
                else:
                    out[..., i - ki, i] = diagonal[..., k[1] - ki, i + offset]
    return out
