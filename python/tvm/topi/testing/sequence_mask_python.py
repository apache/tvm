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
"""Sequence mask in python"""
import numpy as np


def sequence_mask(data, valid_length, mask_value, axis):
    """batch_matmul operator implemented in numpy.

    Parameters
    ----------
    data : numpy.ndarray
        N-D with shape [batch_size, MAX_LENGTH, ...] or [MAX_LENGTH, batch_size, ...]

    valid_length : numpy.ndarray
        1-D with shape [batch_size,]

    mask_value : float
        Masking value

    axis : int
        The axis of the length dimension

    Returns
    -------
    out : numpy.ndarray
        N-D with shape same as data
    """
    in_shape = data.shape
    max_length = data.shape[axis]
    val_len_expand_shape = [1 for _ in range(len(in_shape))]
    val_len_expand_shape[1 - axis] = in_shape[1 - axis]
    seq_len_expand_shape = [1 for _ in range(len(in_shape))]
    seq_len_expand_shape[axis] = in_shape[axis]
    mask = np.broadcast_to(
        np.arange(max_length).reshape(seq_len_expand_shape), in_shape
    ) >= valid_length.reshape(val_len_expand_shape)
    out = data * (1 - mask) + mask_value * mask
    return out
