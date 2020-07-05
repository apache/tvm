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
"""Sequence last in python"""
import numpy as np

def sequence_last(data, valid_length, axis):
    """sequence_last operator implemented in numpy.

    Parameters
    ----------
    data : numpy.ndarray
        N-D with shape [batch_size, MAX_LENGTH, ...] or [MAX_LENGTH, batch_size, ...]

    valid_length : numpy.ndarray
        1-D with shape [batch_size,]

    axis : int
        The axis of the length dimension

    Returns
    -------
    out : numpy.ndarray
        N-D with shape same as data
    """
    data = np.moveaxis(data, axis, 1)
    dims = data.shape
    if valid_length is None:
        return data[:, -1]
    lengths = list(valid_length)
    return np.array([data[i, int(lengths[i]) - 1] for i in range(dims[0])])
