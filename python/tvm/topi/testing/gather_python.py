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
# pylint: disable=invalid-name, line-too-long, unused-variable, too-many-locals
"""gather in python"""
import numpy as np


def gather_python(data, axis, indices):
    """Python version of Gather operator

    Parameters
    ----------
    data : numpy.ndarray
        Numpy array

    axis: int
        integer

    indices : numpy.ndarray
        Numpy array

    Returns
    -------
    b_np : numpy.ndarray
        Numpy array
    """
    shape_indices = indices.shape
    out = np.zeros(shape_indices, dtype=data.dtype)
    for index in np.ndindex(*shape_indices):
        new_index = list(index)
        new_index[axis] = indices[index]
        out[index] = data[tuple(new_index)]
    return out
