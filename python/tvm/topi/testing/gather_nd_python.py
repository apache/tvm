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
"""gather_nd in python"""
import numpy as np


def gather_nd_python(a_np, indices_np):
    """Python version of GatherND operator

    Parameters
    ----------
    a_np : numpy.ndarray
        Numpy array

    indices_np : numpy.ndarray
        Numpy array

    Returns
    -------
    b_np : numpy.ndarray
        Numpy array
    """
    a_shape = a_np.shape
    indices_np = indices_np.astype("int32")
    indices_shape = indices_np.shape
    assert len(indices_shape) > 1
    assert indices_shape[0] <= len(a_shape)
    b_shape = list(indices_shape[1:])
    for i in range(indices_shape[0], len(a_shape)):
        b_shape.append(a_shape[i])
    b_np = np.zeros(b_shape)
    for idx in np.ndindex(*indices_shape[1:]):
        a_idx = []
        for i in range(indices_shape[0]):
            indices_pos = tuple([i] + list(idx))
            a_idx.append(indices_np[indices_pos])
        b_np[idx] = a_np[tuple(a_idx)]
    return b_np
