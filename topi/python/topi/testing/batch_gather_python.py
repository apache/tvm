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
"""batch_gather in python"""
import numpy as np


def batch_gather_python(a_np, indices_np):
    """ Python version of GatherND operator

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
    indices_rank = len(indices_shape)
    assert indices_rank >= 1
    assert len(a_shape) >= 2

    b_shape = [a_shape[0]] + list(indices_np.shape) + list(a_shape[2:])
    b_np = np.zeros(b_shape)
    for idx in np.ndindex(*b_np.shape):
        a_idx = tuple([idx[0], indices_np[idx[1 : indices_rank + 1]]]
                      + list(idx[1 + indices_rank :]))
        b_np[idx] = a_np[a_idx]
    return b_np
