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


def matrix_set_diag(input_np, diagonal):
    """matrix_set_diag operator implemented in numpy.

    Returns a numpy array with the diagonal of input array
    replaced with the provided diagonal values.

    Parameters
    ----------
    input : numpy.ndarray
        Input Array.
        Shape = [D1, D2, D3, ... , Dn-1 , Dn]
    diagonal : numpy.ndarray
        Values to be filled in the diagonal.
        Shape = [D1, D2, D3, ... , Dn-1]

    Returns
    -------
    result : numpy.ndarray
        New Array with given diagonal values.
        Shape = [D1, D2, D3, ... , Dn-1 , Dn]
    """
    out = np.array(input_np, copy=True)
    n = min(input_np.shape[-1], input_np.shape[-2])
    for i in range(n):
        out[..., i, i] = diagonal[..., i]

    return out
