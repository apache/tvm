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
"""Batch matmul in python"""
import numpy as np

def batch_matmul(x, y):
    """batch_matmul operator implemented in numpy.

    Parameters
    ----------
    x : numpy.ndarray
        3-D with shape [batch, M, K]

    y : numpy.ndarray
        3-D with shape [batch, N, K]

    Returns
    -------
    out : numpy.ndarray
        3-D with shape [batch, M, N]
    """
    batch, M, _ = x.shape
    N = y.shape[1]
    out = np.zeros((batch, M, N)).astype(x.dtype)
    for i in range(batch):
        out[i] = np.dot(x[i], y[i].T)
    return out
