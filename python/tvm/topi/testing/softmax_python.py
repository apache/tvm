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
# pylint: disable=invalid-name, trailing-whitespace
"""Softmax and log_softmax operation in python"""
import numpy as np


def softmax_python(a_np):
    """Softmax operator.
    Parameters
    ----------
    a_np : numpy.ndarray
        2-D input data

    Returns
    -------
    output_np : numpy.ndarray
        2-D output with same shape
    """
    assert len(a_np.shape) == 2, "only support 2-dim softmax"
    max_elem = np.amax(a_np, axis=1)
    max_elem = max_elem.reshape(max_elem.shape[0], 1)
    e = np.exp(a_np - max_elem)
    expsum = np.sum(e, axis=1)
    out_np = e / expsum[:, None]
    return out_np


def log_softmax_python(a_np):
    """Log_softmax operator.
    Parameters
    ----------
    a_np : numpy.ndarray
        2-D input data

    Returns
    -------
    output_np : numpy.ndarray
        2-D output with same shape
    """
    assert len(a_np.shape) == 2, "only support 2-dim log_softmax"
    max_elem = np.amax(a_np, axis=1)
    max_elem = max_elem.reshape(max_elem.shape[0], 1)
    e = np.exp(a_np - max_elem)
    expsum = np.sum(e, axis=1)
    out_np = a_np - max_elem - np.log(expsum[:, None])
    return out_np
