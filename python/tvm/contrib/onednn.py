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
"""External function interface to OneDNN library."""
# pylint: disable-msg=C0103
import ctypes
import numpy as np
import tvm

import tvm._ffi
from tvm import te

def exists():
    """
    Checks whether the local machine can use OneDNN.

    Returns
    -------
        exists: bool

            True if OneDNN support is enabled and a OneDNN-capable GPU
            exists.  Otherwise, False.
    """
    func = tvm.get_global_func("tvm.contrib.onednn.exists", allow_missing=True)
    if func is None:
        return False

    return bool(func())

def softmax(x, axis=-1):
    """Compute softmax using OneDNN

    Parameters
    ----------
    x : tvm.te.Tensor
        The input tensor

    axis : int
        The axis to compute the softmax

    Returns
    -------
    ret : tvm.te.Tensor
        The result tensor
    """
    return te.extern(
        x.shape,
        [x],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.onednn.softmax.forward", ins[0], outs[0], axis
        ),
        name="y",
    )


def log_softmax(x, axis=-1):
    """Compute log_softmax using OneDNN

    Parameters
    ----------
    x : tvm.te.Tensor
        The input tensor

    axis : int
        The axis to compute log softmax over

    Returns
    -------
    ret : tvm.te.Tensor
        The result tensor
    """
    return te.extern(
        x.shape,
        [x],
        lambda ins, outs: tvm.tir.call_packed(
            "tvm.contrib.onednn.log_softmax.forward", ins[0], outs[0], axis
        ),
        name="y",
    )
