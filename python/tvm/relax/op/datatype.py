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
# pylint: disable=redefined-builtin
"""Datatype operators."""
from tvm import _ffi

from ..expr import Call
from . import ty
from . import ty_guard as tg

# pylint: disable=invalid-name


## (TVM-TOOL) py_op begin datatype/*
def astype(
    x: ty.Tensor,
    dtype: ty.DType,
) -> Call:
    """TBD

    Parameters
    ----------
    x : ty.Tensor
        TODO(tvm-unity-team): add doc
    dtype : ty.DType
        TODO(tvm-unity-team): add doc

    Returns
    -------
    ret : ty.Tensor
        TODO(tvm-unity-team): add doc
    """
    x = tg.check(0, "x", tg.Tensor([]), x)
    dtype = tg.check(1, "dtype", tg.DType(), dtype)
    _ffi_func = _ffi.get_global_func("relax.op.astype")
    return _ffi_func(x, dtype)


## (TVM-TOOL) py_op end datatype/*
