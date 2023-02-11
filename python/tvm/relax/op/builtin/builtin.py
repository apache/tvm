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
"""The builtin Relax operators."""

from ...expr import Call, Expr
from ...utils import args_converter
from . import _ffi_api


@args_converter.auto
def alloc_tensor(shape: Expr, dtype: str, runtime_device_index: int) -> Call:
    """Construct a Call to allocate a tensor with specific shape, dtype, runtime_device_index.

    Parameters
    ----------
    shape : Expr
        The shape of the tensor to be allocated.

    dtype : str
        The datatype of the tensor to be allocated.

    runtime_device_index : int
        The device index indicating on which device the tensor is to be allocated at runtime.
        Index -1 is reserved for the host device.

    Returns
    -------
    result : Call
        A relax Call, which gets the allocated tensor.
    """
    return _ffi_api.alloc_tensor(shape, dtype, runtime_device_index)  # type: ignore
