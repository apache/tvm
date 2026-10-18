# isort: skip_file
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
"""TVM runtime namespace."""

from tvm_ffi import load_module
from tvm_ffi._dtype import DataTypeCode

# class exposures
from .device import Device
from .memory_scope import MemoryScope
from ._tensor import Tensor, tensor, empty

# function exposures
from ._tensor import cpu, cuda, opencl, vulkan, metal
from ._tensor import vpi, rocm, ext_dev, hexagon
from . import _ffi_api


def enabled(target):
    """Whether module runtime is enabled for target

    Parameters
    ----------
    target : str or Dict[str, Any] or tvm_tirx.target.Target
        The target device type.

    Returns
    -------
    enabled : bool
        Whether runtime is enabled.

    Examples
    --------
    The following code checks if gpu is enabled.

    >>> tvm_tirx.runtime.enabled("gpu")
    """
    if isinstance(target, dict):
        target = target.get("kind", "")
    elif hasattr(target, "kind"):
        target = target.kind.name
    return _ffi_api.RuntimeEnabled(target)


def num_threads() -> int:
    """Get the number of threads in use by the TVM runtime.

    Returns
    -------
    int
        Number of threads in use.
    """
    return _ffi_api.NumThreads()
