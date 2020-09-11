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
"""Code generation related functions."""
from . import _ffi_api
from .target import Target


def build_module(mod, target):
    """Build IRModule into Module.

    Parameters
    ----------
    mod : tvm.IRModule
        The ir module.

    target : str
        The target module type.

    Returns
    -------
    module : runtime.Module
        The corressponding module.
    """
    target = Target(target) if isinstance(target, str) else target
    return _ffi_api.Build(mod, target)


def llvm_lookup_intrinsic_id(name):
    """Lookup LLVM intrinsic id by name.

    Parameters
    ----------
    name : str
        The name of the intrinsic.

    Returns
    -------
    intrin_id : int
        The intrinsic id.
    """
    return _ffi_api.llvm_lookup_intrinsic_id(name)


def llvm_version_major(allow_none=False):
    """Get the major LLVM version.

    Parameters
    ----------
    allow_none : bool
        Whether do we allow none.

    Returns
    -------
    major : int
        The major LLVM version.
    """
    try:
        return _ffi_api.llvm_version_major()
    except AttributeError:
        if allow_none:
            return None
        raise RuntimeError("LLVM version is not available, please check if you build with LLVM")
