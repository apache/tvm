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
from ..ir.container import Array


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


def target_has_features(cpu_features, target=None):
    """Check CPU features for the target's `-mtriple` and `-mcpu` and `-mattr`.

    Parameters
    ----------
    target : Target
        The TVM target.
    cpu_features : str or Array
        CPU Feature(s) to check.

    Returns
    -------
    has_features : bool
        True if target has the feature(s).
    """
    assert isinstance(target, Target) or target is None
    assert isinstance(cpu_features, (Array, list, tuple, str))
    has_feats = True
    cpu_features = [cpu_features] if isinstance(cpu_features, str) else cpu_features
    for feat in cpu_features:
        has_feats &= _ffi_api.target_has_feature(feat, target)
    return has_feats


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


def llvm_get_intrinsic_name(intrin_id: int) -> str:
    """Get the name of an intrinsic for a given id.

    Parameters
    ----------
    intrin_id : int
        The id of the intrinsic.

    Returns
    -------
    name : str
        The name of the intrinsic.
    """
    return _ffi_api.llvm_get_intrinsic_name(intrin_id)


def llvm_get_system_x86_vendor():
    """Get system x86 vendor info.

    Parameters
    ----------

    Returns
    -------
    vendor : str
        The current system's cpu vendor.
    """
    return _ffi_api.llvm_get_system_x86_vendor()


def llvm_get_system_triple():
    """Get system host triple.

    Parameters
    ----------

    Returns
    -------
    triple : str
        The current system's triple.
    """
    return _ffi_api.llvm_get_system_triple()


def llvm_get_system_cpu():
    """Get system host cpu name.

    Parameters
    ----------

    Returns
    -------
    cpu_name : str
        The current system's cpu name.
    """
    return _ffi_api.llvm_get_system_cpu()


def llvm_get_targets():
    """Get LLVM target list.

    Parameters
    ----------

    Returns
    -------
    llvm_targets : list[str]
        List of available LLVM targets.
    """
    return _ffi_api.llvm_get_targets()


def llvm_get_cpu_archlist(target=None):
    """Get CPU architectures for the target's `-mtriple`.

    Parameters
    ----------
    target : Target
        The TVM target.

    Returns
    -------
    cpu_archlist : list[str]
        List of available CPU architectures.
    """
    assert isinstance(target, Target) or target is None
    return _ffi_api.llvm_get_cpu_archlist(target)


def llvm_get_cpu_features(target=None):
    """Get CPU features for the target's `-mtriple` and `-mcpu` and considering `-mattr`.

    Parameters
    ----------
    target : Target
        The TVM target.

    Returns
    -------
    cpu_features : list[str]
        List of available CPU features.
    """
    assert isinstance(target, Target) or target is None
    return _ffi_api.llvm_get_cpu_features(target)


def llvm_cpu_has_features(cpu_features, target=None):
    """Check CPU features for the target's `-mtriple` and `-mcpu` and considering `-mattr`.

    Parameters
    ----------
    target : Target
        The TVM target.
    cpu_features : str or Array
        CPU Feature(s) to check.

    Returns
    -------
    has_features : bool
        True if target CPU has the feature(s).
    """
    assert isinstance(target, Target) or target is None
    assert isinstance(cpu_features, (Array, list, tuple, str))
    has_feats = True
    cpu_features = [cpu_features] if isinstance(cpu_features, str) else cpu_features
    for feat in cpu_features:
        has_feats &= _ffi_api.llvm_cpu_has_feature(feat, target)
    return has_feats


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
        raise RuntimeError("LLVM version is not available, please check if you built TVM with LLVM")
