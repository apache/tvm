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

from tvm_ffi import Array, get_global_func

from tvm_tirx.target import Target

from . import llvm
from .executable import Executable, load_static_library


def build_module(mod, target):
    """Build IRModule into Module.

    Parameters
    ----------
    mod : tvm_tirx.IRModule
        The ir module.

    target : str
        The target module type.

    Returns
    -------
    module : runtime.Module
        The corresponding module.
    """
    target = Target(target) if isinstance(target, str) else target
    return get_global_func("codegen.Build")(mod, target)


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
    assert isinstance(cpu_features, Array | list | tuple | str)
    has_feats = True
    cpu_features = [cpu_features] if isinstance(cpu_features, str) else cpu_features
    for feat in cpu_features:
        has_feats &= get_global_func("codegen.target_has_feature")(feat, target)
    return has_feats
