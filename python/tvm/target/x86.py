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
"""Common x86 related utilities"""
from .._ffi import register_func
from . import _ffi_api
from ..ir.container import Array


@register_func("tvm.target.x86.target_has_features")
def target_has_features(features, target=None):
    """Check X86 CPU features.
    Parameters
    ----------
    features : str or Array
        Feature(s) to check.
    target : Target
        Optional TVM target, default `None` use the global context target.
    Returns
    -------
    has_feats : bool
        True if feature(s) are in the target arch.
    """
    has_feats = True
    assert isinstance(features, (Array, str))
    features = [features] if isinstance(features, str) else features
    for feat in features:
        has_feats &= _ffi_api.llvm_x86_has_feature(feat, target)
    return has_feats


@register_func("tvm.topi.x86.utils.get_simd_32bit_lanes")
def get_simd_32bit_lanes():
    """X86 SIMD optimal vector length lookup.
    Parameters
    ----------
    Returns
    -------
     vec_len : int
        The optimal vector length of CPU from the global context target.
    """
    vec_len = 4
    # avx512f:  llvm.x86.avx512.addpd.w.512 (LLVM auto, added)
    # avx512bw: llvm.x86.avx512.pmaddubs.w.512" (TVM required)
    #         + llvm.x86.avx512.pmaddw.d.512"
    if target_has_features(["avx512bw", "avx512f"]):
        vec_len = 16
    elif target_has_features("avx2"):
        vec_len = 8
    return vec_len
