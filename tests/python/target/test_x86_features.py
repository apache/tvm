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
import pytest

import tvm
from tvm.target import _ffi_api, codegen, Target
from tvm.target.codegen import target_has_features

LLVM_VERSION = codegen.llvm_version_major()

min_llvm_version, tvm_target, x86_feature, is_supported = tvm.testing.parameters(
    # sse4.1
    (-1, "llvm -mtriple=x86_64-- -mcpu=btver2", "sse4a", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=penryn", "sse4.1", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=silvermont", "sse4.2", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=slm", "sse4.2", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=goldmont", "sse4.2", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=goldmont-plus", "sse4.2", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=tremont", "sse4.2", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=nehalem", "sse4.2", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=corei7", "sse4.2", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=westmere", "sse4.2", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=bdver1", "sse4.2", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=bdver2", "sse4.2", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=bdver3", "sse4.2", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=x86-64-v2", "sse4.2", True),
    # avx
    (-1, "llvm -mtriple=x86_64-- -mcpu=sandybridge", "avx", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=corei7-avx", "avx", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=ivybridge", "avx", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=core-avx-i", "avx", True),
    # avx2
    (-1, "llvm -mtriple=x86_64-- -mcpu=haswell", "avx2", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=core-avx2", "avx2", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=broadwell", "avx2", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=skylake", "avx2", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=bdver4", "avx2", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=znver1", "avx2", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=znver2", "avx2", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=znver3", "avx2", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=x86-64-v3", "avx2", True),
    # avx512bw
    (-1, "llvm -mtriple=x86_64-- -mcpu=skylake-avx512", "avx512bw", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=skx", "avx512bw", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=knl", "avx512bw", False),
    (-1, "llvm -mtriple=x86_64-- -mcpu=knl", "avx512f", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=knl", ["avx512bw", "avx512f"], False),
    (11, "llvm -mtriple=x86_64-- -mcpu=knl", ("avx512bw", "avx512f"), False),
    (-1, "llvm -mtriple=x86_64-- -mcpu=knl", "avx512cd", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=knl", ["avx512cd", "avx512f"], True),
    (11, "llvm -mtriple=x86_64-- -mcpu=knl", ("avx512cd", "avx512f"), True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=knl", "avx512er", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=knl", "avx512pf", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=knm", "avx512bw", False),
    (-1, "llvm -mtriple=x86_64-- -mcpu=knm", "avx512f", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=knm", "avx512cd", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=knm", "avx512er", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=knm", "avx512pf", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=x86-64-v4", "avx512bw", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=cannonlake", "avx512bw", True),
    # explicit enumeration of VNNI capable due to collision with alderlake
    (11, "llvm -mtriple=x86_64-- -mcpu=alderlake", "avx512bw", False),
    (-1, "llvm -mtriple=x86_64-- -mcpu=cascadelake", "avx512bw", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=icelake-client", "avx512bw", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=icelake-server", "avx512bw", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=rocketlake", "avx512bw", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=tigerlake", "avx512bw", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=cooperlake", "avx512bw", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=sapphirerapids", "avx512bw", True),
    # avx512vnni
    (11, "llvm -mtriple=x86_64-- -mcpu=alderlake", "avx512vnni", False),
    (11, "llvm -mtriple=x86_64-- -mcpu=alderlake", "avxvnni", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=cascadelake", "avx512vnni", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=icelake-client", "avx512vnni", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=icelake-server", "avx512vnni", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=rocketlake", "avx512vnni", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=tigerlake", "avx512vnni", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=cooperlake", "avx512vnni", True),
    (11, "llvm -mtriple=x86_64-- -mcpu=sapphirerapids", "avx512vnni", True),
    # amx-int8
    (11, "llvm -mtriple=x86_64-- -mcpu=sapphirerapids", "amx-int8", True),
    # generic CPU (no features) but with extra -mattr
    (-1, "llvm -mtriple=x86_64-- -mcpu=x86-64 -mattr=+sse4.1,+avx2", "avx2", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=x86-64 -mattr=+sse4.1,+avx2", "sse4.1", True),
    # enabling +sse4.1 implies ssse3 presence in LLVM
    (-1, "llvm -mtriple=x86_64-- -mcpu=x86-64 -mattr=+sse4.1,+avx2", "ssse3", True),
    (-1, "llvm -mtriple=x86_64-- -mcpu=ivybridge -mattr=-ssse3", "ssse3", False),
    # disabling avx512f (foundation) also disables avx512bw
    (-1, "llvm -mtriple=x86_64-- -mcpu=cascadelake -mattr=-avx512f", "avx512bw", False),
)


def test_x86_target_features(min_llvm_version, tvm_target, x86_feature, is_supported):
    """Test X86 features support for different targets.

    Parameters
    ----------
    min_llvm_version : int
        Minimal LLVM version.
    tvm_target : str
        TVM target.
    x86_feature : str
        X86 CPU feature.
    is_supported : bool
        Expected result.
    """

    ##
    ## no context
    ##

    # check for feature via the python api (no explicit target, no context target)
    try:
        assert target_has_features(x86_feature) == is_supported
        assert False
    except tvm.error.InternalError as e:
        msg = str(e)
        assert (
            msg.find(
                "InternalError: Check failed: (allow_not_defined) is false: Target context required"
            )
            != -1
        )

    if isinstance(x86_feature, str):
        # check for feature via the ffi llvm api (no explicit target, no context target)
        try:
            assert _ffi_api.target_has_feature(x86_feature, None) == is_supported
            assert False
        except tvm.error.InternalError as e:
            msg = str(e)
            assert (
                msg.find(
                    "InternalError: Check failed: (allow_not_defined) is false: Target context required"
                )
                != -1
            )

    # skip test on llvm_version
    if LLVM_VERSION < min_llvm_version:
        return

    # check for feature via the python api (with explicit target, no context target)
    assert target_has_features(x86_feature, Target(tvm_target)) == is_supported
    if isinstance(x86_feature, str):
        # check for feature via the ffi llvm api (with explicit target, no context target)
        assert _ffi_api.target_has_feature(x86_feature, Target(tvm_target)) == is_supported

    ##
    ## with context
    ##

    with Target(tvm_target):
        mcpu = Target.current(False).mcpu
        # check for feature via the python api (current context target)
        assert target_has_features(x86_feature) == is_supported
        # check for feature via the python api (with explicit target)
        assert target_has_features(x86_feature, Target(tvm_target)) == is_supported
        # check for feature via the ffi llvm api (current context target)
        (sum(_ffi_api.target_has_feature(feat, None) for feat in x86_feature) > 0) == is_supported
        # check for feature in target's llvm full x86 CPU feature list
        if (not Target(tvm_target).mattr) and isinstance(x86_feature, str):
            assert (x86_feature in codegen.llvm_get_cpu_features()) == is_supported
