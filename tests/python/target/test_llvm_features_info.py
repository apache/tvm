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

LLVM_VERSION = codegen.llvm_version_major()


def test_llvm_targets(capfd):

    ##
    ## check LLVM backend
    ##

    # check blank results
    assert len(codegen.llvm_get_targets())
    assert len(codegen.llvm_get_system_cpu())
    assert len(codegen.llvm_get_system_triple())
    assert len(codegen.llvm_get_system_x86_vendor())
    # check ffi vs python
    assert codegen.llvm_get_system_cpu() == _ffi_api.llvm_get_system_cpu()
    assert codegen.llvm_get_system_triple() == _ffi_api.llvm_get_system_triple()
    assert codegen.llvm_get_system_x86_vendor() == _ffi_api.llvm_get_system_x86_vendor()
    assert str(codegen.llvm_get_targets()) == str(_ffi_api.llvm_get_targets())

    tvm.target.codegen.llvm_get_cpu_features(
        tvm.target.Target("llvm -mtriple=x86_64-linux-gnu -mcpu=dummy")
    )
    expected_str = (
        " with `-mcpu=dummy` is not valid in "
        "`-mtriple=x86_64-linux-gnu`, using default `-mcpu=generic`"
    )
    readout_error = capfd.readouterr().err
    assert "Error: Using LLVM " in readout_error
    assert expected_str in readout_error


min_llvm_version, llvm_target, cpu_arch, cpu_features, is_supported = tvm.testing.parameters(
    (-1, "x86_64", "sandybridge", "sse4.1", True),
    (-1, "x86_64", "ivybridge", ["sse4.1", "ssse3"], True),
    (-1, "x86_64", "ivybridge", ["sse4.1", "ssse3", "avx512bw"], False),
    # 32bit vs 64bit
    (-1, "aarch64", "cortex-a55", "neon", True),
    (-1, "aarch64", "cortex-a55", "dotprod", True),
    (-1, "aarch64", "cortex-a55", "dsp", False),
    (-1, "arm", "cortex-a55", "dsp", True),
    (-1, "aarch64", "cortex-a55", ["neon", "dotprod"], True),
    (-1, "aarch64", "cortex-a55", ["neon", "dotprod", "dsp"], False),
    (-1, "arm", "cortex-a55", ["neon", "dotprod"], True),
    (-1, "aarch64", "cortex-a55", ["neon", "dotprod", "dsp"], False),
    (-1, "arm", "cortex-a55", ["neon", "dotprod", "dsp"], True),
)


def test_target_features(min_llvm_version, llvm_target, cpu_arch, cpu_features, is_supported):

    target = Target("llvm -mtriple=%s-- -mcpu=%s" % (llvm_target, cpu_arch))

    ##
    ## legalize llvm_target
    ##

    assert llvm_target in codegen.llvm_get_targets()

    ##
    ## legalize cpu_arch
    ##

    ### with context
    with target:
        assert cpu_arch in codegen.llvm_get_cpu_archlist()
    ### no context but with expicit target
    assert cpu_arch in codegen.llvm_get_cpu_archlist(target)
    # check ffi vs python
    assert str(codegen.llvm_get_cpu_archlist(target)) == str(_ffi_api.llvm_get_cpu_archlist(target))

    ##
    ## check has_features
    ##

    ### with context
    with target:
        assert codegen.llvm_cpu_has_features(cpu_features) == is_supported
    ### no context but with expicit target
    assert codegen.llvm_cpu_has_features(cpu_features, target) == is_supported
    # check ffi vs python
    for feat in cpu_features:
        assert str(codegen.llvm_cpu_has_features(feat, target)) == str(
            _ffi_api.llvm_cpu_has_feature(feat, target)
        )
