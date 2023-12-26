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
"""
Tests to verify Python interactions with Target Parsing
"""

import pytest

from tvm.target import Target


@pytest.mark.parametrize(["cpu_target"], [["c"], ["llvm"]])
def test_target_parser_mprofile(cpu_target):
    parsed_target = Target(f"{cpu_target} -mcpu=cortex-m55")
    assert len(parsed_target.keys) == 2
    assert parsed_target.keys[0] == "arm_cpu"
    assert parsed_target.keys[1] == "cpu"
    assert parsed_target.features
    assert parsed_target.features.has_dsp
    assert parsed_target.features.has_mve


@pytest.mark.parametrize(["cpu_target"], [["c"], ["llvm"]])
def test_target_parser_mprofile_no_mve(cpu_target):
    parsed_target = Target(f"{cpu_target} -mcpu=cortex-m7")
    assert len(parsed_target.keys) == 2
    assert parsed_target.keys[0] == "arm_cpu"
    assert parsed_target.keys[1] == "cpu"
    assert parsed_target.features
    assert parsed_target.features.has_dsp
    assert not parsed_target.features.has_mve


@pytest.mark.parametrize(["cpu_target"], [["c"], ["llvm"]])
def test_target_parser_mprofile_no_dsp(cpu_target):
    parsed_target = Target(f"{cpu_target} -mcpu=cortex-m3")
    assert len(parsed_target.keys) == 2
    assert parsed_target.keys[0] == "arm_cpu"
    assert parsed_target.keys[1] == "cpu"
    assert parsed_target.features
    assert not parsed_target.features.has_dsp
    assert not parsed_target.features.has_mve


@pytest.mark.parametrize(["cpu_target"], [["llvm"]])
def test_target_parser_mprofile_mattr(cpu_target):
    parsed_target = Target(f"{cpu_target} -mcpu=cortex-m55 -mattr=+nomve,+woof")
    assert len(parsed_target.keys) == 2
    assert parsed_target.keys[0] == "arm_cpu"
    assert parsed_target.keys[1] == "cpu"
    assert parsed_target.features
    assert parsed_target.features.has_dsp
    assert not parsed_target.features.has_mve


if __name__ == "__main__":
    tvm.testing.main()
