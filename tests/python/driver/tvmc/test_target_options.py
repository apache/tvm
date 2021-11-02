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

import argparse

import pytest

from tvm.driver import tvmc
from tvm.driver.tvmc.common import TVMCException
from tvm.driver.tvmc.target import generate_target_args, reconstruct_target_args


def test_target_to_argparse():
    parser = argparse.ArgumentParser()
    generate_target_args(parser)
    parsed, _ = parser.parse_known_args(
        ["--target=llvm", "--target-llvm-mattr=+fp,+mve", "--target-llvm-mcpu=cortex-m3"]
    )
    assert parsed.target == "llvm"
    assert parsed.target_llvm_mcpu == "cortex-m3"
    assert parsed.target_llvm_mattr == "+fp,+mve"


def test_mapping_target_args():
    parser = argparse.ArgumentParser()
    generate_target_args(parser)
    parsed, _ = parser.parse_known_args(["--target=llvm", "--target-llvm-mcpu=cortex-m3"])
    assert reconstruct_target_args(parsed) == {"llvm": {"mcpu": "cortex-m3"}}


def test_target_recombobulation_single():
    tvm_target, _ = tvmc.common.target_from_cli("llvm", {"llvm": {"mcpu": "cortex-m3"}})

    assert str(tvm_target) == "llvm -keys=cpu -link-params=0 -mcpu=cortex-m3"


def test_target_recombobulation_many():
    tvm_target, _ = tvmc.common.target_from_cli(
        "opencl -device=mali, llvm -mtriple=aarch64-linux-gnu",
        {"llvm": {"mcpu": "cortex-m3"}, "opencl": {"max_num_threads": 404}},
    )

    assert "-max_num_threads=404" in str(tvm_target)
    assert "-device=mali" in str(tvm_target)
    assert "-mtriple=aarch64-linux-gnu" in str(tvm_target.host)
    assert "-mcpu=cortex-m3" in str(tvm_target.host)


def test_error_if_target_missing():
    with pytest.raises(
        TVMCException,
        match="Passed --target-opencl-max_num_threads but did not specify opencl target",
    ):
        tvmc.common.target_from_cli(
            "llvm",
            {"opencl": {"max_num_threads": 404}},
        )
