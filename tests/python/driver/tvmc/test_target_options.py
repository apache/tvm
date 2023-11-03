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

import tvm
from tvm.driver.tvmc import TVMCException
from tvm.driver.tvmc.target import generate_target_args, reconstruct_target_args, target_from_cli


def test_target_to_argparse():
    parser = argparse.ArgumentParser()
    generate_target_args(parser)
    parsed, _ = parser.parse_known_args(
        ["--target=llvm", "--target-llvm-mattr=+fp,+mve", "--target-llvm-mcpu=cortex-m3"]
    )
    assert parsed.target == "llvm"
    assert parsed.target_llvm_mcpu == "cortex-m3"
    assert parsed.target_llvm_mattr == "+fp,+mve"


@tvm.testing.requires_cmsisnn
def test_target_to_argparse_known_codegen():
    parser = argparse.ArgumentParser()
    generate_target_args(parser)
    parsed, _ = parser.parse_known_args(
        [
            "--target=cmsis-nn,llvm",
            "--target-cmsis-nn-mcpu=cortex-m3",
            "--target-llvm-mattr=+fp,+mve",
            "--target-llvm-mcpu=cortex-m3",
        ]
    )
    assert parsed.target == "cmsis-nn,llvm"
    assert parsed.target_llvm_mcpu == "cortex-m3"
    assert parsed.target_llvm_mattr == "+fp,+mve"
    assert parsed.target_cmsis_nn_mcpu == "cortex-m3"


def test_mapping_target_args():
    parser = argparse.ArgumentParser()
    generate_target_args(parser)
    parsed, _ = parser.parse_known_args(["--target=llvm", "--target-llvm-mcpu=cortex-m3"])
    assert reconstruct_target_args(parsed) == {"llvm": {"mcpu": "cortex-m3"}}


@tvm.testing.requires_vitis_ai
def test_composite_target_cmd_line_help():
    parser = argparse.ArgumentParser()
    generate_target_args(parser)
    assert parser._option_string_actions["--target-vitis-ai-dpu"].help == "Vitis AI DPU identifier"
    assert (
        parser._option_string_actions["--target-vitis-ai-build_dir"].help
        == "Build directory to be used (optional, debug)"
    )
    assert (
        parser._option_string_actions["--target-vitis-ai-work_dir"].help
        == "Work directory to be used (optional, debug)"
    )
    assert (
        parser._option_string_actions["--target-vitis-ai-export_runtime_module"].help
        == "Export the Vitis AI runtime module to this file"
    )
    assert (
        parser._option_string_actions["--target-vitis-ai-load_runtime_module"].help
        == "Load the Vitis AI runtime module to this file"
    )


@tvm.testing.requires_cmsisnn
def test_include_known_codegen():
    parser = argparse.ArgumentParser()
    generate_target_args(parser)
    parsed, _ = parser.parse_known_args(
        ["--target=cmsis-nn,c", "--target-cmsis-nn-mcpu=cortex-m55", "--target-c-mcpu=cortex-m55"]
    )
    assert reconstruct_target_args(parsed) == {
        "c": {"mcpu": "cortex-m55"},
        "cmsis-nn": {"mcpu": "cortex-m55"},
    }


@tvm.testing.requires_ethosu
def test_ethosu_compiler_attrs():
    # It is checked that the represented string and boolean types in the
    # EthosUCompilerConfigNode structure can be passed via the command line
    parser = argparse.ArgumentParser()
    generate_target_args(parser)
    parsed, _ = parser.parse_known_args(
        ["--target-ethos-u-accelerator_config=ethos-u55-32", "--target-ethos-u-enable_cascader=1"]
    )
    assert reconstruct_target_args(parsed) == {
        "ethos-u": {"accelerator_config": "ethos-u55-32", "enable_cascader": 1},
    }


def test_skip_target_from_codegen():
    parser = argparse.ArgumentParser()
    generate_target_args(parser)
    parsed, left = parser.parse_known_args(
        ["--target=cmsis-nn, c", "--target-cmsis-nn-from_device=1", "--target-c-mcpu=cortex-m55"]
    )
    assert left == ["--target-cmsis-nn-from_device=1"]
    assert reconstruct_target_args(parsed) == {"c": {"mcpu": "cortex-m55"}}


def test_target_recombobulation_single():
    tvm_target, _ = target_from_cli("llvm", {"llvm": {"mcpu": "cortex-m3"}})

    assert str(tvm_target) == "llvm -keys=arm_cpu,cpu -mcpu=cortex-m3"


def test_target_recombobulation_many():
    tvm_target, _ = target_from_cli(
        "opencl -device=mali, llvm -mtriple=aarch64-linux-gnu",
        {"llvm": {"mcpu": "cortex-m3"}, "opencl": {"max_num_threads": 404}},
    )

    assert "-max_num_threads=404" in str(tvm_target)
    assert "-device=mali" in str(tvm_target)
    assert "-mtriple=aarch64-linux-gnu" in str(tvm_target.host)
    assert "-mcpu=cortex-m3" in str(tvm_target.host)


def test_target_recombobulation_codegen():
    tvm_target, extras = target_from_cli(
        "cmsis-nn, c -mcpu=cortex-m55",
        {"cmsis-nn": {"mcpu": "cortex-m55"}},
    )

    assert "-mcpu=cortex-m55" in str(tvm_target)
    assert len(extras) == 1
    assert extras[0]["name"] == "cmsis-nn"
    assert extras[0]["opts"] == {"mcpu": "cortex-m55"}


def test_error_if_target_missing():
    with pytest.raises(
        TVMCException,
        match="Passed --target-opencl-max_num_threads but did not specify opencl target",
    ):
        target_from_cli(
            "llvm",
            {"opencl": {"max_num_threads": 404}},
        )
