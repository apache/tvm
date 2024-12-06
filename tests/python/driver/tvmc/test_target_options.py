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


@tvm.testing.requires_mrvl
def test_target_to_argparse_for_mrvl_hybrid():
    parser = argparse.ArgumentParser()
    generate_target_args(parser)
    parsed, _ = parser.parse_known_args(
        [
            "--target=mrvl, llvm",
            "--target-mrvl-mattr=wb_pin_ocm=1,quantize=fp16",
            "--target-mrvl-num_tiles=2",
            "--target-mrvl-mcpu=cnf10kb",
        ]
    )

    assert parsed.target == "mrvl, llvm"
    assert parsed.target_mrvl_mattr == "wb_pin_ocm=1,quantize=fp16"
    assert parsed.target_mrvl_num_tiles == 2
    assert parsed.target_mrvl_mcpu == "cnf10kb"


@tvm.testing.requires_mrvl
def test_default_arg_for_mrvl_hybrid():
    parser = argparse.ArgumentParser()
    generate_target_args(parser)
    parsed, _ = parser.parse_known_args(
        [
            "--target=mrvl, llvm",
        ]
    )
    assert parsed.target == "mrvl, llvm"
    assert parsed.target_mrvl_mcpu == "cn10ka"
    assert parsed.target_mrvl_num_tiles == 8


@tvm.testing.requires_mrvl
# Test for default(LLVM) target, when built with USE_MRVL=ON
def test_mrvl_build_with_llvm_only_target():
    parser = argparse.ArgumentParser()
    generate_target_args(parser)
    parsed, _ = parser.parse_known_args(
        [
            "--target=llvm",
        ]
    )
    assert parsed.target == "llvm"


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
