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
import os
import shutil

from inspect import isfunction
from os import path

import pytest

import tvm

from tvm.driver import tvmc

from tvm.driver.tvmc import TVMCException


def test_get_codegen_names():
    names = tvmc.composite_target.get_codegen_names()

    assert "ethos-n" in names
    assert "vitis-ai" in names
    assert len(names) > 0


def test_valid_codegen():
    codegen = tvmc.composite_target.get_codegen_by_target("compute-library")

    assert codegen is not None
    assert codegen["pass_pipeline"] is not None


def test_invalid_codegen():
    with pytest.raises(TVMCException):
        _ = tvmc.composite_target.get_codegen_by_target("invalid")


def test_all_codegens_contain_pass_pipeline():
    for name in tvmc.composite_target.get_codegen_names():
        codegen = tvmc.composite_target.get_codegen_by_target(name)
        assert "pass_pipeline" in codegen, f"{name} does not contain a pass_pipeline"
        assert isfunction(codegen["pass_pipeline"])


def test_all_pass_pipelines_are_functions():
    for name in tvmc.composite_target.get_codegen_names():
        codegen = tvmc.composite_target.get_codegen_by_target(name)
        assert isfunction(codegen["pass_pipeline"]), f"pass_pipeline for {name} is not a function"
