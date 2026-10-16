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
"""Scalar arguments retain their inferred external-kernel ABI types."""

import math

import numpy as np

from tvm_tirx import ir
from tvm_tirx.backend.cuda import compiler
from tvm_tirx.tirx.script.ir_builder.external_kernel import SourceKernel


def test_source_kernel_scalar_argument_types(monkeypatch):
    arguments = [True, 2**31, np.int64(1), np.float64(1), 3.402e39, math.inf, math.nan]
    expected_types = ["bool", "int64", "int64", "float64", "float64", "float64", "float64"]
    expression = ir.const(3, "int32")
    arguments.append(expression)
    expected_types.append("int32")
    module = object()
    metadata = {}

    def create_module(self, binary, arg_types, launch_tags, kernel_name, **kwargs):
        metadata.update(arg_types=arg_types, launch_tags=launch_tags, kernel_name=kernel_name)
        return module

    monkeypatch.setattr(
        compiler, "compile_source", lambda *args: compiler.CompilationResult(b"", "ptx", {})
    )
    monkeypatch.setattr(SourceKernel, "_create_cuda_module", create_module)
    name, result, runtime_args = SourceKernel("// kernel source").compile_to_device_module(
        [[1], [32]], *arguments, kernel_name="kernel"
    )
    assert name == "kernel" and result is module
    assert metadata == {
        "arg_types": expected_types,
        "launch_tags": ["blockIdx.x", "threadIdx.x"],
        "kernel_name": "kernel",
    }
    assert [str(arg.ty.dtype) for arg in runtime_args[:-2]] == expected_types
    assert runtime_args[-3].same_as(expression)
    assert runtime_args[-2:] == [1, 32]
    assert math.isnan(runtime_args[6].value)
