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
""" Test Meta Schedule Builder """

import os
import sys
import time
from typing import List

import pytest
import tvm.testing

from tvm import script
from tvm._ffi import register_func
from tvm.meta_schedule.builder import (
    BuilderInput,
    BuilderResult,
    LocalBuilder,
    PyBuilder,
)
from tvm.runtime import Module
from tvm.script import tir as T
from tvm.target import Target


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


@script.ir_module
class MatmulModule:
    @T.prim_func
    def matmul(a: T.handle, b: T.handle, c: T.handle) -> None:  # pylint: disable=no-self-argument
        T.func_attr({"global_symbol": "matmul", "tir.noalias": True})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@script.ir_module
class MatmulReluModule:
    @T.prim_func
    def matmul_relu(  # pylint: disable=no-self-argument
        a: T.handle, b: T.handle, d: T.handle
    ) -> None:
        T.func_attr({"global_symbol": "matmul_relu", "tir.noalias": True})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        D = T.match_buffer(d, (1024, 1024), "float32")
        C = T.alloc_buffer((1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
        for i, j in T.grid(1024, 1024):
            with T.block("relu"):
                vi, vj = T.axis.remap("SS", [i, j])
                D[vi, vj] = T.max(C[vi, vj], 0.0)


@script.ir_module
class BatchMatmulModule:
    @T.prim_func
    def batch_matmul(  # pylint: disable=no-self-argument
        a: T.handle, b: T.handle, c: T.handle
    ) -> None:
        T.func_attr({"global_symbol": "batch_matmul", "tir.noalias": True})
        A = T.match_buffer(a, [16, 128, 128])
        B = T.match_buffer(b, [16, 128, 128])
        C = T.match_buffer(c, [16, 128, 128])
        for n, i, j, k in T.grid(16, 128, 128, 128):
            with T.block("update"):
                vn, vi, vj, vk = T.axis.remap("SSSR", [n, i, j, k])
                with T.init():
                    C[vn, vi, vj] = 0.0
                C[vn, vi, vj] = C[vn, vi, vj] + A[vn, vi, vk] * B[vn, vj, vk]


# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,missing-docstring


def _check_build_results(builder_results: List[BuilderResult]):
    """Simple check whether the build is successful"""
    for result in builder_results:
        artifact_path = result.artifact_path
        error_msg = result.error_msg
        assert artifact_path is not None
        assert error_msg is None
        os.remove(artifact_path)
        os.rmdir(os.path.dirname(artifact_path))


def test_meta_schedule_single_build():
    """Test meta schedule builder for a single build"""
    mod = MatmulModule
    builder = LocalBuilder()
    builder_inputs = [BuilderInput(mod, Target("llvm"))]
    builder_results = builder.build(builder_inputs)
    assert len(builder_results) == len(builder_inputs)
    _check_build_results(builder_results)


def test_meta_schedule_multiple_build():
    """Test meta schedule builder for multiple builds"""
    builder = LocalBuilder()
    builder_inputs = [
        BuilderInput(MatmulModule, Target("llvm")),
        BuilderInput(MatmulReluModule, Target("llvm")),
        BuilderInput(BatchMatmulModule, Target("llvm")),
    ]
    builder_results = builder.build(builder_inputs)
    assert len(builder_results) == len(builder_inputs)
    _check_build_results(builder_results)


def test_meta_schedule_error_handle_test_builder():
    """Test the error handing during building"""

    class TestBuilder(PyBuilder):
        def build(  # pylint: disable=no-self-use
            self,
            build_inputs: List[BuilderInput],
        ) -> List[BuilderResult]:
            return [BuilderResult(None, "error") for w in build_inputs]

    builder = TestBuilder()
    builder_inputs = [
        BuilderInput(MatmulModule, Target("llvm")),
        BuilderInput(MatmulReluModule, Target("llvm")),
        BuilderInput(BatchMatmulModule, Target("llvm")),
    ]
    builder_results = builder.build(builder_inputs)
    assert len(builder_results) == len(builder_inputs)
    for result in builder_results:
        artifact_path = result.artifact_path
        error_msg = result.error_msg
        assert artifact_path is None
        assert error_msg == "error"


def test_meta_schedule_error_handle_build_func():
    """Test the error handing during building"""

    def initializer():
        @register_func("meta_schedule.builder.test_build")
        def test_build(mod: Module, target: Target, _) -> None:  # pylint: disable=unused-variable
            raise ValueError("Builder intended Test Error (build func).")

    builder = LocalBuilder(f_build="meta_schedule.builder.test_build", initializer=initializer)
    builder_inputs = [BuilderInput(MatmulModule, Target("llvm"))]
    builder_results = builder.build(builder_inputs)
    assert len(builder_results) == len(builder_inputs)
    for result in builder_results:
        artifact_path = result.artifact_path
        error_msg = result.error_msg
        assert artifact_path is None
        assert error_msg.startswith("LocalBuilder: An exception occurred")


def test_meta_schedule_error_handle_export_func():
    """Test the error handing during building"""

    def initializer():
        @register_func("meta_schedule.builder.test_export")
        def test_build(mod: Module) -> str:  # pylint: disable=unused-variable
            raise ValueError("Builder intended Test Error (export func).")

    builder = LocalBuilder(f_export="meta_schedule.builder.test_export", initializer=initializer)
    builder_inputs = [BuilderInput(MatmulModule, Target("llvm"))]
    builder_results = builder.build(builder_inputs)
    assert len(builder_results) == len(builder_inputs)
    for result in builder_results:
        artifact_path = result.artifact_path
        error_msg = result.error_msg
        assert artifact_path is None
        assert error_msg.startswith("LocalBuilder: An exception occurred")


def test_meta_schedule_error_handle_time_out():
    """Test the error handing time out during building"""

    def initializer():
        @register_func("meta_schedule.builder.test_time_out")
        def timeout_build(mod, target, _):  # pylint: disable=unused-argument, unused-variable
            time.sleep(2)

    builder = LocalBuilder(
        timeout_sec=1,
        f_build="meta_schedule.builder.test_time_out",
        initializer=initializer,
    )
    builder_inputs = [BuilderInput(MatmulModule, Target("llvm"))]
    builder_results = builder.build(builder_inputs)
    assert len(builder_results) == len(builder_inputs)
    for result in builder_results:
        artifact_path = result.artifact_path
        error_msg = result.error_msg
        assert artifact_path is None
        assert error_msg.startswith("LocalBuilder: Timeout")


def test_meta_schedule_missing_build_func():
    with pytest.raises(ValueError):
        LocalBuilder(f_build="wrong-name")


if __name__ == "__main__":
    tvm.testing.main()
