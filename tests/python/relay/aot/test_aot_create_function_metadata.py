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
# pylint: disable=line-too-long,missing-class-docstring,missing-module-docstring,missing-function-docstring,no-self-argument,unused-argument,invalid-name
import numpy as np

import tvm
import tvm.testing
from tvm.script import tir as T
from tvm.runtime.ndarray import array
from tvm.relay.backend.aot import CreateFunctionMetadata
from tvm.ir.memory_pools import AllocatedPoolInfo, ConstantPoolInfo, WorkspacePoolInfo, ConstantInfo


def _check_function_metadata(function_metadata, expected_infos):
    for symbol, expected_info in expected_infos.items():
        func_info = function_metadata[symbol]
        # Check workspace_sizes
        key, value = func_info.workspace_sizes.items()[0]
        assert str(key) == expected_info["target"]
        assert value == expected_info["workspace_sizes"]
        # Check io_sizes
        key, value = func_info.io_sizes.items()[0]
        assert str(key) == expected_info["target"]
        assert value == expected_info["io_sizes"]
        # Check constant_sizes
        key, value = func_info.constant_sizes.items()[0]
        assert str(key) == expected_info["target"]
        assert value == expected_info["constant_sizes"]
        # Check tir_primfuncs
        key, value = func_info.tir_primfuncs.items()[0]
        assert str(key) == expected_info["target"]
        tvm.ir.assert_structural_equal(value, expected_info["tir_primfuncs"])


def test_create_function_metadata_workspace_allocate_only():
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def __tvm_main__(a: T.handle, output: T.handle) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]})})
            a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
            output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
            # body
            sid_3 = T.allocate([140], "int8", "global.workspace")
            sid_2 = T.allocate([140], "int8", "global.workspace")
            sid_1 = T.allocate([140], "int8", "global.workspace")
            T.evaluate(T.tvm_call_cpacked("test_fused_add_0", a_buffer.data, sid_1, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
            T.evaluate(T.tvm_call_cpacked("test_fused_add_0", sid_1, sid_2, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
            T.evaluate(T.tvm_call_cpacked("test_fused_add_0", sid_2, sid_3, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
            T.evaluate(T.tvm_call_cpacked("test_fused_add_1", sid_2, sid_3, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    expected_infos = {
        "__tvm_main__": {
            "target": "llvm -keys=cpu ",
            "workspace_sizes": 432,
            "io_sizes": 280,
            "constant_sizes": 0,
            "tir_primfuncs": Module["__tvm_main__"],
        }
    }

    function_metadata = CreateFunctionMetadata(Module, 16, 1)

    _check_function_metadata(function_metadata, expected_infos)


def test_create_function_metadata_constant_allocate_only():
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def __tvm_main__(a: T.handle, output: T.handle) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "num_inputs": 1, "num_outputs": 1})
            a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
            output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
            # body
            constant_0 = T.allocate_const([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "float32", [5, 7])
            T.evaluate(T.tvm_call_cpacked("test_fused_add", a_buffer.data, constant_0, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    expected_infos = {
        "__tvm_main__": {
            "target": "llvm -keys=cpu ",
            "workspace_sizes": 0,
            "io_sizes": 280,
            "constant_sizes": 140,
            "tir_primfuncs": Module["__tvm_main__"],
        }
    }

    function_metadata = CreateFunctionMetadata(Module, 16, 1)

    _check_function_metadata(function_metadata, expected_infos)


def test_create_function_metadata_constant_pool_only():
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def __tvm_main__(a: T.handle, output: T.handle) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "num_inputs": 1, "num_outputs": 1})
            a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
            output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
            # body
            T.evaluate(T.tvm_call_cpacked("test_fused_add", a_buffer.data, a_buffer.data, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    expected_infos = {
        "__tvm_main__": {
            "target": "llvm -keys=cpu ",
            "workspace_sizes": 0,
            "io_sizes": 280,
            "constant_sizes": 256,
            "tir_primfuncs": Module["__tvm_main__"],
        }
    }

    target = Module["__tvm_main__"].attrs["target"]
    mod = Module.with_attr(
        "pool_args",
        [
            AllocatedPoolInfo(
                ConstantPoolInfo(
                    "flash",
                    [target],
                    [ConstantInfo("a", 0, array(np.array([0])))],
                ),
                256,
            ),
        ],
    )

    function_metadata = CreateFunctionMetadata(mod, 16, 1)

    _check_function_metadata(function_metadata, expected_infos)


def test_create_function_metadata_workspace_pool_only():
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def __tvm_main__(a: T.handle, output: T.handle) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "num_inputs": 1, "num_outputs": 1})
            a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
            output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
            # body
            T.evaluate(T.tvm_call_cpacked("test_fused_add", a_buffer.data, a_buffer.data, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    expected_infos = {
        "__tvm_main__": {
            "target": "llvm -keys=cpu ",
            "workspace_sizes": 256,
            "io_sizes": 280,
            "constant_sizes": 0,
            "tir_primfuncs": Module["__tvm_main__"],
        }
    }

    target = Module["__tvm_main__"].attrs["target"]
    mod = Module.with_attr(
        "pool_args",
        [
            AllocatedPoolInfo(
                WorkspacePoolInfo("sram", [target]),
                256,
            ),
        ],
    )

    function_metadata = CreateFunctionMetadata(mod, 16, 1)

    _check_function_metadata(function_metadata, expected_infos)


def test_create_function_metadata_all_single_func():
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def __tvm_main__(a: T.handle, output: T.handle) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]})})
            a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
            output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
            # body
            sid_3 = T.allocate([140], "int8", "global.workspace")
            sid_2 = T.allocate([140], "int8", "global.workspace")
            sid_1 = T.allocate([140], "int8", "global.workspace")
            constant_0 = T.allocate_const([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "float32", [5, 7])
            T.evaluate(T.tvm_call_cpacked("test_fused_add_0", a_buffer.data, sid_1, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
            T.evaluate(T.tvm_call_cpacked("test_fused_add_0", sid_1, constant_0, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
            T.evaluate(T.tvm_call_cpacked("test_fused_add_0", sid_2, sid_3, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
            T.evaluate(T.tvm_call_cpacked("test_fused_add_1", sid_2, sid_3, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    expected_infos = {
        "__tvm_main__": {
            "target": "llvm -keys=cpu ",
            "workspace_sizes": 688,
            "io_sizes": 280,
            "constant_sizes": 652,
            "tir_primfuncs": Module["__tvm_main__"],
        }
    }

    target = Module["__tvm_main__"].attrs["target"]
    mod = Module.with_attr(
        "pool_args",
        [
            AllocatedPoolInfo(
                ConstantPoolInfo(
                    "flash",
                    [target],
                    [ConstantInfo("a", 0, array(np.array([0])))],
                ),
                512,
            ),
            AllocatedPoolInfo(
                WorkspacePoolInfo("sram", [target]),
                256,
            ),
        ],
    )

    function_metadata = CreateFunctionMetadata(mod, 16, 1)

    _check_function_metadata(function_metadata, expected_infos)


def test_create_function_metadata_workspace_multi_funcs():
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def __tvm_main__(a: T.handle, output: T.handle) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "num_inputs": 1, "num_outputs": 1})
            a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
            output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
            # body
            T.evaluate(T.tvm_call_cpacked("test_fused_add", a_buffer.data, a_buffer.data, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))

        @T.prim_func
        def test_fused_add(a: T.handle, b: T.handle, output: T.handle, device_context_unused: T.handle) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "test_mod_test_fused_add", "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]})})
            a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
            b_buffer = T.match_buffer(b, [5, 7], dtype="float32", align=16)
            output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
            # body
            sid_0 = T.allocate([140], "int8", "global.workspace")
            constant_0 = T.allocate_const([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "float32", [5, 7])
            T.evaluate(T.tvm_call_cpacked("magic", a_buffer.data, b_buffer.data, sid_0, constant_0, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    expected_infos = {
        "__tvm_main__": {
            "target": "llvm -keys=cpu ",
            "workspace_sizes": 0,
            "io_sizes": 280,
            "constant_sizes": 0,
            "tir_primfuncs": Module["__tvm_main__"],
        },
        "test_fused_add": {
            "target": "llvm -keys=cpu ",
            "workspace_sizes": 144,
            "io_sizes": 420,
            "constant_sizes": 140,
            "tir_primfuncs": Module["test_fused_add"],
        },
    }

    function_metadata = CreateFunctionMetadata(Module, 16, 1)

    _check_function_metadata(function_metadata, expected_infos)


if __name__ == "__main__":
    tvm.testing.main()
