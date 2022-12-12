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
from tvm.relay.backend import Executor
from tvm.relay.backend.aot import CreateExecutorMetadata
from tvm.relay import TensorType
from tvm.tir.usmp.utils import PoolAllocation
from tvm.ir.memory_pools import AllocatedPoolInfo, ConstantPoolInfo, WorkspacePoolInfo, ConstantInfo


def _check_executor_metadata(executor_metadata, expected_metadata):
    assert list(executor_metadata.inputs) == expected_metadata["inputs"]
    assert list(executor_metadata.input_tensor_types) == expected_metadata["input_tensor_types"]
    assert list(executor_metadata.outputs) == expected_metadata["outputs"]
    assert list(executor_metadata.output_tensor_types) == expected_metadata["output_tensor_types"]
    assert list(executor_metadata.pools) == expected_metadata["pools"]
    assert executor_metadata.devices == expected_metadata["devices"]
    assert executor_metadata.executor == expected_metadata["executor"]
    assert executor_metadata.mod_name == expected_metadata["mod_name"]
    assert executor_metadata.interface_api == expected_metadata["interface_api"]
    assert executor_metadata.unpacked_api == expected_metadata["unpacked_api"]
    assert executor_metadata.workspace_alignment == expected_metadata["workspace_alignment"]
    assert executor_metadata.constant_alignment == expected_metadata["constant_alignment"]
    assert set(executor_metadata.pool_inputs.keys()) == set(expected_metadata["pool_inputs"].keys())
    assert set(executor_metadata.io_pool_allocations.keys()) == set(
        expected_metadata["io_pool_allocations"].keys()
    )


def test_create_executor_metadata_single_func():
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def __tvm_main__(
            a: T.handle, output: T.handle, workspace: T.Ptr[T.uint8], constants: T.Ptr[T.uint8]
        ) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind": "llvm", "tag": "", "keys": ["cpu"]}), "input_vars": [a], "output_vars": [output], "devices": ["test_device"]})
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

    target = Module["__tvm_main__"].attrs["target"]
    executor = Executor("aot", {"interface-api": "c"})
    workspace_pool_info = AllocatedPoolInfo(
        WorkspacePoolInfo("sram", [target]),
        256,
        3,
    )
    constant_pool_info = AllocatedPoolInfo(
        ConstantPoolInfo(
            "flash",
            [target],
            [ConstantInfo("a", 0, array(np.array([0])))],
        ),
        512,
        2,
    )
    io_pool_allocations = {
        "a": PoolAllocation(WorkspacePoolInfo("sram", [target]), 0),
        "output": PoolAllocation(WorkspacePoolInfo("sram", [target]), 0),
    }
    mod = Module.with_attr("io_tensor_pool_allocations", io_pool_allocations)
    mod["__tvm_main__"] = mod["__tvm_main__"].with_attr(
        "pool_args",
        [
            constant_pool_info,
            workspace_pool_info,
        ],
    )
    f = mod["__tvm_main__"]
    expected_metadata = {
        "inputs": [f.params[0]],
        "input_tensor_types": [TensorType((5, 7), "float32")],
        "outputs": ["output"],
        "output_tensor_types": [TensorType((5, 7), "float32")],
        "pools": f.params[2:],
        "devices": f.attrs["devices"],
        "executor": "aot",
        "mod_name": "test_mod",
        "interface_api": "c",
        "unpacked_api": False,
        "workspace_alignment": 16,
        "constant_alignment": 1,
        "pool_inputs": {
            f.params[2]: workspace_pool_info,
            f.params[3]: constant_pool_info,
        },
        "io_pool_allocations": io_pool_allocations,
    }

    executor_metadata = CreateExecutorMetadata(mod, "test_mod", executor, 16, 1)

    _check_executor_metadata(executor_metadata, expected_metadata)


def test_create_executor_metadata_no_usmp():
    # fmt: off
    @tvm.script.ir_module
    class Module:
        @T.prim_func
        def __tvm_main__(
            a: T.handle, output: T.handle
        ) -> None:
            # function attr dict
            T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind": "llvm", "tag": "", "keys": ["cpu"]}), "input_vars": [a], "output_vars": [output], "devices": ["test_device"]})
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

    executor = Executor("aot", {"interface-api": "c"})
    mod = Module
    f = mod["__tvm_main__"]
    expected_metadata = {
        "inputs": [f.params[0]],
        "input_tensor_types": [TensorType((5, 7), "float32")],
        "outputs": ["output"],
        "output_tensor_types": [TensorType((5, 7), "float32")],
        "pools": f.params[2:],
        "devices": f.attrs["devices"],
        "executor": "aot",
        "mod_name": "test_mod",
        "interface_api": "c",
        "unpacked_api": False,
        "workspace_alignment": 16,
        "constant_alignment": 1,
        "pool_inputs": {},
        "io_pool_allocations": {},
    }

    executor_metadata = CreateExecutorMetadata(mod, "test_mod", executor, 16, 1)

    _check_executor_metadata(executor_metadata, expected_metadata)


if __name__ == "__main__":
    tvm.testing.main()
