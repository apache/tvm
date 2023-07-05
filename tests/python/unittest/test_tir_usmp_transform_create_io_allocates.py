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
from typing import NamedTuple, List

import tvm
from tvm.script import tir as T


# fmt: off
@tvm.script.ir_module
class SingleInputSingleOutput:
    @T.prim_func
    def tvmgen_default_fused_cast_subtract(placeholder_2: T.handle, placeholder_3: T.handle, T_subtract: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_2, [150528], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_3, [1], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        T_subtract_1 = T.match_buffer(T_subtract, [452], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        # body
        for ax0_ax1_fused_1 in T.serial(0, 224):
            for ax2_1, ax3_inner_1 in T.grid(224, 3):
                T_subtract_1[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)] = (T.cast(placeholder_4[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)], "int16") - placeholder_5[0])

    @T.prim_func
    def __tvm_main__(input: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "__tvm_main__", "runner_function": True})
        input_buffer_var = T.match_buffer(input, [150528], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        output_buffer_var = T.match_buffer(output, [452], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("tvmgen_default_fused_cast_subtract", input_buffer_var.data, T.lookup_param("p0", dtype="handle"), output_buffer_var.data, dtype="int32"))
# fmt: on


# fmt: off
@tvm.script.ir_module
class TwoInputSingleOutput:
    @T.prim_func
    def tvmgen_default_fused_cast_subtract(placeholder_2: T.handle, placeholder_3: T.handle, T_subtract: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_2, [150528], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_3, [1], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        T_subtract_1 = T.match_buffer(T_subtract, [452], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        # body
        for ax0_ax1_fused_1 in T.serial(0, 224):
            for ax2_1, ax3_inner_1 in T.grid(224, 3):
                T_subtract_1[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)] = (T.cast(placeholder_4[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)], "int16") - placeholder_5[0])

    @T.prim_func
    def __tvm_main__(input1: T.handle, input2: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "__tvm_main__", "runner_function": True})
        input1_buffer_var = T.match_buffer(input1, [150528], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        input2_buffer_var = T.match_buffer(input2, [1], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        output_buffer_var = T.match_buffer(output, [452], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("tvmgen_default_fused_cast_subtract", input1_buffer_var.data, input2_buffer_var.data, output_buffer_var.data, dtype="int32"))
# fmt: on


# fmt: off
@tvm.script.ir_module
class TwoInputTwoOutput:
    @T.prim_func
    def tvmgen_default_fused_cast_subtract(placeholder_2: T.handle, placeholder_3: T.handle, T_subtract: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_2, [150528], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_3, [1], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        T_subtract_1 = T.match_buffer(T_subtract, [452], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        # body
        for ax0_ax1_fused_1 in T.serial(0, 224):
            for ax2_1, ax3_inner_1 in T.grid(224, 3):
                T_subtract_1[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)] = (T.cast(placeholder_4[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)], "int16") - placeholder_5[0])

    @T.prim_func
    def __tvm_main__(input1: T.handle, input2: T.handle, output1: T.handle, output2: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "__tvm_main__", "runner_function": True})
        input1_buffer_var = T.match_buffer(input1, [150528], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        input2_buffer_var = T.match_buffer(input2, [150528], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        output1_buffer_var = T.match_buffer(output1, [452], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        output2_buffer_var = T.match_buffer(output2, [452], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("tvmgen_default_fused_cast_subtract", input1_buffer_var.data, T.lookup_param("p0", dtype="handle"), output1_buffer_var.data, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_cast_subtract", input2_buffer_var.data, T.lookup_param("p1", dtype="handle"), output2_buffer_var.data, dtype="int32"))
# fmt: on


# fmt: off
@tvm.script.ir_module
class SingleInputTwoOutput:
    @T.prim_func
    def tvmgen_default_fused_cast_subtract(placeholder_2: T.handle, placeholder_3: T.handle, T_subtract: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "tvmgen_default_fused_cast_subtract", "tir.noalias": True})
        placeholder_4 = T.match_buffer(placeholder_2, [150528], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        placeholder_5 = T.match_buffer(placeholder_3, [1], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        T_subtract_1 = T.match_buffer(T_subtract, [452], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        # body
        for ax0_ax1_fused_1 in T.serial(0, 224):
            for ax2_1, ax3_inner_1 in T.grid(224, 3):
                T_subtract_1[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)] = (T.cast(placeholder_4[(((ax0_ax1_fused_1*672) + (ax2_1*3)) + ax3_inner_1)], "int16") - placeholder_5[0])

    @T.prim_func
    def __tvm_main__(input: T.handle, output1: T.handle, output2: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "__tvm_main__", "runner_function": True})
        input_buffer_var = T.match_buffer(input, [150528], dtype="uint8", elem_offset=0, align=64, offset_factor=1)
        output1_buffer_var = T.match_buffer(output1, [452], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        output2_buffer_var = T.match_buffer(output2, [452], dtype="int16", elem_offset=0, align=64, offset_factor=1)
        # body
        T.evaluate(T.call_extern("tvmgen_default_fused_cast_subtract", input_buffer_var.data, T.lookup_param("p0", dtype="handle"), output1_buffer_var.data, dtype="int32"))
        T.evaluate(T.call_extern("tvmgen_default_fused_cast_subtract", input_buffer_var.data, T.lookup_param("p1", dtype="handle"), output2_buffer_var.data, dtype="int32"))
# fmt: on


class IOInfo(NamedTuple):
    """A data structure to hold test outputs per I/O tensor"""

    name: str
    shape: list
    dtype: str


def check_io_allocations(mod: tvm.IRModule, inputs: List[IOInfo], outputs: List[IOInfo]):
    """This function checks whether outer most allocates correspond to I/O tensors"""
    found_non_io_allocate_node = False

    input_name_to_info = {}
    for input in inputs:
        input_name_to_info[input.name] = input
    output_name_to_info = {}
    for output in outputs:
        output_name_to_info[output.name] = output

    def _visit(stmt):
        nonlocal found_non_io_allocate_node
        if isinstance(stmt, tvm.tir.Allocate) and not found_non_io_allocate_node:
            allocate = stmt
            if dict(allocate.annotations).get("input_tensor"):
                input_tensor_name = str(dict(allocate.annotations).get("input_tensor"))
                assert input_tensor_name in input_name_to_info.keys()
                assert input_name_to_info[input_tensor_name].shape == list(allocate.extents)
                assert input_name_to_info[input_tensor_name].dtype == str(allocate.dtype)
                del input_name_to_info[input_tensor_name]
            if dict(allocate.annotations).get("output_tensor"):
                output_tensor_name = str(dict(allocate.annotations).get("output_tensor"))
                assert output_tensor_name in output_name_to_info.keys()
                assert output_name_to_info[output_tensor_name].shape == list(allocate.extents)
                assert output_name_to_info[output_tensor_name].dtype == str(allocate.dtype)
                del output_name_to_info[output_tensor_name]
        else:
            found_non_io_allocate_node = True

    main = mod["__tvm_main__"]
    tvm.tir.stmt_functor.ir_transform(main.body, _visit, None, ["tir.Allocate", "tir.Call"])
    assert len(input_name_to_info) == 0
    assert len(output_name_to_info) == 0


@pytest.mark.parametrize(
    "test_mod, input_names, output_names",
    [
        (
            SingleInputSingleOutput,
            [IOInfo("input", [150528], "uint8")],
            [IOInfo("output", [452], "int16")],
        ),
        (
            SingleInputTwoOutput,
            [IOInfo("input", [150528], "uint8")],
            [IOInfo("output1", [452], "int16"), IOInfo("output2", [452], "int16")],
        ),
        (
            TwoInputSingleOutput,
            [IOInfo("input1", [150528], "uint8"), IOInfo("input2", [1], "int16")],
            [IOInfo("output", [452], "int16")],
        ),
        (
            TwoInputTwoOutput,
            [IOInfo("input1", [150528], "uint8"), IOInfo("input2", [150528], "uint8")],
            [IOInfo("output1", [452], "int16"), IOInfo("output2", [452], "int16")],
        ),
    ],
)
def test_mobilenet_subgraph(test_mod, input_names, output_names):
    CreateAllocatesForIO = tvm.get_global_func("tir.usmp.transform.CreateAllocatesForIO")
    test_mod = CreateAllocatesForIO()(test_mod)
    check_io_allocations(test_mod, input_names, output_names)
