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
# pylint: disable=wrong-import-position

"""
Test memory pressure is calculated correctly from used memory annotations.
"""

import pytest

pytest.importorskip("ethosu.vela")

import tvm
from tvm import relay
from tvm.relay.backend.contrib.ethosu.codegen import _calculate_memory_pressure
from tvm.contrib.ethosu.cascader.scheduler import extract_memory_info
from tvm import WorkspacePoolInfo, PoolInfoProperties


def _npu_and_non_npu_functions():
    mod = tvm.IRModule({})

    # NPU function 1
    x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
    max_pool = relay.nn.max_pool2d(x)
    composite_func = relay.Function([x], max_pool)
    composite_func = composite_func.with_attr("Composite", "ethos-u.pooling")
    inp = relay.var("input", shape=(1, 2, 2, 4), dtype="int8")
    compiler_func = relay.Function([inp], composite_func)
    compiler_func = compiler_func.with_attr("used_memory", [32])
    npu_compiler_func1 = compiler_func.with_attr("Compiler", "ethos-u")
    g1 = relay.GlobalVar("g1")
    mod[g1] = npu_compiler_func1

    # Non-NPU function
    x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
    max_pool = relay.abs(x)
    composite_func = relay.Function([x], max_pool)
    composite_func = composite_func.with_attr("Composite", "foo.unary_elementwise")
    inp = relay.var("input", shape=(1, 2, 2, 4), dtype="int8")
    compiler_func = relay.Function([inp], composite_func)
    compiler_func = compiler_func.with_attr("used_memory", [32])
    non_npu_compiler_func = compiler_func.with_attr("Compiler", "foo")
    g2 = relay.GlobalVar("g2")
    mod[g2] = non_npu_compiler_func

    # NPU function 2
    x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
    max_pool = relay.abs(x)
    composite_func = relay.Function([x], max_pool)
    composite_func = composite_func.with_attr("Composite", "ethos-u.unary_elementwise")
    inp = relay.var("input", shape=(1, 2, 2, 4), dtype="int8")
    compiler_func = relay.Function([inp], composite_func)
    compiler_func = compiler_func.with_attr("used_memory", [32])
    npu_compiler_func2 = compiler_func.with_attr("Compiler", "ethos-u")
    g3 = relay.GlobalVar("g3")
    mod[g3] = npu_compiler_func2

    # Main
    inp = relay.var("main_input", shape=(1, 2, 2, 4), dtype="int8")
    call1 = relay.Call(g1, [inp])
    call2 = relay.Call(g2, [call1])
    call3 = relay.Call(g3, [call2])
    main_func = relay.Function([inp], call3)
    main_func = main_func.with_attr("io_used_memory", 32)
    mod["main"] = main_func
    return mod


def _parallel_npu_functions():
    mod = tvm.IRModule({})

    # NPU function 1
    x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
    max_pool = relay.nn.max_pool2d(x)
    composite_func = relay.Function([x], max_pool)
    composite_func = composite_func.with_attr("Composite", "ethos-u.pooling")
    inp = relay.var("input", shape=(1, 2, 2, 4), dtype="int8")
    compiler_func = relay.Function([inp], composite_func)
    compiler_func = compiler_func.with_attr("used_memory", [32])
    npu_compiler_func1 = compiler_func.with_attr("Compiler", "ethos-u")
    g1 = relay.GlobalVar("g1")
    mod[g1] = npu_compiler_func1

    # NPU function 2
    x = relay.var("x", shape=(1, 2, 2, 4), dtype="int8")
    abs_op = relay.abs(x)
    composite_func = relay.Function([x], abs_op)
    composite_func = composite_func.with_attr("Composite", "ethos-u.unary_elementwise")
    inp = relay.var("input", shape=(1, 2, 2, 4), dtype="int8")
    compiler_func = relay.Function([inp], composite_func)
    compiler_func = compiler_func.with_attr("used_memory", [32 + 16])
    npu_compiler_func2 = compiler_func.with_attr("Compiler", "ethos-u")
    g2 = relay.GlobalVar("g2")
    mod[g2] = npu_compiler_func2

    # Main
    inp = relay.var("main_input", shape=(1, 2, 2, 4), dtype="int8")
    call1 = relay.Call(g1, [inp])
    call2 = relay.Call(g2, [inp])
    concat = relay.concatenate([call1, call2], axis=3)
    main_func = relay.Function([inp], concat)
    main_func = main_func.with_attr("io_used_memory", 32)
    mod["main"] = main_func
    return mod


def _full_offload():
    mod = tvm.IRModule({})

    # NPU function
    x = relay.var("x", shape=(1, 4, 4, 16), dtype="int8")
    max_pool = relay.nn.max_pool2d(x)
    composite_func = relay.Function([x], max_pool)
    composite_func = composite_func.with_attr("Composite", "ethos-u.pooling")
    inp = relay.var("input", shape=(1, 4, 4, 16), dtype="int8")
    compiler_func = relay.Function([inp], composite_func)
    compiler_func = compiler_func.with_attr("used_memory", [256 + 256])
    npu_compiler_func = compiler_func.with_attr("Compiler", "ethos-u")
    g1 = relay.GlobalVar("g1")
    mod[g1] = npu_compiler_func

    # Main
    inp = relay.var("main_input", shape=(1, 4, 4, 16), dtype="int8")
    call = relay.Call(g1, [inp])
    main_func = relay.Function([inp], call)
    main_func = main_func.with_attr("io_used_memory", 256 + 256)
    mod["main"] = main_func
    return mod


@pytest.mark.parametrize(
    "model_func,use_workspace_io,expected_memory_pressure",
    [
        (_npu_and_non_npu_functions, True, (16 + 16) + (16 + 16)),
        (_npu_and_non_npu_functions, False, (16 + 16) + (16 + 16) - (16 + 16)),
        (_parallel_npu_functions, True, (16 + 16) + (16 + 16 + 16)),
        (_parallel_npu_functions, False, (16 + 16) + (16 + 16 + 16) - (16 + 16)),
        (_full_offload, True, (256 + 256)),
        (_full_offload, False, (256 + 256) - (256 + 256)),
    ],
)
def test_calculate_memory_pressure_pass(model_func, use_workspace_io, expected_memory_pressure):
    """
    Test that memory pressure is correctly calculated for NPU external functions.
    """

    mod = model_func()
    with tvm.transform.PassContext(config={"tir.usmp.use_workspace_io": use_workspace_io}):
        memory_pressure = _calculate_memory_pressure(mod)
    assert memory_pressure == expected_memory_pressure


def test_extract_memory_info():
    """
    Test memory pressure value correctly reduces the workspace size.
    """
    initial_pool_size = 2000
    memory_pressure = 500
    memory_pool = WorkspacePoolInfo(
        "SRAM",
        [tvm.target.Target("c"), tvm.target.Target("ethos-u")],
        PoolInfoProperties(
            size_hint_bytes=initial_pool_size,
            read_bandwidth_bytes_per_cycle=16,
            write_bandwidth_bytes_per_cycle=16,
            target_burst_bytes={tvm.target.Target("ethos-u"): 1},
        ),
    )

    sram = extract_memory_info(memory_pool, memory_pressure)
    assert sram.size == initial_pool_size - memory_pressure
