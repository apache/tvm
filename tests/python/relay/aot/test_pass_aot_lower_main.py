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
import pytest
import tvm
import tvm.testing
from tvm.ir import assert_structural_equal
from tvm.relay.backend.aot import AOTLowerMain, CallType
from tvm.script import tir as T


def _make_const(dtype, shape):
    return tvm.relay.const(np.zeros(shape).astype(dtype))


def _make_consts(dtype, shapes):
    return [_make_const(dtype, shape) for shape in shapes]


def _plan_devices(mod):
    host_target = tvm.target.Target("llvm")
    prim_target = tvm.target.Target("llvm", host=host_target)
    ctxt = tvm.transform.PassContext()
    config = tvm.target.make_compilation_config(ctxt, prim_target)
    mod = tvm.relay.transform.PlanDevices(config)(mod)
    mod = tvm.relay.transform.InferType()(mod)
    return mod, config


def _assert_lowered_main(mod, main_func, call_type, print_script=False):
    mod, config = _plan_devices(mod)
    mod = AOTLowerMain("test_mod", config, call_type)(mod)
    if print_script:
        print(mod["__tvm_main__"].script())

    assert_structural_equal(mod["__tvm_main__"], main_func)


def test_single_call_cpacked():
    mod = tvm.relay.parse(
        """
#[version = "0.0.5"]
def @test_fused_add(%x: Tensor[(5, 7), float32]) { %x }

def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
  %0 = (%a,) /* ty=(Tensor[(5, 7), float32],) */;
  call_lowered(@test_fused_add, %0) /* ty=Tensor[(5, 7), float32] */
}
        """,
    )

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a], "output_vars": [output], "devices": []})
        a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
        # body
        T.evaluate(T.tvm_call_cpacked("test_fused_add", a_buffer.data, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    _assert_lowered_main(mod, func, CallType.CPacked)


def test_single_call_packed():
    mod = tvm.relay.parse(
        """
#[version = "0.0.5"]
def @test_fused_add(%x: Tensor[(5, 7), float32]) { %x }

def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
  %0 = (%a,) /* ty=(Tensor[(5, 7), float32],) */;
  call_lowered(@test_fused_add, %0) /* ty=Tensor[(5, 7), float32] */
}
        """,
    )

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a], "output_vars": [output], "devices": []})
        a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
        # body
        T.evaluate(T.tvm_check_return(0, -1, T.tvm_call_packed("test_fused_add", a_buffer.data, output_buffer.data, dtype="int32"), dtype="int32"))
    # fmt: on

    _assert_lowered_main(mod, func, CallType.Packed)


def test_single_call_unpacked():
    mod = tvm.relay.parse(
        """
#[version = "0.0.5"]
def @test_fused_add(%x: Tensor[(5, 7), float32]) { %x }

def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
  %0 = (%a,) /* ty=(Tensor[(5, 7), float32],) */;
  call_lowered(@test_fused_add, %0) /* ty=Tensor[(5, 7), float32] */
}
        """,
    )

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a], "output_vars": [output], "devices": []})
        a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
        # body
        T.evaluate(T.tvm_check_return(0, -1, T.call_extern("test_fused_add", a_buffer.data, output_buffer.data, dtype="int32"), dtype="int32"))
    # fmt: on

    _assert_lowered_main(mod, func, CallType.Unpacked)


def test_constant():
    mod = tvm.relay.parse(
        """
#[version = "0.0.5"]
def @test_fused_add(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32]) { %x }

def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
  %0 = (%a, meta[relay.Constant][0]) /* ty=(Tensor[(5, 7), float32], Tensor[(5, 7), float32]) */;
  call_lowered(@test_fused_add, %0) /* ty=Tensor[(5, 7), float32] */
}
        """,
        init_meta_table={"relay.Constant": _make_consts("float32", [(5, 7)])},
    )

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "global_symbol": "test_mod___tvm_main__", "input_vars": [a], "output_vars": [output], "devices": []})
        a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
        # body
        constant_0 = T.allocate_const([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], "float32", [5, 7])
        T.evaluate(T.tvm_call_cpacked("test_fused_add", a_buffer.data, constant_0, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    _assert_lowered_main(mod, func, CallType.CPacked)


# TODO(@mbaret) There seems to be a TVMScript round-trip bug causing this to fail
@pytest.mark.xfail()
def test_copy_to_output():
    mod = tvm.relay.parse(
        """
#[version = "0.0.5"]
def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
  %a
}
        """,
    )

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a], "output_vars": [output], "devices": []})
        tmp_read = T.handle("uint8", "")
        # buffer definition
        tmp_read_1 = T.Buffer([T.uint64(140)], dtype="uint8", data=tmp_read)
        a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
        # body
        tmp_write: T.handle("uint8") = output_buffer.data
        tmp_write_1 = T.Buffer([T.uint64(140)], dtype="uint8", data=tmp_write)
        for i in T.serial(140):
            tmp_write_1[i] = T.Let(tmp_read_1[i], where={tmp_read : a_buffer.data})
    # fmt: on

    _assert_lowered_main(mod, func, CallType.CPacked)


def test_two_calls():
    mod = tvm.relay.parse(
        """
#[version = "0.0.5"]
def @test_fused_add(%x: Tensor[(5, 7), float32]) { %x }

def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
  %0 = (%a,) /* ty=(Tensor[(5, 7), float32],) */;
  %1 = call_lowered(@test_fused_add, %0) /* ty=Tensor[(5, 7), float32] */;
  %2 = (%1,) /* ty=(Tensor[(5, 7), float32],) */;
  call_lowered(@test_fused_add, %2) /* ty=Tensor[(5, 7), float32] */
}
        """,
    )

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a], "output_vars": [output], "devices": []})
        a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
        # body
        sid_2 = T.allocate([140], "int8", "global.workspace")
        T.evaluate(T.tvm_call_cpacked("test_fused_add", a_buffer.data, sid_2, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
        T.evaluate(T.tvm_call_cpacked("test_fused_add", sid_2, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    _assert_lowered_main(mod, func, CallType.CPacked)


def test_tuple_output():
    mod = tvm.relay.parse(
        """
#[version = "0.0.5"]
def @test_fused_add(%x: Tensor[(5, 7), float32]) { (%x, %x) }

def @main(%a: Tensor[(5, 7), float32]) -> (Tensor[(5, 7), float32], Tensor[(5, 7), float32]) {
  %0 = (%a,) /* ty=(Tensor[(5, 7), float32],) */;
  call_lowered(@test_fused_add, %0) /* ty=(Tensor[(5, 7), float32], Tensor[(5, 7), float32]) */
}
        """,
    )

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output0: T.handle, output1: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a], "output_vars": [output0, output1], "devices": []})
        a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
        output0_buffer = T.match_buffer(output0, [5, 7], dtype="float32", align=16)
        output1_buffer = T.match_buffer(output1, [5, 7], dtype="float32", align=16)
        # body
        T.evaluate(T.tvm_call_cpacked("test_fused_add", a_buffer.data, output0_buffer.data, output1_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    _assert_lowered_main(mod, func, CallType.CPacked)


def test_tuple_intermediate():
    mod = tvm.relay.parse(
        """
#[version = "0.0.5"]
def @test_fused_add_0(%x: Tensor[(5, 7), float32]) -> (Tensor[(5, 7), float32], Tensor[(5, 7), float32]) { (%x, %x) }
def @test_fused_add_1(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] { %x }

def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
  %0 = (%a,);
  %1 = call_lowered(@test_fused_add_0, %0);
  %2 = (%1.0, %1.1);
  call_lowered(@test_fused_add_1, %2)
}
        """,
    )

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a], "output_vars": [output], "devices": []})
        a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
        # body
        sid_3 = T.allocate([140], "int8", "global.workspace")
        sid_2 = T.allocate([140], "int8", "global.workspace")
        T.evaluate(T.tvm_call_cpacked("test_fused_add_0", a_buffer.data, sid_2, sid_3, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
        T.evaluate(T.tvm_call_cpacked("test_fused_add_1", sid_2, sid_3, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    _assert_lowered_main(mod, func, CallType.CPacked)


def test_multi_input():
    mod = tvm.relay.parse(
        """
#[version = "0.0.5"]
def @test_fused_add(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32]) { %x }

def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
  %0 = (%a, %b) /* ty=(Tensor[(5, 7), float32], Tensor[(5, 7), float32]) */;
  call_lowered(@test_fused_add, %0) /* ty=Tensor[(5, 7), float32] */
}
        """,
    )

    # fmt: off
    @T.prim_func
    def func(a: T.handle, b: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a, b], "output_vars": [output], "devices": []})
        a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
        b_buffer = T.match_buffer(b, [5, 7], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
        # body
        T.evaluate(T.tvm_call_cpacked("test_fused_add", a_buffer.data, b_buffer.data, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    _assert_lowered_main(mod, func, CallType.CPacked)


def test_let_binding():
    mod = tvm.relay.parse(
        """
#[version = "0.0.5"]
def @test_fused_add(%x: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] { %x }

def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
  %0 = (%a,);
  let %v1 = call_lowered(@test_fused_add, %0);
  %v1
}
        """,
    )

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a], "output_vars": [output], "devices": []})
        a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
        # body
        T.evaluate(T.tvm_call_cpacked("test_fused_add", a_buffer.data, output_buffer.data, T.reinterpret(T.uint64(0), dtype="handle"), dtype="int32"))
    # fmt: on

    _assert_lowered_main(mod, func, CallType.CPacked)


def test_let_binding_branch():
    mod = tvm.relay.parse(
        """
#[version = "0.0.5"]
def @test_fused_add_0(%x: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] { %x }
def @test_fused_add_1(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] { %x }

def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
  %0 = (%a,);
  let %v0 = call_lowered(@test_fused_add_0, %0);
  %1 = (%v0,);
  let %v1 = call_lowered(@test_fused_add_0, %1);
  %2 = (%v1,);
  let %v2 = call_lowered(@test_fused_add_0, %2);
  %3 = (%v1, %v2);
  let %v3 = call_lowered(@test_fused_add_1, %3);
  %v3
}
        """,
    )

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a], "output_vars": [output], "devices": []})
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

    _assert_lowered_main(mod, func, CallType.CPacked)


def test_device_hooks():
    mod = tvm.relay.parse(
        """
#[version = "0.0.5"]
def @test_fused_add(%x: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] { %x }

def @main(%a: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
  %0 = (%a,);
  %1 = call_lowered(@test_fused_add, %0);
  %2 = (%1,);
  call_lowered(@test_fused_add, %2)
}
        """,
    )

    # fmt: off
    @T.prim_func
    def func(a: T.handle, output: T.handle, device_context_example_target_hook: T.handle) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "test_mod___tvm_main__", "runner_function": True, "target": T.target({"kind":"llvm", "tag":"", "keys":["cpu"]}), "input_vars": [a], "output_vars": [output], "devices": ["example_target_hook"]})
        a_buffer = T.match_buffer(a, [5, 7], dtype="float32", align=16)
        output_buffer = T.match_buffer(output, [5, 7], dtype="float32", align=16)
        # body
        T.evaluate(T.tvm_check_return(0, -1, T.call_extern("TVMDeviceExampleTargetHookActivate", device_context_example_target_hook, dtype="int32"), dtype="int32"))
        with T.allocate([140], "int8", "global.workspace") as sid_2:
            T.evaluate(T.tvm_check_return(0, -1, T.call_extern("TVMDeviceExampleTargetHookOpen", device_context_example_target_hook, dtype="int32"), dtype="int32"))
            T.evaluate(T.tvm_call_cpacked("test_fused_add", a_buffer.data, sid_2, device_context_example_target_hook, dtype="int32"))
            T.evaluate(T.tvm_check_return(0, -1, T.call_extern("TVMDeviceExampleTargetHookClose", device_context_example_target_hook, dtype="int32"), dtype="int32"))
            T.evaluate(T.tvm_check_return(0, -1, T.call_extern("TVMDeviceExampleTargetHookOpen", device_context_example_target_hook, dtype="int32"), dtype="int32"))
            T.evaluate(T.tvm_call_cpacked("test_fused_add", sid_2, output_buffer.data, device_context_example_target_hook, dtype="int32"))
            T.evaluate(T.tvm_check_return(0, -1, T.call_extern("TVMDeviceExampleTargetHookClose", device_context_example_target_hook, dtype="int32"), dtype="int32"))
        T.evaluate(T.tvm_check_return(0, -1, T.call_extern("TVMDeviceExampleTargetHookDeactivate", device_context_example_target_hook, dtype="int32"), dtype="int32"))
    # fmt: on

    device_contexts = {}
    for gv in mod.get_global_vars():
        device_contexts[gv] = "example_target_hook"

    mod = mod.with_attr("device_contexts", device_contexts)

    _assert_lowered_main(mod, func, CallType.CPacked)


if __name__ == "__main__":
    tvm.testing.main()
