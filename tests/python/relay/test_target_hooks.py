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
"""Unit tests for target hooks."""
import sys
import numpy as np
import pytest
import logging

import tvm
import tvm.testing
from tvm import relay, IRModule

from utils.external_codegen import (
    parametrize_external_codegen_checks,
    set_external_func_attr,
    check_graph_executor_result,
    check_vm_result,
)

logging.basicConfig(level=logging.INFO)


@parametrize_external_codegen_checks
def test_tir_external_generation_inline_without_target_instance(check_result):
    shape = (8,)
    x_data = np.random.randint(255, size=shape).astype("float32")
    y_data = np.random.randint(255, size=shape).astype("float32")
    inputs = {"x": x_data, "y": y_data}

    x0 = relay.var("x0", shape=shape, dtype="float32")
    y0 = relay.var("y0", shape=shape, dtype="float32")
    z = x0 + y0
    f = relay.Function([x0, y0], z)
    f = set_external_func_attr(f, "example_target_hook", "replace_add_with_subtract")

    x = relay.var("x", shape=(8,), dtype="float32")
    y = relay.var("y", shape=(8,), dtype="float32")
    call = relay.Call(f, [x, y])
    func = IRModule.from_expr(call)

    check_result(func, inputs, (8,), x_data - y_data)


# TODO(mbs): The check_aot_executor_result does not support list-of-targets, mostly because
# tvm.testing.aot.compile_and_run requires the target to be a kind name string, and
# tvm.testing.aot.compile_models requires a single Target object. However, code outside of
# tvm.testing.aot is ready for this more general form.
@pytest.mark.parametrize("check_result", [check_graph_executor_result, check_vm_result])
def test_tir_external_generation_outline_with_target_instance(check_result):
    shape = (8,)
    x_data = np.random.randint(255, size=shape).astype("float32")
    y_data = np.random.randint(255, size=shape).astype("float32")
    inputs = {"x": x_data, "y": y_data}
    # Compile with an instance of the hooked target kind to demonstrate plumbing target attributes
    # into custom passes.
    host_target = tvm.target.Target("llvm")
    generic_target = tvm.target.Target("llvm", host=host_target)
    extern_codegen_target = tvm.target.Target(
        "example_target_hook -example_attribute=42", host=host_target
    )
    mod = tvm.relay.fromtext(
        """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(8), float32], %y: Tensor[(8), float32]) -> Tensor[(8), float32] {
              @replace_add_with_subtract(%x, %y) * 2.0f
            }

            def @replace_add_with_subtract(%x: Tensor[(8), float32], %y: Tensor[(8), float32],
                                           Inline=1,
                                           Primitive=1,
                                           Compiler="example_target_hook",
                                           global_symbol="replace_add_with_subtract") -> Tensor[(8), float32] {
              %x + %y  // will be rewritten to TIR implementing %x - %y - 42.0f by custom pass
            }
        """
    )

    check_result(
        mod,
        inputs,
        (8,),
        (x_data - y_data - 42.0) * 2.0,
        target=[generic_target, extern_codegen_target],
    )


@pytest.mark.parametrize("check_result", [check_graph_executor_result])
def test_runtime_module_generation(check_result):
    shape = (8,)
    x_data = np.random.randint(255, size=shape).astype("float32")
    y_data = np.random.randint(255, size=shape).astype("float32")
    inputs = {"x": x_data, "y": y_data}

    x0 = relay.var("x0", shape=shape, dtype="float32")
    y0 = relay.var("y0", shape=shape, dtype="float32")
    z = x0 + y0
    func = relay.Function([x0, y0], z)
    func = set_external_func_attr(func, "example_target_hook", "replace_add_with_subtract")
    # Test hook to trigger TIRToRuntime code generation
    func = func.with_attr("tir_to_runtime", True)

    x = relay.var("x", shape=(8,), dtype="float32")
    y = relay.var("y", shape=(8,), dtype="float32")
    call = relay.Call(func, [x, y])
    func = IRModule.from_expr(call)

    check_result(func, inputs, (8,), x_data * y_data)


if __name__ == "__main__":
    tvm.testing.main()
