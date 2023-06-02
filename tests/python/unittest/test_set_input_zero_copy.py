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
# pylint: disable=missing-function-docstring,missing-module-docstring
import tvm
from tvm import relay
import numpy as np
from tvm.contrib import graph_executor
from tvm import testing
import numpy as np
import pytest


dev = tvm.cpu(0)
target = tvm.target.Target("llvm")


def build_relay_module(func):
    mod = tvm.IRModule()
    mod["main"] = func
    lib = relay.build(mod, target=target)

    return graph_executor.GraphModule(lib["default"](dev))


@testing.requires_llvm
def test_simple_graph():
    # Simple relay func:
    # 1. y = x + 1
    # 2. return y
    shape = (2, 2)
    x = relay.var("x", shape=shape, dtype="float32")
    y = relay.add(x, relay.ones(shape, dtype="float32"))
    func = relay.Function([x], y)

    # Build 2 exactly same relay modules.
    mod = build_relay_module(func)
    mod_zero_copy = build_relay_module(func)
    x_np = np.random.uniform(size=shape).astype(np.float32)

    # Use set_input()
    x_nd = tvm.nd.array(x_np, device=dev)
    mod.set_input("x", x_nd)
    mod.run()

    # Use set_input_zero_copy()
    x_nd_zero_copy = tvm.nd.array(x_np, device=dev)
    index = mod_zero_copy.get_input_index("x")
    mod_zero_copy.module["set_input_zero_copy"](index, x_nd_zero_copy)
    mod_zero_copy.run()

    # Expect get same output "x".
    testing.assert_allclose(mod.get_output(0).numpy(), mod_zero_copy.get_output(0).numpy())


@testing.requires_llvm
def test_input_in_output():
    # Relay func that input is also in output:
    # 1. y = x + 1
    # 2. return [x, y]
    shape = (3, 4)
    x = relay.var("x", shape=shape, dtype="float32")
    y = relay.add(x, relay.ones(shape, dtype="float32"))
    func = relay.Function([x], relay.expr.Tuple([x, y]))

    # Build 2 exactly same relay modules.
    mod = build_relay_module(func)
    mod_zero_copy = build_relay_module(func)

    x_np = np.random.uniform(size=shape).astype(np.float32)

    # Use set_input()
    x_nd = tvm.nd.array(x_np, device=dev)
    mod.set_input("x", x_nd)
    mod.run()

    # Use set_input_zero_copy()
    x_nd_zero_copy = tvm.nd.array(x_np, device=dev)
    index = mod_zero_copy.get_input_index("x")
    mod_zero_copy.module["set_input_zero_copy"](index, x_nd_zero_copy)
    mod_zero_copy.run()

    # Expect get same output "x".
    testing.assert_allclose(mod.get_output(0).numpy(), mod_zero_copy.get_output(0).numpy())


@testing.requires_llvm
def test_reshape_after_input():
    # Relay func that a reshape op follows immediately after input:
    # 1. y = x + 1
    # 2. return [x, y]
    shape = (3, 4)
    x = relay.var("x", shape=shape, dtype="float32")
    y = relay.reshape(x, (1, 12))
    z = relay.add(y, relay.ones((1, 12), dtype="float32"))
    func = relay.Function([x], relay.expr.Tuple([x, y, z]))

    # Build 2 exactly same relay modules.
    mod = build_relay_module(func)
    mod_zero_copy = build_relay_module(func)

    x_np = np.random.uniform(size=shape).astype(np.float32)

    # Use set_input()
    x_nd = tvm.nd.array(x_np, device=dev)
    mod.set_input("x", x_nd)
    mod.run()

    # Use set_input_zero_copy()
    x_nd_zero_copy = tvm.nd.array(x_np, device=dev)
    index = mod_zero_copy.get_input_index("x")
    mod_zero_copy.module["set_input_zero_copy"](index, x_nd_zero_copy)
    mod_zero_copy.run()

    # Expect get same output "x".
    testing.assert_allclose(mod.get_output(0).numpy(), mod_zero_copy.get_output(0).numpy())
    # Expect get same output "y".
    testing.assert_allclose(mod.get_output(1).numpy(), mod_zero_copy.get_output(1).numpy())


if __name__ == "__main__":
    test_simple_graph()
    test_input_in_output()
    test_reshape_after_input()
