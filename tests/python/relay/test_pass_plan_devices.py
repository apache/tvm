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
# under the License


"""Unit tests for the PlanDevices pass. We check:
    - The pass alone given the expected AST, though we need to manually run InferTypes.
    - The pass is idempotent.
    - Execution on the VM backend yields the correct result."""

import tvm
from tvm import relay
from tvm.script import tir as T
import tvm.testing
import numpy as np
import os

HOST_DEVICE = tvm.device("cpu")
HOST_TARGET = tvm.target.Target("llvm")

CPU_DEVICE = tvm.device("cpu")
CPU_TARGET = tvm.target.Target("llvm").with_host(HOST_TARGET)

GPU_DEVICE = tvm.device("cuda")
GPU_TARGET = tvm.target.Target("cuda").with_host(HOST_TARGET)

TARGETS = [CPU_TARGET, GPU_TARGET]

HOST = tvm.target.VirtualDevice(HOST_DEVICE, HOST_TARGET)  # device_type=1
CPU = tvm.target.VirtualDevice(CPU_DEVICE, CPU_TARGET)  # device_type=1
GPU = tvm.target.VirtualDevice(GPU_DEVICE, GPU_TARGET)  # device_type=2
DEFAULT = GPU

CPU_SCOPE_A = tvm.target.VirtualDevice(CPU_DEVICE, CPU_TARGET, memory_scope="scopeA")
CPU_SCOPE_B = tvm.target.VirtualDevice(CPU_DEVICE, CPU_TARGET, memory_scope="scopeB")

CTXT = tvm.transform.PassContext(config={"relay.fallback_device_type": DEFAULT.device_type_int})

core = tvm.IRModule()
core.import_from_std("core.rly")

recover_virtual_device_map = tvm._ffi.get_global_func("relay.transform.RecoverVirtualDeviceMap")


def rewrite_and_assert(in_mod, expected_mod):
    """Manually run the pass and assert it's structurally equals to the expected."""
    config = tvm.target.make_compilation_config(CTXT, TARGETS)
    actual_mod = relay.transform.InferType()(in_mod)
    actual_mod = relay.transform.PlanDevices(config)(actual_mod)
    actual_mod = relay.transform.InferType()(actual_mod)
    expected_mod = relay.transform.InferType()(expected_mod)
    if not tvm.ir.structural_equal(actual_mod, expected_mod, True):
        # Print everything in full so we can see what's going on when things fail.
        print("Input module:")
        print(in_mod)
        print("Expected module:")
        print(expected_mod)
        print("Actual module:")
        print(actual_mod)
        # Assert again so as to see the actual disagreeing sub-expressions.
        tvm.ir.assert_structural_equal(actual_mod, expected_mod, True)


def eval_and_assert(in_mod: tvm.IRModule, reference_func, args):
    """Test the standard compilation flow gives us a function which agrees with the Numpy
    reference implementation."""
    if not tvm.runtime.enabled("cuda"):
        print("Not evaluating since GPU is not available")
        return
    with tvm.transform.PassContext(opt_level=3):
        compiled = relay.create_executor(
            "vm", mod=in_mod, device=GPU_DEVICE, target=GPU_TARGET
        ).evaluate()
        actual = compiled(*args).numpy()
        expected = reference_func(*args)
        tvm.testing.assert_allclose(actual, expected)


def rand(shape):
    return np.random.rand(*shape).astype("float32")


def rands(shape, n):
    return [rand(shape) for i in range(n)]


def exercise(in_mod: tvm.IRModule, expected_mod: tvm.IRModule, reference_func, args):
    """Test in_mod against expected_mod and reference_func using args."""
    # Correctness
    rewrite_and_assert(in_mod, expected_mod)
    # Idempotence
    rewrite_and_assert(expected_mod, expected_mod)
    # The VM can compile and possibly even run the module
    if not (reference_func is None) and not (args is None):
        eval_and_assert(in_mod, reference_func, args)


def test_plain():
    metatable = {"VirtualDevice": [CPU, GPU]}

    # Everything defaults to GPU
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);
              %1 = add(%c, %d);
              subtract(%0, %1)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], %b {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32],
                      %c {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], %d {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][1]) {
              %0 = add(%a, %b);
              %1 = add(%c, %d);
              subtract(%0, %1)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.add(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_left_add_on_cpu():
    metatable = {"VirtualDevice": [CPU, GPU]}

    # Force some args to be on CPU, rest default to GPU.
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][0]);
              %2 = add(%c, %d);
              subtract(%1, %2)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %b {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      %c {virtual_device= meta[VirtualDevice][1]}: Tensor[(5, 7), float32], %d {virtual_device= meta[VirtualDevice][1]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][1]) {
              %0 = add(%a, %b);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %2 = device_copy(%1, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              %3 = add(%c, %d);
              subtract(%2, %3)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.add(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_left_add_on_cpu_via_copy():
    metatable = {"VirtualDevice": [CPU, GPU]}

    # As for test_left_add_on_cpu, but with an explicit device_copy.
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);
              %1 = device_copy(%0, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              %2 = add(%c, %d);
              subtract(%1, %2)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %b {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      %c {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], %d {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][1]) {
              %0 = add(%a, %b);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %2 = device_copy(%1, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              %3 = add(%c, %d);
              subtract(%2, %3)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.add(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_left_add_on_cpu_via_copy_as_map():
    metatable = {"VirtualDevice": [CPU, GPU]}

    # As for test_left_add_on_cpu, but with an explicit device_copy.
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);
              %1 = device_copy(%0, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              %2 = add(%c, %d);
              subtract(%1, %2)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    config = tvm.target.make_compilation_config(CTXT, TARGETS, HOST_TARGET)
    actual_mod = relay.transform.InferType()(input())
    actual_mod = relay.transform.PlanDevices(config)(actual_mod)
    actual_mod = relay.transform.CapturePostDfsIndexInSpans()(actual_mod)

    # Same expected result as for test_left_add_on_cpu, but we'll include indexes to help
    # the test make sense.
    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], // index 0
                      %b {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], // index 1
                      %c {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], // index 2
                      %d {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], // index 3
                      virtual_device=meta[VirtualDevice][1]) {
              %0 = add(%a, %b);                                                            // index 8
              %1 = on_device(%0,
                             virtual_device=meta[VirtualDevice][0],
                             constrain_result=True);                                       // index 9
              %2 = device_copy(%1,
                               src_virtual_device=meta[VirtualDevice][0],
                               dst_virtual_device=meta[VirtualDevice][1]);                 // index 10
              %3 = add(%c, %d);                                                            // index 11
              subtract(%2, %3)                                                             // index 12
            }                                                                              // index 13
        """,
            "from_string",
            None,
            metatable,
        )

    # Make sure actual matches.
    tvm.ir.assert_structural_equal(actual_mod, expected(), True)

    # Recover all the inferred virtual devices in map form
    raw_map = recover_virtual_device_map(actual_mod, actual_mod["main"])
    # Rewrite the map to be from post-dfs indexes to device types
    map = {e.span.line: d.device_type for e, d in raw_map.items()}
    # Now we can express the expected map
    expected_map = {
        0: CPU.device_type,  # %a
        1: CPU.device_type,  # %b
        2: GPU.device_type,  # %c
        3: GPU.device_type,  # %d
        8: CPU.device_type,  # first add
        9: CPU.device_type,  # on_device
        10: GPU.device_type,  # device_copy
        11: GPU.device_type,  # second add
        12: GPU.device_type,  # subtract
        13: GPU.device_type,  # @main
    }
    assert map == expected_map


def test_both_adds_on_cpu():
    metatable = {"VirtualDevice": [CPU, GPU]}

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);
              %1 = add(%c, %d);
              %2 = on_device(%0, virtual_device=meta[VirtualDevice][0]);
              %3 = on_device(%1, virtual_device=meta[VirtualDevice][0]);
              subtract(%2, %3)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %b {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      %c {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %d {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][1]) {
              %0 = add(%a, %b);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %2 = add(%c, %d);
              %3 = on_device(%2, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %4 = device_copy(%1, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              %5 = device_copy(%3, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              subtract(%4, %5)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.add(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_sharing():
    metatable = {"VirtualDevice": [CPU, GPU]}

    # The same add sub-expression is annotated twice.
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][0]);
              %2 = on_device(%0, virtual_device=meta[VirtualDevice][0]);
              subtract(%1, %2)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %b {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][1]) {
              %0 = add(%a, %b);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %2 = on_device(%0, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %3 = device_copy(%1, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              %4 = device_copy(%2, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              subtract(%3, %4)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(a, b):
        x = np.add(a, b)
        return np.subtract(x, x)

    exercise(input(), expected(), ref, rands((5, 7), 2))


def test_let_on_cpu():
    metatable = {"VirtualDevice": [CPU, GPU]}

    # The device for a let-bound expression can flow from uses of the let-bound var.
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              let %l = add(%a, %b);
              let %r = add(%c, %d);
              %0 = on_device(%l, virtual_device=meta[VirtualDevice][0]);
              subtract(%0, %r)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %b {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      %c {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], %d {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][1]) {
              %0 = add(%a, %b);
              let %l = on_device(%0, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              let %r = on_device(add(%c, %d), virtual_device=meta[VirtualDevice][1], constrain_result=True);
              %1 = device_copy(%l, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              subtract(%1, %r)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.add(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_func_param_on_cpu():
    metatable = {"VirtualDevice": [CPU, GPU]}

    # Devices for function parameters flow to call sites.
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              let %f = fn (%x, %y) {
                %0 = add(%x, %y);
                on_device(%0, virtual_device=meta[VirtualDevice][0])
              };
              %1 = %f(%a, %b);
              %2 = add(%c, %d);
              subtract(%1, %2)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %b {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      %c {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %d {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][0]) {
              let %f = fn (%x {virtual_device=meta[VirtualDevice][0]}, %y {virtual_device=meta[VirtualDevice][0]},
                           virtual_device=meta[VirtualDevice][0]) {
                add(%x, %y)
              };
              %0 = %f(%a, %b);
              %1 = add(%c, %d);
              subtract(%0, %1)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.add(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_func_result_on_cpu():
    metatable = {"VirtualDevice": [CPU, GPU]}

    # Devices for call sites flow to function results.
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              let %f = fn (%x, %y) {
                add(%x, %y)
              };
              %0 = %f(%a, %b);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][0]);
              %2 = add(%c, %d);
              subtract(%1, %2)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %b {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      %c {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], %d {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][1]) {
              let %f = fn (%x {virtual_device=meta[VirtualDevice][0]}, %y {virtual_device=meta[VirtualDevice][0]},
                           virtual_device=meta[VirtualDevice][0]) {
                add(%x, %y)
              };
              %1 = %f(%a, %b);
              %2 = on_device(%1, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %3 = device_copy(%2, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              %4 = add(%c, %d);
              subtract(%3, %4)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.add(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_higher_order():
    metatable = {"VirtualDevice": [CPU, GPU]}

    # The constraint on %a flows back to %y via %f and %h
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32]) {
              let %f = fn (%g) {
                fn (%a) {
                  %0 = on_device(%a, virtual_device=meta[VirtualDevice][0]);
                  %1 = %g(%0);
                  add(%1, %x)
                }
              };
              let %h = fn (%b) {
                negative(%b)
              };
              %2 = %f(%h);
              %3 = %2(%y);
              subtract(%x, %3)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], %y {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][1]) {
              let %f = fn (%g {virtual_device=meta[VirtualDevice][1]}, virtual_device=meta[VirtualDevice][1]) {
                fn (%a {virtual_device=meta[VirtualDevice][0]}, virtual_device=meta[VirtualDevice][1]) {
                  %0 = device_copy(%a, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
                  %1 = %g(%0);
                  add(%1, %x)
                }
              };
              let %h = fn (%b  {virtual_device=meta[VirtualDevice][1]}, virtual_device=meta[VirtualDevice][1]) {
                negative(%b)
              };
              %2 = %f(%h);
              %3 = %2(%y);
              subtract(%x, %3)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x, y):
        def f(g):
            return lambda a: np.add(g(a), x)

        def h(b):
            return np.negative(b)

        return np.subtract(x, f(h)(y))

    exercise(input(), expected(), ref, rands((5, 7), 2))


def test_function_in_tuple():
    metatable = {"VirtualDevice": [CPU, GPU]}

    # Since %f ends up in a tuple its argument and result is forced to be on the CPU
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32]) {
              let %f = fn (%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32]) {
                %0 = on_device(%b, virtual_device=meta[VirtualDevice][0]);
                add(%a, %0)
              };
              let %t = (%f, %x);
              %1 = %t.1;
              %2 = %t.0;
              %2(%1, %y)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %y {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][0]) {
              let %f = fn (%a {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %b {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                           virtual_device=meta[VirtualDevice][0]) {
                add(%a, %b)
              };
              let %t = on_device((%f, %x), virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %0 = %t.1;
              %1 = %t.0;
              %1(%0, %y)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x, y):
        return np.add(x, y)

    exercise(input(), expected(), ref, rands((5, 7), 2))


def test_device_copy():
    const = rand((5, 7))
    metatable = {"VirtualDevice": [CPU, GPU], "relay.Constant": [relay.const(const)]}

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32]) {
              %0 = device_copy(%x, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              add(%0, meta[relay.Constant][0])
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][1]) {
              %0 = device_copy(%x, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              add(%0, meta[relay.Constant][0])
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x):
        return np.add(x, const)

    exercise(input(), expected(), ref, rands((5, 7), 1))


def test_shape_of():
    metatable = {"VirtualDevice": [HOST, GPU]}

    # We need to use constrain_result=True in the on_device call so that the tensor will be on the GPU. Otherwise the
    # result defaults to the result device for @main which is the CPU, thus forcing a copy.
    # TODO(mbs): Perhaps the defaulting heuristics are being too clever?
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(?, ?), float32]) {
              %0 = on_device(%x, virtual_device=meta[VirtualDevice][1], constrain_result=True);
              vm.shape_of(%0, dtype="int64")
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x {virtual_device=meta[VirtualDevice][1]}: Tensor[(?, ?), float32],
                      virtual_device=meta[VirtualDevice][0]) {
              vm.shape_of(%x, dtype="int64")
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x):
        return x.shape

    exercise(input(), expected(), ref, rands((5, 7), 1))


def test_alloc_storage():
    metatable = {"VirtualDevice": [HOST, GPU]}

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%size: int64, %alignment: int64) {
              memory.alloc_storage(%size, %alignment, virtual_device=meta[VirtualDevice][1])
            }
        """,
            "from_string",
            core,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%size {virtual_device=meta[VirtualDevice][0]}: int64, %alignment {virtual_device=meta[VirtualDevice][0]}: int64,
                      virtual_device=meta[VirtualDevice][1]) {
              memory.alloc_storage(%size, %alignment, virtual_device=meta[VirtualDevice][1])
            }
        """,
            "from_string",
            core,
            metatable,
        )

    # Don't try to execute, too fiddly to setup.
    exercise(input(), expected(), None, None)


def test_alloc_tensor():
    shape = np.array([3, 2])
    metatable = {
        "VirtualDevice": [HOST, GPU],
        "relay.Constant": [relay.const(shape, dtype="int64")],
    }

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%sto: Storage[]) {
              memory.alloc_tensor(%sto, 0, meta[relay.Constant][0],
                                  const_shape=meta[relay.Constant][0], assert_shape=[])
            }
        """,
            "from_string",
            core,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%sto {virtual_device=meta[VirtualDevice][1]}: Storage[], virtual_device=meta[VirtualDevice][1]) {
              %0 = on_device(0, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %1 = on_device(meta[relay.Constant][0], virtual_device=meta[VirtualDevice][0], constrain_result=True);
              memory.alloc_tensor(%sto, %0, %1, const_shape=meta[relay.Constant][0], assert_shape=[])
            }
        """,
            "from_string",
            core,
            metatable,
        )

    # Don't try to execute, too fiddly to setup.
    exercise(input(), expected(), None, None)


def test_reshape_tensor():
    newshape = [2, 4, 2]
    metatable = {
        "VirtualDevice": [HOST, GPU],
        "relay.Constant": [relay.const(newshape, dtype="int64")],
    }

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(2, 8), float32]) {
              vm.reshape_tensor(%x, meta[relay.Constant][0], newshape=[2, 4, 2])
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x {virtual_device=meta[VirtualDevice][1]}: Tensor[(2, 8), float32],
                      virtual_device=meta[VirtualDevice][1]) {
              %0 = on_device(meta[relay.Constant][0], virtual_device=meta[VirtualDevice][0], constrain_result=True);
              vm.reshape_tensor(%x, %0, newshape=[2, 4, 2])
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x):
        return np.reshape(x, newshape)

    exercise(input(), expected(), ref, rands((2, 8), 1))


def test_dynamic_input():
    metatable = {"VirtualDevice": [GPU]}

    # There's nothing special about inferring devices for partially unknown types.
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x0: Tensor[(?, ?), float32], %x1: Tensor[(?, ?), float32]) {
              add(%x0, %x1)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x0 {virtual_device=meta[VirtualDevice][0]}: Tensor[(?, ?), float32], %x1 {virtual_device=meta[VirtualDevice][0]}: Tensor[(?, ?), float32],
                      virtual_device=meta[VirtualDevice][0]) {
              add(%x0, %x1)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x0, x1):
        return np.add(x0, x1)

    exercise(input(), expected(), ref, rands((5, 7), 2))


def test_redundant_annotation():
    metatable = {"VirtualDevice": [CPU, GPU]}

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32], %z: Tensor[(5, 7), float32]) {
              %0 = add(%x, %y);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][0]);
              %2 = subtract(%1, %z);
              %3 = on_device(%0, virtual_device=meta[VirtualDevice][0]);
              add(%2, %3)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %y {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %z {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][1]) {
              %0 = add(%x, %y);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %2 = device_copy(%1, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              %3 = on_device(%0, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %4 = subtract(%2, %z);
              %5 = device_copy(%3, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              add(%4, %5)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x, y, z):
        a = np.add(x, y)
        return np.add(np.subtract(a, z), a)

    exercise(input(), expected(), ref, rands((5, 7), 3))


def test_annotate_expr():
    metatable = {"VirtualDevice": [CPU, GPU]}

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32], %z: Tensor[(5, 7), float32]) {
              %0 = add(%x, %y);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][1]);
              %2 = subtract(%1, %z);
              on_device(%2, virtual_device=meta[VirtualDevice][0])
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], %y {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], %z {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][0]) {
              %0 = add(%x, %y);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][1], constrain_result=True);
              %2 = device_copy(%1, src_virtual_device=meta[VirtualDevice][1], dst_virtual_device=meta[VirtualDevice][0]);
              subtract(%2, %z)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x, y, z):
        return np.subtract(np.add(x, y), z)

    exercise(input(), expected(), ref, rands((5, 7), 3))


def test_annotate_all():
    metatable = {"VirtualDevice": [CPU, GPU]}

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32], %z: Tensor[(5, 7), float32]) {
              %0 = add(%x, %y);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][0]);
              %2 = subtract(%1, %z);
              on_device(%2, virtual_device=meta[VirtualDevice][0])
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %y {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %z {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][0]) {
              %0 = add(%x, %y);
              subtract(%0, %z)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x, y, z):
        return np.subtract(np.add(x, y), z)

    exercise(input(), expected(), ref, rands((5, 7), 3))


def test_conv_network():
    r"""The network and devices are as follows:
    data1     data2    <--- CPU
      |         |
    conv2d    conv2d   <--- CPU
       \       /
        \     /
          add          <--- GPU
           |
         conv2d        <--- CPU
           |
        <result>       <--- CPU
    """
    metatable = {"VirtualDevice": [CPU, GPU]}

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%data1: Tensor[(1, 64, 56, 56), float32], %data2: Tensor[(1, 64, 56, 56), float32],
                      %weight: Tensor[(64, 64, 3, 3), float32]) {
              %0 = nn.conv2d(%data1, %weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
              %1 = nn.conv2d(%data2, %weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
              %2 = on_device(%0, virtual_device=meta[VirtualDevice][0]);
              %3 = on_device(%1, virtual_device=meta[VirtualDevice][0]);
              %4 = add(%2, %3);
              %5 = on_device(%4, virtual_device=meta[VirtualDevice][1]);
              %6 = nn.conv2d(%5, %weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
              on_device(%6, virtual_device=meta[VirtualDevice][0])
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%data1 {virtual_device=meta[VirtualDevice][0]}: Tensor[(1, 64, 56, 56), float32], %data2 {virtual_device=meta[VirtualDevice][0]}: Tensor[(1, 64, 56, 56), float32],
                      %weight {virtual_device=meta[VirtualDevice][0]}: Tensor[(64, 64, 3, 3), float32],
                      virtual_device=meta[VirtualDevice][0]) {
              %0 = nn.conv2d(%data1, %weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %2 = nn.conv2d(%data2, %weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
              %3 = on_device(%2, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %4 = device_copy(%1, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              %5 = device_copy(%3, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              %6 = add(%4, %5);
              %7 = on_device(%6, virtual_device=meta[VirtualDevice][1], constrain_result=True);
              %8 = device_copy(%7, src_virtual_device=meta[VirtualDevice][1], dst_virtual_device=meta[VirtualDevice][0]);
              nn.conv2d(%8, %weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
            }
        """,
            "from_string",
            None,
            metatable,
        )

    # Don't try to execute, we don't have a reference conv2d
    exercise(input(), expected(), None, None)


def test_tuple_get_item():
    metatable = {"VirtualDevice": [CPU, GPU]}

    # Note that the device copy should be placed after projection rather than before. This is handled by
    # a heuristic in the pass.
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(3, 3, 4), float32]) {
              let %t = split(%x, indices_or_sections=3);
              %0 = on_device(%t, virtual_device=meta[VirtualDevice][0]);
              %1 = on_device(%t, virtual_device=meta[VirtualDevice][0]);
              %2 = %0.0;
              %3 = %1.1;
              %4 = subtract(%2, %3);
              on_device(%4, virtual_device=meta[VirtualDevice][1])
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x {virtual_device=meta[VirtualDevice][0]}: Tensor[(3, 3, 4), float32],
                      virtual_device=meta[VirtualDevice][1]) {
              %0 = split(%x, indices_or_sections=3);
              let %t = on_device(%0, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %1 = %t.0;
              %2 = on_device(%1, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %3 = %t.1;
              %4 = on_device(%3, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %5 = device_copy(%2, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              %6 = device_copy(%4, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              subtract(%5, %6)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x):
        t = np.split(x, 3)
        return np.subtract(t[0], t[1])

    exercise(input(), expected(), ref, rands((3, 3, 4), 1))


def test_propogation():
    r""" The network and devices are as follows:
                  x             <--- CPU
                  |
                negative        <--- CPU
                /   \
          negative  negative    <--- GPU
                \   /
                 add            <--- GPU
                  |
                negative        <--- CPU
                  |
               <result>         <--- CPU
    """
    metatable = {"VirtualDevice": [CPU, GPU]}

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32]) {
              %0 = negative(%x);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][0]);
              %2 = negative(%1);
              %3 = on_device(%0, virtual_device=meta[VirtualDevice][0]);
              %4 = negative(%3);
              %5 = on_device(%2, virtual_device=meta[VirtualDevice][1]);
              %6 = on_device(%4, virtual_device=meta[VirtualDevice][1]);
              %7 = add(%5, %6);
              %8 = on_device(%7, virtual_device=meta[VirtualDevice][1]);
              %9 = negative(%8);
              on_device(%9, virtual_device=meta[VirtualDevice][0])
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][0]) {
              %0 = negative(%x);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %2 = device_copy(%1, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              %3 = on_device(%0, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %4 = device_copy(%3, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              %5 = negative(%2);
              %6 = negative(%4);
              %7 = add(%5, %6);
              %8 = on_device(%7, virtual_device=meta[VirtualDevice][1], constrain_result=True);
              %9 = device_copy(%8, src_virtual_device=meta[VirtualDevice][1], dst_virtual_device=meta[VirtualDevice][0]);
              negative(%9)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x):
        y = np.negative(x)
        return np.negative(np.add(np.negative(y), np.negative(y)))

    exercise(input(), expected(), ref, rands((5, 7), 1))


def test_fusible_network():
    r""" The network is as follows:
               x     y      <--- GPU
                \   /
                 add        <--- GPU
                /   \
           negative  \      <--- CPU
              \       \
               \  negative  <--- GPU
                \   /
                 add        <--- GPU
                  |
               negative     <--- CPU
                  |
               <result>     <--- CPU
    """
    metatable = {"VirtualDevice": [CPU, GPU]}

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32]) {
              %0 = add(%x, %y);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][1]);
              %2 = negative(%1);
              %3 = on_device(%2, virtual_device=meta[VirtualDevice][0]);
              %4 = negative(%0);
              %5 = add(%3, %4);
              %6 = on_device(%5, virtual_device=meta[VirtualDevice][1]);
              %7 = negative(%6);
              on_device(%7, virtual_device=meta[VirtualDevice][0])
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], %y {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][0]) {
              %0 = add(%x, %y);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][1], constrain_result=True);
              %2 = device_copy(%1, src_virtual_device=meta[VirtualDevice][1], dst_virtual_device=meta[VirtualDevice][0]);
              %3 = negative(%2);
              %4 = on_device(%3, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %5 = device_copy(%4, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              %6 = negative(%0);
              %7 = add(%5, %6);
              %8 = on_device(%7, virtual_device=meta[VirtualDevice][1], constrain_result=True);
              %9 = device_copy(%8, src_virtual_device=meta[VirtualDevice][1], dst_virtual_device=meta[VirtualDevice][0]);
              negative(%9)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x, y):
        z = np.add(x, y)
        return np.negative(np.add(np.negative(z), np.negative(z)))

    exercise(input(), expected(), ref, rands((5, 7), 2))


def test_unpropagatable_graph():
    r"""The network is as follows:
    a      b            <--- CPU
    \     /
     \   /   c     d    <--- GPU
      \ /    \     /
      add     \   /     <--- CPU
       \       \ /
        \    multiply   <--- GPU
         \     /
        subtract        <--- CPU
           |
        <result>        <--- CPU
    """
    metatable = {"VirtualDevice": [CPU, GPU]}

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);
              %1 = multiply(%c, %d);
              %2 = on_device(%0, virtual_device=meta[VirtualDevice][0]);
              %3 = on_device(%1, virtual_device=meta[VirtualDevice][1]);
              %4 = subtract(%2, %3);
              on_device(%4, virtual_device=meta[VirtualDevice][0])
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%a {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %b {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      %c {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], %d {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][0]) {
              %0 = multiply(%c, %d);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][1], constrain_result=True);
              %2 = add(%a, %b);
              %3 = device_copy(%1, src_virtual_device=meta[VirtualDevice][1], dst_virtual_device=meta[VirtualDevice][0]);
              subtract(%2, %3)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.multiply(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_conditional():
    metatable = {"VirtualDevice": [CPU, GPU]}

    # The conditional is over a function type, thus exercising the first-order/higher-order domain handling.
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x: bool, %y: Tensor[(5, 7), float32], %z: Tensor[(5, 7), float32]) {
              let %f = fn (%a) {
                %0 = on_device(%y, virtual_device=meta[VirtualDevice][0], constrain_result=True);
                add(%a, %0)
              };
              let %g = fn (%a1) {
                subtract(%a1, %y)
              };
              let %h = if (%x) {
                %f
              } else {
                %g
              };
              %h(%z)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x {virtual_device=meta[VirtualDevice][0]}: bool, %y {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %z {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][0]) {
              let %f = fn (%a {virtual_device=meta[VirtualDevice][0]}, virtual_device=meta[VirtualDevice][0]) {
                add(%a, %y)
              };
              let %g = fn (%a1 {virtual_device=meta[VirtualDevice][0]}, virtual_device=meta[VirtualDevice][0]) {
                subtract(%a1, %y)
              };
              let %h = on_device(if (%x) {
                %f
              } else {
                %g
              }, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %h(%z)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x, y, z):
        def f(a):
            return np.add(a, y)

        def g(a):
            return np.subtract(a, y)

        h = f if x else g
        return h(z)

    exercise(input(), expected(), ref, [True, rand((5, 7)), rand((5, 7))])


def test_global():
    metatable = {"VirtualDevice": [CPU, GPU]}

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @f(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
              %0 = on_device(%b, virtual_device=meta[VirtualDevice][0]);
              add(%a, %0)
            }

            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
              @f(%y, %x)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @f(%a {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], %b {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                   virtual_device=meta[VirtualDevice][1]) -> Tensor[(5, 7), float32] {
              %0 = device_copy(%b, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              add(%a, %0)
            }

            def @main(%x {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %y {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][1]) -> Tensor[(5, 7), float32] {
              @f(%y, %x)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x, y):
        def f(a, b):
            return np.add(a, b)

        return f(x, y)

    exercise(input(), expected(), ref, rands((5, 7), 2))


def test_ref():
    metatable = {"VirtualDevice": [CPU, GPU]}

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32]) {
              let %r = ref(%x);
              %0 = on_device(%y, virtual_device=meta[VirtualDevice][0]);
              ref_write(%r, %0);
              %1 = ref_read(%r);
              add(%x, %1)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], %y {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][1]) {
              let %r = on_device(ref(%x), virtual_device=meta[VirtualDevice][1], constrain_result=True);
              %0 = device_copy(%y, src_virtual_device=meta[VirtualDevice][0], dst_virtual_device=meta[VirtualDevice][1]);
              on_device(ref_write(%r, %0), virtual_device=meta[VirtualDevice][1], constrain_result=True);
              %1 = ref_read(%r);
              add(%x, %1)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x, y):
        r = {"value": x}
        r["value"] = y
        return np.add(x, r["value"])

    # Don't try to execute, no backend currently supports both hetrogeneous devices and references.
    exercise(input(), expected(), None, None)


def test_adt():
    metatable = {"VirtualDevice": [CPU, GPU]}

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            type List[A] {
              Cons(A, List[A]),
              Nil,
            }
            def @main(%x : Tensor[(5, 7), float32], %y : Tensor[(5, 7), float32]) {
              %0 = on_device(%y, virtual_device=meta[VirtualDevice][0], constrain_result=True);
              %1 = Nil;
              %2 = Cons(%0, %1);
              let %l = Cons(%x, %2);
              match? (%l) {
                Cons(%z, _) => %z
              }
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            type List[A] {
              Cons(A, List[A]),
              Nil,
            }
            def @main(%x {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %y {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][0]) {
              %0 = Nil;
              %1 = Cons(%y, %0);
              let %l = on_device(Cons(%x, %1), virtual_device=meta[VirtualDevice][0], constrain_result=True);
              match? (%l) {
                Cons(%z, _) => %z
              }
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def ref(x, y):
        l = [x, y]
        return l[0]

    exercise(input(), expected(), ref, rands((5, 7), 2))


def test_free_on_device():
    """Tests that the 'free' form of on_device (ie with constrain_body=False) can be used to allow
    a device_copy to be inserted if necessary, but otherwise does not prevent the flow of
    device information."""
    metatable = {
        "VirtualDevice": [
            CPU,  # no memory scope constraint
            CPU_SCOPE_A,  # constrain to scopeA
            CPU_SCOPE_B,
        ]
    }  # constrain to scopeB

    # Everything defaults to GPU
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @on_scope_b(%x {virtual_device=meta[VirtualDevice][2]}: Tensor[(5, 7), float32],
                            virtual_device=meta[VirtualDevice][2]) -> Tensor[(5, 7), float32] {
              %x
            }
            def @main(%a {virtual_device=meta[VirtualDevice][0]}: Tensor[(5, 7), float32], %b {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], %c {virtual_device=meta[VirtualDevice][2]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][1]) {
              // %a's memory scope is unconstrained, so will take on "scopeB" and on_device has no effect
              %0 = @on_scope_b(on_device(%a, virtual_device=meta[VirtualDevice][0], constrain_body=False));
              // %b's memory scope is "scopeA", so will require a "scopeA"->"scopeB" copy.
              %1 = @on_scope_b(on_device(%b, virtual_device=meta[VirtualDevice][0], constrain_body=False));
              // %c's memory scope is "scopeB", so no copy required.
              %2 = @on_scope_b(on_device(%c, virtual_device=meta[VirtualDevice][0], constrain_body=False));
              // result's memory scope is is on "scopeA", so will require a "scopeB"->"scopeA" copy.
              %3 = add(add(%0, %1), %2);
              on_device(%3, virtual_device=meta[VirtualDevice][0], constrain_body=False)
            }
        """,
            "from_string",
            None,
            metatable,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @on_scope_b(%x {virtual_device=meta[VirtualDevice][2]}: Tensor[(5, 7), float32],
                            virtual_device=meta[VirtualDevice][2]) -> Tensor[(5, 7), float32] {
              %x
            }
            def @main(%a {virtual_device=meta[VirtualDevice][2]}: Tensor[(5, 7), float32], %b {virtual_device=meta[VirtualDevice][1]}: Tensor[(5, 7), float32], %c {virtual_device=meta[VirtualDevice][2]}: Tensor[(5, 7), float32],
                      virtual_device=meta[VirtualDevice][1]) {
              %0 = @on_scope_b(%a);
              %1 = device_copy(%b, src_virtual_device=meta[VirtualDevice][1], dst_virtual_device=meta[VirtualDevice][2]);
              %2 = @on_scope_b(%1);
              %3 = @on_scope_b(%c);
              %4 = add(add(%0, %2), %3);
              %5 = on_device(%4, virtual_device=meta[VirtualDevice][2], constrain_result=True);
              device_copy(%5, src_virtual_device=meta[VirtualDevice][2], dst_virtual_device=meta[VirtualDevice][1])
            }
        """,
            "from_string",
            None,
            metatable,
        )

    exercise(input(), expected(), None, None)


def test_lowered():
    """
    Tests propagation of memory scopes from PrimFuncs and insertion
    of device_copies to mediate any scope changes.
    """

    @T.prim_func
    def input_gem(a: T.handle, b: T.handle, c: T.handle, d: T.handle) -> None:
        A = T.match_buffer(a, [128, 128], scope="scopeA")  # will flow out
        B = T.match_buffer(b, [128, 128], scope="")  # will flow in
        C = T.match_buffer(c, [128, 128], scope="scopeB")  # will flow out
        D = T.match_buffer(d, [128, 128], scope="scopeA")  # will flow out

        for i, j, k in T.grid(128, 128, 128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    D[vi, vj] = C[vi, vj]
                D[vi, vj] = D[vi, vj] + A[vi, vk] * B[vj, vk]

    @T.prim_func
    def expected_gem(a: T.handle, b: T.handle, c: T.handle, d: T.handle) -> None:
        A = T.match_buffer(a, [128, 128], scope="scopeA")
        B = T.match_buffer(b, [128, 128], scope="scopeB")  # flowed in
        C = T.match_buffer(c, [128, 128], scope="scopeB")
        D = T.match_buffer(d, [128, 128], scope="scopeA")

        for i, j, k in T.grid(128, 128, 128):
            with T.block("update"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    D[vi, vj] = C[vi, vj]
                D[vi, vj] = D[vi, vj] + A[vi, vk] * B[vj, vk]

    metatable = {
        "VirtualDevice": [
            CPU,  # meta[VirtualDevice][0], no memory scope
            CPU_SCOPE_A,  # meta[VirtualDevice][1], "scopeA"
            CPU_SCOPE_B,
        ]
    }  # meta[VirtualDevice][2], "scopeB"
    gem_ty = relay.FuncType(
        [
            relay.TensorType((128, 128), "float32"),
            relay.TensorType((128, 128), "float32"),
            relay.TensorType((128, 128), "float32"),
        ],
        relay.TensorType((128, 128), "float32"),
    )
    gem_gv = relay.GlobalVar("gem", type_annot=gem_ty)

    def input():
        mod = tvm.ir.IRModule()
        mod[gem_gv] = input_gem
        # - %x on CPU, no memory scope constraint, so will be constrained by first param of gem to "scopeA".
        # - %y on CPU "scopeB", so will flow in to second param of gem.
        # - %z on CPU "scopeA", so will clash with third param of gem and will need device_copy.
        # - result on CPU "scopeB", but result of gem on "scopeA" so will need device_copy
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x {virtual_device=meta[VirtualDevice][0]}: Tensor[(128, 128), float32],
                      %y {virtual_device=meta[VirtualDevice][2]}: Tensor[(128, 128), float32],
                      %z {virtual_device=meta[VirtualDevice][1]}: Tensor[(128, 128), float32],
                      virtual_device=meta[VirtualDevice][2]) {
              call_lowered(@gem, (%x, %y, %z))
            }
            """,
            "from_string",
            mod,
            metatable,
        )

    def expected():
        mod = tvm.ir.IRModule()
        mod[gem_gv] = expected_gem
        # - %x now on CPU "scopeA", no device_copy needed.
        # - %y still on CPU "scopeB", no device_copy needed.
        # - %z still on CPU "scopeA", needs device_copy to "scopeB".
        # - result still on CPU "scopeB", needs device_copy  from "scopeA".
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x {virtual_device=meta[VirtualDevice][1]}: Tensor[(128, 128), float32],
                      %y {virtual_device=meta[VirtualDevice][2]}: Tensor[(128, 128), float32],
                      %z {virtual_device=meta[VirtualDevice][1]}: Tensor[(128, 128), float32],
                      virtual_device=meta[VirtualDevice][2]) {
              %0 = device_copy(%z, src_virtual_device=meta[VirtualDevice][1], dst_virtual_device=meta[VirtualDevice][2]);
              %1 = on_device(%0, virtual_device=meta[VirtualDevice][2], constrain_result=True);
              %2 = call_lowered(@gem, (%x, %y, %1));
              %3 = on_device(%2, virtual_device=meta[VirtualDevice][1], constrain_result=True);
              device_copy(%3, src_virtual_device=meta[VirtualDevice][1], dst_virtual_device=meta[VirtualDevice][2])
            }
            """,
            "from_string",
            mod,
            metatable,
        )

    exercise(input(), expected(), None, None)


def test_stack_overflow():
    metatable = {"VirtualDevice": [CPU, GPU]}

    # Everything defaults to GPU
    def input():
        tmp = "test_stack_overflow_input.txt"
        mod = """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
            %0 = add(%a, %b);
            %1 = add(%c, %d);
            """

        end = 1555
        for i in range(2, end):
            s1 = "\n\t" + "%" + str(i) + " = add(%" + str(i - 1) + ", %" + str(i - 2) + ");"
            mod += s1
        mod += "\n\t" + "add(%" + str(end - 1) + ", %" + str(end - 2) + ")"
        mod += "\n\t}"

        return tvm.parser.parse(
            mod,
            "from_string",
            None,
            metatable,
        )

    config = tvm.target.make_compilation_config(CTXT, TARGETS)
    actual_mod = relay.transform.InferType()(input())
    actual_mod = relay.transform.PlanDevices(config)(actual_mod)
    relay.transform.InferType()(actual_mod)


def test_primitive():
    """Annotations on Primitive functions should be accepted, even though the body
    of the Primitive function is not considered during PlanDevices."""
    global_virtual_device = tvm.target.VirtualDevice(memory_scope="global")
    texture_virtual_device = tvm.target.VirtualDevice(memory_scope="global.texture")
    metatable = {
        "VirtualDevice": [
            global_virtual_device,
            texture_virtual_device,
        ]
    }

    mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%data1: Tensor[(1, 32, 40, 40), float32],
                  %data2: Tensor[(1, 32, 40, 40), float32]) {
          %0 = fn (%a, Primitive=1) {
            layout_transform(%a, src_layout="NCHW", dst_layout="NCHW4c")
          };
          %1 = %0(%data1);
          %3 = %0(%data2);
          %5 = fn (%a {virtual_device=meta[VirtualDevice][0]},  // global
                   %b {virtual_device=meta[VirtualDevice][0]},  // global
                   virtual_device=meta[VirtualDevice][1],       // texture
                   Primitive=1) {
            add(%a, %b)
          };
          %6 = %5(%1, %3);
          %10 = fn (%a,
                    virtual_device=meta[VirtualDevice][0],      // global
                    Primitive=1) {
            layout_transform(%a, src_layout="NCHW4c", dst_layout="NCHW")
          };
          %10(%6)
        }
        """,
        "from_string",
        None,
        metatable,
    )
    print(mod)

    config = tvm.target.make_compilation_config(CTXT, GPU_TARGET)
    mod = relay.transform.InferType()(mod)
    # PlanDevices should succeed.
    mod = relay.transform.PlanDevices(config)(mod)
    print(mod)


if __name__ == "__main__":
    tvm.testing.main()
