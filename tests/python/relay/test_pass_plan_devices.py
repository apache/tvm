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
import tvm.testing
import numpy as np

CPU = tvm.device("cpu")  # device_type=1
GPU = tvm.device("cuda")  # device_type=2
DEFAULT = GPU

core = tvm.IRModule()
core.import_from_std("core.rly")


def rewrite_and_assert(in_mod, expected_mod):
    """Manually run the pass and assert it's structurally equals to the expected."""
    actual_mod = relay.transform.InferType()(in_mod)
    actual_mod = relay.transform.PlanDevices(DEFAULT)(actual_mod)
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
        compiled = relay.create_executor("vm", mod=in_mod, device=GPU, target="cuda").evaluate()
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
    # Everything defaults to GPU
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);
              %1 = add(%c, %d);
              subtract(%0, %1)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32],
                      param_device_types=[2, 2, 2, 2], result_device_type=2) {
              %0 = add(%a, %b);
              %1 = add(%c, %d);
              subtract(%0, %1)
            }
        """
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.add(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_left_add_on_cpu():
    # Force some args to be on CPU, rest default to GPU.
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);
              %1 = on_device(%0, device_type=1);
              %2 = add(%c, %d);
              subtract(%1, %2)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32],
                      param_device_types=[1, 1, 2, 2], result_device_type=2) {
              %0 = add(%a, %b);
              %1 = on_device(%0, device_type=1, is_fixed=True);
              %2 = device_copy(%1, src_dev_type=1, dst_dev_type=2);
              %3 = add(%c, %d);
              subtract(%2, %3)
            }
        """
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.add(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_left_add_on_cpu_via_copy():
    # As for test_left_add_on_cpu, but with an explicit device_copy.
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);
              %1 = device_copy(%0, src_dev_type=1, dst_dev_type=2);
              %2 = add(%c, %d);
              subtract(%1, %2)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32],
                      param_device_types=[1, 1, 2, 2], result_device_type=2) {
              %0 = add(%a, %b);
              %1 = on_device(%0, device_type=1, is_fixed=True);
              %2 = device_copy(%1, src_dev_type=1, dst_dev_type=2);
              %3 = add(%c, %d);
              subtract(%2, %3)
            }
        """
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.add(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_both_adds_on_cpu():
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);
              %1 = add(%c, %d);
              %2 = on_device(%0, device_type=1);
              %3 = on_device(%1, device_type=1);
              subtract(%2, %3)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32],
                      param_device_types=[1, 1, 1, 1], result_device_type=2) {
              %0 = add(%a, %b);
              %1 = on_device(%0, device_type=1, is_fixed=True);
              %2 = add(%c, %d);
              %3 = on_device(%2, device_type=1, is_fixed=True);
              %4 = device_copy(%1, src_dev_type=1, dst_dev_type=2);
              %5 = device_copy(%3, src_dev_type=1, dst_dev_type=2);
              subtract(%4, %5)
            }
        """
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.add(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_sharing():
    # The same add sub-expression is annotated twice.
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);
              %1 = on_device(%0, device_type=1);
              %2 = on_device(%0, device_type=1);
              subtract(%1, %2)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      param_device_types=[1, 1], result_device_type=2) {
              %0 = add(%a, %b);
              %1 = on_device(%0, device_type=1, is_fixed=True);
              %2 = on_device(%0, device_type=1, is_fixed=True);
              %3 = device_copy(%1, src_dev_type=1, dst_dev_type=2);
              %4 = device_copy(%2, src_dev_type=1, dst_dev_type=2);
              subtract(%3, %4)
            }
        """
        )

    def ref(a, b):
        x = np.add(a, b)
        return np.subtract(x, x)

    exercise(input(), expected(), ref, rands((5, 7), 2))


def test_let_on_cpu():
    # The device for a let-bound expression can flow from uses of the let-bound var.
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              let %l = add(%a, %b);
              let %r = add(%c, %d);
              %0 = on_device(%l, device_type=1);
              subtract(%0, %r)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32],
                      param_device_types=[1, 1, 2, 2], result_device_type=2) {
              %0 = add(%a, %b);
              let %l = on_device(%0, device_type=1, is_fixed=True);
              let %r = add(%c, %d);
              %1 = device_copy(%l, src_dev_type=1, dst_dev_type=2);
              subtract(%1, %r)
            }
        """
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.add(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_func_param_on_cpu():
    # Devices for function parameters flow to call sites.
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              let %f = fn (%x, %y) {
                %0 = add(%x, %y);
                on_device(%0, device_type=1)
              };
              %1 = %f(%a, %b);
              %2 = add(%c, %d);
              subtract(%1, %2)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32],
                      param_device_types=[1, 1, 1, 1], result_device_type=1) {
              let %f = fn (%x, %y, param_device_types=[1, 1], result_device_type=1) {
                add(%x, %y)
              };
              %0 = %f(%a, %b);
              %1 = add(%c, %d);
              subtract(%0, %1)
            }
        """
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.add(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_func_result_on_cpu():
    # Devices for call sites flow to function results.
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              let %f = fn (%x, %y) {
                add(%x, %y)
              };
              %0 = %f(%a, %b);
              %1 = on_device(%0, device_type=1);
              %2 = add(%c, %d);
              subtract(%1, %2)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32],
                      param_device_types=[1, 1, 2, 2], result_device_type=2) {
              let %f = fn (%x, %y, param_device_types=[1, 1], result_device_type=1) {
                add(%x, %y)
              };
              %1 = %f(%a, %b);
              %2 = on_device(%1, device_type=1, is_fixed=True);
              %3 = device_copy(%2, src_dev_type=1, dst_dev_type=2);
              %4 = add(%c, %d);
              subtract(%3, %4)
            }
        """
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.add(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_higher_order():
    # The constraint on %a flows back to %y via %f and %h
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32]) {
              let %f = fn (%g) {
                fn (%a) {
                  %0 = on_device(%a, device_type=1);
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
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32],
                      param_device_types=[2, 1], result_device_type=2) {
              let %f = fn (%g, param_device_types=[2], result_device_type=2) {
                fn (%a, param_device_types=[1], result_device_type=2) {
                  %0 = device_copy(%a, src_dev_type=1, dst_dev_type=2);
                  %1 = %g(%0);
                  add(%1, %x)
                }
              };
              let %h = fn (%b, param_device_types=[2], result_device_type=2) {
                negative(%b)
              };
              %2 = %f(%h);
              %3 = %2(%y);
              subtract(%x, %3)
            }
        """
        )

    def ref(x, y):
        def f(g):
            return lambda a: np.add(g(a), x)

        def h(b):
            return np.negative(b)

        return np.subtract(x, f(h)(y))

    exercise(input(), expected(), ref, rands((5, 7), 2))


def test_function_in_tuple():
    # Since %f ends up in a tuple its argument and result is forced to be on the CPU
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32]) {
              let %f = fn (%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32]) {
                %0 = on_device(%b, device_type=1);
                add(%a, %0)
              };
              let %t = (%f, %x);
              %1 = %t.1;
              %2 = %t.0;
              %2(%1, %y)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"] 
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32],
                      param_device_types=[1, 1], result_device_type=1) {
              let %f = fn (%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                           param_device_types=[1, 1], result_device_type=1) {
                add(%a, %b)
              };
              let %t = (%f, %x);
              %0 = %t.1;
              %1 = %t.0;
              %1(%0, %y)
            }
        """
        )

    def ref(x, y):
        return np.add(x, y)

    exercise(input(), expected(), ref, rands((5, 7), 2))


def test_device_copy():
    const = rand((5, 7))
    metatable = {"relay.Constant": [relay.const(const)]}

    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"] 
            def @main(%x: Tensor[(5, 7), float32]) {
              %0 = device_copy(%x, src_dev_type=1, dst_dev_type=2);
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
            def @main(%x: Tensor[(5, 7), float32], param_device_types=[1], result_device_type=2) {
              %0 = device_copy(%x, src_dev_type=1, dst_dev_type=2);
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


def test_shape_func():
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"] 
            def @main(%x: Tensor[(?), float32], %s: Tensor[(1), int64]) {
              %0 = fn (%y: Tensor[(?), float32]) {
                nn.relu(%y)
              };
              let %p = on_device(%0, device_type=2, is_fixed=True);
              %1 = on_device(%x, device_type=2, is_fixed=True);
              %2 = vm.shape_of(%1, dtype="int64");
              %3 = (%2,);
              %4 = (%s,);
              vm.shape_func(%p, %3, %4, is_input=[False])
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"] 
            def @main(%x: Tensor[(?), float32], %s: Tensor[(1), int64],
                      param_device_types=[2, 1], result_device_type=1) {
              let %p = fn (%y: Tensor[(?), float32], param_device_types=[2], result_device_type=2) {
                nn.relu(%y)
              };
              %1 = vm.shape_of(%x, dtype="int64");
              %2 = (%1,);
              %3 = (%s,);
              vm.shape_func(%p, %2, %3, is_input=[False])
            }
        """
        )

    # Don't try to execute, too fiddly to setup.
    exercise(input(), expected(), None, None)


def test_shape_of():
    # We need to use is_fixed=True in the on_device call so that the tensor will be on the GPU. Otherwise the
    # result defaults to the result device for @main which is the CPU, thus forcing a copy.
    # TODO(mbs): Perhaps the defaulting heuristics are being too clever?
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"] 
            def @main(%x: Tensor[(?, ?), float32]) {
              %0 = on_device(%x, device_type=2, is_fixed=True);
              vm.shape_of(%0, dtype="int64")
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(?, ?), float32], param_device_types=[2], result_device_type=1) {
              vm.shape_of(%x, dtype="int64")
            }
        """
        )

    def ref(x):
        return x.shape

    exercise(input(), expected(), ref, rands((5, 7), 1))


def test_alloc_storage():
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%size: int64, %alignment: int64) {
              memory.alloc_storage(%size, %alignment, device_id=0, device_type=2)
            }
        """,
            "from_string",
            core,
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%size: int64, %alignment: int64, param_device_types=[1, 1], result_device_type=2) {
              memory.alloc_storage(%size, %alignment, device_id=0, device_type=2)
            }
        """,
            "from_string",
            core,
        )

    # Don't try to execute, too fiddly to setup.
    exercise(input(), expected(), None, None)


def test_alloc_tensor():
    shape = np.array([3, 2])
    metatable = {"relay.Constant": [relay.const(shape, dtype="int64")]}

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
            def @main(%sto: Storage[], param_device_types=[2], result_device_type=2) {
              %0 = on_device(0, device_type=1, is_fixed=True);
              %1 = on_device(meta[relay.Constant][0], device_type=1, is_fixed=True);
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
    metatable = {"relay.Constant": [relay.const(newshape, dtype="int64")]}

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
            def @main(%x: Tensor[(2, 8), float32], param_device_types=[2], result_device_type=2) {
              %0 = on_device(meta[relay.Constant][0], device_type=1, is_fixed=True);
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
    # There's nothing special about inferring devices for partially unknown types.
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x0: Tensor[(?, ?), float32], %x1: Tensor[(?, ?), float32]) {
              add(%x0, %x1)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x0: Tensor[(?, ?), float32], %x1: Tensor[(?, ?), float32],
                      param_device_types=[2, 2], result_device_type=2) {
              add(%x0, %x1)
            }
        """
        )

    def ref(x0, x1):
        return np.add(x0, x1)

    exercise(input(), expected(), ref, rands((5, 7), 2))


def test_redundant_annotation():
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32], %z: Tensor[(5, 7), float32]) {
              %0 = add(%x, %y);
              %1 = on_device(%0, device_type=1);
              %2 = subtract(%1, %z);
              %3 = on_device(%0, device_type=1);
              add(%2, %3)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32], %z: Tensor[(5, 7), float32],
                      param_device_types=[1, 1, 2], result_device_type=2) {
              %0 = add(%x, %y);
              %1 = on_device(%0, device_type=1, is_fixed=True);
              %2 = device_copy(%1, src_dev_type=1, dst_dev_type=2);
              %3 = on_device(%0, device_type=1, is_fixed=True);
              %4 = subtract(%2, %z);
              %5 = device_copy(%3, src_dev_type=1, dst_dev_type=2);
              add(%4, %5)
            }
        """
        )

    def ref(x, y, z):
        a = np.add(x, y)
        return np.add(np.subtract(a, z), a)

    exercise(input(), expected(), ref, rands((5, 7), 3))


def test_annotate_expr():
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32], %z: Tensor[(5, 7), float32]) {
              %0 = add(%x, %y);
              %1 = on_device(%0, device_type=2);
              %2 = subtract(%1, %z);
              on_device(%2, device_type=1)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32], %z: Tensor[(5, 7), float32],
                      param_device_types=[2, 2, 1], result_device_type=1) {
              %0 = add(%x, %y);
              %1 = on_device(%0, device_type=2, is_fixed=True);
              %2 = device_copy(%1, src_dev_type=2, dst_dev_type=1);
              subtract(%2, %z)
            }
        """
        )

    def ref(x, y, z):
        return np.subtract(np.add(x, y), z)

    exercise(input(), expected(), ref, rands((5, 7), 3))


def test_annotate_all():
    def input():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32], %z: Tensor[(5, 7), float32]) {
              %0 = add(%x, %y);
              %1 = on_device(%0, device_type=1);
              %2 = subtract(%1, %z);
              on_device(%2, device_type=1)
            }
        """
        )

    def expected():
        return tvm.parser.parse(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32], %z: Tensor[(5, 7), float32],
                      param_device_types=[1, 1, 1], result_device_type=1) {
              %0 = add(%x, %y);
              subtract(%0, %z)
            }
        """
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

    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%data1: Tensor[(1, 64, 56, 56), float32], %data2: Tensor[(1, 64, 56, 56), float32],
                      %weight: Tensor[(64, 64, 3, 3), float32]) {
              %0 = nn.conv2d(%data1, %weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
              %1 = nn.conv2d(%data2, %weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
              %2 = on_device(%0, device_type=1);
              %3 = on_device(%1, device_type=1);
              %4 = add(%2, %3);
              %5 = on_device(%4, device_type=2);
              %6 = nn.conv2d(%5, %weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
              on_device(%6, device_type=1)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%data1: Tensor[(1, 64, 56, 56), float32], %data2: Tensor[(1, 64, 56, 56), float32],
                      %weight: Tensor[(64, 64, 3, 3), float32], param_device_types=[1, 1, 1], result_device_type=1) {
              %0 = nn.conv2d(%data1, %weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
              %1 = on_device(%0, device_type=1, is_fixed=True);
              %2 = nn.conv2d(%data2, %weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
              %3 = on_device(%2, device_type=1, is_fixed=True);
              %4 = device_copy(%1, src_dev_type=1, dst_dev_type=2);
              %5 = device_copy(%3, src_dev_type=1, dst_dev_type=2);
              %6 = add(%4, %5);
              %7 = on_device(%6, device_type=2, is_fixed=True);
              %8 = device_copy(%7, src_dev_type=2, dst_dev_type=1);
              nn.conv2d(%8, %weight, padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3])
            }
        """
        )

    # Don't try to execute, we don't have a reference conv2d
    exercise(input(), expected(), None, None)


def test_tuple_get_item():
    # Note that the device copy should be placed after projection rather than before. This is handled by
    # a heuristic in the pass.
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(3, 3, 4), float32]) {
              let %t = split(%x, indices_or_sections=3);
              %0 = on_device(%t, device_type=1);
              %1 = on_device(%t, device_type=1);
              %2 = %0.0;
              %3 = %1.1;
              %4 = subtract(%2, %3);
              on_device(%4, device_type=2)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(3, 3, 4), float32], param_device_types=[1], result_device_type=2) {
              %0 = split(%x, indices_or_sections=3);
              let %t = on_device(%0, device_type=1, is_fixed=True);
              %1 = %t.0;
              %2 = on_device(%1, device_type=1, is_fixed=True);
              %3 = %t.1;
              %4 = on_device(%3, device_type=1, is_fixed=True);
              %5 = device_copy(%2, src_dev_type=1, dst_dev_type=2);
              %6 = device_copy(%4, src_dev_type=1, dst_dev_type=2);
              subtract(%5, %6)
            }
        """
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

    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32]) {
              %0 = negative(%x);
              %1 = on_device(%0, device_type=1);
              %2 = negative(%1);
              %3 = on_device(%0, device_type=1);
              %4 = negative(%3);
              %5 = on_device(%2, device_type=2);
              %6 = on_device(%4, device_type=2);
              %7 = add(%5, %6);
              %8 = on_device(%7, device_type=2);
              %9 = negative(%8);
              on_device(%9, device_type=1)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], param_device_types=[1], result_device_type=1) {
              %0 = negative(%x);
              %1 = on_device(%0, device_type=1, is_fixed=True);
              %2 = device_copy(%1, src_dev_type=1, dst_dev_type=2);
              %3 = on_device(%0, device_type=1, is_fixed=True);
              %4 = device_copy(%3, src_dev_type=1, dst_dev_type=2);
              %5 = negative(%2);
              %6 = negative(%4);
              %7 = add(%5, %6);
              %8 = on_device(%7, device_type=2, is_fixed=True);
              %9 = device_copy(%8, src_dev_type=2, dst_dev_type=1);
              negative(%9)
            }
        """
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

    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32]) {
              %0 = add(%x, %y);
              %1 = on_device(%0, device_type=2);
              %2 = negative(%1);
              %3 = on_device(%2, device_type=1);
              %4 = negative(%0);
              %5 = add(%3, %4);
              %6 = on_device(%5, device_type=2);
              %7 = negative(%6);
              on_device(%7, device_type=1)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32], param_device_types=[2, 2], result_device_type=1) {
              %0 = add(%x, %y);
              %1 = on_device(%0, device_type=2, is_fixed=True);
              %2 = device_copy(%1, src_dev_type=2, dst_dev_type=1);
              %3 = negative(%2);
              %4 = on_device(%3, device_type=1, is_fixed=True);
              %5 = device_copy(%4, src_dev_type=1, dst_dev_type=2);
              %6 = negative(%0);
              %7 = add(%5, %6);
              %8 = on_device(%7, device_type=2, is_fixed=True);
              %9 = device_copy(%8, src_dev_type=2, dst_dev_type=1);
              negative(%9)
            }
        """
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

    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32]) {
              %0 = add(%a, %b);
              %1 = multiply(%c, %d);
              %2 = on_device(%0, device_type=1);
              %3 = on_device(%1, device_type=2);
              %4 = subtract(%2, %3);
              on_device(%4, device_type=1)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                      %c: Tensor[(5, 7), float32], %d: Tensor[(5, 7), float32],
                      param_device_types=[1, 1, 2, 2], result_device_type=1) {
              %0 = multiply(%c, %d);
              %1 = on_device(%0, device_type=2, is_fixed=True);
              %2 = add(%a, %b);
              %3 = device_copy(%1, src_dev_type=2, dst_dev_type=1);
              subtract(%2, %3)
            }
        """
        )

    def ref(a, b, c, d):
        return np.subtract(np.add(a, b), np.multiply(c, d))

    exercise(input(), expected(), ref, rands((5, 7), 4))


def test_conditional():
    # The conditional is over a function type, thus exercising the first-order/higher-order domain handling.
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: bool, %y: Tensor[(5, 7), float32], %z: Tensor[(5, 7), float32]) {
              let %f = fn (%a) {
                %0 = on_device(%y, device_type=1, is_fixed=True);
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
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: bool, %y: Tensor[(5, 7), float32], %z: Tensor[(5, 7), float32],
                      param_device_types=[1, 1, 1], result_device_type=1) {
              let %f = fn (%a, param_device_types=[1], result_device_type=1) {
                add(%a, %y)
              };
              let %g = fn (%a1, param_device_types=[1], result_device_type=1) {
                subtract(%a1, %y)
              };
              let %h = if (%x) {
                %f
              } else {
                %g
              };
              %h(%z)
            }
        """
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
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @f(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
              %0 = on_device(%b, device_type=1);
              add(%a, %0)
            }
            
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32]) -> Tensor[(5, 7), float32] {
              @f(%y, %x)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @f(%a: Tensor[(5, 7), float32], %b: Tensor[(5, 7), float32],
                   param_device_types=[2, 1], result_device_type=2) -> Tensor[(5, 7), float32] {
              %0 = device_copy(%b, src_dev_type=1, dst_dev_type=2);
              add(%a, %0)
            }
            
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32],
                      param_device_types=[1, 2], result_device_type=2) -> Tensor[(5, 7), float32] {
              @f(%y, %x)
            }
        """
        )

    def ref(x, y):
        def f(a, b):
            return np.add(a, b)

        return f(x, y)

    exercise(input(), expected(), ref, rands((5, 7), 2))


def test_ref():
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32]) {
              let %r = ref(%x);
              %0 = on_device(%y, device_type=1);
              ref_write(%r, %0);
              %1 = ref_read(%r);
              add(%x, %1)
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            def @main(%x: Tensor[(5, 7), float32], %y: Tensor[(5, 7), float32],
                      param_device_types=[2, 1], result_device_type=2) {
              let %r = ref(%x);
              %0 = device_copy(%y, src_dev_type=1, dst_dev_type=2);
              ref_write(%r, %0);
              %1 = ref_read(%r);
              add(%x, %1)
            }
        """
        )

    def ref(x, y):
        r = {"value": x}
        r["value"] = y
        return np.add(x, r["value"])

    # Don't try to execute, no backend currently supports both hetrogeneous devices and references.
    exercise(input(), expected(), None, None)


def test_adt():
    def input():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            type List[A] {
              Cons(A, List[A]),
              Nil,
            }
            def @main(%x : Tensor[(5, 7), float32], %y : Tensor[(5, 7), float32]) {
              %0 = on_device(%y, device_type=1, is_fixed=True);
              %1 = Nil;
              %2 = Cons(%0, %1);
              let %l = Cons(%x, %2);
              match? (%l) {
                Cons(%z, _) => %z
              }
            }
        """
        )

    def expected():
        return tvm.parser.fromtext(
            """
            #[version = "0.0.5"]
            type List[A] {
              Cons(A, List[A]),
              Nil,
            }
            def @main(%x : Tensor[(5, 7), float32], %y : Tensor[(5, 7), float32],
                      param_device_types=[1, 1], result_device_type=1) {
              %0 = Nil;
              %1 = Cons(%y, %0);
              let %l = Cons(%x, %1);
              match? (%l) {
                Cons(%z, _) => %z
              }
            }
        """
        )

    def ref(x, y):
        l = [x, y]
        return l[0]

    exercise(input(), expected(), ref, rands((5, 7), 2))


if __name__ == "__main__":
    import sys
    import pytest

    sys.exit(pytest.main([__file__] + sys.argv[1:]))
