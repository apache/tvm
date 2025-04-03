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

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.script import ir as I, tir as T, relax as R


class BaseCompare(tvm.testing.CompareBeforeAfter):
    transform = tvm.relax.transform.InjectDebugCallback()


class TestSimple(BaseCompare):
    """The debug callback is called after each variable binding"""

    @I.ir_module
    class Before:
        @R.function
        def main():
            A = R.const([1.0, 2.0], "float64")
            B = R.const([3.0, 4.0], "float64")
            C = A + B
            return C

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(debug_callback: R.Callable(ret=R.Tuple([]))):
            A = R.const([1.0, 2.0], "float64")
            debug_callback(R.str("A"), A)
            B = R.const([3.0, 4.0], "float64")
            debug_callback(R.str("B"), B)
            C = A + B
            debug_callback(R.str("C"), C)
            return C


class TestCallbackDelayedUntilAfterDataflow(BaseCompare):
    """The debug callback is not inserted within a dataflow block.

    Dataflow blocks may not contain impure calls, and the callback is
    impure.

    """

    @I.ir_module
    class Before:
        @R.function
        def main():
            with R.dataflow():
                A = R.const([1.0, 2.0], "float64")
                B = R.const([3.0, 4.0], "float64")
                C = A + B
                R.output(A, B, C)

            return (A, B, C)

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(debug_callback: R.Callable(ret=R.Tuple([]))):
            with R.dataflow():
                A = R.const([1.0, 2.0], "float64")
                B = R.const([3.0, 4.0], "float64")
                C = A + B
                R.output(A, B, C)
            debug_callback(R.str("A"), A)
            debug_callback(R.str("B"), B)
            debug_callback(R.str("C"), C)
            return (A, B, C)


class TestDelayedCallbacksDoNotIncludeDataflowVar(BaseCompare):
    """The delayed callbacks only include non-dataflow variables

    The impure callback must occur after the dataflow block, but
    dataflow variables may only be accessed within the dataflow block.
    As a result, the callback is skipped for all dataflow vars.

    """

    @I.ir_module
    class Before:
        @R.function
        def main():
            with R.dataflow():
                A = R.const([1.0, 2.0], "float64")
                B = R.const([3.0, 4.0], "float64")
                C = A + B
                R.output(C)

            return C

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(debug_callback: R.Callable(ret=R.Tuple([]))):
            with R.dataflow():
                A = R.const([1.0, 2.0], "float64")
                B = R.const([3.0, 4.0], "float64")
                C = A + B
                R.output(C)
            debug_callback(R.str("C"), C)
            return C


class TestCallbackParameterPreservedNumInputAttribute(BaseCompare):
    """The callback function counts as a runtime input

    The `attr::kNumInput` ("num_input") attribute indicates which
    parameters are provided at runtime, and which are known at
    compile-time, such as model weights.  When the debug callback is
    inserted, any existing `attr::kNumInput` attributes must be
    updated.

    """

    @I.ir_module
    class Before:
        @R.function
        def main(
            activations: R.Tensor([16, 1024], dtype="float16"),
            weights: R.Tensor([1024, 1024], dtype="float16"),
            bias: R.Tensor([1024], dtype="float16"),
        ):
            R.func_attr({"num_input": 1})
            after_matmul = R.matmul(activations, weights)
            after_bias = R.add(after_matmul, bias)

            return after_bias

    @I.ir_module
    class Expected:
        @R.function(pure=False)
        def main(
            debug_callback: R.Callable(ret=R.Tuple([])),
            activations: R.Tensor([16, 1024], dtype="float16"),
            weights: R.Tensor([1024, 1024], dtype="float16"),
            bias: R.Tensor([1024], dtype="float16"),
        ):
            R.func_attr({"num_input": 2})
            after_matmul = R.matmul(activations, weights)
            debug_callback(R.str("after_matmul"), after_matmul)
            after_bias = R.add(after_matmul, bias)
            debug_callback(R.str("after_bias"), after_bias)

            return after_bias


@tvm.testing.parametrize_targets("llvm", "cuda")
def test_inject_debug_check_for_nan(target, dev):
    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([2], "float32")):
            B = A + R.prim_value(T.float32(1.0))
            C = R.sqrt(B)
            D = A + C
            return D

    target = tvm.target.Target(target)
    if "gpu" in target.keys:
        Module = tvm.ir.transform.Sequential(
            [
                tvm.relax.transform.LegalizeOps(),
                tvm.tir.transform.BindTarget(target),
                tvm.tir.transform.DefaultGPUSchedule(),
            ]
        )(Module)

    built = tvm.relax.build(Module, target)
    vm = tvm.relax.VirtualMachine(built, dev)

    # Suppose a function can be called with most outputs, producing
    # valid outputs.
    np_input = np.array([1.0, 2.0], dtype="float32")
    expected = np.sqrt(np_input + 1.0) + np_input
    tvm_input = tvm.nd.array(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    tvm.testing.assert_allclose(expected, tvm_output.numpy())

    # However, for some inputs, the function produces incorrect values
    np_input = np.array([-5.0, 5.0], dtype="float32")
    vm["main"](tvm.nd.array(np_input, dev))

    # We'd like to have some assertion in order to determine where the
    # error occurs.  However, we only have visibility to the final
    # output of the end-to-end function.

    def assert_not_nan(var_name, var_value):
        if isinstance(var_value, tvm.runtime.NDArray):
            contains_nan = np.isnan(var_value.numpy()).any()
            assert not contains_nan, f"Variable {var_name} contained NaN"

    # A callback can be inserted with `InjectDebugCallback`.  After
    # applying this pass, all externally-exposed functions take a
    # callback function as their first parameter.

    Module = tvm.relax.transform.InjectDebugCallback()(Module)

    built = tvm.relax.build(Module, target)
    vm = tvm.relax.VirtualMachine(built, dev)

    # The valid inputs can be inspected, and still produce the same
    # output.
    np_input = np.array([1.0, 2.0], dtype="float32")
    expected = np.sqrt(np_input + 1.0) + np_input
    tvm_input = tvm.nd.array(np_input, dev)
    tvm_output = vm["main"](assert_not_nan, tvm_input)
    tvm.testing.assert_allclose(expected, tvm_output.numpy())

    # However, the invalid inputs can be caught in the debug function
    # and inspected.
    np_input = np.array([-5.0, 5.0], dtype="float32")
    with pytest.raises(AssertionError, match="Variable C contained NaN"):
        vm["main"](assert_not_nan, tvm.nd.array(np_input, dev))


if __name__ == "__main__":
    tvm.testing.main()
