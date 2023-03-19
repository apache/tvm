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

import sys
import tempfile

import numpy as np
import tvm
import tvm.testing
from tvm import relax
from tvm._ffi.base import TVMError
from tvm.script import relax as R


@tvm.script.ir_module
class InputModule:
    @R.function
    def foo(x: R.Tensor(("m", "n"), "int64")):
        y = R.unique(x, sorted=False)
        y_sorted = R.unique(x)
        return y, y_sorted


def run_cpu(mod, func_name, *input):
    target = tvm.target.Target("llvm")
    ex = relax.build(mod, target)
    vm = relax.VirtualMachine(ex, tvm.cpu())
    return vm[func_name](*input)


def test_unique():

    # TODO(prakalp): also add test for compiling and running on cuda device.
    data_numpy = np.random.randint(0, 16, (16, 16))
    data = tvm.nd.array(data_numpy)
    result, result_sorted = run_cpu(InputModule, "foo", data)

    expected_output_sorted, indices = np.unique(data_numpy, return_index=True)
    expected_output = [data_numpy.flatten()[index] for index in sorted(indices, reverse=True)]

    np.testing.assert_array_equal(expected_output_sorted, result_sorted.numpy())
    np.testing.assert_array_equal(expected_output, result.numpy())


@tvm.script.ir_module
class PrintTest:
    @R.function
    def foo(x: R.Tensor((), "int32")):
        # results have to be bound, but we don't use them
        # TODO: We should allow calls whose results are not bound for side effects;
        #       it would be easy syntactic sugar to add.
        p1 = R.print(x)
        p2 = R.print(x, format="Number: {}")
        t = (x, x)
        p3 = R.print(t, format="Tuple: {}")
        p4 = R.print(x, t)
        p5 = R.print(x, x, format="Custom print: {} {}")
        p6 = R.print(x, t, format="Another print: {} {}")
        return x


def test_print():
    try:
        stdout = sys.stdout
        with tempfile.TemporaryFile(mode="w+") as test_out:
            sys.stdout = test_out
            run_cpu(PrintTest, "foo", tvm.nd.array(np.array(1).astype("int32")))
            test_out.seek(0)
            printed_text = str(test_out.read())
            expected = "1\nNumber: 1\nTuple: (1, 1)\n1 (1, 1)\nCustom print: 1 1\nAnother print: 1 (1, 1)\n"
            assert printed_text in expected, ("printed_text is ", printed_text)
    finally:
        sys.stdout = stdout


@tvm.script.ir_module
class AssertOpTest:
    @R.function
    def passes(x: R.Tensor((), "int32")):
        p1 = R.assert_op(relax.const(True))
        return x

    @R.function
    def pass_with_args(x: R.Tensor((), "int32")):
        p1 = R.assert_op(relax.const(True), x, format="You won't see me")
        return x

    @R.function
    def simple_fail(x: R.Tensor((), "int32")):
        p1 = R.assert_op(relax.const(False))
        return x

    @R.function
    def fail_with_message(x: R.Tensor((), "int32")):
        p1 = R.assert_op(relax.const(False), format="I failed...")
        return x

    @R.function
    def fail_with_args(x: R.Tensor((), "int32")):
        # no format
        p1 = R.assert_op(relax.const(False), [x, x])
        return x

    @R.function
    def fail_with_formatted_message(x: R.Tensor((), "int32")):
        p1 = R.assert_op(relax.const(False), x, format="Number: {}")
        return x


def test_assert_op():
    def check_assertion_error(func_name, func_arg, expected_message):
        passed = False
        try:
            run_cpu(AssertOpTest, func_name, func_arg)
            passed = True
        except TVMError as e:
            # TVM will print out a TVMError that will contain the
            # generated error at the bottom of a stack trace
            assert "AssertionError" in e.args[0]
            assert expected_message in e.args[0]
        assert not passed

    run_cpu(AssertOpTest, "passes", tvm.nd.array(np.array(1).astype("int32")))
    run_cpu(AssertOpTest, "pass_with_args", tvm.nd.array(np.array(2).astype("int32")))
    check_assertion_error(
        "simple_fail", tvm.nd.array(np.array(3).astype("int32")), "Assertion Failed"
    )
    check_assertion_error(
        "fail_with_message", tvm.nd.array(np.array(4).astype("int32")), "I failed..."
    )
    check_assertion_error("fail_with_args", tvm.nd.array(np.array(5).astype("int32")), "5, 5")
    check_assertion_error(
        "fail_with_formatted_message", tvm.nd.array(np.array(6).astype("int32")), "Number: 6"
    )


@tvm.script.ir_module
class ShapeOfTest:
    @R.function
    def get_shape(t: R.Tensor(ndim=-1, dtype="int32")) -> R.Shape(ndim=-1):
        return R.shape_of(t)

    @R.function
    def get_constrained_shape(t: R.Tensor(ndim=1, dtype="int32")) -> R.Shape(ndim=1):
        # require the input tensor to have rank 1
        return R.shape_of(t)

    @R.function
    def get_scalar_shape() -> R.Shape(()):
        x: R.Tensor((), "int32") = R.const(1, dtype="int32")
        return R.shape_of(x)

    @R.function
    def get_constant_shape() -> R.Shape((2, 2)):
        x: R.Tensor((2, 2), "int32") = R.const(
            np.array([[1, 2], [3, 4]], dtype="int32"), dtype="int32"
        )
        return R.shape_of(x)


def test_op_shape_of():
    unit_shape = run_cpu(ShapeOfTest, "get_scalar_shape")
    assert unit_shape == tvm.runtime.ShapeTuple([])

    const_shape = run_cpu(ShapeOfTest, "get_constant_shape")
    assert const_shape == tvm.runtime.ShapeTuple([2, 2])

    scalar_shape = run_cpu(ShapeOfTest, "get_shape", tvm.nd.array(np.array(1, dtype="int32")))
    assert scalar_shape == tvm.runtime.ShapeTuple([])

    tensor_shape = run_cpu(
        ShapeOfTest, "get_shape", tvm.nd.array(np.zeros((1, 2, 3)).astype("int32"))
    )
    assert tensor_shape == tvm.runtime.ShapeTuple([1, 2, 3])

    constrained_shape = run_cpu(
        ShapeOfTest, "get_constrained_shape", tvm.nd.array(np.zeros((1,)).astype("int32"))
    )
    assert constrained_shape == tvm.runtime.ShapeTuple([1])


if __name__ == "__main__":
    tvm.testing.main()
