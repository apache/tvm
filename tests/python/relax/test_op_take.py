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

import tvm
import tvm.testing
from tvm.script import ir as I, relax as R, tir as T

import numpy as np

axis = tvm.testing.parameter(0, 1)


@tvm.testing.parametrize_targets("llvm")
def test_take_scalar_tensor_as_index(target, dev, axis):
    """The index of R.take may be a scalar tensor

    Using a scalar tensor as the index reduces the dimension of the
    output.

    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16, 16], "float16")):
            output = R.take(A, R.const(1), axis=axis)
            return output

    built = tvm.relax.build(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, dev)

    np_input = np.random.random(size=[16, 16]).astype("float16")
    tvm_input = tvm.nd.array(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np_input.take(1, axis=axis)

    tvm.testing.assert_allclose(tvm_output.numpy(), np_expected)


@tvm.testing.parametrize_targets("llvm")
def test_take_1d_tensor_as_index(target, dev, axis):
    """The index of R.take may be a non-scalar tensor

    In general, `R.take` outputs a tensor of dimension
    `data.ndim + indices.ndim - 1`.
    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16, 16], "float16")):
            output = R.take(A, R.const([1]), axis=axis)
            return output

    built = tvm.relax.build(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, dev)

    np_input = np.random.random(size=[16, 16]).astype("float16")
    tvm_input = tvm.nd.array(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np_input.take([1], axis=axis)

    tvm.testing.assert_allclose(tvm_output.numpy(), np_expected)


@tvm.testing.parametrize_targets("llvm")
def test_take_2d_tensor_as_index(target, dev, axis):
    """The index of R.take may be a 2-d tensor"""

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16, 16], "float16")):
            output = R.take(A, R.const([[1, 3], [5, 7]]), axis=axis)
            return output

    built = tvm.relax.build(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, dev)

    np_input = np.random.random(size=[16, 16]).astype("float16")
    tvm_input = tvm.nd.array(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np_input.take([[1, 3], [5, 7]], axis=axis)

    tvm.testing.assert_allclose(tvm_output.numpy(), np_expected)


@tvm.testing.parametrize_targets("llvm")
def test_take_constant_prim_value_as_index(target, dev, axis):
    """The index of R.take may be a R.prim_value

    The `R.prim_value` produces output equivalent to a scalar
    tensor.

    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor([16, 16], "float16")):
            output = R.take(A, R.prim_value(1), axis=axis)
            return output

    built = tvm.relax.build(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, dev)

    np_input = np.random.random(size=[16, 16]).astype("float16")
    tvm_input = tvm.nd.array(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np_input.take(1, axis=axis)

    tvm.testing.assert_allclose(tvm_output.numpy(), np_expected)


@tvm.testing.parametrize_targets("llvm")
def test_take_dynamic_prim_value_as_index(target, dev, axis):
    """The index of R.take may be a dynamic R.prim_value

    The `R.prim_value` produces output equivalent to a scalar
    tensor.

    """

    @I.ir_module
    class Module:
        @R.function
        def main(A: R.Tensor(["n", "n"], "float16")):
            n = T.int64()
            output = R.take(A, R.prim_value(n - 1), axis=axis)
            return output

    built = tvm.relax.build(Module, target=target)
    vm = tvm.relax.VirtualMachine(built, dev)

    np_input = np.random.random(size=[16, 16]).astype("float16")
    tvm_input = tvm.nd.array(np_input, dev)
    tvm_output = vm["main"](tvm_input)
    np_expected = np_input.take(15, axis=axis)

    tvm.testing.assert_allclose(tvm_output.numpy(), np_expected)


if __name__ == "__main__":
    tvm.testing.main()
