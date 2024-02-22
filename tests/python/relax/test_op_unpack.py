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

import tvm.testing

from tvm import relax
from tvm.ir import Op
from tvm.script import ir as I, relax as R

# Parameterization for reading dtype of DLTensor.  Chosen to have
# multiple distinct type codes, number of lanes, and widths.
dtype = tvm.testing.parameter(
    "int32",
    "int64",
    "float32",
    "float32x4",
    "bfloat",
    "e4m3_float8",
)
shape = tvm.testing.parameter(
    [],
    [16],
    [128, 256],
    [1] * 64,
)


def test_tensor_dtype_code(dtype):
    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor):
            return A.dtype.type_code

    built = relax.build(mod)
    vm = relax.VirtualMachine(built, tvm.cpu())

    arg = tvm.nd.empty([16], dtype)
    res = vm["main"](arg)

    expected_type_code = tvm.runtime.DataType(dtype).type_code
    assert res == expected_type_code


def test_tensor_dtype_bits(dtype):
    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor):
            return A.dtype.bits

    built = relax.build(mod)
    vm = relax.VirtualMachine(built, tvm.cpu())

    arg = tvm.nd.empty([16], dtype)
    res = vm["main"](arg)

    expected_type_bits = tvm.runtime.DataType(dtype).bits
    assert res == expected_type_bits


def test_tensor_dtype_lanes(dtype):
    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor):
            return A.dtype.lanes

    built = relax.build(mod)
    vm = relax.VirtualMachine(built, tvm.cpu())

    arg = tvm.nd.empty([16], dtype)
    res = vm["main"](arg)

    expected_type_lanes = tvm.runtime.DataType(dtype).lanes
    assert res == expected_type_lanes


def test_tensor_ndim(shape):
    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor):
            return A.ndim

    built = relax.build(mod)
    vm = relax.VirtualMachine(built, tvm.cpu())

    arg = tvm.nd.empty(shape, "int32")
    res = vm["main"](arg)

    assert res == len(shape)


def test_tensor_shape(shape):
    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor, axis: R.Prim("int64")):
            return A.shape[axis]

    built = relax.build(mod)
    vm = relax.VirtualMachine(built, tvm.cpu())

    arg = tvm.nd.empty(shape, "int32")

    res = [vm["main"](arg, i) for i, _ in enumerate(shape)]

    tvm.ir.assert_structural_equal(res, shape)


if __name__ == "__main__":
    tvm.testing.main()
