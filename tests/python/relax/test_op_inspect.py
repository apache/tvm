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

import ctypes

import numpy as np
import pytest

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

elem_offset = tvm.testing.parameter(0, 64, 128)


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


def _get_compact_striding(shape):
    strides = []
    product = 1
    for dim in reversed(shape):
        strides.append(product)
        product *= dim
    return list(reversed(strides))


def test_strides_of_compact_tensor(shape):
    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor, axis: R.Prim("int64")):
            return A.strides[axis]

    built = relax.build(mod)
    vm = relax.VirtualMachine(built, tvm.cpu())

    arg = tvm.nd.empty(shape, "int32")

    res = [vm["main"](arg, i) for i, _ in enumerate(shape)]
    expected = _get_compact_striding(shape)

    tvm.ir.assert_structural_equal(res, expected)


def test_strides_of_non_compact_tensor():
    backing_shape = [64, 64]
    view_shape = [16, 16]
    expected_strides = [backing_shape[0], 1]

    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor, axis: R.Prim("int64")):
            return A.strides[axis]

    built = relax.build(mod)
    vm = relax.VirtualMachine(built, tvm.cpu())

    backing_ndarray = tvm.nd.empty(backing_shape, "int32")

    # Manually overwrite the DLTensor fields to make a view into the
    # tensor.
    view = backing_ndarray.handle[0]
    np_shape = np.array([16, 16], "int64")
    view.shape = np_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    np_strides = np.array([64, 1], "int64")
    view.strides = np_strides.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    backing_ndarray.handle[0] = view

    res = [vm["main"](backing_ndarray, i) for i, _ in enumerate(view_shape)]

    tvm.ir.assert_structural_equal(res, expected_strides)


def test_byte_offset(elem_offset):
    backing_shape = [64, 64]
    view_shape = [16, 16]
    byte_offset = elem_offset * 4

    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor):
            return A.byte_offset

    built = relax.build(mod)
    vm = relax.VirtualMachine(built, tvm.cpu())

    backing_ndarray = tvm.nd.empty(backing_shape, "int32")

    # Manually overwrite the DLTensor fields to make a view into the
    # tensor.
    view = backing_ndarray.handle[0]
    np_shape = np.array(view_shape, "int64")
    view.shape = np_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    view.byte_offset = byte_offset
    backing_ndarray.handle[0] = view

    res = vm["main"](backing_ndarray)

    assert res == byte_offset


def test_elem_offset(elem_offset, dtype):
    tvm_dtype = tvm.runtime.DataType(dtype)

    backing_shape = [64, 64]
    view_shape = [16, 16]
    element_bytes = (tvm_dtype.bits * tvm_dtype.lanes) // 8
    byte_offset = elem_offset * element_bytes

    @I.ir_module
    class mod:
        @R.function
        def main(A: R.Tensor):
            return A.elem_offset

    built = relax.build(mod)
    vm = relax.VirtualMachine(built, tvm.cpu())

    backing_ndarray = tvm.nd.empty(backing_shape, dtype)

    # Manually overwrite the DLTensor fields to make a view into the
    # tensor.
    view = backing_ndarray.handle[0]
    np_shape = np.array(view_shape, "int64")
    view.shape = np_shape.ctypes.data_as(ctypes.POINTER(ctypes.c_long))
    view.byte_offset = byte_offset
    backing_ndarray.handle[0] = view

    res = vm["main"](backing_ndarray)

    assert res == elem_offset


if __name__ == "__main__":
    tvm.testing.main()
