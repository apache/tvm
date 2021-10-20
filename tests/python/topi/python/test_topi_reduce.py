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
"""Test code for reduce."""
import os
import sys

import numpy as np
import pytest

import tvm
import tvm.testing
import tvm.topi.testing

from tvm import te, topi

in_shape, axis, keepdims, reduce_type, dtype = tvm.testing.parameters(
    ((32,), 0, False, "argmax", "float32"),
    ((128, 24, 128, 24), (1, 2, 3), True, "sum", "float32"),
    ((2, 3), None, True, "all", "bool"),
    ((128, 24 * 128 * 24), (1,), False, "max", "float32"),
    ((32, 128, 24), None, True, "sum", "float32"),
    ((32, 128, 24), None, True, "all", "bool"),
    ((128, 24, 128, 24), (0, 2), False, "min", "float32"),
    ((32, 128), 1, True, "argmax", "float32"),
    ((32, 24, 32, 24), 2, False, "argmin", "float32"),
    ((31, 21, 15), None, True, "argmax", "float32"),
    ((31, 21, 15), None, False, "sum", "float32"),
    ((128, 24, 128, 24), (1, 2, 3), True, "sum", "float64"),
    ((2, 3), None, True, "any", "bool"),
    ((32, 128, 24), None, True, "any", "bool"),
    ((1, 4, 7), 1, True, "any", "bool"),
    ((128, 24, 128, 24), 2, False, "any", "bool"),
)


@tvm.testing.fixture(cache_return_value=True)
def ref_data(in_shape, axis, keepdims, reduce_type, dtype):
    # Test
    if dtype == "bool":
        in_npy_map = in_npy = np.random.choice([True, False], size=in_shape)
    else:
        in_npy = np.random.uniform(-1, 1, size=in_shape).astype(dtype)
        in_npy_map = np.sqrt(np.exp(in_npy)).astype(dtype)

    if reduce_type == "sum":
        out_npy = in_npy_map.sum(axis=axis, keepdims=keepdims)
    elif reduce_type == "all" and dtype == "bool":
        out_npy = in_npy_map.all(axis=axis, keepdims=keepdims)
    elif reduce_type == "any" and dtype == "bool":
        out_npy = in_npy_map.any(axis=axis, keepdims=keepdims)
    elif reduce_type == "max":
        out_npy = in_npy_map.max(axis=axis, keepdims=keepdims)
    elif reduce_type == "min":
        out_npy = in_npy_map.min(axis=axis, keepdims=keepdims)
    elif reduce_type == "argmax":
        out_npy = _my_npy_argmax(in_npy_map, axis=axis, keepdims=keepdims)
    elif reduce_type == "argmin":
        out_npy = _my_npy_argmin(in_npy_map, axis=axis, keepdims=keepdims)
    else:
        raise NotImplementedError

    return in_npy, in_npy_map, out_npy


def _my_npy_argmax(arr, axis, keepdims):
    if not keepdims:
        return arr.argmax(axis=axis)
    else:
        if axis is None:
            out_shape = [1 for _ in arr.shape]
        else:
            out_shape = list(arr.shape)
            out_shape[axis] = 1

        return arr.argmax(axis=axis).reshape(out_shape)


def _my_npy_argmin(arr, axis, keepdims):
    if not keepdims:
        return arr.argmin(axis=axis)
    else:
        if axis is None:
            out_shape = [1 for _ in arr.shape]
        else:
            out_shape = list(arr.shape)
            out_shape[axis] = 1
        return arr.argmin(axis=axis).reshape(out_shape)


def test_reduce_map(target, dev, ref_data, in_shape, axis, keepdims, reduce_type, dtype):
    target = tvm.target.Target(target)
    if target.kind.name == "vulkan" and reduce_type in ["sum", "any", "all"]:
        pytest.xfail(f"Vulkan backend has known errors on {reduce_type}")

    in_npy, in_npy_map, out_npy = ref_data

    # Build the logic and compile the function
    A = te.placeholder(shape=in_shape, name="A", dtype=dtype)
    A1 = topi.sqrt(topi.exp(A))
    out_dtype = dtype
    if reduce_type == "sum":
        B = topi.sum(A1, axis=axis, keepdims=keepdims)
    elif reduce_type == "all":
        B = topi.all(A, axis=axis, keepdims=keepdims)
    elif reduce_type == "any":
        B = topi.any(A, axis=axis, keepdims=keepdims)
    elif reduce_type == "max":
        B = topi.max(A1, axis=axis, keepdims=keepdims)
    elif reduce_type == "min":
        B = topi.min(A1, axis=axis, keepdims=keepdims)
    elif reduce_type == "argmax":
        B = topi.argmax(A1, axis=axis, keepdims=keepdims)
        out_dtype = "int32"
    elif reduce_type == "argmin":
        B = topi.argmin(A1, axis=axis, keepdims=keepdims)
        out_dtype = "int32"
    else:
        raise NotImplementedError

    with tvm.target.Target(target):
        s = tvm.topi.testing.get_reduce_schedule(target)(B)

    foo = tvm.build(s, [A, B], target, name=reduce_type)

    data_tvm = tvm.nd.array(in_npy, device=dev)
    out_tvm = tvm.nd.empty(shape=out_npy.shape, device=dev, dtype=out_dtype)
    foo(data_tvm, out_tvm)

    if reduce_type == "argmax" or reduce_type == "argmin":
        out_tvm_indices = out_tvm.numpy()
        if keepdims:
            out_tvm_indices = np.take(out_tvm_indices, indices=0, axis=axis)
        if axis is None:
            out_tvm_val = in_npy_map.ravel()[out_tvm_indices]
        else:
            other_indices = tuple(np.indices(in_shape[0:axis] + in_shape[(axis + 1) :]))
            sel_indices = other_indices[0:axis] + (out_tvm_indices,) + other_indices[axis:]
            out_tvm_val = in_npy_map[sel_indices]
        if reduce_type == "argmax":
            tvm.testing.assert_allclose(out_tvm_val, in_npy_map.max(axis=axis), 1e-3, 1e-3)
        elif reduce_type == "argmin":
            tvm.testing.assert_allclose(out_tvm_val, in_npy_map.min(axis=axis), 1e-3, 1e-3)
    else:
        tvm.testing.assert_allclose(out_tvm.numpy(), out_npy, 1e-3, 1e-3)


def test_complex_reduce(target, dev):
    in_shape = (2, 3)
    dtype = "float32"
    axis = 0
    keepdims = False
    A = te.placeholder(shape=in_shape, name="A", dtype=dtype)
    B = topi.sum(A, axis=axis, keepdims=keepdims)
    C = topi.add(B, B)
    D = topi.multiply(B, B)
    E = topi.add(C, D)

    with tvm.target.Target(target):
        s = tvm.topi.testing.get_reduce_schedule(target)(E)
    foo = tvm.build(s, [A, E], target, name="sum")

    in_npy = np.random.uniform(-1, 1, size=in_shape).astype(dtype)
    sum_npy = in_npy.sum(axis=axis, keepdims=keepdims)
    out_npy = sum_npy * 2 + sum_npy * sum_npy

    data_tvm = tvm.nd.array(in_npy, device=dev)
    out_tvm = tvm.nd.empty(shape=out_npy.shape, device=dev, dtype=dtype)
    foo(data_tvm, out_tvm)
    tvm.testing.assert_allclose(out_tvm.numpy(), out_npy, 1e-3, 1e-3)


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv))
