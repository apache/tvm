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
import pytest
import tvm
import tvm.testing
from tvm import relax
from tvm.relax.frontend.nn.core import Tensor

import numpy as np


def test_tensor_from_numpy():
    x = np.random.rand(1, 10)
    tensor_x = Tensor.from_const(x)
    assert tensor_x.shape == [1, 10]
    assert tensor_x.ndim == 2
    assert tensor_x.dtype == "float32"
    assert repr(tensor_x) == 'Tensor([1, 10], "float32")'


def test_tensor_from_scalar():
    x = 123.321
    tensor_x = Tensor.from_scalar(x, dtype="float16")
    assert tensor_x.shape == []
    assert tensor_x.ndim == 0
    assert tensor_x.dtype == "float16"
    assert repr(tensor_x) == 'Tensor([], "float16")'


def test_tensor_op_binary_tensor_tensor():
    np_x = np.random.rand(1, 10)
    np_y = np.random.rand(2, 1)
    x, y = Tensor.from_const(np_x), Tensor.from_const(np_y)

    bb = relax.BlockBuilder()
    with bb.function("test"):
        add_out = x + y
        mul_out = x * y
        div_out = x / y
        max_out = x.maximum(y)
        min_out = x.minimum(y)
        bb.emit_func_output(x._expr, [])

    assert isinstance(add_out, Tensor) and add_out.shape == [2, 10]
    assert isinstance(mul_out, Tensor) and mul_out.shape == [2, 10]
    assert isinstance(div_out, Tensor) and div_out.shape == [2, 10]
    assert isinstance(max_out, Tensor) and max_out.shape == [2, 10]
    assert isinstance(min_out, Tensor) and min_out.shape == [2, 10]


def test_tensor_op_binary_tensor_saclar():
    np_x = np.random.rand(2, 10)
    y = 10
    x = Tensor.from_const(np_x)

    bb = relax.BlockBuilder()
    with bb.function("test"):
        add_out = x + y
        radd_out = y + x
        mul_out = x * y
        div_out = x / y
        max_out = x.maximum(y)
        min_out = x.minimum(y)
        bb.emit_func_output(x._expr, [])

    assert isinstance(add_out, Tensor) and add_out.shape == [2, 10]
    assert isinstance(radd_out, Tensor) and radd_out.shape == [2, 10]
    assert isinstance(mul_out, Tensor) and mul_out.shape == [2, 10]
    assert isinstance(div_out, Tensor) and div_out.shape == [2, 10]
    assert isinstance(max_out, Tensor) and max_out.shape == [2, 10]
    assert isinstance(min_out, Tensor) and min_out.shape == [2, 10]


def test_tensor_op_datatype():
    np_x = np.random.rand(2, 1, 10).astype("float32")
    x = Tensor.from_const(np_x)
    bb = relax.BlockBuilder()
    with bb.function("test"):
        astype_out = x.astype("float16")
        bb.emit_func_output(x._expr, [])

    assert (
        isinstance(astype_out, Tensor)
        and astype_out.shape == [2, 1, 10]
        and astype_out.dtype == "float16"
    )


def test_tensor_op_manipulate():
    np_x = np.random.rand(2, 1, 10)
    x = Tensor.from_const(np_x)

    bb = relax.BlockBuilder()
    with bb.function("test"):
        reshape_out = x.reshape([2, 5, 2])
        permute_dims_out = x.permute_dims([2, 1, 0])
        repeat_out = x.repeat(2, axis=1)
        bb.emit_func_output(x._expr, [])

    assert isinstance(reshape_out, Tensor) and reshape_out.shape == [2, 5, 2]
    assert isinstance(permute_dims_out, Tensor) and permute_dims_out.shape == [10, 1, 2]
    assert isinstance(repeat_out, Tensor) and repeat_out.shape == [2, 2, 10]


if __name__ == "__main__":
    tvm.testing.main()
