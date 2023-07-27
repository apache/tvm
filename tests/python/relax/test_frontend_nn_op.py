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
from tvm import relax, te, tir
from tvm.relax.frontend.nn import op
from tvm.relax.frontend.nn.core import Tensor

import numpy as np


def test_binary():
    np_x = np.random.rand(1, 10)
    np_y = np.random.rand(10, 1)
    x, y = Tensor.from_const(np_x), Tensor.from_const(np_y)

    bb = relax.BlockBuilder()
    with bb.function("test"):
        add_out = op.add(x, y)
        multiply_out = op.multiply(x, y)
        div_out = op.divide(x, y)
        matmul_out = op.matmul(x, y)
        max_out = op.maximum(x, y)
        min_out = op.minimum(x, y)
        bb.emit_func_output(x._expr, [])

    assert isinstance(add_out, Tensor) and add_out.shape == [10, 10]
    assert isinstance(multiply_out, Tensor) and multiply_out.shape == [10, 10]
    assert isinstance(div_out, Tensor) and div_out.shape == [10, 10]
    assert isinstance(matmul_out, Tensor) and matmul_out.shape == [1, 1]
    assert isinstance(max_out, Tensor) and max_out.shape == [10, 10]
    assert isinstance(min_out, Tensor) and min_out.shape == [10, 10]


def test_manipulate():
    np_x = np.random.rand(1, 5, 2)
    x = Tensor.from_const(np_x)

    bb = relax.BlockBuilder()
    with bb.function("test"):
        broadcast_to_out = op.broadcast_to(x, [2, 5, 2])
        permute_dims_out = op.permute_dims(x, [2, 1, 0])
        reshape_out = op.reshape(x, [1, 10])
        repeat_out = op.repeat(x, repeats=2, axis=1)
        squeeze_out = op.squeeze(x, 0)
        bb.emit_func_output(x._expr, [])

    assert isinstance(broadcast_to_out, Tensor) and broadcast_to_out.shape == [2, 5, 2]
    assert isinstance(permute_dims_out, Tensor) and permute_dims_out.shape == [2, 5, 1]
    assert isinstance(reshape_out, Tensor) and reshape_out.shape == [1, 10]
    assert isinstance(repeat_out, Tensor) and repeat_out.shape == [1, 10, 2]
    assert isinstance(squeeze_out, Tensor) and squeeze_out.shape == [5, 2]


def test_index():
    np_x = np.random.rand(2, 1, 10)
    np_y = np.random.randint(0, 5, (5,))
    x, y = Tensor.from_const(np_x), Tensor.from_const(np_y)

    bb = relax.BlockBuilder()
    with bb.function("test"):
        take_out = op.take(x, y, axis=2)
        bb.emit_func_output(x._expr, [])

    assert isinstance(take_out, Tensor) and take_out.shape == [2, 1, 5]


def test_datatype():
    np_x = np.random.rand(2, 1, 10).astype("float32")
    x = Tensor.from_const(np_x)

    bb = relax.BlockBuilder()
    with bb.function("test"):
        astype_out = op.astype(x, "float16")
        bb.emit_func_output(x._expr, [])

    assert (
        isinstance(astype_out, Tensor)
        and astype_out.shape == [2, 1, 10]
        and astype_out.dtype == "float16"
    )


def test_nn():
    np_x = np.random.rand(2, 3, 4, 5)
    np_weight = np.random.rand(4, 5)
    np_bias = np.random.rand(4, 5)
    x = Tensor.from_const(np_x)
    weight = Tensor.from_const(np_weight)
    bias = Tensor.from_const(np_bias)

    bb = relax.BlockBuilder()
    with bb.function("test"):
        silu_out = op.silu(x)
        softmax_out = op.softmax(x, axis=2)
        rms_norm_out = op.rms_norm(x, weight, bias, axes=[-2, -1])
        rms_norm_with_bias_out = op.rms_norm(x, weight, bias, axes=[-2, -1])
        bb.emit_func_output(x._expr, [])

    assert isinstance(silu_out, Tensor) and silu_out.shape == [2, 3, 4, 5]
    assert isinstance(softmax_out, Tensor) and softmax_out.shape == [2, 3, 4, 5]
    assert isinstance(rms_norm_out, Tensor) and rms_norm_out.shape == [2, 3, 4, 5]
    assert isinstance(rms_norm_with_bias_out, Tensor) and rms_norm_with_bias_out.shape == [
        2,
        3,
        4,
        5,
    ]


def test_create():
    np_x = np.random.rand(10, 10)
    x = Tensor.from_const(np_x)

    bb = relax.BlockBuilder()
    with bb.function("test"):
        triu_out = op.triu(x)
        full_with_scalar_out = op.full([10, 10], fill_value=10)
        full_with_FloatImm_out = op.full(
            [10, 10], fill_value=tir.FloatImm(dtype="float32", value=10)
        )
        full_with_Tensor_out = op.full([10, 10], fill_value=Tensor.from_scalar(10, dtype="float32"))
        full_with_scalar_fp16_out = op.full([10, 10], fill_value=10, dtype="float16")
        zeros_out = op.zeros([10, 10])
        zeros_fp16_out = op.zeros([10, 10], dtype="float16")
        bb.emit_func_output(x._expr, [])

    assert isinstance(triu_out, Tensor) and triu_out.shape == [10, 10]
    assert isinstance(full_with_scalar_out, Tensor) and full_with_scalar_out.shape == [10, 10]
    assert isinstance(full_with_FloatImm_out, Tensor) and full_with_FloatImm_out.shape == [10, 10]
    assert isinstance(full_with_Tensor_out, Tensor) and full_with_Tensor_out.shape == [10, 10]
    assert (
        isinstance(full_with_scalar_fp16_out, Tensor)
        and full_with_scalar_fp16_out.shape == [10, 10]
        and full_with_scalar_fp16_out.dtype == "float16"
    )
    assert (
        isinstance(zeros_out, Tensor)
        and zeros_out.shape == [10, 10]
        and zeros_out.dtype == "float32"
    )
    assert (
        isinstance(zeros_fp16_out, Tensor)
        and zeros_fp16_out.shape == [10, 10]
        and zeros_fp16_out.dtype == "float16"
    )


def test_tensor_expr_op():
    np_x = np.random.rand(10, 10)
    x = Tensor.from_const(np_x)

    bb = relax.BlockBuilder()
    with bb.function("test"):
        tensor_expr_op_out = op.tensor_expr_op(
            tensor_expr_func=lambda x: x + 1, name_hint="add_one", args=[x]
        )
        bb.emit_func_output(x._expr, [])

    assert isinstance(tensor_expr_op_out, Tensor) and tensor_expr_op_out.shape == [10, 10]


if __name__ == "__main__":
    tvm.testing.main()
