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

import numpy as np
import pytest

import tvm
import tvm.testing

from tvm import te, relay
from tvm.relay.testing import check_grad, run_infer_type
from tvm.relay.transform import gradient

executor_kind = tvm.testing.parameter("debug")


def sigmoid(x):
    one = np.ones_like(x)
    return one / (one + np.exp(-x))


def relu(x):
    x_copy = np.copy(x)
    np.maximum(x_copy, 0, x_copy)
    return x_copy


class TestUnaryOp:
    config = {
        "log": (tvm.relay.log, lambda x, g: g * (1 / x)),
        "exp": (tvm.relay.exp, lambda x, g: g * np.exp(x)),
        "sigmoid": (tvm.relay.sigmoid, lambda x, g: g * sigmoid(x) * (1 - sigmoid(x))),
        "tanh": (tvm.relay.tanh, lambda x, g: g * (1 - np.tanh(x) * np.tanh(x))),
        "sqrt": (tvm.relay.sqrt, lambda x, g: g * 0.5 * np.power(x, -0.5)),
        "abs": (tvm.relay.abs, lambda x, g: np.where(x < 0, -g, g)),
        "relu": (relay.nn.relu, lambda x, g: np.where(x < 0, np.zeros_like(x), g)),
        "erf": (tvm.relay.erf, lambda x, g: g * (2.0 / (np.pi ** (0.5)) * np.exp(-x * x))),
        "cos": (tvm.relay.cos, lambda x, g: g * -1.0 * np.sin(x)),
        "sin": (tvm.relay.sin, lambda x, g: g * np.cos(x)),
        "tan": (tvm.relay.tan, lambda x, g: g * (1.0 / (np.cos(x) ** 2))),
        "atan": (tvm.relay.atan, lambda x, g: g * (1 / (1 + np.power(x, 2.0)))),
        "log2": (tvm.relay.log2, lambda x, g: g * (1 / (np.log(2) * x))),
        "log10": (tvm.relay.log10, lambda x, g: g * (1 / (np.log(10) * x))),
        "cosh": (tvm.relay.cosh, lambda x, g: g * (np.sinh(x))),
        "sinh": (tvm.relay.sinh, lambda x, g: g * (np.cosh(x))),
        "asin": (tvm.relay.asin, lambda x, g: g * (1.0 / (1.0 - x**2) ** (1.0 / 2.0))),
        "acos": (tvm.relay.acos, lambda x, g: g * (-1.0 / (1.0 - x**2.0) ** (1.0 / 2.0))),
        "acosh": (tvm.relay.acosh, lambda x, g: g * (1.0 / (x**2 - 1.0) ** (1.0 / 2.0))),
        "asinh": (tvm.relay.asinh, lambda x, g: g * (1.0 / (x**2 + 1.0) ** (1.0 / 2.0))),
        "atanh": (tvm.relay.atanh, lambda x, g: g * (-1.0 / (x**2 - 1.0))),
    }

    relay_op, ref_func = tvm.testing.parameters(*config.values(), ids=config.keys())
    dtype = tvm.testing.parameter("float32", "float64")
    shape = tvm.testing.parameter((10, 4))

    def test_op(self, target, dev, executor_kind, relay_op, ref_func, shape, dtype):

        target = tvm.target.Target(target)
        if target.kind.name == "vulkan":

            known_breaks = {
                "float32": [
                    tvm.relay.erf,
                    tvm.relay.tan,
                    tvm.relay.atan,
                    tvm.relay.log10,
                    tvm.relay.cosh,
                    tvm.relay.sinh,
                    tvm.relay.asin,
                    tvm.relay.acos,
                    tvm.relay.acosh,
                    tvm.relay.asinh,
                    tvm.relay.atanh,
                ],
                "float64": [
                    tvm.relay.log,
                    tvm.relay.exp,
                    tvm.relay.sigmoid,
                    tvm.relay.tanh,
                    tvm.relay.sqrt,
                    tvm.relay.erf,
                    tvm.relay.cos,
                    tvm.relay.sin,
                    tvm.relay.tan,
                    tvm.relay.atan,
                    tvm.relay.log2,
                    tvm.relay.log10,
                    tvm.relay.cosh,
                    tvm.relay.sinh,
                    tvm.relay.asin,
                    tvm.relay.acos,
                    tvm.relay.acosh,
                    tvm.relay.asinh,
                    tvm.relay.atanh,
                ],
            }

            if relay_op in known_breaks[dtype]:
                pytest.xfail(f"{dtype} {relay_op.__name__} not yet supported on Vulkan runtime")

        tp = relay.TensorType(shape, dtype)
        x = relay.var("x", tp)
        g = relay.var("g", tp)
        y = relay_op(x) * g

        fwd_func = relay.Function([x, g], y)
        fwd_func = run_infer_type(fwd_func)
        bwd_func = run_infer_type(gradient(fwd_func))

        data_in = np.random.rand(*shape).astype(dtype)
        grad_in = np.random.rand(*shape).astype(dtype)
        ref_grad_out = ref_func(data_in, grad_in)

        op_res, (op_grad, _) = relay.create_executor(
            executor_kind, device=dev, target=target
        ).evaluate(bwd_func)(data_in, grad_in)
        np.testing.assert_allclose(op_grad.numpy(), ref_grad_out, rtol=0.01)


class TestBinaryOp:
    config = {
        "add": (relay.add, lambda x, y: [np.ones_like(x), np.ones_like(y)]),
        "subtract": (relay.subtract, lambda x, y: [np.ones_like(x), -np.ones_like(y)]),
        "multiply": (relay.multiply, lambda x, y: [y, x]),
        "divide": (relay.divide, lambda x, y: [1 / y, -x / (y**2)]),
    }

    relay_op, ref_func = tvm.testing.parameters(*config.values(), ids=config.keys())
    dtype = tvm.testing.parameter("float32", "float64")
    shape = tvm.testing.parameter((5, 10, 5))

    def test_binary_op(self, target, dev, executor_kind, relay_op, ref_func, shape, dtype):
        t = relay.TensorType(shape, dtype=dtype)
        x = relay.var("x", t)
        y = relay.var("y", t)
        z = relay_op(x, y)

        x_data = np.random.rand(*shape).astype(t.dtype)
        y_data = np.random.rand(*shape).astype(t.dtype)
        ref_grad0, ref_grad1 = ref_func(x_data, y_data)
        fwd_func = relay.Function([x, y], z)
        fwd_func = run_infer_type(fwd_func)
        bwd_func = run_infer_type(gradient(fwd_func))

        op_res, (op_grad0, op_grad1) = relay.create_executor(
            executor_kind, device=dev, target=target
        ).evaluate(bwd_func)(x_data, y_data)
        np.testing.assert_allclose(op_grad0.numpy(), ref_grad0, rtol=0.01)
        np.testing.assert_allclose(op_grad1.numpy(), ref_grad1, rtol=0.01)


def test_softmax_grad(executor_kind, target, dev):
    target = tvm.target.Target(target)
    if target.kind.name == "vulkan":
        pytest.xfail("Known failure on vulkan")

    data = relay.var("data", relay.TensorType((1, 16), "float64"))
    fwd_func = relay.Function([data], relay.nn.softmax(data))
    check_grad(fwd_func, scale=1, target_devices=[(target, dev)], executor_kind=executor_kind)


def test_log_softmax_grad(executor_kind, target, dev):
    target = tvm.target.Target(target)
    if target.kind.name == "vulkan":
        pytest.xfail("Known failure on vulkan")

    data = relay.var("data", relay.TensorType((2, 16), "float64"))
    fwd_func = relay.Function([data], relay.nn.log_softmax(data))
    check_grad(fwd_func, scale=1, target_devices=[(target, dev)], executor_kind=executor_kind)


class TestBiasAddGrad:
    d_shape, b_shape, axis = tvm.testing.parameters(
        ((1, 16), (16,), 1),
        ((1, 8, 2, 2), (8,), 1),
        ((1, 2, 2, 8), (8,), 3),
        ((4, 8), (8,), 1),
    )

    def test_bias_add(self, executor_kind, target, dev, d_shape, b_shape, axis):
        data = relay.var("data", relay.TensorType(d_shape, "float32"))
        bias = relay.var("bias", relay.TensorType(b_shape, "float32"))
        fwd_func = relay.Function([data, bias], relay.nn.bias_add(data, bias, axis=axis))
        check_grad(fwd_func, target_devices=[(target, dev)], executor_kind=executor_kind)


def test_expand_dims_grad(executor_kind, target, dev):
    data = relay.var("data", shape=(2, 3), dtype="float64")
    fwd_func = relay.Function([data], relay.expand_dims(data, axis=1, num_newaxis=2))
    check_grad(fwd_func, target_devices=[(target, dev)], executor_kind=executor_kind)


def test_concatenate_grad(executor_kind, target, dev):
    x = relay.var("x", shape=(2, 2, 5))
    y = relay.var("y", shape=(2, 1, 5))
    z = relay.var("z", shape=(2, 4, 5))
    fwd_func = relay.Function([x, y, z], relay.concatenate([x, y, z], axis=1))
    check_grad(fwd_func, target_devices=[(target, dev)], executor_kind=executor_kind)


if __name__ == "__main__":
    tvm.testing.main()
