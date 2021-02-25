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
from tvm import te
from tvm import relay
from tvm.relay.testing import check_grad, run_infer_type
from tvm.relay.transform import gradient
import tvm.testing


def sigmoid(x):
    one = np.ones_like(x)
    return one / (one + np.exp(-x))


def relu(x):
    x_copy = np.copy(x)
    np.maximum(x_copy, 0, x_copy)
    return x_copy


@tvm.testing.uses_gpu
def test_unary_op():
    def check_single_op(opfunc, ref, dtype):
        shape = (10, 4)
        tp = relay.TensorType(shape, dtype)
        x = relay.var("x", tp)
        g = relay.var("g", tp)
        y = opfunc(x) * g

        if ref is not None:
            data = np.random.rand(*shape).astype(dtype)
            grad_in = np.random.rand(*shape).astype(dtype)
            ref_grad = ref(data, grad_in)
            fwd_func = relay.Function([x, g], y)
            fwd_func = run_infer_type(fwd_func)
            bwd_func = run_infer_type(gradient(fwd_func))

            for target, ctx in tvm.testing.enabled_targets():
                intrp = relay.create_executor(ctx=ctx, target=target)
                op_res, (op_grad, _) = intrp.evaluate(bwd_func)(data, grad_in)
                np.testing.assert_allclose(op_grad.asnumpy(), ref_grad, rtol=0.01)

    for opfunc, ref in [
        (tvm.relay.log, lambda x, g: g * (1 / x)),
        (tvm.relay.exp, lambda x, g: g * np.exp(x)),
        (tvm.relay.sigmoid, lambda x, g: g * sigmoid(x) * (1 - sigmoid(x))),
        (tvm.relay.tanh, lambda x, g: g * (1 - np.tanh(x) * np.tanh(x))),
        (tvm.relay.sqrt, lambda x, g: g * 0.5 * np.power(x, -0.5)),
        (tvm.relay.abs, lambda x, g: np.where(x < 0, -g, g)),
        (relay.nn.relu, lambda x, g: np.where(x < 0, np.zeros_like(x), g)),
        (tvm.relay.erf, lambda x, g: g * (2.0 / (np.pi ** (0.5)) * np.exp(-x * x))),
        (tvm.relay.cos, lambda x, g: g * -1.0 * np.sin(x)),
        (tvm.relay.sin, lambda x, g: g * np.cos(x)),
        (tvm.relay.tan, lambda x, g: g * (1.0 / (np.cos(x) ** 2))),
        (tvm.relay.atan, lambda x, g: g * (1 / (1 + np.power(x, 2.0)))),
        (tvm.relay.log2, lambda x, g: g * (1 / (np.log(2) * x))),
        (tvm.relay.log10, lambda x, g: g * (1 / (np.log(10) * x))),
        (tvm.relay.cosh, lambda x, g: g * (np.sinh(x))),
        (tvm.relay.sinh, lambda x, g: g * (np.cosh(x))),
        (tvm.relay.asin, lambda x, g: g * (1.0 / (1.0 - x ** 2) ** (1.0 / 2.0))),
        (tvm.relay.acos, lambda x, g: g * (-1.0 / (1.0 - x ** 2.0) ** (1.0 / 2.0))),
        (tvm.relay.acosh, lambda x, g: g * (1.0 / (x ** 2 - 1.0) ** (1.0 / 2.0))),
        (tvm.relay.asinh, lambda x, g: g * (1.0 / (x ** 2 + 1.0) ** (1.0 / 2.0))),
        (tvm.relay.atanh, lambda x, g: g * (-1.0 / (x ** 2 - 1.0))),
    ]:
        for dtype in ("float32", "float64"):
            check_single_op(opfunc, ref, dtype)


@tvm.testing.uses_gpu
def test_binary_op():
    def inst(vars, sh):
        return [vars.get(s, s) for s in sh]

    def check_binary_op(opfunc, ref, dtype):
        s = (5, 10, 5)
        t = relay.TensorType((5, 10, 5), dtype=dtype)
        x = relay.var("x", t)
        y = relay.var("y", t)
        z = opfunc(x, y)

        x_data = np.random.rand(*s).astype(t.dtype)
        y_data = np.random.rand(*s).astype(t.dtype)
        ref_grad0, ref_grad1 = ref(x_data, y_data)
        fwd_func = relay.Function([x, y], z)
        fwd_func = run_infer_type(fwd_func)
        bwd_func = run_infer_type(gradient(fwd_func))

        for target, ctx in tvm.testing.enabled_targets():
            intrp = relay.create_executor(ctx=ctx, target=target)
            op_res, (op_grad0, op_grad1) = intrp.evaluate(bwd_func)(x_data, y_data)
            np.testing.assert_allclose(op_grad0.asnumpy(), ref_grad0, rtol=0.01)
            np.testing.assert_allclose(op_grad1.asnumpy(), ref_grad1, rtol=0.01)

    for opfunc, ref in [
        (relay.add, lambda x, y: [np.ones_like(x), np.ones_like(y)]),
        (relay.subtract, lambda x, y: [np.ones_like(x), -np.ones_like(y)]),
        (relay.multiply, lambda x, y: [y, x]),
        (relay.divide, lambda x, y: [1 / y, -x / (y ** 2)]),
    ]:
        for dtype in ("float32", "float64"):
            check_binary_op(opfunc, ref, dtype)


def test_softmax_grad():
    data = relay.var("data", relay.TensorType((1, 16), "float64"))
    fwd_func = relay.Function([data], relay.nn.softmax(data))
    check_grad(fwd_func, scale=1)


def test_log_softmax_grad():
    data = relay.var("data", relay.TensorType((2, 16), "float64"))
    fwd_func = relay.Function([data], relay.nn.log_softmax(data))
    check_grad(fwd_func, scale=1)


def verify_bias_add(d_shape, b_shape, axis=1):
    data = relay.var("data", relay.TensorType(d_shape, "float32"))
    bias = relay.var("bias", relay.TensorType(b_shape, "float32"))
    fwd_func = relay.Function([data, bias], relay.nn.bias_add(data, bias, axis=axis))
    check_grad(fwd_func)


def test_bias_add_grad():
    verify_bias_add((1, 16), (16,))
    verify_bias_add((1, 8, 2, 2), (8,))
    verify_bias_add((1, 2, 2, 8), (8,), 3)
    verify_bias_add((4, 8), (8,))


def test_expand_dims_grad():
    data = relay.var("data", shape=(2, 3), dtype="float64")
    fwd_func = relay.Function([data], relay.expand_dims(data, axis=1, num_newaxis=2))
    check_grad(fwd_func)


def test_concatenate_grad():
    x = relay.var("x", shape=(2, 2, 5))
    y = relay.var("y", shape=(2, 1, 5))
    z = relay.var("z", shape=(2, 4, 5))
    fwd_func = relay.Function([x, y, z], relay.concatenate([x, y, z], axis=1))
    check_grad(fwd_func)


if __name__ == "__main__":
    pytest.main([__file__])
