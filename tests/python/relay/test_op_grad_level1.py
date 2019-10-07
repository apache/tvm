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
from tvm import relay
from tvm.relay.testing import check_grad, ctx_list, run_infer_type
from tvm.relay.transform import gradient


def sigmoid(x):
    one = np.ones_like(x)
    return one / (one + np.exp(-x))


def relu(x):
    x_copy = np.copy(x)
    np.maximum(x_copy, 0, x_copy)
    return x_copy


def test_unary_op():
    def check_single_op(opfunc, ref):
        shape = (10, 4)
        dtype = 'float32'
        tp = relay.TensorType(shape, dtype)
        x = relay.var("x", tp)
        y = opfunc(x)

        if ref is not None:
            data = np.random.rand(*shape).astype(dtype)
            ref_grad = ref(data)
            fwd_func = relay.Function([x], y)
            fwd_func = run_infer_type(fwd_func)
            bwd_func = run_infer_type(gradient(fwd_func))

            for target, ctx in ctx_list():
                intrp = relay.create_executor(ctx=ctx, target=target)
                op_res, (op_grad, ) = intrp.evaluate(bwd_func)(data)
                np.testing.assert_allclose(op_grad.asnumpy(), ref_grad, rtol=0.01)

    for opfunc, ref in [(tvm.relay.log, lambda x: 1 / x),
                        (tvm.relay.exp, np.exp),
                        (tvm.relay.sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x))),
                        (tvm.relay.tanh, lambda x: 1 - np.tanh(x) * np.tanh(x)),
                        (tvm.relay.sqrt, lambda x: 0.5 * np.power(x, -0.5)),
                        (tvm.relay.abs, lambda x: np.where(x < 0, -np.ones_like(x), np.ones_like(x))),
                        (relay.nn.relu, lambda x: np.where(x < 0, np.zeros_like(x), np.ones_like(x))),
                        (tvm.relay.cos, lambda x: -1.0 * np.sin(x)),
                        (tvm.relay.sin, lambda x: np.cos(x)),
                        (tvm.relay.atan, lambda x: 1 / (1 + np.power(x, 2.0)))]:
        check_single_op(opfunc, ref)


def test_binary_op():
    def inst(vars, sh):
        return [vars.get(s, s) for s in sh]

    def check_binary_op(opfunc, ref):
        s = (5, 10, 5)
        t = relay.TensorType((5, 10, 5))
        x = relay.var("x", t)
        y = relay.var("y", t)
        z = opfunc(x, y)

        x_data = np.random.rand(*s).astype(t.dtype)
        y_data = np.random.rand(*s).astype(t.dtype)
        ref_grad0, ref_grad1 = ref(x_data, y_data)
        fwd_func = relay.Function([x, y], z)
        fwd_func = run_infer_type(fwd_func)
        bwd_func = run_infer_type(gradient(fwd_func))

        for target, ctx in ctx_list():
            intrp = relay.create_executor(ctx=ctx, target=target)
            op_res, (op_grad0, op_grad1) = intrp.evaluate(bwd_func)(x_data, y_data)
            np.testing.assert_allclose(op_grad0.asnumpy(), ref_grad0, rtol=0.01)
            np.testing.assert_allclose(op_grad1.asnumpy(), ref_grad1, rtol=0.01)

    for opfunc, ref in [(relay.add, lambda x, y: [np.ones_like(x), np.ones_like(y)]),
                        (relay.subtract, lambda x, y: [np.ones_like(x), -np.ones_like(y)]),
                        (relay.multiply, lambda x, y: [y, x]),
                        (relay.divide, lambda x, y: [1 / y, - x / (y**2)])]:
        check_binary_op(opfunc, ref)


def test_softmax_grad():
    data = relay.var("data", relay.TensorType((1, 16), "float64"))
    fwd_func = relay.Function([data], relay.nn.softmax(data))
    check_grad(fwd_func, scale=1)


def test_log_softmax_grad():
    data = relay.var("data", relay.TensorType((2, 16), "float64"))
    fwd_func = relay.Function([data], relay.nn.log_softmax(data))
    check_grad(fwd_func, scale=1)


def test_bias_add_grad():
    data = relay.var("data", relay.TensorType((1, 16), "float32"))
    bias = relay.var("bias", relay.TensorType((16,), "float32"))
    fwd_func = relay.Function([data, bias], relay.nn.bias_add(data, bias))
    check_grad(fwd_func)


if __name__ == "__main__":
    pytest.main([__file__])
