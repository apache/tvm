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
import numpy as np
import torch
from tvm import relay
from tvm.relay.ir_pass import gradient, infer_type
from tvm.relay.testing import ctx_list

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
            bwd_func = infer_type(gradient(fwd_func))

            for target, ctx in ctx_list():
                intrp = relay.create_executor(ctx=ctx, target=target)
                op_res, (op_grad, ) = intrp.evaluate(bwd_func)(data)
                np.testing.assert_allclose(op_grad.asnumpy(), ref_grad, rtol=0.01)

    for opfunc, ref in [(relay.log, lambda x: 1 / x),
                        (relay.exp, np.exp),
                        (relay.sigmoid, lambda x: sigmoid(x) * (1 - sigmoid(x))),
                        (relay.tanh, lambda x: 1 - np.tanh(x) * np.tanh(x)),
                        (relay.sqrt, lambda x: 0.5 * np.power(x, -0.5)),
                        (relay.nn.relu, lambda x: np.where(x < 0, np.zeros_like(x), np.ones_like(x))),
                        (relay.negative, lambda x: -1 * np.ones_like(x)),
                        (relay.zeros_like, lambda x: np.zeros_like(x)),
                        (relay.ones_like, lambda x: np.zeros_like(x)),
                        (relay.shape_of, lambda x: np.zeros_like(x))]:
        check_single_op(opfunc, ref)


def test_binary_op():
    def inst(vars, sh):
        return [vars.get(s, s) for s in sh]

    def check_binary_op(opfunc, ref):
        s = (5, 10, 5)
        t = relay.TensorType(s)
        x = relay.var("x", t)
        y = relay.var("y", t)
        z = opfunc(x, y)

        x_data = np.random.rand(*s).astype(t.dtype)
        y_data = np.random.rand(*s).astype(t.dtype)
        ref_grad0, ref_grad1 = ref(x_data, y_data)
        fwd_func = relay.Function([x, y], z)
        bwd_func = infer_type(gradient(fwd_func))

        for target, ctx in ctx_list():
            intrp = relay.create_executor(ctx=ctx, target=target)
            op_res, (op_grad0, op_grad1) = intrp.evaluate(bwd_func)(x_data, y_data)
            np.testing.assert_allclose(op_grad0.asnumpy(), ref_grad0, rtol=0.01)
            np.testing.assert_allclose(op_grad1.asnumpy(), ref_grad1, rtol=0.01)

    for opfunc, ref in [(relay.add, lambda x, y: [np.ones_like(x), np.ones_like(y)]),
                        (relay.subtract, lambda x, y: [np.ones_like(x), -np.ones_like(y)]),
                        (relay.multiply, lambda x, y: [y, x]),
                        (relay.divide, lambda x, y: [1 / y, - x / (y**2)]),
                        (relay.collapse_sum_like, lambda x, y: [np.ones_like(x), np.zeros_like(y)])]:
        check_binary_op(opfunc, ref)

def test_reduce_op():
    def softmax_grad_torch(x_data, axis):
        x = torch.tensor(x_data, requires_grad=True)
        sm = torch.nn.Softmax(axis)(x)
        sm.backward(torch.ones(x.shape))
        return x.grad.numpy()

    def max_grad_torch(x_data, axis):
        x = torch.tensor(x_data, requires_grad=True)
        m = x.max(axis).values
        m.backward(torch.ones(m.shape))
        return x.grad.numpy()

    def check_reduce_op(opfunc, ref):
        s = (5, 10, 5)
        t = relay.TensorType(s)
        x = relay.var("x", t)
        axis = 0
        z = opfunc(x, axis)

        x_data = np.random.randn(*s).astype(t.dtype)
        ref_grad = ref(x_data, axis)
        fwd_func = relay.Function([x], z)
        bwd_func = infer_type(gradient(fwd_func))

        for target, ctx in ctx_list():
            intrp = relay.create_executor(ctx=ctx, target=target)
            op_res, (op_grad,) = intrp.evaluate(bwd_func)(x_data)
            np.testing.assert_allclose(op_grad.asnumpy(), ref_grad, rtol=0.01, atol=10e-8)

    for opfunc, ref in [(relay.nn.softmax, softmax_grad_torch),
                        (relay.max, max_grad_torch),
                        (tvm.relay.sum, lambda x, axis: np.ones_like(x))]:
        check_reduce_op(opfunc, ref)

if __name__ == "__main__":
    test_unary_op()
    test_binary_op()
    test_reduce_op()
