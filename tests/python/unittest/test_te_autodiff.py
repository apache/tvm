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
from tvm import te, topi
from tvm.testing import assert_allclose
from tvm.topi.utils import get_const_tuple


def check_grad(
    out, inputs, args=[], data_range=(-10, 10), desired_grads=None, assert_no_jacobian=True
):
    inputs = inputs if isinstance(inputs, list) else [inputs]

    def check_device(device, host="llvm"):
        dev = tvm.device(device, 0)
        if not tvm.testing.device_enabled(host):
            return

        sout = te.create_schedule(out.op)
        mout = tvm.build(sout, [out] + inputs + args)
        out_shape = get_const_tuple(out.shape)

        l, h = data_range
        input_data = [
            tvm.nd.array(
                np.random.uniform(l, h, size=get_const_tuple(input.shape)).astype(input.dtype)
            )
            for input in inputs
        ]
        arg_vals = [
            tvm.nd.array(np.random.uniform(l, h, size=get_const_tuple(arg.shape)).astype(arg.dtype))
            for arg in args
        ]

        ones = topi.full_like(out, 1.0)
        # we provide head to sum and reduce the output dimension,
        # which equals to grad(out.sum(), inputs)
        grads = te.gradient(out, inputs, head=ones)
        grad_sched = te.create_schedule([grad.op for grad in grads])
        mgrad = tvm.build(grad_sched, list(grads) + inputs + args)
        if assert_no_jacobian:
            # TODO(yzhliu): it is better to visit the expression and do assertion
            lowered_ir = str(tvm.lower(grad_sched, list(grads) + inputs + args, simple_mode=True))
            assert "jacobian" not in lowered_ir, lowered_ir

        grad_data = [tvm.nd.empty(get_const_tuple(i.shape), g.dtype) for i, g in zip(inputs, grads)]

        mgrad(*grad_data, *input_data, *arg_vals)
        g_res = [g.numpy() for g in grad_data]

        if desired_grads:
            assert isinstance(desired_grads, list)
            for actual, desired in zip(g_res, desired_grads):
                assert_allclose(actual, desired, rtol=0.1, atol=1e-2)
        else:

            def forward(*in_data):
                out_data = tvm.nd.empty(out_shape, out.dtype)
                mout(out_data, *[tvm.nd.array(d) for d in list(in_data)])
                return out_data.numpy().sum()

            tvm.testing.check_numerical_grads(
                forward, [d.numpy() for d in input_data + arg_vals], g_res
            )

    check_device("cpu")


def test_basic_operation():
    np.random.seed(0)
    shape = (10, 10)
    x = te.var("x", dtype="float32")
    k = te.reduce_axis((0, 10), name="k")
    l = te.reduce_axis((0, 10), name="l")
    A0 = te.placeholder(shape, name="A0")
    A1 = te.placeholder(shape, name="A1")
    zeros = np.zeros(shape)

    B = te.compute(shape, lambda i, j: A0[i, j], name="B")
    check_grad(B, [A0])

    B = te.compute(shape, lambda i, j: A0[i, j] + A1[i, j], name="B")
    check_grad(B, [A0, A1])

    B = te.compute(shape, lambda i, j: A0[i, j] + A0[j, i], name="B")
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: te.floor(A0[i, j]), name="B")
    check_grad(B, A0, desired_grads=[zeros])

    B = te.compute(shape, lambda i, j: te.ceil(A0[i, j]), name="B")
    check_grad(B, A0, desired_grads=[zeros])

    B = te.compute(shape, lambda i, j: te.trunc(A0[i, j]), name="B")
    check_grad(B, A0, desired_grads=[zeros])

    B = te.compute(shape, lambda i, j: te.round(A0[i, j]), name="B")
    check_grad(B, A0, desired_grads=[zeros])

    B = te.compute(shape, lambda i, j: A0[i, j] + te.exp(A0[j, i]), name="B")
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: te.log(0.1 + te.abs(A0[i, j] + te.exp(A0[j, i]))), name="B")
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: te.sigmoid(A0[i, j] * A0[i, j] * A0[j, i]), name="B")
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: te.tanh(A0[i, j] * A0[i, j] * A0[j, i]), name="B")
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: te.sqrt(A0[i, j] * A0[i, j] * A0[j, i]), name="B")
    check_grad(B, A0, data_range=(0.1, 10))

    B = te.compute(shape, lambda i, j: te.power(te.abs(A0[i, j]), A0[j, i]), name="B")
    check_grad(B, A0, data_range=(-4, 4))

    B = te.compute(shape, lambda i, j: A0[i, j] * A0[j, i], name="B")
    check_grad(B, A0)

    B = te.compute((10,), lambda i: te.sum(A0[i, k] * A0[k, i], axis=k), name="B")
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: te.sum(A0[i, k] * A0[k, i] + 5, axis=k), name="B")
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: te.max(A0[i, k] * A0[k, j] + 5, axis=k), name="B")
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: A0[i, j] * (A1[j, i] + A0[j, i]), name="B")
    check_grad(B, [A0, A1])

    B = te.compute(
        shape, lambda i, j: te.sum(A0[k, k] - A0[te.min(j + k, 9), j] * A0[i, k], axis=k), name="B"
    )
    check_grad(B, A0)

    def fcombine(x, y):
        return x * y

    def fidentity(t0):
        return tvm.tir.const(1, t0)

    prod = te.comm_reducer(fcombine, fidentity, name="prod")
    B = te.compute((10, 10), lambda i, j: prod(A0[i, k] + A0[k, i], axis=k), name="B")
    check_grad(B, A0)

    X = te.placeholder((10,), name="X")
    A = te.compute((10,), lambda i: X[i] + X[9 - i])
    B = te.compute((10,), lambda i: X[i] * X[9 - i])
    Y = topi.tensordot(A, B, 1)
    check_grad(Y, X)

    X = te.placeholder((3, 3), name="X")
    Y = topi.einsum("ii->i", (X))
    check_grad(Y, X)


def test_topi():
    X = te.placeholder((1, 2, 4, 4), name="X")
    W = te.placeholder((5, 2, 3, 3), name="W")
    W1 = te.placeholder((2, 5, 3, 3), name="W1")
    W2 = te.placeholder((1,), name="W2")

    R = topi.nn.conv2d(X, W, 1, 1, 1)
    check_grad(R, [X, W])

    R1 = topi.nn.conv2d(topi.nn.relu(R), W1, 1, 0, 1)
    check_grad(R1, [X, W, W1])

    R = topi.broadcast_to(W2, (5, 2, 3, 3))
    check_grad(R, [W2])

    R = topi.nn.conv2d(X, topi.broadcast_to(W2, (5, 2, 3, 3)), 1, 1, 1)
    check_grad(R, [X, W2])

    R = topi.nn.pool2d(X, [2, 2], [1, 1], [2, 2], [0, 0, 0, 0], "avg")
    check_grad(R, X)

    R = topi.nn.pool2d(X, [2, 2], [1, 1], [2, 2], [0, 0, 0, 0], "max")
    check_grad(R, X)

    X = te.placeholder((1, 2, 5, 5), name="X")
    R = topi.reshape(X, (1, 32))
    check_grad(R, [X])

    X = te.placeholder((1, 2, 5, 5), name="X")
    W = te.placeholder((2, 2, 3, 3), name="W")

    S = topi.reshape(X, (1, 50))
    check_grad(S, [X])

    R = X + topi.nn.conv2d(X + topi.nn.conv2d(X, W, 1, 1, 1), W, 1, 1, 1)
    check_grad(R, [X, W])

    S = topi.nn.softmax(topi.reshape(R, (1, 50)))
    check_grad(S, [X, W])

    S = topi.sigmoid(topi.reshape(R, (1, 50)))
    check_grad(S, [X, W])

    S = topi.tanh(topi.reshape(R, (1, 50)))
    check_grad(S, [X, W])

    S = topi.nn.log_softmax(topi.reshape(R, (1, 50)))
    check_grad(S, [X, W])
    check_grad(S, [W], [X])

    X = te.placeholder((1, 2, 3, 5), name="X")
    Y = te.placeholder((1, 2, 7, 5), name="Y")
    S = topi.concatenate((X, Y), 2)
    check_grad(S, [X, Y])

    X = te.placeholder((1, 2, 6, 5), name="X")
    (S, R) = topi.split(X, 2, 2)
    check_grad(S, [X])
    check_grad(R, [X])
    R1 = topi.concatenate((S, R), 2)
    check_grad(R1, [X])
    R2 = topi.concatenate((R, S), 2)
    check_grad(R2, [X])

    X = te.placeholder((4, 5), name="X")
    I = te.placeholder((100,), name="I", dtype="int32")
    R = topi.take(X, topi.abs(I))
    check_grad(R, [X], [I])

    W = te.placeholder((5, 5), name="W")
    exps = topi.exp(topi.nn.dense(X, W))
    sumexps = topi.sum(exps, axis=-1, keepdims=True)
    R = exps / sumexps
    check_grad(R, [X, W], data_range=(-1, 1))


def test_stride_dilation():
    X = te.placeholder((1, 2, 10, 10), name="X")
    W = te.placeholder((2, 2, 1, 1), name="W")

    Y = topi.nn.conv2d(X, W, 1, 0, 1)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 2, 0, 1)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 3, 0, 1)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 1, 0, 2)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 2, 0, 2)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 3, 0, 2)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 1, 0, 3)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 2, 0, 3)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 3, 0, 3)
    check_grad(Y, [X, W])

    W = te.placeholder((2, 2, 2, 2), name="W")

    Y = topi.nn.conv2d(X, W, 1, 0, 1)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 2, 0, 1)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 3, 0, 1)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 1, 0, 2)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 2, 0, 2)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 3, 0, 2)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 1, 0, 3)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 2, 0, 3)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 3, 0, 3)
    check_grad(Y, [X, W])

    W = te.placeholder((2, 2, 3, 3), name="W")

    Y = topi.nn.conv2d(X, W, 1, 0, 1)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 2, 0, 1)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 3, 0, 1)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 1, 0, 2)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 2, 0, 2)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 3, 0, 2)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 1, 0, 3)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 2, 0, 3)
    check_grad(Y, [X, W])
    Y = topi.nn.conv2d(X, W, 3, 0, 3)
    check_grad(Y, [X, W])

    Y = topi.nn.pool2d(X, [1, 1], [1, 1], [1, 1], [0, 0, 0, 0], "max")
    check_grad(Y, [X])
    Y = topi.nn.pool2d(X, [1, 1], [1, 1], [2, 2], [0, 0, 0, 0], "max")
    check_grad(Y, [X])
    Y = topi.nn.pool2d(X, [1, 1], [1, 1], [3, 3], [0, 0, 0, 0], "max")
    check_grad(Y, [X])
    Y = topi.nn.pool2d(X, [2, 2], [1, 1], [1, 1], [0, 0, 0, 0], "max")
    check_grad(Y, [X])
    Y = topi.nn.pool2d(X, [2, 2], [1, 1], [2, 2], [0, 0, 0, 0], "max")
    check_grad(Y, [X])
    Y = topi.nn.pool2d(X, [2, 2], [1, 1], [3, 3], [0, 0, 0, 0], "max")
    check_grad(Y, [X])
    Y = topi.nn.pool2d(X, [3, 3], [1, 1], [1, 1], [0, 0, 0, 0], "max")
    check_grad(Y, [X])
    Y = topi.nn.pool2d(X, [3, 3], [1, 1], [2, 2], [0, 0, 0, 0], "max")
    check_grad(Y, [X])
    Y = topi.nn.pool2d(X, [3, 3], [1, 1], [3, 3], [0, 0, 0, 0], "max")
    check_grad(Y, [X])


@pytest.mark.xfail
def test_reduction_init():
    np.random.seed(0)
    shape = (10, 10)
    k = te.reduce_axis((0, 10), name="k")
    A0 = te.placeholder(shape, name="A0")

    B = te.compute((10,), lambda i: te.sum(A0[i, k] * A0[k, i], axis=k, init=0.0), name="B")
    check_grad(B, A0)


if __name__ == "__main__":
    test_basic_operation()
    test_topi()
    test_stride_dilation()
