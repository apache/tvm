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
from tvm import te
from tvm.testing import check_numerical_grads, assert_allclose
from tvm import topi
from tvm.topi.util import get_const_tuple

import numpy as np


def check_grad(out, inputs, data_range=(-10, 10), desired_grads=None):
    inputs = inputs if isinstance(inputs, list) else [inputs]

    def check_device(device, host="llvm"):
        ctx = tvm.context(device, 0)
        if not tvm.runtime.enabled(host):
            return
        if not ctx.exist:
            print("skip because %s is not enabled.." % device)
            return

        sout = te.create_schedule(out.op)
        mout = tvm.build(sout, [out] + inputs)
        out_shape = get_const_tuple(out.shape)

        l, h = data_range
        input_data = [tvm.nd.array(
            np.random.uniform(l, h, size=get_const_tuple(input.shape)).astype(input.dtype))
            for input in inputs]

        ones = topi.full_like(out, 1.0)
        # we provide head to sum and reduce the output dimension,
        # which equals to grad(out.sum(), inputs)
        grads = te.gradient(out, inputs, head=ones)
        grad_sched = te.create_schedule([grad.op for grad in grads])
        mgrad = tvm.build(grad_sched, list(grads) + inputs)
        # print(tvm.lower(grad_sched, list(grads) + inputs, simple_mode=True))

        grad_data = [tvm.nd.empty(get_const_tuple(i.shape), g.dtype)
                     for i, g in zip(inputs, grads)]

        mgrad(*grad_data, *input_data)
        g_res = [g.asnumpy() for g in grad_data]

        if desired_grads:
            assert isinstance(desired_grads, list)
            for actual, desired in zip(g_res, desired_grads):
                assert_allclose(actual, desired, rtol=0.1, atol=1e-2)
        else:
            def forward(*in_data):
                out_data = tvm.nd.empty(out_shape, out.dtype)
                mout(out_data, *[tvm.nd.array(d) for d in list(in_data)])
                return out_data.asnumpy().sum()
            check_numerical_grads(forward, [d.asnumpy() for d in input_data], g_res)

    check_device("cpu")


def test_basic_operation():
    np.random.seed(0)
    shape = (10, 10)
    x = te.var("x", dtype='float32')
    k = te.reduce_axis((0, 10), name="k")
    l = te.reduce_axis((0, 10), name="l")
    A0 = te.placeholder(shape, name='A0')
    A1 = te.placeholder(shape, name='A1')
    zeros = np.zeros(shape)

    B = te.compute(shape, lambda i, j: A0[i, j], name='B')
    check_grad(B, [A0])

    B = te.compute(shape, lambda i, j: A0[i, j] + A1[i, j], name='B')
    check_grad(B, [A0, A1])

    B = te.compute(shape, lambda i, j: A0[i, j] + A0[j, i], name='B')
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: te.floor(A0[i, j]), name='B')
    check_grad(B, A0, desired_grads=[zeros])

    B = te.compute(shape, lambda i, j: te.ceil(A0[i, j]), name='B')
    check_grad(B, A0, desired_grads=[zeros])

    B = te.compute(shape, lambda i, j: te.trunc(A0[i, j]), name='B')
    check_grad(B, A0, desired_grads=[zeros])

    B = te.compute(shape, lambda i, j: te.round(A0[i, j]), name='B')
    check_grad(B, A0, desired_grads=[zeros])

    B = te.compute(shape, lambda i, j: A0[i, j] + te.exp(A0[j, i]), name='B')
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: te.log(0.1 + te.abs(A0[i, j] + te.exp(A0[j, i]))), name='B')
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: te.sigmoid(A0[i, j]*A0[i, j]*A0[j, i]), name='B')
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: te.tanh(A0[i, j]*A0[i, j]*A0[j, i]), name='B')
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: te.sqrt(A0[i, j]*A0[i, j]*A0[j, i]), name='B')
    check_grad(B, A0, data_range=(0.1, 10))

    B = te.compute(shape, lambda i, j: te.power(te.abs(A0[i, j]), A0[j, i]), name='B')
    check_grad(B, A0, data_range=(-4, 4))

    B = te.compute(shape, lambda i, j: A0[i, j] * A0[j, i], name='B')
    check_grad(B, A0)

    B = te.compute((10,), lambda i: te.sum(A0[i, k]*A0[k, i], axis=k), name='B')
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: te.sum(A0[i, k]*A0[k, i] + 5, axis=k), name='B')
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: te.max(A0[i, k]*A0[k, j] + 5, axis=k), name='B')
    check_grad(B, A0)

    B = te.compute(shape, lambda i, j: A0[i, j] * (A1[j, i] + A0[j, i]), name='B')
    check_grad(B, [A0, A1])

    B = te.compute(shape, lambda i, j: te.sum(A0[k, k] -
                                              A0[te.min(j + k, 9), j]*A0[i, k],
                                              axis=k), name='B')
    check_grad(B, A0)

    def fcombine(x, y):
        return x*y

    def fidentity(t0):
        return tvm.tir.const(1, t0)

    prod = te.comm_reducer(fcombine, fidentity, name='prod')
    B = te.compute((10, 10), lambda i, j: prod(A0[i, k] + A0[k, i], axis=k), name='B')
    check_grad(B, A0)

    X = te.placeholder((10,), name='X')
    A = te.compute((10,), lambda i: X[i] + X[9 - i])
    B = te.compute((10,), lambda i: X[i] * X[9 - i])
    Y = topi.tensordot(A, B, 1)
    check_grad(Y, X)


def test_conv2d():
    np.random.seed(0)
    X = te.placeholder((1, 2, 4, 4), name='X')
    W = te.placeholder((5, 2, 3, 3), name='W')

    R = topi.nn.conv2d(X, W, 1, 1, 1)
    check_grad(R, [X, W])


if __name__ == "__main__":
    test_basic_operation()
    test_conv2d()
