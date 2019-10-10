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
import tvm
import topi
import topi.testing
from topi.util import get_const_tuple


def test_operator_type_and_tags():
    k = 1
    n = tvm.var('n')
    A = tvm.placeholder((), name='A')
    B = tvm.placeholder((10, 5), name='B')
    B1 = B[0]
    B2 = B[0,0]

    assert isinstance(k + n, tvm.expr.Expr)
    assert isinstance(n + n, tvm.expr.Expr)
    assert isinstance(k + A, tvm.tensor.Tensor)
    assert isinstance(A + k, tvm.tensor.Tensor)
    assert isinstance(n + A, tvm.tensor.Tensor)
    assert isinstance(A + n, tvm.tensor.Tensor)
    assert isinstance(A + A, tvm.tensor.Tensor)

    assert isinstance(k + B, tvm.tensor.Tensor)
    assert isinstance(B + k, tvm.tensor.Tensor)
    assert isinstance(n + B, tvm.tensor.Tensor)
    assert isinstance(B + n, tvm.tensor.Tensor)
    assert isinstance(A + B, tvm.tensor.Tensor)
    assert isinstance(B + A, tvm.tensor.Tensor)
    assert isinstance(B + B, tvm.tensor.Tensor)

    assert (k + B).op.tag == topi.tag.ELEMWISE
    assert (B + k).op.tag == topi.tag.ELEMWISE
    assert (n + B).op.tag == topi.tag.ELEMWISE
    assert (B + n).op.tag == topi.tag.ELEMWISE
    assert (A + B).op.tag == topi.tag.BROADCAST
    assert (B + A).op.tag == topi.tag.BROADCAST
    assert (B + B).op.tag == topi.tag.BROADCAST

    assert isinstance(k + B2, tvm.expr.Expr)
    assert isinstance(B2 + k, tvm.expr.Expr)
    assert isinstance(n + B2, tvm.expr.Expr)
    assert isinstance(B2 + n, tvm.expr.Expr)
    assert isinstance(B2 + B2, tvm.expr.Expr)
    assert isinstance(B2 + A, tvm.tensor.Tensor)
    assert isinstance(A + B2, tvm.tensor.Tensor)
    assert isinstance(B2 + B, tvm.tensor.Tensor)
    assert isinstance(B + B2, tvm.tensor.Tensor)


def test_combination():
    k = 3
    n = 5
    m = 10
    x = tvm.var('x')
    A = tvm.placeholder((n, m), name='A')
    B = tvm.placeholder((n, m), name='B')
    C = tvm.placeholder((n, m), name='C')
    D = k + A - B * C + x
    s = tvm.create_schedule(D.op)
    foo = tvm.build(s, [x, A, B, C, D], "llvm")
    ctx = tvm.cpu(0)
    x = 2
    a = tvm.nd.array(np.random.uniform(size=(n, m)).astype(A.dtype), ctx)
    b = tvm.nd.array(np.random.uniform(size=(n, m)).astype(B.dtype), ctx)
    c = tvm.nd.array(np.random.uniform(size=(n, m)).astype(C.dtype), ctx)
    d = tvm.nd.array(np.zeros((n, m), dtype=D.dtype), ctx)
    foo(x, a, b, c, d)
    tvm.testing.assert_allclose(d.asnumpy(), k + a.asnumpy() - b.asnumpy() * c.asnumpy() + x)


def verify_tensor_scalar_bop(shape, typ="add"):
    """Verify non-constant Tensor and scalar binary operations."""
    sh = [tvm.var('n%d' % i) for i in range(0, len(shape))]
    k = tvm.var('k')
    A = tvm.placeholder(sh, name='A')
    if typ == "add":
        B = A + k
    elif typ == "sub":
        B = A - k
    elif typ == "mul":
        B = A * k
    elif typ == "div":
        B = A / k
    else:
        raise NotImplementedError()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_elemwise(B)

        k_ = 2
        foo = tvm.build(s, [A, B, k] + sh, device, name="tensor_scalar_" + typ)
        a_npy = np.random.uniform(size=shape).astype(A.dtype)
        if typ == "add":
            b_npy = a_npy + k_
        elif typ == "sub":
            b_npy = a_npy - k_
        elif typ == "mul":
            b_npy = a_npy * k_
        elif typ == "div":
            b_npy = a_npy / k_
        else:
            raise NotImplementedError()

        a_nd = tvm.nd.array(a_npy, ctx)
        b_nd = tvm.nd.array(np.empty(b_npy.shape).astype(B.dtype), ctx)
        foo(a_nd, b_nd, k_, *shape)
        tvm.testing.assert_allclose(b_nd.asnumpy(), b_npy, rtol=1e-5)

    for device in ['llvm', 'cuda', 'opencl', 'metal', 'rocm', 'vulkan']:
        check_device(device)


def verify_broadcast_bop(lhs_shape, rhs_shape, typ="add"):
    A = tvm.placeholder(shape=lhs_shape, name="A")
    B = tvm.placeholder(shape=rhs_shape, name="B")
    if typ == "add":
        C = A + B
    elif typ == "sub":
        C = A - B
    elif typ == "mul":
        C = A * B
    elif typ == "div":
        C = A / B
    else:
        raise NotImplementedError()

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_broadcast(C)

        foo = tvm.build(s, [A, B, C], device, name="broadcast_binary" + "_" + typ)
        lhs_npy = np.random.uniform(size=lhs_shape).astype(A.dtype)
        rhs_npy = np.random.uniform(size=rhs_shape).astype(A.dtype)
        if typ == "add":
            out_npy = lhs_npy + rhs_npy
        elif typ == "sub":
            out_npy = lhs_npy - rhs_npy
        elif typ == "mul":
            out_npy = lhs_npy * rhs_npy
        elif typ == "div":
            rhs_npy = np.abs(rhs_npy) + 0.001
            out_npy = lhs_npy / rhs_npy
        else:
            raise NotImplementedError()

        lhs_nd = tvm.nd.array(lhs_npy, ctx)
        rhs_nd = tvm.nd.array(rhs_npy, ctx)
        out_nd = tvm.nd.array(np.empty(out_npy.shape).astype(B.dtype), ctx)
        for _ in range(1):
            foo(lhs_nd, rhs_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.asnumpy(), out_npy, rtol=1E-4, atol=1E-4)

    for device in ['llvm', 'cuda', 'opencl', 'metal', 'rocm', 'vulkan']:
        check_device(device)


def verify_conv2d_scalar_bop(batch, in_size, in_channel, num_filter, kernel, stride, padding, typ="add"):
    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)

        k = 10.0
        dilation = (1, 1)
        with tvm.target.create(device):
            A = tvm.placeholder((batch, in_channel, in_size, in_size), name='A')
            W = tvm.placeholder((num_filter, in_channel, kernel, kernel), name='W')
            B = topi.nn.conv2d(A, W, stride, padding, dilation)
            if typ == "add":
                C = B + k
            elif typ == "sub":
                C = B - k
            elif typ == "mul":
                C = B * k
            elif typ == "div":
                C = B / k
            else:
                raise NotImplementedError()
            s = topi.generic.schedule_conv2d_nchw([C])

        foo = tvm.build(s, [A, W, B, C], device, name="conv2d_scalar_" + typ)

        a_npy = np.random.uniform(size=get_const_tuple(A.shape)).astype(A.dtype)
        w_npy = np.random.uniform(size=get_const_tuple(W.shape)).astype(W.dtype)
        b_npy = topi.testing.conv2d_nchw_python(a_npy, w_npy, stride, padding)
        c_npy = np.random.uniform(size=get_const_tuple(B.shape)).astype(B.dtype)
        if typ == "add":
            c_npy = b_npy + k
        elif typ == "sub":
            c_npy = b_npy - k
        elif typ == "mul":
            c_npy = b_npy * k
        elif typ == "div":
            c_npy = b_npy / k
        else:
            raise NotImplementedError()

        a_nd = tvm.nd.array(a_npy, ctx)
        w_nd = tvm.nd.array(w_npy, ctx)
        b_nd = tvm.nd.array(np.empty(b_npy.shape).astype(B.dtype), ctx)
        c_nd = tvm.nd.array(np.empty(c_npy.shape).astype(C.dtype), ctx)
        foo(a_nd, w_nd, b_nd, c_nd)
        tvm.testing.assert_allclose(c_nd.asnumpy(), c_npy, rtol=1E-4, atol=1E-4)

    for device in ['llvm', 'cuda', 'opencl', 'metal', 'rocm', 'vulkan']:
        check_device(device)


def test_tensor_scalar_bop():
    verify_tensor_scalar_bop((1,), typ="add")
    verify_tensor_scalar_bop((3, 5), typ="sub")
    verify_tensor_scalar_bop((1, 3, 5), typ="mul")
    verify_tensor_scalar_bop((2, 3, 1, 32), typ="div")


def test_broadcast_bop():
    verify_broadcast_bop((2, 3), (), typ="add")
    verify_broadcast_bop((5, 2, 3), (1,), typ="add")
    verify_broadcast_bop((1, 32), (64, 32), typ="sub")
    verify_broadcast_bop((5, 64, 128), (2, 5, 64, 1), typ="mul")
    verify_broadcast_bop((2, 3, 1, 32), (64, 32), typ="div")


def test_conv2d_scalar_bop():
    verify_conv2d_scalar_bop(1, 16, 4, 4, 3, 1, 1, typ="add")
    verify_conv2d_scalar_bop(1, 32, 2, 1, 3, 1, 1, typ="sub")
    verify_conv2d_scalar_bop(1, 32, 1, 1, 3, 1, 1, typ="mul")
    verify_conv2d_scalar_bop(1, 16, 2, 1, 3, 1, 1, typ="div")


if __name__ == "__main__":
    test_operator_type_and_tags()
    test_combination()
    test_tensor_scalar_bop()
    test_broadcast_bop()
    test_conv2d_scalar_bop()
