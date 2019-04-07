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
"""Test code for broadcasting operators."""
from common import get_all_backend
import numpy as np
import tvm
import topi

def verify_broadcast_to_ele(in_shape, out_shape, fbcast):
    # Build the logic and compile the function
    A = tvm.placeholder(shape=in_shape, name="A")
    B = fbcast(A, out_shape)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_broadcast(B)
        foo = tvm.build(s, [A, B], device, name="broadcast_to")
        data_npy = np.random.uniform(size=in_shape).astype(A.dtype)
        out_npy = np.broadcast_to(data_npy, out_shape)
        data_nd = tvm.nd.array(data_npy, ctx)
        out_nd = tvm.nd.array(np.empty(out_shape).astype(B.dtype), ctx)
        foo(data_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    for target in get_all_backend():
        check_device(target)
    check_device("sdaccel")


def verify_broadcast_binary_ele(lhs_shape, rhs_shape,
                                ftopi, fnumpy,
                                lhs_min=-100, lhs_max=100,
                                rhs_min=-100, rhs_max=100,
                                dtype="float32"):
    # Build the logic and compile the function
    A = (tvm.var("A", dtype=dtype) if lhs_shape is None
         else tvm.placeholder(shape=lhs_shape, name="A", dtype=dtype))
    B = (tvm.var("B", dtype=dtype) if rhs_shape is None
         else tvm.placeholder(shape=rhs_shape, name="B", dtype=dtype))
    C = ftopi(A, B)
    if isinstance(A, tvm.expr.Expr) and isinstance(B, tvm.expr.Expr):
        assert(isinstance(C, tvm.expr.Expr))
        return

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.generic.schedule_broadcast(C)
        foo = tvm.build(s, [A, B, C], device, name="broadcast_binary" + "_" + ftopi.__name__)
        if lhs_shape is None:
            lhs_npy = float(np.random.uniform(low=lhs_min, high=lhs_max))
            if dtype.startswith('int'):
                lhs_npy = int(lhs_npy)
            lhs_nd = lhs_npy
        else:
            lhs_npy = np.random.uniform(low=lhs_min, high=lhs_max,
                                        size=lhs_shape).astype(A.dtype)
            lhs_nd = tvm.nd.array(lhs_npy, ctx)

        if rhs_shape is None:
            rhs_npy = float(np.random.uniform(low=rhs_min, high=rhs_max))
            if dtype.startswith('int'):
                rhs_npy = int(rhs_npy)
            rhs_nd = rhs_npy
        else:
            rhs_npy = np.random.uniform(low=rhs_min, high=rhs_max,
                                        size=rhs_shape).astype(A.dtype)
            rhs_nd = tvm.nd.array(rhs_npy, ctx)

        out_npy = fnumpy(lhs_npy, rhs_npy)
        out_nd = tvm.nd.array(np.empty(out_npy.shape).astype(C.dtype), ctx)
        foo(lhs_nd, rhs_nd, out_nd)
        tvm.testing.assert_allclose(out_nd.asnumpy(), out_npy, rtol=1E-4, atol=1E-4)

    for target in get_all_backend():
        check_device(target)
    check_device("sdaccel")

def test_broadcast_to():
    verify_broadcast_to_ele((1,), (10,), topi.broadcast_to)
    verify_broadcast_to_ele((), (10,), topi.broadcast_to)
    verify_broadcast_to_ele((1, 1, 5, 4), (3, 4, 4, 4, 5, 4), topi.broadcast_to)
    verify_broadcast_to_ele((1, 128, 1, 32), (64, 128, 64, 32), topi.broadcast_to)

def test_add():
    verify_broadcast_binary_ele(
        (), (), topi.add, np.add)
    verify_broadcast_binary_ele(
        (5, 2, 3), (2, 1), topi.add, np.add)

def test_subtract():
    verify_broadcast_binary_ele(
        (5, 2, 3), (), topi.subtract, np.subtract)
    verify_broadcast_binary_ele(
        (5, 2, 3), None, topi.subtract, np.subtract)
    verify_broadcast_binary_ele(
        None, None, topi.subtract, np.subtract)
    verify_broadcast_binary_ele(
        (1, 32), (64, 32), topi.subtract, np.subtract)

def test_multiply():
    verify_broadcast_binary_ele(
        (5, 64, 128), (2, 5, 64, 1), topi.multiply, np.multiply)

def test_divide():
    verify_broadcast_binary_ele(
        None, (10,), topi.divide, np.divide, rhs_min=0.0001)
    verify_broadcast_binary_ele(
        (), None, topi.divide, np.divide, rhs_min=0.0001)
    verify_broadcast_binary_ele(
        (2, 3, 1, 32), (64, 32), topi.divide, np.divide, rhs_min=0.0001)

def test_maximum_minmum():
    verify_broadcast_binary_ele(
        (32,), (64, 32), topi.maximum, np.maximum)
    verify_broadcast_binary_ele(
        (1, 2, 2, 1, 32), (64, 32), topi.minimum, np.minimum)

def test_power():
    verify_broadcast_binary_ele(
        (1, 2, 2), (2,), topi.power, np.power, lhs_min=0.001, rhs_min=0.001, rhs_max=2)

def test_mod():
    verify_broadcast_binary_ele(
        (1, 2, 2), (2,), topi.mod, np.mod, lhs_min=0.001, rhs_min=1, dtype="int32")

def test_cmp():
    # explicit specify the output type
    def greater(x, y):
        return topi.greater(x, y).astype("int8")
    def less(x, y):
        return topi.less(x, y).astype("int8")
    def equal(x, y):
        return topi.equal(x, y).astype("int8")
    def not_equal(x, y):
        return topi.not_equal(x, y).astype("int8")
    def greater_equal(x, y):
        return topi.greater_equal(x, y).astype("int8")
    def less_equal(x, y):
        return topi.less_equal(x, y).astype("int8")
    verify_broadcast_binary_ele(
        (1, 2, 2), (2,), greater, np.greater)
    verify_broadcast_binary_ele(
        (2, 1, 2), (2, 3, 1), less, np.less)
    verify_broadcast_binary_ele(
        (2, 1, 2), (2, 3, 1), equal, np.equal,
        lhs_min=-2, lhs_max=2, rhs_min=-2, rhs_max=2, dtype='int32')
    verify_broadcast_binary_ele(
        (2, 1, 2), (2, 3, 1), not_equal, np.not_equal,
        lhs_min=-2, lhs_max=2, rhs_min=-2, rhs_max=2, dtype='int32')
    verify_broadcast_binary_ele(
        (7, 1, 5), (7, 3, 1), greater_equal, np.greater_equal,
        lhs_min=-3, lhs_max=3, rhs_min=-3, rhs_max=3, dtype='int32')
    verify_broadcast_binary_ele(
        (7, 1, 5), (7, 3, 1), less_equal, np.less_equal,
        lhs_min=-3, lhs_max=3, rhs_min=-3, rhs_max=3, dtype='int32')

def test_shift():
    # explicit specify the output type
    verify_broadcast_binary_ele(
        (2, 1, 2), None, topi.right_shift, np.right_shift,
        dtype="int32", rhs_min=0, rhs_max=32)

    verify_broadcast_binary_ele(
        (1, 2, 2), (2,), topi.left_shift, np.left_shift,
        dtype="int32", rhs_min=0, rhs_max=32)

    verify_broadcast_binary_ele(
        (1, 2, 2), (2,), topi.left_shift, np.left_shift,
        dtype="int8", rhs_min=0, rhs_max=32)


if __name__ == "__main__":
    test_add()
    test_shift()
    test_cmp()
    test_mod()
    test_subtract()
    test_multiply()
    test_divide()
    test_maximum_minmum()
    test_power()
    test_broadcast_to()
