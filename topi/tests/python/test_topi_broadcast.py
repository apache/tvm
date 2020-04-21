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
import numpy as np
import tvm
from tvm import te
import topi
import topi.testing
from common import get_all_backend


def verify_broadcast_to_ele(in_shape, out_shape, fbcast):
    # Build the logic and compile the function
    A = te.placeholder(shape=in_shape, name="A")
    B = fbcast(A, out_shape)

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.testing.get_broadcast_schedule(device)(B)
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
    A = (te.var("A", dtype=dtype) if lhs_shape is None
         else te.placeholder(shape=lhs_shape, name="A", dtype=dtype))
    B = (te.var("B", dtype=dtype) if rhs_shape is None
         else te.placeholder(shape=rhs_shape, name="B", dtype=dtype))
    C = ftopi(A, B)
    if isinstance(A, tvm.tir.PrimExpr) and isinstance(B, tvm.tir.PrimExpr):
        assert(isinstance(C, tvm.tir.PrimExpr))
        return

    def gen_operand(shape, low, high, ctx):
        if shape is None:
            npy = float(np.random.uniform(low=low, high=high))
            if dtype.startswith('int'):
                npy = int(npy)
            nd = npy
        else:
            npy = np.random.uniform(low=low, high=high,
                                    size=shape).astype(dtype)
            nd = tvm.nd.array(npy, ctx)
        return npy, nd

    def check_device(device):
        ctx = tvm.context(device, 0)
        if not ctx.exist:
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        with tvm.target.create(device):
            s = topi.testing.get_broadcast_schedule(device)(C)
        foo = tvm.build(s, [A, B, C], device, name="broadcast_binary" + "_" + ftopi.__name__)

        lhs_npy, lhs_nd = gen_operand(lhs_shape, lhs_min, lhs_max, ctx)
        rhs_npy, rhs_nd = gen_operand(rhs_shape, rhs_min, rhs_max, ctx)
        out_npy = fnumpy(lhs_npy, rhs_npy)

        if fnumpy == np.floor_divide:
            # avoid check too close to X.5 and X.0
            # FIXME: floor_divide(94.90735, 0.6731018) behaves as floor(div(94.90735, 0.6731018))
            # However the result is somehow incorrect - need to further investigate.
            # And looks like numpy's floor_div(a,b) is implemented different from floor(div(a,b))
            mask = np.logical_or(np.abs(np.abs(np.fmod(lhs_npy / rhs_npy, 1)) - 0.5) < 1e-6,
                                 np.abs(np.fmod(lhs_npy / rhs_npy, 1)) < 1e-6)
            if mask.any():
                lhs_npy = lhs_npy + mask * 1e-3  * rhs_npy
                lhs_npy = lhs_npy.astype(dtype)
                lhs_nd = tvm.nd.array(lhs_npy, ctx) if lhs_shape is not None else lhs_npy.item()
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

def test_floor_divide():
    verify_broadcast_binary_ele(
        None, (10,), topi.floor_divide, np.floor_divide, rhs_min=0.0001)
    verify_broadcast_binary_ele(
        (), None, topi.floor_divide, np.floor_divide, rhs_min=0.0001)
    verify_broadcast_binary_ele(
        (2, 3, 64, 32), (64, 32), topi.floor_divide, np.floor_divide, rhs_min=0.0001)

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

def test_floor_mod():
    verify_broadcast_binary_ele(
        (1, 2, 2), (2,), topi.floor_mod, np.fmod, lhs_min=0.001, rhs_min=1, dtype="int32")
    verify_broadcast_binary_ele(
        (3, 4, 5), (3, 4, 5), topi.floor_mod, np.fmod, lhs_min=0.001, rhs_min=1, dtype="float32")

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


def test_logical_single_ele():
    def test_apply(
            func,
            name,
            f_numpy,
            indata,
            dtype="bool",
    ):
        # Build the logic and compile the function
        A = te.placeholder(shape=indata.shape, name="A", dtype=dtype)
        B = func(A)
        if isinstance(A, tvm.tir.PrimExpr):
            assert (isinstance(B, tvm.tir.PrimExpr))
            return

        def check_device(device):
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                return
            print("Running on target: %s" % device)
            with tvm.target.create(device):
                s = topi.testing.get_broadcast_schedule(device)(B)
            foo = tvm.build(s, [A, B], device, name=name)

            data_npy = indata.astype(A.dtype)
            data_nd = tvm.nd.array(data_npy, ctx)

            out_npy = f_numpy(indata)
            out_nd = tvm.nd.array(np.empty(data_npy.shape).astype(B.dtype), ctx)
            foo(data_nd, out_nd)
            tvm.testing.assert_allclose(out_nd.asnumpy(), out_npy)

        for device in get_all_backend():
            check_device(device)

    test_apply(topi.logical_not, "logical_not", np.logical_not, np.array([True, False, 0, 1]))
    test_apply(topi.logical_not, "logical_not", np.logical_not, np.array(np.arange(5) < 3))


def test_bitwise_not():
    def test_apply(
            func,
            name,
            f_numpy,
            shape,
            dtype="int32",
    ):
        # Build the logic and compile the function
        A = te.placeholder(shape=shape, name="A", dtype=dtype)
        B = func(A)

        if isinstance(A, tvm.tir.PrimExpr):
            assert (isinstance(B, tvm.tir.PrimExpr))
            return

        def check_device(device):
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                return
            print("Running on target: %s" % device)
            with tvm.target.create(device):
                s = topi.testing.get_broadcast_schedule(device)(B)
            foo = tvm.build(s, [A, B], device, name=name)

            data_npy = np.random.uniform(size=shape).astype(A.dtype)
            data_nd = tvm.nd.array(data_npy, ctx)

            out_npy = f_numpy(data_npy)
            out_nd = tvm.nd.array(np.empty(data_npy.shape).astype(B.dtype), ctx)
            foo(data_nd, out_nd)
            tvm.testing.assert_allclose(out_nd.asnumpy(), out_npy)

        for device in get_all_backend():
            check_device(device)

    test_apply(topi.bitwise_not, "bitwise_not", np.bitwise_not, ())
    test_apply(topi.bitwise_not, "bitwise_not", np.bitwise_not, (2, 1, 2))


def test_logical_binary_ele():
    def test_apply(
            func,
            name,
            f_numpy,
            lhs,
            rhs,
            dtype="bool",
    ):
        # Build the logic and compile the function
        A = (te.var("A", dtype=dtype))
        B = (te.var("B", dtype=dtype))
        C = func(A, B)
        if isinstance(A, tvm.tir.PrimExpr) and isinstance(B, tvm.tir.PrimExpr):
            assert (isinstance(C, tvm.tir.PrimExpr))
            return

        def check_device(device):
            ctx = tvm.context(device, 0)
            if not ctx.exist:
                print("Skip because %s is not enabled" % device)
                return
            print("Running on target: %s" % device)
            with tvm.target.create(device):
                s = topi.testing.get_broadcast_schedule(device)(C)
            foo = tvm.build(s, [A, B, C], device, name=name)

            lhs_nd = tvm.nd.array(lhs, ctx)
            rhs_nd = tvm.nd.array(rhs, ctx)

            out_npy = f_numpy(lhs, rhs)
            out_nd = tvm.nd.array(np.empty(out_npy.shape).astype(C.dtype), ctx)
            foo(lhs_nd, rhs_nd, out_nd)
            tvm.testing.assert_allclose(out_nd.asnumpy(), out_npy, rtol=1E-4, atol=1E-4)

        for device in get_all_backend():
            check_device(device)

    test_apply(topi.logical_and, "logical_and", np.logical_and, True, False)
    test_apply(topi.logical_and, "logical_and", np.logical_and, [True, False], [False, False])
    test_apply(topi.logical_or, "logical_or", np.logical_or, True, False)
    test_apply(topi.logical_or, "logical_or", np.logical_or, [True, False], [False, False])
    test_apply(topi.logical_xor, "logical_xor", np.logical_xor, True, False)
    test_apply(topi.logical_xor, "logical_xor", np.logical_xor, [True, False], [False, False])


def test_bitwise_and():
    verify_broadcast_binary_ele(
        None, None, topi.bitwise_and, np.bitwise_and,
        dtype="int32")
    verify_broadcast_binary_ele(
        (2, 1, 2), (2, 1, 2), topi.bitwise_and, np.bitwise_and,
        dtype="int32")


def test_bitwise_or():
    verify_broadcast_binary_ele(
        None, None, topi.bitwise_or, np.bitwise_or,
        dtype="int32")
    verify_broadcast_binary_ele(
        (2, 1, 2), (2, 1, 2), topi.bitwise_or, np.bitwise_or,
        dtype="int32")


def test_bitwise_xor():
    verify_broadcast_binary_ele(
        None, None, topi.bitwise_xor, np.bitwise_xor,
        dtype="int32")
    verify_broadcast_binary_ele(
        (2, 1, 2), (2, 1, 2), topi.bitwise_xor, np.bitwise_xor,
        dtype="int32")


if __name__ == "__main__":
    test_add()
    test_shift()
    test_cmp()
    test_mod()
    test_floor_mod()
    test_subtract()
    test_multiply()
    test_divide()
    test_floor_divide()
    test_maximum_minmum()
    test_power()
    test_broadcast_to()
    test_logical_single_ele()
    test_bitwise_not()
    test_logical_binary_ele()
    test_bitwise_and()
    test_bitwise_or()
    test_bitwise_xor()
