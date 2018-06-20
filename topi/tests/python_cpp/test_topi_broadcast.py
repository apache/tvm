"""Test code for broadcasting operators."""
import os
import numpy as np
import tvm
import topi

def verify_broadcast_to_ele(in_shape, out_shape):
    # Build the logic and compile the function
    A = tvm.placeholder(shape=in_shape, name="A")
    B = topi.cpp.broadcast_to(A, out_shape)
    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        s = topi.cpp.cuda.schedule_injective(target, [B])
        ctx = tvm.context(device, 0)
        foo = tvm.build(s, [A, B], device, name="broadcast_to")
        data_npy = np.random.uniform(size=in_shape).astype(A.dtype)
        out_npy = np.broadcast_to(data_npy, out_shape)
        data_nd = tvm.nd.array(data_npy, ctx)
        out_nd = tvm.nd.array(np.empty(out_shape).astype(B.dtype), ctx)
        for _ in range(1):
            foo(data_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy)

    check_device("opencl")
    check_device("cuda")
    #check_device("metal")
    #check_device("rocm")


def verify_broadcast_binary_ele(lhs_shape, rhs_shape, typ="add"):
    # Build the logic and compile the function
    A = tvm.placeholder(shape=lhs_shape, name="A")
    B = tvm.placeholder(shape=rhs_shape, name="B")
    if typ == "add":
        C = topi.cpp.broadcast_add(A, B)
    elif typ == "sub":
        C = topi.cpp.broadcast_sub(A, B)
    elif typ == "div":
        C = topi.cpp.broadcast_div(A, B)
    elif typ == "mul":
        C = topi.cpp.broadcast_mul(A, B)
    elif typ == "maximum":
        C = topi.cpp.broadcast_maximum(A, B)
    elif typ == "minimum":
        C = topi.cpp.broadcast_minimum(A, B)
    elif typ == "pow":
        C = topi.cpp.broadcast_pow(A, B)
    else:
        raise NotImplementedError
    def check_device(device):
        if not tvm.module.enabled(device):
            print("Skip because %s is not enabled" % device)
            return
        print("Running on target: %s" % device)
        target = topi.cpp.TEST_create_target(device)
        s = topi.cpp.cuda.schedule_injective(target, [C])
        ctx = tvm.context(device, 0)
        foo = tvm.build(s, [A, B, C], device, name="broadcast_binary" + "_" + typ)
        lhs_npy = np.random.uniform(size=lhs_shape).astype(A.dtype)
        rhs_npy = np.random.uniform(size=rhs_shape).astype(A.dtype)
        if typ == "add":
            out_npy = lhs_npy + rhs_npy
        elif typ == "sub":
            out_npy = lhs_npy - rhs_npy
        elif typ == "div":
            rhs_npy = np.abs(rhs_npy) + 0.001
            out_npy = lhs_npy / rhs_npy
        elif typ == "mul":
            out_npy = lhs_npy * rhs_npy
        elif typ == "maximum":
            out_npy = np.maximum(lhs_npy, rhs_npy)
        elif typ == "minimum":
            out_npy = np.minimum(lhs_npy, rhs_npy)
        elif typ == "pow":
            out_npy = lhs_npy ** rhs_npy
        else:
            raise NotImplementedError
        lhs_nd = tvm.nd.array(lhs_npy, ctx)
        rhs_nd = tvm.nd.array(rhs_npy, ctx)
        out_nd = tvm.nd.array(np.empty(out_npy.shape).astype(B.dtype), ctx)
        for _ in range(1):
            foo(lhs_nd, rhs_nd, out_nd)
        np.testing.assert_allclose(out_nd.asnumpy(), out_npy, rtol=1E-4, atol=1E-4)

    check_device("opencl")
    check_device("cuda")
    #check_device("metal")
    #check_device("rocm")

def test_broadcast_to():
    verify_broadcast_to_ele((1,), (10,))
    verify_broadcast_to_ele((), (10,))
    verify_broadcast_to_ele((1, 1, 5, 4), (3, 4, 4, 4, 5, 4))
    verify_broadcast_to_ele((1, 128, 1, 32), (64, 128, 64, 32))


def test_broadcast_binary():
    verify_broadcast_binary_ele((5, 2, 3), (2, 1), typ="add")
    verify_broadcast_binary_ele((5, 2, 3), (), typ="add")
    verify_broadcast_binary_ele((5, 64, 128), (2, 5, 64, 1), typ="mul")
    verify_broadcast_binary_ele((2, 3, 1, 32), (64, 32), typ="div")
    verify_broadcast_binary_ele((1, 32), (64, 32), typ="sub")
    verify_broadcast_binary_ele((32,), (64, 32), typ="maximum")
    verify_broadcast_binary_ele((1, 2, 2, 1, 32), (64, 32), typ="minimum")
    verify_broadcast_binary_ele((1, 32), (64, 32), typ="pow")


if __name__ == "__main__":
    test_broadcast_to()
    test_broadcast_binary()
