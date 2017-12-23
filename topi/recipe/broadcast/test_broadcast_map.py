import os
import tvm
from tvm.contrib import nvcc
import numpy as np

import topi


TASK = "reduce_map"
USE_MANUAL_CODE = False


@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx")
    return ptx


def write_code(code, fname):
    with open(fname, "w") as f:
        f.write(code)


@tvm.register_func
def tvm_callback_cuda_postproc(code):
    if not os.path.exists("perf"):
        os.mkdir("perf")
    write_code(code, "perf/%s_generated.cu" % TASK)
    if USE_MANUAL_CODE:
        code = open("perf/%s_manual.cu" % TASK).read()
    return code


def test_broadcast_to(in_shape, out_shape):
    global TASK
    TASK = "bcast_to_i" + "_".join([str(ele) for ele in in_shape])\
           + "o" + "_".join([str(ele) for ele in out_shape])
    # Build the logic and compile the function
    A = tvm.placeholder(shape=in_shape, name="A")
    B = topi.broadcast_to(A, out_shape)
    s = topi.cuda.schedule_broadcast(B)
    fcuda = tvm.build(s, [A, B], "cuda", name="broadcast_to")

    data_npy = np.random.uniform(size=in_shape).astype(A.dtype)
    out_npy = np.broadcast_to(data_npy, out_shape)

    data_nd = tvm.nd.array(data_npy, tvm.gpu())
    out_nd = tvm.nd.array(np.empty(out_shape).astype(B.dtype), tvm.gpu())
    for _ in range(2):
        fcuda(data_nd, out_nd)
    np.testing.assert_allclose(out_nd.asnumpy(), out_npy)


def test_broadcast_binary_op(lhs_shape, rhs_shape, typ="add"):
    global TASK
    TASK = "bcast_binary_" + typ + "_lhs" +\
           "_".join([str(ele) for ele in lhs_shape]) +\
           "rhs" + "_".join([str(ele) for ele in rhs_shape])
    A = tvm.placeholder(shape=lhs_shape, name="A")
    B = tvm.placeholder(shape=rhs_shape, name="B")
    if typ == "add":
        C = topi.broadcast_add(A, B)
    elif typ == "sub":
        C = topi.broadcast_sub(A, B)
    elif typ == "div":
        C = topi.broadcast_div(A, B)
    elif typ == "mul":
        C = topi.broadcast_mul(A, B)
    elif typ == "maximum":
        C = topi.broadcast_maximum(A, B)
    elif typ == "minimum":
        C = topi.broadcast_minimum(A, B)
    else:
        raise NotImplementedError
    s = topi.cuda.schedule_broadcast(C)
    fcuda = tvm.build(s, [A, B, C], "cuda", name="broadcast_binary" + "_" + typ)

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
    lhs_nd = tvm.nd.array(lhs_npy, tvm.gpu())
    rhs_nd = tvm.nd.array(rhs_npy, tvm.gpu())
    out_nd = tvm.nd.array(np.empty(out_npy.shape).astype(B.dtype), tvm.gpu())
    for _ in range(2):
        fcuda(lhs_nd, rhs_nd, out_nd)
    np.testing.assert_allclose(out_nd.asnumpy(), out_npy)


if __name__ == "__main__":
    test_broadcast_to((1,), (10,))
    test_broadcast_to((1, 1, 5, 4),  (3, 4, 4, 4, 5, 4))
    test_broadcast_to((1, 128, 1, 32),  (64, 128, 64, 32))
    test_broadcast_binary_op((5, 2, 3), (2, 1), typ="add")
    test_broadcast_binary_op((5, 64, 128), (2, 5, 64, 1), typ="mul")
    test_broadcast_binary_op((2, 3, 1, 32), (64, 32), typ="div")
    test_broadcast_binary_op((1, 32), (64, 32), typ="sub")
    test_broadcast_binary_op((32,), (64, 32), typ="maximum")
    test_broadcast_binary_op((1, 2, 2, 1, 32), (64, 32), typ="minimum")
