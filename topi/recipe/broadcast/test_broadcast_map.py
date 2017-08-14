import os
import tvm
from tvm.contrib import nvcc
import numpy as np

import topi


TASK = "reduce_map"
USE_MANUAL_CODE = False


@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx = nvcc.compile_cuda(code, target="ptx", options=["-arch=sm_52"])
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
    s = topi.cuda.schedule_broadcast_to(B.op)
    fcuda = tvm.build(s, [A, B], "cuda", name="broadcast_to")

    data_npy = np.random.uniform(size=in_shape).astype(A.dtype)
    out_npy = np.broadcast_to(data_npy, out_shape)

    data_nd = tvm.nd.array(data_npy, tvm.gpu())
    out_nd = tvm.nd.array(np.empty(out_shape).astype(B.dtype), tvm.gpu())
    for _ in range(2):
        fcuda(data_nd, out_nd)
    np.testing.assert_allclose(out_nd.asnumpy(), out_npy)


if __name__ == "__main__":
    test_broadcast_to((1,), (10,))
    test_broadcast_to((1, 1, 5, 4),  (3, 4, 4, 4, 5, 4))
    test_broadcast_to((1, 128, 1, 32),  (64, 128, 64, 32))
