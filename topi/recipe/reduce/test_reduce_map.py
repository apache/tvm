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


def test_reduce_map(in_shape, axis, keepdims, type="sum", test_id=0):
    global TASK
    # Build the logic and compile the function
    A = tvm.placeholder(shape=in_shape, name="A")
    if type == "sum":
        TASK = "sum_map_id%d" %test_id
        B = topi.sum(A, axis=axis, keepdims=keepdims)
    elif type == "max":
        TASK = "max_map_id%d" %test_id
        B = topi.max(A, axis=axis, keepdims=keepdims)
    elif type == "min":
        TASK = "min_map_id%d" %test_id
        B = topi.min(A, axis=axis, keepdims=keepdims)
    else:
        raise NotImplementedError
    s = topi.cuda.schedule_reduce(B)
    with tvm.build_config(auto_unroll_max_step=16,
                          auto_unroll_min_depth=0):
        fcuda = tvm.build(s, [A, B], "cuda", name="sum")

    # Test
    in_npy = np.random.normal(size=in_shape).astype(np.float32)
    if type == "sum":
        out_npy = in_npy.sum(axis=axis, keepdims=keepdims)
    elif type == "max":
        out_npy = in_npy.max(axis=axis, keepdims=keepdims)
    elif type == "min":
        out_npy = in_npy.min(axis=axis, keepdims=keepdims)
    else:
        raise NotImplementedError

    data_tvm = tvm.nd.array(in_npy, ctx=tvm.gpu())
    out_tvm = tvm.nd.empty(shape=out_npy.shape, ctx=tvm.gpu())

    for _ in range(2):
        fcuda(data_tvm, out_tvm)
    np.testing.assert_allclose(out_tvm.asnumpy(), out_npy, 4E-4, 4E-4)

if __name__ == "__main__":
    test_reduce_map(in_shape=(128, 24, 128, 24),
                    axis=(1, 2, 3),
                    keepdims=True,
                    type="sum",
                    test_id=0)
    test_reduce_map(in_shape=(128, 24 * 128 * 24),
                    axis=(1,),
                    keepdims=False,
                    type="max",
                    test_id=1)
    test_reduce_map(in_shape=(32, 128, 24),
                    axis=None,
                    keepdims=True,
                    type="sum",
                    test_id=2)
    test_reduce_map(in_shape=(128, 24, 128, 24),
                    axis=(0, 2),
                    keepdims=False,
                    type="min",
                    test_id=3)
