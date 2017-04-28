import tvm
import os
from tvm.contrib import nvcc_compiler
import numpy as np

TASK="gemm"
USE_MANUAL_CODE = False

@tvm.register_func
def tvm_callback_cuda_compile(code):
    ptx =  nvcc_compiler.compile_source(code, target="ptx", options=["-arch=sm_52"])
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

def test_gemm():
    # graph
    nn = 2048
    n = tvm.var('n')
    n = tvm.convert(nn)
    m, l = n, n
    A = tvm.placeholder((l, n), name='A')
    B = tvm.placeholder((l, m), name='B')
    k = tvm.reduce_axis((0, l), name='k')
    C = tvm.compute(
        (m, n),
        lambda ii, jj: tvm.sum(A[k, jj] * B[k, ii], axis=k),
        name='C')

    # schedule
    s = tvm.create_schedule(C.op)
    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])
    AL = s.cache_read(AA, "local", [C])
    BL = s.cache_read(BB, "local", [C])
    CC = s.cache_write(C, "local")

    scale = 8
    num_thread = 8
    block_factor = scale * num_thread
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread), "threadIdx.x")
    block_y = tvm.thread_axis("blockIdx.y")
    thread_y = tvm.thread_axis((0, num_thread), "threadIdx.y")
    thread_xz = tvm.thread_axis((0, 2), "vthread", name="vx")
    thread_yz = tvm.thread_axis((0, 2), "vthread", name="vy")

    by, yi = s[C].split(C.op.axis[0], factor=block_factor)
    bx, xi = s[C].split(C.op.axis[1], factor=block_factor)
    s[C].bind(by, block_y)
    s[C].bind(bx, block_x)
    s[C].reorder(by, bx, yi, xi)

    tyz, yi = s[C].split(yi, nparts=2)
    ty, yi = s[C].split(yi, nparts=num_thread)
    txz, xi = s[C].split(xi, nparts=2)
    tx, xi = s[C].split(xi, nparts=num_thread)
    s[C].bind(tyz, thread_yz)
    s[C].bind(txz, thread_xz)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].reorder(tyz, txz, ty, tx, yi, xi)
    s[CC].compute_at(s[C], tx)

    yo, xo = CC.op.axis
    ko, ki = s[CC].split(k, factor=8)
    kt, ki = s[CC].split(ki, factor=1)
    s[CC].reorder(ko, kt, ki, yo, xo)
    s[AA].compute_at(s[CC], ko)
    s[BB].compute_at(s[CC], ko)
    s[AL].compute_at(s[CC], kt)
    s[BL].compute_at(s[CC], kt)
    # Schedule for A's shared memory load
    ty, xi = s[AA].split(s[AA].op.axis[0], nparts=num_thread)
    _, xi = s[AA].split(s[AA].op.axis[1], factor=num_thread * 4)
    tx, xi = s[AA].split(xi, nparts=num_thread)
    s[AA].bind(ty, thread_y)
    s[AA].bind(tx, thread_x)
    s[AA].vectorize(xi)
    # Schedule for B' shared memory load
    ty, xi = s[BB].split(s[BB].op.axis[0], nparts=num_thread)
    _, xi = s[BB].split(s[BB].op.axis[1], factor=num_thread * 4)
    tx, xi = s[BB].split(xi, nparts=num_thread)
    s[BB].bind(ty, thread_y)
    s[BB].bind(tx, thread_x)
    s[BB].vectorize(xi)
    max_auto_unroll_step = 8

    # correctness
    def check_device(device, host="stackvm"):
        if not tvm.codegen.enabled(host):
            return
        if not tvm.codegen.enabled(device):
            return
        f = tvm.build(s, [A, B, C], device, host,
                      max_auto_unroll_step=max_auto_unroll_step)
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        # launch the kernel.
        n, m, l = nn, nn, nn
        a_np = np.random.uniform(size=(n, l)).astype(A.dtype)
        b_np = np.random.uniform(size=(m, l)).astype(B.dtype)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(b_np, ctx)
        c = tvm.nd.array(np.zeros((n, m), dtype=C.dtype), ctx)
        for i in range(2):
            f(a, b, c)
        np.testing.assert_allclose(
            c.asnumpy(), np.dot(b_np.T, a_np), rtol=1e-5)

    check_device("cuda")

if __name__ == "__main__":
    test_gemm()
