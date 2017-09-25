"""Matrix exponential example.

This is an example for matrix exponential,
which calculates the following recursion formula

```math
X[t] = dot(X[t-1], W)
```
"""
import tvm
import time
import os
import argparse
from tvm.contrib import nvcc
import numpy as np

# Quick knobs
TASK="matexp"
USE_MANUAL_CODE = False
PERSIST_KERNEL = True
DETECT_GLOBAL_BARRIER = PERSIST_KERNEL
SKIP_CHECK = False

@tvm.register_func
def tvm_callback_cuda_compile(code):
    """Use nvcc compiler for better perf."""
    ptx =  nvcc.compile_cuda(code, target="ptx")
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

def rnn_matexp():
    n_num_step = 128
    n_num_hidden = 1152
    n_batch_size = 4
    detect_global_barrier = DETECT_GLOBAL_BARRIER

    num_step = tvm.var("num_step")
    num_hidden = tvm.convert(n_num_hidden)
    batch_size = tvm.convert(n_batch_size)
    num_thread_y = 8
    num_thread_x = 16 * 3
    num_sm = 24

    Whh = tvm.placeholder((num_hidden, num_hidden), name="Whh")
    s_init = tvm.compute((1, batch_size, num_hidden),
                         lambda _, i, j: 1.0, name="init")
    s_state = tvm.placeholder((num_step, batch_size, num_hidden))
    kh = tvm.reduce_axis((0, num_hidden), name="kh")
    s_update = tvm.compute(
        (num_step, batch_size, num_hidden),
        lambda t, i, j: tvm.sum(s_state[t-1, i, kh] * Whh[kh, j], axis=kh),
        name="update")
    s_scan = tvm.scan(s_init, s_update, s_state)
    # schedule
    s = tvm.create_schedule(s_scan.op)
    CL = s_update
    SS = s.cache_read(s_state, "shared", [CL])
    SL = s.cache_read(SS, "local", [CL])
    WhhL = s.cache_read(Whh, "local", [CL])
    ko, ki = s[CL].split(s[CL].op.reduce_axis[0], nparts=num_thread_y)
    CLF = s.rfactor(CL, ko)

    block_x = tvm.thread_axis((0, num_sm), "blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")
    if PERSIST_KERNEL:
        s[s_scan.op].env_threads([block_x, thread_y, thread_x])

    bx, xi = s[s_init].split(s_init.op.axis[2], nparts=num_sm)
    tx, xi = s[s_init].split(xi, nparts=num_thread_x)
    s[s_init].bind(bx, block_x)
    s[s_init].bind(tx, thread_x)

    bx, xi = s[s_update].split(s[CL].op.axis[2], nparts=num_sm)
    tx, xi = s[s_update].split(xi, nparts=num_thread_x)
    s[s_update].bind(bx, block_x)
    s[s_update].bind(tx, thread_x)
    s[CL].bind(s[CL].op.reduce_axis[0], thread_y)
    s[CLF].compute_at(s[CL], s[CL].op.reduce_axis[0])
    # Duplicate store predicate.
    s[CL].set_store_predicate(thread_y.equal(0))

    if PERSIST_KERNEL:
        s[WhhL].compute_at(s[s_scan], thread_x)
        s[WhhL].unroll(WhhL.op.axis[0])
    else:
        s[WhhL].compute_at(s[CLF], CLF.op.axis[3])

    kr, ki = s[CLF].split(CLF.op.reduce_axis[0], nparts=1)
    ko, ki = s[CLF].split(ki, factor=4)
    s[SS].compute_at(s[CLF], kr)
    s[SL].compute_at(s[CLF], ko)

    xo, xi = s[SS].split(SS.op.axis[2], factor=num_thread_x * num_thread_y * 3)
    ty, xi = s[SS].split(xi, nparts=num_thread_y)
    tx, xi = s[SS].split(xi, nparts=num_thread_x)
    s[SS].bind(ty, thread_y)
    s[SS].bind(tx, thread_x)

    def check_device(target):
        with tvm.build_config(
                detect_global_barrier=detect_global_barrier,
                auto_unroll_min_depth=2,
                auto_unroll_max_step=128,
                unroll_explicit=False):
            f = tvm.build(s, [s_scan, Whh], target)
        ctx = tvm.gpu(0) if target == "cuda" else tvm.cl(0)
        # launch the kernel.
        res_np = np.zeros(
            (n_num_step, n_batch_size, n_num_hidden)).astype("float32")
        Whh_np = np.zeros((n_num_hidden, n_num_hidden)).astype("float32")
        Whh_np[:] = 2.0 / n_num_hidden
        Whh_np[:, n_num_hidden//2:] = 0

        res_a = tvm.nd.array(res_np, ctx)
        Whh_a = tvm.nd.array(Whh_np, ctx)
        # Skip first pass as it is compilation
        f(res_a, Whh_a)
        ctx.sync()
        # measure time cost of second step.
        tstart = time.time()
        f(res_a, Whh_a)
        ctx.sync()
        tgap = time.time() - tstart
        print("Time cost=%g" % tgap)
        # correctness
        if not SKIP_CHECK:
            res_gpu = res_a.asnumpy()
            res_cmp = np.ones_like(res_np).astype("float64")
            Whh_np = Whh_np.astype("float64")
            for t in range(1, n_num_step):
                res_cmp[t][:] = np.dot(res_cmp[t - 1], Whh_np)
            for i  in range(n_num_step):
                for j in range(n_num_hidden):
                    if abs(res_cmp[i,0,j] - res_gpu[i,0,j]) > 1e-5:
                        print("%d, %d: %g vs %g" % (i,j, res_cmp[i,0,j], res_gpu[i,0,j]))
            np.testing.assert_allclose(res_gpu, res_cmp, rtol=1e-3)
    check_device("cuda")

if __name__ == "__main__":
    rnn_matexp()
