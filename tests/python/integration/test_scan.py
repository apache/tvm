import tvm
import numpy as np

def test_scan():
    m = tvm.Var("m")
    n = tvm.Var("n")
    t = tvm.IterVar((1, m), name="t")
    X = tvm.placeholder((m, n), name="X")
    s_state = tvm.placeholder((m, n))
    s_init = tvm.compute((1, n), lambda _, i: X[0, i])
    s_update = tvm.compute((m, n), lambda t, i: s_state[t-1, i] + X[t, i])
    res = tvm.scan(s_init, s_update, s_state)

    # schedule
    s = tvm.Schedule(res.op)
    num_thread = 256
    block_x = tvm.IterVar(thread_tag="blockIdx.x")
    thread_x = tvm.IterVar((0, num_thread), thread_tag="threadIdx.x")
    _, x = s[s_init].split(s_init.op.axis[1], factor=num_thread, outer=block_x)
    _, x = s[s_init].split(x, outer=thread_x)
    _, x = s[s_update].split(s_update.op.axis[1], factor=num_thread, outer=block_x)
    _, x = s[s_update].split(x, outer=thread_x)

    # one line to build the function.
    def check_device(device, host="stackvm"):
        if not tvm.codegen.target_enabled(host):
            return
        if not tvm.codegen.target_enabled(device):
            return
        fscan = tvm.build(s, [X, res],
                          device, host,
                          name="myscan")
        ctx = tvm.gpu(0) if device == "cuda" else tvm.cl(0)
        # launch the kernel.
        n = 1024
        m = 10
        a_np = np.random.uniform(size=(m, n)).astype(res.dtype)
        a = tvm.nd.array(a_np, ctx)
        b = tvm.nd.array(np.zeros((m, n), dtype=res.dtype), ctx)
        fscan(a, b)
        np.testing.assert_allclose(
            b.asnumpy(), np.cumsum(a_np, axis=0))

    tvm.init_opencl()
    check_device("cuda")
    check_device("opencl")


if __name__ == "__main__":
    test_scan()
