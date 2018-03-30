import tvm

def test_lower_warp_mem():
    m = 128
    A = tvm.placeholder((m,), name='A')
    B = tvm.compute((m,), lambda i: A[i] + 3, name='B')

    s = tvm.create_schedule(B.op)
    AA = s.cache_read(A, "warp", [B])
    xo, xi = s[B].split(B.op.axis[0], 32)
    xi0, xi1 = s[B].split(xi, factor=16)
    tx = tvm.thread_axis("threadIdx.x")
    s[B].bind(xi1, tx)
    s[B].bind(xo, tvm.thread_axis("blockIdx.x"))
    s[AA].compute_at(s[B], xo)
    xo, xi = s[AA].split(s[AA].op.axis[0], 16)
    s[AA].bind(xi, tx)

    f = tvm.lower(s, [A, B])
    fhost, fdevice = tvm.ir_pass.SplitHostDevice(f)
    fdevice = tvm.ir_pass.LowerWarpMemory(fdevice, 16)
    assert(fdevice.body.body.value.value == "local")
    assert(fdevice.body.body.body.extents[0].value == 2)


if __name__ == "__main__":
    test_lower_warp_mem()
